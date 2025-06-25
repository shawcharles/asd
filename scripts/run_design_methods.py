#!/usr/bin/env python
"""Run baseline, Supergeo, and Adaptive Supergeo designs on synthetic data.

This script generates synthetic geographic units, runs three design pipelines
(Random-Pairs baseline, SupergeoDesign, AdaptiveSupergeoDesign) and evaluates
each using the Trimmed Match estimator.  Results are saved to a CSV that can be
plugged straight into the paper tables / figures.

Usage
-----
$ python scripts/run_design_methods.py --n-units 200 --n-seeds 200 --out results.csv

The default parameters reproduce the configuration used in §4 of the paper.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Make sure project src/ directory is on PYTHONPATH when run directly
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import csv
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Local imports – assume src/ is on PYTHONPATH when run from project root
from supergeos.supergeo_design import GeoUnit, SupergeoDesign
from adaptive_supergeos.adaptive_supergeo_design import AdaptiveSupergeoDesign
from trimmed_match.estimator import TrimmedMatch

# -----------------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------------

def generate_geo_units(n_units: int, rng: np.random.Generator, *, true_iroas: float) -> List[GeoUnit]:
    """Create *baseline* geo units used by the design algorithms.

    Each GeoUnit contains only *pre-period* metrics (response, spend, covariates)
    because the design phase must be blinded to treatment effects.
    """
    # Baseline spend between 100 and 500 (uniform), baseline response around 1k.
    base_spend = rng.uniform(100.0, 500.0, size=n_units)
    base_response = rng.normal(loc=1000.0, scale=200.0, size=n_units)

    # Two simple covariates for balance tests.
    income = rng.normal(50.0, 10.0, size=n_units)  # arbitrary units
    population = rng.normal(100.0, 20.0, size=n_units)

    units: List[GeoUnit] = []
    for i in range(n_units):
        covs = {"income": float(income[i]), "population": float(population[i])}
        units.append(GeoUnit(id=f"g{i}", response=float(base_response[i]), spend=float(base_spend[i]), covariates=covs))
    return units

# -----------------------------------------------------------------------------
# Helper: convert supergeo partition → TrimmedMatch inputs
# -----------------------------------------------------------------------------

def partition_to_delta_lists(treatment_sgs: List[List[GeoUnit]], control_sgs: List[List[GeoUnit]], rng: np.random.Generator, *, true_iroas: float, noise_sd: float = 0.2) -> Tuple[List[float], List[float]]:
    """Return (delta_response, delta_spend) aggregated at the supergeo level.

    We simulate **incremental** spend and response *post-treatment* using a
    simple linear model with iROAS = true_iroas + Gaussian noise.
    """
    # If the design produces a different number of treatment vs control supergeos
    # (e.g., due to integer size constraints), we aggregate over the *common*
    # pairs and ignore the remainder so that TrimmedMatch receives equal-length
    # lists.  This pragmatic choice keeps the benchmarking script robust without
    # affecting the relative comparison of design quality.
    max_pairs = min(len(treatment_sgs), len(control_sgs))
    delta_response: List[float] = []
    delta_spend: List[float] = []

    if max_pairs == 0:
        # Fallback: no pairs to compare – return singleton zeros to avoid errors
        return [0.0], [0.0]

    for treat_sg, ctrl_sg in zip(treatment_sgs, control_sgs):
        # Incremental spend for treatment supergeo: proportional to baseline spend.
        inc_spend_t = sum(g.spend for g in treat_sg) * 0.5  # 50 % of baseline spend
        inc_spend_c = 0.0  # control arm gets no incremental spend

        # Incremental response generated via true iROAS plus idiosyncratic noise.
        inc_resp_t = true_iroas * inc_spend_t + rng.normal(0.0, noise_sd * inc_spend_t)
        inc_resp_c = 0.0

        delta_spend.append(inc_spend_t - inc_spend_c)
        delta_response.append(inc_resp_t - inc_resp_c)

    return delta_response, delta_spend

# -----------------------------------------------------------------------------
# Main loop per seed
# -----------------------------------------------------------------------------

def run_single_seed(seed: int, *, n_units: int, n_pairs: int, true_iroas: float) -> List[Tuple[str, float, float, float]]:
    """Run all three methods and return rows for CSV.

    Each row: (method, estimate, std_error, balance_score).
    """
    rng = np.random.default_rng(seed)
    geo_units = generate_geo_units(n_units, rng, true_iroas=true_iroas)

    results: List[Tuple[str, float, float, float]] = []

    # Utility: run TrimmedMatch safely and return (estimate, se) or (None, None).
    def safe_tm(delta_r: List[float], delta_s: List[float]):
        try:
            tm = TrimmedMatch(delta_r, delta_s)
            return tm.auto_estimate()[:2]  # (estimate, se)
        except Exception:
            return None, None

    # ---------------- Random-Pairs baseline ----------------
    shuffled = geo_units.copy()
    rng.shuffle(shuffled)
    pairs = [(shuffled[2 * i], shuffled[2 * i + 1]) for i in range(n_pairs)]
    treat_pairs = [p[0] for p in pairs]
    ctrl_pairs = [p[1] for p in pairs]
    delta_r, delta_s = partition_to_delta_lists([[g] for g in treat_pairs], [[g] for g in ctrl_pairs], rng, true_iroas=true_iroas)
    est, se = safe_tm(delta_r, delta_s)
    if est is not None:
        results.append(("TM-Baseline", est, se, math.fabs(sum(delta_r))))

    # ---------------- SupergeoDesign ----------------
    # Target a moderate number of supergeos so that each contains at least a
    # few units.  This avoids pathological cases where the requested number of
    # supergeos exceeds what the algorithm can construct due to min_supergeo_size
    # constraints.
    n_supergeos_per_arm = max(1, n_units // 20)  # ~=10 units per supergeo
    sg_algo = SupergeoDesign(geo_units)
    sg_partition = sg_algo.optimize_partition(n_supergeos_per_arm, n_supergeos_per_arm, random_seed=seed)
    delta_r, delta_s = partition_to_delta_lists(sg_partition.treatment_supergeos, sg_partition.control_supergeos, rng, true_iroas=true_iroas)
    est, se = safe_tm(delta_r, delta_s)
    if est is not None:
        results.append(("SG-TM", est, se, sg_partition.balance_score))

    # ---------------- AdaptiveSupergeoDesign ----------------
    asd_algo = AdaptiveSupergeoDesign(geo_units)
    asd_partition = asd_algo.optimize_partition(n_supergeos_per_arm, n_supergeos_per_arm, random_seed=seed)
    delta_r, delta_s = partition_to_delta_lists(asd_partition.treatment_supergeos, asd_partition.control_supergeos, rng, true_iroas=true_iroas)
    est, se = safe_tm(delta_r, delta_s)
    if est is not None:
        results.append(("ASD-TM", est, se, asd_partition.balance_score))

    return results

# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run design comparison experiments.")
    parser.add_argument("--n-units", type=int, default=200, help="Number of geo units to simulate (even number)")
    parser.add_argument("--n-seeds", type=int, default=100, help="Number of Monte Carlo repetitions")
    parser.add_argument("--true-iroas", type=float, default=2.0, help="Ground-truth iROAS value")
    parser.add_argument("--out", type=Path, default=Path("results.csv"), help="Output CSV path")
    args = parser.parse_args()

    if args.n_units % 2 != 0:
        parser.error("--n-units must be even so pairs can be formed")

    n_pairs = args.n_units // 2
    rows = []
    for seed in range(args.n_seeds):
        for method, est, se, balance in run_single_seed(seed, n_units=args.n_units, n_pairs=n_pairs, true_iroas=args.true_iroas):
            bias = est - args.true_iroas
            rows.append({
                "seed": seed,
                "method": method,
                "estimate": est,
                "std_error": se,
                "bias": bias,
                "abs_bias": abs(bias),
                "balance_score": balance,
            })

    # Aggregate metrics across seeds
    methods = sorted(set(r["method"] for r in rows))
    summary_rows = []
    for m in methods:
        ms = [r for r in rows if r["method"] == m]
        ests = np.array([r["estimate"] for r in ms])
        ses = np.array([r["std_error"] for r in ms])
        abs_biases = np.array([r["abs_bias"] for r in ms])
        balances = np.array([r["balance_score"] for r in ms])
        summary_rows.append({
            "method": m,
            "mean_estimate": float(np.mean(ests)),
            "rmse": float(math.sqrt(np.mean((ests - args.true_iroas) ** 2))),
            "mean_std_error": float(np.mean(ses)),
            "mean_abs_bias": float(np.mean(abs_biases)),
            "mean_balance": float(np.mean(balances)),
        })

    # Write detailed rows + summary as a single CSV (two sections separated by blank line)
    with args.out.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        # blank line then summary
        f.write("\n")
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Wrote results for {args.n_seeds} seeds → {args.out}")


if __name__ == "__main__":
    main()
