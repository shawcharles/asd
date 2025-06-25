#!/usr/bin/env python3
"""Benchmark design runtime scalability.

Runs SupergeoDesign and AdaptiveSupergeoDesign for increasing numbers of
geo-units and records wall-clock runtime.  Results are stored in
``paper_assets/scalability.csv`` and a corresponding PDF line-plot is
produced.  The benchmark is intentionally lightweight – we use a single trial
per N with a fixed random seed so that it finishes in a couple of minutes.
"""
from __future__ import annotations

import csv
import sys, pathlib
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# Ensure project src in path
SRC_DIR = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from supergeos.supergeo_design import SupergeoDesign, GeoUnit
from adaptive_supergeos.adaptive_supergeo_design import AdaptiveSupergeoDesign

# -----------------------------------------------------------------------------
# Helper to create synthetic GeoUnits
# -----------------------------------------------------------------------------

def synthetic_units(n: int, rng: np.random.Generator) -> List[GeoUnit]:
    units: List[GeoUnit] = []
    for i in range(n):
        response = rng.normal(loc=2.0, scale=0.2)  # centred near 2 for analogy
        spend = rng.lognormal(mean=1.0, sigma=0.3)  # positive
        units.append(GeoUnit(id=str(i), response=response, spend=spend))
    return units


# -----------------------------------------------------------------------------
# Benchmark loop
# -----------------------------------------------------------------------------

def time_method(method_name: str, algo, n_supergeos_per_arm: int) -> float:
    start = time.perf_counter()
    try:
        algo.optimize_partition(n_supergeos_per_arm, n_supergeos_per_arm, random_seed=42)
        duration = time.perf_counter() - start
    except Exception:
        duration = float("nan")
    return duration


def benchmark(out_csv: Path, out_pdf: Path) -> None:
    Ns = [50, 100, 200, 400, 800, 1000]
    rows = []
    rng = np.random.default_rng(0)

    for N in Ns:
        print(f"Benchmarking N={N} …", flush=True)
        units = synthetic_units(N, rng)
        n_supergeos_per_arm = max(1, N // 20)

        # SupergeoDesign – only run for moderately sized problems to avoid timeouts
        if N <= 400:
            sg_algo = SupergeoDesign(units)
            sg_t = time_method("SG", sg_algo, n_supergeos_per_arm)
        else:
            sg_t = float("nan")  # skipped to cap runtime

        # Adaptive Supergeo
        asd_algo = AdaptiveSupergeoDesign(units)
        asd_t = time_method("ASD", asd_algo, n_supergeos_per_arm)

        rows.append({"N": N, "method": "SG-TM", "runtime_sec": sg_t})
        rows.append({"N": N, "method": "ASD-TM", "runtime_sec": asd_t})

    # Write CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["N", "method", "runtime_sec"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] wrote {out_csv}")

    # Plot
    Ns_unique = sorted({r["N"] for r in rows})
    plt.figure(figsize=(4, 3))
    for method in {r["method"] for r in rows}:
        ys = [next((r["runtime_sec"] for r in rows if r["N"]==N and r["method"]==method), float('nan')) for N in Ns_unique]
        plt.plot(Ns_unique, ys, marker="o", label=method)
    plt.yscale("log")
    plt.xlabel("Number of geos (N)")
    plt.ylabel("Wall-clock runtime (s, log scale)")
    plt.title("Design optimisation runtime vs N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"[OK] wrote {out_pdf}")


if __name__ == "__main__":
    benchmark(Path("paper_assets/scalability.csv"), Path("paper_assets/scalability_plot.pdf"))
