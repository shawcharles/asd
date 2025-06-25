#!/usr/bin/env python3
"""Generate supporting plots for the Adaptive Supergeo Design paper.

Reads a CSV produced by ``scripts/run_design_methods.py`` (detailed rows +
summary) and creates PDF figures under ``paper_assets/`` that can be included
in the LaTeX.  At minimum we output a box–plot of absolute bias by method.

Usage
-----
$ python3 scripts/plot_results.py --csv paper_assets/results_synthetic.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_detailed_rows(csv_path: Path) -> List[Dict[str, str]]:
    """Return list of dict rows until first blank line (summary starts after)."""
    rows: List[Dict[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Stop when we reach blank line (summary section separator)
            if not any(row.values()):
                break  # reached blank separator
            # Guard against second header row accidentally parsed
            if row.get("method", "").lower() == "method":
                break
            rows.append(row)
    return rows


def plot_abs_bias_boxplot(rows: List[Dict[str, str]], out_path: Path) -> None:
    # Gather abs_bias values per method
    data: Dict[str, List[float]] = {}
    for r in rows:
        m = r["method"].strip()
        # Skip rows that are actually summary header or invalid (numeric etc.)
        if "-" not in m:
            continue
        try:
            data.setdefault(m, []).append(float(r["abs_bias"]))
        except (ValueError, TypeError):
            continue

    methods = sorted(data.keys())
    values = [data[m] for m in methods]

    plt.figure(figsize=(4, 3))
    plt.boxplot(values, labels=methods, showfliers=False)
    plt.ylabel("Absolute bias")
    plt.title("Distribution of |Bias| across 50 seeds")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] wrote {out_path}")


def plot_rmse_bar(summary_csv: Path, out_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(summary_csv, skip_blank_lines=True)
    if "rmse" not in df.columns:
        return
    plt.figure(figsize=(4,3))
    plt.bar(df["method"], df["rmse"], color="skyblue")
    plt.ylabel("RMSE")
    plt.title("RMSE by method (50 seeds)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] wrote {out_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("paper_assets"))
    args = parser.parse_args()

    rows = load_detailed_rows(args.csv)
    plot_abs_bias_boxplot(rows, args.out_dir / "abs_bias_boxplot.pdf")

    # RMSE bar plot uses summary section (read with pandas)
    plot_rmse_bar(args.csv, args.out_dir / "rmse_bar.pdf")


if __name__ == "__main__":
    main()
