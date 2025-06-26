# Adaptive Supergeo Design (ASD)

A scalable, open-source framework for designing high-quality geographic experiments at marketing scale. Pre-print: http://arxiv.org/abs/2506.20499

Geographic ("geo-lift") experiments remain the gold-standard for measuring the incremental return on ad spend (**iROAS**) of large advertising campaigns, yet their design is non-trivial: the number of markets is small, heterogeneity is large, and the exact Supergeo partitioning problem is NP-hard.  **ASD** solves these challenges with a two-stage approach:

1. A bespoke graph-neural network learns geo-embeddings and proposes a concise candidate set of *supergeos*.
2. A CP-SAT solver selects a treatment/control partition that balances both baseline outcomes and effect modifiers.

Under mild community-structure assumptions ASD’s objective is within $(1+\varepsilon)$ of optimal.  Simulations with up to **1 000** Designated Market Areas (DMAs) finish in **< 60 s** on a single core, keep every media dollar in market, and cut iROAS bias by ≈ 30 % versus existing methods.

---

## Quick start

```bash
# clone your fork then
python3 -m venv venv && source venv/bin/activate
pip install -e .[dev]            # installs ASD and its dependencies

# design an experiment (example synthetic data)
python scripts/asd_design.py  \
       --input paper_assets/synthetic_units.csv \
       --output my_design.csv

# run the scalability benchmark & plot (Fig 2 of the paper)
python scripts/benchmark_scalability.py \
       --csv paper_assets/runtime_results.csv \
       --pdf paper_assets/scalability_plot.pdf
```

> **Note** `benchmark_scalability.py` automatically skips the (slow) exact ILP solver for *N* > 400.

---

## Repository layout

| Path | Purpose |
|------|---------|
| `Adaptive_Supergeo_Design/` | LaTeX source for the accompanying paper |
| `paper_assets/` | Pre-generated figures & benchmark CSVs |
| `scripts/` | CLI entry-points (`asd_design.py`, `asd_analysis.py`, `benchmark_scalability.py`, …) |
| `src/asd/` | Core Python implementation (GNN, clustering, optimisation) |
| `tests/` | Unit tests (run with `pytest`) |

---

## Building the paper

```bash
cd Adaptive_Supergeo_Design
latexmk -pdf main.tex   # compile, repeat to resolve references
```
The build produces `main.pdf`, which is the version submitted to *Marketing Science*.

---

## Contributing
Pull requests are welcome!  Please open an issue first to discuss major changes.

---

## License
Apache 2.0 — see [`LICENSE`](LICENSE) for details.

## Citation
If you use this codebase in academic work, please cite:

```
@article{Shaw2025ASD,
  title   = {Adaptive Supergeo Design: A Scalable Framework for Geographic Marketing Experiments},
  author  = {Charles Shaw},
  journal = {arXiv preprint},
  year    = {2025}
}
```
