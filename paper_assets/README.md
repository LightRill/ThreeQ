# Paper Assets

This directory contains paper-facing ThreeQ figures, tables, and a Chinese summary generated from completed experiments.

Run:

```bash
python paper_assets/scripts/build_paper_assets.py
```

The script does not train models. It reads existing CSV/PNG artifacts under `experiments/` and selected legacy convergence outputs, then writes:

- `figures/`: numbered figures for inference convergence, training convergence, MNIST performance, DThreeQ progress, and mechanism diagnostics.
- `tables/`: compact CSV/Markdown tables.
- `data/`: figure manifests and copied provenance metadata.
- `PAPER_SUMMARY.md`: Chinese paper-writing summary.
- `latex/threeq_convergence_report.tex`: paper-ready LaTeX report for ThreeQ inference/training convergence, including rho > 1 divergence evidence.

Additional convergence report build:

```bash
python paper_assets/scripts/build_convergence_report.py
cd paper_assets/latex && xelatex -interaction=nonstopmode -halt-on-error threeq_convergence_report.tex
```
