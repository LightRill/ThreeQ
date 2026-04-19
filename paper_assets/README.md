# Paper Assets

This directory contains paper-facing ThreeQ figures, tables, and a Chinese summary generated from completed experiments.

Run:

```bash
python paper_assets/scripts/build_paper_assets.py
```

The script does not train models. It reads existing CSV/PNG artifacts under `experiments/` and selected legacy convergence outputs, then writes:

- `figures/`: numbered figures for inference convergence, training convergence, MNIST performance, DThreeQ progress, mechanism diagnostics, and legacy CNNThreeQ structure exploration.
- `tables/`: compact CSV/Markdown tables.
- `data/`: figure manifests and copied provenance metadata.
- `PAPER_SUMMARY.md`: Chinese paper-writing summary.
- `latex/threeq_systematic_report.tex`: paper-ready systematic LaTeX report covering ThreeQ, EPThreeQ, DThreeQ, CNNThreeQ, convergence/divergence, MNIST behavior, and mechanism diagnostics.
- `latex/threeq_convergence_report.tex`: compatibility copy of the systematic report, retaining the previous convergence-report entry point.

Additional convergence report build:

```bash
python paper_assets/scripts/build_convergence_report.py
python paper_assets/scripts/build_systematic_report.py
cd paper_assets/latex && xelatex -interaction=nonstopmode -halt-on-error threeq_systematic_report.tex
```
