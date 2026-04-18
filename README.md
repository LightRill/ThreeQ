# ThreeQ

This repository studies ThreeQ-style energy models based on bidirectional local
prediction, truncated-gradient state inference, and EP/DThreeQ training
variants.

## Current Paper-Facing Entry Points

- `RESEARCH_REPORT.md`: living research report with theory, experiment history,
  conclusions, and next-step plans.
- `paper_assets/`: paper-oriented figures, tables, and Chinese summary generated
  from completed experiments.
- `paper_assets/PAPER_SUMMARY.md`: concise writing aid for the current paper.
- `paper_assets/scripts/build_paper_assets.py`: regenerates the paper figures and
  tables from existing result CSVs and selected cached convergence evidence.

## Main Code Layers

| Path | Role |
|---|---|
| `Provement/` | Fixed-point and local convergence proof notes. |
| `AllConnected3QNotTrained/`, `AllConnected3QTrained/` | Early convergence demonstrations for the theoretical mother model. |
| `Base3Q/`, `EPBase3Q/` | Legacy MLP ThreeQ and EPThreeQ implementations used for MNIST reproduction. |
| `CNN3Q/`, `EPCNN3Q/` | Legacy convolutional/transpose-convolutional symmetric prototypes. |
| `threeq_common/` | Shared implementation for comparable ThreeQ, EPThreeQ, DThreeQ, and mechanism diagnostics. |
| `experiments/` | Reproducible radas experiment suites and materialized results. |
| `paper_assets/` | Consolidated tables and figures for paper writing. |

## Rebuilding Paper Assets

The asset build does not run training:

```bash
python paper_assets/scripts/build_paper_assets.py
```

For large new training or sweep experiments, use the radas workflow and then
update `RESEARCH_REPORT.md` and `paper_assets/`.

## Current Headline Results

- ThreeQ inference convergence follows the local spectral-radius condition:
  runs with $\rho(J)<1$ show fast residual decay, while larger local spectral
  radius slows or prevents convergence.
- Legacy `EPBase3Q` is substantially stronger than direct `Base3Q` on MNIST:
  5k/1k, 8 epoch best accuracy is 77.5% vs 60.2%; 10k/2k, 15 epoch EP tuning
  reaches about 86.5%.
- DThreeQ improves from near-random small-budget behavior to about 86.5% on
  MNIST 10k/2k with CE-style output nudging, and about 88.0% selected accuracy
  on full MNIST with restore-best. Final checkpoints still degrade.
- Current Dplus/residual-delta updates are EP-like but not BP-like: forward
  cosine against BP is around 0.03 to 0.08 for plus targets, and plusminus
  cancels.
