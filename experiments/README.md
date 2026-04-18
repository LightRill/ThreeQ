# Experiments

This directory contains reproducible radas experiment suites. Older exploratory
code is retained when it is needed to reproduce a result in `RESEARCH_REPORT.md`
or `paper_assets/`.

## Paper-Core Suites

| Suite | Purpose |
|---|---|
| `minimal_suite/` | Comparable two-moons ThreeQ/EPThreeQ baselines. |
| `mnist_legacy_repro/` | Direct reproduction-style screen using legacy `Base3Q` and `EPBase3Q` classes. |
| `mnist_epthreeq_tune/` | EPThreeQ MNIST tuning around the legacy architecture. |
| `mnist_dthreeq_supervision_budget/` | DThreeQ CE/readout/input-energy supervision audit on MNIST 10k/2k. |
| `mnist_dthreeq_fullbudget_confirm/` | DThreeQ full MNIST confirmation and restore-best check. |
| `mechanism_diagnostic/` | BP/EP/Dplus update-vector diagnostic. |
| `dplus_fix_diagnostic/` | Residual normalization, scale calibration, plus-only, layer-gain, and one-step loss diagnostic. |

## Provenance / Superseded Screens

These suites are useful for tracing the research path, but their conclusions are
superseded by the paper-core suites above.

| Suite | Status |
|---|---|
| `mnist_suite/` | Small-budget stress screen; useful as a negative control, not a legacy MNIST benchmark. |
| `mnist_dthreeq_focus/` | First legacy-scale DThreeQ screen; superseded by longrun and supervision-budget suites. |
| `mnist_dthreeq_longrun/` | Establishes the stable EP-nudge DThreeQ baseline; kept for the improvement path. |
| `mnist_dthreeq_objective_audit/` | Shows plus-energy and target-matching objectives do not beat EP-nudge. |
| `mnist_dthreeq_activation_boost/` | Shows high learning rate is transient and sigmoid/relu variants are worse. |
| `mnist_dthreeq_clip01_boost/` | Shows nonnegative/clipped DThreeQ states saturate and underperform. |
| `mnist_dthreeq_prediction_audit/` | Shows linear edge prediction and activated initialization do not solve the plateau. |
| `dthreeq_suite/` | Two-moons DThreeQ feasibility screen. |
| `analysis_summary/` | Historical consolidated plots; `paper_assets/` is now the paper-facing replacement. |

## Cleanup Policy

Generated caches, downloaded datasets, `__pycache__`, and ignored legacy figure
directories should not be committed. Paper-critical figures and compact CSVs are
materialized under `paper_assets/` instead.
