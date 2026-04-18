# Consolidated ThreeQ Experiment Analysis

This report materializes all currently saved training-run data from:

- `experiments/minimal_suite/results/results_raw.csv`
- `experiments/dthreeq_suite/results/results_raw.csv`

The materialized CSV files do not contain per-epoch histories. The figures
therefore visualize the full available per-trial data: best/final error,
seed spread, learning-rate split, state convergence diagnostics, weight
scale, update scale, and runtime.

## Figures

- `figures/minimal_error_by_variant.png`
- `figures/minimal_diagnostics_by_variant.png`
- `figures/minimal_seed_best_error.png`
- `figures/dthreeq_error_by_variant.png`
- `figures/dthreeq_diagnostics_by_variant.png`
- `figures/dthreeq_seed_best_error.png`
- `figures/dthreeq_best_error_vs_lr.png`
- `figures/dthreeq_best_error_heatmap.png`
- `figures/combined_top_configs.png`

## Minimal Suite

| variant           | best_valid_error_mean   | best_valid_error_std   | final_valid_error_mean   | rho_mean   | saturation_mean   | state_delta_l2_mean   | weight_abs_mean_mean   | duration_sec_mean   |
|:------------------|:------------------------|:-----------------------|:-------------------------|:-----------|:------------------|:----------------------|:-----------------------|:--------------------|
| ep_cost_w50       | 0.1783                  | 0.0325                 | 0.2817                   | 0.641      | 0.4084            | 0.2618                | 1.0956                 | 111.676             |
| ep_cost_w10       | 0.2767                  | 0.1674                 | 0.2817                   | 0.4808     | 0.4628            | 0.1604                | 0.2929                 | 85.3098             |
| direct_cost_w50   | 0.2867                  | 0.2152                 | 0.4933                   | 0.5558     | 0.2232            | 0.2813                | 0.2217                 | 164.9663            |
| ep_tied_w10       | 0.3167                  | 0.1962                 | 0.6667                   | 0.6607     | 0.569             | 0.166                 | 0.3626                 | 81.1396             |
| direct_linear_w10 | 0.3267                  | 0.1828                 | 0.5267                   | 0.4814     | 0.2803            | 0.042                 | 0.22                   | 104.0073            |
| direct_cost_w10   | 0.3933                  | 0.2036                 | 0.46                     | 0.4713     | 0.2915            | 0.162                 | 0.2169                 | 107.4008            |
| ep_cost_w2        | 0.4117                  | 0.2093                 | 0.52                     | 0.3787     | 0.5312            | 0.0461                | 0.2218                 | 77.3743             |

Main conclusions:

- `ep_cost_w50` gives the best mean best-validation error, but its final
  validation error is worse than its best checkpoint and it has the largest
  weight scale in this suite. It is an early-best winner, not a stable
  endpoint.
- `ep_cost_w10` is less aggressive and more stable than `ep_cost_w50`,
  but does not reach the same best checkpoint.
- `ep_cost_w2` is too weak: the weak phase does not supply enough training
  signal.
- Strict transpose sharing (`ep_tied_w10`) still hurts final accuracy in
  this minimal implementation, so tying alone is not a sufficient fix.

## DThreeQ Screen

| variant                     | best_test_error_mean   | best_test_error_std   | best_test_error_min   | final_test_error_mean   | state_delta_mean   | saturation_mean   | weight_abs_mean_mean   | duration_sec_mean   |
|:----------------------------|:-----------------------|:----------------------|:----------------------|:------------------------|:-------------------|:------------------|:-----------------------|:--------------------|
| ep_gradual_100_0p01_plus    | 0.1565                 | 0.0518                | 0.1178                | 0.2264                  | 0.0933             | 0.0007            | 0.3143                 | 497.3748            |
| ep_nudge_0p01_plus          | 0.1931                 | 0.1061                | 0.1154                | 0.2023                  | 0.0233             | 0.0               | 0.1808                 | 254.413             |
| bp_tanh                     | 0.3365                 | 0.2177                | 0.1178                | 0.3407                  | n/a                | n/a               | n/a                    | 13.508              |
| dplus_direct                | 0.3502                 | 0.1137                | 0.2163                | 0.3502                  | 0.1779             | 0.0               | 0.06                   | 257.7372            |
| dplus_nudge_0p1_plus        | 0.3506                 | 0.1133                | 0.2188                | 0.3506                  | 0.1434             | 0.0               | 0.0598                 | 240.8813            |
| dplus_gradual_100_0p01_plus | 0.3554                 | 0.1089                | 0.2404                | 0.3554                  | 0.1116             | 0.0               | 0.0596                 | 494.4412            |
| dplus_nudge_0p01_plus       | 0.3578                 | 0.1069                | 0.2548                | 0.3578                  | 0.0284             | 0.0               | 0.0588                 | 230.2441            |
| dplus_nudge_0p001_plus      | 0.359                  | 0.1058                | 0.2668                | 0.359                   | 0.0054             | 0.0               | 0.0586                 | 231.8193            |
| dplus_nudge_0p01_plusminus  | 0.359                  | 0.1058                | 0.2668                | 0.359                   | 0.0301             | 0.0               | 0.0586                 | 306.1952            |

Best DThreeQ/BP/EP settings by variant and LR:

| variant                  | weight_lr   | best_test_error_mean   | best_test_error_std   | final_test_error_mean   | final_minus_best_mean   |
|:-------------------------|:------------|:-----------------------|:----------------------|:------------------------|:------------------------|
| ep_gradual_100_0p01_plus | 1e-03       | 0.129                  | 0.0084                | 0.3742                  | 0.2452                  |
| bp_tanh                  | 1e-03       | 0.1362                 | 0.0193                | 0.1522                  | 0.016                   |
| ep_nudge_0p01_plus       | 1e-03       | 0.1362                 | 0.0193                | 0.1595                  | 0.0232                  |
| ep_gradual_100_0p01_plus | 1e-04       | 0.1426                 | 0.0204                | 0.1587                  | 0.016                   |
| ep_gradual_100_0p01_plus | 5e-05       | 0.145                  | 0.0332                | 0.1571                  | 0.012                   |
| ep_nudge_0p01_plus       | 1e-04       | 0.1538                 | 0.0411                | 0.1562                  | 0.0024                  |
| ep_nudge_0p01_plus       | 5e-05       | 0.1755                 | 0.0537                | 0.1835                  | 0.008                   |
| ep_gradual_100_0p01_plus | 1e-05       | 0.2091                 | 0.086                 | 0.2155                  | 0.0064                  |
| bp_tanh                  | 1e-04       | 0.2179                 | 0.0824                | 0.2187                  | 0.0008                  |
| ep_nudge_0p01_plus       | 1e-05       | 0.3069                 | 0.173                 | 0.3101                  | 0.0032                  |
| dplus_direct             | 1e-03       | 0.3277                 | 0.1513                | 0.3277                  | 0.0                     |
| dplus_nudge_0p1_plus     | 1e-03       | 0.3285                 | 0.1504                | 0.3285                  | 0.0                     |

Main conclusions:

- The current Dplus/DThreeQ update does not pass the two-moons screen.
  Across learning rates, the Dplus variants stay around 0.35 mean best
  test error and many runs remain near the 0.5 chance-error regime.
- The strongest DThreeQ-screen rows are still EP and BP controls. The best
  mean row is `ep_gradual_100_0p01_plus` at `weight_lr=1e-3`, but its final
  error is unstable; the BP/EP rows at smaller learning rates are more
  stable endpoints.
- Dplus has near-zero saturation, so the failure is unlikely to be caused
  by tanh state clipping alone. The likely issue is update/objective
  alignment: the residual target, detach path, or sign/beta construction is
  not producing a useful descent direction.

## Cross-Suite Ranking

| suite   | config                            | best_error_mean   | best_error_std   | final_error_mean   |
|:--------|:----------------------------------|:------------------|:-----------------|:-------------------|
| dthreeq | ep_gradual_100_0p01_plus lr=1e-03 | 0.129             | 0.0084           | 0.3742             |
| dthreeq | bp_tanh lr=1e-03                  | 0.1362            | 0.0193           | 0.1522             |
| dthreeq | ep_nudge_0p01_plus lr=1e-03       | 0.1362            | 0.0193           | 0.1595             |
| dthreeq | ep_gradual_100_0p01_plus lr=1e-04 | 0.1426            | 0.0204           | 0.1587             |
| dthreeq | ep_gradual_100_0p01_plus lr=5e-05 | 0.145             | 0.0332           | 0.1571             |
| dthreeq | ep_nudge_0p01_plus lr=1e-04       | 0.1538            | 0.0411           | 0.1562             |
| dthreeq | ep_nudge_0p01_plus lr=5e-05       | 0.1755            | 0.0537           | 0.1835             |
| minimal | ep_cost_w50                       | 0.1783            | 0.0325           | 0.2817             |
| dthreeq | ep_gradual_100_0p01_plus lr=1e-05 | 0.2091            | 0.086            | 0.2155             |
| dthreeq | bp_tanh lr=1e-04                  | 0.2179            | 0.0824           | 0.2187             |
| minimal | ep_cost_w10                       | 0.2767            | 0.1674           | 0.2817             |
| minimal | direct_cost_w50                   | 0.2867            | 0.2152           | 0.4933             |

Overall decision:

- Keep the minimal ThreeQ/EP suite as the comparable baseline.
- Do not scale the current Dplus/DThreeQ rule to MNIST/Fashion yet.
- The next experiment should be a mechanism diagnostic rather than a larger
  benchmark: compare Dplus update vectors against BP/EP update vectors
  using cosine similarity, norm ratio, sign agreement, and a one-step loss
  decrease test on the same mini-batches.
