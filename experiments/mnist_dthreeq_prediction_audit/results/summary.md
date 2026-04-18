# DThreeQ MNIST Focused Screen

Focused MNIST screen for DThreeQ variants using legacy-scale hidden width and a learning-rate sweep. This is the next step after the small-budget MNIST stress test.

## Run Configuration

| train_subset   | test_subset   | n_epochs   | batch_size   | hidden_sizes   | device   |
|:---------------|:--------------|:-----------|:-------------|:---------------|:---------|
| 1e+04          | 2000          | 30         | 64           | [500]          | cuda     |

## Summary By Variant

| variant                                 | family   | best_test_error_mean   | best_test_error_std   | final_test_error_mean   | final_train_error_mean   | state_delta_mean   | saturation_mean   | duration_sec_mean   |
|:----------------------------------------|:---------|:-----------------------|:----------------------|:------------------------|:-------------------------|:-------------------|:------------------|:--------------------|
| dthreeq_ep_nudge0p1_lr3e3               | dthreeq  | 0.1606                 | 0.006215              | 0.1646                  | 0.1513                   | 0.04981            | 6.83e-07          | 288.1               |
| dthreeq_ep_linear_tanh_nudge0p1_lr3e3   | dthreeq  | 0.1624                 | 0.00656               | 0.1646                  | 0.1516                   | 0.05204            | 9.757e-08         | 251.3               |
| dthreeq_ep_initact_tanh_nudge0p1_lr3e3  | dthreeq  | 0.1638                 | 0.007941              | 0.1689                  | 0.1565                   | 0.05236            | 0                 | 287                 |
| dthreeq_ep_linear_tanh_nudge0p1_lr1e3   | dthreeq  | 0.1682                 | 0.007941              | 0.1711                  | 0.1654                   | 0.05285            | 6.996e-05         | 287.8               |
| dthreeq_ep_linear_clip01_nudge0p1_lr3e3 | dthreeq  | 0.416                  | 0.03038               | 0.4268                  | 0.427                    | 0.04997            | 0.9704            | 246.8               |
| dthreeq_ep_linear_clip01_nudge0p1_lr1e3 | dthreeq  | 0.4226                 | 0.01899               | 0.4434                  | 0.4368                   | 0.05303            | 0.9434            | 253.3               |

## Notes

The dashed 0.10 test-error line in the figures marks the 90% accuracy threshold reported by external DThreeQ experiments.
