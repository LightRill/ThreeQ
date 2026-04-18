# DThreeQ MNIST Focused Screen

Focused MNIST screen for DThreeQ variants using legacy-scale hidden width and a learning-rate sweep. This is the next step after the small-budget MNIST stress test.

## Run Configuration

| train_subset   | test_subset   | n_epochs   | batch_size   | hidden_sizes   | device   |
|:---------------|:--------------|:-----------|:-------------|:---------------|:---------|
| 1e+04          | 2000          | 30         | 64           | [500]          | cuda     |

## Summary By Variant

| variant                           | family   | best_test_error_mean   | best_test_error_std   | final_test_error_mean   | final_train_error_mean   | state_delta_mean   | saturation_mean   | duration_sec_mean   |
|:----------------------------------|:---------|:-----------------------|:----------------------|:------------------------|:-------------------------|:-------------------|:------------------|:--------------------|
| dthreeq_ep_nudge0p1_lr3e3         | dthreeq  | 0.1606                 | 0.006215              | 0.1646                  | 0.1513                   | 0.04981            | 5.854e-07         | 280.7               |
| dthreeq_ep_nudge0p2_lr3e3         | dthreeq  | 0.1621                 | 0.01105               | 0.1689                  | 0.1568                   | 0.06709            | 2.322e-05         | 272.1               |
| dthreeq_ep_nudge0p1_lr1e3_steps20 | dthreeq  | 0.1714                 | 0.007596              | 0.1729                  | 0.1615                   | 0.06721            | 0.0001654         | 416.2               |
| dthreeq_ep_nudge0p1_lr1e3         | dthreeq  | 0.1716                 | 0.007941              | 0.1721                  | 0.1657                   | 0.05219            | 2.712e-05         | 318.4               |
| dthreeq_ep_nudge0p05_lr1e3        | dthreeq  | 0.1716                 | 0.00587               | 0.1743                  | 0.1651                   | 0.03309            | 2.556e-05         | 314.9               |
| dthreeq_ep_nudge0p2_lr1e3         | dthreeq  | 0.179                  | 0.01692               | 0.1804                  | 0.1746                   | 0.07056            | 8.937e-05         | 245.8               |

## Notes

The dashed 0.10 test-error line in the figures marks the 90% accuracy threshold reported by external DThreeQ experiments.
