# DThreeQ MNIST Focused Screen

Focused MNIST screen for DThreeQ variants using legacy-scale hidden width and a learning-rate sweep. This is the next step after the small-budget MNIST stress test.

## Run Configuration

| train_subset   | test_subset   | n_epochs   | batch_size   | hidden_sizes   | device   |
|:---------------|:--------------|:-----------|:-------------|:---------------|:---------|
| 1e+04          | 2000          | 30         | 64           | [500]          | cuda     |

## Summary By Variant

| variant                                  | family   | best_test_error_mean   | best_test_error_std   | final_test_error_mean   | final_train_error_mean   | state_delta_mean   | saturation_mean   | duration_sec_mean   |
|:-----------------------------------------|:---------|:-----------------------|:----------------------|:------------------------|:-------------------------|:-------------------|:------------------|:--------------------|
| dthreeq_ep_tanh_nudge0p1_lr5e3           | dthreeq  | 0.1545                 | 0.001726              | 0.4331                  | 0.2156                   | 0.07465            | 0.002622          | 332.5               |
| dthreeq_ep_nudge0p1_lr3e3                | dthreeq  | 0.1606                 | 0.006215              | 0.1646                  | 0.1513                   | 0.04981            | 6.83e-07          | 334.6               |
| dthreeq_ep_clip01_nudge0p2_lr3e3         | dthreeq  | 0.2969                 | 0.08286               | 0.5327                  | 0.5165                   | 0.06414            | 0.9144            | 296                 |
| dthreeq_ep_clip01_nudge0p1_lr3e3         | dthreeq  | 0.3293                 | 0.06249               | 0.8423                  | 0.8591                   | 0.05887            | 0.9813            | 293.9               |
| dthreeq_ep_clip01_nudge0p1_lr5e3_decay20 | dthreeq  | 0.3318                 | 0.07423               | 0.8733                  | 0.8699                   | 0.05899            | 0.9817            | 291                 |
| dthreeq_ep_clip01_nudge0p05_lr3e3        | dthreeq  | 0.4028                 | 0.05662               | 0.8691                  | 0.8533                   | 0.03818            | 0.9806            | 299.2               |

## Notes

The dashed 0.10 test-error line in the figures marks the 90% accuracy threshold reported by external DThreeQ experiments.
