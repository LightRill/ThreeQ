# DThreeQ MNIST Focused Screen

Focused MNIST screen for DThreeQ variants using legacy-scale hidden width and a learning-rate sweep. This is the next step after the small-budget MNIST stress test.

## Run Configuration

| train_subset   | test_subset   | n_epochs   | batch_size   | hidden_sizes   | device   |
|:---------------|:--------------|:-----------|:-------------|:---------------|:---------|
| 1e+04          | 2000          | 30         | 64           | [500]          | cuda     |

## Summary By Variant

| variant                            | family   | best_test_error_mean   | best_test_error_std   | final_test_error_mean   | final_train_error_mean   | state_delta_mean   | saturation_mean   | duration_sec_mean   |
|:-----------------------------------|:---------|:-----------------------|:----------------------|:------------------------|:-------------------------|:-------------------|:------------------|:--------------------|
| dthreeq_ep_tanh_nudge0p1_lr5e3     | dthreeq  | 0.1545                 | 0.001726              | 0.4331                  | 0.2156                   | 0.07465            | 0.001975          | 310.8               |
| dthreeq_ep_nudge0p1_lr3e3          | dthreeq  | 0.1606                 | 0.006215              | 0.1646                  | 0.1513                   | 0.04981            | 5.854e-07         | 304.1               |
| dthreeq_ep_relu_nudge0p1_lr1e3     | dthreeq  | 0.3933                 | 0.009322              | 0.8865                  | 0.8108                   | 0.09362            | 0.03903           | 246.2               |
| dthreeq_ep_relu_nudge0p1_lr3e3     | dthreeq  | 0.4097                 | 0.004834              | 0.8984                  | 0.8982                   | 0.1031             | 0.08292           | 251.7               |
| dthreeq_ep_sigmoid_nudge0p1_lr5e3  | dthreeq  | 0.6814                 | 0.01623               | 0.853                   | 0.7294                   | 0.0624             | 0                 | 261.7               |
| dthreeq_ep_sigmoid_nudge0p1_lr3e3  | dthreeq  | 0.7236                 | 0.01174               | 0.8613                  | 0.643                    | 0.05888            | 9.757e-08         | 259.6               |
| dthreeq_ep_sigmoid_nudge0p2_lr3e3  | dthreeq  | 0.7249                 | 0.04592               | 0.8718                  | 0.6665                   | 0.07949            | 9.757e-08         | 259.8               |
| dthreeq_ep_sigmoid_nudge0p05_lr3e3 | dthreeq  | 0.7285                 | 0.01933               | 0.853                   | 0.6717                   | 0.03708            | 0                 | 321.8               |

## Notes

The dashed 0.10 test-error line in the figures marks the 90% accuracy threshold reported by external DThreeQ experiments.
