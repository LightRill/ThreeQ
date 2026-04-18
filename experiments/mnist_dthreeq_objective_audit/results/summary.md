# DThreeQ MNIST Focused Screen

Focused MNIST screen for DThreeQ variants using legacy-scale hidden width and a learning-rate sweep. This is the next step after the small-budget MNIST stress test.

## Run Configuration

| train_subset   | test_subset   | n_epochs   | batch_size   | hidden_sizes   | device   |
|:---------------|:--------------|:-----------|:-------------|:---------------|:---------|
| 1e+04          | 2000          | 30         | 64           | [500]          | cuda     |

## Summary By Variant

| variant                               | family                 | best_test_error_mean   | best_test_error_std   | final_test_error_mean   | final_train_error_mean   | state_delta_mean   | saturation_mean   | duration_sec_mean   |
|:--------------------------------------|:-----------------------|:-----------------------|:----------------------|:------------------------|:-------------------------|:-------------------|:------------------|:--------------------|
| dthreeq_ep_nudge0p1_lr3e3             | dthreeq                | 0.1606                 | 0.006215              | 0.1646                  | 0.1513                   | 0.04981            | 5.854e-07         | 304.6               |
| dthreeq_plus_energy_direct_lr1e3      | dthreeq_plus_energy    | 0.4509                 | 0.04109               | 0.5483                  | 0.5305                   | 0.09354            | 0.898             | 290.5               |
| dthreeq_plus_energy_direct_lr3e4      | dthreeq_plus_energy    | 0.5002                 | 0.02451               | 0.5002                  | 0.522                    | 0.107              | 0.3102            | 321.3               |
| dthreeq_plus_energy_nudge0p1_lr1e3    | dthreeq_plus_energy    | 0.543                  | 0.04903               | 0.6318                  | 0.6222                   | 0.07305            | 0.9011            | 253.3               |
| dthreeq_forward_target_direct_lr1e3   | dthreeq_forward_target | 0.7217                 | 0.02555               | 0.7217                  | 0.745                    | 0.1279             | 0.00207           | 253                 |
| dthreeq_forward_target_nudge0p1_lr1e3 | dthreeq_forward_target | 0.7896                 | 0.0221                | 0.7896                  | 0.8056                   | 0.09279            | 0.002072          | 252.2               |
| dthreeq_bidir_target_direct_lr1e3     | dthreeq_bidir_target   | 0.8201                 | 0.02106               | 0.8201                  | 0.8372                   | 0.1378             | 0.002024          | 252.5               |

## Notes

The dashed 0.10 test-error line in the figures marks the 90% accuracy threshold reported by external DThreeQ experiments.
