# DThreeQ MNIST Focused Screen

Focused MNIST screen for DThreeQ variants using legacy-scale hidden width and a learning-rate sweep. This is the next step after the small-budget MNIST stress test.

## Run Configuration

| train_subset   | test_subset   | n_epochs   | batch_size   | hidden_sizes   | device   |
|:---------------|:--------------|:-----------|:-------------|:---------------|:---------|
| 1e+04          | 2000          | 12         | 64           | [500]          | cuda     |

## Summary By Variant

| variant                       | family                    | best_test_error_mean   | best_test_error_std   | final_test_error_mean   | final_train_error_mean   | state_delta_mean   | saturation_mean   | duration_sec_mean   |
|:------------------------------|:--------------------------|:-----------------------|:----------------------|:------------------------|:-------------------------|:-------------------|:------------------|:--------------------|
| bp_mlp_500_adam               | bp_mlp                    | 0.05469                | 0.005524              | 0.05859                 | 0.007464                 | n/a                | n/a               | 40.42               |
| dthreeq_ep_nudge0p1_lr1e3     | dthreeq                   | 0.1965                 | 0.02037               | 0.1965                  | 0.2019                   | 0.05612            | 9.513e-05         | 122.4               |
| dthreeq_ep_nudge0p01_lr1e3    | dthreeq                   | 0.2524                 | 0.006215              | 0.2737                  | 0.2441                   | 0.01013            | 1.912e-05         | 157                 |
| dthreeq_ep_nudge0p01_lr3e4    | dthreeq                   | 0.3389                 | 0.001381              | 0.3389                  | 0.3528                   | 0.0115             | 4.781e-06         | 159                 |
| dthreeq_dplus_direct_lr1e3    | dthreeq                   | 0.896                  | 0.001381              | 0.8967                  | 0.9045                   | 0.1611             | 0.002005          | 122.7               |
| dthreeq_dplus_layergain_lr1e3 | dthreeq_variant_objective | 0.896                  | 0.001381              | 0.897                   | 0.9045                   | 0.1611             | 0.002003          | 123.4               |
| dthreeq_dplus_direct_lr3e4    | dthreeq                   | 0.9094                 | 0.003798              | 0.9097                  | 0.9162                   | 0.1807             | 0.002121          | 122.2               |

## Notes

The dashed 0.10 test-error line in the figures marks the 90% accuracy threshold reported by external DThreeQ experiments.
