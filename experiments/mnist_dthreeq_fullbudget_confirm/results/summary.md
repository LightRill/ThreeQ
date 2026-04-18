# DThreeQ MNIST Supervision And Budget Screen

This screen isolates four hypotheses: input reconstruction dominance, MSE one-hot versus CE-style output supervision, extra readout or label encoding differences, and high-LR stabilization by schedule or best checkpoint selection.

## Run Configuration

| train_subset   | test_subset   | n_epochs   | batch_size   | hidden_sizes   | device   |
|:---------------|:--------------|:-----------|:-------------|:---------------|:---------|
| 6e+04          | 1e+04         | 30         | 64           | [500]          | cuda     |

## Summary By Variant

| variant                                          | family          | best_test_error_mean   | best_test_error_std   | selected_test_error_mean   | selected_test_accuracy_mean   | final_test_error_mean   | final_state_test_error_mean   | final_train_error_mean   | state_delta_mean   | saturation_mean   | input_recon_energy_frac_mean   | weighted_input_recon_energy_frac_mean   | weight_update_rel_mean_mean   | duration_sec_mean   |
|:-------------------------------------------------|:----------------|:-----------------------|:----------------------|:---------------------------|:------------------------------|:------------------------|:------------------------------|:-------------------------|:-------------------|:------------------|:-------------------------------|:----------------------------------------|:------------------------------|:--------------------|
| dthreeq_ep_signed_nudge0p1_lr3e3_restorebest     | dthreeq         | 0.1197                 | 0.003871              | 0.1197                     | 0.8803                        | 0.1621                  | 0.1621                        | 0.3265                   | 0.06889            | 0.3578            | 0.5352                         | 0.5352                                  | 0.00196                       | 1403                |
| dthreeq_ep_ce_nudge0p1_lr3e3_restorebest         | dthreeq         | 0.121                  | 0.0009852             | 0.121                      | 0.879                         | 0.1852                  | 0.1852                        | 0.2171                   | 0.04735            | 0.2568            | 0.7125                         | 0.7125                                  | 0.0008679                     | 1563                |
| dthreeq_ep_ce_nudge0p1_lr5e3_decay15_restorebest | dthreeq         | 0.1238                 | 0.002111              | 0.1238                     | 0.8762                        | 0.7777                  | 0.7777                        | 0.6263                   | 0.07468            | 0.603             | 0.573                          | 0.573                                   | 0.005629                      | 574.9               |
| dthreeq_ep_ce_nudge0p1_lr3e3                     | dthreeq         | 0.121                  | 0.0009852             | 0.1852                     | 0.8148                        | 0.1852                  | 0.1852                        | 0.2171                   | 0.04735            | 0.2568            | 0.7125                         | 0.7125                                  | 0.0008679                     | 1542                |
| dthreeq_ep_readout_nudge0p1_lr3e3                | dthreeq_readout | 0.1238                 | 0.000563              | 0.7582                     | 0.2418                        | 0.7582                  | 0.8368                        | 0.8245                   | 0.2893             | 0.7104            | 0.3488                         | 0.3488                                  | 0.006852                      | 1356                |
| dthreeq_ep_nudge0p1_lr3e3                        | dthreeq         | 0.1477                 | 0.002533              | 0.8368                     | 0.1632                        | 0.8368                  | 0.8368                        | 0.8618                   | 0.2893             | 0.7104            | 0.3488                         | 0.3488                                  | 0.006852                      | 1576                |

## Notes

`selected_test_error` equals `best_test_error` only for restore-best variants; otherwise it equals final test error. The dashed 0.10 line in the figures marks 90% accuracy.
