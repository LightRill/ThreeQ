# Legacy MNIST Reproduction

Direct reproduction-style screen using original `Base3Q/ThreeQ.py` and `EPBase3Q/ThreeQ.py` network classes with net1 architecture and hyperparameters, on a 5k/1k MNIST subset for 8 epochs.

## Summary

| variant                          | method    | best_valid_error_mean   | final_valid_error_mean   | final_train_error_mean   | rho_mean_mean   | rho_max_mean   | forward_update_rel_mean_mean   | backward_update_rel_mean_mean   | duration_sec_mean   |
|:---------------------------------|:----------|:------------------------|:-------------------------|:-------------------------|:----------------|:---------------|:-------------------------------|:--------------------------------|:--------------------|
| epbase3q_legacy_10k_e15_beta1    | ep_threeq | 0.1345                  | 0.1475                   | 0.1151                   | 0.3848          | 0.5194         | 0.0004569                      | 0.00015                         | 479.4               |
| epbase3q_legacy_10k_e15_w5       | ep_threeq | 0.135                   | 0.1535                   | 0.128                    | 0.3329          | 0.5265         | 0.001305                       | 0.002878                        | 870.5               |
| epbase3q_legacy_10k_e15_base     | ep_threeq | 0.1495                  | 0.1675                   | 0.1313                   | 0.364           | 0.5675         | 0.0007545                      | 0.001129                        | 776.4               |
| epbase3q_legacy_10k_e15_alpha_hi | ep_threeq | 0.15                    | 0.1595                   | 0.168                    | 0.3693          | 0.6103         | 0.002102                       | 0.002597                        | 488.3               |
