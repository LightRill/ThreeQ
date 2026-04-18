# Legacy MNIST Reproduction

Direct reproduction-style screen using original `Base3Q/ThreeQ.py` and `EPBase3Q/ThreeQ.py` network classes with net1 architecture and hyperparameters, on a 5k/1k MNIST subset for 8 epochs.

## Summary

| variant                    | method        | best_valid_error_mean   | final_valid_error_mean   | final_train_error_mean   | rho_mean_mean   | rho_max_mean   | forward_update_rel_mean_mean   | backward_update_rel_mean_mean   | duration_sec_mean   |
|:---------------------------|:--------------|:------------------------|:-------------------------|:-------------------------|:----------------|:---------------|:-------------------------------|:--------------------------------|:--------------------|
| epbase3q_legacy_net1_5k_e8 | ep_threeq     | 0.225                   | 0.228                    | 0.1786                   | 0.3615          | 0.4312         | 0.0004293                      | 0.0001223                       | 230                 |
| base3q_legacy_net1_5k_e8   | threeq_direct | 0.398                   | 0.398                    | 0.3554                   | 0.3789          | 0.4179         | 0.0005151                      | 0.001864                        | 426                 |
