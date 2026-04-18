# DThreeQ Mechanism Diagnostic

All methods are evaluated on the same mini-batches and the same initial DThreeQ parameter state. BP updates affect only forward weights/biases; backward parameters are zero-filled in the full vector, so forward-only metrics are the cleanest direction comparison.

## Main Findings

- Dplus is not aligned with BP. Forward-only `dplus_vs_bp` cosine stays near zero for all targets, and sign agreement is close to chance.
- Dplus is often aligned with EP for one-sided plus targets, but its norm is much smaller than EP. This points to a scale-collapse problem, not only a direction problem.
- The `plusminus` Dplus variant is the clearest failure mode: it has negative Dplus-vs-EP cosine and non-positive raw one-step free-MSE change, consistent with cancellation between the two signs.
- EP raw steps reduce the free-MSE objective most strongly, but mainly because the EP vector norm is tens to hundreds of times larger than BP. BP-normed comparisons are therefore the right scale-controlled view.

## Forward Pair Metrics

| target_name           | pair        | cosine   | norm_ratio   | sign_agreement   |
|:----------------------|:------------|:---------|:-------------|:-----------------|
| direct_plus           | dplus_vs_bp | 0.07598  | 0.1705       | 0.5685           |
| gradual_100_0p01_plus | dplus_vs_bp | 0.06418  | 0.08171      | 0.5303           |
| nudge_0p001_plus      | dplus_vs_bp | 0.03631  | 0.003567     | 0.5013           |
| nudge_0p01_plus       | dplus_vs_bp | 0.07114  | 0.02588      | 0.5064           |
| nudge_0p01_plusminus  | dplus_vs_bp | -0.1574  | 0.001345     | 0.4973           |
| nudge_0p1_plus        | dplus_vs_bp | 0.075    | 0.1393       | 0.5468           |
| direct_plus           | dplus_vs_ep | 0.9971   | 0.08308      | 0.9879           |
| gradual_100_0p01_plus | dplus_vs_ep | 0.8466   | 0.000699     | 0.9831           |
| nudge_0p001_plus      | dplus_vs_ep | 0.3804   | 2.553e-05    | 0.9902           |
| nudge_0p01_plus       | dplus_vs_ep | 0.9276   | 0.0007701    | 0.9898           |
| nudge_0p01_plusminus  | dplus_vs_ep | -0.829   | 4.62e-05     | 0.5167           |
| nudge_0p1_plus        | dplus_vs_ep | 0.9967   | 0.008305     | 0.9877           |
| direct_plus           | ep_vs_bp    | 0.08059  | 2.053        | 0.5695           |
| gradual_100_0p01_plus | ep_vs_bp    | 0.07008  | 116.6        | 0.5315           |
| nudge_0p001_plus      | ep_vs_bp    | 0.02145  | 134.4        | 0.502            |
| nudge_0p01_plus       | ep_vs_bp    | 0.07076  | 33.58        | 0.5062           |
| nudge_0p01_plusminus  | ep_vs_bp    | 0.07728  | 32.5         | 0.5756           |
| nudge_0p1_plus        | ep_vs_bp    | 0.07808  | 16.77        | 0.5468           |

## One-Step Loss Metrics

| target_name           | method   | free_mse_decrease_raw   | free_mse_decrease_bp_normed   | free_mse_decrease_raw_positive_rate   | feedforward_ce_decrease_raw   |
|:----------------------|:---------|:------------------------|:------------------------------|:--------------------------------------|:------------------------------|
| direct_plus           | bp       | 0.0001342               | 0.0001342                     | 1                                     | 0.0001348                     |
| direct_plus           | dplus    | 3.573e-05               | 0.0002343                     | 1                                     | 1.553e-06                     |
| direct_plus           | ep       | 0.0004658               | 0.0001647                     | 1                                     | 2.005e-05                     |
| gradual_100_0p01_plus | bp       | 0.0001342               | 0.0001342                     | 1                                     | 0.0001348                     |
| gradual_100_0p01_plus | dplus    | 1.666e-05               | 0.0002129                     | 1                                     | 6.01e-07                      |
| gradual_100_0p01_plus | ep       | 0.02431                 | 0.0001244                     | 1                                     | 0.0009815                     |
| nudge_0p001_plus      | bp       | 0.0001342               | 0.0001342                     | 1                                     | 0.0001348                     |
| nudge_0p001_plus      | dplus    | 6.73e-07                | 9.541e-05                     | 0.9722                                | 6.623e-09                     |
| nudge_0p001_plus      | ep       | 0.008209                | 3.013e-05                     | 1                                     | 0.0003575                     |
| nudge_0p01_plus       | bp       | 0.0001342               | 0.0001342                     | 1                                     | 0.0001348                     |
| nudge_0p01_plus       | dplus    | 5.367e-06               | 0.0002256                     | 1                                     | 2.103e-07                     |
| nudge_0p01_plus       | ep       | 0.006787                | 0.00013                       | 1                                     | 0.0002873                     |
| nudge_0p01_plusminus  | bp       | 0.0001342               | 0.0001342                     | 1                                     | 0.0001348                     |
| nudge_0p01_plusminus  | dplus    | -2.964e-07              | -4.708e-05                    | 0                                     | -3.311e-08                    |
| nudge_0p01_plusminus  | ep       | 0.007199                | 0.0001549                     | 1                                     | 0.0003032                     |
| nudge_0p1_plus        | bp       | 0.0001342               | 0.0001342                     | 1                                     | 0.0001348                     |
| nudge_0p1_plus        | dplus    | 2.915e-05               | 0.0002342                     | 1                                     | 1.243e-06                     |
| nudge_0p1_plus        | ep       | 0.003719                | 0.0001577                     | 1                                     | 0.0001584                     |
