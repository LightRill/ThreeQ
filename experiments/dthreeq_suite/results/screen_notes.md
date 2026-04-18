# DThreeQ Two Moons Screen Notes

Experiment: `dthreeq-twomoons-screen-v1`

Canonical radas id: `_workspace_ThreeQ_dthreeq_twomoons_screen_v1`

Status: 108 / 108 trials completed, 0 failed trials.

## Main Result

The first-stage screen does not support the current Dplus implementation as a
competitive optimization rule. Under the residual-difference loss implemented
from the PDF notes, Dplus variants consistently lag behind EP and BP on two
moons.

Best aggregate rows by variant:

| variant | mean best test error | min best test error | max best test error |
|:--|--:|--:|--:|
| `ep_gradual_100_0p01_plus` | 0.1565 | 0.1178 | 0.2885 |
| `ep_nudge_0p01_plus` | 0.1931 | 0.1154 | 0.5000 |
| `bp_tanh` | 0.3365 | 0.1178 | 0.7812 |
| `dplus_direct` | 0.3502 | 0.2163 | 0.5000 |
| `dplus_nudge_0p1_plus` | 0.3506 | 0.2188 | 0.5000 |
| `dplus_gradual_100_0p01_plus` | 0.3554 | 0.2404 | 0.5000 |
| `dplus_nudge_0p01_plus` | 0.3578 | 0.2548 | 0.5000 |
| `dplus_nudge_0p001_plus` | 0.3590 | 0.2668 | 0.5000 |
| `dplus_nudge_0p01_plusminus` | 0.3590 | 0.2668 | 0.5000 |

## Interpretation

- EP is the strongest local-rule baseline in this screen. `ep_gradual_100_0p01_plus`
  has the best mean best-test-error, though its `lr=1e-3` final error is unstable.
- BP reaches competitive best error at `lr=1e-3`, but is much more learning-rate
  sensitive than EP.
- Dplus generally stays around `0.35` mean best error and often gets stuck at
  `0.5` for one seed. This suggests the current residual-difference objective
  is not transmitting a useful supervised update reliably.
- The failure is not caused by state saturation in this screen: Dplus saturation
  is essentially 0. The more likely issue is update alignment, sign, or the fact
  that the current Dplus objective only differentiates through the clamped
  residual while using free residuals as a detached target.

## Decision

Do not launch the planned MNIST/FashionMNIST grid from this exact Dplus rule.
The next useful experiment is a mechanism/debug run that compares Dplus weight
updates against BP and EP update directions on small batches, plus alternate
residual objectives where the free and clamped residual terms have controlled
gradient paths.

