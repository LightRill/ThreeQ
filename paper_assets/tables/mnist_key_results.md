| section | dataset_budget | family | variant | best_error | final_error | best_accuracy | selected_accuracy | rho_mean | saturation | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnist_dthreeq_focus | MNIST 10k/2k, 12 epochs | DThreeQ EP | dthreeq_ep_nudge0p1_lr1e3 | 0.1965 | 0.1965 | 0.8035 | 0.8035 |  | 9.51e-05 | 扩大到 legacy 尺度后 DThreeQ EP-nudge 明显学习。 |
| mnist_dthreeq_fullbudget | MNIST 60k/10k, 30 epochs | DThreeQ | dthreeq_ep_signed_nudge0p1_lr3e3_restorebest | 0.1197 | 0.1621 | 0.8803 | 0.8803 |  | 0.3578 | Full-data restore-best 接近 88%，但 final 后期崩塌。 |
| mnist_dthreeq_fullbudget | MNIST 60k/10k, 30 epochs | DThreeQ | dthreeq_ep_ce_nudge0p1_lr3e3_restorebest | 0.1210 | 0.1852 | 0.8790 | 0.8790 |  | 0.2568 | Full-data restore-best 接近 88%，但 final 后期崩塌。 |
| mnist_dthreeq_fullbudget | MNIST 60k/10k, 30 epochs | DThreeQ | dthreeq_ep_ce_nudge0p1_lr3e3 | 0.1210 | 0.1852 | 0.8790 | 0.8148 |  | 0.2568 | Full-data restore-best 接近 88%，但 final 后期崩塌。 |
| mnist_dthreeq_fullbudget | MNIST 60k/10k, 30 epochs | DThreeQ | dthreeq_ep_nudge0p1_lr3e3 | 0.1477 | 0.8368 | 0.8523 | 0.1632 |  | 0.7104 | Full-data restore-best 接近 88%，但 final 后期崩塌。 |
| mnist_dthreeq_longrun | MNIST 10k/2k, 30 epochs | DThreeQ EP | dthreeq_ep_nudge0p1_lr3e3 | 0.1606 | 0.1646 | 0.8394 | 0.8394 |  | 5.85e-07 | 当前稳定 EP-nudge baseline，约 83.9% best accuracy。 |
| mnist_dthreeq_supervision | MNIST 10k/2k, 30 epochs | DThreeQ | dthreeq_ep_ce_nudge0p1_lr3e3 | 0.1316 | 0.1345 | 0.8684 | 0.8655 |  | 0.0108 | CE-style nudge 是当前最明确正向改动。 |
| mnist_dthreeq_supervision | MNIST 10k/2k, 30 epochs | DThreeQ | dthreeq_ep_readout_nudge0p1_lr3e3 | 0.1370 | 0.1418 | 0.8630 | 0.8582 |  | 6.83e-07 | CE-style nudge 是当前最明确正向改动。 |
| mnist_dthreeq_supervision | MNIST 10k/2k, 30 epochs | DThreeQ | dthreeq_ep_noinput_nudge0p1_lr3e3 | 0.1570 | 0.1814 | 0.8430 | 0.8186 |  | 5.29e-04 | CE-style nudge 是当前最明确正向改动。 |
| mnist_dthreeq_supervision | MNIST 10k/2k, 30 epochs | DThreeQ | dthreeq_ep_nudge0p1_lr3e3 | 0.1606 | 0.1646 | 0.8394 | 0.8354 |  | 6.83e-07 | CE-style nudge 是当前最明确正向改动。 |
| mnist_epthreeq_tune | MNIST 10k/2k, 15 epochs | EPThreeQ legacy | epbase3q_legacy_10k_e15_beta1 | 0.1345 | 0.1475 | 0.8655 | 0.8655 | 0.3848 |  | EPThreeQ 回到约 86.5% accuracy。 |
| mnist_epthreeq_tune | MNIST 10k/2k, 15 epochs | EPThreeQ legacy | epbase3q_legacy_10k_e15_w5 | 0.1350 | 0.1535 | 0.8650 | 0.8650 | 0.3329 |  | EPThreeQ 回到约 86.5% accuracy。 |
| mnist_epthreeq_tune | MNIST 10k/2k, 15 epochs | EPThreeQ legacy | epbase3q_legacy_10k_e15_base | 0.1495 | 0.1675 | 0.8505 | 0.8505 | 0.3640 |  | EPThreeQ 回到约 86.5% accuracy。 |
| mnist_legacy_repro | MNIST 5k/1k, 8 epochs | EPThreeQ legacy | epbase3q_legacy_net1_5k_e8 | 0.2250 | 0.2280 | 0.7750 | 0.7750 | 0.3615 |  | 直接调用原始类，验证原始 ThreeQ/EPThreeQ 可学习。 |
| mnist_legacy_repro | MNIST 5k/1k, 8 epochs | ThreeQ legacy | base3q_legacy_net1_5k_e8 | 0.3980 | 0.3980 | 0.6020 | 0.6020 | 0.3789 |  | 直接调用原始类，验证原始 ThreeQ/EPThreeQ 可学习。 |
| mnist_small_budget | MNIST 3k/1k, 3 epochs | DThreeQ EP | dthreeq_ep_nudge_0p01 | 0.8668 | 0.8668 | 0.1332 | 0.1332 |  | 0.0026 | 小预算公共框架压力测试，接近随机。 |
