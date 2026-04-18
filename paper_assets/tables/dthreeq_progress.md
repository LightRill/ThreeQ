| stage | variant | dataset_budget | best_accuracy | final_accuracy | interpretation |
| --- | --- | --- | --- | --- | --- |
| small public screen | dthreeq_ep_nudge_0p01 | 3k/1k, 3 epochs | 0.1332 | 0.1332 | 公共框架小预算接近随机，不能代表原始设定。 |
| legacy-scale focus | dthreeq_ep_nudge0p1_lr1e3 | 10k/2k, 12 epochs | 0.8035 | 0.8035 | 放大 hidden size 与训练预算后离开随机。 |
| longrun EP-nudge | dthreeq_ep_nudge0p1_lr3e3 | 10k/2k, 30 epochs | 0.8394 | 0.8354 | 稳定 baseline 约 83.5% 到 83.9%。 |
| CE-style nudge | dthreeq_ep_ce_nudge0p1_lr3e3 | 10k/2k, 30 epochs | 0.8684 | 0.8655 | 输出监督从 MSE nudge 改为 CE-style 后到约 86.5%。 |
| full-data restore-best | dthreeq_ep_signed_nudge0p1_lr3e3_restorebest | 60k/10k, 30 epochs | 0.8803 | 0.8379 | best checkpoint 达 88.0%，但 final 仍回退。 |
