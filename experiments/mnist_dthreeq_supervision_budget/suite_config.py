from __future__ import annotations

from ray import tune


EXPERIMENT_NAME = "threeq-mnist-dthreeq-supervision-budget-v2"

VARIANT_NAMES = [
    "dthreeq_ep_nudge0p1_lr3e3",
    "dthreeq_ep_tanh_nudge0p1_lr5e3",
    "dthreeq_ep_noinput_nudge0p1_lr3e3",
    "dthreeq_ep_input1over784_nudge0p1_lr3e3",
    "dthreeq_ep_input1over28_nudge0p1_lr3e3",
    "dthreeq_ep_ce_nudge0p05_lr3e3",
    "dthreeq_ep_ce_nudge0p1_lr3e3",
    "dthreeq_ep_signed_nudge0p1_lr3e3",
    "dthreeq_ep_readout_nudge0p1_lr3e3",
    "dthreeq_ep_readout_noinput_nudge0p1_lr3e3",
    "dthreeq_ep_tanh_nudge0p1_lr5e3_decay15",
    "dthreeq_ep_tanh_nudge0p1_lr5e3_restorebest",
    "dthreeq_ep_tanh_nudge0p1_lr5e3_decay15_restorebest",
]
SEEDS = [42, 2025]


def param_space():
    return {
        "variant": tune.grid_search(VARIANT_NAMES),
        "seed": tune.grid_search(SEEDS),
        "dataset": "mnist",
        "train_subset": 10_000,
        "test_subset": 2_000,
        "n_epochs": 30,
        "batch_size": 64,
        "infer_lr": 0.05,
        "activation": "tanh",
        "bias": True,
        "state_clip": 1.0,
        "residual_norm_eps": 1e-3,
    }
