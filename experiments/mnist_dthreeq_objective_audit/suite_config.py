from __future__ import annotations

from ray import tune


EXPERIMENT_NAME = "threeq-mnist-dthreeq-objective-audit-v2"

VARIANT_NAMES = [
    "dthreeq_ep_nudge0p1_lr3e3",
    "dthreeq_plus_energy_direct_lr1e3",
    "dthreeq_plus_energy_direct_lr3e4",
    "dthreeq_plus_energy_nudge0p1_lr1e3",
    "dthreeq_forward_target_direct_lr1e3",
    "dthreeq_forward_target_nudge0p1_lr1e3",
    "dthreeq_bidir_target_direct_lr1e3",
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
