from __future__ import annotations

from ray import tune


EXPERIMENT_NAME = "threeq-mnist-dthreeq-activation-boost-v1"

VARIANT_NAMES = [
    "dthreeq_ep_nudge0p1_lr3e3",
    "dthreeq_ep_tanh_nudge0p1_lr5e3",
    "dthreeq_ep_sigmoid_nudge0p05_lr3e3",
    "dthreeq_ep_sigmoid_nudge0p1_lr3e3",
    "dthreeq_ep_sigmoid_nudge0p1_lr5e3",
    "dthreeq_ep_sigmoid_nudge0p2_lr3e3",
    "dthreeq_ep_relu_nudge0p1_lr1e3",
    "dthreeq_ep_relu_nudge0p1_lr3e3",
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
