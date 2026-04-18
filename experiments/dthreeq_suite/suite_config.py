from __future__ import annotations

from ray import tune


EXPERIMENT_NAME = "dthreeq-twomoons-screen-v1"

VARIANTS = [
    "bp_tanh",
    "dplus_direct",
    "dplus_nudge_0p1_plus",
    "dplus_nudge_0p01_plus",
    "dplus_nudge_0p001_plus",
    "dplus_nudge_0p01_plusminus",
    "dplus_gradual_100_0p01_plus",
    "ep_nudge_0p01_plus",
    "ep_gradual_100_0p01_plus",
]

SEEDS = [42, 2025, 7]
WEIGHT_LRS = [1e-3, 1e-4, 5e-5, 1e-5]


def param_space():
    return {
        "variant": tune.grid_search(VARIANTS),
        "weight_lr": tune.grid_search(WEIGHT_LRS),
        "seed": tune.grid_search(SEEDS),
        "dataset": "twomoons",
        "n_samples": 2000,
        "test_size": 0.2,
        "noise": 0.1,
        "hidden_sizes": [128, 64],
        "batch_size": 32,
        "n_epochs": 30,
        "infer_steps": 15,
        "infer_lr": 0.05,
        "activation": "tanh",
        "bias": True,
        "state_clip": 1.0,
    }

