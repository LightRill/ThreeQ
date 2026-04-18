from __future__ import annotations

from ray import tune


EXPERIMENT_NAME = "threeq-minimal-comparable-v1"

VARIANTS = [
    "direct_cost_w10",
    "direct_cost_w50",
    "direct_linear_w10",
    "ep_cost_w2",
    "ep_cost_w10",
    "ep_cost_w50",
    "ep_tied_w10",
]

SEEDS = [0, 1, 2]


def param_space():
    return {
        "variant": tune.grid_search(VARIANTS),
        "seed": tune.grid_search(SEEDS),
        "n_samples": 1000,
        "noise": 0.1,
        "hidden_sizes": [32],
        "batch_size": 20,
        "n_epochs": 20,
        "free_steps": 50,
        "epsilon": 0.05,
        "beta": 0.5,
        "alphas": [0.05, 0.01],
    }
