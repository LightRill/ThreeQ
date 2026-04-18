from __future__ import annotations

from ray import tune


EXPERIMENT_NAME = "threeq-mnist-epthreeq-tune-v1"

VARIANT_NAMES = [
    "epbase3q_legacy_10k_e15_base",
    "epbase3q_legacy_10k_e15_w5",
    "epbase3q_legacy_10k_e15_alpha_hi",
    "epbase3q_legacy_10k_e15_beta1",
]
SEEDS = [42]


def param_space():
    return {
        "variant": tune.grid_search(VARIANT_NAMES),
        "seed": tune.grid_search(SEEDS),
        "dataset": "mnist",
    }
