from __future__ import annotations

from ray import tune

from threeq_common.legacy_mnist import LEGACY_MNIST_VARIANTS


EXPERIMENT_NAME = "threeq-mnist-legacy-repro-v1"

VARIANT_NAMES = list(LEGACY_MNIST_VARIANTS)
SEEDS = [42]


def param_space():
    return {
        "variant": tune.grid_search(VARIANT_NAMES),
        "seed": tune.grid_search(SEEDS),
        "dataset": "mnist",
    }
