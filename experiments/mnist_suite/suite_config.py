from __future__ import annotations

from ray import tune

from threeq_common.mnist import MNIST_VARIANTS


EXPERIMENT_NAME = "threeq-mnist-comparable-screen-v1"

VARIANT_NAMES = list(MNIST_VARIANTS)
SEEDS = [42, 2025]


def param_space():
    return {
        "variant": tune.grid_search(VARIANT_NAMES),
        "seed": tune.grid_search(SEEDS),
        "dataset": "mnist",
        "train_subset": 3000,
        "test_subset": 1000,
        "n_epochs": 3,
        "batch_size": 64,
        "hidden_sizes": [128, 64],
        "free_steps": 15,
        "weak_steps": 10,
        "epsilon": 0.05,
        "beta": 0.5,
        "infer_steps": 10,
        "infer_lr": 0.05,
        "weight_lr": 1e-4,
        "activation": "tanh",
        "bias": True,
        "state_clip": 1.0,
        "residual_norm_eps": 1e-3,
        "conv_channels": [1, 8],
        "kernel_sizes": [5],
        "strides": [2],
        "paddings": [2],
        "conv_free_steps": 10,
        "conv_weak_steps": 10,
        "conv_epsilon": 0.05,
    }
