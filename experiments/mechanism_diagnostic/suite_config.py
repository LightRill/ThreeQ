from __future__ import annotations

from ray import tune

from threeq_common.mechanism import TARGET_SPECS


EXPERIMENT_NAME = "dthreeq-mechanism-diagnostic-v1"

TARGET_NAMES = list(TARGET_SPECS)
SEEDS = [42, 2025, 7]
STEP_LRS = [1e-3, 1e-4]


def param_space():
    return {
        "target_name": tune.grid_search(TARGET_NAMES),
        "seed": tune.grid_search(SEEDS),
        "step_lr": tune.grid_search(STEP_LRS),
        "dataset": "twomoons",
        "n_samples": 2000,
        "test_size": 0.2,
        "noise": 0.1,
        "hidden_sizes": [128, 64],
        "batch_size": 32,
        "n_batches": 6,
        "infer_steps": 15,
        "infer_lr": 0.05,
        "activation": "tanh",
        "bias": True,
        "state_clip": 1.0,
    }
