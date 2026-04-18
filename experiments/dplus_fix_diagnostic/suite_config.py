from __future__ import annotations

from ray import tune

from threeq_common.dplus_fix import OBJECTIVE_SPECS, PLUS_TARGET_SPECS


EXPERIMENT_NAME = "dplus-fix-diagnostic-v2"

TARGET_NAMES = list(PLUS_TARGET_SPECS)
OBJECTIVE_NAMES = list(OBJECTIVE_SPECS)
SEEDS = [42, 2025, 7]


def param_space():
    return {
        "target_name": tune.grid_search(TARGET_NAMES),
        "objective_name": tune.grid_search(OBJECTIVE_NAMES),
        "seed": tune.grid_search(SEEDS),
        "step_lr": 1e-3,
        "dataset": "twomoons",
        "n_samples": 2000,
        "test_size": 0.2,
        "noise": 0.1,
        "hidden_sizes": [128, 64],
        "batch_size": 32,
        "n_batches": 6,
        "infer_steps": 15,
        "infer_lr": 0.05,
        "residual_norm_eps": 1e-3,
        "activation": "tanh",
        "bias": True,
        "state_clip": 1.0,
    }
