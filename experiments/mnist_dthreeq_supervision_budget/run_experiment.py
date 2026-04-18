from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import radas.clusters.ftjob as radas_ftjob
from radas import run_experiment


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_dthreeq_supervision_budget.results_utils import write_summaries
from experiments.mnist_dthreeq_supervision_budget.suite_config import (
    EXPERIMENT_NAME,
    param_space,
)
from experiments.mnist_dthreeq_supervision_budget.trainable_def import trainable


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_PYTHON = "$HOME/miniconda3/envs/radas-cluster/bin/python"

RUNTIME_ENV = {
    "excludes": [
        "!ftjob_generated_*.py",
        "!ftjob_generated_*.pkl",
        "/data/",
        "/AllConnected3QNotTrained/data/",
        "/AllConnected3QNotTrained/g(x)=*/",
        "/AllConnected3QNotTrained/__pycache__/",
        "/AllConnected3QTrained/data/",
        "/AllConnected3QTrained/png/",
        "/AllConnected3QTrained/g=*/",
        "/AllConnected3QTrained/__pycache__/",
        "/Base3Q/data/",
        "/Base3Q/logs/",
        "/Base3Q/__pycache__/",
        "/Base3QClampWithLinear/data/",
        "/Base3QClampWithLinear/logs/",
        "/Base3QClampWithLinear/__pycache__/",
        "/CNN3Q/data/",
        "/CNN3Q/logs/",
        "/CNN3Q/__pycache__/",
        "/EPBase3Q/data/",
        "/EPBase3Q/logs/",
        "/EPBase3Q/__pycache__/",
        "/EPBase3QClampWithLinear/data/",
        "/EPBase3QClampWithLinear/logs/",
        "/EPBase3QClampWithLinear/__pycache__/",
        "/EPCNN3Q/data/",
        "/EPCNN3Q/logs/",
        "/EPCNN3Q/__pycache__/",
        "/threeq_common/__pycache__/",
        "/experiments/minimal_suite/results/",
        "/experiments/dthreeq_suite/results/",
        "/experiments/analysis_summary/results/",
        "/experiments/mechanism_diagnostic/results/",
        "/experiments/dplus_fix_diagnostic/results/",
        "/experiments/mnist_suite/results/",
        "/experiments/mnist_legacy_repro/results/",
        "/experiments/mnist_dthreeq_focus/results/",
        "/experiments/mnist_dthreeq_longrun/results/",
        "/experiments/mnist_dthreeq_objective_audit/results/",
        "/experiments/mnist_dthreeq_activation_boost/results/",
        "/experiments/mnist_dthreeq_clip01_boost/results/",
        "/experiments/mnist_dthreeq_prediction_audit/results/",
        "/experiments/mnist_dthreeq_supervision_budget/__pycache__/",
        "/experiments/mnist_dthreeq_supervision_budget/results/",
    ]
}


async def main() -> None:
    os.chdir(ROOT)
    radas_ftjob.CLUSTER_PYTHON = CLUSTER_PYTHON
    results = await run_experiment(
        experiment_name=EXPERIMENT_NAME,
        trainable=trainable,
        param_space=param_space(),
        run_with="cluster:atol-gpu-5090",
        resources_per_trial={"cpu": 10, "gpu": 0.25},
        runtime_env=RUNTIME_ENV,
        warn_if_gpu_trainable_looks_cpu_only=False,
        run_choice="run",
    )
    df = results["df"]
    write_summaries(df, SCRIPT_DIR / "results")
    print("display_experiment_name:", results.get("display_experiment_name"))
    print("canonical_experiment_name:", results.get("canonical_experiment_name"))
    print(df)


if __name__ == "__main__":
    asyncio.run(main())
