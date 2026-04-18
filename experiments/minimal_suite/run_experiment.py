from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
import radas.clusters.ftjob as radas_ftjob
from radas import run_experiment

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.minimal_suite.suite_config import EXPERIMENT_NAME, param_space
from experiments.minimal_suite.trainable_def import trainable

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_PYTHON = "$HOME/miniconda3/envs/radas-cluster/bin/python"

RUNTIME_ENV = {
    "excludes": [
        "!ftjob_generated_*.py",
        "!ftjob_generated_*.pkl",
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
        "/experiments/minimal_suite/__pycache__/",
        "/experiments/minimal_suite/results/",
    ]
}


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results_raw.csv", index=False)
    variant_col = "variant" if "variant" in df.columns else "config/variant"
    keep = [
        "best_valid_error",
        "final_valid_error",
        "rho",
        "saturation",
        "state_delta_l1",
        "state_delta_l2",
        "weight_abs_mean",
        "duration_sec",
    ]
    available = [col for col in keep if col in df.columns]
    summary = (
        df.groupby(variant_col)[available]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("best_valid_error", "mean"))
    )
    summary.to_csv(output_dir / "summary_by_variant.csv")
    with (output_dir / "summary_by_variant.md").open("w", encoding="utf-8") as f:
        f.write(summary.to_markdown())
        f.write("\n")
    compact = pd.DataFrame(
        {
            "best_valid_error_mean": summary[("best_valid_error", "mean")],
            "best_valid_error_std": summary[("best_valid_error", "std")],
            "final_valid_error_mean": summary[("final_valid_error", "mean")],
            "rho_mean": summary[("rho", "mean")],
            "saturation_mean": summary[("saturation", "mean")],
            "state_delta_l1_mean": summary[("state_delta_l1", "mean")],
            "state_delta_l2_mean": summary[("state_delta_l2", "mean")],
            "weight_abs_mean": summary[("weight_abs_mean", "mean")],
            "duration_sec_mean": summary[("duration_sec", "mean")],
        }
    )
    compact.to_csv(output_dir / "summary_compact.csv")
    with (output_dir / "summary_compact.md").open("w", encoding="utf-8") as f:
        f.write(compact.round(4).to_markdown())
        f.write("\n")


async def main() -> None:
    os.chdir(ROOT)
    radas_ftjob.CLUSTER_PYTHON = CLUSTER_PYTHON
    results = await run_experiment(
        experiment_name=EXPERIMENT_NAME,
        trainable=trainable,
        param_space=param_space(),
        run_with="cluster:atol-gpu-5090",
        resources_per_trial={"cpu": 4, "gpu": 0.1},
        runtime_env=RUNTIME_ENV,
        warn_if_gpu_trainable_looks_cpu_only=False,
        run_choice="run",
    )
    df = results["df"]
    write_summary(df, SCRIPT_DIR / "results")
    print("display_experiment_name:", results.get("display_experiment_name"))
    print("canonical_experiment_name:", results.get("canonical_experiment_name"))
    print(df)


if __name__ == "__main__":
    asyncio.run(main())
