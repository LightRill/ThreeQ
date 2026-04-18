from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pandas as pd
import radas

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.minimal_suite.suite_config import EXPERIMENT_NAME, param_space

SCRIPT_DIR = Path(__file__).resolve().parent


def write_summary(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
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
    df.to_csv(output_dir / "results_raw.csv", index=False)
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
    output_dir = SCRIPT_DIR / "results"
    results = await radas.restore_and_materialize(
        experiment_name=EXPERIMENT_NAME,
        param_space=param_space(),
        output_dir=output_dir,
    )
    write_summary(results["df"], output_dir)
    print("restore_source:", results.get("restore_source"))
    print(results["df"])


if __name__ == "__main__":
    asyncio.run(main())
