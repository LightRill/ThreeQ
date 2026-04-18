from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results_raw.csv", index=False)
    group_cols = ["variant", "weight_lr"]
    keep = [
        "best_test_error",
        "final_test_error",
        "final_train_error",
        "state_delta",
        "saturation",
        "weight_abs_mean",
        "weight_update_rel_mean",
        "duration_sec",
    ]
    available = [col for col in keep if col in df.columns]
    summary = (
        df.groupby(group_cols)[available]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("best_test_error", "mean"))
    )
    summary.to_csv(output_dir / "summary_by_variant_lr.csv")
    with (output_dir / "summary_by_variant_lr.md").open("w", encoding="utf-8") as f:
        f.write(summary.to_markdown())
        f.write("\n")

    compact = pd.DataFrame(
        {
            "best_test_error_mean": summary[("best_test_error", "mean")],
            "best_test_error_std": summary[("best_test_error", "std")],
            "final_test_error_mean": summary[("final_test_error", "mean")],
            "final_train_error_mean": summary[("final_train_error", "mean")],
            "state_delta_mean": summary.get(("state_delta", "mean")),
            "saturation_mean": summary.get(("saturation", "mean")),
            "weight_abs_mean": summary.get(("weight_abs_mean", "mean")),
            "duration_sec_mean": summary[("duration_sec", "mean")],
        }
    )
    compact.to_csv(output_dir / "summary_compact.csv")
    with (output_dir / "summary_compact.md").open("w", encoding="utf-8") as f:
        f.write(compact.round(4).to_markdown())
        f.write("\n")

    by_variant = (
        df.groupby("variant")[available]
        .agg(["mean", "std", "min", "max"])
        .sort_values(("best_test_error", "mean"))
    )
    by_variant.to_csv(output_dir / "summary_by_variant.csv")
    with (output_dir / "summary_by_variant.md").open("w", encoding="utf-8") as f:
        f.write(by_variant.to_markdown())
        f.write("\n")

