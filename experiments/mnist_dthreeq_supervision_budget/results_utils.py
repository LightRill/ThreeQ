from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.mnist_dthreeq_focus.results_utils import (
    _fill_curve_final_metrics,
    _format_markdown_table,
    _plot_curves,
    _plot_diagnostics,
    _plot_error,
    _save_table,
    _mean_summary,
)
from threeq_common.mnist_dthreeq_focus import write_focus_curve_rows


def write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    df = _fill_curve_final_metrics(df)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "results_raw.csv"
    df.to_csv(raw_path, index=False)
    curves_path = output_dir / "curves.csv"
    write_focus_curve_rows(raw_path, curves_path)
    curves = pd.read_csv(curves_path) if curves_path.exists() else pd.DataFrame()

    metrics = [
        "best_test_error",
        "selected_test_error",
        "selected_test_accuracy",
        "final_test_error",
        "final_state_test_error",
        "final_test_cost",
        "final_train_error",
        "final_train_cost",
        "final_train_energy",
        "state_delta",
        "saturation",
        "input_recon_energy_frac",
        "weighted_input_recon_energy_frac",
        "weight_abs_mean",
        "weight_update_rel_mean",
        "duration_sec",
    ]
    summary = _mean_summary(df, ["variant", "family"], metrics).sort_values(
        "selected_test_error_mean"
    )
    _save_table(summary, output_dir / "summary_by_variant")
    compact_cols = [
        "variant",
        "family",
        "best_test_error_mean",
        "best_test_error_std",
        "selected_test_error_mean",
        "selected_test_accuracy_mean",
        "final_test_error_mean",
        "final_state_test_error_mean",
        "final_train_error_mean",
        "state_delta_mean",
        "saturation_mean",
        "input_recon_energy_frac_mean",
        "weighted_input_recon_energy_frac_mean",
        "weight_update_rel_mean_mean",
        "duration_sec_mean",
    ]
    compact = summary[[col for col in compact_cols if col in summary.columns]].copy()
    _save_table(compact, output_dir / "summary_compact")
    _plot_error(summary, figures_dir / "mnist_dthreeq_supervision_error.png")
    _plot_curves(curves, figures_dir / "mnist_dthreeq_supervision_curves.png")
    _plot_diagnostics(summary, figures_dir / "mnist_dthreeq_supervision_diagnostics.png")

    config_cols = [
        "train_subset",
        "test_subset",
        "n_epochs",
        "batch_size",
        "hidden_sizes",
        "device",
    ]
    config = (
        df[config_cols].drop_duplicates().head(12)
        if set(config_cols).issubset(df.columns)
        else pd.DataFrame()
    )
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# DThreeQ MNIST Supervision And Budget Screen\n\n")
        f.write(
            "This screen isolates four hypotheses: input reconstruction dominance, "
            "MSE one-hot versus CE-style output supervision, extra readout or label "
            "encoding differences, and high-LR stabilization by schedule or best "
            "checkpoint selection.\n\n"
        )
        if not config.empty:
            f.write("## Run Configuration\n\n")
            f.write(_format_markdown_table(config))
            f.write("\n\n")
        f.write("## Summary By Variant\n\n")
        f.write(_format_markdown_table(compact))
        f.write("\n\n")
        f.write("## Notes\n\n")
        f.write(
            "`selected_test_error` equals `best_test_error` only for restore-best "
            "variants; otherwise it equals final test error. The dashed 0.10 line "
            "in the figures marks 90% accuracy.\n"
        )
