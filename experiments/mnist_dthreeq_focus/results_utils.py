from __future__ import annotations

from pathlib import Path
from typing import List
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threeq_common.mnist_dthreeq_focus import write_focus_curve_rows


def _format_markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(
            lambda value: "n/a" if pd.isna(value) else f"{value:.{digits}g}"
        )
    out = out.astype(object).where(pd.notna(out), "n/a")
    return out.to_markdown(index=False, disable_numparse=True)


def _save_table(df: pd.DataFrame, stem: Path, digits: int = 4) -> None:
    df.to_csv(stem.with_suffix(".csv"), index=False)
    with stem.with_suffix(".md").open("w", encoding="utf-8") as f:
        f.write(_format_markdown_table(df, digits=digits))
        f.write("\n")


def _mean_summary(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    parts = []
    for metric in [metric for metric in metrics if metric in df.columns]:
        stat = (
            df.groupby(group_cols, dropna=False)[metric]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"{metric}_mean",
                    "std": f"{metric}_std",
                    "min": f"{metric}_min",
                    "max": f"{metric}_max",
                }
            )
        )
        parts.append(stat)
    out = parts[0] if parts else pd.DataFrame(columns=group_cols)
    for part in parts[1:]:
        out = out.merge(part, on=group_cols, how="outer")
    return out


def _fill_curve_final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "input_recon_energy",
        "input_recon_energy_frac",
        "weighted_input_recon_energy_frac",
    ]
    if "curve_json" not in df.columns:
        return df
    out = df.copy()
    for metric in metrics:
        if metric not in out.columns:
            out[metric] = np.nan
    for idx, value in out["curve_json"].items():
        try:
            curves = json.loads(value)
        except (TypeError, ValueError):
            continue
        if not curves:
            continue
        final = curves[-1]
        for metric in metrics:
            if pd.isna(out.at[idx, metric]) and metric in final:
                out.at[idx, metric] = final[metric]
    return out


def _plot_error(summary: pd.DataFrame, output: Path) -> None:
    table = summary.sort_values("best_test_error_mean")
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    ax.bar(x - 0.18, table["best_test_error_mean"], width=0.36, label="best")
    ax.bar(x + 0.18, table["final_test_error_mean"], width=0.36, label="final")
    ax.axhline(0.10, color="black", linestyle="--", linewidth=1.0, label="90% acc")
    ax.set_xticks(x)
    ax.set_xticklabels(table["variant"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("test error")
    ax.set_title("DThreeQ MNIST focused screen")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_curves(curves: pd.DataFrame, output: Path) -> None:
    if curves.empty:
        return
    group = curves.groupby(["variant", "epoch"], as_index=False)["test_error"].mean()
    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    for variant, sub in group.groupby("variant"):
        sub = sub.sort_values("epoch")
        ax.plot(sub["epoch"], sub["test_error"], marker="o", linewidth=1.5, label=variant)
    ax.axhline(0.10, color="black", linestyle="--", linewidth=1.0, label="90% acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test error")
    ax.set_title("DThreeQ MNIST focused curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_diagnostics(summary: pd.DataFrame, output: Path) -> None:
    cols = [
        "state_delta_mean",
        "saturation_mean",
        "input_recon_energy_frac_mean",
        "weighted_input_recon_energy_frac_mean",
        "weight_update_rel_mean_mean",
        "weight_abs_mean_mean",
        "duration_sec_mean",
    ]
    available = [col for col in cols if col in summary.columns]
    if not available:
        return
    table = summary.sort_values("best_test_error_mean")
    fig, axes = plt.subplots(len(available), 1, figsize=(12.8, 3.1 * len(available)))
    if len(available) == 1:
        axes = [axes]
    x = np.arange(len(table))
    for ax, col in zip(axes, available):
        ax.bar(x, table[col])
        ax.set_ylabel(col.replace("_mean", ""))
        ax.set_xticks(x)
        ax.set_xticklabels(table["variant"], rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("DThreeQ MNIST focused diagnostics", y=0.995)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


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
        "best_test_error_mean"
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
    _plot_error(summary, figures_dir / "mnist_dthreeq_focus_error.png")
    _plot_curves(curves, figures_dir / "mnist_dthreeq_focus_curves.png")
    _plot_diagnostics(summary, figures_dir / "mnist_dthreeq_focus_diagnostics.png")

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
        f.write("# DThreeQ MNIST Focused Screen\n\n")
        f.write(
            "Focused MNIST screen for DThreeQ variants using legacy-scale hidden "
            "width and a learning-rate sweep. This is the next step after the "
            "small-budget MNIST stress test.\n\n"
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
            "The dashed 0.10 test-error line in the figures marks the 90% accuracy "
            "threshold reported by external DThreeQ experiments.\n"
        )
