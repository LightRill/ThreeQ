from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threeq_common.legacy_mnist import write_curve_rows


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


def _plot_curves(curves: pd.DataFrame, output: Path) -> None:
    if curves.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    group = curves.groupby(["variant", "epoch"], as_index=False)["valid_error"].mean()
    for variant, sub in group.groupby("variant"):
        sub = sub.sort_values("epoch")
        ax.plot(sub["epoch"], sub["valid_error"], marker="o", label=variant)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation error")
    ax.set_title("Legacy MNIST reproduction curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_error(summary: pd.DataFrame, output: Path) -> None:
    table = summary.sort_values("best_valid_error_mean")
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(x - 0.18, table["best_valid_error_mean"], width=0.36, label="best")
    ax.bar(x + 0.18, table["final_valid_error_mean"], width=0.36, label="final")
    ax.set_xticks(x)
    ax.set_xticklabels(table["variant"], rotation=25, ha="right")
    ax.set_ylabel("validation error")
    ax.set_title("Legacy MNIST reproduction summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "results_raw.csv"
    df.to_csv(raw_path, index=False)
    curves_path = output_dir / "curves.csv"
    write_curve_rows(raw_path, curves_path)
    curves = pd.read_csv(curves_path) if curves_path.exists() else pd.DataFrame()
    metrics = [
        "best_valid_error",
        "final_valid_error",
        "final_train_error",
        "final_train_cost",
        "final_valid_cost",
        "rho_mean",
        "rho_max",
        "forward_update_rel_mean",
        "backward_update_rel_mean",
        "duration_sec",
    ]
    summary = _mean_summary(df, ["variant", "method"], metrics).sort_values(
        "best_valid_error_mean"
    )
    _save_table(summary, output_dir / "summary_by_variant")
    compact_cols = [
        "variant",
        "method",
        "best_valid_error_mean",
        "final_valid_error_mean",
        "final_train_error_mean",
        "rho_mean_mean",
        "rho_max_mean",
        "forward_update_rel_mean_mean",
        "backward_update_rel_mean_mean",
        "duration_sec_mean",
    ]
    compact = summary[[col for col in compact_cols if col in summary.columns]].copy()
    _save_table(compact, output_dir / "summary_compact")
    _plot_curves(curves, figures_dir / "legacy_mnist_curves.png")
    _plot_error(summary, figures_dir / "legacy_mnist_error_by_variant.png")
    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Legacy MNIST Reproduction\n\n")
        f.write(
            "Direct reproduction-style screen using original `Base3Q/ThreeQ.py` "
            "and `EPBase3Q/ThreeQ.py` network classes with net1 architecture and "
            "hyperparameters, on a 5k/1k MNIST subset for 8 epochs.\n\n"
        )
        f.write("## Summary\n\n")
        f.write(_format_markdown_table(compact))
        f.write("\n")
