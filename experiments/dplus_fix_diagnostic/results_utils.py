from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _format_markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if col == "step_lr":
            out[col] = out[col].map(lambda value: "n/a" if pd.isna(value) else f"{value:.0e}")
        else:
            out[col] = out[col].map(
                lambda value: "n/a" if pd.isna(value) else f"{value:.{digits}g}"
            )
    out = out.astype(object).where(pd.notna(out), "n/a")
    return out.to_markdown(index=False, disable_numparse=True)


def _save_table(df: pd.DataFrame, path_stem: Path, digits: int = 4) -> None:
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    with path_stem.with_suffix(".md").open("w", encoding="utf-8") as f:
        f.write(_format_markdown_table(df, digits=digits))
        f.write("\n")


def _mean_summary(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    parts = []
    available = [metric for metric in metrics if metric in df.columns]
    for metric in available:
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
    if not parts:
        return pd.DataFrame(columns=group_cols)
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=group_cols, how="outer")
    return out


def _heatmap(
    table: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    title: str,
    output: Path,
    cmap: str = "viridis",
    center_zero: bool = False,
) -> None:
    pivot = table.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(9.5, 1.05 * len(pivot.columns)), max(4.8, 0.55 * len(pivot.index))))
    data = pivot.to_numpy(dtype=float)
    if center_zero:
        vmax = np.nanmax(np.abs(data))
        vmax = 1.0 if not np.isfinite(vmax) or vmax == 0 else vmax
        vmin = -vmax
    else:
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(columns)
    ax.set_ylabel(index)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=values)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _bar_top(table: pd.DataFrame, output: Path) -> None:
    top = table.head(18).copy()
    labels = top["objective_name"] + " / " + top["target_name"]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(13.0, 6.0))
    ax.bar(x, top["direction_score"], color="#4f7cac")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Top Dplus fix candidates by direction score")
    ax.set_ylabel("score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results_raw.csv", index=False)

    metrics = [
        "dplus_vs_bp_forward_cosine",
        "dplus_vs_bp_forward_norm_ratio",
        "dplus_vs_bp_forward_sign_agreement",
        "dplus_vs_ep_forward_cosine",
        "dplus_vs_ep_forward_norm_ratio",
        "dplus_vs_ep_forward_sign_agreement",
        "dplus_forward_norm",
        "bp_forward_norm",
        "ep_forward_norm",
        "dplus_raw_free_mse_decrease",
        "dplus_bp_scaled_free_mse_decrease",
        "dplus_ep_scaled_free_mse_decrease",
        "dplus_raw_feedforward_ce_decrease",
        "dplus_bp_scaled_feedforward_ce_decrease",
        "dplus_ep_scaled_feedforward_ce_decrease",
        "dplus_raw_free_mse_decrease_positive_rate",
        "dplus_bp_scaled_free_mse_decrease_positive_rate",
        "dplus_ep_scaled_free_mse_decrease_positive_rate",
        "dplus_objective",
        "dplus_state_delta",
        "duration_sec",
    ]
    summary = _mean_summary(df, ["objective_name", "target_name"], metrics)
    summary["direction_score"] = (
        summary["dplus_vs_bp_forward_cosine_mean"].fillna(-1.0)
        + 0.25 * summary["dplus_bp_scaled_free_mse_decrease_positive_rate_mean"].fillna(0.0)
        + 0.25 * summary["dplus_ep_scaled_free_mse_decrease_positive_rate_mean"].fillna(0.0)
    )
    summary["regularizer_pass"] = (
        (summary["dplus_bp_scaled_free_mse_decrease_positive_rate_mean"] >= 0.95)
        & (summary["dplus_ep_scaled_free_mse_decrease_positive_rate_mean"] >= 0.95)
        & (summary["dplus_vs_bp_forward_cosine_mean"] > 0.0)
    )
    summary = summary.sort_values(
        ["regularizer_pass", "direction_score", "dplus_vs_bp_forward_cosine_mean"],
        ascending=[False, False, False],
    )
    _save_table(summary, output_dir / "candidate_summary")

    compact = summary[
        [
            "objective_name",
            "target_name",
            "regularizer_pass",
            "direction_score",
            "dplus_vs_bp_forward_cosine_mean",
            "dplus_vs_bp_forward_sign_agreement_mean",
            "dplus_vs_bp_forward_norm_ratio_mean",
            "dplus_vs_ep_forward_cosine_mean",
            "dplus_vs_ep_forward_sign_agreement_mean",
            "dplus_vs_ep_forward_norm_ratio_mean",
            "dplus_raw_free_mse_decrease_mean",
            "dplus_bp_scaled_free_mse_decrease_mean",
            "dplus_ep_scaled_free_mse_decrease_mean",
            "dplus_raw_feedforward_ce_decrease_mean",
            "dplus_bp_scaled_feedforward_ce_decrease_mean",
            "dplus_ep_scaled_feedforward_ce_decrease_mean",
            "dplus_bp_scaled_free_mse_decrease_positive_rate_mean",
            "dplus_ep_scaled_free_mse_decrease_positive_rate_mean",
        ]
    ].copy()
    _save_table(compact, output_dir / "candidate_compact")

    best_by_objective = (
        compact.sort_values("direction_score", ascending=False)
        .groupby("objective_name", as_index=False)
        .head(1)
        .sort_values("direction_score", ascending=False)
    )
    _save_table(best_by_objective, output_dir / "best_by_objective")

    _heatmap(
        compact,
        index="target_name",
        columns="objective_name",
        values="dplus_vs_bp_forward_cosine_mean",
        title="Dplus-vs-BP forward cosine",
        output=figures_dir / "dplus_vs_bp_cosine_heatmap.png",
        cmap="coolwarm",
        center_zero=True,
    )
    _heatmap(
        compact,
        index="target_name",
        columns="objective_name",
        values="dplus_vs_bp_forward_sign_agreement_mean",
        title="Dplus-vs-BP forward sign agreement",
        output=figures_dir / "dplus_vs_bp_sign_agreement_heatmap.png",
        cmap="viridis",
        center_zero=False,
    )
    _heatmap(
        compact,
        index="target_name",
        columns="objective_name",
        values="dplus_vs_ep_forward_cosine_mean",
        title="Dplus-vs-EP forward cosine",
        output=figures_dir / "dplus_vs_ep_cosine_heatmap.png",
        cmap="coolwarm",
        center_zero=True,
    )
    log_norm = compact.copy()
    log_norm["log10_dplus_vs_bp_norm_ratio"] = np.log10(
        log_norm["dplus_vs_bp_forward_norm_ratio_mean"].clip(lower=1e-12)
    )
    _heatmap(
        log_norm,
        index="target_name",
        columns="objective_name",
        values="log10_dplus_vs_bp_norm_ratio",
        title="log10 Dplus/BP forward norm ratio",
        output=figures_dir / "dplus_vs_bp_norm_ratio_log_heatmap.png",
        cmap="magma",
        center_zero=False,
    )
    _heatmap(
        compact,
        index="target_name",
        columns="objective_name",
        values="dplus_bp_scaled_free_mse_decrease_mean",
        title="BP-scaled Dplus one-step free-MSE decrease",
        output=figures_dir / "bp_scaled_free_mse_decrease_heatmap.png",
        cmap="viridis",
        center_zero=True,
    )
    _heatmap(
        compact,
        index="target_name",
        columns="objective_name",
        values="dplus_bp_scaled_feedforward_ce_decrease_mean",
        title="BP-scaled Dplus one-step feedforward CE decrease",
        output=figures_dir / "bp_scaled_feedforward_ce_decrease_heatmap.png",
        cmap="viridis",
        center_zero=True,
    )
    _heatmap(
        compact,
        index="target_name",
        columns="objective_name",
        values="dplus_ep_scaled_free_mse_decrease_mean",
        title="EP-scaled Dplus one-step free-MSE decrease",
        output=figures_dir / "ep_scaled_free_mse_decrease_heatmap.png",
        cmap="viridis",
        center_zero=True,
    )
    _bar_top(compact.sort_values("direction_score", ascending=False), figures_dir / "candidate_score_top.png")

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Dplus Fix Diagnostic\n\n")
        f.write(
            "This suite tests plus-only Dplus objective variants with beta scaling, "
            "residual RMS normalization, layer-wise gain, and BP/EP forward-norm "
            "calibrated one-step loss checks. Tables include cosine similarity, "
            "norm ratio, sign agreement, and one-step loss decrease. "
            "`regularizer_pass` requires positive BP-scaled and EP-scaled one-step "
            "free-MSE decrease rates and positive Dplus-vs-BP forward cosine.\n\n"
        )
        f.write("## Best Candidate Per Objective\n\n")
        f.write(_format_markdown_table(best_by_objective))
        f.write("\n\n## Top Candidate Ranking\n\n")
        f.write(_format_markdown_table(compact.head(20)))
        f.write("\n")
