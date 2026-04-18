from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threeq_common.mechanism import METHODS, PAIRS, SCOPES


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


def _pair_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for left, right in PAIRS:
            for scope in SCOPES:
                prefix = f"{left}_vs_{right}_{scope}"
                rows.append(
                    {
                        "target_name": row["target_name"],
                        "target_mode": row["target_mode"],
                        "beta_sign": row["beta_sign"],
                        "seed": row["seed"],
                        "step_lr": row["step_lr"],
                        "pair": f"{left}_vs_{right}",
                        "scope": scope,
                        "cosine": row.get(f"{prefix}_cosine"),
                        "norm_ratio": row.get(f"{prefix}_norm_ratio"),
                        "sign_agreement": row.get(f"{prefix}_sign_agreement"),
                    }
                )
    return pd.DataFrame(rows)


def _loss_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for method in METHODS:
            rows.append(
                {
                    "target_name": row["target_name"],
                    "target_mode": row["target_mode"],
                    "beta_sign": row["beta_sign"],
                    "seed": row["seed"],
                    "step_lr": row["step_lr"],
                    "method": method,
                    "direction_norm_full": row.get(f"{method}_full_direction_norm"),
                    "direction_norm_forward": row.get(f"{method}_forward_direction_norm"),
                    "free_mse_decrease_raw": row.get(f"{method}_free_mse_decrease_raw"),
                    "free_mse_decrease_bp_normed": row.get(
                        f"{method}_free_mse_decrease_bp_normed"
                    ),
                    "free_mse_decrease_raw_positive_rate": row.get(
                        f"{method}_free_mse_decrease_raw_positive_rate"
                    ),
                    "feedforward_ce_decrease_raw": row.get(
                        f"{method}_feedforward_ce_decrease_raw"
                    ),
                    "feedforward_ce_decrease_bp_normed": row.get(
                        f"{method}_feedforward_ce_decrease_bp_normed"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _mean_summary(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    available = [col for col in metrics if col in df.columns]
    parts = []
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
    value_col: str,
    title: str,
    output: Path,
    cmap: str = "coolwarm",
) -> None:
    pivot = table.pivot_table(
        index="target_name", columns="pair", values=value_col, aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8.5, max(4.8, 0.55 * len(pivot.index))))
    values = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if "cosine" in value_col else np.nanmax(values)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax if "cosine" in value_col else 0.0
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("pair")
    ax.set_ylabel("target")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _loss_plot(table: pd.DataFrame, value_col: str, title: str, output: Path) -> None:
    targets = list(table["target_name"].drop_duplicates())
    methods = METHODS
    x = np.arange(len(targets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for i, method in enumerate(methods):
        sub = table[table["method"] == method].set_index("target_name")
        values = sub.reindex(targets)[value_col].to_numpy()
        ax.bar(x + (i - 1) * width, values, width=width, label=method)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("loss decrease; positive is better")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "results_raw.csv", index=False)

    pair_long = _pair_long(df)
    loss_long = _loss_long(df)
    _save_table(pair_long, output_dir / "pair_metrics_long")
    _save_table(loss_long, output_dir / "loss_metrics_long")

    pair_summary = _mean_summary(
        pair_long,
        ["target_name", "pair", "scope"],
        ["cosine", "norm_ratio", "sign_agreement"],
    ).sort_values(["scope", "pair", "target_name"])
    _save_table(pair_summary, output_dir / "pair_summary")

    loss_summary = _mean_summary(
        loss_long,
        ["target_name", "method", "step_lr"],
        [
            "direction_norm_full",
            "direction_norm_forward",
            "free_mse_decrease_raw",
            "free_mse_decrease_bp_normed",
            "free_mse_decrease_raw_positive_rate",
            "feedforward_ce_decrease_raw",
            "feedforward_ce_decrease_bp_normed",
        ],
    ).sort_values(["step_lr", "target_name", "method"])
    _save_table(loss_summary, output_dir / "loss_summary")

    forward_pairs = pair_long[pair_long["scope"] == "forward"]
    forward_pair_means = (
        forward_pairs.groupby(["target_name", "pair"], as_index=False)[
            ["cosine", "norm_ratio", "sign_agreement"]
        ]
        .mean()
        .sort_values(["pair", "target_name"])
    )
    _save_table(forward_pair_means, output_dir / "forward_pair_compact")

    loss_compact = (
        loss_long.groupby(["target_name", "method"], as_index=False)[
            [
                "free_mse_decrease_raw",
                "free_mse_decrease_bp_normed",
                "free_mse_decrease_raw_positive_rate",
                "feedforward_ce_decrease_raw",
            ]
        ]
        .mean()
        .sort_values(["target_name", "method"])
    )
    _save_table(loss_compact, output_dir / "loss_compact")

    _heatmap(
        forward_pair_means,
        "cosine",
        "Forward-only update cosine similarity",
        figures_dir / "forward_cosine_heatmap.png",
        cmap="coolwarm",
    )
    _heatmap(
        forward_pair_means,
        "sign_agreement",
        "Forward-only update sign agreement",
        figures_dir / "forward_sign_agreement_heatmap.png",
        cmap="viridis",
    )
    _heatmap(
        forward_pair_means,
        "norm_ratio",
        "Forward-only update norm ratio",
        figures_dir / "forward_norm_ratio_heatmap.png",
        cmap="magma",
    )
    raw_step = loss_summary[loss_summary["step_lr"] == loss_summary["step_lr"].max()]
    raw_step_compact = raw_step.rename(
        columns={"free_mse_decrease_raw_mean": "free_mse_decrease_raw"}
    )
    _loss_plot(
        raw_step_compact,
        "free_mse_decrease_raw",
        "Raw one-step free-MSE decrease at largest step_lr",
        figures_dir / "raw_free_mse_decrease.png",
    )
    normed_loss = loss_compact.rename(
        columns={"free_mse_decrease_bp_normed": "free_mse_decrease_bp_normed"}
    )
    _loss_plot(
        normed_loss,
        "free_mse_decrease_bp_normed",
        "BP-normed one-step free-MSE decrease",
        figures_dir / "bp_normed_free_mse_decrease.png",
    )

    with (output_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# DThreeQ Mechanism Diagnostic\n\n")
        f.write(
            "All methods are evaluated on the same mini-batches and the same initial "
            "DThreeQ parameter state. BP updates affect only forward weights/biases; "
            "backward parameters are zero-filled in the full vector, so forward-only "
            "metrics are the cleanest direction comparison.\n\n"
        )
        f.write("## Main Findings\n\n")
        f.write(
            "- Dplus is not aligned with BP. Forward-only `dplus_vs_bp` cosine stays "
            "near zero for all targets, and sign agreement is close to chance.\n"
        )
        f.write(
            "- Dplus is often aligned with EP for one-sided plus targets, but its "
            "norm is much smaller than EP. This points to a scale-collapse problem, "
            "not only a direction problem.\n"
        )
        f.write(
            "- The `plusminus` Dplus variant is the clearest failure mode: it has "
            "negative Dplus-vs-EP cosine and non-positive raw one-step free-MSE "
            "change, consistent with cancellation between the two signs.\n"
        )
        f.write(
            "- EP raw steps reduce the free-MSE objective most strongly, but mainly "
            "because the EP vector norm is tens to hundreds of times larger than BP. "
            "BP-normed comparisons are therefore the right scale-controlled view.\n\n"
        )
        f.write("## Forward Pair Metrics\n\n")
        f.write(_format_markdown_table(forward_pair_means))
        f.write("\n\n## One-Step Loss Metrics\n\n")
        f.write(_format_markdown_table(loss_compact))
        f.write("\n")
