from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

MINIMAL_RAW = ROOT / "experiments" / "minimal_suite" / "results" / "results_raw.csv"
DTHREEQ_RAW = ROOT / "experiments" / "dthreeq_suite" / "results" / "results_raw.csv"


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, stem: str, digits: int = 4) -> None:
    csv_path = TABLES_DIR / f"{stem}.csv"
    md_path = TABLES_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(format_markdown_table(df, digits=digits))
        f.write("\n")


def format_markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    markdown_df = df.copy()
    for col in markdown_df.select_dtypes(include=[np.number]).columns:
        if col == "weight_lr":
            markdown_df[col] = markdown_df[col].map(
                lambda value: "n/a" if pd.isna(value) else f"{value:.0e}"
            )
        else:
            markdown_df[col] = markdown_df[col].round(digits)
    markdown_df = markdown_df.astype(object).where(pd.notna(markdown_df), "n/a")
    return markdown_df.to_markdown(index=False, disable_numparse=True)


def mean_summary(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    available = [col for col in metrics if col in df.columns]
    parts: list[pd.DataFrame] = []
    for metric in available:
        stat = (
            df.groupby(group_cols, dropna=False)[metric]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        stat = stat.rename(
            columns={
                "mean": f"{metric}_mean",
                "std": f"{metric}_std",
                "min": f"{metric}_min",
                "max": f"{metric}_max",
            }
        )
        parts.append(stat)
    if not parts:
        return pd.DataFrame(columns=group_cols)
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=group_cols, how="outer")
    return out


def ordered_variants(summary: pd.DataFrame, sort_col: str) -> list[str]:
    return summary.sort_values(sort_col)["variant"].tolist()


def plot_grouped_error_bars(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    order: list[str],
    best_col: str,
    final_col: str,
    title: str,
    output: Path,
) -> None:
    x = np.arange(len(order))
    width = 0.36
    by_variant = summary.set_index("variant")
    best_mean = by_variant.loc[order, f"{best_col}_mean"].to_numpy()
    final_mean = by_variant.loc[order, f"{final_col}_mean"].to_numpy()
    best_std = by_variant.loc[order, f"{best_col}_std"].fillna(0.0).to_numpy()
    final_std = by_variant.loc[order, f"{final_col}_std"].fillna(0.0).to_numpy()

    fig, ax = plt.subplots(figsize=(max(10.0, 0.95 * len(order)), 5.6))
    ax.bar(x - width / 2, best_mean, width, yerr=best_std, label="best", color="#4f7cac")
    ax.bar(x + width / 2, final_mean, width, yerr=final_std, label="final", color="#d9825b")

    for i, variant in enumerate(order):
        sub = df[df["variant"] == variant]
        best_jitter = np.linspace(-0.09, 0.09, max(len(sub), 1))
        final_jitter = np.linspace(-0.09, 0.09, max(len(sub), 1))
        ax.scatter(
            np.full(len(sub), x[i] - width / 2) + best_jitter,
            sub[best_col],
            s=18,
            color="#1f4e79",
            alpha=0.75,
            linewidths=0,
        )
        ax.scatter(
            np.full(len(sub), x[i] + width / 2) + final_jitter,
            sub[final_col],
            s=18,
            color="#9c4a2f",
            alpha=0.75,
            linewidths=0,
        )

    ax.set_title(title)
    ax.set_ylabel("error rate")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_metric_panels(
    df: pd.DataFrame,
    order: list[str],
    metrics: list[tuple[str, str]],
    title: str,
    output: Path,
) -> None:
    available = [(col, label) for col, label in metrics if col in df.columns]
    if not available:
        return

    n_cols = 2
    n_rows = int(np.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(11.0, 0.95 * len(order)), 3.0 * n_rows),
        squeeze=False,
    )
    fig.suptitle(title, y=1.01)
    x = np.arange(len(order))
    for ax, (metric, label) in zip(axes.ravel(), available):
        grouped = df.groupby("variant")[metric].agg(["mean", "std"])
        means = grouped.reindex(order)["mean"].to_numpy()
        stds = grouped.reindex(order)["std"].fillna(0.0).to_numpy()
        ax.bar(x, means, yerr=stds, color="#6aa477")
        for i, variant in enumerate(order):
            sub = df[df["variant"] == variant]
            jitter = np.linspace(-0.10, 0.10, max(len(sub), 1))
            ax.scatter(
                np.full(len(sub), x[i]) + jitter,
                sub[metric],
                s=14,
                color="#315f3a",
                alpha=0.72,
                linewidths=0,
            )
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.22)
    for ax in axes.ravel()[len(available) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_seed_matrix(
    df: pd.DataFrame,
    order: list[str],
    value_col: str,
    title: str,
    output: Path,
) -> None:
    pivot = df.pivot_table(index="variant", columns="seed", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(order)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.52 * len(pivot.columns)), max(4.8, 0.44 * len(order))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_title(title)
    ax.set_xlabel("seed")
    ax.set_ylabel("variant")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=0)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_lr_lines(
    grouped: pd.DataFrame,
    title: str,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for variant, sub in grouped.groupby("variant", sort=False):
        sub = sub.sort_values("weight_lr")
        ax.plot(
            sub["weight_lr"],
            sub["best_test_error_mean"],
            marker="o",
            linewidth=1.6,
            label=variant,
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_title(title)
    ax.set_xlabel("weight_lr")
    ax.set_ylabel("mean best test error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_lr_heatmap(grouped: pd.DataFrame, title: str, output: Path) -> None:
    pivot = grouped.pivot(index="variant", columns="weight_lr", values="best_test_error_mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot = pivot[sorted(pivot.columns, reverse=True)]
    fig, ax = plt.subplots(figsize=(8.8, max(5.2, 0.42 * len(pivot.index))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_title(title)
    ax.set_xlabel("weight_lr")
    ax.set_ylabel("variant")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.0e}" for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean best test error")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_combined_top_configs(combined: pd.DataFrame, output: Path) -> None:
    top = combined.head(16).copy()
    labels = top["suite"] + ": " + top["config"]
    x = np.arange(len(top))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13.0, 6.4))
    ax.bar(
        x - width / 2,
        top["best_error_mean"],
        width,
        label="best",
        color="#4f7cac",
    )
    ax.bar(
        x + width / 2,
        top["final_error_mean"],
        width,
        label="final",
        color="#d9825b",
    )
    ax.set_title("Top configurations across materialized suites")
    ax.set_ylabel("error rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def analyze_minimal() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(MINIMAL_RAW)
    metrics = [
        "best_valid_error",
        "final_valid_error",
        "final_train_error",
        "rho",
        "saturation",
        "state_delta_l1",
        "state_delta_l2",
        "weight_abs_mean",
        "weight_update_rel_mean",
        "duration_sec",
    ]
    summary = mean_summary(df, ["variant"], metrics).sort_values("best_valid_error_mean")
    summary["final_minus_best_mean"] = (
        summary["final_valid_error_mean"] - summary["best_valid_error_mean"]
    )
    save_table(summary, "minimal_variant_summary")
    order = ordered_variants(summary, "best_valid_error_mean")

    plot_grouped_error_bars(
        df,
        summary,
        order,
        "best_valid_error",
        "final_valid_error",
        "ThreeQ minimal suite: best vs final validation error",
        FIGURES_DIR / "minimal_error_by_variant.png",
    )
    plot_metric_panels(
        df,
        order,
        [
            ("rho", "local spectral radius"),
            ("saturation", "state saturation"),
            ("state_delta_l1", "state delta L1"),
            ("state_delta_l2", "state delta L2"),
            ("weight_abs_mean", "mean absolute weight"),
            ("duration_sec", "duration seconds"),
        ],
        "ThreeQ minimal suite diagnostics",
        FIGURES_DIR / "minimal_diagnostics_by_variant.png",
    )
    plot_seed_matrix(
        df,
        order,
        "best_valid_error",
        "ThreeQ minimal suite: seed-level best validation error",
        FIGURES_DIR / "minimal_seed_best_error.png",
    )

    combined_rows = df.copy()
    combined_rows["suite"] = "minimal"
    combined_rows["config"] = combined_rows["variant"]
    combined_rows["best_error"] = combined_rows["best_valid_error"]
    combined_rows["final_error"] = combined_rows["final_valid_error"]
    return df, combined_rows[["suite", "config", "variant", "seed", "best_error", "final_error"]]


def analyze_dthreeq() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DTHREEQ_RAW)
    df["final_minus_best"] = df["final_test_error"] - df["best_test_error"]
    metrics = [
        "best_test_error",
        "final_test_error",
        "final_train_error",
        "final_minus_best",
        "objective",
        "state_delta",
        "saturation",
        "weight_abs_mean",
        "weight_update_rel_mean",
        "duration_sec",
    ]
    summary = mean_summary(df, ["variant"], metrics).sort_values("best_test_error_mean")
    save_table(summary, "dthreeq_variant_summary")

    by_lr = mean_summary(df, ["variant", "weight_lr"], metrics).sort_values("best_test_error_mean")
    save_table(by_lr, "dthreeq_variant_lr_summary")
    order = ordered_variants(summary, "best_test_error_mean")

    plot_grouped_error_bars(
        df,
        summary,
        order,
        "best_test_error",
        "final_test_error",
        "DThreeQ screen: best vs final test error",
        FIGURES_DIR / "dthreeq_error_by_variant.png",
    )
    plot_metric_panels(
        df,
        order,
        [
            ("final_minus_best", "final minus best error"),
            ("objective", "training objective"),
            ("state_delta", "state delta"),
            ("saturation", "state saturation"),
            ("weight_abs_mean", "mean absolute weight"),
            ("duration_sec", "duration seconds"),
        ],
        "DThreeQ screen diagnostics",
        FIGURES_DIR / "dthreeq_diagnostics_by_variant.png",
    )
    plot_seed_matrix(
        df,
        order,
        "best_test_error",
        "DThreeQ screen: seed-level best test error",
        FIGURES_DIR / "dthreeq_seed_best_error.png",
    )
    plot_lr_lines(
        by_lr,
        "DThreeQ screen: learning-rate sensitivity",
        FIGURES_DIR / "dthreeq_best_error_vs_lr.png",
    )
    plot_lr_heatmap(
        by_lr,
        "DThreeQ screen: mean best test error by variant and LR",
        FIGURES_DIR / "dthreeq_best_error_heatmap.png",
    )

    combined_rows = df.copy()
    combined_rows["suite"] = "dthreeq"
    combined_rows["config"] = (
        combined_rows["variant"] + " lr=" + combined_rows["weight_lr"].map(lambda v: f"{v:.0e}")
    )
    combined_rows["best_error"] = combined_rows["best_test_error"]
    combined_rows["final_error"] = combined_rows["final_test_error"]
    return df, combined_rows[["suite", "config", "variant", "seed", "best_error", "final_error"]]


def analyze_combined(rows: list[pd.DataFrame]) -> pd.DataFrame:
    combined_trials = pd.concat(rows, ignore_index=True)
    combined = mean_summary(combined_trials, ["suite", "config"], ["best_error", "final_error"])
    combined = combined.sort_values(["best_error_mean", "final_error_mean"]).reset_index(drop=True)
    save_table(combined, "combined_config_summary")
    save_table(combined.head(20), "combined_top20")
    plot_combined_top_configs(combined, FIGURES_DIR / "combined_top_configs.png")
    return combined


def write_summary(
    minimal_summary: pd.DataFrame,
    dthreeq_summary: pd.DataFrame,
    dthreeq_by_lr: pd.DataFrame,
    combined: pd.DataFrame,
) -> None:
    minimal_compact = minimal_summary[
        [
            "variant",
            "best_valid_error_mean",
            "best_valid_error_std",
            "final_valid_error_mean",
            "rho_mean",
            "saturation_mean",
            "state_delta_l2_mean",
            "weight_abs_mean_mean",
            "duration_sec_mean",
        ]
    ].copy()
    dthreeq_compact = dthreeq_summary[
        [
            "variant",
            "best_test_error_mean",
            "best_test_error_std",
            "best_test_error_min",
            "final_test_error_mean",
            "state_delta_mean",
            "saturation_mean",
            "weight_abs_mean_mean",
            "duration_sec_mean",
        ]
    ].copy()
    dthreeq_lr_compact = dthreeq_by_lr[
        [
            "variant",
            "weight_lr",
            "best_test_error_mean",
            "best_test_error_std",
            "final_test_error_mean",
            "final_minus_best_mean",
        ]
    ].head(12)
    combined_compact = combined[
        ["suite", "config", "best_error_mean", "best_error_std", "final_error_mean"]
    ].head(12)

    summary_path = RESULTS_DIR / "summary.md"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(
            dedent(
                """
                # Consolidated ThreeQ Experiment Analysis

                This report materializes all currently saved training-run data from:

                - `experiments/minimal_suite/results/results_raw.csv`
                - `experiments/dthreeq_suite/results/results_raw.csv`

                The materialized CSV files do not contain per-epoch histories. The figures
                therefore visualize the full available per-trial data: best/final error,
                seed spread, learning-rate split, state convergence diagnostics, weight
                scale, update scale, and runtime.

                ## Figures

                - `figures/minimal_error_by_variant.png`
                - `figures/minimal_diagnostics_by_variant.png`
                - `figures/minimal_seed_best_error.png`
                - `figures/dthreeq_error_by_variant.png`
                - `figures/dthreeq_diagnostics_by_variant.png`
                - `figures/dthreeq_seed_best_error.png`
                - `figures/dthreeq_best_error_vs_lr.png`
                - `figures/dthreeq_best_error_heatmap.png`
                - `figures/combined_top_configs.png`

                ## Minimal Suite

                """
            ).lstrip()
        )
        f.write(format_markdown_table(minimal_compact))
        f.write(
            dedent(
                """

                Main conclusions:

                - `ep_cost_w50` gives the best mean best-validation error, but its final
                  validation error is worse than its best checkpoint and it has the largest
                  weight scale in this suite. It is an early-best winner, not a stable
                  endpoint.
                - `ep_cost_w10` is less aggressive and more stable than `ep_cost_w50`,
                  but does not reach the same best checkpoint.
                - `ep_cost_w2` is too weak: the weak phase does not supply enough training
                  signal.
                - Strict transpose sharing (`ep_tied_w10`) still hurts final accuracy in
                  this minimal implementation, so tying alone is not a sufficient fix.

                ## DThreeQ Screen

                """
            )
        )
        f.write(format_markdown_table(dthreeq_compact))
        f.write("\n\nBest DThreeQ/BP/EP settings by variant and LR:\n\n")
        f.write(format_markdown_table(dthreeq_lr_compact))
        f.write(
            dedent(
                """

                Main conclusions:

                - The current Dplus/DThreeQ update does not pass the two-moons screen.
                  Across learning rates, the Dplus variants stay around 0.35 mean best
                  test error and many runs remain near the 0.5 chance-error regime.
                - The strongest DThreeQ-screen rows are still EP and BP controls. The best
                  mean row is `ep_gradual_100_0p01_plus` at `weight_lr=1e-3`, but its final
                  error is unstable; the BP/EP rows at smaller learning rates are more
                  stable endpoints.
                - Dplus has near-zero saturation, so the failure is unlikely to be caused
                  by tanh state clipping alone. The likely issue is update/objective
                  alignment: the residual target, detach path, or sign/beta construction is
                  not producing a useful descent direction.

                ## Cross-Suite Ranking

                """
            )
        )
        f.write(format_markdown_table(combined_compact))
        f.write(
            dedent(
                """

                Overall decision:

                - Keep the minimal ThreeQ/EP suite as the comparable baseline.
                - Do not scale the current Dplus/DThreeQ rule to MNIST/Fashion yet.
                - The next experiment should be a mechanism diagnostic rather than a larger
                  benchmark: compare Dplus update vectors against BP/EP update vectors
                  using cosine similarity, norm ratio, sign agreement, and a one-step loss
                  decrease test on the same mini-batches.
                """
            )
        )


def main() -> None:
    ensure_dirs()
    minimal_df, minimal_combined = analyze_minimal()
    dthreeq_df, dthreeq_combined = analyze_dthreeq()
    dthreeq_summary = pd.read_csv(TABLES_DIR / "dthreeq_variant_summary.csv")
    dthreeq_by_lr = pd.read_csv(TABLES_DIR / "dthreeq_variant_lr_summary.csv")
    minimal_summary = pd.read_csv(TABLES_DIR / "minimal_variant_summary.csv")
    combined = analyze_combined([minimal_combined, dthreeq_combined])
    write_summary(minimal_summary, dthreeq_summary, dthreeq_by_lr, combined)
    print(f"minimal_trials={len(minimal_df)}")
    print(f"dthreeq_trials={len(dthreeq_df)}")
    print(f"wrote={RESULTS_DIR}")


if __name__ == "__main__":
    main()
