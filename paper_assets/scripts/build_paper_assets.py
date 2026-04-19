"""Build paper-facing figures and summary tables from existing ThreeQ results.

This script intentionally does not launch training. It materializes a compact,
paper-oriented view of the already-completed experiment suites.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_assets"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data"


def ensure_dirs() -> None:
    for path in (FIG, TAB, DATA):
        path.mkdir(parents=True, exist_ok=True)


def read_csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def compact_variant(name: str, max_len: int = 34) -> str:
    replacements = {
        "dthreeq_ep_": "D3Q EP ",
        "dthreeq_dplus_": "D3Q D+ ",
        "epbase3q_legacy_": "EP3Q ",
        "base3q_legacy_": "3Q ",
        "_restorebest": " best",
        "_nudge0p1": " b0.1",
        "_nudge0p05": " b0.05",
        "_lr3e3": " lr3e-3",
        "_lr5e3": " lr5e-3",
        "_decay15": " decay",
        "_10k_e15": " 10k/e15",
        "_net1_5k_e8": " 5k/e8",
        "_": " ",
    }
    out = name
    for old, new in replacements.items():
        out = out.replace(old, new)
    if len(out) > max_len:
        out = out[: max_len - 1] + "..."
    return out


def fmt_float(x: object, digits: int = 4) -> str:
    if x is None:
        return ""
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(value):
        return ""
    if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, float_digits: int = 4) -> str:
    rows = []
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows.extend([header, sep])
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(fmt_float(value, float_digits))
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def save_table(df: pd.DataFrame, name: str, float_digits: int = 4) -> None:
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(markdown_table(df, float_digits) + "\n", encoding="utf-8")


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()


def row(df: pd.DataFrame, variant: str) -> pd.Series:
    matches = df[df["variant"] == variant]
    if matches.empty:
        raise KeyError(f"Missing variant {variant}")
    return matches.iloc[0]


def build_key_result_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minimal = read_csv("experiments/minimal_suite/results/summary_compact.csv")
    legacy = read_csv("experiments/mnist_legacy_repro/results/summary_compact.csv")
    ep_tune = read_csv("experiments/mnist_epthreeq_tune/results/summary_compact.csv")
    mnist_suite = read_csv("experiments/mnist_suite/results/summary_compact.csv")
    d_focus = read_csv("experiments/mnist_dthreeq_focus/results/summary_compact.csv")
    d_long = read_csv("experiments/mnist_dthreeq_longrun/results/summary_compact.csv")
    d_super = read_csv("experiments/mnist_dthreeq_supervision_budget/results/summary_compact.csv")
    d_full = read_csv("experiments/mnist_dthreeq_fullbudget_confirm/results/summary_compact.csv")

    rows = []

    def add(
        section: str,
        dataset_budget: str,
        family: str,
        variant: str,
        best_error: float | None,
        final_error: float | None,
        selected_accuracy: float | None,
        rho_mean: float | None = None,
        saturation: float | None = None,
        note: str = "",
    ) -> None:
        rows.append(
            {
                "section": section,
                "dataset_budget": dataset_budget,
                "family": family,
                "variant": variant,
                "best_error": best_error,
                "final_error": final_error,
                "best_accuracy": None if best_error is None else 1.0 - best_error,
                "selected_accuracy": selected_accuracy,
                "rho_mean": rho_mean,
                "saturation": saturation,
                "note": note,
            }
        )

    for variant in ["ep_cost_w50", "ep_cost_w10", "direct_cost_w50", "direct_cost_w10"]:
        r = row(minimal, variant)
        add(
            "two_moons_minimal",
            "2D moons, 3 seeds",
            "EPThreeQ" if variant.startswith("ep") else "ThreeQ",
            variant,
            r["best_valid_error_mean"],
            r["final_valid_error_mean"],
            None,
            r["rho_mean"],
            r["saturation_mean"],
            "EP 更新优于 direct；w50 early-best 最好但 final 回退。",
        )

    for variant, family in [
        ("base3q_legacy_net1_5k_e8", "ThreeQ legacy"),
        ("epbase3q_legacy_net1_5k_e8", "EPThreeQ legacy"),
    ]:
        r = row(legacy, variant)
        add(
            "mnist_legacy_repro",
            "MNIST 5k/1k, 8 epochs",
            family,
            variant,
            r["best_valid_error_mean"],
            r["final_valid_error_mean"],
            1.0 - r["best_valid_error_mean"],
            r["rho_mean_mean"],
            None,
            "直接调用原始类，验证原始 ThreeQ/EPThreeQ 可学习。",
        )

    for variant in [
        "epbase3q_legacy_10k_e15_beta1",
        "epbase3q_legacy_10k_e15_w5",
        "epbase3q_legacy_10k_e15_base",
    ]:
        r = row(ep_tune, variant)
        add(
            "mnist_epthreeq_tune",
            "MNIST 10k/2k, 15 epochs",
            "EPThreeQ legacy",
            variant,
            r["best_valid_error_mean"],
            r["final_valid_error_mean"],
            1.0 - r["best_valid_error_mean"],
            r["rho_mean_mean"],
            None,
            "EPThreeQ 回到约 86.5% accuracy。",
        )

    for variant in ["dthreeq_ep_nudge_0p01"]:
        r = row(mnist_suite, variant)
        add(
            "mnist_small_budget",
            "MNIST 3k/1k, 3 epochs",
            "DThreeQ EP",
            variant,
            r["best_test_error_mean"],
            r["final_test_error_mean"],
            1.0 - r["best_test_error_mean"],
            None,
            r.get("saturation_mean", np.nan),
            "小预算公共框架压力测试，接近随机。",
        )

    for variant in ["dthreeq_ep_nudge0p1_lr1e3"]:
        r = row(d_focus, variant)
        add(
            "mnist_dthreeq_focus",
            "MNIST 10k/2k, 12 epochs",
            "DThreeQ EP",
            variant,
            r["best_test_error_mean"],
            r["final_test_error_mean"],
            1.0 - r["best_test_error_mean"],
            None,
            r["saturation_mean"],
            "扩大到 legacy 尺度后 DThreeQ EP-nudge 明显学习。",
        )

    for variant in ["dthreeq_ep_nudge0p1_lr3e3"]:
        r = row(d_long, variant)
        add(
            "mnist_dthreeq_longrun",
            "MNIST 10k/2k, 30 epochs",
            "DThreeQ EP",
            variant,
            r["best_test_error_mean"],
            r["final_test_error_mean"],
            1.0 - r["best_test_error_mean"],
            None,
            r["saturation_mean"],
            "当前稳定 EP-nudge baseline，约 83.9% best accuracy。",
        )

    for variant in [
        "dthreeq_ep_ce_nudge0p1_lr3e3",
        "dthreeq_ep_readout_nudge0p1_lr3e3",
        "dthreeq_ep_nudge0p1_lr3e3",
        "dthreeq_ep_noinput_nudge0p1_lr3e3",
    ]:
        r = row(d_super, variant)
        add(
            "mnist_dthreeq_supervision",
            "MNIST 10k/2k, 30 epochs",
            "DThreeQ",
            variant,
            r["best_test_error_mean"],
            r["final_test_error_mean"],
            r["selected_test_accuracy_mean"],
            None,
            r["saturation_mean"],
            "CE-style nudge 是当前最明确正向改动。",
        )

    for variant in [
        "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest",
        "dthreeq_ep_ce_nudge0p1_lr3e3_restorebest",
        "dthreeq_ep_ce_nudge0p1_lr3e3",
        "dthreeq_ep_nudge0p1_lr3e3",
    ]:
        r = row(d_full, variant)
        add(
            "mnist_dthreeq_fullbudget",
            "MNIST 60k/10k, 30 epochs",
            "DThreeQ",
            variant,
            r["best_test_error_mean"],
            r["final_test_error_mean"],
            r["selected_test_accuracy_mean"],
            None,
            r["saturation_mean"],
            "Full-data restore-best 接近 88%，但 final 后期崩塌。",
        )

    key = pd.DataFrame(rows)
    key = key.sort_values(["section", "best_error"], ignore_index=True)
    save_table(key, "key_results", 4)

    mnist = key[key["section"].str.startswith("mnist")].copy()
    save_table(mnist, "mnist_key_results", 4)

    dthreeq_progress = pd.DataFrame(
        [
            {
                "stage": "small public screen",
                "variant": "dthreeq_ep_nudge_0p01",
                "dataset_budget": "3k/1k, 3 epochs",
                "best_accuracy": 1.0 - row(mnist_suite, "dthreeq_ep_nudge_0p01")["best_test_error_mean"],
                "final_accuracy": 1.0 - row(mnist_suite, "dthreeq_ep_nudge_0p01")["final_test_error_mean"],
                "interpretation": "公共框架小预算接近随机，不能代表原始设定。",
            },
            {
                "stage": "legacy-scale focus",
                "variant": "dthreeq_ep_nudge0p1_lr1e3",
                "dataset_budget": "10k/2k, 12 epochs",
                "best_accuracy": 1.0 - row(d_focus, "dthreeq_ep_nudge0p1_lr1e3")["best_test_error_mean"],
                "final_accuracy": 1.0 - row(d_focus, "dthreeq_ep_nudge0p1_lr1e3")["final_test_error_mean"],
                "interpretation": "放大 hidden size 与训练预算后离开随机。",
            },
            {
                "stage": "longrun EP-nudge",
                "variant": "dthreeq_ep_nudge0p1_lr3e3",
                "dataset_budget": "10k/2k, 30 epochs",
                "best_accuracy": 1.0 - row(d_long, "dthreeq_ep_nudge0p1_lr3e3")["best_test_error_mean"],
                "final_accuracy": 1.0 - row(d_long, "dthreeq_ep_nudge0p1_lr3e3")["final_test_error_mean"],
                "interpretation": "稳定 baseline 约 83.5% 到 83.9%。",
            },
            {
                "stage": "CE-style nudge",
                "variant": "dthreeq_ep_ce_nudge0p1_lr3e3",
                "dataset_budget": "10k/2k, 30 epochs",
                "best_accuracy": 1.0 - row(d_super, "dthreeq_ep_ce_nudge0p1_lr3e3")["best_test_error_mean"],
                "final_accuracy": 1.0 - row(d_super, "dthreeq_ep_ce_nudge0p1_lr3e3")["final_test_error_mean"],
                "interpretation": "输出监督从 MSE nudge 改为 CE-style 后到约 86.5%。",
            },
            {
                "stage": "full-data restore-best",
                "variant": "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest",
                "dataset_budget": "60k/10k, 30 epochs",
                "best_accuracy": 1.0 - row(d_full, "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest")["best_test_error_mean"],
                "final_accuracy": 1.0 - row(d_full, "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest")["final_test_error_mean"],
                "interpretation": "best checkpoint 达 88.0%，但 final 仍回退。",
            },
        ]
    )
    save_table(dthreeq_progress, "dthreeq_progress", 4)

    return key, mnist, dthreeq_progress


def plot_inference_convergence() -> None:
    base = ROOT / "AllConnected3QNotTrained/data/n_30,n_clamped_10,n_tracked_10,seed_0"
    lgs = [5, 8, 10, 12, 15]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for lg in lgs:
        legacy_path = base / f"n_30,n_clamped_10,n_tracked_10,Lg_{lg},seed_0.csv"
        cached_path = DATA / f"allconnected_inference_n30_c10_Lg{lg}.csv"
        if legacy_path.exists():
            shutil.copy2(legacy_path, cached_path)
            path = legacy_path
        elif cached_path.exists():
            path = cached_path
        else:
            continue
        df = pd.read_csv(path, comment="#")
        label = f"Lg={lg}"
        axes[0].semilogy(df["epoch"], df["delta"], label=label, linewidth=2)
        axes[1].plot(df["epoch"], df["rho"], label=label, linewidth=2)
        axes[2].plot(df["epoch"], df["E"], label=label, linewidth=2)

    axes[0].set_title("Inference residual decay")
    axes[0].set_xlabel("state step")
    axes[0].set_ylabel("delta, log scale")
    axes[0].grid(alpha=0.25)

    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="rho=1")
    axes[1].set_title("Local spectral radius")
    axes[1].set_xlabel("state step")
    axes[1].set_ylabel("rho(J)")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Energy during inference")
    axes[2].set_xlabel("state step")
    axes[2].set_ylabel("energy")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best", fontsize=8)

    savefig("fig01_threeq_inference_convergence.png")


def plot_training_convergence() -> None:
    legacy = read_csv("experiments/mnist_legacy_repro/results/curves.csv")
    ep_tune = read_csv("experiments/mnist_epthreeq_tune/results/curves.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8))
    for variant, df in legacy.groupby("variant"):
        axes[0].plot(df["epoch"], 1.0 - df["valid_error"], marker="o", label=compact_variant(variant, 26))
    axes[0].set_title("Legacy ThreeQ / EPThreeQ")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("validation accuracy")
    axes[0].set_ylim(0.15, 0.9)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for variant in [
        "epbase3q_legacy_10k_e15_beta1",
        "epbase3q_legacy_10k_e15_w5",
        "epbase3q_legacy_10k_e15_base",
        "epbase3q_legacy_10k_e15_alpha_hi",
    ]:
        df = ep_tune[ep_tune["variant"] == variant]
        axes[1].plot(df["epoch"], 1.0 - df["valid_error"], marker="o", label=compact_variant(variant, 26))
    axes[1].set_title("EPThreeQ tune")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_ylim(0.2, 0.92)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)

    for variant in [
        "epbase3q_legacy_10k_e15_beta1",
        "epbase3q_legacy_10k_e15_w5",
        "base3q_legacy_net1_5k_e8",
        "epbase3q_legacy_net1_5k_e8",
    ]:
        df = pd.concat([legacy, ep_tune])
        sub = df[df["variant"] == variant]
        if not sub.empty:
            axes[2].plot(sub["epoch"], sub["rho_mean"], marker="o", label=compact_variant(variant, 24))
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[2].set_title("rho during training")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("rho mean")
    axes[2].set_ylim(0.2, 0.7)
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=7)

    savefig("fig02_threeq_training_convergence.png")


def plot_mnist_performance(mnist: pd.DataFrame) -> None:
    selections = [
        ("mnist_legacy_repro", "base3q_legacy_net1_5k_e8"),
        ("mnist_legacy_repro", "epbase3q_legacy_net1_5k_e8"),
        ("mnist_epthreeq_tune", "epbase3q_legacy_10k_e15_beta1"),
        ("mnist_dthreeq_longrun", "dthreeq_ep_nudge0p1_lr3e3"),
        ("mnist_dthreeq_supervision", "dthreeq_ep_ce_nudge0p1_lr3e3"),
        ("mnist_dthreeq_fullbudget", "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest"),
    ]
    rows = []
    for section, variant in selections:
        matched = mnist[(mnist["section"] == section) & (mnist["variant"] == variant)]
        if matched.empty:
            raise KeyError((section, variant))
        rows.append(matched.iloc[0])
    sub = pd.DataFrame(rows)
    sub["plot_accuracy"] = sub["selected_accuracy"].fillna(sub["best_accuracy"])
    sub["label"] = sub["variant"].map(lambda x: compact_variant(x, 31))

    plt.figure(figsize=(10.5, 4.4))
    colors = ["#8da0cb", "#66c2a5", "#1b9e77", "#fc8d62", "#e78ac3", "#a6d854"]
    bars = plt.bar(sub["label"], sub["plot_accuracy"] * 100, color=colors)
    plt.ylabel("accuracy (%)")
    plt.title("MNIST current performance: ThreeQ, EPThreeQ, DThreeQ")
    plt.ylim(0, 95)
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=18, ha="right")
    for bar, value in zip(bars, sub["plot_accuracy"] * 100):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
    savefig("fig03_mnist_current_performance.png")


def plot_dthreeq_progress(progress: pd.DataFrame) -> None:
    plt.figure(figsize=(10.5, 4.3))
    x = np.arange(len(progress))
    width = 0.38
    plt.bar(x - width / 2, progress["best_accuracy"] * 100, width, label="best / selected", color="#66c2a5")
    plt.bar(x + width / 2, progress["final_accuracy"] * 100, width, label="final", color="#fc8d62")
    plt.ylabel("accuracy (%)")
    plt.title("DThreeQ MNIST improvement path")
    plt.xticks(x, progress["stage"], rotation=18, ha="right")
    plt.ylim(0, 95)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    for idx, value in enumerate(progress["best_accuracy"] * 100):
        plt.text(idx - width / 2, value + 1.3, f"{value:.1f}", ha="center", fontsize=8)
    savefig("fig04_dthreeq_improvement_process.png")


def plot_dthreeq_training_curves() -> None:
    super_curves = read_csv("experiments/mnist_dthreeq_supervision_budget/results/curves.csv")
    full_curves = read_csv("experiments/mnist_dthreeq_fullbudget_confirm/results/curves.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.0))

    variants_10k = [
        "dthreeq_ep_nudge0p1_lr3e3",
        "dthreeq_ep_ce_nudge0p1_lr3e3",
        "dthreeq_ep_readout_nudge0p1_lr3e3",
        "dthreeq_ep_tanh_nudge0p1_lr5e3_restorebest",
        "dthreeq_ep_noinput_nudge0p1_lr3e3",
    ]
    for variant in variants_10k:
        sub = super_curves[super_curves["variant"] == variant]
        if sub.empty:
            continue
        grouped = sub.groupby("epoch", as_index=False)["test_error"].mean()
        axes[0].plot(grouped["epoch"], 1.0 - grouped["test_error"], label=compact_variant(variant, 28), linewidth=2)
    axes[0].set_title("DThreeQ 10k/2k curves")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("test accuracy")
    axes[0].set_ylim(0.0, 0.92)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7)

    variants_full = [
        "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest",
        "dthreeq_ep_ce_nudge0p1_lr3e3_restorebest",
        "dthreeq_ep_ce_nudge0p1_lr5e3_decay15_restorebest",
        "dthreeq_ep_nudge0p1_lr3e3",
    ]
    for variant in variants_full:
        sub = full_curves[full_curves["variant"] == variant]
        if sub.empty:
            continue
        grouped = sub.groupby("epoch", as_index=False)["test_error"].mean()
        axes[1].plot(grouped["epoch"], 1.0 - grouped["test_error"], label=compact_variant(variant, 30), linewidth=2)
    axes[1].set_title("DThreeQ full-data curves")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test accuracy")
    axes[1].set_ylim(0.0, 0.92)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7)

    savefig("fig05_dthreeq_training_curves.png")


def build_mechanism_tables_and_plots() -> pd.DataFrame:
    diag = read_csv("experiments/mechanism_diagnostic/results/results_raw.csv")
    fix = read_csv("experiments/dplus_fix_diagnostic/results/results_raw.csv")

    targets = [
        "direct_plus",
        "nudge_0p1_plus",
        "nudge_0p01_plus",
        "nudge_0p001_plus",
        "nudge_0p01_plusminus",
        "gradual_100_0p01_plus",
    ]
    rows = []
    for target in targets:
        sub = diag[diag["target_name"] == target]
        if sub.empty:
            continue
        rows.append(
            {
                "target": target,
                "dplus_vs_bp_cosine": sub["dplus_vs_bp_forward_cosine"].mean(),
                "dplus_vs_bp_norm_ratio": sub["dplus_vs_bp_forward_norm_ratio"].mean(),
                "dplus_vs_bp_sign": sub["dplus_vs_bp_forward_sign_agreement"].mean(),
                "dplus_vs_ep_cosine": sub["dplus_vs_ep_forward_cosine"].mean(),
                "dplus_vs_ep_norm_ratio": sub["dplus_vs_ep_forward_norm_ratio"].mean(),
                "dplus_raw_free_mse_decrease": sub["dplus_free_mse_decrease_raw"].mean(),
                "dplus_bp_normed_free_mse_decrease": sub["dplus_free_mse_decrease_bp_normed"].mean(),
            }
        )
    mechanism = pd.DataFrame(rows)
    save_table(mechanism, "mechanism_diagnostic_key_metrics", 4)

    best_fix = (
        fix.groupby(["objective_name", "target_name"], as_index=False)
        .agg(
            dplus_vs_bp_forward_cosine=("dplus_vs_bp_forward_cosine", "mean"),
            dplus_vs_bp_forward_sign_agreement=("dplus_vs_bp_forward_sign_agreement", "mean"),
            dplus_vs_bp_forward_norm_ratio=("dplus_vs_bp_forward_norm_ratio", "mean"),
            dplus_vs_ep_forward_cosine=("dplus_vs_ep_forward_cosine", "mean"),
            dplus_bp_scaled_free_mse_decrease=("dplus_bp_scaled_free_mse_decrease", "mean"),
            dplus_ep_scaled_free_mse_decrease=("dplus_ep_scaled_free_mse_decrease", "mean"),
        )
        .sort_values("dplus_vs_bp_forward_cosine", ascending=False)
        .head(12)
    )
    save_table(best_fix, "dplus_fix_top_candidates", 4)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    labels = [compact_variant(x, 18) for x in mechanism["target"]]
    x = np.arange(len(mechanism))
    axes[0].bar(x, mechanism["dplus_vs_bp_cosine"], color="#8da0cb", label="Dplus vs BP")
    axes[0].bar(x, mechanism["dplus_vs_ep_cosine"], color="#66c2a5", alpha=0.6, label="Dplus vs EP")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("forward cosine")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylim(-1.0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(x, mechanism["dplus_vs_bp_sign"], color="#fc8d62")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Dplus vs BP sign agreement")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].set_ylim(0.4, 0.62)
    axes[1].grid(axis="y", alpha=0.25)

    norm_ratio = mechanism["dplus_vs_bp_norm_ratio"].replace(0, np.nan)
    axes[2].bar(x, norm_ratio, color="#e78ac3")
    axes[2].set_yscale("log")
    axes[2].set_title("Dplus / BP norm ratio")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].grid(axis="y", alpha=0.25)

    savefig("fig06_mechanism_direction_metrics.png")

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.bar(x - 0.2, mechanism["dplus_raw_free_mse_decrease"], 0.4, label="raw", color="#66c2a5")
    ax.bar(x + 0.2, mechanism["dplus_bp_normed_free_mse_decrease"], 0.4, label="BP-normed", color="#fc8d62")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title("Dplus one-step free-MSE decrease")
    ax.set_ylabel("loss decrease")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    savefig("fig07_mechanism_one_step_loss.png")

    return mechanism


def plot_input_supervision() -> None:
    super_df = read_csv("experiments/mnist_dthreeq_supervision_budget/results/summary_compact.csv")
    full_df = read_csv("experiments/mnist_dthreeq_fullbudget_confirm/results/summary_compact.csv")
    super_df = super_df.assign(suite="10k/2k")
    full_df = full_df.assign(suite="60k/10k")
    df = pd.concat([super_df, full_df], ignore_index=True)
    keep = df[df["variant"].str.contains("dthreeq_ep_(?:ce|readout|noinput|input1over|nudge0p1|signed)", regex=True)].copy()
    keep["accuracy"] = keep["selected_test_accuracy_mean"]
    keep["label"] = keep["variant"].map(lambda x: compact_variant(x, 26))

    plt.figure(figsize=(10, 4.5))
    colors = keep["suite"].map({"10k/2k": "#66c2a5", "60k/10k": "#fc8d62"}).fillna("#8da0cb")
    plt.scatter(
        keep["weighted_input_recon_energy_frac_mean"],
        keep["accuracy"] * 100,
        s=90,
        c=colors,
        edgecolors="black",
        linewidth=0.4,
    )
    for _, r in keep.iterrows():
        if "input1over" in r["variant"] or "noinput" in r["variant"] or "ce_nudge0p1" in r["variant"] or "signed" in r["variant"]:
            plt.text(
                r["weighted_input_recon_energy_frac_mean"] + 0.012,
                r["accuracy"] * 100 + 0.5,
                compact_variant(r["variant"], 18),
                fontsize=7,
            )
    plt.xlabel("weighted input reconstruction energy fraction")
    plt.ylabel("selected/final accuracy (%)")
    plt.title("Input energy vs supervision outcomes")
    plt.ylim(10, 91)
    plt.grid(alpha=0.25)
    savefig("fig08_input_energy_supervision.png")


def copy_existing_evidence() -> None:
    existing_dir = FIG / "existing"
    existing_dir.mkdir(exist_ok=True)
    files = [
        (
            "AllConnected3QTrained/g=relu/n_30,n_clamped_20,n_tracked_10,seed_0/n_30,n_clamped_20,n_tracked_10,Lg_10,seed_0_train.png",
            "allconnected_train_convergence.png",
        ),
        (
            "AllConnected3QTrained/g=relu/n_30,n_clamped_20,n_tracked_10,seed_0/n_30,n_clamped_20,n_tracked_10,Lg_10,seed_0_train_rho.png",
            "allconnected_train_rho.png",
        ),
        ("Base3QClampWithLinear/twomoons_decision_boundary.png", "base3q_clamp_twomoons.png"),
        ("EPBase3Q/twomoons_decision_boundary.png", "epbase3q_twomoons.png"),
        ("CNN3Q/twomoons_decision_boundary.png", "cnn3q_twomoons.png"),
        ("EPCNN3Q/twomoons_decision_boundary.png", "epcnn3q_twomoons.png"),
        (
            "experiments/mnist_dthreeq_supervision_budget/results/figures/mnist_dthreeq_supervision_error.png",
            "dthreeq_supervision_10k_error.png",
        ),
        (
            "experiments/mnist_dthreeq_fullbudget_confirm/results/figures/mnist_dthreeq_supervision_error.png",
            "dthreeq_fullbudget_error.png",
        ),
    ]
    manifest = []
    for rel, output_name in files:
        src = ROOT / rel
        dst = existing_dir / output_name
        if src.exists():
            shutil.copy2(src, dst)
        if dst.exists():
            manifest.append({"source": rel, "copied_to": str(dst.relative_to(OUT))})
    pd.DataFrame(manifest).to_csv(DATA / "existing_figure_manifest.csv", index=False)


def write_paper_summary(key: pd.DataFrame, progress: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    ep_best = key[key["variant"] == "epbase3q_legacy_10k_e15_beta1"].iloc[0]
    d_best = key[key["variant"] == "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest"].iloc[0]
    d_ce = key[
        (key["section"] == "mnist_dthreeq_supervision")
        & (key["variant"] == "dthreeq_ep_ce_nudge0p1_lr3e3")
    ].iloc[0]
    legacy_base = key[key["variant"] == "base3q_legacy_net1_5k_e8"].iloc[0]
    legacy_ep = key[key["variant"] == "epbase3q_legacy_net1_5k_e8"].iloc[0]
    plusminus = mechanism[mechanism["target"] == "nudge_0p01_plusminus"].iloc[0]
    direct = mechanism[mechanism["target"] == "direct_plus"].iloc[0]

    text = f"""# ThreeQ 论文材料汇总

生成日期：2026-04-19

这个目录把目前已有实验整理成论文写作可直接引用的图、表和结论。所有图表均由已有 `experiments/*/results` 与 legacy 收敛性文件生成，没有重新训练模型。

## 1. 当前核心结论

1. **ThreeQ inference 收敛性有清晰实验证据。** `fig01_threeq_inference_convergence.png` 展示了固定点推断中 $\\rho(J)<1$ 时 `delta` 呈指数式下降；当局部谱半径长期超过 1，例如 `Lg=15`，`delta` 明显停滞。这与理论证明中的充分条件一致：若固定点处 $\\rho(J)<1$，局部 inference 指数稳定。
2. **$\\rho(J)>1$ 的发散侧证已补齐。** `fig09_controlled_inference_rho_divergence.png` 使用受控线性 ThreeQ 母模型 $T(u)=\\rho u$，展示 $\\rho=1.05,1.20$ 时扰动范数指数增长；`fig10_training_rho_boundary_stress_test.png` 展示 bounded-gradient 大步训练更新跨过 $\\rho=1$ 后 post-update inference 发散。
3. **训练过程也基本符合“小步更新保持稳定裕度”的判断。** `fig02_threeq_training_convergence.png` 显示 legacy ThreeQ/EPThreeQ 训练中 `rho_mean` 持续低于 1；同时 EPThreeQ validation accuracy 上升更快。`EPBase3Q` 在 MNIST 5k/1k、8 epoch 下 best accuracy 为 {legacy_ep['best_accuracy']*100:.1f}%，而 direct `Base3Q` 为 {legacy_base['best_accuracy']*100:.1f}%。
4. **EPThreeQ 是当前最稳的原始 ThreeQ 改进线。** 在 MNIST 10k/2k、15 epoch 中，`epbase3q_legacy_10k_e15_beta1` 达到 best accuracy {ep_best['best_accuracy']*100:.1f}%，final accuracy {100*(1-ep_best['final_error']):.1f}%。这支持“原始版本能到 85% 左右，EP 更稳定”的判断。
5. **DThreeQ 的 EP-nudge/CE-nudge 版本可学习，但还没有稳定复现 90%+。** 10k/2k、30 epoch 下 CE-style nudge selected accuracy 为 {d_ce['selected_accuracy']*100:.1f}%；60k/10k full-data + restore-best 的最好结果为 {d_best['selected_accuracy']*100:.1f}%，但 final accuracy 回退到 {100*(1-d_best['final_error']):.1f}%。
6. **Dplus/residual-delta 主训练规则目前不是 BP-like。** 在同一批 mini-batch 上，`direct_plus` 的 Dplus-vs-BP forward cosine 只有 {direct['dplus_vs_bp_cosine']:.3f}，sign agreement 为 {direct['dplus_vs_bp_sign']:.3f}；`plusminus` 变体 cosine 为 {plusminus['dplus_vs_bp_cosine']:.3f} 且 one-step free-MSE decrease 为负，说明符号抵消是明确失败模式。
7. **输入重构项不是简单删掉就能解决。** `fig08_input_energy_supervision.png` 显示 input residual weight 设为 0 或极小会降低 final accuracy；输入项既压制分类监督，也在维持表示。下一步应改为 layer-normalized/boundary-clamped input energy，而不是直接移除。

## 2. 图表索引

| 编号 | 文件 | 论文用途 |
|---|---|---|
| Fig.1 | `figures/fig01_threeq_inference_convergence.png` | 证明式实验：inference 残差、能量与 $\\rho(J)$ 的关系 |
| Fig.2 | `figures/fig02_threeq_training_convergence.png` | 训练收敛：legacy ThreeQ/EPThreeQ accuracy 与 `rho_mean` |
| Fig.3 | `figures/fig03_mnist_current_performance.png` | MNIST 当前性能总览：ThreeQ、EPThreeQ、DThreeQ |
| Fig.4 | `figures/fig04_dthreeq_improvement_process.png` | DThreeQ 从小预算失败到 full-data 88% 的改进路径 |
| Fig.5 | `figures/fig05_dthreeq_training_curves.png` | DThreeQ 10k/full-data 训练曲线，显示 early-best 与后期崩塌 |
| Fig.6 | `figures/fig06_mechanism_direction_metrics.png` | Dplus、BP、EP 更新向量方向诊断 |
| Fig.7 | `figures/fig07_mechanism_one_step_loss.png` | Dplus one-step loss decrease 与 plusminus 失败 |
| Fig.8 | `figures/fig08_input_energy_supervision.png` | 输入重构能量比例与分类 accuracy 的关系 |
| Fig.9 | `figures/fig09_controlled_inference_rho_divergence.png` | 受控线性 ThreeQ 母模型中 $\\rho>1$ 的指数发散证据 |
| Fig.10 | `figures/fig10_training_rho_boundary_stress_test.png` | 训练更新跨过 $\\rho=1$ 后 post-update inference 发散 |

## 3. 表格索引

| 文件 | 内容 |
|---|---|
| `tables/key_results.csv` / `.md` | two moons、MNIST、EPThreeQ、DThreeQ 关键结果总表 |
| `tables/mnist_key_results.csv` / `.md` | MNIST 相关结果子表 |
| `tables/dthreeq_progress.csv` / `.md` | DThreeQ 改进过程分阶段表 |
| `tables/mechanism_diagnostic_key_metrics.csv` / `.md` | Dplus/BP/EP cosine、norm ratio、sign agreement 与 one-step loss |
| `tables/dplus_fix_top_candidates.csv` / `.md` | residual normalization、scale calibration、layer gain 的候选诊断 |
| `tables/controlled_inference_rho_summary.csv` / `.md` | 受控线性 inference 收敛/发散摘要 |
| `tables/training_stability_stress_summary.csv` / `.md` | 小步/大步训练更新稳定性压力测试摘要 |
| `latex/threeq_convergence_report.tex` / `.pdf` | 可迁移到论文的 LaTeX 学术报告 |

## 4. 可用于论文正文的判断

**理论与实验闭环。** 目前可以把理论部分表述为局部充分条件，而不是全局收敛定理：$\\rho(J)<1$ 保证固定点局部稳定，小学习率保证稳定裕度局部保持。Fig.1 和 Fig.2 正好支撑这个说法，同时也展示了条件不满足时的失败边界。

**ThreeQ/EPThreeQ 的经验事实。** 原始 ThreeQ 架构并非不能学 MNIST；问题在于慢收敛与超参敏感。EPThreeQ 在同等 legacy 设置下明显优于 direct ThreeQ，说明自由相/受扰相差分比直接优化 weak-state total energy 更合理。

**DThreeQ 的当前定位。** DThreeQ 的 EP-nudge 分支已经可作为可行能量模型路线继续推进；Dplus/residual-delta 分支仍应作为机制研究对象，而不是主 benchmark 规则。外部 90%+ 若真实存在，最可能差异来自 CE/softmax 输出、best checkpoint、完整预算、额外 readout/pretraining、layer-normalized energy 或更强的后期稳定化。

**下一步最值得写入论文展望的方向。** 优先做 MNIST mini-batch mechanism diagnostic、boundary-clamped input energy、CE beta schedule、spectral/damping regularization，以及 signed residual target / output-to-hidden residual injection。继续只调 residual norm、$\\beta$ 缩放或固定 layer gain 的边际收益很低。

## 5. DThreeQ 改进过程简表

{markdown_table(progress, 4)}
"""
    (OUT / "PAPER_SUMMARY.md").write_text(text, encoding="utf-8")


def write_readme() -> None:
    text = """# Paper Assets

This directory contains paper-facing ThreeQ figures, tables, and a Chinese summary generated from completed experiments.

Run:

```bash
python paper_assets/scripts/build_paper_assets.py
```

The script does not train models. It reads existing CSV/PNG artifacts under `experiments/` and selected legacy convergence outputs, then writes:

- `figures/`: numbered figures for inference convergence, training convergence, MNIST performance, DThreeQ progress, and mechanism diagnostics.
- `tables/`: compact CSV/Markdown tables.
- `data/`: figure manifests and copied provenance metadata.
- `PAPER_SUMMARY.md`: Chinese paper-writing summary.
- `latex/threeq_convergence_report.tex`: paper-ready LaTeX report for ThreeQ inference/training convergence, including rho > 1 divergence evidence.

Additional convergence report build:

```bash
python paper_assets/scripts/build_convergence_report.py
cd paper_assets/latex && xelatex -interaction=nonstopmode -halt-on-error threeq_convergence_report.tex
```
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    key, mnist, progress = build_key_result_tables()
    plot_inference_convergence()
    plot_training_convergence()
    plot_mnist_performance(mnist)
    plot_dthreeq_progress(progress)
    plot_dthreeq_training_curves()
    mechanism = build_mechanism_tables_and_plots()
    plot_input_supervision()
    copy_existing_evidence()
    write_paper_summary(key, progress, mechanism)
    write_readme()
    print(f"Wrote paper assets to {OUT}")


if __name__ == "__main__":
    main()
