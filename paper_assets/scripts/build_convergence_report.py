"""Build convergence/divergence figures and a LaTeX report for ThreeQ.

The generated material is intentionally paper-facing. It combines existing
legacy ThreeQ convergence data with controlled mother-model stress tests that
isolate the rho < 1 and rho > 1 regimes.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_assets"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data"
LATEX = OUT / "latex"


def ensure_dirs() -> None:
    for path in (FIG, TAB, DATA, LATEX):
        path.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=220, bbox_inches="tight")
    plt.close()


def fmt_float(x: object, digits: int = 4) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(value):
        return ""
    if abs(value) >= 1000 or (0 < abs(value) < 1e-3):
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = [header, sep]
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(fmt_float(value, digits))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def save_table(df: pd.DataFrame, name: str, digits: int = 4) -> None:
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.md").write_text(markdown_table(df, digits) + "\n", encoding="utf-8")


def build_controlled_inference() -> pd.DataFrame:
    """Generate exact linear ThreeQ mother-model inference curves.

    With identity activation and W=(n-1)rho I, the mother-model update
    u_{t+1}=T(u_t; W) has Jacobian rho I at the fixed point u*=0.
    """

    steps = np.arange(0, 51)
    initial_norm = 1.0e-3
    cases = [
        ("stable_rho_0p80", 0.80),
        ("stable_rho_0p95", 0.95),
        ("unstable_rho_1p05", 1.05),
        ("unstable_rho_1p20", 1.20),
    ]
    rows = []
    for name, rho in cases:
        norms = initial_norm * (rho ** steps)
        residual = np.abs(rho - 1.0) * norms
        for step, norm, res in zip(steps, norms, residual):
            rows.append(
                {
                    "case": name,
                    "rho": rho,
                    "step": int(step),
                    "state_norm": float(norm),
                    "fixed_point_residual": float(res),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "controlled_inference_rho_curves.csv", index=False)

    summary = (
        df.groupby(["case", "rho"], as_index=False)
        .agg(
            initial_state_norm=("state_norm", "first"),
            final_state_norm=("state_norm", "last"),
            growth_factor=("state_norm", lambda s: float(s.iloc[-1] / s.iloc[0])),
            final_residual=("fixed_point_residual", "last"),
        )
        .sort_values("rho")
    )
    save_table(summary, "controlled_inference_rho_summary", 4)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for name, rho in cases:
        sub = df[df["case"] == name]
        label = f"rho={rho:.2f}"
        style = "-" if rho < 1 else "--"
        axes[0].semilogy(sub["step"], sub["state_norm"], style, linewidth=2.2, label=label)
        axes[1].semilogy(sub["step"], sub["fixed_point_residual"], style, linewidth=2.2, label=label)
    axes[0].set_title("Controlled ThreeQ inference")
    axes[0].set_xlabel("inference step")
    axes[0].set_ylabel("state perturbation norm")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Fixed-point residual")
    axes[1].set_xlabel("inference step")
    axes[1].set_ylabel("residual norm")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    savefig("fig09_controlled_inference_rho_divergence.png")

    return summary


def build_training_stress_test() -> pd.DataFrame:
    """Generate a bounded-gradient update stress test.

    The scalar rho parameter follows rho_{k+1}=rho_k + eta. The increment eta
    is a bounded training update. Small eta keeps rho below one; large eta
    crosses the stability boundary, after which post-update inference diverges.
    """

    epochs = np.arange(0, 25)
    initial_rho = 0.70
    initial_norm = 1.0e-3
    infer_steps = np.arange(0, 41)
    cases = [
        ("small_step_eta_0p010", 0.010),
        ("large_step_eta_0p035", 0.035),
    ]

    train_rows = []
    infer_rows = []
    for name, eta in cases:
        for epoch in epochs:
            rho = initial_rho + eta * epoch
            final_norm = initial_norm * (rho ** infer_steps[-1])
            train_rows.append(
                {
                    "case": name,
                    "eta": eta,
                    "epoch": int(epoch),
                    "rho": float(rho),
                    "post_update_final_norm": float(final_norm),
                    "stable": bool(rho < 1.0),
                }
            )
        final_rho = initial_rho + eta * epochs[-1]
        for step in infer_steps:
            infer_rows.append(
                {
                    "case": name,
                    "eta": eta,
                    "final_rho": float(final_rho),
                    "inference_step": int(step),
                    "state_norm": float(initial_norm * (final_rho ** step)),
                }
            )

    train_df = pd.DataFrame(train_rows)
    infer_df = pd.DataFrame(infer_rows)
    train_df.to_csv(DATA / "training_stability_stress_epochs.csv", index=False)
    infer_df.to_csv(DATA / "training_stability_post_update_inference.csv", index=False)

    summary_rows = []
    for name, eta in cases:
        sub = train_df[train_df["case"] == name]
        crossed = sub[sub["rho"] >= 1.0]
        post = infer_df[infer_df["case"] == name]
        summary_rows.append(
            {
                "case": name,
                "eta": eta,
                "initial_rho": initial_rho,
                "final_rho": sub["rho"].iloc[-1],
                "first_epoch_rho_ge_1": "" if crossed.empty else int(crossed["epoch"].iloc[0]),
                "post_update_growth_factor": post["state_norm"].iloc[-1] / post["state_norm"].iloc[0],
                "post_update_final_norm": post["state_norm"].iloc[-1],
            }
        )
    summary = pd.DataFrame(summary_rows)
    save_table(summary, "training_stability_stress_summary", 4)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    for name, eta in cases:
        sub = train_df[train_df["case"] == name]
        label = "small step" if "small" in name else "large step"
        axes[0].plot(sub["epoch"], sub["rho"], marker="o", linewidth=2.1, label=f"{label}, eta={eta:.3f}")
        axes[1].semilogy(sub["epoch"], sub["post_update_final_norm"], marker="o", linewidth=2.1, label=label)
        post = infer_df[infer_df["case"] == name]
        axes[2].semilogy(post["inference_step"], post["state_norm"], linewidth=2.2, label=f"{label}, final rho={post['final_rho'].iloc[0]:.2f}")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Training update crosses rho=1")
    axes[0].set_xlabel("training epoch")
    axes[0].set_ylabel("rho")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].axhline(initial_norm, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_title("Post-update inference final norm")
    axes[1].set_xlabel("training epoch")
    axes[1].set_ylabel("norm after 40 inference steps")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].set_title("Final-epoch inference")
    axes[2].set_xlabel("inference step")
    axes[2].set_ylabel("state perturbation norm")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    savefig("fig10_training_rho_boundary_stress_test.png")

    return summary


def read_existing_legacy_summary() -> pd.DataFrame:
    rows = []
    for lg in [5, 8, 10, 12, 15]:
        path = DATA / f"allconnected_inference_n30_c10_Lg{lg}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, comment="#")
        rows.append(
            {
                "Lg": lg,
                "initial_delta": df["delta"].iloc[0],
                "final_delta": df["delta"].iloc[-1],
                "final_rho": df["rho"].iloc[-1],
                "delta_reduction_factor": df["delta"].iloc[-1] / df["delta"].iloc[0],
                "classification": "convergent" if df["delta"].iloc[-1] < 1.0e-3 and df["rho"].iloc[-1] < 1.0 else "non_convergent_or_slow",
            }
        )
    summary = pd.DataFrame(rows)
    save_table(summary, "legacy_inference_boundary_summary", 4)
    return summary


def latex_table_rows(df: pd.DataFrame, columns: list[str]) -> str:
    def latex_escape(value: object) -> str:
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    lines = []
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(fmt_float(value, 4))
            else:
                cells.append(latex_escape(value))
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def write_latex_report(
    controlled_summary: pd.DataFrame,
    training_summary: pd.DataFrame,
    legacy_summary: pd.DataFrame,
) -> Path:
    legacy_rows = latex_table_rows(
        legacy_summary[["Lg", "initial_delta", "final_delta", "final_rho", "delta_reduction_factor", "classification"]],
        ["Lg", "initial_delta", "final_delta", "final_rho", "delta_reduction_factor", "classification"],
    )
    controlled_rows = latex_table_rows(
        controlled_summary[["case", "rho", "initial_state_norm", "final_state_norm", "growth_factor", "final_residual"]],
        ["case", "rho", "initial_state_norm", "final_state_norm", "growth_factor", "final_residual"],
    )
    training_rows = latex_table_rows(
        training_summary[
            [
                "case",
                "eta",
                "initial_rho",
                "final_rho",
                "first_epoch_rho_ge_1",
                "post_update_growth_factor",
                "post_update_final_norm",
            ]
        ],
        [
            "case",
            "eta",
            "initial_rho",
            "final_rho",
            "first_epoch_rho_ge_1",
            "post_update_growth_factor",
            "post_update_final_norm",
        ],
    )

    text = rf"""\documentclass[UTF8,zihao=-4]{{ctexart}}
\usepackage[a4paper,margin=2.4cm]{{geometry}}
\usepackage{{amsmath,amssymb,amsthm,bm}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{float}}
\graphicspath{{{{../figures/}}{{../figures/existing/}}}}

\title{{ThreeQ 推断与训练收敛性的实验验证报告}}
\author{{ThreeQ Research}}
\date{{2026-04-19}}

\newtheorem{{proposition}}{{命题}}

\begin{{document}}
\maketitle

\begin{{abstract}}
本文整理 ThreeQ 母模型及其 legacy 训练实验中的收敛性证据。已有图像主要展示了 $\rho(J)<1$ 时推断残差下降和训练过程稳定；为补足反例侧证，本文新增受控线性 ThreeQ 母模型实验，在完全相同的固定点更新形式下设置 $\rho(J)>1$，直接观察扰动范数指数发散。同时，本文给出 bounded-gradient 训练更新的 small-step/large-step 对照：小步更新保持 $\rho<1$ 并维持推断稳定，大步更新跨过 $\rho=1$ 后导致 post-update inference 发散。该报告中的公式、文字和图片可直接迁移到论文的收敛性实验章节。
\end{{abstract}}

\section{{模型与理论命题}}

ThreeQ 母模型将全局状态记为 $u\in\mathbb{{R}}^n$，局部预测映射写作
\begin{{equation}}
T(u;W)=\frac{{1}}{{n-1}}Wg(u),
\end{{equation}}
其中 $g$ 是逐元素激活函数。固定点 $u^\star$ 满足
\begin{{equation}}
u^\star=T(u^\star;W).
\end{{equation}}
在固定点附近定义扰动 $\delta=u-u^\star$，局部 Jacobian 为
\begin{{equation}}
J(u^\star,W)=\frac{{\partial T}}{{\partial u}}(u^\star,W)
       =\frac{{1}}{{n-1}}W\operatorname{{diag}}(g'(u^\star)).
\end{{equation}}

\begin{{proposition}}[局部推断稳定性]
若固定点 $u^\star$ 处满足 $\rho(J(u^\star,W))<1$，则离散固定点迭代 $u^{{t+1}}=T(u^t;W)$ 的线性化扰动满足 $\delta_{{t+1}}\approx J\delta_t$，并在局部指数收敛。若存在特征值 $|\lambda|>1$，则沿对应不稳定特征方向的扰动在局部线性化系统中指数增长。
\end{{proposition}}

训练稳定性命题则是上述条件的参数连续性推论。令训练更新为
\begin{{equation}}
W^+=W-\eta \nabla_W E(u^\star(W),W).
\end{{equation}}
若当前点存在稳定裕度 $\rho(J)\le 1-\varepsilon$，且梯度范数有界，则足够小的 $\eta$ 会保持更新后 $\rho(J(W^+))<1$。该命题并不声称任意大步长都稳定；相反，大步长可以把系统推过 $\rho=1$ 的边界，导致下一轮推断不稳定。

\section{{实验设计}}

本文使用三类互补实验。

\paragraph{{实验 A：legacy 非线性 ThreeQ 推断。}}
使用已有 \texttt{{AllConnected3QNotTrained}} 结果，比较不同 $L_g$ 下的推断残差、能量和局部谱半径。该实验来自原始非线性实现，因此可展示实际代码中的 $\rho<1$ 收敛以及 $\rho>1$ 时的非收敛/慢收敛边界。

\paragraph{{实验 B：受控线性 ThreeQ 推断。}}
为了直接观察 $\rho>1$ 的指数发散，设 $g(u)=u$，并取 $W=(n-1)\rho I$。此时 $T(u;W)=\rho u$，固定点为 $u^\star=0$，且 $J=\rho I$。因此 $\rho<1$ 与 $\rho>1$ 的收敛/发散差异不受激活饱和、截断或非线性边界影响。

\paragraph{{实验 C：bounded-gradient 训练更新压力测试。}}
令标量稳定性参数满足 $\rho_{{k+1}}=\rho_k+\eta$。这等价于一个范数有界的参数更新方向。small-step 设置保持 $\rho_k<1$；large-step 设置跨过 $\rho=1$。每个训练 epoch 后运行固定步数推断，观察 post-update inference 的最终扰动范数。

\paragraph{{实验 D：legacy MNIST 训练稳定性。}}
使用已有 \texttt{{mnist\_legacy\_repro}} 与 \texttt{{mnist\_epthreeq\_tune}} 曲线，展示实际训练中 EPThreeQ 的 validation accuracy 上升，同时 \texttt{{rho\_mean}} 保持在 1 以下。

\section{{实验结果}}

\subsection{{非线性 legacy 推断：$\rho<1$ 收敛，$\rho>1$ 非收敛}}

图~\ref{{fig:legacy-inference}} 显示，当 $L_g=5,8,10$ 时，后期 $\rho(J)$ 明显低于 1，推断残差下降到 $10^{{-5}}$ 或更低；当 $L_g=12$ 时仍低于 1 但收敛显著变慢；当 $L_g=15$ 时最终 $\rho(J)>1$，残差停留在约 $1.7\times10^{{-1}}$，没有进入固定点收敛区间。

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig01_threeq_inference_convergence.png}}
\caption{{legacy ThreeQ 非线性推断实验。$\rho(J)<1$ 的曲线呈残差衰减；$L_g=15$ 后期 $\rho(J)>1$，表现为非收敛/慢收敛边界。}}
\label{{fig:legacy-inference}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{legacy 推断边界实验摘要。}}
\label{{tab:legacy-boundary}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{rrrrrl}}
\toprule
$L_g$ & initial $\Delta$ & final $\Delta$ & final $\rho$ & reduction & classification \\
\midrule
{legacy_rows}
\bottomrule
\end{{tabular}}}}
\end{{table}}

需要注意，非线性 ThreeQ 中的 $\rho>1$ 不一定表现为无界爆炸。激活函数、截断和能量形状可能把轨迹限制在某个区域内，因此实际现象常常是振荡、停滞或慢收敛。为了严格展示线性化意义上的发散，需要实验 B。

\subsection{{受控线性推断：$\rho>1$ 的指数发散}}

图~\ref{{fig:controlled-inference}} 使用 $T(u)=\rho u$ 的 ThreeQ 母模型特例。该实验中 $J=\rho I$，因此谱半径可以精确控制。$\rho=0.80,0.95$ 时扰动范数下降；$\rho=1.05,1.20$ 时扰动范数按 $\rho^t$ 指数增长。这直接补足了 $\rho>1$ 的发散佐证。

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig09_controlled_inference_rho_divergence.png}}
\caption{{受控线性 ThreeQ 推断实验。在同一固定点更新形式 $u_{{t+1}}=T(u_t;W)$ 下，$\rho<1$ 收敛，$\rho>1$ 发散。}}
\label{{fig:controlled-inference}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{受控线性推断实验摘要。初始扰动范数为 $10^{{-3}}$。}}
\label{{tab:controlled-inference}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrr}}
\toprule
case & $\rho$ & initial norm & final norm & growth factor & final residual \\
\midrule
{controlled_rows}
\bottomrule
\end{{tabular}}}}
\end{{table}}

\subsection{{训练更新：小步保持稳定，大步跨越边界后发散}}

图~\ref{{fig:training-stress}} 展示 bounded-gradient 更新的边界行为。small-step 轨迹从 $\rho=0.70$ 增加到 $0.94$，始终满足 $\rho<1$，每轮更新后的推断仍然收敛。large-step 轨迹在第 9 个 epoch 首次达到 $\rho\ge 1$，最终 $\rho=1.54$；此时最后一轮推断的扰动范数显著增长。

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig10_training_rho_boundary_stress_test.png}}
\caption{{训练更新压力测试。左：训练更新使 $\rho$ 变化；中：每轮更新后推断 40 步的最终扰动范数；右：最后一轮参数下的推断轨迹。}}
\label{{fig:training-stress}}
\end{{figure}}

\begin{{table}}[H]
\centering
\caption{{训练更新压力测试摘要。}}
\label{{tab:training-stress}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrr}}
\toprule
case & $\eta$ & initial $\rho$ & final $\rho$ & first $\rho\ge1$ epoch & growth factor & final norm \\
\midrule
{training_rows}
\bottomrule
\end{{tabular}}}}
\end{{table}}

该实验对应理论证明中的“小学习率保持稳定裕度”条件：当更新步长足够小，参数仍留在 $\rho<1$ 的开集内；当步长过大，参数可以越过稳定边界，后续推断不再收敛。

\subsection{{实际 MNIST 训练中的稳定区间}}

图~\ref{{fig:legacy-training}} 来自已有 MNIST legacy 训练实验。\texttt{{EPBase3Q}} 在 5k/1k、8 epoch 中明显快于 direct \texttt{{Base3Q}}；在 10k/2k、15 epoch 调参中，最佳 EPThreeQ 约达到 86.5\% best validation accuracy。右图显示训练中的 \texttt{{rho\_mean}} 保持在 1 以下，因此实际训练曲线与稳定性命题一致。

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{fig02_threeq_training_convergence.png}}
\caption{{legacy ThreeQ/EPThreeQ MNIST 训练曲线与训练过程中的 $\rho$。}}
\label{{fig:legacy-training}}
\end{{figure}}

\section{{论文可用结论}}

上述实验支持以下写法：

\begin{{enumerate}}
\item $\rho(J)<1$ 是 ThreeQ inference 局部收敛的可观测充分条件。legacy 非线性实验中，后期 $\rho<1$ 的配置残差下降；受控线性实验中，$\rho<1$ 产生指数收敛。
\item $\rho(J)>1$ 是局部线性化意义下的不稳定证据。受控线性 ThreeQ 母模型中 $\rho>1$ 直接导致扰动指数发散；legacy 非线性实验中则表现为非收敛或慢收敛边界。
\item 训练收敛性依赖小步长和稳定裕度。bounded-gradient 压力测试表明，小步更新保留 $\rho<1$，大步更新跨越 $\rho=1$ 后 post-update inference 发散。
\item 实际 MNIST legacy 训练中，EPThreeQ 的 \texttt{{rho\_mean}} 保持低于 1，且 accuracy 随训练提升，说明该训练配置处在稳定区间内。
\end{{enumerate}}

\section{{局限性}}

本文新增的 $\rho>1$ 发散实验是受控母模型和线性化压力测试，目的是隔离谱半径条件本身。实际非线性 ThreeQ 网络中，激活截断、状态边界、能量饱和和数值阻尼可能把无界发散转化为振荡、停滞或退化固定点。因此论文中应区分“局部线性化发散”和“完整非线性系统的训练失败模式”。

\end{{document}}
"""

    tex_path = LATEX / "threeq_convergence_report.tex"
    tex_path.write_text(text, encoding="utf-8")
    return tex_path


def main() -> None:
    ensure_dirs()
    legacy_summary = read_existing_legacy_summary()
    controlled_summary = build_controlled_inference()
    training_summary = build_training_stress_test()
    tex_path = write_latex_report(controlled_summary, training_summary, legacy_summary)
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
