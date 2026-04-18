import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ThreeQ_Conv import Network


def setup_logger(run_name: str) -> logging.Logger:
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{ts}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info(f"log_path = {log_path}")
    return logger


# -----------------------------
# 评估：只做 free phase（不更新参数）
# 注意：不使用 @torch.no_grad() 装饰器，
# 因为 relax_states 内部需要梯度来做状态松弛
# -----------------------------
def eval_free_phase(
    net: Network, x: torch.Tensor, y: torch.Tensor
) -> Tuple[float, float, float]:
    net.eval()

    B = x.shape[0]
    x = x.to(net.device, non_blocking=True).view(B, -1).float()
    y = y.to(net.device, non_blocking=True).long()
    y_one_hot = F.one_hot(y, num_classes=net.num_classes).float()

    states0 = [torch.zeros(B, dim, device=net.device) for dim in net.layer_sizes[1:]]

    states_free = net.relax_states(
        x=x,
        states_init=states0,
        y_one_hot=None,
        beta=0.0,
        n_steps=net.free_steps,
        epsilon=net.epsilon,
    )

    with torch.no_grad():
        layers_free = [x] + states_free
        E_mean = net.energy(layers_free).mean().item()
        C_mean = net.cost(layers_free, y_one_hot).mean().item()
        y_pred = torch.argmax(states_free[-1], dim=1)
        err = (y_pred != y).float().mean().item()

    return E_mean, C_mean, err


@torch.no_grad()
def inference_grid(net: Network, grid_tensor: torch.Tensor) -> torch.Tensor:
    """对网格点做 free phase 推理，返回输出层状态。"""
    net.eval()
    B = grid_tensor.shape[0]
    x = grid_tensor.to(net.device).view(B, -1).float()

    states0 = [torch.zeros(B, dim, device=net.device) for dim in net.layer_sizes[1:]]
    # relax_states 内部已使用 torch.enable_grad()，可安全在 no_grad 下调用
    states_free = net.relax_states(
        x=x,
        states_init=states0,
        y_one_hot=None,
        beta=0.0,
        n_steps=net.free_steps,
        epsilon=net.epsilon,
    )
    return states_free[-1]  # shape (B, num_classes)


def save_checkpoint(
    path: str,
    net: Network,
    hyper: Dict[str, Any],
    curves: Dict[str, List[float]],
    epoch: int,
):
    payload = {
        "epoch": epoch,
        "model_state": net.state_dict(),
        "hyperparameters": hyper,
        "training_curves": curves,
    }
    torch.save(payload, path)


# -----------------------------
# 训练主函数
# -----------------------------
def train_twomoons(name: str, hyper: Dict[str, Any], device: torch.device):
    conv_channels = hyper["conv_channels"]
    kernel_sizes = hyper["kernel_sizes"]
    strides_list = hyper["strides"]
    paddings_list = hyper["paddings"]
    input_size = hyper.get("input_size", (1, 1))
    num_classes = hyper.get("num_classes", 2)
    n_epochs = hyper["n_epochs"]
    batch_size = hyper["batch_size"]
    n_it_neg = hyper["n_it_neg"]
    n_it_pos = hyper["n_it_pos"]
    epsilon = float(hyper["epsilon"])
    beta_abs = float(hyper["beta"])
    alphas = [float(a) for a in hyper["alphas"]]
    n_samples = hyper.get("n_samples", 1000)
    noise = hyper.get("noise", 0.1)

    logger = setup_logger(name)
    n_connections = len(alphas)

    # ---------- 数据集 ----------
    X_all, y_all = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # 归一化到 [0, 1]（rho 激活函数将状态 clamp 到 [0,1]）
    x_min = X_all.min(axis=0)
    x_max = X_all.max(axis=0)
    X_norm = (X_all - x_min) / (x_max - x_min + 1e-8)

    # 80/20 划分
    n_train = int(0.8 * n_samples)
    X_train, y_train = X_norm[:n_train], y_all[:n_train]
    X_valid, y_valid = X_norm[n_train:], y_all[n_train:]

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    valid_ds = TensorDataset(
        torch.tensor(X_valid, dtype=torch.float32),
        torch.tensor(y_valid, dtype=torch.long),
    )
    loader_train = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    loader_valid = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ---------- 构建网络 ----------
    net = Network(
        alphas=alphas,
        beta=beta_abs,
        free_steps=n_it_neg,
        weak_steps=n_it_pos,
        epsilon=epsilon,
        conv_channels=conv_channels,
        kernel_sizes=kernel_sizes,
        strides=strides_list,
        paddings=paddings_list,
        input_size=input_size,
        num_classes=num_classes,
        device=device,
    ).to(device)

    # ---------- 训练曲线 ----------
    curves: Dict[str, List[float]] = {
        "training error": [],
        "validation error": [],
        "training loss": [],
        "training energy": [],
        "training cost": [],
    }
    for k in range(n_connections):
        curves[f"dlogW{k}"] = []

    # ---------- 打印超参信息 ----------
    arch_ch = "->".join(str(c) for c in conv_channels)
    logger.info(f"name = {name}")
    logger.info(f"conv_channels = {arch_ch} -> FC -> {num_classes}")
    logger.info(f"layer_sizes (flat) = {net.layer_sizes}")
    logger.info(f"spatial_sizes = {net.spatial_sizes}")
    logger.info(f"kernel_sizes = {kernel_sizes}")
    logger.info(f"strides = {strides_list}")
    logger.info(f"paddings = {paddings_list}")
    logger.info(f"input_size = {input_size}")
    logger.info(f"n_epochs = {n_epochs}")
    logger.info(f"batch_size = {batch_size}")
    logger.info(f"n_it_neg = {n_it_neg}")
    logger.info(f"n_it_pos = {n_it_pos}")
    logger.info(f"epsilon = {epsilon:.3f}")
    logger.info(f"beta = {beta_abs:.3f}")
    logger.info(
        "learning rates: "
        + " ".join([f"alpha_{k}={a:.4f}" for k, a in enumerate(alphas)])
    )
    n_params = sum(p.numel() for p in net.parameters())
    logger.info(f"total parameters = {n_params}")
    logger.info(f"dataset: two moons, n_samples={n_samples}, noise={noise}")

    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        # =================== 训练 ===================
        net.train()

        measures_sum = [0.0, 0.0, 0.0, 0.0]  # E, C, err, loss
        gW = [0.0] * n_connections

        for step, (xb, yb) in enumerate(loader_train):
            net.beta = float(beta_abs)

            E_mean, C_mean, err, loss, delta_logs = net.train_batch(xb, yb)

            measures_sum[0] += E_mean
            measures_sum[1] += C_mean
            measures_sum[2] += err
            measures_sum[3] += loss
            for i in range(n_connections):
                gW[i] += float(delta_logs[i])

            k = step + 1
            avg = [m / k for m in measures_sum]
            avg[2] *= 100.0
            seen = k * batch_size
            print(
                f"\r{epoch:3d}-train-{seen:5d} "
                f"E={avg[0]:.3f} C={avg[1]:.5f} "
                f"error={avg[2]:.2f}% loss={avg[3]:.4f}",
                end="",
                flush=True,
            )

        print()

        # epoch 级训练统计
        n_bt = len(loader_train)
        train_E = measures_sum[0] / max(1, n_bt)
        train_C = measures_sum[1] / max(1, n_bt)
        train_err = measures_sum[2] / max(1, n_bt) * 100.0
        train_loss = measures_sum[3] / max(1, n_bt)
        curves["training error"].append(train_err)
        curves["training loss"].append(train_loss)
        curves["training energy"].append(train_E)
        curves["training cost"].append(train_C)

        logger.info(
            f"epoch {epoch:03d} train: E={train_E:.4f} C={train_C:.5f} "
            f"error={train_err:.2f}% loss={train_loss:.4f}"
        )

        dlogW = [100.0 * g / n_bt for g in gW]
        logger.info(
            "   " + " ".join([f"dlogW{k}={v:.3f}%" for k, v in enumerate(dlogW)])
        )
        for k, v in enumerate(dlogW):
            curves[f"dlogW{k}"].append(v)

        # =================== 验证 ===================
        v_sum = [0.0, 0.0, 0.0]

        for step, (xb, yb) in enumerate(loader_valid):
            e, c, er = eval_free_phase(net, xb, yb)

            v_sum[0] += e
            v_sum[1] += c
            v_sum[2] += er

            k = step + 1
            va = [m / k for m in v_sum]
            va[-1] *= 100.0
            seen = k * batch_size
            print(
                f"\r   valid-{seen:5d} "
                f"E={va[0]:.3f} C={va[1]:.5f} error={va[2]:.2f}%",
                end="",
                flush=True,
            )

        print()
        n_bv = len(loader_valid)
        valid_E = v_sum[0] / max(1, n_bv)
        valid_C = v_sum[1] / max(1, n_bv)
        valid_err = v_sum[2] / max(1, n_bv) * 100.0
        curves["validation error"].append(valid_err)
        logger.info(
            f"epoch {epoch:03d} valid: E={valid_E:.4f} C={valid_C:.5f} "
            f"error={valid_err:.2f}%"
        )

        dur = (time.perf_counter() - start_time) / 60.0
        logger.info(f"   duration={dur:.1f} min")

        save_checkpoint(f"{name}.pt", net, hyper, curves, epoch)

    # ---------- 可视化：决策边界 ----------
    logger.info("Generating decision boundary visualization...")

    margin = 0.5
    res = 200
    x0_lo, x0_hi = X_all[:, 0].min() - margin, X_all[:, 0].max() + margin
    x1_lo, x1_hi = X_all[:, 1].min() - margin, X_all[:, 1].max() + margin
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_lo, x0_hi, res),
        np.linspace(x1_lo, x1_hi, res),
    )
    grid_raw = np.column_stack([xx0.ravel(), xx1.ravel()])
    # 使用与训练数据相同的归一化参数
    grid_norm = (grid_raw - x_min) / (x_max - x_min + 1e-8)
    grid_t = torch.tensor(grid_norm, dtype=torch.float32)

    # 分批推理
    chunk = 4096
    outputs = []
    for i in range(0, grid_t.shape[0], chunk):
        out = inference_grid(net, grid_t[i : i + chunk])
        outputs.append(out.cpu())
    output = torch.cat(outputs, dim=0)

    # 决策值：output[:,1] - output[:,0]
    decision = (output[:, 1] - output[:, 0]).detach().numpy().reshape(xx0.shape)

    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.contourf(xx0, xx1, decision, levels=50, cmap="RdBu", alpha=0.85)
    ax.contour(xx0, xx1, decision, levels=[0.0], colors="k", linewidths=2)
    plt.colorbar(cf, ax=ax, label="Output: P(class 1) - P(class 0)")

    colors = ["#d73027", "#4575b4"]
    for cls in [0, 1]:
        mask = y_all == cls
        ax.scatter(
            X_all[mask, 0],
            X_all[mask, 1],
            c=colors[cls],
            edgecolors="k",
            s=25,
            linewidth=0.5,
            label=f"Class {cls}",
            zorder=5,
        )

    arch_str = f"Conv {arch_ch} -> FC -> {num_classes}"
    ax.legend(loc="upper right")
    ax.set_xlabel("$x_1$", fontsize=13)
    ax.set_ylabel("$x_2$", fontsize=13)
    ax.set_title(
        f"CNN 3Q Network — Two Moons Decision Boundary\n"
        f"arch={arch_str}  epochs={n_epochs}  "
        f"valid error={curves['validation error'][-1]:.1f}%",
        fontsize=13,
    )
    fig.tight_layout()

    out_path = "twomoons_decision_boundary.png"
    fig.savefig(out_path, dpi=150)
    logger.info(f"Saved: {out_path}")
    plt.close(fig)

    return net, curves, out_path


# -----------------------------
# 超参配置
# -----------------------------
# 输入 2D 点作为 (2 channels, 1x1 spatial)，kernel_size=1 使卷积等价于逐点线性变换
twomoons_config = (
    "twomoons_conv",
    {
        "conv_channels": [2, 32],
        "kernel_sizes": [1],
        "strides": [1],
        "paddings": [0],
        "input_size": (1, 1),
        "num_classes": 2,
        "n_epochs": 5,
        "batch_size": 20,
        "n_it_neg": 50,
        "n_it_pos": 50,
        "epsilon": np.float32(0.1),
        "beta": np.float32(0.5),
        "alphas": [np.float32(0.02), np.float32(0.01)],
        "n_samples": 1000,
        "noise": 0.1,
    },
)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net, curves, img_path = train_twomoons(*twomoons_config, device=device)
    print(f"\nDone. Visualization saved to {img_path}")
