import time
import math
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ThreeQ_Conv import Network


def setup_logger(run_name: str) -> logging.Logger:
    """
    创建并返回一个 logger，同时写入控制台与日志文件。
    日志文件默认保存到 ./logs 目录。
    """
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    # 日志目录与文件名
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
# MNIST 数据集 / DataLoader
# -----------------------------
def build_mnist_loaders(
    batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """构建 MNIST 训练/验证 DataLoader。"""
    tfm = transforms.Compose([transforms.ToTensor()])

    # 下载/加载完整训练集 (60000)
    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)

    # 简单划分：前 50000 做训练，后 10000 做验证
    train_idx = list(range(0, 50000))
    valid_idx = list(range(50000, 60000))
    ds_train = Subset(full_train, train_idx)
    ds_valid = Subset(full_train, valid_idx)

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    loader_valid = DataLoader(
        ds_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader_train, loader_valid


# -----------------------------
# 验证阶段（仅自由相）
# -----------------------------
def eval_free_phase(
    net: Network, x: torch.Tensor, y: torch.Tensor
) -> Tuple[float, float, float]:
    """
    仅使用自由相评估 (E_mean, C_mean, err)。
    注意：relax_states 内部需要 enable_grad，因此这里不包 @torch.no_grad。
    """

    # 切到评估模式（禁用 Dropout/BN 等）
    net.eval()

    B = x.shape[0]
    x = x.to(net.device, non_blocking=True).view(B, -1).float()
    y = y.to(net.device, non_blocking=True).long()
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=net.num_classes).float()

    # 初始隐藏状态全 0
    states0 = [torch.zeros(B, dim, device=net.device) for dim in net.layer_sizes[1:]]

    # 自由相松弛
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


# -----------------------------
# 保存模型与曲线
# -----------------------------
def save_checkpoint(
    path: str,
    net: Network,
    hyper: Dict[str, Any],
    curves: Dict[str, List[float]],
    epoch: int,
):
    """保存模型权重、超参数与训练曲线。"""
    payload = {
        "epoch": epoch,
        "model_state": net.state_dict(),
        "hyperparameters": hyper,
        "training_curves": curves,
    }
    torch.save(payload, path)


# -----------------------------
# 训练流程
# -----------------------------
def train_net(name: str, hyper: Dict[str, Any], device: torch.device):
    """
    训练 EP 网络的主流程。

    关键超参数说明：
    - n_it_neg: 自由相松弛迭代步数
    - n_it_pos: 受扰相松弛迭代步数
    - epsilon: 状态更新步长
    - beta: 受扰强度（当前实现固定为正值）
    - alphas: 每条连接的学习率（卷积层 + FC）
    """
    conv_channels = hyper["conv_channels"]
    kernel_sizes = hyper["kernel_sizes"]
    strides_list = hyper["strides"]
    paddings_list = hyper["paddings"]
    n_epochs = hyper["n_epochs"]
    batch_size = hyper["batch_size"]
    n_it_neg = hyper["n_it_neg"]
    n_it_pos = hyper["n_it_pos"]
    epsilon = float(hyper["epsilon"])
    beta_abs = float(hyper["beta"])
    alphas = [float(a) for a in hyper["alphas"]]
    input_size = hyper.get("input_size", (28, 28))
    num_classes = hyper.get("num_classes", 10)

    logger = setup_logger(name)

    # 连接数量：卷积层数量 + 1 个 FC
    n_connections = len(alphas)

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

    # 构建 DataLoader
    loader_train, loader_valid = build_mnist_loaders(
        batch_size=batch_size, num_workers=2
    )

    # 训练曲线记录
    curves: Dict[str, List[float]] = {
        "training error": [],
        "validation error": [],
        "training loss": [],
        "training energy": [],
        "training cost": [],
    }
    for k in range(n_connections):
        curves[f"dlogW{k}"] = []

    # ---------- 打印网络结构与超参数 ----------
    arch_ch = "->".join(str(c) for c in conv_channels)
    logger.info(f"name = {name}")
    logger.info(f"conv_channels = {arch_ch} -> FC -> {num_classes}")
    logger.info(f"layer_sizes (flat) = {net.layer_sizes}")
    logger.info(f"spatial_sizes = {net.spatial_sizes}")
    logger.info(f"kernel_sizes = {kernel_sizes}")
    logger.info(f"strides = {strides_list}")
    logger.info(f"paddings = {paddings_list}")
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

    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        # =================== 训练 ===================
        net.train()

        measures_sum = [0.0, 0.0, 0.0, 0.0]  # E(自由), C(自由), err(自由), loss(受扰)
        gW = [0.0] * n_connections

        for step, (x, y) in enumerate(loader_train):
            net.beta = float(beta_abs)

            E_mean, C_mean, err, loss, delta_logs = net.train_batch(x, y)

            measures_sum[0] += E_mean
            measures_sum[1] += C_mean
            measures_sum[2] += err
            measures_sum[3] += loss
            for i in range(n_connections):
                gW[i] += float(delta_logs[i])

            k = step + 1
            measures_avg = [m / k for m in measures_sum]
            measures_avg[2] *= 100.0
            seen = k * batch_size
            # 训练进度实时输出
            print(
                f"\r{epoch:2d}-train-{seen:5d} "
                f"E={measures_avg[0]:.1f} C={measures_avg[1]:.5f} "
                f"error={measures_avg[2]:.3f}% loss={measures_avg[3]:.3f}",
                end="",
                flush=True,
            )

        print()

        n_batches_train = len(loader_train)
        train_E = measures_sum[0] / max(1, n_batches_train)
        train_C = measures_sum[1] / max(1, n_batches_train)
        train_err = measures_sum[2] / max(1, n_batches_train) * 100.0
        train_loss = measures_sum[3] / max(1, n_batches_train)
        curves["training error"].append(train_err)
        curves["training loss"].append(train_loss)
        curves["training energy"].append(train_E)
        curves["training cost"].append(train_C)

        logger.info(
            f"epoch {epoch:03d} train: E={train_E:.3f} C={train_C:.5f} "
            f"error={train_err:.3f}% loss={train_loss:.4f}"
        )

        # dlogW: 各连接权重的相对更新幅度
        dlogW = [100.0 * g / n_batches_train for g in gW]
        logger.info(
            "   " + " ".join([f"dlogW{k}={v:.3f}%" for k, v in enumerate(dlogW)])
        )
        for k, v in enumerate(dlogW):
            curves[f"dlogW{k}"].append(v)

        # =================== 验证 ===================
        measures_sum = [0.0, 0.0, 0.0]

        for step, (x, y) in enumerate(loader_valid):
            E_mean, C_mean, err = eval_free_phase(net, x, y)

            measures_sum[0] += E_mean
            measures_sum[1] += C_mean
            measures_sum[2] += err

            k = step + 1
            measures_avg = [m / k for m in measures_sum]
            measures_avg[-1] *= 100.0
            seen = k * batch_size
            # 验证进度实时输出
            print(
                f"\r   valid-{seen:5d} "
                f"E={measures_avg[0]:.1f} C={measures_avg[1]:.5f} "
                f"error={measures_avg[2]:.2f}%",
                end="",
                flush=True,
            )

        print()
        n_batches_valid = len(loader_valid)
        valid_E = measures_sum[0] / max(1, n_batches_valid)
        valid_C = measures_sum[1] / max(1, n_batches_valid)
        valid_err = measures_sum[2] / max(1, n_batches_valid) * 100.0
        curves["validation error"].append(valid_err)
        logger.info(
            f"epoch {epoch:03d} valid: E={valid_E:.3f} C={valid_C:.5f} "
            f"error={valid_err:.3f}%"
        )

        duration_min = (time.perf_counter() - start_time) / 60.0
        logger.info(f"   duration={duration_min:.1f} min")

        # 每个 epoch 结束后保存 checkpoint
        ckpt_path = f"{name}.pt"
        save_checkpoint(ckpt_path, net, hyper, curves, epoch)

    return net, curves


# -------------------------------------------------------
# 预设网络配置
# -------------------------------------------------------

# 1 层卷积 + FC
net1 = (
    "net1_conv",
    {
        "conv_channels": [1, 32],
        "kernel_sizes": [5],
        "strides": [2],
        "paddings": [2],
        # 输入(1,28,28) -> (32,14,14) | FC: 6272 -> 10
        "n_epochs": 25,
        "batch_size": 20,
        "n_it_neg": 50,
        "n_it_pos": 50,
        "epsilon": np.float32(0.05),
        "beta": np.float32(0.5),
        "alphas": [np.float32(0.01), np.float32(0.005)],
        # alphas[0] -> conv，alphas[1] -> FC
    },
)

# 2 层卷积 + FC
net2 = (
    "net2_conv",
    {
        "conv_channels": [1, 32, 64],
        "kernel_sizes": [5, 5],
        "strides": [2, 2],
        "paddings": [2, 2],
        # 输入(1,28,28) -> (32,14,14) -> (64,7,7) | FC: 3136 -> 10
        "n_epochs": 60,
        "batch_size": 20,
        "n_it_neg": 150,
        "n_it_pos": 6,
        "epsilon": np.float32(0.5),
        "beta": np.float32(1.0),
        "alphas": [np.float32(0.4), np.float32(0.1), np.float32(0.01)],
    },
)

# 3 层卷积 + FC
net3 = (
    "net3_conv",
    {
        "conv_channels": [1, 16, 32, 64],
        "kernel_sizes": [5, 3, 3],
        "strides": [2, 2, 1],
        "paddings": [2, 1, 1],
        # 输入(1,28,28) -> (16,14,14) -> (32,7,7) -> (64,7,7) | FC: 3136 -> 10
        "n_epochs": 500,
        "batch_size": 20,
        "n_it_neg": 500,
        "n_it_pos": 8,
        "epsilon": np.float32(0.5),
        "beta": np.float32(1.0),
        "alphas": [
            np.float32(0.128),
            np.float32(0.032),
            np.float32(0.008),
            np.float32(0.002),
        ],
    },
)


if __name__ == "__main__":
    # 默认训练 net1，需要可切换为 net2 / net3
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_net(*net1, device=device)
