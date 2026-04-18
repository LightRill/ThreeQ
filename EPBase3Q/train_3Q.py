import time
import math
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ThreeQ import Network


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
# 工具：构建 MNIST 的 train/valid loader
# 按论文脚本：60000 中前 50000 训练，后 10000 验证
# -----------------------------
def build_mnist_loaders(
    batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),  # -> [0,1], shape=(1,28,28)
            # 论文原实现本质上是把输入 clamp 在 [0,1]；ToTensor 已满足
        ]
    )

    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)

    train_idx = list(range(0, 50000))
    valid_idx = list(range(50000, 60000))
    ds_train = Subset(full_train, train_idx)
    ds_valid = Subset(full_train, valid_idx)

    loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,  # 没有持久粒子，shuffle 完全没问题
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
# 工具：只做 free phase 的评估（不更新参数）
# 对齐论文 validation：free_phase -> measure
# -----------------------------
@torch.no_grad()
def _zero_states(
    batch_size: int, layer_sizes: List[int], device: torch.device
) -> List[torch.Tensor]:
    # layer_sizes: [784, h1, ..., 10]，状态只需要 hidden+output
    return [
        torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
        for dim in layer_sizes[1:]
    ]


def eval_free_phase(
    net: Network, x: torch.Tensor, y: torch.Tensor
) -> Tuple[float, float, float]:
    """
    返回 (E_mean, C_mean, err)，对应论文 measure() 的三个量
    注意：这里不做 weak phase，也不做参数更新。
    """
    net.eval()

    B = x.shape[0]
    x = x.to(net.device, non_blocking=True).view(B, -1).float()
    y = y.to(net.device, non_blocking=True).long()
    y_one_hot = torch.nn.functional.one_hot(y, num_classes=net.layer_sizes[-1]).float()

    # 从零初始化状态（不含持久粒子）
    states0 = [torch.zeros(B, dim, device=net.device) for dim in net.layer_sizes[1:]]

    # free phase 松弛
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
# 保存 checkpoint（替代 Theano 的 .save）
# -----------------------------
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
# 训练主函数：对齐论文 train.py 的输出风格
# -----------------------------
def train_net(name: str, hyper: Dict[str, Any], device: torch.device):
    hidden_sizes = hyper["hidden_sizes"]
    n_epochs = hyper["n_epochs"]
    batch_size = hyper["batch_size"]
    n_it_neg = hyper["n_it_neg"]
    n_it_pos = hyper["n_it_pos"]
    epsilon = float(hyper["epsilon"])
    beta_abs = float(hyper["beta"])
    alphas = [float(a) for a in hyper["alphas"]]

    # 架构：784-...-10
    layer_sizes = [28 * 28] + list(hidden_sizes) + [10]
    logger = setup_logger(name)
    if len(alphas) != len(layer_sizes) - 1:
        raise ValueError(
            f"alphas length ({len(alphas)}) must be N-1 ({len(layer_sizes) - 1})"
        )

    # 构建网络（当前实现使用固定正 beta）
    net = Network(
        layer_sizes=layer_sizes,
        alphas=alphas,
        beta=beta_abs,
        free_steps=n_it_neg,
        weak_steps=n_it_pos,
        epsilon=epsilon,
        device=device,
    ).to(device)

    # DataLoader
    loader_train, loader_valid = build_mnist_loaders(
        batch_size=batch_size, num_workers=2
    )

    # 训练曲线
    curves = {
        "training error": [],
        "validation error": [],
        "training loss": [],
        "training energy": [],
        "training cost": [],
        "training rho mean": [],
        "training rho max": [],
    }
    for k in range(len(alphas)):
        curves[f"forward_dlogW{k+1}"] = []
    for k in range(len(alphas)):
        curves[f"backward_dlogW{k+1}"] = []

    # 打印超参信息
    arch = "784-" + "-".join([str(n) for n in hidden_sizes]) + "-10"
    logger.info(f"name = {name}")
    logger.info(f"architecture = {arch}")
    logger.info(f"number of epochs = {n_epochs}")
    logger.info(f"batch_size = {batch_size}")
    logger.info(f"n_it_neg = {n_it_neg}")
    logger.info(f"n_it_pos = {n_it_pos}")
    logger.info(f"epsilon = {epsilon:.3f}")
    logger.info(f"beta = {beta_abs:.3f}")
    logger.info(
        "learning rates: "
        + " ".join([f"alpha_W{k+1}={a:.3f}" for k, a in enumerate(alphas)])
    )

    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        # ----------------- TRAINING -----------------
        net.train()

        measures_sum = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # E, C, error 的累计和
        forward_gW = [0.0 for _ in alphas]  # 每层 dlogW 累计（来自 delta_logs）
        backward_gW = [0.0 for _ in alphas]  # 每层 dlogW 累计（来自 delta_logs）

        for step, (x, y) in enumerate(loader_train):
            net.beta = float(beta_abs)

            # 训练一个 batch（内部：free -> weak -> update）
            (
                E_mean,
                C_mean,
                err,
                loss,
                rho_mean,
                rho_max,
                forward_delta_logs,
                backward_delta_logs,
            ) = net.train_batch(x, y)

            # 累计
            measures_sum[0] += E_mean
            measures_sum[1] += C_mean
            measures_sum[2] += err
            measures_sum[3] += loss
            measures_sum[4] += rho_mean
            measures_sum[5] += rho_max
            for i in range(len(forward_gW)):
                forward_gW[i] += float(forward_delta_logs[i])
            for i in range(len(backward_gW)):
                backward_gW[i] += float(backward_delta_logs[i])

            # 平均并打印（对齐原脚本风格）
            k = step + 1
            measures_avg = [m / k for m in measures_sum]
            measures_avg[2] *= 100.0  # error -> percent
            seen = k * batch_size
            print(
                f"\r{epoch:2d}-train-{seen:5d} "
                f"E={measures_avg[0]:.1f} C={measures_avg[1]:.5f} "
                f"error={measures_avg[2]:.3f}% loss={measures_avg[3]:.3f} "
                f"rho={measures_avg[4]:.3f}/{measures_avg[5]:.3f}",
                end="",
                flush=True,
            )

        print()  # 换行

        # 训练集 epoch 统计
        n_batches_train = len(loader_train)
        train_E = measures_sum[0] / max(1, n_batches_train)
        train_C = measures_sum[1] / max(1, n_batches_train)
        train_err = measures_sum[2] / max(1, n_batches_train) * 100.0
        train_loss = measures_sum[3] / max(1, n_batches_train)
        train_rho_mean = measures_sum[4] / max(1, n_batches_train)
        train_rho_max = measures_sum[5] / max(1, n_batches_train)
        curves["training error"].append(train_err)
        curves["training loss"].append(train_loss)
        curves["training energy"].append(train_E)
        curves["training cost"].append(train_C)
        curves["training rho mean"].append(train_rho_mean)
        curves["training rho max"].append(train_rho_max)

        logger.info(
            f"epoch {epoch:03d} train: E={train_E:.3f} C={train_C:.5f} "
            f"error={train_err:.3f}% loss={train_loss:.4f} "
            f"rho={train_rho_mean:.4f}/{train_rho_max:.4f}"
        )

        # epoch 级 dlogW（对齐原脚本：乘 100%）
        dlogW = [100.0 * gw / n_batches_train for gw in forward_gW]
        logger.info(
            "   "
            + " ".join([f"forward_dlogW{k+1}={v:.3f}%" for k, v in enumerate(dlogW)])
        )
        for k, v in enumerate(dlogW):
            curves[f"forward_dlogW{k+1}"].append(v)

        dlogW = [100.0 * gw / n_batches_train for gw in backward_gW]
        logger.info(
            "   "
            + " ".join([f"backward_dlogW{k+1}={v:.3f}%" for k, v in enumerate(dlogW)])
        )
        for k, v in enumerate(dlogW):
            curves[f"backward_dlogW{k+1}"].append(v)

        # ----------------- VALIDATION -----------------
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
            print(
                f"\r   valid-{seen:5d} "
                f"E={measures_avg[0]:.1f} C={measures_avg[1]:.5f} error={measures_avg[2]:.2f}%",
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

        # 保存 checkpoint（每个 epoch 一次）
        ckpt_path = f"{name}.pt"
        save_checkpoint(ckpt_path, net, hyper, curves, epoch)

    return net, curves


# -----------------------------
# 对齐论文给的三组超参
# -----------------------------
net1 = (
    "net1",
    {
        "hidden_sizes": [500],
        "n_epochs": 25,
        "batch_size": 20,
        "n_it_neg": 50,
        "n_it_pos": 2,
        "epsilon": np.float32(0.1),
        "beta": np.float32(0.5),
        "alphas": [np.float32(0.05), np.float32(0.01)],
    },
)

net2 = (
    "net2",
    {
        "hidden_sizes": [500, 500],
        "n_epochs": 60,
        "batch_size": 20,
        "n_it_neg": 150,
        "n_it_pos": 6,
        "epsilon": np.float32(0.5),
        "beta": np.float32(1.0),
        "alphas": [np.float32(0.4), np.float32(0.1), np.float32(0.01)],
    },
)

net3 = (
    "net3",
    {
        "hidden_sizes": [500, 500, 500],
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 训练 1 隐藏层网络（对齐论文 main）
    train_net(*net1, device=device)
