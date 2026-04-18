from __future__ import annotations

import random
import time
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import BidirectionalMLP


VARIANTS: Dict[str, Dict[str, Any]] = {
    "direct_cost_w10": {
        "weight_update": "direct",
        "weak_mode": "cost",
        "weak_steps": 10,
        "transpose_tied": False,
    },
    "direct_cost_w50": {
        "weight_update": "direct",
        "weak_mode": "cost",
        "weak_steps": 50,
        "transpose_tied": False,
    },
    "direct_linear_w10": {
        "weight_update": "direct",
        "weak_mode": "linear_clamp",
        "weak_steps": 10,
        "transpose_tied": False,
    },
    "ep_cost_w2": {
        "weight_update": "ep",
        "weak_mode": "cost",
        "weak_steps": 2,
        "transpose_tied": False,
    },
    "ep_cost_w10": {
        "weight_update": "ep",
        "weak_mode": "cost",
        "weak_steps": 10,
        "transpose_tied": False,
    },
    "ep_cost_w50": {
        "weight_update": "ep",
        "weak_mode": "cost",
        "weak_steps": 50,
        "transpose_tied": False,
    },
    "ep_tied_w10": {
        "weight_update": "ep",
        "weak_mode": "cost",
        "weak_steps": 10,
        "transpose_tied": True,
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_twomoons(
    n_samples: int, noise: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    outer_theta = np.linspace(0.0, np.pi, n_outer)
    inner_theta = np.linspace(0.0, np.pi, n_inner)

    outer = np.stack([np.cos(outer_theta), np.sin(outer_theta)], axis=1)
    inner = np.stack([1.0 - np.cos(inner_theta), 1.0 - np.sin(inner_theta) - 0.5], axis=1)
    X = np.concatenate([outer, inner], axis=0)
    y = np.concatenate(
        [
            np.zeros(n_outer, dtype=np.int64),
            np.ones(n_inner, dtype=np.int64),
        ]
    )
    if noise > 0.0:
        X = X + rng.normal(scale=noise, size=X.shape)
    perm = rng.permutation(n_samples)
    return X[perm].astype(np.float32), y[perm]


def build_twomoons_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    n_samples = int(config.get("n_samples", 1000))
    noise = float(config.get("noise", 0.1))
    seed = int(config.get("seed", 0))
    batch_size = int(config.get("batch_size", 20))
    X, y = make_twomoons(n_samples=n_samples, noise=noise, random_state=42 + seed)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    n_train = int(0.8 * n_samples)
    train_ds = TensorDataset(
        torch.tensor(X[:n_train], dtype=torch.float32),
        torch.tensor(y[:n_train], dtype=torch.long),
    )
    valid_ds = TensorDataset(
        torch.tensor(X[n_train:], dtype=torch.float32),
        torch.tensor(y[n_train:], dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def mean_metrics(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(items)
    if not items:
        return {}
    keys = items[0].keys()
    return {key: float(sum(item[key] for item in items) / len(items)) for key in keys}


def train_one(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    config = dict(config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    variant_name = str(config["variant"])
    variant = dict(VARIANTS[variant_name])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_sizes = list(config.get("hidden_sizes", [32]))
    layer_sizes = [2] + hidden_sizes + [2]
    train_loader, valid_loader = build_twomoons_loaders(config)
    weak_steps = int(config.get("weak_steps", variant["weak_steps"]))
    model = BidirectionalMLP(
        layer_sizes=layer_sizes,
        alphas=list(config.get("alphas", [0.05, 0.01])),
        beta=float(config.get("beta", 0.5)),
        free_steps=int(config.get("free_steps", 50)),
        weak_steps=weak_steps,
        epsilon=float(config.get("epsilon", 0.05)),
        weight_update=variant["weight_update"],
        weak_mode=variant["weak_mode"],
        transpose_tied=bool(variant["transpose_tied"]),
        device=device,
    ).to(device)

    n_epochs = int(config.get("n_epochs", 20))
    best_valid_error = float("inf")
    best_epoch = -1
    last_train: Dict[str, float] = {}
    last_valid: Dict[str, float] = {}
    for epoch in range(n_epochs):
        train_epoch = []
        for x, y in train_loader:
            train_epoch.append(model.train_batch(x, y))
        valid_epoch = []
        for x, y in valid_loader:
            valid_epoch.append(model.eval_batch(x, y))
        last_train = mean_metrics(train_epoch)
        last_valid = mean_metrics(valid_epoch)
        if last_valid["valid_error"] < best_valid_error:
            best_valid_error = last_valid["valid_error"]
            best_epoch = epoch

    result: Dict[str, Any] = {
        "variant": variant_name,
        "seed": seed,
        "dataset": "twomoons",
        "n_epochs": n_epochs,
        "free_steps": int(config.get("free_steps", 50)),
        "weak_steps": weak_steps,
        "epsilon": float(config.get("epsilon", 0.05)),
        "beta": float(config.get("beta", 0.5)),
        "weight_update": variant["weight_update"],
        "weak_mode": variant["weak_mode"],
        "transpose_tied": bool(variant["transpose_tied"]),
        "best_epoch": int(best_epoch),
        "best_valid_error": float(best_valid_error),
        "final_valid_error": float(last_valid.get("valid_error", float("nan"))),
        "final_valid_cost": float(last_valid.get("valid_cost", float("nan"))),
        "final_valid_energy": float(last_valid.get("valid_energy", float("nan"))),
        "final_train_error": float(last_train.get("train_error", float("nan"))),
        "final_train_cost": float(last_train.get("train_cost", float("nan"))),
        "final_train_energy": float(last_train.get("train_energy", float("nan"))),
        "rho": float(last_train.get("rho", float("nan"))),
        "saturation": float(last_train.get("saturation", float("nan"))),
        "state_delta_l1": float(last_train.get("state_delta_l1", float("nan"))),
        "state_delta_l2": float(last_train.get("state_delta_l2", float("nan"))),
        "weight_abs_mean": float(last_train.get("weight_abs_mean", float("nan"))),
        "weight_update_rel_mean": float(
            last_train.get("weight_update_rel_mean", float("nan"))
        ),
        "duration_sec": float(time.perf_counter() - start),
        "device": str(device),
    }
    return result
