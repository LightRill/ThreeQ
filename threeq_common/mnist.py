from __future__ import annotations

import csv
import fcntl
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .dplus_fix import _dplus_variant_objective
from .dthreeq import BPMLP, DThreeQMLP, set_seed
from .models import BidirectionalMLP
from .training import mean_metrics


ROOT = Path(__file__).resolve().parents[1]


MNIST_VARIANTS: Dict[str, Dict[str, Any]] = {
    "bp_mlp_tanh": {
        "family": "bp_mlp",
        "optimizer": "adam",
        "weight_lr": 1e-3,
    },
    "bp_cnn": {
        "family": "bp_cnn",
        "optimizer": "adam",
        "weight_lr": 1e-3,
    },
    "threeq_direct_cost_w10": {
        "family": "threeq_mlp",
        "weight_update": "direct",
        "weak_mode": "cost",
        "weak_steps": 10,
        "transpose_tied": False,
        "alphas": [0.01, 0.005, 0.001],
    },
    "epthreeq_cost_w10": {
        "family": "threeq_mlp",
        "weight_update": "ep",
        "weak_mode": "cost",
        "weak_steps": 10,
        "transpose_tied": False,
        "alphas": [0.01, 0.005, 0.001],
    },
    "epthreeq_tied_w10": {
        "family": "threeq_mlp",
        "weight_update": "ep",
        "weak_mode": "cost",
        "weak_steps": 10,
        "transpose_tied": True,
        "alphas": [0.01, 0.005, 0.001],
    },
    "dthreeq_ep_nudge_0p01": {
        "family": "dthreeq",
        "loss_mode": "ep",
        "target_mode": "nudge_0.01",
        "beta_sign": "plus",
        "weight_lr": 1e-4,
    },
    "dthreeq_dplus_direct": {
        "family": "dthreeq",
        "loss_mode": "dplus",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "weight_lr": 1e-4,
    },
    "dthreeq_dplus_layergain_direct": {
        "family": "dthreeq_variant_objective",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "objective_name": "layergain_beta_div1",
        "weight_lr": 1e-4,
    },
    "cnnthreeq_direct": {
        "family": "legacy_conv",
        "module_dir": "CNN3Q",
        "alphas": [0.005, 0.001],
        "weak_steps": 10,
    },
    "epcnnthreeq_ep": {
        "family": "legacy_conv",
        "module_dir": "EPCNN3Q",
        "alphas": [0.005, 0.001],
        "weak_steps": 10,
    },
}


class SmallBPCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x.flatten(1))


def _mean_union(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    items = list(items)
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    out: Dict[str, float] = {}
    for key in keys:
        vals = [float(item[key]) for item in items if key in item]
        out[key] = float(sum(vals) / max(1, len(vals)))
    return out


def build_mnist_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    seed = int(config.get("seed", 0))
    batch_size = int(config.get("batch_size", 64))
    train_subset = int(config.get("train_subset", 3000))
    test_subset = int(config.get("test_subset", 1000))
    data_dir = Path(str(config.get("data_dir", ROOT / "data" / "mnist")))
    data_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    lock_path = data_dir / "download.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        train_full = datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=transform
        )
        test_full = datasets.MNIST(
            root=str(data_dir), train=False, download=True, transform=transform
        )
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    train_gen = torch.Generator().manual_seed(10_000 + seed)
    test_gen = torch.Generator().manual_seed(20_000 + seed)
    train_idx = torch.randperm(len(train_full), generator=train_gen)[:train_subset].tolist()
    test_idx = torch.randperm(len(test_full), generator=test_gen)[:test_subset].tolist()
    train_ds = Subset(train_full, train_idx)
    test_ds = Subset(test_full, test_idx)
    loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_gen,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def _make_optimizer(
    params: Iterable[torch.nn.Parameter], variant: Dict[str, Any], config: Dict[str, Any]
) -> torch.optim.Optimizer:
    lr = float(variant.get("weight_lr", config.get("weight_lr", 1e-3)))
    if str(variant.get("optimizer", "sgd")) == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr)


def _train_bp_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    flatten: bool,
) -> Dict[str, float]:
    model.train()
    rows = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        model_input = x.view(x.shape[0], -1) if flatten else x
        logits = model(model_input)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = logits.detach().argmax(dim=1)
        rows.append(
            {
                "train_error": (pred != y).float().mean().item(),
                "train_cost": loss.detach().item(),
            }
        )
    return mean_metrics(rows)


@torch.no_grad()
def _eval_bp(
    model: nn.Module, loader: DataLoader, device: torch.device, flatten: bool
) -> Dict[str, float]:
    model.eval()
    rows = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        model_input = x.view(x.shape[0], -1) if flatten else x
        logits = model(model_input)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        rows.append(
            {
                "test_error": (pred != y).float().mean().item(),
                "test_cost": loss.item(),
            }
        )
    return mean_metrics(rows)


def _train_dthreeq_variant_batch(
    model: DThreeQMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    target_mode: str,
    beta_sign: str,
    objective_name: str,
    residual_norm_eps: float,
) -> Dict[str, float]:
    x = x.to(model.device, non_blocking=True).view(x.shape[0], -1).float()
    y = y.to(model.device, non_blocking=True).long()
    y_one_hot = F.one_hot(y, num_classes=model.layer_sizes[-1]).float()
    states0 = model.initial_states(x)
    states_free = model.relax(x, states0, clamped=False)
    layers_free = model.layers_from_states(x, [s.detach() for s in states_free])
    clamped_rows = []
    objectives = []
    for sign in model.signs():
        states_clamped = model.relax(
            x, states_free, y_one_hot=y_one_hot, sign=sign, clamped=True
        )
        layers_clamped = model.layers_from_states(
            x, [s.detach() for s in states_clamped]
        )
        objectives.append(
            _dplus_variant_objective(
                model=model,
                layers_free=layers_free,
                layers_clamped=layers_clamped,
                target_mode=target_mode,
                objective_name=objective_name,
                residual_norm_eps=residual_norm_eps,
            )
        )
        clamped_rows.append(states_clamped)
    objective = sum(objectives) / max(1, len(objectives))
    update_rel = model.update_weights(objective)
    with torch.no_grad():
        pred = states_free[-1].argmax(dim=1)
        state_delta = 0.0
        for states_clamped in clamped_rows:
            state_delta += sum(
                (c - f).abs().mean().item()
                for c, f in zip(states_clamped, states_free)
            ) / len(states_free)
        state_delta /= max(1, len(clamped_rows))
        return {
            "train_error": (pred != y).float().mean().item(),
            "train_energy": model.energy(layers_free).mean().item(),
            "train_cost": F.mse_loss(states_free[-1], y_one_hot).item(),
            "objective": objective.detach().item(),
            "state_delta": float(state_delta),
            "saturation": model.state_saturation(states_free),
            "weight_abs_mean": model.weight_abs_mean(),
            "weight_update_rel_mean": update_rel,
        }


def _load_conv_network(module_dir: str):
    module_path = ROOT / module_dir / "ThreeQ_Conv.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_threeq_conv", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Network


def _conv_state_saturation(states: List[torch.Tensor]) -> float:
    total = sum(state.numel() for state in states)
    if total == 0:
        return 0.0
    saturated = sum(
        ((state <= 0.0) | (state >= 1.0)).float().sum().item() for state in states
    )
    return float(saturated / total)


def _eval_legacy_conv(model: Any, loader: DataLoader) -> Dict[str, float]:
    rows = []
    for x, y in loader:
        batch_size = x.shape[0]
        x = x.to(model.device, non_blocking=True).view(batch_size, -1).float()
        y = y.to(model.device, non_blocking=True).long()
        y_one_hot = F.one_hot(y, num_classes=model.num_classes).float()
        states0 = [
            torch.zeros(batch_size, dim, device=model.device, dtype=torch.float32)
            for dim in model.layer_sizes[1:]
        ]
        states_free = model.relax_states(
            x, states0, None, 0.0, model.free_steps, model.epsilon
        )
        layers_free = [x] + states_free
        with torch.no_grad():
            pred = states_free[-1].argmax(dim=1)
            rows.append(
                {
                    "test_error": (pred != y).float().mean().item(),
                    "test_cost": model.cost(layers_free, y_one_hot).mean().item(),
                    "test_energy": model.energy(layers_free).mean().item(),
                    "test_saturation": _conv_state_saturation(states_free),
                }
            )
    return mean_metrics(rows)


def _train_legacy_conv_epoch(model: Any, loader: DataLoader) -> Dict[str, float]:
    rows = []
    for x, y in loader:
        energy, cost, error, loss, update_logs = model.train_batch(x, y)
        rows.append(
            {
                "train_error": float(error),
                "train_energy": float(energy),
                "train_cost": float(cost),
                "weak_cost": float(loss),
                "weight_update_rel_mean": float(
                    sum(update_logs) / max(1, len(update_logs))
                ),
            }
        )
    return mean_metrics(rows)


def _record_curve(
    curves: List[Dict[str, float]],
    epoch: int,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    row = {"epoch": float(epoch)}
    row.update(train_metrics)
    row.update(test_metrics)
    curves.append({key: float(value) for key, value in row.items()})


def _summarize_result(
    config: Dict[str, Any],
    variant_name: str,
    variant: Dict[str, Any],
    curves: List[Dict[str, float]],
    start_time: float,
    device: torch.device,
) -> Dict[str, Any]:
    best_epoch = min(curves, key=lambda row: row.get("test_error", float("inf")))
    final = curves[-1] if curves else {}
    result: Dict[str, Any] = {
        "variant": variant_name,
        "family": str(variant["family"]),
        "seed": int(config.get("seed", 0)),
        "dataset": "mnist",
        "train_subset": int(config.get("train_subset", 3000)),
        "test_subset": int(config.get("test_subset", 1000)),
        "n_epochs": int(config.get("n_epochs", 3)),
        "batch_size": int(config.get("batch_size", 64)),
        "best_epoch": int(best_epoch.get("epoch", -1)),
        "best_test_error": float(best_epoch.get("test_error", float("nan"))),
        "final_test_error": float(final.get("test_error", float("nan"))),
        "final_test_cost": float(final.get("test_cost", float("nan"))),
        "final_train_error": float(final.get("train_error", float("nan"))),
        "final_train_cost": float(final.get("train_cost", float("nan"))),
        "final_train_energy": float(final.get("train_energy", float("nan"))),
        "state_delta": float(final.get("state_delta", float("nan"))),
        "saturation": float(final.get("saturation", float("nan"))),
        "weight_abs_mean": float(final.get("weight_abs_mean", float("nan"))),
        "weight_update_rel_mean": float(
            final.get("weight_update_rel_mean", float("nan"))
        ),
        "duration_sec": float(time.perf_counter() - start_time),
        "device": str(device),
        "curve_json": json.dumps(curves, sort_keys=True),
    }
    if "weight_lr" in variant or "weight_lr" in config:
        result["effective_weight_lr"] = float(
            variant.get("weight_lr", config.get("weight_lr", float("nan")))
        )
    for key in [
        "weight_update",
        "weak_mode",
        "transpose_tied",
        "target_mode",
        "beta_sign",
        "loss_mode",
        "objective_name",
        "module_dir",
    ]:
        if key in variant:
            result[key] = variant[key]
    return result


def train_one_mnist(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    config = dict(config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    variant_name = str(config["variant"])
    variant = dict(MNIST_VARIANTS[variant_name])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_mnist_loaders(config)
    n_epochs = int(config.get("n_epochs", 3))
    curves: List[Dict[str, float]] = []
    hidden_sizes = list(config.get("hidden_sizes", [128, 64]))

    if variant["family"] == "bp_mlp":
        model = BPMLP(
            layer_sizes=[784] + hidden_sizes + [10],
            activation=str(config.get("activation", "tanh")),
            bias=bool(config.get("bias", True)),
        ).to(device)
        optimizer = _make_optimizer(model.parameters(), variant, config)
        for epoch in range(n_epochs):
            train_metrics = _train_bp_epoch(model, train_loader, optimizer, device, True)
            test_metrics = _eval_bp(model, test_loader, device, True)
            _record_curve(curves, epoch, train_metrics, test_metrics)
        return _summarize_result(config, variant_name, variant, curves, start, device)

    if variant["family"] == "bp_cnn":
        model = SmallBPCNN(num_classes=10).to(device)
        optimizer = _make_optimizer(model.parameters(), variant, config)
        for epoch in range(n_epochs):
            train_metrics = _train_bp_epoch(model, train_loader, optimizer, device, False)
            test_metrics = _eval_bp(model, test_loader, device, False)
            _record_curve(curves, epoch, train_metrics, test_metrics)
        return _summarize_result(config, variant_name, variant, curves, start, device)

    if variant["family"] == "threeq_mlp":
        model = BidirectionalMLP(
            layer_sizes=[784] + hidden_sizes + [10],
            alphas=list(config.get("alphas", variant["alphas"])),
            beta=float(config.get("beta", 0.5)),
            free_steps=int(config.get("free_steps", 15)),
            weak_steps=int(config.get("weak_steps", variant["weak_steps"])),
            epsilon=float(config.get("epsilon", 0.05)),
            weight_update=str(variant["weight_update"]),
            weak_mode=str(variant["weak_mode"]),
            transpose_tied=bool(variant["transpose_tied"]),
            device=device,
        ).to(device)
        for epoch in range(n_epochs):
            train_metrics = mean_metrics([model.train_batch(x, y) for x, y in train_loader])
            valid_metrics = mean_metrics([model.eval_batch(x, y) for x, y in test_loader])
            test_metrics = {
                key.replace("valid_", "test_"): value
                for key, value in valid_metrics.items()
            }
            _record_curve(curves, epoch, train_metrics, test_metrics)
        return _summarize_result(config, variant_name, variant, curves, start, device)

    if variant["family"] in {"dthreeq", "dthreeq_variant_objective"}:
        model = DThreeQMLP(
            layer_sizes=[784] + hidden_sizes + [10],
            activation=str(config.get("activation", "tanh")),
            infer_steps=int(config.get("infer_steps", 10)),
            infer_lr=float(config.get("infer_lr", 0.05)),
            weight_lr=float(variant.get("weight_lr", config.get("weight_lr", 1e-4))),
            target_mode=str(variant["target_mode"]),
            beta_sign=str(variant["beta_sign"]),
            loss_mode=str(variant.get("loss_mode", "dplus")),
            bias=bool(config.get("bias", True)),
            state_clip=float(config.get("state_clip", 1.0)),
            device=device,
        ).to(device)
        for epoch in range(n_epochs):
            if variant["family"] == "dthreeq_variant_objective":
                train_metrics = mean_metrics(
                    [
                        _train_dthreeq_variant_batch(
                            model,
                            x,
                            y,
                            target_mode=str(variant["target_mode"]),
                            beta_sign=str(variant["beta_sign"]),
                            objective_name=str(variant["objective_name"]),
                            residual_norm_eps=float(
                                config.get("residual_norm_eps", 1e-3)
                            ),
                        )
                        for x, y in train_loader
                    ]
                )
            else:
                train_metrics = mean_metrics(
                    [model.train_batch(x, y) for x, y in train_loader]
                )
            test_metrics = mean_metrics([model.eval_batch(x, y) for x, y in test_loader])
            _record_curve(curves, epoch, train_metrics, test_metrics)
        return _summarize_result(config, variant_name, variant, curves, start, device)

    if variant["family"] == "legacy_conv":
        Network = _load_conv_network(str(variant["module_dir"]))
        model = Network(
            alphas=list(config.get("conv_alphas", variant["alphas"])),
            beta=float(config.get("beta", 0.5)),
            free_steps=int(config.get("conv_free_steps", config.get("free_steps", 10))),
            weak_steps=int(config.get("conv_weak_steps", variant["weak_steps"])),
            epsilon=float(config.get("conv_epsilon", config.get("epsilon", 0.05))),
            conv_channels=list(config.get("conv_channels", [1, 8])),
            kernel_sizes=list(config.get("kernel_sizes", [5])),
            strides=list(config.get("strides", [2])),
            paddings=list(config.get("paddings", [2])),
            input_size=(28, 28),
            num_classes=10,
            device=device,
        ).to(device)
        for epoch in range(n_epochs):
            train_metrics = _train_legacy_conv_epoch(model, train_loader)
            test_metrics = _eval_legacy_conv(model, test_loader)
            _record_curve(curves, epoch, train_metrics, test_metrics)
        return _summarize_result(config, variant_name, variant, curves, start, device)

    raise ValueError(f"unknown MNIST family: {variant['family']}")


def write_curve_rows(raw_csv: Path, curves_csv: Path) -> None:
    rows = []
    with raw_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for result in reader:
            curve_json = result.get("curve_json", "[]")
            try:
                curve = json.loads(curve_json)
            except json.JSONDecodeError:
                curve = []
            for item in curve:
                row = {
                    "variant": result["variant"],
                    "family": result["family"],
                    "seed": result["seed"],
                }
                row.update(item)
                rows.append(row)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with curves_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
