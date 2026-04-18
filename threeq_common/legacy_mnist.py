from __future__ import annotations

import csv
import fcntl
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .dthreeq import set_seed
from .training import mean_metrics


ROOT = Path(__file__).resolve().parents[1]


LEGACY_MNIST_VARIANTS: Dict[str, Dict[str, Any]] = {
    "base3q_legacy_net1_5k_e8": {
        "module_dir": "Base3Q",
        "method": "threeq_direct",
        "hidden_sizes": [500],
        "n_epochs": 8,
        "train_subset": 5000,
        "valid_subset": 1000,
        "batch_size": 20,
        "n_it_neg": 50,
        "n_it_pos": 50,
        "epsilon": 0.05,
        "beta": 0.5,
        "alphas": [0.05, 0.01],
    },
    "epbase3q_legacy_net1_5k_e8": {
        "module_dir": "EPBase3Q",
        "method": "ep_threeq",
        "hidden_sizes": [500],
        "n_epochs": 8,
        "train_subset": 5000,
        "valid_subset": 1000,
        "batch_size": 20,
        "n_it_neg": 50,
        "n_it_pos": 2,
        "epsilon": 0.1,
        "beta": 0.5,
        "alphas": [0.05, 0.01],
    },
}


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


def _load_legacy_network(module_dir: str):
    module_path = ROOT / module_dir / "ThreeQ.py"
    spec = importlib.util.spec_from_file_location(f"{module_dir}_legacy_threeq", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Network


def build_legacy_mnist_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    batch_size = int(config.get("batch_size", 20))
    train_subset = int(config.get("train_subset", 5000))
    valid_subset = int(config.get("valid_subset", 1000))
    data_dir = Path(str(config.get("data_dir", ROOT / "data" / "mnist")))
    data_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.ToTensor()
    lock_path = data_dir / "download.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        full_train = datasets.MNIST(
            root=str(data_dir), train=True, download=True, transform=transform
        )
        fcntl.flock(lock_file, fcntl.LOCK_UN)

    train_end = min(train_subset, 50_000)
    valid_end = min(50_000 + valid_subset, len(full_train))
    train_ds = Subset(full_train, list(range(0, train_end)))
    valid_ds = Subset(full_train, list(range(50_000, valid_end)))
    generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader


def _eval_free_phase(net: Any, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    batch_size = x.shape[0]
    x = x.to(net.device, non_blocking=True).view(batch_size, -1).float()
    y = y.to(net.device, non_blocking=True).long()
    y_one_hot = F.one_hot(y, num_classes=net.layer_sizes[-1]).float()
    states0 = [
        torch.zeros(batch_size, dim, device=net.device, dtype=torch.float32)
        for dim in net.layer_sizes[1:]
    ]
    states_free = net.relax_states(
        x=x,
        states_init=states0,
        y_one_hot=None,
        beta=0.0,
        n_steps=net.free_steps,
        epsilon=net.epsilon,
    )
    layers_free = [x] + states_free
    with torch.no_grad():
        pred = states_free[-1].argmax(dim=1)
        return {
            "valid_energy": net.energy(layers_free).mean().item(),
            "valid_cost": net.cost(layers_free, y_one_hot).mean().item(),
            "valid_error": (pred != y).float().mean().item(),
        }


def _train_epoch(net: Any, loader: DataLoader) -> Dict[str, float]:
    rows = []
    for x, y in loader:
        (
            energy,
            cost,
            error,
            loss,
            rho_mean,
            rho_max,
            forward_delta_logs,
            backward_delta_logs,
        ) = net.train_batch(x, y)
        rows.append(
            {
                "train_energy": float(energy),
                "train_cost": float(cost),
                "train_error": float(error),
                "weak_cost": float(loss),
                "rho_mean": float(rho_mean),
                "rho_max": float(rho_max),
                "forward_update_rel_mean": float(
                    sum(forward_delta_logs) / max(1, len(forward_delta_logs))
                ),
                "backward_update_rel_mean": float(
                    sum(backward_delta_logs) / max(1, len(backward_delta_logs))
                ),
            }
        )
    return mean_metrics(rows)


def _record_curve(
    curves: List[Dict[str, float]],
    epoch: int,
    train_metrics: Dict[str, float],
    valid_metrics: Dict[str, float],
) -> None:
    row = {"epoch": float(epoch)}
    row.update(train_metrics)
    row.update(valid_metrics)
    curves.append({key: float(value) for key, value in row.items()})


def train_one_legacy_mnist(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    config = dict(config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    variant_name = str(config["variant"])
    variant = dict(LEGACY_MNIST_VARIANTS[variant_name])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = build_legacy_mnist_loaders(variant | config)
    hidden_sizes = list(variant["hidden_sizes"])
    layer_sizes = [784] + hidden_sizes + [10]
    Network = _load_legacy_network(str(variant["module_dir"]))
    net = Network(
        layer_sizes=layer_sizes,
        alphas=list(variant["alphas"]),
        beta=float(variant["beta"]),
        free_steps=int(variant["n_it_neg"]),
        weak_steps=int(variant["n_it_pos"]),
        epsilon=float(variant["epsilon"]),
        device=device,
    ).to(device)
    curves: List[Dict[str, float]] = []
    for epoch in range(int(variant["n_epochs"])):
        train_metrics = _train_epoch(net, train_loader)
        valid_metrics = _mean_union([_eval_free_phase(net, x, y) for x, y in valid_loader])
        _record_curve(curves, epoch, train_metrics, valid_metrics)

    best = min(curves, key=lambda row: row.get("valid_error", float("inf")))
    final = curves[-1]
    return {
        "variant": variant_name,
        "method": str(variant["method"]),
        "module_dir": str(variant["module_dir"]),
        "seed": seed,
        "dataset": "mnist",
        "hidden_sizes": str(hidden_sizes),
        "train_subset": int(variant["train_subset"]),
        "valid_subset": int(variant["valid_subset"]),
        "n_epochs": int(variant["n_epochs"]),
        "batch_size": int(variant["batch_size"]),
        "n_it_neg": int(variant["n_it_neg"]),
        "n_it_pos": int(variant["n_it_pos"]),
        "epsilon": float(variant["epsilon"]),
        "beta": float(variant["beta"]),
        "alphas": str(list(variant["alphas"])),
        "best_epoch": int(best.get("epoch", -1)),
        "best_valid_error": float(best.get("valid_error", float("nan"))),
        "final_valid_error": float(final.get("valid_error", float("nan"))),
        "final_train_error": float(final.get("train_error", float("nan"))),
        "final_train_cost": float(final.get("train_cost", float("nan"))),
        "final_valid_cost": float(final.get("valid_cost", float("nan"))),
        "rho_mean": float(final.get("rho_mean", float("nan"))),
        "rho_max": float(final.get("rho_max", float("nan"))),
        "forward_update_rel_mean": float(
            final.get("forward_update_rel_mean", float("nan"))
        ),
        "backward_update_rel_mean": float(
            final.get("backward_update_rel_mean", float("nan"))
        ),
        "duration_sec": float(time.perf_counter() - start),
        "device": str(device),
        "curve_json": json.dumps(curves, sort_keys=True),
    }


def write_curve_rows(raw_csv: Path, curves_csv: Path) -> None:
    rows = []
    with raw_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for result in reader:
            try:
                curve = json.loads(result.get("curve_json", "[]"))
            except json.JSONDecodeError:
                curve = []
            for item in curve:
                row = {
                    "variant": result["variant"],
                    "method": result["method"],
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
