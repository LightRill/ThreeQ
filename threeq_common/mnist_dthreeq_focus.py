from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn.functional as F

from .dthreeq import BPMLP, DThreeQMLP, set_seed
from .mnist import (
    _eval_bp,
    _make_optimizer,
    _mean_union,
    _record_curve,
    _train_bp_epoch,
    _train_dthreeq_variant_batch,
    build_mnist_loaders,
    write_curve_rows,
)
from .training import mean_metrics


ROOT = Path(__file__).resolve().parents[1]


DTHREEQ_MNIST_FOCUS_VARIANTS: Dict[str, Dict[str, Any]] = {
    "bp_mlp_500_adam": {
        "family": "bp_mlp",
        "hidden_sizes": [500],
        "optimizer": "adam",
        "weight_lr": 1e-3,
    },
    "dthreeq_ep_nudge0p01_lr1e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.01",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_nudge0p01_lr3e4": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.01",
        "beta_sign": "plus",
        "weight_lr": 3e-4,
        "infer_steps": 10,
    },
    "dthreeq_ep_nudge0p1_lr1e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_nudge0p05_lr1e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.05",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_nudge0p2_lr1e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.2",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_nudge0p2_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.2",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_nudge0p1_lr1e3_steps20": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 20,
    },
    "dthreeq_dplus_direct_lr1e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "dplus",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_dplus_direct_lr3e4": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "dplus",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "weight_lr": 3e-4,
        "infer_steps": 10,
    },
    "dthreeq_dplus_layergain_lr1e3": {
        "family": "dthreeq_variant_objective",
        "hidden_sizes": [500],
        "loss_mode": "dplus",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "objective_name": "layergain_beta_div1",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
}


def _merge_config(config: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    merged.update(variant)
    return merged


def _summarize_result(
    config: Dict[str, Any],
    variant_name: str,
    variant: Dict[str, Any],
    curves: List[Dict[str, float]],
    start_time: float,
    device: torch.device,
) -> Dict[str, Any]:
    best = min(curves, key=lambda row: row.get("test_error", float("inf")))
    final = curves[-1]
    result: Dict[str, Any] = {
        "variant": variant_name,
        "family": str(variant["family"]),
        "seed": int(config.get("seed", 0)),
        "dataset": "mnist",
        "train_subset": int(config.get("train_subset", 10_000)),
        "test_subset": int(config.get("test_subset", 2_000)),
        "n_epochs": int(config.get("n_epochs", 12)),
        "batch_size": int(config.get("batch_size", 64)),
        "hidden_sizes": str(list(config.get("hidden_sizes", [500]))),
        "best_epoch": int(best.get("epoch", -1)),
        "best_test_error": float(best.get("test_error", float("nan"))),
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
    for key in [
        "optimizer",
        "weight_lr",
        "infer_steps",
        "infer_lr",
        "activation",
        "bias",
        "state_clip",
        "loss_mode",
        "target_mode",
        "beta_sign",
        "objective_name",
    ]:
        if key in config:
            result[key] = config[key]
    return result


def _train_dthreeq_epoch(
    model: DThreeQMLP,
    loader: Iterable,
    config: Dict[str, Any],
    variant: Dict[str, Any],
) -> Dict[str, float]:
    if variant["family"] == "dthreeq_variant_objective":
        return mean_metrics(
            [
                _train_dthreeq_variant_batch(
                    model,
                    x,
                    y,
                    target_mode=str(config["target_mode"]),
                    beta_sign=str(config["beta_sign"]),
                    objective_name=str(config["objective_name"]),
                    residual_norm_eps=float(config.get("residual_norm_eps", 1e-3)),
                )
                for x, y in loader
            ]
        )
    return mean_metrics([model.train_batch(x, y) for x, y in loader])


def _eval_dthreeq(model: DThreeQMLP, loader: Iterable) -> Dict[str, float]:
    return mean_metrics([model.eval_batch(x, y) for x, y in loader])


def train_one_mnist_dthreeq_focus(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    config = dict(config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    variant_name = str(config["variant"])
    variant = dict(DTHREEQ_MNIST_FOCUS_VARIANTS[variant_name])
    run_cfg = _merge_config(config, variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_mnist_loaders(run_cfg)
    hidden_sizes = list(run_cfg.get("hidden_sizes", [500]))
    n_epochs = int(run_cfg.get("n_epochs", 12))
    curves: List[Dict[str, float]] = []

    if variant["family"] == "bp_mlp":
        model = BPMLP(
            layer_sizes=[784] + hidden_sizes + [10],
            activation=str(run_cfg.get("activation", "tanh")),
            bias=bool(run_cfg.get("bias", True)),
        ).to(device)
        optimizer = _make_optimizer(model.parameters(), variant, run_cfg)
        for epoch in range(n_epochs):
            train_metrics = _train_bp_epoch(model, train_loader, optimizer, device, True)
            test_metrics = _eval_bp(model, test_loader, device, True)
            _record_curve(curves, epoch, train_metrics, test_metrics)
        return _summarize_result(run_cfg, variant_name, variant, curves, start, device)

    model = DThreeQMLP(
        layer_sizes=[784] + hidden_sizes + [10],
        activation=str(run_cfg.get("activation", "tanh")),
        infer_steps=int(run_cfg.get("infer_steps", 10)),
        infer_lr=float(run_cfg.get("infer_lr", 0.05)),
        weight_lr=float(run_cfg.get("weight_lr", 1e-3)),
        target_mode=str(run_cfg["target_mode"]),
        beta_sign=str(run_cfg["beta_sign"]),
        loss_mode=str(run_cfg.get("loss_mode", "dplus")),
        bias=bool(run_cfg.get("bias", True)),
        state_clip=float(run_cfg.get("state_clip", 1.0)),
        device=device,
    ).to(device)
    for epoch in range(n_epochs):
        train_metrics = _train_dthreeq_epoch(model, train_loader, run_cfg, variant)
        test_metrics = _eval_dthreeq(model, test_loader)
        _record_curve(curves, epoch, train_metrics, test_metrics)
    return _summarize_result(run_cfg, variant_name, variant, curves, start, device)


def write_focus_curve_rows(raw_csv: Path, curves_csv: Path) -> None:
    write_curve_rows(raw_csv, curves_csv)
