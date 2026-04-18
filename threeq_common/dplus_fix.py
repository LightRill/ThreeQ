from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dthreeq import DThreeQMLP, build_twomoons_loaders, parse_target_mode, set_seed
from .mechanism import (
    _apply_flat_update,
    _bp_directions,
    _feedforward_ce,
    _flatten_params,
    _free_mse,
    _local_directions,
    _prepare_batch,
    _set_flat_params,
    _vector_stats,
    _zero_param_grads,
)
from .training import mean_metrics


PLUS_TARGET_SPECS: Dict[str, Dict[str, str]] = {
    "direct_plus": {
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
    },
    "nudge_0p1_plus": {
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
    },
    "nudge_0p01_plus": {
        "target_mode": "nudge_0.01",
        "beta_sign": "plus",
    },
    "nudge_0p001_plus": {
        "target_mode": "nudge_0.001",
        "beta_sign": "plus",
    },
    "gradual_100_0p01_plus": {
        "target_mode": "gradual_clamp_100_0.01",
        "beta_sign": "plus",
    },
}


OBJECTIVE_SPECS: Dict[str, Dict[str, Any]] = {
    "raw": {
        "beta_power": 0.0,
        "residual_norm": False,
        "layer_gain": False,
    },
    "beta_div1": {
        "beta_power": 1.0,
        "residual_norm": False,
        "layer_gain": False,
    },
    "beta_div2": {
        "beta_power": 2.0,
        "residual_norm": False,
        "layer_gain": False,
    },
    "resnorm_beta_div1": {
        "beta_power": 1.0,
        "residual_norm": True,
        "layer_gain": False,
    },
    "resnorm_beta_div2": {
        "beta_power": 2.0,
        "residual_norm": True,
        "layer_gain": False,
    },
    "layergain_beta_div1": {
        "beta_power": 1.0,
        "residual_norm": False,
        "layer_gain": True,
    },
    "resnorm_layergain_beta_div1": {
        "beta_power": 1.0,
        "residual_norm": True,
        "layer_gain": True,
    },
}


def _effective_beta(target_mode: str) -> float:
    mode = parse_target_mode(target_mode)
    if mode.kind == "nudge":
        return max(abs(mode.beta), 1e-8)
    if mode.kind == "gradual":
        return max(abs(mode.gamma), 1e-8)
    return 1.0


def _residual_gain(index: int, n_residuals: int, use_layer_gain: bool) -> float:
    if not use_layer_gain:
        return 1.0
    edge_idx = index // 2
    n_edges = max(1, n_residuals // 2)
    distance_from_output = max(1, n_edges - edge_idx)
    return math.sqrt(float(distance_from_output))


def _dplus_variant_objective(
    model: DThreeQMLP,
    layers_free: List[torch.Tensor],
    layers_clamped: List[torch.Tensor],
    target_mode: str,
    objective_name: str,
    residual_norm_eps: float,
) -> torch.Tensor:
    spec = OBJECTIVE_SPECS[objective_name]
    beta_scale = _effective_beta(target_mode) ** float(spec["beta_power"])
    beta_scale = max(beta_scale, 1e-12)
    residuals_free = [r.detach() for r in model.residuals(layers_free)]
    residuals_clamped = model.residuals(layers_clamped)
    terms = []
    for index, (r_clamped, r_free) in enumerate(zip(residuals_clamped, residuals_free)):
        delta = r_clamped - r_free
        if bool(spec["residual_norm"]):
            denom = r_free.pow(2).mean().sqrt().detach().clamp_min(residual_norm_eps)
            delta = delta / denom
        gain = _residual_gain(index, len(residuals_free), bool(spec["layer_gain"]))
        delta = gain * delta
        terms.append(0.5 * delta.pow(2).mean())
    return sum(terms) / max(1, len(terms)) / beta_scale


def _dplus_variant_directions(
    model: DThreeQMLP,
    x: torch.Tensor,
    y_one_hot: torch.Tensor,
    target_mode: str,
    beta_sign: str,
    objective_name: str,
    residual_norm_eps: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    model.target_mode = parse_target_mode(target_mode)
    model.beta_sign = beta_sign
    states0 = model.initial_states(x)
    states_free = model.relax(x, states0, clamped=False)
    layers_free = model.layers_from_states(x, [s.detach() for s in states_free])
    objectives = []
    clamped_states_all = []
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
        clamped_states_all.append(states_clamped)
    objective = sum(objectives) / max(1, len(objectives))
    _zero_param_grads(model)
    objective.backward()
    directions = {
        "full": torch.cat(
            [
                torch.zeros_like(param).flatten()
                if param.grad is None
                else (-param.grad).detach().flatten()
                for _, param in model.named_parameters()
            ]
        ),
        "forward": torch.cat(
            [
                torch.zeros_like(param).flatten()
                if param.grad is None
                else (-param.grad).detach().flatten()
                for name, param in model.named_parameters()
                if name.startswith("forward_")
            ]
        ),
    }
    _zero_param_grads(model)
    with torch.no_grad():
        state_delta = 0.0
        for states_clamped in clamped_states_all:
            state_delta += sum(
                (c - f).abs().mean().item()
                for c, f in zip(states_clamped, states_free)
            ) / len(states_free)
        state_delta /= max(1, len(clamped_states_all))
        stats = {
            "dplus_objective": float(objective.detach().item()),
            "dplus_state_delta": float(state_delta),
            "dplus_saturation": model.state_saturation(states_free),
        }
    return directions, stats


def _finite(value: float) -> float:
    if math.isfinite(value):
        return float(value)
    return float("nan")


def _evaluate_updates(
    model: DThreeQMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    y_one_hot: torch.Tensor,
    updates: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    original = _flatten_params(model, "full")
    base_free_mse = float(_free_mse(model, x, y_one_hot).item())
    base_ce = float(_feedforward_ce(model, x, y).detach().item())
    out: Dict[str, float] = {
        "base_free_mse": base_free_mse,
        "base_feedforward_ce": base_ce,
    }
    for name, update in updates.items():
        _set_flat_params(model, original, "full")
        _apply_flat_update(model, update, "full")
        try:
            after_free_mse = float(_free_mse(model, x, y_one_hot).item())
            after_ce = float(_feedforward_ce(model, x, y).detach().item())
        except RuntimeError:
            after_free_mse = float("nan")
            after_ce = float("nan")
        out[f"{name}_free_mse_decrease"] = _finite(base_free_mse - after_free_mse)
        out[f"{name}_feedforward_ce_decrease"] = _finite(base_ce - after_ce)
        _set_flat_params(model, original, "full")
    _zero_param_grads(model)
    return out


def _calibrated_update(
    full_direction: torch.Tensor,
    forward_direction: torch.Tensor,
    reference_forward_direction: torch.Tensor,
    step_lr: float,
) -> torch.Tensor:
    eps = 1e-12
    direction_norm = torch.linalg.vector_norm(forward_direction)
    reference_norm = torch.linalg.vector_norm(reference_forward_direction)
    if direction_norm.item() <= eps or reference_norm.item() <= eps:
        return torch.zeros_like(full_direction)
    scale = reference_norm / direction_norm
    return step_lr * scale * full_direction


def _diagnose_fixed_batch(
    model: DThreeQMLP,
    x_raw: torch.Tensor,
    y_raw: torch.Tensor,
    target_mode: str,
    beta_sign: str,
    objective_name: str,
    step_lr: float,
    residual_norm_eps: float,
) -> Dict[str, float]:
    x, y, y_one_hot = _prepare_batch(model, x_raw, y_raw)
    bp_dirs = _bp_directions(model, x, y)
    ep_dirs, ep_stats = _local_directions(
        model, x, y_one_hot, target_mode, beta_sign, method="ep"
    )
    dplus_dirs, dplus_stats = _dplus_variant_directions(
        model,
        x,
        y_one_hot,
        target_mode=target_mode,
        beta_sign=beta_sign,
        objective_name=objective_name,
        residual_norm_eps=residual_norm_eps,
    )
    row: Dict[str, float] = {}
    row.update(ep_stats)
    row.update(dplus_stats)
    for ref_name, ref_dirs in [("bp", bp_dirs), ("ep", ep_dirs)]:
        stats = _vector_stats(dplus_dirs["forward"], ref_dirs["forward"])
        for key, value in stats.items():
            row[f"dplus_vs_{ref_name}_forward_{key}"] = value
    row["dplus_forward_norm"] = float(torch.linalg.vector_norm(dplus_dirs["forward"]).item())
    row["bp_forward_norm"] = float(torch.linalg.vector_norm(bp_dirs["forward"]).item())
    row["ep_forward_norm"] = float(torch.linalg.vector_norm(ep_dirs["forward"]).item())

    updates = {
        "bp_raw": step_lr * bp_dirs["full"],
        "ep_raw": step_lr * ep_dirs["full"],
        "dplus_raw": step_lr * dplus_dirs["full"],
        "dplus_bp_scaled": _calibrated_update(
            dplus_dirs["full"], dplus_dirs["forward"], bp_dirs["forward"], step_lr
        ),
        "dplus_ep_scaled": _calibrated_update(
            dplus_dirs["full"], dplus_dirs["forward"], ep_dirs["forward"], step_lr
        ),
    }
    row.update(_evaluate_updates(model, x, y, y_one_hot, updates))
    return row


def run_dplus_fix_diagnostic(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    config = dict(config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    target_name = str(config["target_name"])
    target_spec = PLUS_TARGET_SPECS[target_name]
    objective_name = str(config["objective_name"])
    if objective_name not in OBJECTIVE_SPECS:
        raise ValueError(f"unknown objective_name: {objective_name}")
    target_mode = str(target_spec["target_mode"])
    beta_sign = str(target_spec["beta_sign"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_sizes = list(config.get("hidden_sizes", [128, 64]))
    layer_sizes = [2] + hidden_sizes + [2]
    train_loader, _ = build_twomoons_loaders(config)
    model = DThreeQMLP(
        layer_sizes=layer_sizes,
        activation=str(config.get("activation", "tanh")),
        infer_steps=int(config.get("infer_steps", 15)),
        infer_lr=float(config.get("infer_lr", 0.05)),
        weight_lr=float(config.get("step_lr", 1e-3)),
        target_mode=target_mode,
        beta_sign=beta_sign,
        loss_mode="dplus",
        bias=bool(config.get("bias", True)),
        state_clip=float(config.get("state_clip", 1.0)),
        device=device,
    ).to(device)
    n_batches = int(config.get("n_batches", 6))
    step_lr = float(config.get("step_lr", 1e-3))
    residual_norm_eps = float(config.get("residual_norm_eps", 1e-3))
    batch_items: List[Dict[str, float]] = []
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        item = _diagnose_fixed_batch(
            model=model,
            x_raw=x,
            y_raw=y,
            target_mode=target_mode,
            beta_sign=beta_sign,
            objective_name=objective_name,
            step_lr=step_lr,
            residual_norm_eps=residual_norm_eps,
        )
        item["batch_idx"] = float(batch_idx)
        batch_items.append(item)
    metrics = mean_metrics(batch_items)
    positive_rates: Dict[str, float] = {}
    for update_name in ["dplus_raw", "dplus_bp_scaled", "dplus_ep_scaled", "bp_raw", "ep_raw"]:
        key = f"{update_name}_free_mse_decrease"
        vals = [item[key] for item in batch_items if key in item and math.isfinite(item[key])]
        positive_rates[f"{key}_positive_rate"] = float(
            np.mean([value > 0 for value in vals])
        ) if vals else float("nan")
    result: Dict[str, Any] = {
        "target_name": target_name,
        "target_mode": target_mode,
        "beta_sign": beta_sign,
        "objective_name": objective_name,
        "objective_beta_power": float(OBJECTIVE_SPECS[objective_name]["beta_power"]),
        "objective_residual_norm": bool(OBJECTIVE_SPECS[objective_name]["residual_norm"]),
        "objective_layer_gain": bool(OBJECTIVE_SPECS[objective_name]["layer_gain"]),
        "seed": seed,
        "dataset": str(config.get("dataset", "twomoons")),
        "n_samples": int(config.get("n_samples", 2000)),
        "noise": float(config.get("noise", 0.1)),
        "hidden_sizes": str(hidden_sizes),
        "batch_size": int(config.get("batch_size", 32)),
        "n_batches": int(len(batch_items)),
        "infer_steps": int(config.get("infer_steps", 15)),
        "infer_lr": float(config.get("infer_lr", 0.05)),
        "step_lr": step_lr,
        "residual_norm_eps": residual_norm_eps,
        "activation": str(config.get("activation", "tanh")),
        "device": str(device),
        "duration_sec": float(time.perf_counter() - start),
    }
    result.update({key: float(value) for key, value in metrics.items()})
    result.update(positive_rates)
    return result
