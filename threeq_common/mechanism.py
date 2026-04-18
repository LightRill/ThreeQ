from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dthreeq import (
    DThreeQMLP,
    build_twomoons_loaders,
    parse_target_mode,
    set_seed,
)
from .training import mean_metrics


TARGET_SPECS: Dict[str, Dict[str, str]] = {
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
    "nudge_0p01_plusminus": {
        "target_mode": "nudge_0.01",
        "beta_sign": "plusminus",
    },
    "gradual_100_0p01_plus": {
        "target_mode": "gradual_clamp_100_0.01",
        "beta_sign": "plus",
    },
}

PAIRS = [
    ("dplus", "bp"),
    ("ep", "bp"),
    ("dplus", "ep"),
]

SCOPES = ["full", "forward"]
METHODS = ["bp", "ep", "dplus"]


def _prepare_batch(
    model: DThreeQMLP, x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.to(model.device, non_blocking=True).view(x.shape[0], -1).float()
    y = y.to(model.device, non_blocking=True).long()
    y_one_hot = F.one_hot(y, num_classes=model.layer_sizes[-1]).float()
    return x, y, y_one_hot


def _zero_param_grads(model: DThreeQMLP) -> None:
    for param in model.parameters():
        param.grad = None


def _selected_named_parameters(
    model: DThreeQMLP, scope: str
) -> Iterable[Tuple[str, torch.nn.Parameter]]:
    if scope not in {"full", "forward"}:
        raise ValueError(f"unknown scope: {scope}")
    for name, param in model.named_parameters():
        if scope == "forward" and not name.startswith("forward_"):
            continue
        yield name, param


def _flatten_params(model: DThreeQMLP, scope: str) -> torch.Tensor:
    return torch.cat(
        [param.detach().flatten() for _, param in _selected_named_parameters(model, scope)]
    )


def _set_flat_params(model: DThreeQMLP, flat: torch.Tensor, scope: str) -> None:
    offset = 0
    with torch.no_grad():
        for _, param in _selected_named_parameters(model, scope):
            n = param.numel()
            param.copy_(flat[offset : offset + n].view_as(param))
            offset += n


def _flatten_update_direction(model: DThreeQMLP, scope: str) -> torch.Tensor:
    chunks = []
    for _, param in _selected_named_parameters(model, scope):
        if param.grad is None:
            chunks.append(torch.zeros_like(param).flatten())
        else:
            chunks.append((-param.grad).detach().flatten())
    return torch.cat(chunks)


def _apply_flat_update(model: DThreeQMLP, update: torch.Tensor, scope: str) -> None:
    offset = 0
    with torch.no_grad():
        for _, param in _selected_named_parameters(model, scope):
            n = param.numel()
            param.add_(update[offset : offset + n].view_as(param))
            offset += n


def _feedforward_logits(model: DThreeQMLP, x: torch.Tensor) -> torch.Tensor:
    current = x
    for i, weight in enumerate(model.forward_weights):
        current = model._act(current) @ weight
        if model.bias:
            current = current + model.forward_biases[i]
        if i < len(model.forward_weights) - 1:
            current = model._act(current)
    return current


def _feedforward_ce(model: DThreeQMLP, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(_feedforward_logits(model, x), y)


def _free_mse(model: DThreeQMLP, x: torch.Tensor, y_one_hot: torch.Tensor) -> torch.Tensor:
    states0 = model.initial_states(x)
    states_free = model.relax(x, states0, clamped=False)
    loss = F.mse_loss(states_free[-1], y_one_hot)
    _zero_param_grads(model)
    return loss.detach()


def _bp_directions(
    model: DThreeQMLP, x: torch.Tensor, y: torch.Tensor
) -> Dict[str, torch.Tensor]:
    _zero_param_grads(model)
    loss = _feedforward_ce(model, x, y)
    loss.backward()
    directions = {scope: _flatten_update_direction(model, scope) for scope in SCOPES}
    _zero_param_grads(model)
    return directions


def _local_objective(
    model: DThreeQMLP,
    x: torch.Tensor,
    y_one_hot: torch.Tensor,
    target_mode: str,
    beta_sign: str,
    method: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        if method == "dplus":
            objectives.append(model.dplus_objective(layers_free, layers_clamped))
        elif method == "ep":
            objectives.append(sign * model.ep_objective(layers_free, layers_clamped))
        else:
            raise ValueError(f"unknown local method: {method}")
        clamped_states_all.append(states_clamped)

    objective = sum(objectives) / max(1, len(objectives))
    with torch.no_grad():
        state_delta = 0.0
        for states_clamped in clamped_states_all:
            state_delta += sum(
                (c - f).abs().mean().item()
                for c, f in zip(states_clamped, states_free)
            ) / len(states_free)
        state_delta /= max(1, len(clamped_states_all))
        stats = {
            f"{method}_objective": float(objective.detach().item()),
            f"{method}_state_delta": float(state_delta),
            f"{method}_saturation": model.state_saturation(states_free),
        }
    return objective, stats


def _local_directions(
    model: DThreeQMLP,
    x: torch.Tensor,
    y_one_hot: torch.Tensor,
    target_mode: str,
    beta_sign: str,
    method: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    objective, stats = _local_objective(
        model=model,
        x=x,
        y_one_hot=y_one_hot,
        target_mode=target_mode,
        beta_sign=beta_sign,
        method=method,
    )
    _zero_param_grads(model)
    objective.backward()
    directions = {scope: _flatten_update_direction(model, scope) for scope in SCOPES}
    _zero_param_grads(model)
    return directions, stats


def _vector_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    eps = 1e-12
    a = a.detach().float()
    b = b.detach().float()
    norm_a = torch.linalg.vector_norm(a).item()
    norm_b = torch.linalg.vector_norm(b).item()
    if norm_a <= eps or norm_b <= eps:
        cosine = float("nan")
    else:
        cosine = F.cosine_similarity(a, b, dim=0).item()
    mask = (a.abs() > eps) | (b.abs() > eps)
    if mask.any():
        sign_agreement = (torch.sign(a[mask]) == torch.sign(b[mask])).float().mean().item()
    else:
        sign_agreement = float("nan")
    return {
        "cosine": float(cosine),
        "norm_ratio": float(norm_a / (norm_b + eps)),
        "sign_agreement": float(sign_agreement),
        "norm_a": float(norm_a),
        "norm_b": float(norm_b),
    }


def _loss_decreases(
    model: DThreeQMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    y_one_hot: torch.Tensor,
    directions_full: Dict[str, torch.Tensor],
    step_lr: float,
) -> Dict[str, float]:
    original = _flatten_params(model, "full")
    base_free_mse = float(_free_mse(model, x, y_one_hot).item())
    base_ce = float(_feedforward_ce(model, x, y).detach().item())
    bp_step_norm = float(torch.linalg.vector_norm(step_lr * directions_full["bp"]).item())
    out: Dict[str, float] = {
        "base_free_mse": base_free_mse,
        "base_feedforward_ce": base_ce,
        "bp_step_norm": bp_step_norm,
    }
    for method in METHODS:
        direction = directions_full[method]
        raw_update = step_lr * direction
        direction_norm = float(torch.linalg.vector_norm(direction).item())
        if direction_norm > 0 and bp_step_norm > 0:
            bp_normed_update = direction / direction_norm * bp_step_norm
        else:
            bp_normed_update = torch.zeros_like(direction)
        for label, update in [
            ("raw", raw_update),
            ("bp_normed", bp_normed_update),
        ]:
            _set_flat_params(model, original, "full")
            _apply_flat_update(model, update, "full")
            after_free_mse = float(_free_mse(model, x, y_one_hot).item())
            after_ce = float(_feedforward_ce(model, x, y).detach().item())
            out[f"{method}_free_mse_decrease_{label}"] = base_free_mse - after_free_mse
            out[f"{method}_feedforward_ce_decrease_{label}"] = base_ce - after_ce
        _set_flat_params(model, original, "full")
    _zero_param_grads(model)
    return out


def _diagnose_batch(
    model: DThreeQMLP,
    x_raw: torch.Tensor,
    y_raw: torch.Tensor,
    target_mode: str,
    beta_sign: str,
    step_lr: float,
) -> Dict[str, float]:
    x, y, y_one_hot = _prepare_batch(model, x_raw, y_raw)
    bp_dirs = _bp_directions(model, x, y)
    ep_dirs, ep_stats = _local_directions(
        model, x, y_one_hot, target_mode, beta_sign, method="ep"
    )
    dplus_dirs, dplus_stats = _local_directions(
        model, x, y_one_hot, target_mode, beta_sign, method="dplus"
    )
    directions = {
        "bp": bp_dirs,
        "ep": ep_dirs,
        "dplus": dplus_dirs,
    }
    row: Dict[str, float] = {}
    row.update(ep_stats)
    row.update(dplus_stats)
    for method in METHODS:
        for scope in SCOPES:
            row[f"{method}_{scope}_direction_norm"] = float(
                torch.linalg.vector_norm(directions[method][scope]).item()
            )
    for left, right in PAIRS:
        for scope in SCOPES:
            stats = _vector_stats(directions[left][scope], directions[right][scope])
            prefix = f"{left}_vs_{right}_{scope}"
            for key, value in stats.items():
                row[f"{prefix}_{key}"] = value
    row.update(
        _loss_decreases(
            model,
            x,
            y,
            y_one_hot,
            {method: directions[method]["full"] for method in METHODS},
            step_lr=step_lr,
        )
    )
    return row


def run_mechanism_diagnostic(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    config = dict(config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    target_name = str(config["target_name"])
    target_spec = TARGET_SPECS[target_name]
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
    n_batches = int(config.get("n_batches", 8))
    step_lr = float(config.get("step_lr", 1e-3))
    batch_items: List[Dict[str, float]] = []
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        batch_row = _diagnose_batch(
            model=model,
            x_raw=x,
            y_raw=y,
            target_mode=target_mode,
            beta_sign=beta_sign,
            step_lr=step_lr,
        )
        batch_row["batch_idx"] = float(batch_idx)
        batch_items.append(batch_row)
    metrics = mean_metrics(batch_items)
    positive_rates: Dict[str, float] = {}
    for method in METHODS:
        for loss_name in ["free_mse", "feedforward_ce"]:
            for label in ["raw", "bp_normed"]:
                key = f"{method}_{loss_name}_decrease_{label}"
                vals = [item[key] for item in batch_items if key in item]
                positive_rates[f"{key}_positive_rate"] = float(
                    np.mean([value > 0 for value in vals])
                )
    result: Dict[str, Any] = {
        "target_name": target_name,
        "target_mode": target_mode,
        "beta_sign": beta_sign,
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
        "activation": str(config.get("activation", "tanh")),
        "device": str(device),
        "duration_sec": float(time.perf_counter() - start),
    }
    result.update({key: float(value) for key, value in metrics.items()})
    result.update(positive_rates)
    return result
