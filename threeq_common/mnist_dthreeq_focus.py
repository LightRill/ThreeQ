from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
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
    "dthreeq_ep_tanh_nudge0p1_lr5e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "tanh",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 5e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_noinput_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "input_residual_weight": 0.0,
        "infer_steps": 10,
    },
    "dthreeq_ep_input1over784_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "input_residual_weight": 1.0 / 784.0,
        "infer_steps": 10,
    },
    "dthreeq_ep_input1over28_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "input_residual_weight": 1.0 / 28.0,
        "infer_steps": 10,
    },
    "dthreeq_ep_ce_nudge0p05_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "ce_nudge_0.05",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_ce_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "ce_nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_ce_nudge0p1_lr3e3_restorebest": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "ce_nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "restore_best": True,
        "infer_steps": 10,
    },
    "dthreeq_ep_ce_nudge0p1_lr5e3_decay15_restorebest": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "ce_nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 5e-3,
        "weight_lr_decay_epoch": 15,
        "weight_lr_decay_factor": 0.2,
        "restore_best": True,
        "early_stop_patience": 8,
        "infer_steps": 10,
    },
    "dthreeq_ep_signed_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "label_encoding": "signed",
        "infer_steps": 10,
    },
    "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "label_encoding": "signed",
        "restore_best": True,
        "infer_steps": 10,
    },
    "dthreeq_ep_readout_nudge0p1_lr3e3": {
        "family": "dthreeq_readout",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "readout_lr": 1e-2,
        "infer_steps": 10,
    },
    "dthreeq_ep_readout_noinput_nudge0p1_lr3e3": {
        "family": "dthreeq_readout",
        "hidden_sizes": [500],
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "readout_lr": 1e-2,
        "input_residual_weight": 0.0,
        "infer_steps": 10,
    },
    "dthreeq_ep_tanh_nudge0p1_lr5e3_decay15": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "tanh",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 5e-3,
        "weight_lr_decay_epoch": 15,
        "weight_lr_decay_factor": 0.2,
        "infer_steps": 10,
    },
    "dthreeq_ep_tanh_nudge0p1_lr5e3_restorebest": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "tanh",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 5e-3,
        "restore_best": True,
        "infer_steps": 10,
    },
    "dthreeq_ep_tanh_nudge0p1_lr5e3_decay15_restorebest": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "tanh",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 5e-3,
        "weight_lr_decay_epoch": 15,
        "weight_lr_decay_factor": 0.2,
        "restore_best": True,
        "early_stop_patience": 8,
        "infer_steps": 10,
    },
    "dthreeq_ep_initact_tanh_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "tanh",
        "activate_initial": True,
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_linear_tanh_nudge0p1_lr1e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "tanh",
        "prediction_activation": "none",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_linear_tanh_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "tanh",
        "prediction_activation": "none",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_linear_clip01_nudge0p1_lr1e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "clip01",
        "state_min": 0.0,
        "prediction_activation": "none",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_linear_clip01_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "clip01",
        "state_min": 0.0,
        "prediction_activation": "none",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_sigmoid_nudge0p05_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "sigmoid",
        "loss_mode": "ep",
        "target_mode": "nudge_0.05",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_sigmoid_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "sigmoid",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_sigmoid_nudge0p1_lr5e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "sigmoid",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 5e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_sigmoid_nudge0p2_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "sigmoid",
        "loss_mode": "ep",
        "target_mode": "nudge_0.2",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_relu_nudge0p1_lr1e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "relu",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_relu_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "relu",
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_clip01_nudge0p05_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "clip01",
        "state_min": 0.0,
        "loss_mode": "ep",
        "target_mode": "nudge_0.05",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_clip01_nudge0p1_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "clip01",
        "state_min": 0.0,
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_clip01_nudge0p2_lr3e3": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "clip01",
        "state_min": 0.0,
        "loss_mode": "ep",
        "target_mode": "nudge_0.2",
        "beta_sign": "plus",
        "weight_lr": 3e-3,
        "infer_steps": 10,
    },
    "dthreeq_ep_clip01_nudge0p1_lr5e3_decay20": {
        "family": "dthreeq",
        "hidden_sizes": [500],
        "activation": "clip01",
        "state_min": 0.0,
        "loss_mode": "ep",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 5e-3,
        "weight_lr_decay_epoch": 20,
        "weight_lr_decay_factor": 0.2,
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
    "dthreeq_plus_energy_direct_lr1e3": {
        "family": "dthreeq_plus_energy",
        "hidden_sizes": [500],
        "loss_mode": "plus_energy",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_plus_energy_direct_lr3e4": {
        "family": "dthreeq_plus_energy",
        "hidden_sizes": [500],
        "loss_mode": "plus_energy",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "weight_lr": 3e-4,
        "infer_steps": 10,
    },
    "dthreeq_plus_energy_nudge0p1_lr1e3": {
        "family": "dthreeq_plus_energy",
        "hidden_sizes": [500],
        "loss_mode": "plus_energy",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_forward_target_direct_lr1e3": {
        "family": "dthreeq_forward_target",
        "hidden_sizes": [500],
        "loss_mode": "forward_target",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_forward_target_nudge0p1_lr1e3": {
        "family": "dthreeq_forward_target",
        "hidden_sizes": [500],
        "loss_mode": "forward_target",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
    },
    "dthreeq_bidir_target_direct_lr1e3": {
        "family": "dthreeq_bidir_target",
        "hidden_sizes": [500],
        "loss_mode": "bidir_target",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
        "weight_lr": 1e-3,
        "infer_steps": 10,
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
    restore_best = bool(config.get("restore_best", False))
    selected = best if restore_best else final
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
        "selected_epoch": int(selected.get("epoch", -1)),
        "selected_test_error": float(selected.get("test_error", float("nan"))),
        "selected_test_accuracy": 1.0
        - float(selected.get("test_error", float("nan"))),
        "final_state_test_error": float(
            final.get("state_test_error", final.get("test_error", float("nan")))
        ),
        "state_delta": float(final.get("state_delta", float("nan"))),
        "saturation": float(final.get("saturation", float("nan"))),
        "input_recon_energy": float(
            final.get("input_recon_energy", float("nan"))
        ),
        "input_recon_energy_frac": float(
            final.get("input_recon_energy_frac", float("nan"))
        ),
        "weighted_input_recon_energy_frac": float(
            final.get("weighted_input_recon_energy_frac", float("nan"))
        ),
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
        "state_min",
        "prediction_activation",
        "activate_initial",
        "loss_mode",
        "target_mode",
        "beta_sign",
        "objective_name",
        "weight_lr_decay_epoch",
        "weight_lr_decay_factor",
        "input_residual_weight",
        "label_encoding",
        "readout_lr",
        "restore_best",
        "early_stop_patience",
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


def _state_delta(
    states_free: List[torch.Tensor],
    clamped_states_all: List[List[torch.Tensor]],
) -> float:
    value = 0.0
    for states_clamped in clamped_states_all:
        value += sum(
            (c - f).abs().mean().item()
            for c, f in zip(states_clamped, states_free)
        ) / len(states_free)
    return float(value / max(1, len(clamped_states_all)))


def _common_dthreeq_batch_setup(
    model: DThreeQMLP,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    List[torch.Tensor],
    List[List[torch.Tensor]],
]:
    x = x.to(model.device, non_blocking=True).view(x.shape[0], -1).float()
    y = y.to(model.device, non_blocking=True).long()
    y_target = model.encode_labels(y)
    states0 = model.initial_states(x)
    states_free = model.relax(x, states0, clamped=False)
    layers_free = model.layers_from_states(x, [s.detach() for s in states_free])
    clamped_states_all = []
    layers_clamped_all = []
    for sign in model.signs():
        states_clamped = model.relax(
            x, states_free, y_one_hot=y_target, sign=sign, clamped=True
        )
        clamped_states_all.append(states_clamped)
        layers_clamped_all.append(
            model.layers_from_states(x, [s.detach() for s in states_clamped])
        )
    return (
        x,
        y,
        y_target,
        states_free,
        layers_free,
        layers_clamped_all,
        clamped_states_all,
    )


def _train_dthreeq_plus_energy_batch(
    model: DThreeQMLP,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    (
        _x,
        y,
        y_one_hot,
        states_free,
        layers_free,
        layers_clamped_all,
        clamped_states_all,
    ) = _common_dthreeq_batch_setup(model, x, y)
    objective = sum(model.energy(layers).mean() for layers in layers_clamped_all)
    objective = objective / max(1, len(layers_clamped_all))
    update_rel = model.update_weights(objective)
    with torch.no_grad():
        pred = states_free[-1].argmax(dim=1)
        energy_diag = model.energy_diagnostics(layers_free)
        return {
            "train_error": (pred != y).float().mean().item(),
            "train_energy": model.energy(layers_free).mean().item(),
            "train_cost": F.mse_loss(states_free[-1], y_one_hot).item(),
            "objective": objective.detach().item(),
            "state_delta": _state_delta(states_free, clamped_states_all),
            "saturation": model.state_saturation(states_free),
            "weight_abs_mean": model.weight_abs_mean(),
            "weight_update_rel_mean": update_rel,
            **energy_diag,
        }


def _target_objective(
    model: DThreeQMLP,
    layers_free: List[torch.Tensor],
    layers_clamped: List[torch.Tensor],
    include_backward: bool,
) -> torch.Tensor:
    forward_free, _ = model.edge_predictions(layers_free)
    terms = [
        F.mse_loss(pred, layers_clamped[i + 1].detach())
        for i, pred in enumerate(forward_free)
    ]
    if include_backward:
        _, backward_clamped = model.edge_predictions(layers_clamped)
        terms.extend(
            F.mse_loss(pred, layers_free[i].detach())
            for i, pred in enumerate(backward_clamped)
        )
    return sum(terms) / max(1, len(terms))


def _train_dthreeq_target_batch(
    model: DThreeQMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    include_backward: bool,
) -> Dict[str, float]:
    (
        _x,
        y,
        y_one_hot,
        states_free,
        layers_free,
        layers_clamped_all,
        clamped_states_all,
    ) = _common_dthreeq_batch_setup(model, x, y)
    objective = sum(
        _target_objective(model, layers_free, layers, include_backward)
        for layers in layers_clamped_all
    ) / max(1, len(layers_clamped_all))
    update_rel = model.update_weights(objective)
    with torch.no_grad():
        pred = states_free[-1].argmax(dim=1)
        energy_diag = model.energy_diagnostics(layers_free)
        return {
            "train_error": (pred != y).float().mean().item(),
            "train_energy": model.energy(layers_free).mean().item(),
            "train_cost": F.mse_loss(states_free[-1], y_one_hot).item(),
            "objective": objective.detach().item(),
            "state_delta": _state_delta(states_free, clamped_states_all),
            "saturation": model.state_saturation(states_free),
            "weight_abs_mean": model.weight_abs_mean(),
            "weight_update_rel_mean": update_rel,
            **energy_diag,
        }


def _train_dthreeq_readout_batch(
    model: DThreeQMLP,
    readout: nn.Module,
    readout_optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    (
        _x,
        y,
        y_target,
        states_free,
        layers_free,
        layers_clamped_all,
        clamped_states_all,
    ) = _common_dthreeq_batch_setup(model, x, y)
    objectives = []
    for sign, layers_clamped in zip(model.signs(), layers_clamped_all):
        if model.loss_mode == "dplus":
            objectives.append(model.dplus_objective(layers_free, layers_clamped))
        else:
            objectives.append(sign * model.ep_objective(layers_free, layers_clamped))
    objective = sum(objectives) / max(1, len(objectives))
    update_rel = model.update_weights(objective)

    logits = readout(states_free[-1].detach())
    readout_loss = F.cross_entropy(logits, y)
    readout_optimizer.zero_grad()
    readout_loss.backward()
    readout_optimizer.step()

    with torch.no_grad():
        state_pred = states_free[-1].argmax(dim=1)
        readout_pred = logits.detach().argmax(dim=1)
        energy_diag = model.energy_diagnostics(layers_free)
        return {
            "train_error": (readout_pred != y).float().mean().item(),
            "state_train_error": (state_pred != y).float().mean().item(),
            "train_energy": model.energy(layers_free).mean().item(),
            "train_cost": readout_loss.detach().item(),
            "state_train_cost": F.mse_loss(states_free[-1], y_target).item(),
            "objective": objective.detach().item(),
            "state_delta": _state_delta(states_free, clamped_states_all),
            "saturation": model.state_saturation(states_free),
            "weight_abs_mean": model.weight_abs_mean(),
            "weight_update_rel_mean": update_rel,
            **energy_diag,
        }


def _train_dthreeq_objective_epoch(
    model: DThreeQMLP,
    loader: Iterable,
    family: str,
) -> Dict[str, float]:
    if family == "dthreeq_plus_energy":
        return mean_metrics([_train_dthreeq_plus_energy_batch(model, x, y) for x, y in loader])
    if family == "dthreeq_forward_target":
        return mean_metrics(
            [_train_dthreeq_target_batch(model, x, y, False) for x, y in loader]
        )
    if family == "dthreeq_bidir_target":
        return mean_metrics(
            [_train_dthreeq_target_batch(model, x, y, True) for x, y in loader]
        )
    raise ValueError(f"unknown objective family: {family}")


def _eval_dthreeq(model: DThreeQMLP, loader: Iterable) -> Dict[str, float]:
    return mean_metrics([model.eval_batch(x, y) for x, y in loader])


def _eval_dthreeq_readout(
    model: DThreeQMLP, readout: nn.Module, loader: Iterable
) -> Dict[str, float]:
    rows = []
    for x, y in loader:
        x = x.to(model.device, non_blocking=True).view(x.shape[0], -1).float()
        y = y.to(model.device, non_blocking=True).long()
        y_target = model.encode_labels(y)
        states0 = model.initial_states(x)
        states_free = model.relax(x, states0, clamped=False)
        layers_free = model.layers_from_states(x, states_free)
        with torch.no_grad():
            logits = readout(states_free[-1])
            pred = logits.argmax(dim=1)
            state_pred = states_free[-1].argmax(dim=1)
            energy_diag = model.energy_diagnostics(layers_free)
            rows.append(
                {
                    "test_error": (pred != y).float().mean().item(),
                    "state_test_error": (state_pred != y).float().mean().item(),
                    "test_energy": model.energy(layers_free).mean().item(),
                    "test_cost": F.cross_entropy(logits, y).item(),
                    "state_test_cost": F.mse_loss(states_free[-1], y_target).item(),
                    "test_saturation": model.state_saturation(states_free),
                    **energy_diag,
                }
            )
    return mean_metrics(rows)


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
    objective_families = {
        "dthreeq_plus_energy",
        "dthreeq_forward_target",
        "dthreeq_bidir_target",
    }

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

    model_loss_mode = str(run_cfg.get("loss_mode", "dplus"))
    if variant["family"] in objective_families:
        model_loss_mode = "ep"

    model = DThreeQMLP(
        layer_sizes=[784] + hidden_sizes + [10],
        activation=str(run_cfg.get("activation", "tanh")),
        infer_steps=int(run_cfg.get("infer_steps", 10)),
        infer_lr=float(run_cfg.get("infer_lr", 0.05)),
        weight_lr=float(run_cfg.get("weight_lr", 1e-3)),
        target_mode=str(run_cfg["target_mode"]),
        beta_sign=str(run_cfg["beta_sign"]),
        loss_mode=model_loss_mode,
        bias=bool(run_cfg.get("bias", True)),
        state_clip=float(run_cfg.get("state_clip", 1.0)),
        state_min=(
            None
            if run_cfg.get("state_min") is None
            else float(run_cfg.get("state_min"))
        ),
        prediction_activation=str(run_cfg.get("prediction_activation", "post")),
        activate_initial=bool(run_cfg.get("activate_initial", False)),
        input_residual_weight=float(run_cfg.get("input_residual_weight", 1.0)),
        label_encoding=str(run_cfg.get("label_encoding", "onehot")),
        device=device,
    ).to(device)
    readout: nn.Module | None = None
    readout_optimizer: torch.optim.Optimizer | None = None
    if variant["family"] == "dthreeq_readout":
        readout = nn.Linear(10, 10).to(device)
        readout_optimizer = torch.optim.Adam(
            readout.parameters(), lr=float(run_cfg.get("readout_lr", 1e-2))
        )
    base_weight_lr = float(run_cfg.get("weight_lr", 1e-3))
    decay_epoch = run_cfg.get("weight_lr_decay_epoch")
    decay_factor = float(run_cfg.get("weight_lr_decay_factor", 1.0))
    best_test_error = float("inf")
    epochs_since_improvement = 0
    early_stop_patience = run_cfg.get("early_stop_patience")
    for epoch in range(n_epochs):
        if decay_epoch is not None and epoch >= int(decay_epoch):
            model.weight_lr = base_weight_lr * decay_factor
        else:
            model.weight_lr = base_weight_lr
        if variant["family"] in objective_families:
            train_metrics = _train_dthreeq_objective_epoch(
                model, train_loader, str(variant["family"])
            )
        elif variant["family"] == "dthreeq_readout":
            if readout is None or readout_optimizer is None:
                raise RuntimeError("readout family requires a readout module")
            train_metrics = mean_metrics(
                [
                    _train_dthreeq_readout_batch(
                        model, readout, readout_optimizer, x, y
                    )
                    for x, y in train_loader
                ]
            )
        else:
            train_metrics = _train_dthreeq_epoch(model, train_loader, run_cfg, variant)
        if variant["family"] == "dthreeq_readout":
            if readout is None:
                raise RuntimeError("readout family requires a readout module")
            test_metrics = _eval_dthreeq_readout(model, readout, test_loader)
        else:
            test_metrics = _eval_dthreeq(model, test_loader)
        _record_curve(curves, epoch, train_metrics, test_metrics)
        current_error = float(test_metrics.get("test_error", float("inf")))
        if current_error < best_test_error:
            best_test_error = current_error
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if (
            early_stop_patience is not None
            and epochs_since_improvement >= int(early_stop_patience)
        ):
            break
    return _summarize_result(run_cfg, variant_name, variant, curves, start, device)


def write_focus_curve_rows(raw_csv: Path, curves_csv: Path) -> None:
    write_curve_rows(raw_csv, curves_csv)
