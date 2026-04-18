from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .training import make_twomoons, mean_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def activation_fn(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "tanh":
        return torch.tanh(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name in {"clip01", "hard_sigmoid"}:
        return torch.clamp(x, 0.0, 1.0)
    if name == "relu":
        return F.relu(x)
    if name == "identity":
        return x
    raise ValueError(f"unknown activation: {name}")


@dataclass(frozen=True)
class TargetMode:
    name: str
    kind: str
    beta: float = 0.0
    tau: int = 0
    gamma: float = 0.0


def parse_target_mode(name: str) -> TargetMode:
    if name == "direct_clamp":
        return TargetMode(name=name, kind="direct")
    if name.startswith("ce_nudge_"):
        return TargetMode(
            name=name, kind="ce_nudge", beta=float(name.rsplit("_", 1)[1])
        )
    if name.startswith("nudge_"):
        return TargetMode(name=name, kind="nudge", beta=float(name.split("_", 1)[1]))
    if name.startswith("gradual_clamp_"):
        _, _, tau, gamma = name.split("_")
        return TargetMode(name=name, kind="gradual", tau=int(tau), gamma=float(gamma))
    raise ValueError(f"unknown target mode: {name}")


def build_twomoons_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    n_samples = int(config.get("n_samples", 2000))
    noise = float(config.get("noise", 0.1))
    seed = int(config.get("seed", 0))
    batch_size = int(config.get("batch_size", 32))
    test_size = float(config.get("test_size", 0.2))
    X, y = make_twomoons(n_samples=n_samples, noise=noise, random_state=42 + seed)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    n_train = int((1.0 - test_size) * n_samples)
    train_ds = TensorDataset(
        torch.tensor(X[:n_train], dtype=torch.float32),
        torch.tensor(y[:n_train], dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X[n_train:], dtype=torch.float32),
        torch.tensor(y[n_train:], dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class BPMLP(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.activation = activation
        layers: List[nn.Module] = []
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(n_in, n_out, bias=bias))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh() if activation == "tanh" else nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DThreeQMLP(nn.Module):
    """Bidirectional local predictor used for DThreeQ feasibility experiments."""

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str,
        infer_steps: int,
        infer_lr: float,
        weight_lr: float,
        target_mode: str,
        beta_sign: str,
        loss_mode: str,
        bias: bool = True,
        state_clip: float = 1.0,
        state_min: float | None = None,
        prediction_activation: str = "post",
        activate_initial: bool = False,
        input_residual_weight: float = 1.0,
        label_encoding: str = "onehot",
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_sizes = list(layer_sizes)
        self.activation = activation
        self.infer_steps = int(infer_steps)
        self.infer_lr = float(infer_lr)
        self.weight_lr = float(weight_lr)
        self.target_mode = parse_target_mode(target_mode)
        self.beta_sign = beta_sign
        self.loss_mode = loss_mode
        self.bias = bool(bias)
        self.state_clip = float(state_clip)
        self.state_min = -self.state_clip if state_min is None else float(state_min)
        self.state_max = self.state_clip
        self.prediction_activation = prediction_activation
        self.activate_initial = bool(activate_initial)
        self.input_residual_weight = float(input_residual_weight)
        self.label_encoding = label_encoding
        self.device = device
        self.n_layers = len(self.layer_sizes)
        if self.beta_sign not in {"plus", "minus", "plusminus"}:
            raise ValueError("beta_sign must be plus, minus, or plusminus")
        if self.loss_mode not in {"dplus", "ep"}:
            raise ValueError("loss_mode must be dplus or ep")
        if self.prediction_activation not in {"post", "none"}:
            raise ValueError("prediction_activation must be post or none")
        if self.label_encoding not in {"onehot", "signed"}:
            raise ValueError("label_encoding must be onehot or signed")

        self.forward_weights = nn.ParameterList()
        self.backward_weights = nn.ParameterList()
        self.forward_biases = nn.ParameterList()
        self.backward_biases = nn.ParameterList()
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            bound = math.sqrt(6.0 / (n_in + n_out))
            wf = torch.empty(n_in, n_out, device=device).uniform_(-bound, bound)
            wb = torch.empty(n_out, n_in, device=device).uniform_(-bound, bound)
            self.forward_weights.append(nn.Parameter(wf))
            self.backward_weights.append(nn.Parameter(wb))
            self.forward_biases.append(nn.Parameter(torch.zeros(n_out, device=device)))
            self.backward_biases.append(nn.Parameter(torch.zeros(n_in, device=device)))

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        return activation_fn(self.activation, x)

    def _clip_states_(self, states: List[torch.Tensor]) -> None:
        if self.state_clip <= 0:
            return
        with torch.no_grad():
            for state in states:
                state.clamp_(self.state_min, self.state_max)

    def initial_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        states: List[torch.Tensor] = []
        current = x
        for i, w in enumerate(self.forward_weights):
            current = self._act(current) @ w
            if self.bias:
                current = current + self.forward_biases[i]
            if self.activate_initial:
                current = self._act(current)
            current = current.clamp(self.state_min, self.state_max)
            states.append(current.detach())
        return states

    def layers_from_states(
        self, x: torch.Tensor, states: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        return [x] + states

    def edge_predictions(
        self, layers: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        forward_preds: List[torch.Tensor] = []
        backward_preds: List[torch.Tensor] = []
        for i in range(self.n_layers - 1):
            vf = self._act(layers[i]) @ self.forward_weights[i]
            vb = self._act(layers[i + 1]) @ self.backward_weights[i]
            if self.bias:
                vf = vf + self.forward_biases[i]
                vb = vb + self.backward_biases[i]
            if self.prediction_activation == "post":
                vf = self._act(vf)
                vb = self._act(vb)
            forward_preds.append(vf)
            backward_preds.append(vb)
        return forward_preds, backward_preds

    def residuals(self, layers: List[torch.Tensor]) -> List[torch.Tensor]:
        forward_preds, backward_preds = self.edge_predictions(layers)
        residuals: List[torch.Tensor] = []
        for i, (vf, vb) in enumerate(zip(forward_preds, backward_preds)):
            residuals.append(vf - layers[i + 1])
            residuals.append(vb - layers[i])
        return residuals

    def residual_weights(self) -> List[float]:
        weights: List[float] = []
        for i in range(self.n_layers - 1):
            weights.append(1.0)
            weights.append(self.input_residual_weight if i == 0 else 1.0)
        return weights

    def encode_labels(self, y: torch.Tensor) -> torch.Tensor:
        y_one_hot = F.one_hot(y, num_classes=self.layer_sizes[-1]).float()
        if self.label_encoding == "signed":
            return 2.0 * y_one_hot - 1.0
        return y_one_hot

    def energy(self, layers: List[torch.Tensor]) -> torch.Tensor:
        energy = torch.zeros(layers[0].shape[0], device=self.device)
        for residual, weight in zip(self.residuals(layers), self.residual_weights()):
            energy = energy + weight * 0.5 * residual.pow(2).sum(dim=1)
        return energy

    @torch.no_grad()
    def energy_diagnostics(self, layers: List[torch.Tensor]) -> Dict[str, float]:
        raw_terms = [
            0.5 * residual.pow(2).sum(dim=1).mean().item()
            for residual in self.residuals(layers)
        ]
        weights = self.residual_weights()
        weighted_terms = [term * weight for term, weight in zip(raw_terms, weights)]
        raw_total = float(sum(raw_terms))
        weighted_total = float(sum(weighted_terms))
        input_recon = float(raw_terms[1]) if len(raw_terms) > 1 else float("nan")
        weighted_input_recon = (
            float(weighted_terms[1]) if len(weighted_terms) > 1 else float("nan")
        )
        return {
            "input_recon_energy": input_recon,
            "input_recon_energy_frac": input_recon / (raw_total + 1e-12),
            "weighted_input_recon_energy_frac": weighted_input_recon
            / (weighted_total + 1e-12),
        }

    def apply_target(
        self,
        output: torch.Tensor,
        y_target: torch.Tensor,
        sign: float,
        step: int,
    ) -> torch.Tensor:
        mode = self.target_mode
        if mode.kind == "direct":
            target = y_target if sign >= 0 else -y_target
            return target
        if mode.kind == "ce_nudge":
            signed_beta = sign * mode.beta
            ce_grad = F.softmax(output, dim=1) - y_target
            return output - signed_beta * ce_grad
        if mode.kind == "nudge":
            signed_beta = sign * mode.beta
            return (1.0 - signed_beta) * output + signed_beta * y_target
        if mode.kind == "gradual":
            signed_gamma = sign * mode.gamma
            return (1.0 - signed_gamma) * output + signed_gamma * y_target
        raise ValueError(f"unknown target mode kind: {mode.kind}")

    def relax(
        self,
        x: torch.Tensor,
        states_init: List[torch.Tensor],
        y_one_hot: torch.Tensor | None = None,
        sign: float = 1.0,
        clamped: bool = False,
    ) -> List[torch.Tensor]:
        states = [s.detach().clone().requires_grad_(True) for s in states_init]
        if clamped:
            states[-1] = self.apply_target(states[-1], y_one_hot, sign, 0).detach()
            states[-1].requires_grad_(self.target_mode.kind != "direct")
        optimizer = torch.optim.SGD(states, lr=self.infer_lr)
        steps = self.target_mode.tau if clamped and self.target_mode.kind == "gradual" else self.infer_steps
        steps = max(1, int(steps))
        for step in range(steps):
            layers = self.layers_from_states(x, states)
            objective = self.energy(layers).mean()
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            if clamped:
                with torch.no_grad():
                    states[-1].copy_(self.apply_target(states[-1], y_one_hot, sign, step))
            self._clip_states_(states)
        return [s.detach() for s in states]

    def signs(self) -> List[float]:
        if self.beta_sign == "plus":
            return [1.0]
        if self.beta_sign == "minus":
            return [-1.0]
        return [1.0, -1.0]

    def dplus_objective(
        self,
        layers_free: List[torch.Tensor],
        layers_clamped: List[torch.Tensor],
    ) -> torch.Tensor:
        residuals_free = [r.detach() for r in self.residuals(layers_free)]
        residuals_clamped = self.residuals(layers_clamped)
        terms = []
        for r_clamped, r_free, weight in zip(
            residuals_clamped, residuals_free, self.residual_weights()
        ):
            terms.append(weight * 0.5 * (r_clamped - r_free).pow(2).mean())
        return sum(terms) / max(1, len(terms))

    def ep_objective(
        self,
        layers_free: List[torch.Tensor],
        layers_clamped: List[torch.Tensor],
    ) -> torch.Tensor:
        beta_scale = 1.0
        if self.target_mode.kind in {"nudge", "ce_nudge"}:
            beta_scale = max(abs(self.target_mode.beta), 1e-6)
        if self.target_mode.kind == "gradual":
            beta_scale = max(abs(self.target_mode.gamma), 1e-6)
        return (self.energy(layers_clamped).mean() - self.energy(layers_free).mean()) / beta_scale

    def update_weights(self, objective: torch.Tensor) -> float:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()
        objective.backward()
        updates = []
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is None:
                    continue
                base = param.abs().mean().item()
                delta = -self.weight_lr * param.grad
                param.add_(delta)
                updates.append(delta.abs().mean().item() / (base + 1e-8))
        return float(sum(updates) / max(1, len(updates)))

    @torch.no_grad()
    def state_saturation(self, states: List[torch.Tensor]) -> float:
        if self.state_clip <= 0:
            return 0.0
        saturated = 0.0
        total = 0
        upper_threshold = self.state_min + 0.98 * (self.state_max - self.state_min)
        lower_threshold = self.state_min + 0.02 * (self.state_max - self.state_min)
        for state in states:
            saturated += (
                (state <= lower_threshold) | (state >= upper_threshold)
            ).float().sum().item()
            total += state.numel()
        return float(saturated / max(1, total))

    @torch.no_grad()
    def weight_abs_mean(self) -> float:
        vals = [p.abs().mean().item() for p in self.parameters()]
        return float(sum(vals) / max(1, len(vals)))

    def train_batch(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        x = x.to(self.device, non_blocking=True).view(x.shape[0], -1).float()
        y = y.to(self.device, non_blocking=True).long()
        y_target = self.encode_labels(y)
        states0 = self.initial_states(x)
        states_free = self.relax(x, states0, clamped=False)
        layers_free = self.layers_from_states(x, [s.detach() for s in states_free])
        objectives = []
        clamped_states_all = []
        for sign in self.signs():
            states_clamped = self.relax(
                x, states_free, y_one_hot=y_target, sign=sign, clamped=True
            )
            layers_clamped = self.layers_from_states(
                x, [s.detach() for s in states_clamped]
            )
            if self.loss_mode == "dplus":
                objectives.append(self.dplus_objective(layers_free, layers_clamped))
            else:
                objectives.append(sign * self.ep_objective(layers_free, layers_clamped))
            clamped_states_all.append(states_clamped)
        objective = sum(objectives) / max(1, len(objectives))
        update_rel = self.update_weights(objective)
        with torch.no_grad():
            y_pred = torch.argmax(states_free[-1], dim=1)
            energy_diag = self.energy_diagnostics(layers_free)
            state_delta = 0.0
            for states_clamped in clamped_states_all:
                state_delta += sum(
                    (c - f).abs().mean().item()
                    for c, f in zip(states_clamped, states_free)
                ) / len(states_free)
            state_delta /= max(1, len(clamped_states_all))
            return {
                "train_error": (y_pred != y).float().mean().item(),
                "train_energy": self.energy(layers_free).mean().item(),
                "train_cost": F.mse_loss(states_free[-1], y_target).item(),
                "objective": objective.detach().item(),
                "state_delta": float(state_delta),
                "saturation": self.state_saturation(states_free),
                "weight_abs_mean": self.weight_abs_mean(),
                "weight_update_rel_mean": update_rel,
                **energy_diag,
            }

    def eval_batch(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        x = x.to(self.device, non_blocking=True).view(x.shape[0], -1).float()
        y = y.to(self.device, non_blocking=True).long()
        y_target = self.encode_labels(y)
        states0 = self.initial_states(x)
        states_free = self.relax(x, states0, clamped=False)
        layers_free = self.layers_from_states(x, states_free)
        with torch.no_grad():
            y_pred = torch.argmax(states_free[-1], dim=1)
            energy_diag = self.energy_diagnostics(layers_free)
            return {
                "test_error": (y_pred != y).float().mean().item(),
                "test_energy": self.energy(layers_free).mean().item(),
                "test_cost": F.mse_loss(states_free[-1], y_target).item(),
                "test_saturation": self.state_saturation(states_free),
                **energy_diag,
            }


def train_bp(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    config = dict(config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_sizes = list(config.get("hidden_sizes", [128, 64]))
    layer_sizes = [2] + hidden_sizes + [2]
    train_loader, test_loader = build_twomoons_loaders(config)
    model = BPMLP(
        layer_sizes=layer_sizes,
        activation=str(config.get("activation", "tanh")),
        bias=bool(config.get("bias", True)),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config.get("weight_lr", 1e-3)))
    n_epochs = int(config.get("n_epochs", 30))
    best_test_error = float("inf")
    best_epoch = -1
    last_train: Dict[str, float] = {}
    last_test: Dict[str, float] = {}
    for epoch in range(n_epochs):
        train_items = []
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).view(x.shape[0], -1).float()
            y = y.to(device, non_blocking=True).long()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(logits.detach(), dim=1)
            train_items.append(
                {
                    "train_error": (pred != y).float().mean().item(),
                    "train_cost": loss.detach().item(),
                }
            )
        test_items = []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device, non_blocking=True).view(x.shape[0], -1).float()
                y = y.to(device, non_blocking=True).long()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                pred = torch.argmax(logits, dim=1)
                test_items.append(
                    {
                        "test_error": (pred != y).float().mean().item(),
                        "test_cost": loss.item(),
                    }
                )
        last_train = mean_metrics(train_items)
        last_test = mean_metrics(test_items)
        if last_test["test_error"] < best_test_error:
            best_test_error = last_test["test_error"]
            best_epoch = epoch
    return {
        "variant": str(config.get("variant", "bp")),
        "method": "bp",
        "seed": seed,
        "dataset": "twomoons",
        "best_epoch": int(best_epoch),
        "best_test_error": float(best_test_error),
        "final_test_error": float(last_test.get("test_error", float("nan"))),
        "final_test_cost": float(last_test.get("test_cost", float("nan"))),
        "final_train_error": float(last_train.get("train_error", float("nan"))),
        "final_train_cost": float(last_train.get("train_cost", float("nan"))),
        "duration_sec": float(time.perf_counter() - start),
        "device": str(device),
        "weight_lr": float(config.get("weight_lr", 1e-3)),
    }


def train_dthreeq(config: Dict[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    config = dict(config)
    seed = int(config.get("seed", 0))
    set_seed(seed)
    variant = DTHREEQ_VARIANTS[str(config["variant"])]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_sizes = list(config.get("hidden_sizes", [128, 64]))
    layer_sizes = [2] + hidden_sizes + [2]
    train_loader, test_loader = build_twomoons_loaders(config)
    model = DThreeQMLP(
        layer_sizes=layer_sizes,
        activation=str(config.get("activation", variant.get("activation", "tanh"))),
        infer_steps=int(config.get("infer_steps", 15)),
        infer_lr=float(config.get("infer_lr", 0.05)),
        weight_lr=float(config.get("weight_lr", 1e-4)),
        target_mode=str(variant["target_mode"]),
        beta_sign=str(variant["beta_sign"]),
        loss_mode=str(variant["loss_mode"]),
        bias=bool(config.get("bias", True)),
        state_clip=float(config.get("state_clip", 1.0)),
        device=device,
    ).to(device)
    n_epochs = int(config.get("n_epochs", 30))
    best_test_error = float("inf")
    best_epoch = -1
    last_train: Dict[str, float] = {}
    last_test: Dict[str, float] = {}
    for epoch in range(n_epochs):
        train_items = []
        for x, y in train_loader:
            train_items.append(model.train_batch(x, y))
        test_items = []
        for x, y in test_loader:
            test_items.append(model.eval_batch(x, y))
        last_train = mean_metrics(train_items)
        last_test = mean_metrics(test_items)
        if last_test["test_error"] < best_test_error:
            best_test_error = last_test["test_error"]
            best_epoch = epoch
    result: Dict[str, Any] = {
        "variant": str(config["variant"]),
        "method": str(variant["loss_mode"]),
        "target_mode": str(variant["target_mode"]),
        "beta_sign": str(variant["beta_sign"]),
        "seed": seed,
        "dataset": "twomoons",
        "n_epochs": n_epochs,
        "infer_steps": int(config.get("infer_steps", 15)),
        "infer_lr": float(config.get("infer_lr", 0.05)),
        "weight_lr": float(config.get("weight_lr", 1e-4)),
        "activation": str(config.get("activation", variant.get("activation", "tanh"))),
        "best_epoch": int(best_epoch),
        "best_test_error": float(best_test_error),
        "final_test_error": float(last_test.get("test_error", float("nan"))),
        "final_test_cost": float(last_test.get("test_cost", float("nan"))),
        "final_test_energy": float(last_test.get("test_energy", float("nan"))),
        "final_train_error": float(last_train.get("train_error", float("nan"))),
        "final_train_cost": float(last_train.get("train_cost", float("nan"))),
        "final_train_energy": float(last_train.get("train_energy", float("nan"))),
        "objective": float(last_train.get("objective", float("nan"))),
        "state_delta": float(last_train.get("state_delta", float("nan"))),
        "saturation": float(last_train.get("saturation", float("nan"))),
        "weight_abs_mean": float(last_train.get("weight_abs_mean", float("nan"))),
        "weight_update_rel_mean": float(
            last_train.get("weight_update_rel_mean", float("nan"))
        ),
        "duration_sec": float(time.perf_counter() - start),
        "device": str(device),
    }
    return result


DTHREEQ_VARIANTS: Dict[str, Dict[str, str]] = {
    "bp_tanh": {
        "loss_mode": "bp",
        "target_mode": "none",
        "beta_sign": "none",
        "activation": "tanh",
    },
    "dplus_direct": {
        "loss_mode": "dplus",
        "target_mode": "direct_clamp",
        "beta_sign": "plus",
    },
    "dplus_nudge_0p1_plus": {
        "loss_mode": "dplus",
        "target_mode": "nudge_0.1",
        "beta_sign": "plus",
    },
    "dplus_nudge_0p01_plus": {
        "loss_mode": "dplus",
        "target_mode": "nudge_0.01",
        "beta_sign": "plus",
    },
    "dplus_nudge_0p001_plus": {
        "loss_mode": "dplus",
        "target_mode": "nudge_0.001",
        "beta_sign": "plus",
    },
    "dplus_nudge_0p01_plusminus": {
        "loss_mode": "dplus",
        "target_mode": "nudge_0.01",
        "beta_sign": "plusminus",
    },
    "dplus_gradual_100_0p01_plus": {
        "loss_mode": "dplus",
        "target_mode": "gradual_clamp_100_0.01",
        "beta_sign": "plus",
    },
    "ep_nudge_0p01_plus": {
        "loss_mode": "ep",
        "target_mode": "nudge_0.01",
        "beta_sign": "plus",
    },
    "ep_gradual_100_0p01_plus": {
        "loss_mode": "ep",
        "target_mode": "gradual_clamp_100_0.01",
        "beta_sign": "plus",
    },
}


def train_one_dthreeq(config: Dict[str, Any]) -> Dict[str, Any]:
    variant = str(config["variant"])
    if DTHREEQ_VARIANTS[variant]["loss_mode"] == "bp":
        return train_bp(config)
    return train_dthreeq(config)
