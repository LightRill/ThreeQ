from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rho(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def rho_prime(x: torch.Tensor) -> torch.Tensor:
    return ((x > 0.0) & (x < 1.0)).to(dtype=x.dtype)


class BidirectionalMLP(nn.Module):
    """Shared implementation for Base3Q-style layered MLP experiments."""

    def __init__(
        self,
        layer_sizes: List[int],
        alphas: List[float],
        beta: float,
        free_steps: int,
        weak_steps: int,
        epsilon: float,
        weight_update: str = "direct",
        weak_mode: str = "cost",
        transpose_tied: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_sizes = list(layer_sizes)
        self.alphas = [float(a) for a in alphas]
        self.beta = float(beta)
        self.free_steps = int(free_steps)
        self.weak_steps = int(weak_steps)
        self.epsilon = float(epsilon)
        self.weight_update = weight_update
        self.weak_mode = weak_mode
        self.transpose_tied = bool(transpose_tied)
        self.device = device
        self.N = len(self.layer_sizes)
        if len(self.alphas) != self.N - 1:
            raise ValueError(f"alphas length {len(self.alphas)} must be {self.N - 1}")
        if self.weight_update not in {"direct", "ep"}:
            raise ValueError("weight_update must be 'direct' or 'ep'")
        if self.weak_mode not in {"cost", "linear_clamp"}:
            raise ValueError("weak_mode must be 'cost' or 'linear_clamp'")

        self.forward_weights = nn.ParameterList()
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            w = torch.empty(n_in, n_out, device=self.device)
            bound = math.sqrt(6.0 / (n_in + n_out))
            with torch.no_grad():
                w.uniform_(-bound, bound)
            self.forward_weights.append(nn.Parameter(w))

        self.backward_weights = nn.ParameterList()
        if not self.transpose_tied:
            for n_in, n_out in zip(self.layer_sizes[1:], self.layer_sizes[:-1]):
                w = torch.empty(n_in, n_out, device=self.device)
                bound = math.sqrt(6.0 / (n_in + n_out))
                with torch.no_grad():
                    w.uniform_(-bound, bound)
                self.backward_weights.append(nn.Parameter(w))

    def _backward_weight(self, conn: int) -> torch.Tensor:
        if self.transpose_tied:
            return self.forward_weights[conn].t()
        return self.backward_weights[conn]

    def energy(self, layers: List[torch.Tensor]) -> torch.Tensor:
        batch_size = layers[0].shape[0]
        energy = torch.zeros(batch_size, device=self.device)
        for i in range(self.N):
            pred = torch.zeros(batch_size, self.layer_sizes[i], device=self.device)
            n_sources = 0
            if i > 0:
                pred = pred + rho(layers[i - 1].detach()) @ self.forward_weights[i - 1]
                n_sources += 1
            if i < self.N - 1:
                pred = pred + rho(layers[i + 1].detach()) @ self._backward_weight(i)
                n_sources += 1
            if n_sources > 1:
                pred = pred / float(n_sources)
            energy = energy + 0.5 * ((pred - layers[i]) ** 2).sum(dim=1)
        return energy

    def cost(self, layers: List[torch.Tensor], y_one_hot: torch.Tensor) -> torch.Tensor:
        return ((layers[-1] - y_one_hot) ** 2).sum(dim=1)

    def total_energy(
        self, layers: List[torch.Tensor], y_one_hot: torch.Tensor, beta: float
    ) -> torch.Tensor:
        if self.weak_mode == "linear_clamp":
            output = (1.0 - beta) * layers[-1] + beta * y_one_hot
            return self.energy(layers[:-1] + [output])
        return self.energy(layers) + beta * self.cost(layers, y_one_hot)

    @torch.no_grad()
    def _clip_states_(self, states: List[torch.Tensor]) -> None:
        for state in states:
            state.clamp_(0.0, 1.0)

    def relax_states(
        self,
        x: torch.Tensor,
        states_init: List[torch.Tensor],
        y_one_hot: Optional[torch.Tensor],
        beta: float,
        n_steps: int,
        epsilon: float,
    ) -> List[torch.Tensor]:
        states = [s.detach().clone().requires_grad_(True) for s in states_init]
        layers = [x] + states
        optimizer = torch.optim.SGD(states, lr=epsilon)
        for _ in range(int(n_steps)):
            with torch.enable_grad():
                if beta == 0.0:
                    objective = self.energy(layers).sum()
                else:
                    if y_one_hot is None:
                        raise ValueError("y_one_hot is required when beta != 0")
                    objective = self.total_energy(layers, y_one_hot, beta).sum()
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            self._clip_states_(states)
        return states

    def _weight_parameters_with_alphas(self) -> Iterable[Tuple[nn.Parameter, float]]:
        for i, w in enumerate(self.forward_weights):
            yield w, self.alphas[i]
        if not self.transpose_tied:
            for i, w in enumerate(self.backward_weights):
                yield w, self.alphas[i]

    def _zero_weight_grads(self) -> None:
        for w, _ in self._weight_parameters_with_alphas():
            if w.grad is not None:
                w.grad.zero_()

    @torch.no_grad()
    def batch_spectral_radius(self, states: List[torch.Tensor]) -> float:
        state_sizes = self.layer_sizes[1:]
        offsets = [0]
        for size in state_sizes:
            offsets.append(offsets[-1] + size)
        total_dim = offsets[-1]
        jac = torch.zeros(total_dim, total_dim, device=self.device)

        def state_slice(layer_idx: int) -> slice:
            state_idx = layer_idx - 1
            return slice(offsets[state_idx], offsets[state_idx + 1])

        representative = [s.mean(dim=0) for s in states]
        for target_layer in range(1, self.N):
            row = state_slice(target_layer)
            n_sources = int(target_layer > 0) + int(target_layer < self.N - 1)
            scale = 1.0 / float(n_sources)
            if target_layer > 1:
                prev_layer = target_layer - 1
                prev = state_slice(prev_layer)
                prime = rho_prime(representative[prev_layer - 1])
                w = self.forward_weights[target_layer - 1]
                jac[row, prev] = w.t() * prime.unsqueeze(0) * scale
            if target_layer < self.N - 1:
                next_layer = target_layer + 1
                nxt = state_slice(next_layer)
                prime = rho_prime(representative[next_layer - 1])
                if self.transpose_tied:
                    block = self.forward_weights[target_layer] * prime.unsqueeze(0)
                else:
                    block = self.backward_weights[target_layer].t() * prime.unsqueeze(0)
                jac[row, nxt] = block * scale

        eigvals = torch.linalg.eigvals(jac)
        return float(torch.abs(eigvals).max().item())

    @torch.no_grad()
    def state_saturation(self, states: List[torch.Tensor]) -> float:
        saturated = 0.0
        total = 0
        for state in states:
            saturated += ((state <= 0.0) | (state >= 1.0)).float().sum().item()
            total += state.numel()
        return float(saturated / max(1, total))

    @torch.no_grad()
    def weight_norm_mean(self) -> float:
        vals = [w.abs().mean().item() for w, _ in self._weight_parameters_with_alphas()]
        return float(sum(vals) / max(1, len(vals)))

    def train_batch(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        batch_size = x.shape[0]
        x = x.to(self.device, non_blocking=True).view(batch_size, -1).float()
        y = y.to(self.device, non_blocking=True).long()
        y_one_hot = F.one_hot(y, num_classes=self.layer_sizes[-1]).float()

        states0 = [
            torch.zeros(batch_size, dim, device=self.device, dtype=torch.float32)
            for dim in self.layer_sizes[1:]
        ]
        states_free = self.relax_states(
            x, states0, None, 0.0, self.free_steps, self.epsilon
        )
        states_beta = self.relax_states(
            x, states_free, y_one_hot, self.beta, self.weak_steps, self.epsilon
        )

        with torch.no_grad():
            layers_free = [x] + states_free
            layers_beta = [x] + states_beta
            y_pred = torch.argmax(states_free[-1], dim=1)
            metrics = {
                "train_energy": self.energy(layers_free).mean().item(),
                "train_cost": self.cost(layers_free, y_one_hot).mean().item(),
                "train_error": (y_pred != y).float().mean().item(),
                "weak_cost": self.cost(layers_beta, y_one_hot).mean().item(),
                "rho": self.batch_spectral_radius(states_free),
                "saturation": self.state_saturation(states_free),
                "weight_abs_mean": self.weight_norm_mean(),
            }
            for idx, (s0, sb) in enumerate(zip(states_free, states_beta), start=1):
                metrics[f"state_delta_l{idx}"] = (sb - s0).abs().mean().item()

        layers_free_det = [x] + [s.detach() for s in states_free]
        layers_beta_det = [x] + [s.detach() for s in states_beta]
        if self.weight_update == "ep":
            objective = (
                self.energy(layers_beta_det).mean()
                - self.energy(layers_free_det).mean()
            ) / self.beta
        else:
            objective = self.total_energy(layers_beta_det, y_one_hot, self.beta).mean()

        self._zero_weight_grads()
        objective.backward()
        with torch.no_grad():
            update_logs = []
            for w, alpha in self._weight_parameters_with_alphas():
                if w.grad is None:
                    update_logs.append(0.0)
                    continue
                w_abs = w.abs().mean().item()
                delta = -float(alpha) * w.grad
                w.add_(delta)
                update_logs.append(delta.abs().mean().item() / (w_abs + 1e-8))
            metrics["weight_update_rel_mean"] = float(
                sum(update_logs) / max(1, len(update_logs))
            )
        return metrics

    def eval_batch(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        batch_size = x.shape[0]
        x = x.to(self.device, non_blocking=True).view(batch_size, -1).float()
        y = y.to(self.device, non_blocking=True).long()
        y_one_hot = F.one_hot(y, num_classes=self.layer_sizes[-1]).float()
        states0 = [
            torch.zeros(batch_size, dim, device=self.device, dtype=torch.float32)
            for dim in self.layer_sizes[1:]
        ]
        states_free = self.relax_states(
            x, states0, None, 0.0, self.free_steps, self.epsilon
        )
        with torch.no_grad():
            layers_free = [x] + states_free
            y_pred = torch.argmax(states_free[-1], dim=1)
            return {
                "valid_energy": self.energy(layers_free).mean().item(),
                "valid_cost": self.cost(layers_free, y_one_hot).mean().item(),
                "valid_error": (y_pred != y).float().mean().item(),
                "valid_saturation": self.state_saturation(states_free),
            }
