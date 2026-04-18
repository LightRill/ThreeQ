import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 激活函数
def rho(s: torch.Tensor) -> torch.Tensor:
    """Hard-sigmoid / firing-rate nonlinearity: clip to [0, 1]."""
    return torch.clamp(s, 0.0, 1.0)


class Network(nn.Module):
    def __init__(
        self,
        alphas: List[float],
        beta: float,
        free_steps: int,
        weak_steps: int,
        epsilon: float,
        layer_sizes: List[int],
        device=device,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.alphas = alphas
        self.beta = beta
        self.free_steps = free_steps
        self.weak_steps = weak_steps
        self.epsilon = epsilon
        self.N = len(layer_sizes)
        self.device = device
        if len(self.alphas) != self.N - 1:
            raise ValueError(
                f"alphas length ({len(self.alphas)}) must be N-1 ({self.N - 1})"
            )

        # 权重初始化
        def xavier_uniform_(W: torch.Tensor, n_in: int, n_out: int):
            bound = math.sqrt(6.0 / (n_in + n_out))
            with torch.no_grad():
                W.uniform_(-bound, bound)

        # x[i] --- fw[i] ---> x[i+1]
        self.forward_weights = nn.ParameterList([])
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            W = torch.empty(n_in, n_out, device=self.device)
            xavier_uniform_(W, n_in, n_out)
            self.forward_weights.append(nn.Parameter(W))

        # x[i] <--- bw[i] --- x[i+1]
        self.backward_weights = nn.ParameterList([])
        for n_in, n_out in zip(self.layer_sizes[1:], self.layer_sizes[:-1]):
            W = torch.empty(n_in, n_out, device=self.device)
            xavier_uniform_(W, n_in, n_out)
            self.backward_weights.append(nn.Parameter(W))

    def energy(self, layers: List[torch.Tensor]) -> torch.Tensor:
        """
        layers: [x, h1, h2, ..., y]，每个张量 shape=(B, dim)
        返回：每个样本的能量 shape=(B,)
        """
        B = layers[0].shape[0]

        energy = torch.zeros(B, device=self.device)

        for i in range(self.N):
            predict = torch.zeros(B, self.layer_sizes[i], device=self.device)
            size = 0
            if i != 0:
                predict = predict + torch.matmul(
                    rho(layers[i - 1].detach()), self.forward_weights[i - 1]
                )
                size += 1
            if i != self.N - 1:
                predict = predict + torch.matmul(
                    rho(layers[i + 1].detach()), self.backward_weights[i]
                )
                size += 1

            if size > 1:
                predict = predict / size

            energy = energy + 0.5 * ((predict - layers[i]) ** 2).sum(dim=1)

        return energy

    def compute_energy(self, layers: List[torch.Tensor]) -> torch.Tensor:
        return self.energy(layers)

    def cost(self, layers: List[torch.Tensor], y_one_hot: torch.Tensor) -> torch.Tensor:
        """
        均方误差
        返回 shape=(B,)
        """
        y = layers[-1]
        return ((y - y_one_hot) ** 2).sum(dim=1)

    def total_energy(
        self, layers: List[torch.Tensor], y_one_hot: torch.Tensor, beta: float
    ) -> torch.Tensor:
        return self.energy(layers) + beta * self.cost(layers, y_one_hot)

    @torch.no_grad()
    def _clip_states_(self, states: List[torch.Tensor]):
        for i in range(len(states)):
            states[i].clamp_(0.0, 1.0)

    @staticmethod
    def _rho_prime(state: torch.Tensor) -> torch.Tensor:
        return ((state > 0.0) & (state < 1.0)).to(dtype=state.dtype)

    @torch.no_grad()
    def _batch_spectral_radius_two_state_layers(
        self, states: List[torch.Tensor]
    ) -> torch.Tensor:
        hidden_state, output_state = states
        hidden_prime = self._rho_prime(hidden_state)
        output_prime = self._rho_prime(output_state)

        # output -> hidden uses backward_weights[1], hidden -> output uses forward_weights[1]
        hidden_from_output = self.backward_weights[1]
        output_from_hidden = self.forward_weights[1]

        # hidden layer averages input/output predictions, output layer uses only previous layer
        hidden_scale = 0.5
        output_scale = 1.0

        block_hidden_output = (
            hidden_from_output.T.unsqueeze(0) * output_prime.unsqueeze(1) * hidden_scale
        )
        block_output_hidden = (
            output_from_hidden.T.unsqueeze(0) * hidden_prime.unsqueeze(1) * output_scale
        )

        reduced_jacobian = block_output_hidden @ block_hidden_output
        eigvals = torch.linalg.eigvals(reduced_jacobian)
        return torch.sqrt(torch.abs(eigvals).amax(dim=1))

    @torch.no_grad()
    def _single_state_jacobian(self, state_vectors: List[torch.Tensor]) -> torch.Tensor:
        state_sizes = self.layer_sizes[1:]
        offsets = [0]
        for size in state_sizes:
            offsets.append(offsets[-1] + size)

        total_dim = offsets[-1]
        jacobian = torch.zeros(
            total_dim,
            total_dim,
            device=self.device,
            dtype=self.forward_weights[0].dtype,
        )

        def state_slice(layer_idx: int) -> slice:
            state_idx = layer_idx - 1
            return slice(offsets[state_idx], offsets[state_idx + 1])

        for target_layer in range(1, self.N):
            row_slice = state_slice(target_layer)
            neighbor_count = 0
            if target_layer != 0:
                neighbor_count += 1
            if target_layer != self.N - 1:
                neighbor_count += 1
            scale = 1.0 / float(neighbor_count)

            if target_layer > 1:
                prev_layer = target_layer - 1
                prev_slice = state_slice(prev_layer)
                prev_state = state_vectors[prev_layer - 1]
                prev_prime = self._rho_prime(prev_state)
                weight_prev = self.forward_weights[target_layer - 1]
                jacobian[row_slice, prev_slice] = (
                    weight_prev.T * prev_prime.unsqueeze(0) * scale
                )

            if target_layer < self.N - 1:
                next_layer = target_layer + 1
                next_slice = state_slice(next_layer)
                next_state = state_vectors[next_layer - 1]
                next_prime = self._rho_prime(next_state)
                weight_next = self.backward_weights[target_layer]
                jacobian[row_slice, next_slice] = (
                    weight_next.T * next_prime.unsqueeze(0) * scale
                )

        return jacobian

    @torch.no_grad()
    def batch_spectral_radius(self, states: List[torch.Tensor]) -> Tuple[float, float]:
        if len(states) != self.N - 1:
            raise ValueError(f"Expected {self.N - 1} state tensors, got {len(states)}.")

        if len(states) == 2:
            radii = self._batch_spectral_radius_two_state_layers(states)
            return radii.mean().item(), radii.max().item()

        representative_states = [state.mean(dim=0) for state in states]
        jacobian = self._single_state_jacobian(representative_states)
        eigvals = torch.linalg.eigvals(jacobian)
        radius = torch.abs(eigvals).max().item()
        return radius, radius

    def relax_states(
        self,
        x: torch.Tensor,
        states_init: List[torch.Tensor],
        y_one_hot: Optional[torch.Tensor],
        beta: float,
        n_steps: int,
        epsilon: float,
    ) -> List[torch.Tensor]:
        """
        用梯度动力学近似求平衡点：
        - free phase: beta=0, y_one_hot=None（或忽略）
        - weak phase: beta>0, y_one_hot 必须提供
        """
        # states: list of (B, dim) tensors
        states = [s.detach().clone().requires_grad_(True) for s in states_init]
        layers = [x] + states
        layers_optim = torch.optim.SGD(states, lr=epsilon)

        for _ in range(n_steps):

            with torch.enable_grad():
                if beta == 0.0:
                    F_sum = self.energy(layers).sum()
                else:
                    assert y_one_hot is not None
                    F_sum = self.total_energy(layers, y_one_hot, beta).sum()

            layers_optim.zero_grad()
            F_sum.backward()
            layers_optim.step()

            with torch.no_grad():
                self._clip_states_(states)

        return states

    def train_batch(
        self,
        x: torch.Tensor,  # shape=(B, input_dim)
        y: torch.Tensor,  # shape=(B,) 类别
    ) -> Tuple[float, float, float, float, float, float, List[float], List[float]]:
        """
        返回：(E_mean, C_mean, error, loss, delta_log_per_weight_layer)
        """

        B = x.shape[0]

        # 将输入/标签放到计算设备（权重所在设备）
        x = x.to(self.device, non_blocking=True).view(B, -1).float()
        y = y.to(self.device, non_blocking=True).long()
        y_one_hot = F.one_hot(y, num_classes=self.layer_sizes[-1]).float()

        states0 = [
            torch.zeros(B, dim, device=self.device, dtype=torch.float32)
            for dim in self.layer_sizes[1:]
        ]

        # ---------- free phase ----------
        states_free = self.relax_states(
            x=x,
            states_init=states0,
            y_one_hot=None,
            beta=0.0,
            n_steps=self.free_steps,
            epsilon=self.epsilon,
        )

        # ---------- weakly clamped phase ----------
        states_beta = self.relax_states(
            x=x,
            states_init=states_free,
            y_one_hot=y_one_hot,
            beta=float(self.beta),
            n_steps=self.weak_steps,
            epsilon=self.epsilon,
        )

        # 计算测量指标
        with torch.no_grad():
            layers_free = [x] + states_free
            E_mean = self.energy(layers_free).mean().item()
            C_mean = self.cost(layers_free, y_one_hot).mean().item()
            y_pred = torch.argmax(states_free[-1], dim=1)
            err = (y_pred != y).float().mean().item()
            rho_mean, rho_max = self.batch_spectral_radius(states_free)
            layers_beta = [x] + states_beta
            loss = self.cost(layers_beta, y_one_hot).mean().item()

        # 权重更新
        layers_beta_det = [x] + [s.detach() for s in states_beta]
        F_beta = self.total_energy(
            layers_beta_det, y_one_hot, beta=float(self.beta)
        ).mean()

        for w in self.forward_weights:
            if w.grad is not None:
                w.grad.zero_()
        for w in self.backward_weights:
            if w.grad is not None:
                w.grad.zero_()
        F_beta.backward()

        forward_delta_logs = []
        backward_delta_logs = []
        eps = 1e-8
        with torch.no_grad():
            for i in range(len(self.forward_weights)):
                w = self.forward_weights[i]
                grad = w.grad
                if grad is None:
                    forward_delta_logs.append(0.0)
                    continue
                w_abs = w.abs().mean().item()
                delta = -self.alphas[i] * grad
                w.add_(delta)
                forward_delta_logs.append(delta.abs().mean().item() / (w_abs + eps))

            for i in range(len(self.backward_weights)):
                w = self.backward_weights[i]
                grad = w.grad
                if grad is None:
                    backward_delta_logs.append(0.0)
                    continue
                w_abs = w.abs().mean().item()
                delta = -self.alphas[i] * grad
                w.add_(delta)
                backward_delta_logs.append(delta.abs().mean().item() / (w_abs + eps))

        return (
            E_mean,
            C_mean,
            err,
            loss,
            rho_mean,
            rho_max,
            forward_delta_logs,
            backward_delta_logs,
        )
