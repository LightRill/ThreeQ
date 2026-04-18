import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

# 设备选择：优先使用 GPU，若不可用则回退到 CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def rho(s: torch.Tensor) -> torch.Tensor:
    """
    截断型激活函数，将状态限制在 [0, 1]。
    在本实现中它对应“对称梯度截断”里的状态投影操作。
    """
    return torch.clamp(s, 0.0, 1.0)


class Network(nn.Module):
    """
    卷积版 EP(Equilibrium Propagation) 网络。

    关键设计：
    1) 连接权重使用同一组参数 W，在正向预测中用 conv2d，
       在反向预测中用 conv_transpose2d，实现对称连接 W 与 W^T。
    2) 计算某层的预测值时，对相邻层状态使用 detach，截断跨层梯度，
       只对当前层状态进行优化（对称梯度截断思想）。
    3) 最后一层使用全连接权重 fc_kernel。

    主要参数：
    - conv_channels: 每层通道数，例如 [1, 32, 64]
    - kernel_sizes / strides / paddings: 各卷积层超参数
    - input_size: 输入图像尺寸 (H, W)
    - num_classes: 类别数
    """

    def __init__(
        self,
        alphas: List[float],
        beta: float,
        free_steps: int,
        weak_steps: int,
        epsilon: float,
        conv_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        input_size: Tuple[int, int] = (28, 28),
        num_classes: int = 10,
        device=device,
    ):
        super().__init__()
        # -----------------
        # 保存超参数
        # -----------------
        self.alphas = alphas
        self.beta = beta
        self.free_steps = free_steps
        self.weak_steps = weak_steps
        self.epsilon = epsilon
        self.device = device
        self.num_classes = num_classes
        self.conv_channels = list(conv_channels)
        self.kernel_sizes = list(kernel_sizes)
        self.strides = list(strides)
        self.paddings = list(paddings)

        # 卷积连接数（conv_channels 长度减 1）
        n_conv = len(conv_channels) - 1
        self.n_conv = n_conv
        # 层数 N = 输入层 + 卷积层 + 输出层
        self.N = n_conv + 2

        if (
            len(kernel_sizes) != n_conv
            or len(strides) != n_conv
            or len(paddings) != n_conv
        ):
            raise ValueError(
                "kernel_sizes, strides, paddings must each have length "
                f"len(conv_channels)-1 = {n_conv}"
            )
        if len(alphas) != self.N - 1:
            raise ValueError(
                f"alphas length ({len(alphas)}) must be N-1 ({self.N - 1})"
            )

        # ---------- 计算每一层的空间尺寸 (H, W) ----------
        self.spatial_sizes: List[Tuple[int, int]] = [input_size]
        h, w = input_size
        for i in range(n_conv):
            # conv2d 输出尺寸公式：H_out = (H + 2p - k) // s + 1
            h = (h + 2 * paddings[i] - kernel_sizes[i]) // strides[i] + 1
            w = (w + 2 * paddings[i] - kernel_sizes[i]) // strides[i] + 1
            self.spatial_sizes.append((h, w))

        # ---------- 计算转置卷积的 output_padding ----------
        # PyTorch 公式：H_out = (H_in-1)*s - 2p + k + output_padding
        self.output_paddings: List[Tuple[int, int]] = []
        for i in range(n_conv):
            h_s, w_s = self.spatial_sizes[i + 1]
            h_b, w_b = self.spatial_sizes[i]
            op_h = h_b - ((h_s - 1) * strides[i] - 2 * paddings[i] + kernel_sizes[i])
            op_w = w_b - ((w_s - 1) * strides[i] - 2 * paddings[i] + kernel_sizes[i])
            self.output_paddings.append((op_h, op_w))

        # ---------- 每层的“展平维度” ----------
        self.layer_sizes: List[int] = []
        for c, (sh, sw) in zip(conv_channels, self.spatial_sizes):
            self.layer_sizes.append(c * sh * sw)
        self.layer_sizes.append(num_classes)

        # ---------- 初始化卷积核参数 ----------
        # conv_kernels[i] 形状: (C_{i+1}, C_i, K, K)
        #   用于 conv2d: i -> i+1
        #   用于 conv_transpose2d: i+1 -> i（实现对称连接）
        self.conv_kernels = nn.ParameterList()
        for i in range(n_conv):
            c_in = conv_channels[i]
            c_out = conv_channels[i + 1]
            k = kernel_sizes[i]
            W = torch.empty(c_out, c_in, k, k, device=self.device)
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))
            self.conv_kernels.append(nn.Parameter(W))

        # ---------- 初始化全连接参数 ----------
        flat_dim = (
            conv_channels[-1] * self.spatial_sizes[-1][0] * self.spatial_sizes[-1][1]
        )
        self.flat_dim = flat_dim
        # fc_kernel: (flat_dim, num_classes)
        self.fc_kernel = nn.Parameter(
            torch.empty(flat_dim, num_classes, device=self.device)
        )
        bound = math.sqrt(6.0 / (flat_dim + num_classes))
        with torch.no_grad():
            self.fc_kernel.uniform_(-bound, bound)

    # ------------------------------------------------------------------ #
    #  工具函数
    # ------------------------------------------------------------------ #
    def _to_spatial(self, flat: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        将展平向量恢复为 (B, C, H, W) 形式。
        layer_idx 用于选择对应层的通道数与空间尺寸。
        """
        B = flat.shape[0]
        c = self.conv_channels[layer_idx]
        h, w = self.spatial_sizes[layer_idx]
        return flat.view(B, c, h, w)

    # ------------------------------------------------------------------ #
    #  能量函数
    # ------------------------------------------------------------------ #
    def energy(self, layers: List[torch.Tensor]) -> torch.Tensor:
        """
        计算网络能量 E(x, h1, h2, ..., y)。

        layers: [x, h1, h2, ..., y]，每个元素形状 (B, dim_i)
        返回: (B,) 每个样本的能量

        连接关系：
        - j in [0, n_conv-1]  使用 conv_kernels[j]
        - j == n_conv         使用 fc_kernel
        """

        B = layers[0].shape[0]
        energy = torch.zeros(B, device=self.device)

        for i in range(self.N):
            # predict 表示对第 i 层的预测值，size 记录预测来源数量
            predict = torch.zeros(B, self.layer_sizes[i], device=self.device)
            size = 0

            # ---- 来自下层 (i-1) 的预测 ----
            if i > 0:
                conn = i - 1
                # 对相邻层状态做 detach，截断跨层梯度传播
                x_below = rho(layers[i - 1].detach())
                if conn < self.n_conv:
                    # 卷积连接：conn -> conn+1
                    x_sp = self._to_spatial(x_below, conn)
                    pred_sp = F.conv2d(
                        x_sp,
                        self.conv_kernels[conn],
                        stride=self.strides[conn],
                        padding=self.paddings[conn],
                    )
                    predict = predict + pred_sp.view(B, -1)
                else:
                    # 全连接：倒数第二层 -> 输出层
                    predict = predict + torch.matmul(x_below, self.fc_kernel)
                size += 1

            # ---- 来自上层 (i+1) 的预测 ----
            if i < self.N - 1:
                conn = i
                # 对相邻层状态做 detach，截断跨层梯度传播
                x_above = rho(layers[i + 1].detach())
                if conn < self.n_conv:
                    # 反向卷积：conn+1 -> conn
                    x_sp = self._to_spatial(x_above, conn + 1)
                    pred_sp = F.conv_transpose2d(
                        x_sp,
                        self.conv_kernels[conn],
                        stride=self.strides[conn],
                        padding=self.paddings[conn],
                        output_padding=self.output_paddings[conn],
                    )
                    predict = predict + pred_sp.view(B, -1)
                else:
                    # 输出层 -> 倒数第二层（使用 W^T）
                    predict = predict + torch.matmul(x_above, self.fc_kernel.t())
                size += 1

            # 如果上下两侧都有预测，取平均
            if size > 1:
                predict = predict / size

            # 能量项：0.5 * ||predict - state||^2
            energy = energy + 0.5 * ((predict - layers[i]) ** 2).sum(dim=1)

        return energy

    def compute_energy(self, layers: List[torch.Tensor]) -> torch.Tensor:
        return self.energy(layers)

    def cost(self, layers: List[torch.Tensor], y_one_hot: torch.Tensor) -> torch.Tensor:
        """监督代价项：输出层与标签的 MSE，返回 (B,)"""
        y = layers[-1]
        return ((y - y_one_hot) ** 2).sum(dim=1)

    def total_energy(
        self,
        layers: List[torch.Tensor],
        y_one_hot: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        return self.energy(layers) + beta * self.cost(layers, y_one_hot)

    @torch.no_grad()
    def _clip_states_(self, states: List[torch.Tensor]):
        # 将状态强制截断到 [0, 1]，保持与 rho 一致
        for s in states:
            s.clamp_(0.0, 1.0)

    # ------------------------------------------------------------------ #
    #  状态松弛 (state relaxation)
    # ------------------------------------------------------------------ #
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
        对隐藏状态进行迭代松弛（梯度下降）。
        - 自由相 (beta=0, y_one_hot=None)
        - 受扰相 (beta>0, y_one_hot 提供监督信号)

        注意：虽然整体处于 no_grad 环境，本函数内部使用
        torch.enable_grad 让状态可求导。
        """

        with torch.enable_grad():
            # 只对状态变量求梯度，不更新权重
            # 重要：clone 避免与 states_init 共享存储。
            # 否则在受扰相中更新会回写到自由相状态，导致 error 总是接近 0。
            states = [s.detach().clone().requires_grad_(True) for s in states_init]
            layers = [x] + states
            layers_optim = torch.optim.SGD(states, lr=epsilon)

            for _ in range(n_steps):
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

    # ------------------------------------------------------------------ #
    #  训练
    # ------------------------------------------------------------------ #
    def train_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[float, float, float, float, List[float]]:
        """
        对一个 mini-batch 执行 EP 训练。

        返回：
        - E_mean: 自由相能量均值
        - C_mean: 自由相代价均值
        - err: 自由相分类错误率 (0-1)
        - loss: 受扰相代价均值
        - kernel_delta_logs: 每层权重相对更新幅度（含 FC 层）
        """

        B = x.shape[0]
        # 输入展平并搬到目标设备
        x = x.to(self.device, non_blocking=True).view(B, -1).float()
        y = y.to(self.device, non_blocking=True).long()
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # 初始化所有隐藏层状态为 0
        states0 = [
            torch.zeros(B, dim, device=self.device, dtype=torch.float32)
            for dim in self.layer_sizes[1:]
        ]

        # ---------- 自由相 ----------
        states_free = self.relax_states(
            x, states0, None, 0.0, self.free_steps, self.epsilon
        )

        # ---------- 受扰相 ----------
        states_beta = self.relax_states(
            x, states_free, y_one_hot, float(self.beta), self.weak_steps, self.epsilon
        )

        # ---------- 统计指标 ----------
        with torch.no_grad():
            layers_free = [x] + states_free
            E_mean = self.energy(layers_free).mean().item()
            C_mean = self.cost(layers_free, y_one_hot).mean().item()
            y_pred = torch.argmax(states_free[-1], dim=1)
            err = (y_pred != y).float().mean().item()
            layers_beta = [x] + states_beta
            loss = self.cost(layers_beta, y_one_hot).mean().item()

        # ---------- 权重更新 ----------
        # 使用受扰相的能量对权重求梯度，然后按层学习率更新
        layers_beta_det = [x] + [s.detach() for s in states_beta]
        F_beta = self.total_energy(
            layers_beta_det, y_one_hot, beta=float(self.beta)
        ).mean()

        for w in self.conv_kernels:
            if w.grad is not None:
                w.grad.zero_()
        if self.fc_kernel.grad is not None:
            self.fc_kernel.grad.zero_()

        F_beta.backward()

        kernel_delta_logs: List[float] = []
        eps_small = 1e-8
        with torch.no_grad():
            # 更新卷积核
            for i in range(self.n_conv):
                w = self.conv_kernels[i]
                grad = w.grad
                if grad is None:
                    kernel_delta_logs.append(0.0)
                    continue
                w_abs = w.abs().mean().item()
                delta = -self.alphas[i] * grad
                w.add_(delta)
                # 记录相对更新幅度
                kernel_delta_logs.append(
                    delta.abs().mean().item() / (w_abs + eps_small)
                )

            # 更新全连接层
            w = self.fc_kernel
            grad = w.grad
            if grad is None:
                kernel_delta_logs.append(0.0)
            else:
                w_abs = w.abs().mean().item()
                delta = -self.alphas[-1] * grad
                w.add_(delta)
                kernel_delta_logs.append(
                    delta.abs().mean().item() / (w_abs + eps_small)
                )

        return E_mean, C_mean, err, loss, kernel_delta_logs
