# ThreeQ 系列研究报告

最后更新：2026-04-18  
当前状态：第一轮理论整理、最小可比较实验、DThreeQ 可行性筛查、DThreeQ 机制诊断、Dplus 修正诊断已完成。

## 1. 执行摘要

本项目研究的是一类基于“对称/双向局部预测 + 截断梯度状态更新”的能量模型。核心目标不是直接复制反向传播，而是构造一个局部状态动力学：每层状态由相邻层双向预测，状态推断使用局部能量下降，权重更新只通过固定后的自由相/受扰相状态产生。

截至目前的结论可以压缩为五点：

1. **理论上，局部 inference 收敛是可以保证的。** 在母模型 $T(u,W)=\frac{1}{n-1}Wg(u)$ 下，如果固定点处 Jacobian $J(u^*,W)$ 满足 $\rho(J)<1$，则固定点局部指数稳定；若训练步长足够小且梯度有界，该稳定性在一次参数更新后保持。
2. **稳定性不等于表达能力或训练有效性。** 截断/饱和激活、局部平均预测、状态 detach 会让系统更稳定，但也削弱非线性表示和跨层 credit assignment。
3. **当前最小 two-moons 可比较实验中，EP 更新优于 direct 更新，但仍不稳。** `ep_cost_w50` 有最好的 mean best validation error 0.1783，但 final validation error 回退到 0.2817，且权重尺度最大；`ep_cost_w10` 更稳但上限较低。
4. **当前 DThreeQ/Dplus 规则没有通过 BP-like 机制诊断。** Dplus 与 BP 的 forward-only 更新方向几乎不对齐，cosine 仅约 0.03 到 0.08；Dplus 与 EP 在 one-sided targets 上高度同向，但 norm ratio 极小，说明它更像“尺度塌缩的 EP-like 更新”，不是 BP-like 更新。
5. **Residual normalization / beta scaling / layer-wise gain 主要修正尺度，不修正方向。** 新的 Dplus fix diagnostic 中，best candidate 的 `dplus_vs_bp` forward cosine 只从 raw direct 的 0.07598 提到 0.07624，sign agreement 仍为 0.5685；但 BP/EP norm-calibrated one-step free-MSE decrease 全部为正，说明该规则有局部下降成分，只是 credit assignment 方向仍没有被改写。

因此，短期主线不应直接扩大到 MNIST/FashionMNIST，而应先把 Dplus 从“可缩放的 EP-like 局部下降”改造成真正能改善跨层 credit assignment 的目标，再建立小规模训练筛查。

## 2. 代码与实验结构

当前仓库可分为四层：

| 层级 | 路径 | 作用 |
|---|---|---|
| 理论母模型 | `Provement/inferenceProve.tex` | 固定点、Jacobian、谱条件、小学习率稳定性证明 |
| 早期原型 | `AllConnected3Q*`、`Base3Q*`、`EPBase3Q*`、`CNN3Q*`、`EPCNN3Q*` | 原始探索脚本和 twomoons/MNIST/CNN 变体 |
| 公共实现 | `threeq_common/` | 可比较 MLP/DThreeQ/机制诊断/Dplus 修正诊断实现 |
| radas 实验 | `experiments/minimal_suite/`、`experiments/dthreeq_suite/`、`experiments/mechanism_diagnostic/`、`experiments/dplus_fix_diagnostic/`、`experiments/analysis_summary/` | 可复现的参数矩阵、结果物化、图表和汇总 |

后续所有实验都应优先落在 `experiments/<suite_name>/` 中，并把结果写入该 suite 的 `results/` 子目录，再更新本报告。

## 3. 理论结果：局部收敛与小步训练鲁棒性

### 3.1 母模型定义

证明文档考虑 $n$ 维状态 $u\in\mathbb{R}^n$，定义局部预测映射：

$$
T(u,W)=\frac{1}{n-1}Wg(u)
$$

连续时间 inference 动力学为：

$$
\dot u=-\beta(u-T(u,W)),\qquad \beta>0
$$

固定点满足：

$$
u^*(W)=T(u^*(W),W)
$$

令 $D(u)=\mathrm{diag}(g'(u_1),\dots,g'(u_n))$，则固定点附近的 Jacobian 为：

$$
J(u,W)=\frac{\partial T}{\partial u}(u,W)=\frac{1}{n-1}WD(u)
$$

### 3.2 局部指数稳定

在固定点 $u^*$ 附近，扰动 $\delta=u-u^*$ 满足：

$$
\dot \delta=-\beta(I-J(u^*,W))\delta+\beta r(\delta)
$$

其中 $r(\delta)=o(\|\delta\|)$。若

$$
\rho(J(u^*,W))<1
$$

则 $-\beta(I-J)$ 是 Hurwitz 矩阵，线性化系统指数稳定；再由 Lyapunov 间接法，原非线性系统在该固定点局部指数稳定。

证明中的更精确条件其实是 $\mathrm{Re}(\lambda_k(J))<1$，而 $\rho(J)<1$ 是更强但更易用于实验诊断的充分条件。

### 3.3 固定点分支与训练鲁棒性

定义残差：

$$
F(u,W)=u-T(u,W)
$$

若 $\det(I-J(u^*,W))\neq 0$，隐函数定理给出局部唯一且 $C^1$ 的固定点分支 $u^*(W)$。由于 $W\mapsto J(u^*(W),W)$ 连续，且谱半径连续，所以满足 $\rho(J)<1$ 的参数集合是开集。

训练步写作：

$$
W^+=W-\eta\nabla_W E(u^*(W),W)
$$

若 $\|\nabla_W E\|\le G$，且当前点有稳定裕度 $\rho(J)\le 1-\varepsilon$，则存在 $\eta_{\max}>0$，当 $0<\eta<\eta_{\max}$ 时，更新后仍有 $\rho(J(W^+))<1$，从而 inference 仍局部收敛。

### 3.4 理论边界

这个证明只保证局部 inference 稳定，不保证：

- 学到的表示有足够非线性；
- 权重更新方向接近 BP；
- 受扰相能产生有效 credit assignment；
- 长训练过程始终留在同一稳定邻域；
- 饱和带来的稳定性不会同时压制梯度信号。

这正是后续实验中观察到“收敛但弱表达、稳定但训练慢”的根源。

## 4. 模型家族

### 4.1 基础 ThreeQ / Base3Q

基础分层 MLP 使用每层状态 $x_i$，由相邻层双向预测：

- 下层到上层：`forward_weights`
- 上层到下层：`backward_weights`
- 激活：hard clipping $\rho(x)=\mathrm{clip}(x,0,1)$
- 状态推断：free phase 后接 weak/clamped phase
- direct 权重更新：对受扰相 total energy 直接求权重梯度

对应实现包括 `Base3Q/`、`Base3QClampWithLinear/`，以及公共版本 `threeq_common/models.py`。

基础 ThreeQ 的主要问题是：hard clipping 提供了稳定性，但很容易让状态进入饱和或近线性区域；同时局部 detach 使跨层误差信号很弱，导致 two moons 和 MNIST 上训练效率低、非线性能力弱。

### 4.2 EPThreeQ / EPBase3Q

EPThreeQ 将权重目标从 direct total energy 改为近似 EP 差分：

$$
\Delta W \propto -\nabla_W\frac{E(s_\beta,W)-E(s_0,W)}{\beta}
$$

对应实现包括 `EPBase3Q/`、`EPBase3QClampWithLinear/` 和公共实验中的 `weight_update="ep"`。

EP 的优点是：相比 direct objective，它更接近“自由相与受扰相能量差”带来的局部学习信号。当前结果也显示 EP 类变体在 two moons 上明显优于 direct 类变体。

EP 的问题是：受扰相步数和 $\beta$ 需要精细平衡。weak steps 太少时信号不足；weak steps 太多时可能出现 early-best 后回退、权重尺度膨胀和 final accuracy 下降。

### 4.3 CNNThreeQ / EPCNN3Q

CNN 版本把 MLP 的局部双向预测推广到卷积结构：

- 正向预测使用 `conv2d`
- 反向预测使用 `conv_transpose2d`
- 卷积核天然对应严格转置共享
- 输出层使用全连接 `fc_kernel` 与转置

对应实现包括 `CNN3Q/` 和 `EPCNN3Q/`。

这类结构在形式上更接近“严格对称结构”，但当前仓库只有 legacy decision-boundary 图和脚本，没有像 minimal suite 那样完整的多 seed 可比较表。因此本报告暂不把 CNN 结果放入数值排名，只把它作为结构推广方向保留。

### 4.4 DThreeQ / Dplus

DThreeQ 使用 tanh 状态、双向局部 residual，并比较自由相与 clamped/nudged 相的 residual 变化：

$$
\mathcal{L}_{D+}=\frac{1}{2}\sum_e\|r_e^{\beta}-r_e^0\|^2
$$

实验中包括：

- direct clamp
- nudge with $\beta=0.1,0.01,0.001$
- gradual clamp
- plus / plusminus sign
- BP 和 EP 对照组

DThreeQ 的初衷是避免直接依赖 EP 能量差，转而学习“受监督扰动导致的局部 residual 改变量”。但当前机制诊断显示：Dplus 与 BP 不对齐，且尺度严重塌缩；plusminus 还会发生符号抵消。

## 5. 可比较实验结果

### 5.1 Minimal Comparable Suite

实验路径：`experiments/minimal_suite/`  
结果路径：`experiments/minimal_suite/results/`  
综合图表：`experiments/analysis_summary/results/figures/`

![Minimal suite error by variant](experiments/analysis_summary/results/figures/minimal_error_by_variant.png)

![Minimal suite diagnostics](experiments/analysis_summary/results/figures/minimal_diagnostics_by_variant.png)

| variant | best valid mean | final valid mean | rho mean | saturation mean | weight abs mean |
|---|---:|---:|---:|---:|---:|
| `ep_cost_w50` | 0.1783 | 0.2817 | 0.6410 | 0.4084 | 1.0956 |
| `ep_cost_w10` | 0.2767 | 0.2817 | 0.4808 | 0.4628 | 0.2929 |
| `direct_cost_w50` | 0.2867 | 0.4933 | 0.5558 | 0.2232 | 0.2217 |
| `ep_tied_w10` | 0.3167 | 0.6667 | 0.6607 | 0.5690 | 0.3626 |
| `direct_linear_w10` | 0.3267 | 0.5267 | 0.4814 | 0.2803 | 0.2200 |
| `direct_cost_w10` | 0.3933 | 0.4600 | 0.4713 | 0.2915 | 0.2169 |
| `ep_cost_w2` | 0.4117 | 0.5200 | 0.3787 | 0.5312 | 0.2218 |

结论：

- `ep_cost_w50` 的 best checkpoint 最好，但 final error 回退，且权重尺度最大。
- `ep_cost_w10` 更稳，但 best performance 不如 `ep_cost_w50`。
- `ep_cost_w2` 受扰相太弱，训练信号不足。
- 严格转置共享的 `ep_tied_w10` final accuracy 很差，说明 tying alone 不是充分条件。

### 5.2 DThreeQ Two-Moons Screen

实验路径：`experiments/dthreeq_suite/`  
结果路径：`experiments/dthreeq_suite/results/`

![DThreeQ error by variant](experiments/analysis_summary/results/figures/dthreeq_error_by_variant.png)

![DThreeQ LR heatmap](experiments/analysis_summary/results/figures/dthreeq_best_error_heatmap.png)

| variant | best test mean | best test min | final test mean | state delta mean | weight abs mean |
|---|---:|---:|---:|---:|---:|
| `ep_gradual_100_0p01_plus` | 0.1565 | 0.1178 | 0.2264 | 0.0933 | 0.3143 |
| `ep_nudge_0p01_plus` | 0.1931 | 0.1154 | 0.2023 | 0.0233 | 0.1808 |
| `bp_tanh` | 0.3365 | 0.1178 | 0.3407 | n/a | n/a |
| `dplus_direct` | 0.3502 | 0.2163 | 0.3502 | 0.1779 | 0.0600 |
| `dplus_nudge_0p1_plus` | 0.3506 | 0.2188 | 0.3506 | 0.1434 | 0.0598 |
| `dplus_gradual_100_0p01_plus` | 0.3554 | 0.2404 | 0.3554 | 0.1116 | 0.0596 |
| `dplus_nudge_0p01_plus` | 0.3578 | 0.2548 | 0.3578 | 0.0284 | 0.0588 |
| `dplus_nudge_0p001_plus` | 0.3590 | 0.2668 | 0.3590 | 0.0054 | 0.0586 |
| `dplus_nudge_0p01_plusminus` | 0.3590 | 0.2668 | 0.3590 | 0.0301 | 0.0586 |

结论：

- 当前 Dplus 变体没有通过 two moons 第一轮筛查。
- Dplus 系列 mean best test error 基本停在 0.35 附近，并且很多 seed/LR 组合接近 chance-error。
- Dplus saturation 接近 0，说明失败不是 tanh clipping 饱和主导。
- 更可能的问题是 update/objective alignment：residual target、detach 路径、符号构造或尺度没有产生有效下降方向。

### 5.3 DThreeQ Mechanism Diagnostic

实验路径：`experiments/mechanism_diagnostic/`  
结果路径：`experiments/mechanism_diagnostic/results/`

该实验在同一批 mini-batch 和同一初始参数上比较 BP、EP、Dplus 的更新向量。BP 只更新 forward weights/biases，backward 参数在 full vector 中补零；因此主要结论使用 forward-only 指标。

![Forward update cosine](experiments/mechanism_diagnostic/results/figures/forward_cosine_heatmap.png)

![Forward update norm ratio](experiments/mechanism_diagnostic/results/figures/forward_norm_ratio_heatmap.png)

![BP-normed one-step free MSE decrease](experiments/mechanism_diagnostic/results/figures/bp_normed_free_mse_decrease.png)

| target | pair | cosine | norm ratio | sign agreement |
|---|---|---:|---:|---:|
| `direct_plus` | `dplus_vs_bp` | 0.0760 | 0.1705 | 0.5685 |
| `gradual_100_0p01_plus` | `dplus_vs_bp` | 0.0642 | 0.0817 | 0.5303 |
| `nudge_0p001_plus` | `dplus_vs_bp` | 0.0363 | 0.0036 | 0.5013 |
| `nudge_0p01_plus` | `dplus_vs_bp` | 0.0711 | 0.0259 | 0.5064 |
| `nudge_0p01_plusminus` | `dplus_vs_bp` | -0.1574 | 0.0013 | 0.4973 |
| `direct_plus` | `dplus_vs_ep` | 0.9971 | 0.0831 | 0.9879 |
| `nudge_0p01_plus` | `dplus_vs_ep` | 0.9276 | 0.0008 | 0.9898 |
| `nudge_0p01_plusminus` | `dplus_vs_ep` | -0.8290 | 4.62e-05 | 0.5167 |

| target | method | raw free-MSE decrease | BP-normed free-MSE decrease | raw positive rate |
|---|---|---:|---:|---:|
| `direct_plus` | BP | 1.34e-04 | 1.34e-04 | 1.0000 |
| `direct_plus` | Dplus | 3.57e-05 | 2.34e-04 | 1.0000 |
| `nudge_0p01_plus` | Dplus | 5.37e-06 | 2.26e-04 | 1.0000 |
| `gradual_100_0p01_plus` | EP | 2.43e-02 | 1.24e-04 | 1.0000 |
| `nudge_0p01_plusminus` | Dplus | -2.96e-07 | -4.71e-05 | 0.0000 |

结论：

- Dplus 与 BP 几乎不对齐，forward cosine 约 0.03 到 0.08，sign agreement 接近随机。
- Dplus 与 EP 在 one-sided plus targets 上高度同向，但 norm ratio 极小，说明它是尺度塌缩的 EP-like 更新。
- `plusminus` 是明确失败模式，会发生方向抵消，raw 和 BP-normed one-step loss 都不下降。
- EP raw step 下降最大，但主要来自极大的向量范数；BP-normed 后 EP 并不稳定优于 BP。

### 5.4 Dplus Fix Diagnostic

实验路径：`experiments/dplus_fix_diagnostic/`
结果路径：`experiments/dplus_fix_diagnostic/results/`

这轮实验直接检验上一节提出的五个修正方向：

- 只保留 one-sided plus targets，禁用 `plusminus` 主线；
- 对 Dplus objective 做 $\beta^{-1}$、$\beta^{-2}$ 缩放；
- 对 $r_\beta-r_0$ 做 residual RMS normalization；
- 对浅层 residual delta 加固定 layer-wise gain；
- 在机制诊断里同时记录 BP/EP forward-norm calibrated one-step loss decrease。

实验矩阵为 5 个 plus-only target $\times$ 7 个 objective variant $\times$ 3 个 seed，共 105 个 trial。

![Dplus fix BP cosine](experiments/dplus_fix_diagnostic/results/figures/dplus_vs_bp_cosine_heatmap.png)

![Dplus fix sign agreement](experiments/dplus_fix_diagnostic/results/figures/dplus_vs_bp_sign_agreement_heatmap.png)

![Dplus fix norm ratio](experiments/dplus_fix_diagnostic/results/figures/dplus_vs_bp_norm_ratio_log_heatmap.png)

![Dplus fix BP-scaled free MSE decrease](experiments/dplus_fix_diagnostic/results/figures/bp_scaled_free_mse_decrease_heatmap.png)

| objective | target | BP cosine | BP sign | Dplus/BP norm | EP cosine | raw free-MSE decrease | BP-scaled free-MSE decrease | BP-scaled CE decrease |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resnorm_layergain_beta_div1` | `direct_plus` | 0.07624 | 0.5685 | 7619 | 0.9975 | 3.21e-02 | 4.27e-04 | 2.23e-05 |
| `layergain_beta_div1` | `direct_plus` | 0.07613 | 0.5685 | 0.1705 | 0.9973 | 6.51e-05 | 4.28e-04 | 2.23e-05 |
| `resnorm_beta_div1` | `direct_plus` | 0.07602 | 0.5685 | 7619 | 0.9972 | 3.21e-02 | 4.27e-04 | 2.23e-05 |
| `raw` | `direct_plus` | 0.07598 | 0.5685 | 0.1705 | 0.9971 | 6.50e-05 | 4.27e-04 | 2.22e-05 |

结论：

- **方向没有被修正。** 最好的 `resnorm_layergain_beta_div1/direct_plus` 只把 BP cosine 从 raw direct 的 0.07598 提到 0.07624，幅度可以忽略；sign agreement 仍停在 0.5685。
- **尺度可以被修正，但这不等于 credit assignment 被修正。** `layergain_beta_div1/direct_plus` 的 raw Dplus/BP norm ratio 仍是 0.1705；BP-norm calibration 后 free-MSE decrease 变成 4.28e-04，超过 BP baseline 的 1.34e-04，但方向仍主要是 EP-like。
- **Residual normalization 有风险。** 它能把 raw free-MSE decrease 放大到约 3.2e-02，但也把 Dplus/BP norm ratio 推到 $10^3$ 到 $10^8$ 量级，并且 raw feedforward CE decrease 可变成负值；因此它只能和显式 norm calibration 配套使用。
- **Layer-wise gain 不是方向解法。** 固定 gain 的数值更可控，但对 BP cosine 几乎没有实质提升。
- **Direction regularizer 的第一版筛选过宽。** 所有 plus-only 候选都通过了 BP/EP-scaled free-MSE positive-rate 筛选，但仍没有拉近 BP 方向；下一版筛选必须把 `dplus_vs_bp` cosine/sign agreement 明确作为硬门槛，而不是只看 one-step free-MSE 是否下降。

因此，本轮 Dplus 修正诊断的判断是：plus-only + norm calibration 可以作为稳定训练时的安全步长控制，但 residual normalization、$\beta$ 缩放和固定 layer gain 还没有解决 Dplus 的核心机制问题。下一步需要改变 residual target 的结构，而不是继续只调标量尺度。

### 5.5 Legacy CNN / Decision Boundary Figures

这些图来自早期脚本，未配套多 seed 可比较表；因此只作为结构探索记录，不参与主排名。

| 模型 | 图 |
|---|---|
| Base3QClampWithLinear | ![Base3QClampWithLinear decision boundary](Base3QClampWithLinear/twomoons_decision_boundary.png) |
| EPBase3Q | ![EPBase3Q decision boundary](EPBase3Q/twomoons_decision_boundary.png) |
| CNN3Q | ![CNN3Q decision boundary](CNN3Q/twomoons_decision_boundary.png) |
| EPCNN3Q | ![EPCNN3Q decision boundary](EPCNN3Q/twomoons_decision_boundary.png) |

需要注意：这些 legacy 图无法替代 radas suite。后续若要评价 CNNThreeQ，应建立独立的 `experiments/cnn_suite/`，固定数据、seed、batch、推断步数、学习率，并记录与 minimal suite 一致的 best/final error、state_delta、rho/saturation、weight scale 和 runtime。

## 6. 总体结论

### 6.1 为什么 ThreeQ 在 MNIST 与 two moons 上效率低、非线性弱

1. **状态截断稳定但压缩非线性。** Hard clipping 把状态限制在 $[0,1]$，有助于控制 Jacobian，但会制造饱和区，降低有效导数。
2. **局部平均预测削弱高阶组合。** 每层只拟合相邻层平均预测，非线性组合能力依赖多次状态迭代间接形成，效率低于显式前向网络。
3. **detach 切断跨层 credit assignment。** 这符合局部学习设定，但也让远层监督信号只能通过状态差间接传播。
4. **受扰相尺度很难调。** Weak phase 太短则信号不足，太长则容易 early-best 后退化。
5. **稳定性条件可能鼓励饱和。** 谱半径变小可以来自 $g'(u^*)$ 变小，但这同时意味着学习信号也变小。

### 6.2 EPThreeQ 的位置

EP 更新比 direct 更新更合理，因为它比较自由相和受扰相的能量差，而不是直接在受扰状态上优化 total energy。实验中 EP 确实表现更好。但 EP 仍存在尺度和稳定终点问题：`ep_cost_w50` best 好但 final 回退，`ep_cost_w10` 稳但上限较低。

### 6.3 CNNThreeQ 的位置

CNN/ConvTranspose 是推广到严格对称结构的自然路线。它在结构上比独立 forward/backward MLP 更干净，但还缺少可比较实验和机制诊断。CNN 方向值得保留，但应在 Dplus/EP 更新机制更清楚后再扩大。

### 6.4 DThreeQ 的位置

DThreeQ 的 Dplus 目标目前还不能作为主训练规则直接扩大 benchmark。它的失败不是简单的“不能下降”，而是：

- 与 BP 更新方向不对齐；
- 与 EP 同向，且尺度可以通过 calibration 修正；
- plusminus 会产生符号抵消，已从主线移除；
- residual normalization、$\beta$ 缩放和固定 layer-wise gain 主要改变范数，不改变 BP-like 方向；
- raw 训练步下 Dplus 更新弱到不足以驱动表示学习，而过强的 residual normalization 又会破坏 feedforward CE。

因此下一步不应继续堆叠标量缩放，而应改 residual target 的结构，让输出扰动能以有符号、有层级权重的形式传到浅层。

## 7. 下一步研究计划

### 7.1 Dplus 修正优先级

已完成第一轮标量修正诊断：

1. **Residual normalization**：可以放大 raw free-MSE 下降，但会制造极大 update norm 和负 CE step，不能单独作为训练规则。
2. **EP/BP-scale calibration**：能让 one-step free-MSE 稳定下降，但不改善 BP cosine。
3. **禁用 plusminus 主线**：已完成，后续默认只保留 plus。
4. **Layer-wise gain**：固定 gain 对数值尺度更友好，但不解决方向。
5. **Direction regularizer**：one-step free-MSE positive-rate 过宽，下一版必须加入 BP cosine/sign agreement 门槛。

下一轮优先尝试以下结构性修改：

1. **Signed residual target**：不只拟合 $\|r_\beta-r_0\|^2$，而是保留 residual delta 的有符号投影，避免平方目标把方向信息平均掉。
2. **Output-to-hidden residual injection**：把输出层 supervised residual 通过严格转置或局部 Jacobian proxy 显式注入浅层 residual target，而不是依赖状态 relaxation 自然传递。
3. **Layer-wise trainable gain with constraint**：允许 gain 学习，但用 softplus/spectral penalty 限制范数，避免 residual normalization 的爆炸。
4. **Hard diagnostic gate**：候选必须同时满足 `dplus_vs_bp` forward cosine 高于 0.15、sign agreement 高于 0.60、BP-scaled one-step free-MSE positive-rate 接近 1，才进入 two-moons 训练筛查。
5. **Short training screen**：只把 `layergain_beta_div1/direct_plus` 和下一版 signed residual target 放入训练比较；`resnorm_*` 只作为 calibrated diagnostic，不作为 raw update baseline。

### 7.2 保留收敛性的非线性增强

可探索的方向：

1. 使用 smooth bounded activation，例如 tanh/softsign，降低 hard clipping 的死区。
2. 使用 layer-wise spectral normalization 或 damping，直接控制局部 Jacobian，而不是依赖饱和。
3. 在状态更新中加入 learnable damping $\lambda$，例如 $s^{t+1}=(1-\lambda)s^t+\lambda T(s^t)$。
4. 使用 block-symmetric 或 tied transpose 结构，但允许每层有独立 gain/normalization。
5. 用 residual/skip 状态变量提高表达能力，同时对每个 block 施加局部谱约束。

### 7.3 实验工程要求

所有新实验必须记录：

- raw per-trial CSV；
- summary CSV/Markdown；
- 至少一张 error 图、一张 diagnostic 图；
- seed、step、learning rate、target mode、device、runtime；
- 关键状态指标：rho 或替代谱指标、saturation、state_delta、weight scale、update scale；
- 对新学习规则，必须附 mechanism diagnostic。

## 8. 持续更新约定

本文件是项目的 living report。之后每次新增实验或理论修正，都必须同步更新以下位置：

1. **执行摘要**：更新当前最重要结论。
2. **代码与实验结构**：如果新增 suite，加入路径和作用。
3. **可比较实验结果**：加入图、表和一句话结论。
4. **总体结论**：若新结果改变判断，直接修改旧结论，不只追加。
5. **下一步研究计划**：把已完成项移出，新增下一项。
6. **更新日志**：记录日期、提交、实验名、核心发现。

### 更新日志

| 日期 | 提交 | 内容 |
|---|---|---|
| 2026-04-18 | `6c54602` | 新增 minimal comparable ThreeQ suite |
| 2026-04-18 | `077900d` | 新增 DThreeQ two-moons feasibility screen |
| 2026-04-18 | `38fab1b` | 新增 consolidated experiment analysis plots |
| 2026-04-18 | `9382a5f` | 新增 DThreeQ mechanism diagnostics |
| 2026-04-18 | `36f8d7d` | 新增本系统研究报告 |
