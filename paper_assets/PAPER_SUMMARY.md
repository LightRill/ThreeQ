# ThreeQ 论文材料汇总

生成日期：2026-04-19

这个目录把目前已有实验整理成论文写作可直接引用的图、表和结论。所有图表均由已有 `experiments/*/results` 与 legacy 收敛性文件生成，没有重新训练模型。

## 1. 当前核心结论

1. **ThreeQ inference 收敛性有清晰实验证据。** `fig01_threeq_inference_convergence.png` 展示了固定点推断中 $\rho(J)<1$ 时 `delta` 呈指数式下降；当局部谱半径长期超过 1，例如 `Lg=15`，`delta` 明显停滞。这与理论证明中的充分条件一致：若固定点处 $\rho(J)<1$，局部 inference 指数稳定。
2. **$\rho(J)>1$ 的发散侧证已补齐。** `fig09_controlled_inference_rho_divergence.png` 使用受控线性 ThreeQ 母模型 $T(u)=\rho u$，展示 $\rho=1.05,1.20$ 时扰动范数指数增长；`fig10_training_rho_boundary_stress_test.png` 展示 bounded-gradient 大步训练更新跨过 $\rho=1$ 后 post-update inference 发散。
3. **训练过程也基本符合“小步更新保持稳定裕度”的判断。** `fig02_threeq_training_convergence.png` 显示 legacy ThreeQ/EPThreeQ 训练中 `rho_mean` 持续低于 1；同时 EPThreeQ validation accuracy 上升更快。`EPBase3Q` 在 MNIST 5k/1k、8 epoch 下 best accuracy 为 77.5%，而 direct `Base3Q` 为 60.2%。
4. **EPThreeQ 是当前最稳的原始 ThreeQ 改进线。** 在 MNIST 10k/2k、15 epoch 中，`epbase3q_legacy_10k_e15_beta1` 达到 best accuracy 86.5%，final accuracy 85.2%。这支持“原始版本能到 85% 左右，EP 更稳定”的判断。
5. **DThreeQ 的 EP-nudge/CE-nudge 版本可学习，但还没有稳定复现 90%+。** 10k/2k、30 epoch 下 CE-style nudge selected accuracy 为 86.5%；60k/10k full-data + restore-best 的最好结果为 88.0%，但 final accuracy 回退到 83.8%。
6. **Dplus/residual-delta 主训练规则目前不是 BP-like。** 在同一批 mini-batch 上，`direct_plus` 的 Dplus-vs-BP forward cosine 只有 0.076，sign agreement 为 0.568；`plusminus` 变体 cosine 为 -0.157 且 one-step free-MSE decrease 为负，说明符号抵消是明确失败模式。
7. **输入重构项不是简单删掉就能解决。** `fig08_input_energy_supervision.png` 显示 input residual weight 设为 0 或极小会降低 final accuracy；输入项既压制分类监督，也在维持表示。下一步应改为 layer-normalized/boundary-clamped input energy，而不是直接移除。

## 2. 图表索引

| 编号 | 文件 | 论文用途 |
|---|---|---|
| Fig.1 | `figures/fig01_threeq_inference_convergence.png` | 证明式实验：inference 残差、能量与 $\rho(J)$ 的关系 |
| Fig.2 | `figures/fig02_threeq_training_convergence.png` | 训练收敛：legacy ThreeQ/EPThreeQ accuracy 与 `rho_mean` |
| Fig.3 | `figures/fig03_mnist_current_performance.png` | MNIST 当前性能总览：ThreeQ、EPThreeQ、DThreeQ |
| Fig.4 | `figures/fig04_dthreeq_improvement_process.png` | DThreeQ 从小预算失败到 full-data 88% 的改进路径 |
| Fig.5 | `figures/fig05_dthreeq_training_curves.png` | DThreeQ 10k/full-data 训练曲线，显示 early-best 与后期崩塌 |
| Fig.6 | `figures/fig06_mechanism_direction_metrics.png` | Dplus、BP、EP 更新向量方向诊断 |
| Fig.7 | `figures/fig07_mechanism_one_step_loss.png` | Dplus one-step loss decrease 与 plusminus 失败 |
| Fig.8 | `figures/fig08_input_energy_supervision.png` | 输入重构能量比例与分类 accuracy 的关系 |
| Fig.9 | `figures/fig09_controlled_inference_rho_divergence.png` | 受控线性 ThreeQ 母模型中 $\rho>1$ 的指数发散证据 |
| Fig.10 | `figures/fig10_training_rho_boundary_stress_test.png` | 训练更新跨过 $\rho=1$ 后 post-update inference 发散 |

## 3. 表格索引

| 文件 | 内容 |
|---|---|
| `tables/key_results.csv` / `.md` | two moons、MNIST、EPThreeQ、DThreeQ 关键结果总表 |
| `tables/mnist_key_results.csv` / `.md` | MNIST 相关结果子表 |
| `tables/dthreeq_progress.csv` / `.md` | DThreeQ 改进过程分阶段表 |
| `tables/mechanism_diagnostic_key_metrics.csv` / `.md` | Dplus/BP/EP cosine、norm ratio、sign agreement 与 one-step loss |
| `tables/dplus_fix_top_candidates.csv` / `.md` | residual normalization、scale calibration、layer gain 的候选诊断 |
| `tables/controlled_inference_rho_summary.csv` / `.md` | 受控线性 inference 收敛/发散摘要 |
| `tables/training_stability_stress_summary.csv` / `.md` | 小步/大步训练更新稳定性压力测试摘要 |
| `latex/threeq_convergence_report.tex` / `.pdf` | 可迁移到论文的 LaTeX 学术报告 |

## 4. 可用于论文正文的判断

**理论与实验闭环。** 目前可以把理论部分表述为局部充分条件，而不是全局收敛定理：$\rho(J)<1$ 保证固定点局部稳定，小学习率保证稳定裕度局部保持。Fig.1 和 Fig.2 正好支撑这个说法，同时也展示了条件不满足时的失败边界。

**ThreeQ/EPThreeQ 的经验事实。** 原始 ThreeQ 架构并非不能学 MNIST；问题在于慢收敛与超参敏感。EPThreeQ 在同等 legacy 设置下明显优于 direct ThreeQ，说明自由相/受扰相差分比直接优化 weak-state total energy 更合理。

**DThreeQ 的当前定位。** DThreeQ 的 EP-nudge 分支已经可作为可行能量模型路线继续推进；Dplus/residual-delta 分支仍应作为机制研究对象，而不是主 benchmark 规则。外部 90%+ 若真实存在，最可能差异来自 CE/softmax 输出、best checkpoint、完整预算、额外 readout/pretraining、layer-normalized energy 或更强的后期稳定化。

**下一步最值得写入论文展望的方向。** 优先做 MNIST mini-batch mechanism diagnostic、boundary-clamped input energy、CE beta schedule、spectral/damping regularization，以及 signed residual target / output-to-hidden residual injection。继续只调 residual norm、$\beta$ 缩放或固定 layer gain 的边际收益很低。

## 5. DThreeQ 改进过程简表

| stage | variant | dataset_budget | best_accuracy | final_accuracy | interpretation |
| --- | --- | --- | --- | --- | --- |
| small public screen | dthreeq_ep_nudge_0p01 | 3k/1k, 3 epochs | 0.1332 | 0.1332 | 公共框架小预算接近随机，不能代表原始设定。 |
| legacy-scale focus | dthreeq_ep_nudge0p1_lr1e3 | 10k/2k, 12 epochs | 0.8035 | 0.8035 | 放大 hidden size 与训练预算后离开随机。 |
| longrun EP-nudge | dthreeq_ep_nudge0p1_lr3e3 | 10k/2k, 30 epochs | 0.8394 | 0.8354 | 稳定 baseline 约 83.5% 到 83.9%。 |
| CE-style nudge | dthreeq_ep_ce_nudge0p1_lr3e3 | 10k/2k, 30 epochs | 0.8684 | 0.8655 | 输出监督从 MSE nudge 改为 CE-style 后到约 86.5%。 |
| full-data restore-best | dthreeq_ep_signed_nudge0p1_lr3e3_restorebest | 60k/10k, 30 epochs | 0.8803 | 0.8379 | best checkpoint 达 88.0%，但 final 仍回退。 |
