# PyEidors 方法路线图

更新时间：2026-04-26

这份文档是项目的“全局地图”。它回答三个问题：

- 当前 PyEidors 已经有哪些正问题和逆问题路线。
- 每条路线解决什么问题，处在 baseline、production、dynamic-quality 还是 research 层级。
- 哪些实验结果支撑当前判断，哪些结论还不能对外过度宣称。

## 一句话主线

PyEidors 当前主线是：

```text
高保真正问题 CEM
  -> 离线构建 RM / prior / dynamic model
  -> 在线 RM @ normalized delta_v
  -> 用 EIDORS-aligned metrics + 动态保真指标复核
```

也就是说，实时主线不是每帧重新 forward、重建 Jacobian、跑完整 GN；实时主线是把重活放到离线或低频 cold path，把在线 hot path 压成矩阵乘法和轻量动态状态更新。

## 项目分层

| 层 | 主要模块 | 目标 | 当前状态 |
|---|---|---|---|
| 数据与协议 | `pyeidors.data` | 差分、归一化、坏通道、权重、动态帧、EIDORS 噪声 | 可用 |
| 正问题 | `pyeidors.forward` | DOLFINx/FEniCSx CEM + PETSc KSP/PC | 3D 主线可用 |
| 一阶逆问题 | `pyeidors.inverse.reconstruction_matrix` | one-step GN/NOSER/Laplace/curvature RM | v1 主线可用 |
| 先验与正则化 | `pyeidors.inverse.prior`, `pyeidors.inverse.dynamic` | 通用 `RtR/R_prior`，TV-IRLS，时空先验 | 可扩展 |
| 动态重建 | `pyeidors.inverse.dynamic` | 4D GN，TV/Huber，Kalman/fixed-lag | benchmark 可用 |
| GREIT | `pyeidors.inverse.greit` | 3D GREIT RM + metrics | 线性化 v0 可用，官方 parity 未完成 |
| 评价体系 | `scripts/benchmarks/*review*` | RMSE + 动态指标 + EIDORS-aligned metrics | 已用于复核 |

## 正问题路线

| 路线 | 作用 | 优点 | 缺点 | 推荐场景 |
|---|---|---|---|---|
| DOLFINx CEM forward | 完整电极模型正问题 | 物理含义最强，适合 3D、接触阻抗、真实几何 | cold path 贵，依赖 Nix/FEniCSx/PETSc | 离线 RM 构建、真实 3D benchmark、论文基准 |
| PETSc `spd_gamg + cuda + matSolve=off` | 当前 3D forward 默认安全路线 | 已避开 Hypre CUDA 和 `KSPMatSolve` 风险 | 不是纯 GPU 全链路，仍有 cold solve 成本 | 48e/5936 这类大 3D forward |
| Dual-model | fine forward mesh + coarse inverse mesh | EIDORS 风格，在线 unknown 少，适合 RM | 分辨率受 coarse inverse 限制 | v1 production 主线 |
| Surrogate / bucket linearized model | 快速可控实验 | 非常快，便于扫采集模式、噪声、拟合策略 | 不是完整 FEM 物理 | 数据管线、噪声、采集策略探索 |

正问题的当前判断：

- 真实 3D 仍以 DOLFINx CEM 为物理基准。
- 在线重建不应把完整 forward solve 放进每帧 hot path。
- bucket/surrogate 不是替代真实 forward，而是用于快速实验设计和参数趋势判断。

## 逆问题路线

| 方法 | 层级 | 解决什么 | 优点 | 主要缺点 |
|---|---|---|---|---|
| Rowwise RM / one-step GN | baseline / v1 | 单帧差分重建 | 在线极快，公式透明，贴近 EIDORS 一阶骨架 | 不利用时间连续性 |
| NOSER RM | baseline / v1 | 官方风格一阶正则 | 稳定，尺度自适应，适合对齐官方 baseline | 易平滑，参数敏感 |
| Laplace RM | baseline / v1 | 图平滑先验 | 稳、简单、默认好解释 | 会抹边界和波前 |
| curvature / graph_ltl | regularization foundation | 曲率型或二阶平滑惩罚 | 与 Laplace 形成可比较平滑族，签名可区分 | 当前 travelling-wave fixture 中与 Laplace 多项打平 |
| TV-IRLS | quality / opt-in | 保边界、保稀疏结构、保波前 | 中心定位和速度保真强 | RMSE 和 solution-error 不一定最好，计算更贵 |
| 3D GREIT RM | v1 enhancement | 固定几何多目标 RM 成像 | 在线快，有 AR/PE/RES/SD/RNG 指标 | 当前是线性化 v0，不是官方完整 parity |
| Matrix-free GN / IRGNM | phase-2 | 避免 dense-J，提升高质量离线能力 | 对大模型更合理，有高级 PC 路线 | 仍不适合作为当前 realtime 默认 |
| SBL / BSBL / SA-SBL | research | 稀疏异常、目标检测、频差 EIT | 对 sparse target 有潜力 | 尚未通过 T70 接受 benchmark |

当前默认策略：

- v1 生产默认：dual-model + cached one-step RM。
- 官方对齐 baseline：NOSER、Laplace、one-step GN。
- 动态质量增强：T66 TV/Huber。
- 研究候选：SBL、full propagation-aware prior、official GREIT parity。

## 正则化体系

当前一阶 GN/NOSER/Laplace 的共同骨架是：

```text
J.T @ W @ J + hp^2 * RtR
```

近期工作已经把这里从“写死某个正则项”推进成通用 `RtR/R_prior` 插件式接口：

- `NOSER`: `RtR = diag(J.T @ J) ** noser_exponent`
- `Laplace`: 逆网格邻接图 Laplacian
- `curvature / graph_ltl`: `L.T @ L`
- `TV-IRLS`: 每轮按当前图差分更新权重，近似 TV/Huber 型稳健先验

这带来的好处：

- RM builder、GN runtime、matrix-free 路线可以共享先验契约。
- cache signature 能区分不同数学先验。
- 未来 SBL、anatomical prior、propagation prior 可以挂接，而不是重写求解器。

## 动态路线

动态路线是为神经活动传导、植物神经或慢脉冲、连续测量准备的。

| 方法 | 作用 | 优点 | 缺点 | 当前建议 |
|---|---|---|---|---|
| Measurement temporal filtering | 测量域 EMA / moving average | 便宜，容易上线，适合先降噪 | 会带来峰值延迟，可能抹快波 | 连续测量第一版工具 |
| T65 4D GN | batch spatiotemporal L2 | 比 rowwise RM 更懂时间连续性 | L2 会平滑突发 | 慢变化 baseline |
| T66 TV/Huber | 时空稳健正则 | 保波前、保突发、快传导保真更好 | 参数更多，batch solve 更重 | 当前动态保真首选 |
| T67 Kalman/fixed-lag | 在线低延迟状态估计 | 可控 latency，实时友好 | identity A 保真不足 | 低延迟 fallback |
| propagation-aware A | Kalman transition 中加入传播速度 | 高噪声下显著增强 Kalman | 目前只接 benchmark，不改默认重建器 | Kalman 增强候选 |
| T68 propagation-aware prior | 方向、速度、路径先验 | 可表达神经/植物传导结构 | 尚未正式实现 | 后期高级路线 |

动态路线的当前判断：

- 追求保真：优先 T66 TV/Huber。
- 追求低延迟：Kalman/fixed-lag，最好配 propagation-aware A。
- 植物慢脉冲：T65 4D GN 已经是很稳的 baseline。
- 神经快传导：T66 + propagation-aware Kalman 组合最值得继续测。

## 评价体系

项目现在不再接受“只看 RMSE”的结论。核心评价分三组：

| 评价组 | 指标 | 作用 |
|---|---|---|
| 像素/场误差 | RMSE、relative RMSE、solution-error | 全局误差和数据拟合 |
| 动态保真 | onset error、peak-time error、speed error、amplitude attenuation、SNR gain | 判断是否保住传导过程 |
| EIDORS-aligned | AR、PE、RES、SD、RNG、NF、solution-error | 对齐 GREIT/EIDORS 风格评价 |

注意：

- EIDORS-aligned 指标没有一个官方单一总分。
- 当前 review 采用 per-metric winner 和多数门槛，而不是把所有指标硬凑成一个分数。
- 对神经传导，动态速度和峰值时间不能被 RMSE 掩盖。

## 当前证据

| 结论 | 证据 |
|---|---|
| 48e/5936 在线 RM hot path 成立 | `reports/runtime_benchmarks/dual_model_rm_48e_5936_t36_20260422/README.md` |
| 512 帧在线 apply 已是毫秒级批处理 | Laplace CUDA 512 帧 `0.036958s`，GREIT CUDA 512 帧 `0.033325s` |
| 3D GREIT EIDORS-component path 已完成但官方等价声明未放行 | T49 48e/5936 surrogate gate 中 `Y/D/PJt/M/noiselev/RM/RM@dv/metrics` 全过；`official_equivalence_claim_allowed=false` 因外部 MATLAB/EIDORS 48e fixture 未接入 |
| 4D GN 比 rowwise RM 更懂连续过程 | `dynamic_validation_4d_gn_vs_rowwise_rm_20260425.md`，plant speed error 从 `0.0570454` 降到 `0.0441677` |
| 高噪声快传导整体保真首选 T66 | `dynamic_t65_t66_t67_high_noise_sweep_20260426.md`，best overall 为 T66 TV/Huber |
| propagation-aware A 是 Kalman 有效增强 | `dynamic_eidors_metric_review_propagation_A_multiseed_high_noise_20260426.md`，5 seed 全部通过，计票 `5/7` 或 `6/7` |
| TV-IRLS 不是全指标赢家 | `prior_travelling_wave_eidors_review_20260426.md`，TV-IRLS 中心和速度更好，Laplace-family RMSE/solution-error 更好 |

## 不能过度宣称的点

- 当前 3D GREIT 的 EIDORS-component 路线 T40-T50 已完成，但 T49 仍是 surrogate gate；接入真实 MATLAB/EIDORS 48e fixture 前，不应称作官方完整等价。
- linearized GREIT RM v0 仍保留为显式 non-parity 快速/基线模式，不能和 EIDORS-component 路线混称。
- propagation-aware A 已通过 benchmark gate，但只是 T67 Kalman benchmark 原型，不是默认重建器。
- SBL/BSBL 仍是 research，未通过 T70 benchmark 前不应作为默认推荐。
- matrix-free GN/IRGNM 有研究价值，但还没有替代 cached RM hot path 的 48e/5936 证据。

## 下一步主线

1. 把 benchmark 证据持续归档到 `reports/runtime_benchmarks/README.md`。
2. 对 dynamic 方法继续做多噪声、多 seed、不同速度范围复核。
3. 推进 T68 propagation-aware prior，但保持 opt-in。
4. 推进 T42 到 T50，完成 official GREIT parity 后再开放“EIDORS 等价”表述。
5. 对 SBL/BSBL 做 T70 接受 benchmark，只有赢了才晋升。
