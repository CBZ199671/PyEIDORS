# PyEidors 方法选择指南

更新时间：2026-04-26

这份文档回答一个更实际的问题：给定一个场景，应该先用哪个方法，为什么，不应该先用什么。

## 快速决策树

```text
是否需要实时在线？
  是 -> 是否固定几何/协议？
       是 -> cached RM: NOSER/Laplace/curvature
       否 -> 先建立固定协议或退回低频 cold solve
  否 -> 是否连续动态数据？
       是 -> 是否快传导/突发波前？
            是 -> T66 TV/Huber
            否 -> T65 4D GN
       否 -> 是否稀疏异常/目标检测？
            是 -> TV-IRLS，SBL 作为研究对照
            否 -> NOSER/Laplace baseline
```

## 场景选择表

| 场景 | 首选 | 备选 | 暂不优先 |
|---|---|---|---|
| 固定硬件实时 3D difference EIT | dual-model cached Laplace/NOSER RM | GREIT RM v0 | 每帧 dense-J GN |
| 官方 baseline 对齐 | one-step GN / NOSER / Laplace | curvature/graph_ltl | SBL |
| 神经快传导 | T66 TV/Huber | propagation-aware Kalman | identity-A Kalman 当最佳保真 |
| 植物慢脉冲 | T65 4D GN | T66 TV/Huber | 过强速度先验 |
| 高噪声连续测量 | T66 TV/Huber | propagation-aware Kalman | 单帧 RMSE 结论 |
| 低延迟在线动态 | Kalman/fixed-lag + propagation-aware A | measurement filtering + RM | 大窗口 batch solve 每帧重跑 |
| 稀疏异常体 | TV-IRLS | SBL/BSBL benchmark | 只用 Laplace 平滑 |
| 采集模式和噪声策略 | bucket all-modes noise sweep | full256/full208 compare | 直接上真实 3D 重 benchmark |
| 论文式 EIDORS 评价 | EIDORS-aligned metrics review | RMSE + dynamic metrics | 只报 RMSE |

## 方法说明

### Rowwise RM / one-step GN

适合：

- 单帧或逐帧 difference reconstruction。
- 官方 baseline。
- 需要极快 online hot path 的系统。

优点：

- 在线只需要 `RM @ delta_v`。
- 易缓存、易部署、易解释。
- 与 EIDORS 一阶解法骨架一致。

缺点：

- 不利用时间连续性。
- 对快传导波前和突发事件没有专门保护。

建议：

- 每条新路线都先和 rowwise RM 比。
- 不要因为高级方法还没调好，就丢掉这个 baseline。

### NOSER

适合：

- 官方风格 baseline。
- 对尺度变化较敏感的单帧问题。
- 需要快速稳定起点。

优点：

- `diag(J.T @ J)` 形式清晰。
- 当前支持 `noser_exponent`，默认更贴近 EIDORS sqrt diag 口径。

缺点：

- 平滑倾向明显。
- 对局部边界、传播波前不一定好。

建议：

- 用作 baseline，不要作为快传导最终质量结论。

### Laplace

适合：

- 稳定平滑重建。
- 大多数 v1 默认 RM baseline。
- 噪声不太极端的固定几何系统。

优点：

- 稳、简单、图结构直观。
- 和 coarse inverse mesh 自然匹配。

缺点：

- 容易抹平尖锐边界。
- 对神经快传导的波前可能不够保真。

建议：

- 当前默认稳健 baseline。
- 和 curvature/TV 做横向对照。

### curvature / graph_ltl

适合：

- 更强曲率约束实验。
- 检查二阶平滑是否优于 Laplace。

优点：

- `L.T @ L` 形式清楚。
- 可以在同一 RM/GN 框架中作为 prior 插件。

缺点：

- 当前 travelling-wave fixture 中与 Laplace 多项指标打平。

建议：

- 当目标是平滑曲率而不是边界保持时使用。
- 不要默认宣称比 Laplace 强，除非目标 fixture 证明。

### TV-IRLS

适合：

- 稀疏异常体。
- 边界、突发、波前。
- 神经快传导保真。

优点：

- 中心定位和传播速度指标强。
- 比 L2 时间平滑更能保留突变。

缺点：

- 计算更贵。
- RMSE、solution-error 不一定最好。
- 需要调 `beta`、outer iterations、Huber/TV 权重。

建议：

- 神经快传导质量路线优先考虑。
- 必须同时报告 RMSE、速度、峰值时间和 EIDORS-aligned 指标。

### T65 4D GN

适合：

- 连续测量 baseline。
- 植物慢脉冲、平滑传播。
- 窗口式离线或准实时分析。

优点：

- 比 rowwise RM 更稳定。
- 时间先验解释性强。

缺点：

- L2 时间先验会过平滑突发。

建议：

- 动态数据第一条 batch baseline。
- 快传导最终质量通常继续看 T66。

### T66 TV/Huber

适合：

- 神经快传导。
- 高噪声动态。
- 波前、onset、peak-time 保真。

优点：

- 当前 high-noise sweep 中整体 fast-conduction 最强。
- 推荐区间已有证据：`lambda_t 0.08..0.35`，`huber_delta 0.02..0.12`。

缺点：

- 比 T65 更复杂。
- 不一定所有 EIDORS-aligned 指标都赢。

建议：

- 作为动态保真首选。
- 若目标是低延迟 online，再与 propagation-aware Kalman 组合比较。

### T67 Kalman / fixed-lag

适合：

- 实时低延迟连续测量。
- 需要明确 latency budget 的系统。

优点：

- 在线状态更新轻。
- fixed-lag 可以在低延迟和保真之间折中。

缺点：

- identity A 时保真不够强。
- 默认不应宣称优于 T66。

建议：

- 作为低延迟 fallback。
- 高噪声快传导优先开启 propagation-aware A benchmark。

### propagation-aware A

适合：

- 已知传播方向或速度范围的连续事件。
- 神经/植物传导中的低延迟 Kalman 增强。

优点：

- 高噪声 multi-seed gate 已通过。
- 在 Kalman 内部相对 identity A 有稳定优势。

缺点：

- 当前只接 benchmark，不改默认重建器。
- 还不是 full propagation-aware prior。

建议：

- 作为 T67 的 opt-in 候选。
- 后续加密速度网格，并扩展到 T68 路径先验。

### GREIT RM

适合：

- 固定几何、固定协议、多帧快速成像。
- 需要 GREIT-style 指标输出的 benchmark。

优点：

- 在线快。
- 输出 AR/PE/RES/SD/RNG。

缺点：

- `linearized v0` 仍只是快速/基线模式，不能和 EIDORS parity path 混称。
- 48e official fixture 已过，但实际测量数是 2160；5936 协议仍需单独 official fixture。

建议：

- 可用作 v1 enhancement。
- 对外可说“48e official fixture parity passed”。
- 对外不要说“48e/5936 official-equivalent”，直到 5936 protocol fixture 单独通过。

## 评价选择

| 目标 | 必报指标 |
|---|---|
| 单帧图像质量 | RMSE、relative RMSE、solution-error |
| 目标定位 | PE、center RMSE、peak position |
| 图像形态 | AR、RES、SD、RNG |
| 噪声鲁棒 | NF、SNR gain、multi-seed stability |
| 神经/植物传导 | onset error、peak-time error、speed error、amplitude attenuation |
| 实时系统 | online apply time、latency frames/seconds、forward/J/KSP counters |

经验规则：

- RMSE 好，不代表传导时间好。
- TV/Huber 可能牺牲全局 RMSE，但改善中心和速度。
- Kalman 低延迟强，但默认 identity A 不应作为最佳保真。
- EIDORS-aligned review 看 per-metric majority，不造一个伪官方总分。

## 推荐工作流

### 新几何/新硬件

1. 跑正问题 smoke。
2. 构建 dual-model coarse inverse。
3. 先建 NOSER/Laplace RM。
4. 检查 online counters 为 0 forward / 0 KSP / 0 Jacobian rebuild。
5. 再加 GREIT 或动态路线。

### 新动态数据

1. Rowwise RM baseline。
2. Measurement temporal filtering。
3. T65 4D GN。
4. T66 TV/Huber。
5. T67 Kalman/fixed-lag。
6. 若是传导问题，额外跑 propagation-aware A sweep。
7. 用 EIDORS-aligned + dynamic metrics 复核。

### 新正则化方法

1. 先实现 `RtR/R_prior` contract。
2. 在 one-step RM 和 GN runtime 都能接入。
3. 跑 small fixture parity。
4. 跑 travelling-wave 或 sparse anomaly fixture。
5. 只有在多指标复核通过后再提升层级。

## Promotion gate

一个方法从 research 或 prototype 晋升，需要满足：

- 有 baseline 对照，不只单独跑自己。
- 有多 seed 或多场景复核。
- 有 EIDORS-aligned metrics，而不只 RMSE。
- 有 runtime/latency 数字。
- 对默认路线的替代必须证明不破坏 hot path。

当前晋升状态：

| 方法 | 当前层级 | 原因 |
|---|---|---|
| Laplace/NOSER RM | baseline / v1 | 公式与测试稳定，hot path 成立 |
| T66 TV/Huber | dynamic-quality | 高噪声快传导表现强 |
| propagation-aware A | opt-in benchmark candidate | multi-seed gate 通过，但未改默认 |
| GREIT EIDORS parity path | v1 enhancement | 48e official fixture 已过；5936 protocol official gate 仍单独 |
| SBL/BSBL | research | 缺 T70 接受 benchmark |
