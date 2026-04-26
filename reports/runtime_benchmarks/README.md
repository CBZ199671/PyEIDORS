# Runtime Benchmark Index

更新时间：2026-04-26

这个目录保存运行时 benchmark、动态重建 sweep、EIDORS-aligned 复核报告。很多原始 JSON/NPZ/HDF5 类 artifact 受 `.gitignore` 管理，本文只维护关键结论和入口路径。

## 快速结论

| 主题 | 当前结论 | 关键报告 |
|---|---|---|
| 48e/5936 dual-model RM | 在线 hot path 已成立，512 帧 CUDA apply 为毫秒级批处理 | `dual_model_rm_48e_5936_t36_20260422/README.md` |
| 48e/5936 GREIT parity gate | PyEIDORS EIDORS-component path、HDF5 artifact、CPU/CUDA hot path 已通过 surrogate gate；外部 MATLAB/EIDORS fixture 未提供，官方等价声明仍不放行 | `greit_eidors_parity_48e_5936_t49_20260426/README.md` |
| 4D GN vs rowwise RM | 4D GN 改善连续测量时间保真，plant slow pulse speed error 改善明显 | `dynamic_validation_4d_gn_vs_rowwise_rm_20260425.md` |
| T66 high-noise dynamic | TV/Huber 是当前高噪声快传导整体保真首选 | `dynamic_t65_t66_t67_high_noise_sweep_20260426.md` |
| propagation-aware A | 在 T67 Kalman 内部多 seed 稳定通过 EIDORS-aligned gate | `dynamic_eidors_metric_review_propagation_A_multiseed_high_noise_20260426.md` |
| prior travelling-wave review | TV-IRLS 改善中心/速度，Laplace-family 在 RMSE/solution-error 更稳 | `prior_travelling_wave_eidors_review_20260426.md` |

## 主线报告

### 48e/5936 dual-model RM

入口：

- `dual_model_rm_48e_5936_t36_20260422/README.md`
- `dual_model_rm_48e_5936_t36_20260422/summary.json`

关键数字：

- forward reference: `spd_gamg + cuda + vec-loop`
- forward setup: `0.111076s`
- forward solve: `4.768245s`
- RM build: NOSER `84.925926s`, Laplace `73.075975s`, GREIT `2.606761s`
- 512 帧 online apply:
  - Laplace CUDA `0.036958s`
  - GREIT CUDA `0.033325s`
- GREIT 旧实现到当前 CUDA 512 帧 speedup: `1078.09x`

使用结论：

- 冷路径可以贵，在线路径必须是 `RM @ delta_v`。
- 这个报告是 v1 realtime RM 主线的主要证据。

### 48e/5936 GREIT parity gate

入口：

- `greit_eidors_parity_48e_5936_t49_20260426/README.md`
- `greit_eidors_parity_48e_5936_t49_20260426/summary.json`

关键数字：

- case: `bad_weighted`, 5936 measurements, 144 voxels, 512 frames
- bad channels: `192`, measurement W: `diagonal`
- finite-target response build: `9.923667s`
- GREIT RM component build: `0.267523s`
- HDF5 artifact write: `12.087613s`
- HDF5 artifact load: `2.265816s`
- 512 帧 online apply:
  - CPU `0.051371s`
  - CUDA `0.133575s`
- parity components: `Y/D/PJt/M/noiselev/RM/RM@dv/metrics` all passed

使用结论：

- V55..V65 的 PyEIDORS 组件路径、HDF5 schema、在线 `RM @ dv` gate 已经能系统化跑通。
- 本次未提供外部 MATLAB/EIDORS 48e fixture，报告中的 `official_equivalence_claim_allowed=false` 是故意保守；正式对外称“官方等价”仍需接入真实官方 fixture 后复跑。

### 4D GN vs rowwise RM

入口：

- `dynamic_validation_4d_gn_vs_rowwise_rm_20260425.md`
- `dynamic_validation_4d_gn_vs_rowwise_rm_20260425.json`

关键数字：

| fixture | rowwise RMSE | 4D GN RMSE | rowwise speed err | 4D GN speed err |
|---|---:|---:|---:|---:|
| travelling_wave | `0.0427355` | `0.0437737` | `0.0186226` | `0.017762` |
| plant_slow_pulse | `0.00683518` | `0.00651256` | `0.0570454` | `0.0441677` |

使用结论：

- 4D GN 不是只为形式好看，它能改善连续过程保真。
- 对慢变化和植物慢脉冲，它是合理动态 baseline。

### T65/T66/T67 high-noise sweep

入口：

- `dynamic_t65_t66_t67_high_noise_sweep_20260426.md`
- `dynamic_t65_t66_t67_high_noise_sweep_20260426.json`

关键数字：

- fixture: travelling wave, `noise_std=0.01`
- best overall: `T66 TV/Huber`
- T66 推荐区间:
  - `lambda_t 0.08..0.35`
  - `huber_delta 0.02..0.12`
- best T66:
  - score `0.137591`
  - speed error `0.0409743`
  - peak MAE `0.00691244`
  - RMSE `0.04165`
- identity-A Kalman gate-passing rows: `0/180`

使用结论：

- 高噪声快传导整体保真优先 T66 TV/Huber。
- identity-A Kalman 更像低延迟 fallback。

### propagation-aware A high-noise review

入口：

- `dynamic_t65_t66_t67_propagation_A_high_noise_20260426.md`
- `dynamic_eidors_metric_review_propagation_A_20260426.md`
- `dynamic_eidors_metric_review_propagation_A_multiseed_high_noise_20260426.md`

单 seed 动态结果：

- T67 推荐 transition: `propagation`
- T67 推荐 velocity: `0.68..0.85`
- T67 gate-passing rows: `118/720`
- best T67: `A=propagation@v=0.85`, `lag=3`, `Q=0.04`, `R=0.32`
- best T67 speed error: `0.0289882`
- best overall 仍是 `T66 TV/Huber`

多 seed EIDORS-aligned gate：

| seed | propagation-A gate |
|---:|---|
| 20260426 | pass, propagation `5/7` |
| 20260427 | pass, propagation `6/7` |
| 20260428 | pass, propagation `5/7` |
| 20260429 | pass, propagation `5/7` |
| 20260430 | pass, propagation `6/7` |

使用结论：

- propagation-aware A 是 Kalman 分支的有效增强。
- 它通过的是 T67 内部 identity vs propagation 门槛，不代表 T67 全面取代 T66。
- 当前保持 opt-in benchmark candidate。

### prior travelling-wave EIDORS review

入口：

- `prior_travelling_wave_eidors_review_20260426.md`
- `prior_travelling_wave_eidors_review_20260426.json`

关键结论：

- Laplace、graph_ltl、curvature 在该 fixture 中数值打平。
- TV-IRLS:
  - center RMSE 最好: `0.00157012`
  - speed_abs_error 最好: `0.00439478`
  - AR_error、PE、RES、RNG 也更好
- Laplace-family:
  - RMSE 更好
  - solution_error 更好

使用结论：

- TV-IRLS 是定位/传播保真增强，不是所有指标赢家。
- 新方法结论必须分指标写。

## 文件命名约定

| 前缀 | 含义 |
|---|---|
| `dual_model_rm_*` | dual-model RM / GREIT online hot path |
| `dynamic_validation_*` | T69 dynamic validation fixture |
| `dynamic_t65_t66_t67_*` | T65 4D GN、T66 TV/Huber、T67 Kalman sweep |
| `dynamic_eidors_metric_review_*` | EIDORS-aligned review over dynamic sweep JSON |
| `prior_travelling_wave_*` | Laplace/curvature/TV-IRLS prior comparison |

## 读报告顺序

如果只想恢复全局掌控，建议按这个顺序读：

1. `docs/METHOD_ROADMAP.md`
2. `docs/METHOD_SELECTION_GUIDE.md`
3. `dual_model_rm_48e_5936_t36_20260422/README.md`
4. `dynamic_validation_4d_gn_vs_rowwise_rm_20260425.md`
5. `dynamic_t65_t66_t67_high_noise_sweep_20260426.md`
6. `dynamic_eidors_metric_review_propagation_A_multiseed_high_noise_20260426.md`
7. `prior_travelling_wave_eidors_review_20260426.md`

## 维护规则

- 新 benchmark 如果改变方法选择结论，必须在本 README 增加一行。
- 只看 RMSE 的报告不能单独作为方法晋升证据。
- 动态方法至少要同时看 speed error、peak-time、onset、RMSE 或 solution-error。
- EIDORS-aligned review 应保留 per-metric winners，不要伪造单一官方总分。
- ignored 原始 artifact 可以保留在本地；长期结论写入 Markdown 或 SPEC。
