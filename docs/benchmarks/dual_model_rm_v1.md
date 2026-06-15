# FEniCSx-EIT-3D-v1 双模型 RM 基准报告

日期：2026-04-21

本文把当前 v1 重构主线固定为一条明确路线：

> 细 3D CEM 正问题网格 + 粗逆问题体素/四面体网格 + 离线重建矩阵
> `RM` + 在线 `RM @ normalize(delta_v)`。

这条 v1 路线采用 EIDORS-style dual-model 差分成像思路。matrix-free
GN-CG、IRGNM、TV、SBL、神经网络后处理继续保留为 phase-2 或 research
方向，但不阻塞 v1 在线主线。

## 范围

本报告测的是 RM 层，而不是 48 电极真实 FEniCSx Jacobian 冷启动重型路径。

benchmark 使用一个确定性的 synthetic linearized CEM-like Jacobian，用来隔离
以下成本：

- dual-mesh 投影成本
- one-step GN / NOSER / Laplace RM 构建成本
- 3D GREIT RM 构建成本
- 在线批量 `RM @ delta_v` 成本
- hot path 诊断：确认在线阶段没有 forward、adjoint、KSP 或 Jacobian rebuild

真实 FEniCSx CEM 双模型 smoke 已由 `tests/unit/test_forward_model_3d_cem.py`
单独守护。

## 可复现实验命令

```bash
nix develop .#complex64-cuda -c python scripts/benchmarks/benchmark_dual_model_rm_v1.py \
  --output-dir reports/runtime_benchmarks/dual_model_rm_v1_20260421 \
  --coarse-shape 6,6,4 \
  --fine-per-coarse 4 \
  --n-measurements 256 \
  --n-frames 512 \
  --devices cpu,auto
```

```bash
nix develop .#complex64-cuda -c python scripts/benchmarks/benchmark_dual_model_rm_v1.py \
  --output-dir reports/runtime_benchmarks/dual_model_rm_v1_20260421_batch8192 \
  --coarse-shape 6,6,4 \
  --fine-per-coarse 4 \
  --n-measurements 256 \
  --n-frames 8192 \
  --devices cpu,auto
```

主要本地 artifact：

- `reports/runtime_benchmarks/dual_model_rm_v1_20260421/summary.json`
- `reports/runtime_benchmarks/dual_model_rm_v1_20260421/forward_rm_benchmark.json`
- `reports/runtime_benchmarks/dual_model_rm_v1_20260421/greit_metrics.json`
- `reports/runtime_benchmarks/dual_model_rm_v1_20260421_batch8192/summary.json`
- `reports/runtime_benchmarks/dual_model_rm_v1_20260421_batch8192/forward_rm_benchmark.json`
- `reports/runtime_benchmarks/dual_model_rm_v1_20260421_batch8192/greit_metrics.json`

注意：`reports/` 目前受 `.gitignore` 管理，这些 JSON/NPZ 是本机可复现
artifact；长期版本化证据写在本文档和 benchmark 脚本里。

## Benchmark 配置

| 字段 | 值 |
|---|---:|
| 粗逆问题网格 | `6 x 6 x 4` voxels |
| 粗未知量 | 144 |
| 细 surrogate cells | 576 |
| `coarse2fine` nnz | 576 |
| 测量数 | 256 |
| 电极 | 48 total, 3 rings |
| 坏通道 | 9 |
| 差分模式 | normalized |
| RM 模式 | Tikhonov, NOSER, Laplace |
| GREIT targets | 每个逆问题 cell 一个 synthetic target |
| 在线设备 | `cpu`, `auto` |

本环境中 `auto` 在线设备解析为 Torch CUDA。当前公开 API 为了兼容 GUI/报告
仍返回 NumPy 数组，所以 CUDA 数字包含 host-to-device 和 device-to-host
往返成本。

## 结果

### 512 帧批处理

| 阶段 | 秒 |
|---|---:|
| 细网格 setup | 0.003116 |
| `coarse2fine` 投影 | 0.001692 |
| dense operator parity check | 0.008072 |
| one-step Tikhonov RM build | 0.021329 |
| one-step NOSER RM build | 0.007386 |
| one-step Laplace RM build | 0.009674 |
| 3D GREIT RM build | 0.031300 |
| 在线 CPU `RM @ dv`, 512 帧 | 0.054610 |
| 在线 auto/CUDA `RM @ dv`, 512 帧 | 0.155236 |

Hot-path metadata：

| 计数器 | 值 |
|---|---:|
| `forward_solve_count` | 0 |
| `adjoint_solve_count` | 0 |
| `ksp_solve_count` | 0 |
| `jacobian_rebuild_count` | 0 |

### 8192 帧批处理

| 阶段 | 秒 |
|---|---:|
| 细网格 setup | 0.002876 |
| `coarse2fine` 投影 | 0.000780 |
| dense operator parity check | 0.005580 |
| one-step Tikhonov RM build | 0.174976 |
| one-step NOSER RM build | 0.073040 |
| one-step Laplace RM build | 0.007559 |
| 3D GREIT RM build | 0.033307 |
| 在线 CPU `RM @ dv`, 8192 帧 | 0.520929 |
| 在线 auto/CUDA `RM @ dv`, 8192 帧 | 0.581530 |

对当前小型 RM 形状 `(144, 256)`，CPU NumPy 仍然略快于 CUDA。这个结果
是合理的：CUDA 路径有 host/device 往返和最终回拷 NumPy 的成本。v1 现在
真正重要的性质已经成立：在线路径是单个 batched matmul，没有 PDE solve。
更大的 RM 尺寸，或未来全 GPU 后处理链路，才是 CUDA 更可能占优的位置。

## GREIT 指标 artifact

benchmark 输出完整的 v1 GREIT 指标集合：

| 指标 | 512 帧 artifact 值 |
|---|---:|
| AR | 0.039996 |
| PE | 0.036791 |
| RES | 0.534589 |
| SD | 0.045455 |
| RNG | 0.228164 |

这些数值不是生理成像质量验收门，而是 schema 和 pipeline sanity check：
`{AR, PE, RES, SD, RNG}` 五个字段必须全部存在、有限，并通过
`write_greit_metrics_artifact` 输出。

## v1 Contract

固定几何、激励/测量协议、背景电导率/接触阻抗、坏通道 mask、噪声权重、
差分模式和正则化超参数后，v1 在线流程是：

1. 构建或加载细 CEM forward mesh。
2. 构建粗 inverse grid 或粗 inverse tetra mesh。
3. 构建 `coarse2fine`。
4. 在粗逆问题空间构建离线 RM：
   - one-step GN/Tikhonov
   - NOSER
   - Laplace prior
   - 3D GREIT
5. 用强数学签名存储 RM。
6. 在线用 `normalize_time_difference` 转换帧数据。
7. 应用同一套 bad-channel 和 measurement-weight contract。
8. 只执行 `RM @ dv` 或批量 `frames @ RM.T`。

RM cache signature 是数学签名。device/backend 不进入签名，因此同一个 RM
可以在 CPU 或 CUDA 上应用。

## 与 48 电极实验的关系

之前的 48 电极、3 层、tetra GUI 实验测到：

- dense/direct CUDA cold wall：119.70 s
- dense/direct Jacobian build：84.98 s
- warm semantic cache path：0.45 s
- CPU cold semantic context 超过 1204 s timeout 后终止

那次实验说明 full dense-J 冷重建不适合作为在线默认路径。本次 v1 RM
benchmark 给出的设计答案是：把模型相关重活放到离线阶段，在线阶段只保留
矩阵乘法。

lazy adjoint / matrix-free 仍然有研究价值，但不再是 v1 实时默认路径。它当前
的瓶颈已经从 dense Jacobian storage 转移到重复 Krylov action 成本。

## Gate Mapping

| Gate | Evidence |
|---|---|
| V25 | `DualMesh`, `coarse2fine`, real CEM smoke |
| V26 | measurement-space one-step RM |
| V27 | NOSER RM |
| V28 | Laplace prior |
| V29 | 3D GREIT RM |
| V30 | GREIT metric writer |
| V33/V39 | normalized time difference parity |
| V34/V35 | bad-channel mask and measurement weights |
| V36 | device-independent RM signature |
| V37 | online path has zero forward/adjoint/KSP/Jacobian rebuild counters |
| V38 | one-step NOSER RM parity against legacy dense baseline |
| V40 | cold RM-build and warm RM-apply times split in artifact |
| V41 | `{AR, PE, RES, SD, RNG}` all emitted |

## 决策

FEniCSx-EIT-3D-v1 默认实时差分成像路线应采用 dual-model offline RM：

- 在线主线：one-step GN/NOSER/Laplace 或 3D GREIT RM
- 在线操作：normalized difference + RM matmul
- cache 边界：RM signature，而不是 Python object identity 或 device
- phase-2：matrix-free GN-CG / IRGNM、TV、电极移动/接触阻抗联合估计

这让 3D 在线路径保持快速、可测试、可缓存，并与 EIDORS dual-model/GREIT
设计哲学一致。
