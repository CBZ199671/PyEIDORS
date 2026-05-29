# WSL2 CUDA 开发与验证路径

本仓库现在把 **CUDA 路径** 明确建模为独立 dev shell，但 GUI 默认入口会自动选择最宽能力路线：

```bash
nix develop .#cuda
```

默认 `nix develop` 仍然是实数 CPU 开发路径；`.#cuda` 是实数 CUDA，`.#complex64-cuda` / `.#complex-cuda` 是复数 CUDA。GUI 的 `auto` 启动器会优先进入复数能力 GPU 路线，让实数和复数输入共用同一个上位机入口。

GUI 官方启动方式：

```bash
bash scripts/gui/run_eit_app.sh
```

这个启动器默认 `--auto`：

- WSL2 能看到 NVIDIA GPU → `.#complex64-cuda`
- 没有 GPU → `.#complex64`
- 需要更高精度 → 加 `--precision complex128`
- 只做专家级实数性能对比 → 用 `--real-gpu` 或 `--real-cpu`

启动器会补齐 GUI 运行所需的 `repo root + src + nix runtime` 路径；需要强制启动前执行 CUDA PETSc probe 时加 `--probe-cuda`。

## 官方依据

本实现和文档只依赖官方资料，不基于社区传言：

- DOLFINx Python PETSc API：
  `https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html`
  - 这里的 `create_matrix(...)` / `assemble_matrix(...)` / `create_vector(...)` / `assemble_vector(...)` 明确支持 `kind`，可为 PETSc Mat/Vec 指定后端类型。
- PETSc CUDA sparse matrix manual pages：
  - `https://petsc.org/release/manualpages/Mat/MATSEQAIJCUSPARSE/`
  - `https://petsc.org/release/manualpages/Mat/MATMPIAIJCUSPARSE/`
- PETSc CUDA vector manual page：
  - `https://petsc.org/release/manualpages/Vec/VECCUDA/`
- PETSc 官方安装文档（CUDA 构建选项）：
  - `https://petsc.org/release/install/install/`

这几个官方点共同限定了当前仓库的 GPU 方案边界：

1. DOLFINx Python 层可以把矩阵/向量创建委托给 PETSc 的指定 `kind`。
2. 真正能不能用 `aijcusparse` / `cuda`，取决于 **PETSc 是否编译进 CUDA backend**。
3. 因此“只改 Python 代码”不够；必须有 CUDA 运行时入口。GUI 默认使用复数能力 CUDA 作为 superset，避免用户手动判断 real/complex profile。

## 当前仓库里的 CUDA 开关

### 1) PETSc 设备策略

`EITSystem` 和统一 CLI 现在支持：

- `petsc_device="auto"`
- `petsc_device="cpu"`
- `petsc_device="cuda"`
- `device="auto"`
- `device="cpu"`
- `device="cuda"`

统一 CLI 对应参数：

```bash
--petsc-device {auto,cpu,cuda}
--device {auto,cpu,cuda}
```

语义如下：

- `cpu`：强制走 CPU PETSc/FEM 路径。
- `auto`：如果当前 shell 的 PETSc CUDA probe 成功，则启用 CUDA；否则自动回退 CPU，并把原因写入 diagnostics。
- `cuda`：要求 PETSc CUDA 真可用；否则直接报错，不再静默回退。

`device` 的语义与之对应，但只控制 Jacobian/Torch/GN inverse runtime：

- `cpu`：强制 inverse runtime 使用 CPU。
- `auto`：仅当 `petsc_device_effective == cuda` 且 `torch.cuda.is_available()` 时启用 CUDA，否则回退 CPU。
- `cuda`：要求 Torch CUDA 真可用；否则直接报错。

### 2) Probe 脚本

仓库新增了一个“真创建对象”的 PETSc CUDA probe：

```bash
python scripts/diagnostics/probe_petsc_cuda.py --pretty
python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty
```

这个 probe 不只检查枚举名是否存在；它会实际尝试创建：

- `PETSc.Mat.Type.AIJCUSPARSE`
- `PETSc.Vec.Type.CUDA`
- `PETSc.Mat.Type.DENSECUDA`

因此它能识别一种常见误判：**符号存在，但运行时创建时报 `Unknown type`**。这种情况会被判定为“CUDA 不可用”。

## 推荐工作流

### CPU 默认路径

```bash
nix develop
python scripts/env/verify_env_manifest.py
python -c "import pyeidors; print(pyeidors.check_environment())"
```

如果你要在当前 WSL2 机器上验证 CPU strict 参考，可以直接跑：

```bash
python scripts/benchmarks/benchmark_3d_runtime.py \
  --solver-mode strict \
  --repeat 1 \
  --perf-report reports/benchmark_3d_runtime_cpu_strict.json
```

这里的 `solver_mode="strict"` 对外语义没有变化。对于 3D `gn-difference` + NOSER 对角正则，只有在 dense strict 内存守卫触发时，内部后端才会从 `dense-param` 自动切到代数等价的 `measurement-exact`。这表示 strict 仍然是精确参考求解，只是避免了参数空间 dense `JᵀJ + λ diag(R)` 的大块内存分配；它不是 fast fallback，也不是近似 iterative strict。2D、小规模 3D、以及不触发守卫的 case 仍保持 `dense-param`。

CPU strict benchmark 报告应重点看：

- `difference_solver.strict_solver_backend_effective`
- `difference_solver.strict_memory_guard_triggered`
- `difference_solver.strict_measurement_system_shape`

当这些字段显示 `measurement-exact`、`true`、以及一个 measurement-space 形状时，表示当前 3D difference strict 使用的是低内存精确后端。`absolute` strict 仍走原主路径，不受 difference strict fallback 污染。

### CUDA 路径

```bash
nix develop .#cuda
python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty
```

如果 probe 通过，可以继续跑 3D benchmark：

在 CPU/CUDA 对照时，CPU strict 报告也应按上面的 difference diagnostics 来解释：`measurement-exact` 表示 strict 参考在内存守卫下切到了精确的 measurement-space 后端，而不是退回 fast；如果字段仍是 `dense-param`，则表示 CPU strict 仍走原 dense 参考路径。

```bash
python scripts/benchmarks/benchmark_3d_runtime.py \
  --petsc-device auto \
  --device auto \
  --forward-mat-solve auto \
  --perf-report reports/benchmark_3d_runtime_cuda.json
```

或者显式强制 CUDA：

```bash
python scripts/benchmarks/benchmark_3d_runtime.py \
  --petsc-device cuda \
  --device cuda \
  --forward-mat-solve auto \
  --perf-report reports/benchmark_3d_runtime_cuda_strict.json
```

统一 CLI 也可直接指定：

```bash
python scripts/run_reconstruction_unified.py \
  --method gn-difference \
  --csv data/example.csv \
  --output-root results/demo \
  --mesh-dim 3 \
  --petsc-device auto \
  --device auto \
  --dry-run
```

## 环境锁与 profile

CUDA 路径现在支持单独的 manifest profile 命名：

```bash
python scripts/env/export_env_manifest.py \
  --profile cuda \
  --output env/manifests/linux-x86_64-cuda.lock.json

python scripts/env/verify_env_manifest.py --profile cuda
```

默认 CPU profile 仍然使用：

- `env/manifests/linux-x86_64.lock.json`

而 CUDA profile 约定使用：

- `env/manifests/linux-x86_64-cuda.lock.json`

如果你当前只是进入了普通 `nix develop`，不要导出 CUDA manifest；那会把 CPU-only 运行时误记成 GPU profile。

## 已实现的运行时行为

本轮代码改动已经把以下链路接上：

- Forward PETSc 路径支持 `petsc_device=auto|cpu|cuda`。
- GN inverse runtime 路径支持 `device=auto|cpu|cuda`，并把 `inverse_device_*` / `execution_profile` 写入 diagnostics。
- CUDA profile 下会优先尝试：
  - PETSc sparse matrix：`AIJCUSPARSE`
  - PETSc vector：`CUDA`
  - 多 RHS `matSolve` 的 dense mat：`DENSECUDA`（如果运行时可用）
- Forward factor cache key 已纳入 PETSc 设备与后端信息，避免 CPU/GPU 误共享。
- Direct Jacobian 的大块收缩现在可在 Torch CUDA 上按阈值切换；小问题仍保留 CPU/NumPy 路径。
- Absolute runtime / benchmark diagnostics 已新增 GPU 相关字段：
  - `petsc_device_requested`
  - `petsc_device_effective`
  - `petsc_mat_type`
  - `petsc_vec_type`
  - `gpu_fallback_reason`
  - `forward_factor_backend`
  - `forward_mat_solve_effective`
  - `inverse_device_requested`
  - `inverse_device_effective`
  - `execution_profile`
  - `jacobian_backend_requested`
  - `jacobian_backend_effective`
  - `jacobian_block_backend`
  - `jacobian_transfer_estimate`
  - `jacobian_cuda_threshold_hit`

## 当前边界与预期

这条 GPU 路线的目标是：

- 优先把 **3D forward-heavy** 的阶段推到 GPU：正问题、多 RHS、伴随求解、line-search 内 forward 调用。
- 再把 Jacobian 的大块张量收缩放到 Torch CUDA。
- 不承诺 DOLFINx Python 层所有 `Expression/interpolate` 都是纯 device-native。
- 不改变对外公共数据结构：`EITData` / `EITImage` 仍然对外暴露 NumPy。

因此，如果你在 `.#cuda` shell 下看到：

- probe 成功；
- benchmark 的 `forward_factor_backend`、`petsc_mat_type`、`petsc_device_effective` 都显示为 CUDA 相关；
- 端到端总时间优于当前 CPU fast；

才说明这条 GPU 路线真正开始带来收益。

如果只有部分阶段变快、但端到端总耗时没有改善，就应把它继续视为 **experimental profile**，而不是默认推荐路径。
