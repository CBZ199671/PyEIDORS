---
status: draft
source: user-research-plus-official-doc-check
date: 2026-04-20
---

# Reference: FEniCSx/PETSc EIT Solver Refactor

## Purpose

本参考材料保存本轮底层重构的总目标、官方依据、关键设计判断和验证风险。它的作用是防止上下文压缩后丢失方向：后续代理必须先读本文件，再读对应 kit 和 plan。

## Total Goal

重构 PyEIDORS 的底层 2D/3D EIT 正逆问题实现，使其从“直接矩阵求解和显式 Jacobian 驱动”转向“DOLFINx 负责弱式/网格/装配，PETSc 负责 KSP/PC/MPI/GPU，多 RHS 复用和 matrix-free inverse 驱动”的体系。

最终状态必须满足：

- 3D forward solve 不以 LU/Cholesky/MUMPS 作为生产主力；直接法仅用于 2D、小 3D、debug、验证或 coarse/reference。
- 同一 `sigma,z` 下 CEM 系统矩阵只装配一次，KSP/PC setup 只做一次，然后求解所有电流激励模式。
- 3D inverse 默认不显式持久化 dense `J`、dense `J^T J` 或 dense GN Hessian。
- 逆问题提供 `Jv`、`J^T r`、`H v = J^T W J v + alpha R v` 操作，并可被 CG/LSQR/LSMR/GN-CG/IRGNM/LM 使用。
- 预条件分层：forward PDE 使用 Hypre/GAMG/AMGx/领域分解候选；inverse Hessian 使用 `alpha R`、NOSER diagonal、prior precision、coarse Hessian 或自定义 shell preconditioner。
- 多变量联合估计，如 `sigma + z_contact`，预留 block/fieldsplit/Schur 结构。
- GPU/MPI 路线必须显式诊断数据是否留在 PETSc/GPU 路径，不能只把最后一小步 KSP 放到 GPU 后宣称全链路 GPU。
- 每一步必须有可自动执行的验收标准，验证命令写入 plan 和 impl tracking。

## Official Documentation Findings

后续代理需要知道这些结论已经和官方文档对齐：

- DOLFINx `dolfinx.fem.petsc.LinearProblem` 使用 PETSc KSP，`NonlinearProblem` 使用 PETSc SNES，并支持 `P=` 作为预条件矩阵形式。
- DOLFINx 文档说明 `petsc_options` 只作用于底层 KSP/SNES；矩阵、向量等 PETSc 对象选项需要用户显式设置。
- PETSc KSP 文档确认 PETSc 原生 AMG 为 `PCGAMG`，也接口到 Hypre、ML、AMGx；`PCGAMG` 可用 `-pc_gamg_type agg` 等选项。
- PETSc 文档确认 `PCFIELDSPLIT` 是 block preconditioner 机制，可由 field/index set 定义块，并支持 Schur 等结构。
- PETSc 文档确认同一矩阵不同 RHS 的连续系统可以重复 `KSPSolve()`；预条件 setup 通常不会在后续 solve 中重复。
- PETSc `KSPSetReusePreconditioner` 可强制复用旧预条件器，但矩阵值变化时可能显著增加迭代次数，因此必须记录迭代数。
- PETSc `KSPMatSolve` 支持以 `MATDENSE` 存储多右端并求解多 RHS。
- PETSc `MATSHELL`/Python matrix 可表达 matrix-free 算子；但 ILU 等依赖显式矩阵条目的 PC 不能直接用于 shell matrix，必须提供显式 `Pmat` 或自定义 `PCSHELL`。
- PETSc 支持 CUDA/HIP/Kokkos/OpenCL 等 GPU 后端；GPU 成效取决于 Mat/Vec/assembly/operator apply/Jv/JTr 是否尽量避免 host-device 往返。

### Official Verification Round 2: 2026-04-20

本轮重新核对后，用户给出的 solver/preconditioner 主线保持成立，但必须加上几个强约束，避免后续实现误用：

- DOLFINx 官方 `dolfinx.fem.petsc` 文档确认：`LinearProblem` 是 PETSc KSP 高层接口，`NonlinearProblem` 是 PETSc SNES 高层接口；二者都支持 `P=` 作为 preconditioner form，并且官方示例把 `preonly + lu + mumps` 作为稳健 reference/direct path，而不是 large-3D production 默认。
- DOLFINx Stokes 官方 demo 展示了 nested/block operator、nullspace、`MINRES + fieldsplit`、上块 `GAMG`、下块 `Jacobi` 的做法。这支持我们把 CEM 约束系统、`sigma + z_contact` 联合估计、Schur/block diagonal 预条件作为官方体系内的自然升级路线。
- PETSc 官方 PCType 列表确认 `jacobi`、`bjacobi`、`ilu`、`icc`、`asm`、`gasm`、`lu`、`cholesky`、`hypre`、`fieldsplit`、`gamg`、`bddc`、`hpddm`、`amgx`、`python`、`shell` 等均是 PETSc 预条件器/直接求解器体系的一部分。
- PETSc KSP 手册确认 `PCGAMG` 是原生 AMG，并接口 Hypre、ML、AMGx；`-pc_gamg_type agg` 是官方支持的 smoothed aggregation AMG 路线。它也说明 `PCGAMG` 需要 AIJ matrix family，因此 GPU/Mat type 和 block/nest 选择必须显式诊断。
- PETSc `KSPMatSolve` 官方页确认多 RHS 可用 `MATDENSE` 表达；这与 EIT 多电流激励模式天然匹配。默认实现仍可保留 repeated `KSPSolve`，但必须只 setup 一次 KSP/PC。
- PETSc `KSPSetReusePreconditioner` 官方页确认可复用旧 PC，但矩阵数值变化后可能显著增加迭代次数。因此 preconditioner lag/reuse 只能作为显式策略，并且必须记录 iteration count。
- PETSc `MATSHELL` 官方页确认 shell matrix 适合 matrix-free；同时明确很多标准 PC 如 `PCILU` 依赖显式矩阵条目，不能直接用于 `MATSHELL`。这确认 3D inverse Hessian 的 PC 必须走 diagonal/NOSER/prior/coarse/Pmat/PCSHELL，而不是把 `GAMG/ILU` 直接套在 shell `H` 上。
- PETSc GPU 官方说明确认 CUDA/HIP/Kokkos 路线、`VECCUDA`、`MATAIJCUSPARSE`、`MATAIJKOKKOS` 等 GPU Mat/Vec 类型，并强调 CPU/GPU 数据来回拷贝慢；PCAMGX 官方页也说明 AmgX 要获得好性能，KSP 本身也必须 GPU accelerated。

Official sources checked:

- DOLFINx PETSc API: https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html
- DOLFINx Stokes demo: https://docs.fenicsproject.org/dolfinx/main/python/demos/demo_stokes.html
- PETSc KSP manual: https://petsc.org/main/manual/ksp/
- PETSc PCType manual page: https://petsc.org/main/manualpages/PC/PCType/
- PETSc PCFIELDSPLIT manual page: https://petsc.org/main/manualpages/PC/PCFIELDSPLIT/
- PETSc KSPMatSolve manual page: https://petsc.org/main/manualpages/KSP/KSPMatSolve/
- PETSc KSPSetReusePreconditioner manual page: https://petsc.org/release/manualpages/KSP/KSPSetReusePreconditioner/
- PETSc MATSHELL manual page: https://petsc.org/release/manualpages/Mat/MATSHELL/
- PETSc GPU getting-started section: https://petsc.org/release/manual/getting_started/
- PETSc PCAMGX manual page: https://petsc.org/release/manualpages/PC/PCAMGX/

### Canonical Solver/Preconditioner Strategy After Verification

This is the strategy future agents must preserve unless a benchmark or failed validation proves a change is needed:

```text
2D EIT forward:
    debug/reference:
        preonly + lu, optional MUMPS
    production:
        CG + ICC/ILU/GAMG/Hypre only when SPD or effectively SPD
        FGMRES + ILU/ASM/BJacobi/GAMG/Hypre when SPD is not guaranteed

2D EIT inverse:
    explicit J is acceptable under size guard
    dense/sparse linear algebra is acceptable for small and medium 2D
    still expose matrix-free Jv/JTr so 2D tests protect the 3D path

3D EIT forward:
    first-choice SPD path:
        CG + Hypre BoomerAMG
        CG + PETSc GAMG
    robust default when CEM/gauge/block structure is not proven SPD:
        FGMRES + GAMG
        FGMRES + Hypre BoomerAMG
    GPU candidates:
        CG + GAMG + CUDA/Kokkos
        CG/FGMRES + AmgX
    complex/indefinite/block:
        MINRES/FGMRES + fieldsplit
        ASM/GASM/BDDC/HPDDM as MPI/domain-decomposition candidates
    debug/reference:
        preonly + lu + MUMPS only for 2D, small 3D, coarse/reference

3D EIT inverse:
    do not:
        dense J
        dense J.T @ J
        direct inverse
        PC setup per current pattern
    do:
        matrix-free Jv/JTr/Hv
        GN-CG / LSQR / CGLS / LSMR / IRGNM / LM-CG
        explicit sparse/action regularization R
        P ~= alpha R + sensitivity diagonal/NOSER/prior/coarse Hessian
        forward KSP/PC reuse across RHS
        coarse inverse mesh and multi-resolution inversion

Contact impedance joint estimation:
    sigma-z block system:
        fieldsplit additive -> multiplicative -> Schur
    sigma block:
        AMG only for explicit compatible Pmat/PDE block
        prior-preconditioned CG or NOSER/prior/coarse action for inverse Hessian
    z block:
        small dense LU, Jacobi, diagonal scaling
    bad electrodes:
        robust loss, outlier rejection, measurement weighting
```

The most important correction to remember: `CG + AMG` is preferred only when the forward operator is SPD after gauge/grounding/constraint handling. If the CEM formulation is monolithic, constrained, indefinite, complex, or otherwise not proven SPD, use `FGMRES` or `MINRES` with an explicit block/fieldsplit strategy.

## Domain Interpretation for EIT

### Forward Problem

CEM forward solve 对固定 `sigma,z` 有相同系统矩阵：

```text
K(sigma, z) x_p = b_p
```

其中 `p` 是电流激励模式。因此正确循环为：

```text
for inverse iteration k:
    assemble K(sigma_k, z_k)
    setup or reuse KSP/PC
    solve all current patterns p
    extract electrode voltages
```

错误循环为：

```text
for each current pattern p:
    assemble K
    setup KSP/PC
    solve
```

### Inverse Problem

3D EIT 的未知量 `N` 可达到 `1e5` 到 `1e7`，测量数 `M` 可达到 `1e3` 到 `1e5`。dense `J in R^(M x N)` 和 dense `J^T J` 会快速失控。默认路线必须是：

```text
Jv(v)
JTr(r)
Rv(v)
Hv(v) = JTr(W * Jv(v)) + alpha * Rv(v)
```

解法优先：

- 差分成像：LSQR、CGLS、CG on normal equation。
- 绝对成像：IRGNM、LM、GN-CG、trust region 或 line search。
- 高噪声硬件数据：差分和归一化差分必须保留为 baseline；绝对成像需要建模接触阻抗、噪声协方差、异常测量、通道校准、边界/电极误差。

## Solver and Preconditioner Policy

### Forward solver presets

- 2D/small/debug: `ksp_type=preonly`, `pc_type=lu`, optional `pc_factor_mat_solver_type=mumps`。
- 3D portable default: `ksp_type=fgmres`, `pc_type=gamg`, `pc_gamg_type=agg`。
- 3D SPD optional: `ksp_type=cg`, `pc_type=gamg|hypre`。
- 3D Hypre optional: `ksp_type=fgmres|cg`, `pc_type=hypre`, `pc_hypre_type=boomeramg`。
- Indefinite/block systems: `minres|fgmres + fieldsplit`。
- Non-Hermitian/complex/nonstandard: `fgmres + hypre|gamg|fieldsplit`。

### Inverse preconditioner policy

Matrix-free Hessian cannot assume ILU/GAMG directly on shell `H`。可接受预条件器为：

- `diag(J^T W J) + alpha diag(R)`。
- NOSER diagonal。
- `alpha R` 或 prior precision 的显式稀疏近似。
- coarse inverse mesh Hessian。
- custom `PCSHELL`。
- block diagonal 或 Schur approximation，用于 `sigma-z` 联合估计。

## Existing Implementation State After First Refactor Pass

首轮重构已经完成以下内容：

- `LinearBackendConfig` 增加 `solver_preset`、`pc_hypre_type`、`pc_gamg_type`、`pc_factor_mat_solver_type`、`petsc_options`。
- 2D 默认解析为 direct；3D 默认解析为 `fgmres + gamg + agg`，支持 `3d_hypre`、`spd_gamg`、`spd_hypre`、`mumps` 等 preset。
- forward KSP 创建路径接入 PC 专项配置和 PETSc options database。
- backend signature 纳入 solver preset 和 PETSc PC/options，避免缓存误用。
- 新增 `JacobianLinearization`，提供 `Jv`、`J^T r`、`J^T W J v + alpha R v` 和 SciPy `LinearOperator`。
- `DirectJacobianCalculator.linearize()` 可返回 matrix-free sensitivity operator。

## Validation Lessons

- 项目 FEniCSx 工作流必须用 `nix develop -c uv run ...`。裸 `uv run` 在当前 `.venv` 可能触发 NumPy 导入错误。
- 单独跑少量测试时要加 `--no-cov`，否则 `pyproject.toml` 的全局覆盖率门槛会让聚焦测试失败。
- 已通过的聚焦验证命令：

```bash
nix develop -c uv run pytest --no-cov \
  tests/unit/test_forward_petsc_helper_branches.py \
  tests/unit/test_forward_solver_branch_suite.py \
  tests/unit/test_forward_mat_solve_policy.py \
  tests/unit/test_perf_capabilities_selection.py \
  tests/unit/test_gn_fast_linear_solver.py \
  tests/unit/test_forward_vectorized_runtime.py \
  tests/unit/test_forward_solve_view_semantics.py \
  tests/unit/test_adjoint_jacobian_helper_branches.py \
  tests/unit/test_jacobian_linearization.py \
  tests/unit/test_forward_solver_presets.py -q
```

结果：`53 passed`。

## Non-Goals

- 本重构不是替换 DOLFINx 弱式/网格/基础装配体系。
- 本重构不是一开始就手写完整 GPU FEM assembly。
- 本重构不是把所有 inverse 路径一次性改成 matrix-free；需要先保持 dense 路径作为 2D/debug/reference，再逐步切换 3D fast mode。
- 本重构不是删除 MUMPS；MUMPS 应保留为验证和小规模 reference。
