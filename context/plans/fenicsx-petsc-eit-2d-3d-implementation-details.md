---
status: ready
source: kit-derived
parent_plan: context/plans/build-site-fenicsx-petsc-eit-refactor.md
kit: context/kits/cavekit-fenicsx-petsc-eit-refactor.md
---

# Implementation Details: 2D/3D FEniCSx/PETSc EIT

## Purpose

本文件记录 2D/3D EIT 正逆问题的具体实施路线、预条件器选择、fallback 策略和验证标准。后续代理在修改 solver、Jacobian、GN runtime、GPU/MPI 或接触阻抗联合估计前，必须先读本文件。

## Global Solver Principles

1. DOLFINx/FEniCSx 负责 weak form、mesh、function space、CEM assembly、baseline correctness。
2. PETSc 负责 KSP/PC、多 RHS、MPI/GPU、diagnostics 和可配置 solver policy。
3. 同一 `sigma,z` 下 forward 矩阵只装配一次，KSP/PC 只 setup 一次，然后求解所有 current patterns。
4. Direct solver 是 debug/reference/small problem 工具，不是 large 3D production 主力。
5. 3D inverse 默认 matrix-free：不持久化 dense `J`，不构造 dense `J^T J`，不构造 dense Hessian。
6. Matrix-free operator 必须配兼容 preconditioner：diagonal/NOSER/prior/coarse/Pmat/PCSHELL，不能假设 ILU/GAMG 能直接用于 shell matrix。
7. 任何 fallback 都必须记录在 diagnostics：requested/effective solver、PC、device、Mat/Vec type、reason、iteration count。

## Canonical Solver and PC Matrix

本节是后续实现的优先级矩阵。它已经按 DOLFINx/PETSc 官方文档二次核对；除非有项目内 benchmark 或失败验收证据，否则不要改动默认方向。

### 2D EIT

调试/reference:

```text
ksp_type = preonly
pc_type  = lu
pc_factor_mat_solver_type = mumps  # optional when available
```

生产:

```text
if system is SPD after grounding/elimination:
    cg + icc/gamg/hypre
else:
    fgmres + ilu/asm/bjacobi/gamg/hypre
```

Notes:

- `ICC` 只用于对称正定/接近 SPD 的显式矩阵路径。
- `ILU` 更适合作为 2D general/serial 或 ASM/BJacobi 子块求解器。
- `GAMG/Hypre` 是 2D 到 3D 迁移的生产候选，但 2D 仍保留 LU/MUMPS reference，方便验证 weak form、边界条件、电极和 Jacobian。

逆问题:

```text
small/medium 2D:
    explicit dense/sparse J allowed
    dense solve, sparse solve, LSQR, CGLS allowed
large 2D or migration-ready path:
    expose matrix-free Jv/JTr/Hv
```

### 3D EIT Forward

首选 SPD 路线:

```text
cg + hypre(boomeramg)
cg + gamg(agg)
```

稳健默认路线:

```text
fgmres + gamg(agg)
fgmres + hypre(boomeramg)
```

GPU:

```text
cg + gamg + CUDA/Kokkos Mat/Vec
cg/fgmres + amgx
```

复杂/不定/块系统:

```text
minres + fieldsplit  # symmetric indefinite
fgmres + fieldsplit  # nonsymmetric/flexible block PC
asm/gasm/bddc/hpddm  # MPI/domain-decomposition candidates
```

Forbidden as default:

```text
large 3D production = preonly + lu/mumps
```

MUMPS remains allowed only for tiny 3D debug, 2D reference, coarse/reference solve, and parity tests.

### 3D EIT Inverse

Do not:

```text
dense J
dense J.T @ J
dense Hessian
direct inverse
PC setup per current pattern
```

Do:

```text
matrix-free Jv/JTr
Hv(v) = JTr(W * Jv(v)) + alpha * Rv(v)
GN-CG / LSQR / CGLS / LSMR / IRGNM / LM-CG
explicit sparse/action regularization R
forward KSP/PC reuse across RHS
coarse inverse mesh
multi-resolution inversion
```

Inverse Hessian preconditioner:

```text
P ~= alpha R + diag sensitivity
P ~= alpha R + NOSER diagonal
P ~= prior precision
P ~= coarse-grid Hessian/Pmat
P = custom PCSHELL action
```

Critical PETSc rule:

```text
MATSHELL H cannot directly assume PCILU/GAMG.
If using matrix-free H, provide compatible Pmat, diagonal/NOSER/prior/coarse action, or PCSHELL.
```

### Contact Impedance and Bad Electrodes

Joint `sigma-z_contact` inverse:

```text
fieldsplit additive -> fieldsplit multiplicative -> fieldsplit schur
```

Block policy:

```text
sigma block:
    prior-preconditioned CG
    NOSER/prior/coarse inverse Hessian PC
    AMG only when an explicit compatible PDE/Pmat block exists

z block:
    small dense LU
    Jacobi
    diagonal scaling

bad electrodes / bad measurements:
    robust loss
    outlier rejection
    measurement weighting
```

## 2D Forward Problem

### Mathematical model

2D forward 使用 Complete Electrode Model。基础未知量：

- domain potential `u`，通常使用 P1 Lagrange function space。
- electrode potentials `U_e`，维度为电极数。
- gauge/grounding unknown 或约束，用于处理电位零空间。
- conductivity `sigma`，当前代码常用 DG0/P0 cell-wise conductivity。
- contact impedance `z`，电极级参数。

固定 `sigma,z` 时：

```text
K_2d(sigma,z) x_p = b_p
```

`p` 为 current pattern。所有 pattern 共享 `K_2d`。

### Default implementation path

| Situation | KSP | PC | Extra options | Reason |
| --- | --- | --- | --- | --- |
| 2D debug/reference | `preonly` | `lu` | optional `pc_factor_mat_solver_type=mumps` | 最稳，用于 weak form、电极、Jacobian parity 验证 |
| 2D production small/medium, SPD after grounding/elimination | `cg` | `gamg` or `hypre` | `pc_gamg_type=agg` or `pc_hypre_type=boomeramg` | 避免直接法依赖，保持向 3D 迁移 |
| 2D production small/medium, monolithic constrained CEM | `fgmres` | `ilu`, `asm`, `bjacobi`, or `fieldsplit` | `sub_pc_type=ilu` for ASM/BJacobi | 约束/块结构可能非 SPD |
| 2D symmetric indefinite block/gauge form | `minres` | `fieldsplit` | `pc_fieldsplit_type=schur` when block metadata exists | 对称不定系统 |
| 2D non-symmetric/complex/nonstandard | `fgmres` | `gamg`, `hypre`, `asm`, or `fieldsplit` | depends on matrix | 保持 flexible Krylov |

### 2D forward preset rules

- `solver_preset=direct` 保留为 2D 默认和 debug reference。
- `solver_preset=mumps` 只在 MUMPS 可用且用户明确请求时使用。
- `solver_preset=spd_gamg` 仅在系统确认为 SPD 时使用。
- `solver_preset=3d_gamg` 可在 2D 测试 AMG portability，但不要覆盖 debug direct reference。

### 2D forward validation

Required gates:

- Gate 1: compile changed forward files.
- Gate 2: `tests/unit/test_forward_solver_presets.py`
- Gate 2: `tests/unit/test_forward_vectorized_runtime.py`
- Gate 2: `tests/unit/test_forward_solver_branch_suite.py`
- Gate 3: small 2D CEM forward parity against SciPy/direct reference.

Acceptance thresholds:

- output voltages finite。
- repeated solves with identical `sigma,z,patterns` deterministic。
- PETSc/SciPy small 2D parity within existing tolerances。
- one `forward_solve(sigma)` must not rebuild KSP per stimulation pattern。

## 3D Forward Problem

### Mathematical model

3D forward 与 2D 共享 CEM 结构，但 DOF、nnz、边界电极面积和局部加密成本显著增加。3D 中 `h` 减半通常单元数约 `x8`，直接法 fill-in 会超线性增长。

固定 `sigma,z` 时：

```text
K_3d(sigma,z) X = B
```

其中 `B` 包含所有 stimulation RHS。实现上可以是 repeated `KSPSolve` 或 `KSPMatSolve(MATDENSE)`。

### Default implementation path

| Situation | KSP | PC | Extra options | Reason |
| --- | --- | --- | --- | --- |
| 3D production portable default | `fgmres` | `gamg` | `pc_gamg_type=agg`, `mg_levels_ksp_type=chebyshev`, `mg_levels_pc_type=jacobi` | CEM/gauge/constraints 可能不严格 SPD；flexible AMG 默认更稳 |
| 3D SPD after grounding/elimination | `cg` | `hypre` | `pc_hypre_type=boomeramg` | 椭圆 SPD 路径优先，Hypre BoomerAMG 常用于大规模 |
| 3D SPD PETSc-native | `cg` | `gamg` | `pc_gamg_type=agg` | 原生 PETSc AMG，便携 |
| 3D symmetric indefinite block/gauge | `minres` | `fieldsplit` | `pc_fieldsplit_type=schur` | 约束系统，不适合 CG |
| 3D non-symmetric/complex/nonstandard | `fgmres` | `hypre`, `gamg`, `asm`, `gasm` | as needed | 非 Hermitian/非对称模型 |
| MPI/domain decomposition candidate | `fgmres` or `cg` | `asm`, `gasm`, `bddc`, `hpddm` | subdomain solver explicit | 大规模并行扩展 |
| GPU candidate | `cg` or `fgmres` | `gamg` or `amgx` | `mat_type=aijcusparse`, `vec_type=cuda` | 仅在 PETSc CUDA capability 通过时 |
| 3D debug/reference only | `preonly` | `lu` | `pc_factor_mat_solver_type=mumps` | 小网格验证，不做 production 主力 |

### 3D forward assembly and reuse loop

Required loop:

```text
for inverse iteration k:
    sigma_k, z_k are fixed
    assemble K_3d(sigma_k, z_k)
    create/setup KSP and PC once
    solve all RHS patterns
    record iterations and diagnostics
```

Forbidden production loop:

```text
for pattern p:
    assemble K_3d
    setup KSP/PC
    solve b_p
```

### 3D forward KSP reuse policy

- Same matrix, different RHS: reuse same KSP/PC automatically。
- Changed `sigma,z` but same sparsity: allow `reuse_preconditioner=true` only as explicit policy; diagnostics must record iteration count and fallback reason。
- If reuse increases iteration count past threshold, implementation must allow disabling reuse。

### 3D forward validation

Required gates:

- Gate 2: preset parsing and diagnostics tests。
- Gate 2: multi-RHS policy tests。
- Gate 3: small 3D CEM AMG smoke。
- Gate 4: benchmark artifact。

Benchmark artifact must include:

```text
mesh_dim
n_cells
n_dofs
n_elec
n_patterns
solver_preset
ksp_type
pc_type
pc_subtype
mat_type
vec_type
setup_seconds
solve_seconds
iterations_per_rhs
mat_solve_effective
petsc_device_requested
petsc_device_effective
fallback_reason
```

## 2D Inverse Problem

### Default reconstruction policy

2D 可以保留 dense Jacobian 路径作为 reference 和 production 候选，因为问题规模通常可控。但所有新设计必须保持 matrix-free-compatible，避免 2D 逻辑阻碍 3D。

| Inverse mode | Jacobian policy | Linear solve | Preconditioner/regularization | Notes |
| --- | --- | --- | --- | --- |
| 2D difference baseline | explicit dense or sparse `J0` acceptable | dense solve, sparse solve, LSQR, CGLS | NOSER diag, Tikhonov, smoothness `R` | 真实硬件 baseline 优先 |
| 2D normalized difference | explicit `J0` acceptable | same as difference | measurement weights + NOSER/Tikhonov | 处理增益/幅值漂移 |
| 2D absolute GN | explicit `J` acceptable for small/medium | GN/LM with dense/sparse linear solve | `J^T W J + alpha R` | 保留 line search |
| 2D absolute robust | explicit or operator | GN-CG/LM | robust weights + prior | 用于坏电极/异常测量 |
| 2D matrix-free compatibility | `Jv/J^T r` | LSQR/CGLS/GN-CG | diagonal/NOSER/prior | 为 3D 迁移验证 |

### 2D inverse preconditioner split

For conductivity-only inverse:

- first choice: NOSER diagonal for difference EIT。
- smoothness/Tikhonov: sparse `R` or `R^T R`。
- if using PCG: diagonal inverse, PyAMG on sparse `R`, CHOLMOD on sparse SPD `R` when available。
- dense direct/Cholesky allowed only when size guard passes。

For `sigma + z_contact` joint inverse:

| Block | Size | Suggested PC | Notes |
| --- | --- | --- | --- |
| `sigma` | cells/elements | NOSER diag, sparse `R`, PyAMG/CHOLMOD on `R`, matrix-free PCG | 不要默认 dense Hessian for large models |
| `z_contact` | electrode count | dense LU, Jacobi, diagonal scaling | 小维度 |
| coupling | rectangular low-rank-ish | block diagonal first, Schur approximation later | fieldsplit upgrade path |

T-FPX-009 implementation contract:

- `pyeidors.inverse.block_system.build_sigma_contact_block_metadata()` defines contiguous `sigma` and `z_contact` parameter blocks, measurement Jacobian coupling shapes, Hessian coupling shapes, regularization labels, and a PETSc-style fieldsplit plan.
- `make_block_diagonal_inverse_action()` provides the first shape-safe block diagonal inverse action. It is an initial approximation for tests and future integration, not a production Schur solver.
- `scale_contact_impedance_update()` applies a finite, positive, globally scaled `z_contact` update so later joint solvers have a tested update guard.
- Future agents must attach PETSc `PCFIELDSPLIT`/Schur wiring at this metadata boundary instead of merging `sigma` and `z_contact` into an opaque dense monolithic vector.

### 2D inverse validation

- Gate 2: dense Jacobian shape and finite tests。
- Gate 2: NOSER/smoothness regularization finite diagonal/lower-bound tests。
- Gate 3: small 2D dense vs matrix-free parity。
- Gate 4: optional runtime comparison for dense direct vs CGLS/LSQR。

Acceptance thresholds:

- reconstruction update finite。
- dense and matrix-free `Jv/J^T r` agree within numeric tolerance。
- no size-guarded direct solver runs when estimated memory exceeds configured threshold。

## 3D Inverse Problem

### Default reconstruction policy

3D inverse production path must be matrix-free. Explicit dense `J` is allowed only for:

- tiny 3D debug mesh。
- algorithm parity tests。
- cached startup reference when size guard explicitly permits。
- paper/diagnostic artifact generation with clear memory budget。

| Inverse mode | Jacobian policy | Linear solve | Preconditioner | Notes |
| --- | --- | --- | --- | --- |
| 3D difference baseline | matrix-free preferred; dense only under size guard | LSQR, CGLS, CG normal eq | NOSER diag + alpha R/prior | hardware baseline |
| 3D normalized difference | matrix-free | LSQR/CGLS | measurement weights + NOSER/prior | handles calibration drift |
| 3D absolute GN | matrix-free `Jv/J^T r` | GN-CG or LM-CG | `alpha R`, prior precision, coarse Hessian | no dense `J^T J` |
| 3D IRGNM | matrix-free | decreasing-alpha GN-CG | prior-conditioned Krylov | preferred absolute path |
| 3D robust inverse | matrix-free | reweighted GN-CG | robust weights + prior | bad electrode/outlier handling |
| 3D reduced-order | projected operator | reduced GN/low-rank | reduced preconditioner | opt-in only |

### 3D sensitivity implementation

Minimum required operator:

```text
Jv(v):
    apply conductivity perturbation direction v
    compute forward sensitivity action for all measurement rows
    return measurement-space vector

JTr(r):
    convert measurement residual r to adjoint current patterns
    solve adjoint fields using reused forward operator when compatible
    accumulate element-wise gradient
    return parameter-space vector

Hv(v):
    y = Jv(v)
    y = W * y
    out = JTr(y)
    out += alpha * Rv(v)
    return out
```

Current transitional implementation may cache gradients in `JacobianLinearization`; later high-scale implementation should replace gradient storage with PDE/adjoint actions and batching.

### 3D inverse preconditioning

Allowed PC choices:

| PC | When to use | Validation |
| --- | --- | --- |
| diagonal `diag(J^T W J) + alpha diag(R)` | first matrix-free baseline | finite positive lower bound |
| NOSER diagonal | difference EIT baseline | parity with dense small mesh |
| `alpha R` sparse/prior precision | smoothness/Bayesian prior | sparse SPD/action test |
| coarse Hessian/Pmat | large 3D with coarse inverse mesh | small smoke + diagnostics |
| custom `PCSHELL` | PETSc shell operator | apply test + convergence diagnostics |
| block diagonal `sigma,z` | joint inverse initial path | block shape tests |
| Schur approximation | advanced fieldsplit | integration test |

Disallowed default:

- dense `J^T J` in 3D fast path。
- ILU/GAMG directly on matrix-free shell Hessian without explicit `Pmat`。
- direct dense solve when `n_param` or memory estimate exceeds guard。

### 3D inverse validation

Required gates:

- Gate 2: `JacobianLinearization` operator unit tests。
- Gate 2: GN fast solver accepts operator and does not require dense `J`。
- Gate 3: tiny 3D dense vs matrix-free parity。
- Gate 4: benchmark records memory estimate and selected solver path。
- Gate 6: manual audit confirms no hidden dense 3D default path。

Acceptance thresholds:

- `Jv` and `J^T r` finite。
- `Hv` finite and shape-compatible。
- CG/LSQR iteration count and residual norm recorded。
- fallback reason recorded if dense reference is used。

## Contact Impedance Joint Estimation

### Target block structure

Joint inverse should be represented as:

```text
[ H_sigma_sigma  H_sigma_z ] [ delta_sigma ] = [ g_sigma ]
[ H_z_sigma      H_z_z     ] [ delta_z     ]   [ g_z     ]
```

Initial production-safe route:

```text
P_block = diag(P_sigma, P_z)
P_sigma = NOSER/prior/coarse/sparse R approximation
P_z     = dense LU or Jacobi
```

Upgrade route:

```text
fieldsplit additive -> fieldsplit multiplicative -> fieldsplit schur
```

### Contact impedance validation

- Gate 2: block metadata shape test。
- Gate 2: `z` update finite and scaled。
- Gate 3: small 2D/3D synthetic joint inverse smoke。
- Gate 6: human review of physical parameterization and scale。

## GPU/MPI Implementation Details

### PETSc GPU policy

GPU path requires capability probe:

```bash
nix develop .#cuda -c uv run python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty
```

GPU PETSc object policy:

- matrix type: `aijcusparse` or detected PETSc CUDA AIJ type。
- vector type: `cuda`。
- dense multi-RHS type: CUDA dense type only if probe confirms。
- PC: `gamg` or `amgx` candidate; Hypre GPU only if runtime proves support。

### GPU acceptance criteria

- `petsc_device=cuda` fails fast if CUDA Mat/Vec unavailable。
- `petsc_device=auto` records fallback reason。
- benchmark says whether forward assembly, KSP, Jv, JTr, regularization, and residual are CPU/GPU/mixed。
- no local/host-device proxy should be called “full GPU” unless data remains on device through the hot path。
- Current diagnostic implementation:
  - `scripts/diagnostics/probe_petsc_cuda.py --pretty` reports PETSc CUDA
    Mat/Vec/Dense probe results and an `mpi` section from
    `probe_mpi_runtime()`。
  - `EITForwardModel.get_backend_diagnostics()` records
    `petsc_device_requested/effective`, PETSc Mat/Vec/Dense type,
    `gpu_fallback_reason`, `gpu_transfer_risk`, and MPI size/rank/support
    fields。
  - `forward_solver_benchmark` records CUDA availability/errors and MPI
    support fields, so CPU fallback and single-rank limitation are visible in
    performance artifacts。

### MPI policy

Current known state:

- Existing code has single-rank restriction in forward model initialization。
- `probe_mpi_runtime()` is the canonical runtime diagnostic for this boundary:
  `mpi_size > 1` sets
  `mpi_fallback_reason=mpi_size_gt_1_not_supported_phase2_single_rank_only`。
- `EITForwardModel` fails fast for MPI size > 1 and includes detected
  `mpi_size`, `mpi_rank`, and the fallback reason in the exception message。
- Until removed and tested, MPI > 1 must be treated as unsupported for production。

MPI upgrade acceptance:

- remove size=1 restriction intentionally。
- use distributed Mat/Vec kinds。
- run at least `mpiexec -n 2` smoke。
- domain decomposition candidates: `asm`, `gasm`, `bddc`, `hpddm`。

## Solver Decision Checklist

Before selecting a solver, answer:

1. Is the assembled system SPD after gauge/grounding handling?
   - yes: `cg + gamg/hypre`
   - no/unknown: `fgmres + gamg/hypre` or `minres + fieldsplit` if symmetric indefinite
2. Is this 3D large production?
   - yes: no direct LU as default
3. Are there many RHS?
   - yes: one KSP/PC setup, repeated `KSPSolve` or `KSPMatSolve`
4. Is inverse problem 3D?
   - yes: matrix-free `Jv/J^T r/Hv`
5. Is preconditioner compatible with matrix-free operator?
   - if no explicit Pmat/PCSHELL: use diagonal/NOSER/prior/coarse action
6. Is CUDA requested?
   - run capability probe and record effective Mat/Vec/backend

## Next Implementation Targets

Priority order:

1. T-FPX-011: create sharded validation commands so future work has strict evidence。
2. T-FPX-006: make GN fast linear solver consume `JacobianLinearization`/`LinearOperator`。
3. T-FPX-003: harden forward KSP/multi-RHS reuse tests。
4. T-FPX-007: define matrix-free preconditioner action contract。
5. T-FPX-004: benchmark forward solver presets and diagnostics。
