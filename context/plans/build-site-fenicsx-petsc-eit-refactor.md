---
status: ready
source: kit-derived
kit: context/kits/cavekit-fenicsx-petsc-eit-refactor.md
complexity: thorough
---

# Build Site: FEniCSx/PETSc EIT Solver Refactor

## Objective

把 PyEIDORS 的底层求解器演进为官方 DOLFINx/PETSc 风格：forward 使用 KSP/PC/multi-RHS reuse，3D production 默认 AMG-family iterative solver，inverse 逐步迁移到 matrix-free `Jv/J^T r/Hv`，并用严格 validation gates 约束每一步。

## Required Reading

1. `context/refs/fenicsx-petsc-eit-refactor-research.md`
2. `context/kits/cavekit-fenicsx-petsc-eit-refactor.md`
3. `context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md`
4. `context/kits/cavekit-forward-solver.md`
5. `context/kits/cavekit-inverse-reconstruction.md`
6. `context/impl/impl-fenicsx-petsc-eit-refactor.md`

## Execution Rules

- 代码任务必须在 WSL2 Ubuntu 项目目录执行。
- FEniCSx/PETSc 验证必须使用 `nix develop -c uv run ...`。
- 聚焦测试默认加 `--no-cov`；覆盖率门槛只在专门 coverage pass 中验证。
- 每个 task 完成或失败后必须更新 implementation tracking。
- 不允许把 large 3D production path 静默回退到 direct LU。
- 不允许把 dense `J` 或 dense `J^T J` 作为 3D fast/inverse 默认路径。
- 2D/3D forward/inverse 的具体 solver 和 preconditioner 选择以
  `fenicsx-petsc-eit-2d-3d-implementation-details.md` 为准。

## Task Graph

| Task ID | Requirement | Task | Status | Validation Gates |
| --- | --- | --- | --- | --- |
| T-FPX-001 | R1 | 完成 forward PETSc solver preset 体系，并保持 2D direct reference。 | DONE | Gate 1, Gate 2 |
| T-FPX-002 | R1,R9 | 把 solver preset、PC subtype、PETSc options 纳入 backend/cache signature。 | DONE | Gate 2 |
| T-FPX-003 | R2,R3 | 加强 forward 多 RHS/KSP reuse 行为测试和 3D AMG smoke。 | DONE | Gate 2, Gate 3 |
| T-FPX-004 | R3,R8 | 建立 forward solver benchmark artifact：setup/solve/iteration/RHS/device diagnostics。 | DONE | Gate 4 |
| T-FPX-005 | R4 | 完成 `JacobianLinearization` 的 `Jv/J^T r/Hv` operator 层。 | DONE | Gate 1, Gate 2 |
| T-FPX-006 | R4,R5 | 将 GN fast linear solver 接入 `JacobianLinearization`/`LinearOperator` 输入。 | DONE | Gate 2, Gate 3 |
| T-FPX-007 | R5,R6 | 实现 matrix-free Hessian action 的 diagonal/NOSER/prior preconditioner 策略。 | DONE | Gate 2 |
| T-FPX-008 | R6 | 设计并 smoke Pmat/coarse Hessian 或 custom PC path。 | DONE | Gate 2, Gate 3 |
| T-FPX-009 | R7 | 为 `sigma + z_contact` 联合估计定义 block-ready interface 和 fieldsplit plan。 | DONE | Gate 2, Gate 6 |
| T-FPX-010 | R8 | 强化 CUDA/MPI capability diagnostics 和 fallback reporting。 | DONE | Gate 2, Gate 4 |
| T-FPX-011 | R10 | 建立分片验证脚本/命令清单，避免全量 unit 超时后没有可恢复信息；原 `gui-hardware` 拆成 default `gui` 与 opt-in `hardware`。 | DONE | Gate 1, Gate 2, Gate 6 |
| T-FPX-012 | R10,R11 | 每轮实施后更新 impl tracking，包括死路、未验证项和 canonical solver matrix 变更。 | DONE | Gate 6 |

## Implementation Sequence

### Phase 0: Stabilize Current First Pass

目标：确认首轮已完成内容不会在上下文压缩后丢失。

Tasks:

- T-FPX-001
- T-FPX-002
- T-FPX-005
- T-FPX-012

Required validation:

```bash
nix develop -c uv run python -m compileall -q \
  src/pyeidors/forward/eit_forward_model.py \
  src/pyeidors/cache/object_signature.py \
  src/pyeidors/inverse/jacobian/direct_jacobian.py \
  src/pyeidors/inverse/jacobian/linearized.py
```

```bash
nix develop -c uv run pytest --no-cov \
  tests/unit/test_forward_solver_presets.py \
  tests/unit/test_jacobian_linearization.py \
  tests/unit/test_forward_petsc_helper_branches.py \
  tests/unit/test_forward_solver_branch_suite.py \
  tests/unit/test_forward_mat_solve_policy.py \
  tests/unit/test_forward_vectorized_runtime.py \
  tests/unit/test_gn_fast_linear_solver.py -q
```

Exit criteria:

- compileall exit code 0。
- targeted tests pass。
- impl tracking updated with command results。

### Phase 1: Forward KSP/PC Reuse Hardening

目标：让 forward solve 的“assemble once, setup once, solve all RHS”成为可测试 contract。

Tasks:

- T-FPX-003
- T-FPX-004

Implementation notes:

- 用 fake PETSc 或 instrumentation 记录一次 `forward_solve(sigma)` 中 `_create_full_matrix_petsc`、`_make_petsc_solver_bundle`、`ksp.solve` 或 `ksp.matSolve` 的调用次数。
- 对小 3D mesh 增加 AMG/Hypre smoke；如果 runtime 没有 Hypre，测试应 skip 或检查 fallback reason，不能失败得不明不白。
- benchmark artifact 至少包含：`mesh_dim`、`n_dofs`、`n_patterns`、`solver_preset`、`ksp_type`、`pc_type`、`pc_subtype`、`setup_seconds`、`solve_seconds`、`iterations`、`mat_solve_effective`、`petsc_device_effective`、`fallback_reason`。
- T-FPX-003 completion note: fake PETSc call-count now verifies one matrix/bundle setup and matSolve/vector-loop solve counts; 3D PETSc `3d_gamg` smoke verifies finite output, GAMG non-LU PC, multi-RHS diagnostics, KSP setup count, and preconditioner reuse request/applied flags. Benchmark artifact remains T-FPX-004.
- T-FPX-004 completion note: `benchmark_3d_runtime.py --forward-only on` emits a normalized `forward_solver_benchmark` JSON block with setup/solve seconds, KSP iterations, KSP convergence reason, Mat/Vec type, device/fallback diagnostics, and finite-output status. Quick artifact: `test_results/benchmarks/tfpx004_forward_solver.json`; `/ck:check` artifact: `test_results/benchmarks/tfpx004_check_forward_solver.json`.

Validation commands:

```bash
nix develop -c uv run pytest --no-cov \
  tests/unit/test_forward_solver_presets.py \
  tests/unit/test_forward_solver_branch_suite.py \
  tests/unit/test_forward_mat_solve_policy.py \
  tests/unit/test_forward_vectorized_runtime.py \
  tests/unit/test_forward_petsc_multirhs.py \
  tests/unit/test_forward_petsc_helper_branches.py \
  tests/unit/test_forward_model_3d_cem.py -q
```

Benchmark gate:

```bash
nix develop -c uv run python scripts/benchmarks/benchmark_3d_runtime.py --help
```

If a real benchmark command is too expensive, document the chosen quick benchmark and output file in impl tracking.

### Phase 2: Matrix-Free GN Runtime Integration

目标：GN fast path 能消费 operator，不强制要求 dense `measurement_jacobian_np`。

Tasks:

- T-FPX-006
- T-FPX-007

Implementation notes:

- 定义输入 contract：dense ndarray、SciPy `LinearOperator`、`JacobianLinearization` 都可进入 fast linear solve。
- 保留 dense path 作为 2D/debug/reference。
- 对小 synthetic problem 比较 dense 和 operator 解。
- 对 regularization 支持 sparse matrix、LinearOperator、callable action。
- solver diagnostics 必须记录 `jacobian_representation = dense|linear_operator|jacobian_linearization`。

Validation commands:

```bash
nix develop -c uv run pytest --no-cov \
  tests/unit/test_jacobian_linearization.py \
  tests/unit/test_gn_fast_linear_solver.py \
  tests/unit/test_gn_runtime_helper_branches.py -q
```

Exit criteria:

- dense/reference tests still pass。
- operator path has at least one direct unit test。
- No default 3D fast path constructs dense `J^T J` without explicit reference/debug mode。

Completion note 2026-04-20:

- `_solve_linear_system_fast()` now normalizes dense ndarray, SciPy `LinearOperator`, and `JacobianLinearization` into `Jv/J^T r` actions.
- Operator mode supports measurement weighting, prior term, callable/LinearOperator/sparse regularization action, and PCG/LSMR Hessian action without dense `J^T J`.
- Solver diagnostics include `jacobian_representation`, `jacobian_shape`, `dense_jacobian_materialized`, and `linear_iterations`.
- Dense Woodbury/ROM/cholmod debug routes remain dense-only and report fallback reasons when given operator input.
- `/ck:check` follow-up wired real GN runtime selection via `jacobian_method=linearized|operator|matrix-free`; operator route now calls `jacobian_calculator.linearize()`, skips dense startup cache/projection, and sends measurement weights to the fast solver.
- Extra 2D/3D real-ish FEM smoke added in `tests/unit/test_gn_linearized_real_smoke.py` to pin `jacobian_method="linearized"` on tagged 2D and 3D CEM meshes.
- T-FPX-007 adds a matrix-free PC contract for `diag`/`noser`/`prior`: operator Hessian PCG now uses finite positive diagonal approximations from explicit Hessian diag, explicit matrix-free diag, NOSER diag, prior/R diag, or identity fallback; metadata records source, floor, min/max, Pmat availability, and fallback reason.
- `petsc-gamg` is no longer treated as a direct preconditioner for matrix-free Hessian without explicit `Pmat`; it falls back to diagonal PC and records `petsc_gamg_not_supported_in_matrix_free`.
- User-facing preconditioner validation now accepts `noser` and `prior` in GN runtime/CLI paths; dense 2D/reference path remains unchanged.
- T-FPX-008 adds the second-layer compatible PC path: matrix-free operator PCG can use explicit sparse/dense `matrix_free_pmat`, `matrix_free_coarse_pmat`/`coarse_hessian_pmat`, or `matrix_free_pc_action` as a PCSHELL-like inverse action. Metadata records source, kind, attr, requested preconditioner, and whether Pmat was available.
- When `petsc-gamg` is requested with an explicit Pmat in the current SciPy fast path, runtime uses the compatible explicit Pmat inverse contract instead of pretending GAMG can act on the shell Hessian directly. Future PETSc shell-H + Pmat/GAMG wiring can attach at this same contract boundary.

### Phase 3: Matrix-Free Preconditioning and Pmat Strategy

目标：避免 “matrix-free shell H + ILU/GAMG” 的错误假设，建立可验证预条件路线。

Tasks:

- T-FPX-008

Implementation notes:

- 第一层：diagonal/NOSER/prior precision 已由 T-FPX-007 完成，当前 contract 会在 matrix-free operator mode 记录 PC source/floor/min/max/Pmat availability。
- 第二层：sparse/dense Pmat、coarse inverse Hessian、PCSHELL-like inverse action 已由 T-FPX-008 完成小规模 smoke。
- PETSc path 如果使用 shell matrix，必须明确 Pmat 或 PCSHELL；当前 SciPy fast path 先用相同 contract 验证元数据和数值通路。
- 若 `petsc-gamg` 被请求但没有 explicit Pmat，必须 fallback 并写清楚原因；若有 explicit Pmat，当前 fast path 使用 Pmat inverse 并记录 requested preconditioner。

Validation commands:

```bash
nix develop -c uv run pytest --no-cov \
  tests/unit/test_gn_fast_linear_solver.py \
  tests/unit/test_perf_capabilities_selection.py -q
```

### Phase 4: Block-Ready Contact Impedance Path

目标：为 `sigma + z_contact` 联合估计留下 fieldsplit/Schur 结构，不把问题揉成不可维护 dense monolith。

Tasks:

- T-FPX-009

Implementation notes:

- 定义 block metadata：`sigma` 参数、`z_contact` 参数、measurement coupling、regularization。T-FPX-009 已在 `pyeidors.inverse.block_system` 提供 shape-safe metadata。
- 初始可只实现 shape-safe block diagonal approximation。T-FPX-009 已提供 `make_block_diagonal_inverse_action()`。
- fieldsplit/Schur 可以先作为 plan/design，不要求一次生产化。`JointInverseBlockMetadata.fieldsplit_plan()` 已输出 additive/multiplicative/Schur upgrade path；生产 PETSc fieldsplit 仍是后续任务。

Validation commands:

```bash
nix develop -c uv run pytest --no-cov tests/unit/test_inverse_block_system.py -q
```

### Phase 5: GPU/MPI Diagnostics and Strict Validation

目标：让 GPU/MPI 能力、fallback 和未完成边界都可见。

Tasks:

- T-FPX-010
- T-FPX-011
- T-FPX-012

Implementation notes:

- CUDA: keep `probe_petsc_cuda.py` as source of truth。
- MPI: 当前代码若仍限制 size=1，diagnostic 必须明确。
- 全量 unit 超时要拆分 test shards，不要只写“测试失败”。
- T-FPX-010 completion note: `probe_petsc_cuda.py --pretty` now emits PETSc
  CUDA availability plus an `mpi` diagnostics section; forward backend
  diagnostics and `forward_solver_benchmark` include CUDA Mat/Vec/Dense
  availability, PETSc CUDA errors, `gpu_transfer_risk`, `mpi_size`,
  `mpi_rank`, `mpi_parallel`, `mpi_size_supported`, and
  `mpi_fallback_reason`. MPI size > 1 still fails fast with the explicit
  phase-2 single-rank limitation.

Validation commands:

```bash
nix develop -c uv run python scripts/diagnostics/probe_petsc_cuda.py --pretty
```

```bash
nix develop -c uv run pytest --no-cov tests/unit/test_perf_capabilities_selection.py -q
```

## Validation Matrix

| Gate | Required For | Command Pattern | Notes |
| --- | --- | --- | --- |
| Gate 1 compile/import | every code change | `nix develop -c uv run python -m compileall -q <files>` | fastest gate |
| Gate 2 targeted unit | every task | `nix develop -c uv run pytest --no-cov <targeted tests> -q` | avoid coverage gate noise |
| Gate 3 integration/smoke | forward/inverse cross-boundary | `nix develop -c uv run pytest --no-cov tests/integration/<target> -q` | may be skipped with documented reason |
| Gate 4 benchmark | solver/cache/GPU | benchmark script with JSON/CSV output | must record environment and diagnostics |
| Gate 5 startup | GUI/runtime only | GUI launcher or CLI smoke | not mandatory for pure solver refactor |
| Gate 6 manual audit | every phase transition | review kit/plan/impl tracking | required before broad refactor continues |

## Detailed Solver Policy

具体 2D/3D 正逆问题的 solver、PC、matrix-free、block split、GPU/MPI 选择见
`context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md`。该文件是本 build site 的执行附录，优先级高于临时聊天上下文。

## File Ownership

| Path | Ownership |
| --- | --- |
| `src/pyeidors/forward/eit_forward_model.py` | forward PETSc solver policy and multi-RHS reuse |
| `src/pyeidors/cache/object_signature.py` | semantic cache invalidation |
| `src/pyeidors/inverse/jacobian/**` | sensitivity operator and Jacobian actions |
| `src/pyeidors/inverse/solvers/gauss_newton_runtime.py` | matrix-free inverse linear subproblems |
| `src/pyeidors/inverse/regularization/**` | R/prior actions and Pmat approximations |
| `tests/unit/test_forward_*` | forward policy/reuse regression |
| `tests/unit/test_jacobian_linearization.py` | matrix-free operator regression |
| `tests/unit/test_gn_*` | inverse fast path regression |
| `context/**fenicsx-petsc-eit-refactor*` | Cavekit continuity documents |

## Completion Definition

本 build site 只能在以下条件都满足时标记 COMPLETE：

- R1-R11 每条至少有一个自动验证或明确人工审计记录。
- R11 的 canonical solver/PC matrix 已记录在 implementation detail appendix；任何默认 solver policy 变更都必须回写该 appendix。
- 3D forward default 不再是 direct LU。
- 3D inverse fast path 有 operator-based route，不强制 dense `J`。
- 关键 fallback 都有 diagnostics。
- impl tracking 记录最新测试健康状态、死路、未验证项和下一步。
