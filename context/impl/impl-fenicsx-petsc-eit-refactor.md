---
status: in_progress
source: cavekit-refactor
last_updated: 2026-04-20
current_phase: Spec/Plan with first implementation pass
---

# Implementation Tracking: FEniCSx/PETSc EIT Solver Refactor

## Status: IN_PROGRESS

**Last Updated:** 2026-04-20  
**Current Phase:** Spec/Plan with first implementation pass  
**Blocking Issues:** None for current Cavekit wave. Full unsharded `tests/unit` remains unsuitable as the only broad gate; use sharded validation.

---

## Task Status

| Task ID | Task | Status | Notes |
| --- | --- | --- | --- |
| T-FPX-001 | Forward PETSc solver preset体系 | DONE | `LinearBackendConfig` 支持 `solver_preset`; 3D auto -> `fgmres+gamg+agg`; 2D auto -> direct。 |
| T-FPX-002 | Backend/cache signature 纳入 solver/PC/options | DONE | `backend_signature_from_forward_model` 已包含 preset、PC subtype、PETSc options。 |
| T-FPX-003 | Forward KSP/multi-RHS reuse hardening | DONE | fake PETSc call-count 覆盖一次 matrix/bundle setup 与 matSolve/vector-loop 路由；新增小 3D PETSc GAMG smoke 和 multi-RHS 诊断断言；follow-up 增加 KSP setup count 与 preconditioner reuse 诊断。 |
| T-FPX-004 | Forward solver benchmark artifact | DONE | `benchmark_3d_runtime.py --forward-only on` 生成 `forward_solver_benchmark` JSON；包含 setup/solve/iteration/RHS/device/fallback/Mat/Vec/finite-output diagnostics。 |
| T-FPX-005 | Matrix-free Jacobian operator layer | DONE | 新增 `JacobianLinearization` 和 `DirectJacobianCalculator.linearize()`。 |
| T-FPX-006 | GN fast linear solver 接入 operator 输入 | DONE | `_solve_linear_system_fast()` 支持 dense ndarray、SciPy `LinearOperator`、`JacobianLinearization`；`run_reconstruction(jacobian_method="linearized")` 可直连 `jacobian_calculator.linearize()`，operator path 通过 `Jv/J^T r` 组装 Hessian action，不形成 dense `J^T J`。 |
| T-FPX-007 | Matrix-free Hessian diagonal/NOSER/prior PC | DONE | Operator Hessian PCG 已支持 finite positive `diag`/`noser`/`prior` diagonal PC contract；`petsc-gamg` 无 Pmat 时明确 fallback。 |
| T-FPX-008 | Pmat/coarse Hessian/custom PC strategy | DONE | Matrix-free PCG 已有 explicit Pmat、coarse Pmat、custom PCSHELL-like inverse action smoke；`petsc-gamg`+Pmat 走兼容 Pmat contract。 |
| T-FPX-009 | Contact impedance block-ready interface | DONE | 新增 `pyeidors.inverse.block_system`，提供 `sigma + z_contact` block metadata、fieldsplit plan、shape-safe block diagonal inverse action 和 scaled finite `z_contact` update guard。 |
| T-FPX-010 | CUDA/MPI diagnostics hardening | DONE | `probe_petsc_cuda.py`、forward diagnostics 和 benchmark artifact 已统一记录 PETSc CUDA Mat/Vec/Dense capability、CUDA errors、transfer risk、MPI size/rank/support/fallback reason；MPI size > 1 仍显式 fail fast。 |
| T-FPX-011 | Sharded validation strategy | DONE | 新增 `scripts/ci/run_sharded_unit_tests.py`、`docs/VALIDATION_SHARDS.md` 和单元测试；原 `gui-hardware` 已拆成 default `gui` 与 opt-in `hardware`。 |
| T-FPX-012 | Cavekit continuity docs | DONE | 本文件、overview、ref、kit、plan 已同步；canonical solver matrix、测试健康、dead ends、next-session recovery notes 已收口。 |

### Task Dependencies

- T-FPX-006 blockedBy T-FPX-005。
- T-FPX-007 blockedBy T-FPX-006。
- T-FPX-008 blockedBy T-FPX-007。
- T-FPX-009 blockedBy T-FPX-006。
- T-FPX-004 dependsOn T-FPX-003。
- T-FPX-011 should run before broad refactor validation claims。

---

## Files Created

| File | Purpose | Spec Reference |
| --- | --- | --- |
| `context/refs/fenicsx-petsc-eit-refactor-research.md` | 保存总目标、官方依据、设计判断、验证经验 | R10 |
| `context/kits/cavekit-fenicsx-petsc-eit-refactor.md` | 重构总 kit，含 R1-R11 和验收标准 | R1-R11 |
| `context/plans/build-site-fenicsx-petsc-eit-refactor.md` | 从 kit 派生的实施计划和验证矩阵 | R10 |
| `context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md` | 2D/3D 正逆问题具体 solver、PC、block/GPU/MPI 细节和 canonical matrix | R1-R8,R11 |
| `context/impl/impl-fenicsx-petsc-eit-refactor.md` | 跨会话实施追踪和测试健康 | R10 |
| `src/pyeidors/inverse/jacobian/linearized.py` | Matrix-free `Jv/J^T r/Hv` operator | R4,R5 |
| `src/pyeidors/inverse/block_system.py` | Joint inverse block metadata, block diagonal inverse action, and scaled finite contact-impedance update guard for `sigma + z_contact` | R7 |
| `tests/unit/test_forward_solver_presets.py` | PETSc solver preset regression tests | R1 |
| `tests/unit/test_jacobian_linearization.py` | Matrix-free Jacobian action tests | R4,R5 |
| `tests/unit/test_inverse_block_system.py` | Block-ready `sigma + z_contact` shape and fieldsplit plan tests | R7 |
| `.cavekit/tasks.json` | Cavekit runtime task registry source for this build site | R10,R11 |
| `scripts/ci/run_sharded_unit_tests.py` | Recoverable unit-test shard runner with per-shard logs, JSON summary, default `gui`, and opt-in `hardware` shard | R10 |
| `tests/unit/test_ci_sharded_unit_validation.py` | Unit tests for shard coverage, no-cov commands, report path handling, shell quoting, and hardware opt-in selection | R10 |
| `docs/VALIDATION_SHARDS.md` | Human-readable commands for listing, dry-running, default GUI software shard, and opt-in hardware shard | R10 |

## Files Modified

| File | Change | Reason |
| --- | --- | --- |
| `src/pyeidors/forward/eit_forward_model.py` | 新增 solver presets、PC subtype/options database 接入、3D auto AMG 默认 | R1,R2,R3 |
| `src/pyeidors/cache/object_signature.py` | Backend signature 加入 preset/PC/options | R9 |
| `src/pyeidors/inverse/jacobian/direct_jacobian.py` | 新增 `linearize()` 返回 `JacobianLinearization` | R4 |
| `src/pyeidors/inverse/jacobian/__init__.py` | 导出 `JacobianLinearization` | R4 |
| `context/kits/cavekit-overview.md` | 挂载新 refactor kit | R10 |
| `context/plans/plan-overview.md` | 挂载新 build site 和 2D/3D implementation appendix | R10 |
| `context/impl/impl-overview.md` | 记录新 active build site | R10 |
| `context/refs/fenicsx-petsc-eit-refactor-research.md` | 追加 2026-04-20 官方二次核对结果、source URLs、canonical solver/PC strategy | R1-R11 |
| `context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md` | 追加 canonical solver and PC matrix，锁定 2D/3D forward/inverse/contact/GPU 路线 | R1-R11 |
| `context/kits/cavekit-fenicsx-petsc-eit-refactor.md` | 增加 R11，要求后续代理保存并同步 solver/PC 决策矩阵 | R11 |
| `context/plans/build-site-fenicsx-petsc-eit-refactor.md` | 将 T-FPX-012 扩展为追踪 canonical solver matrix 变更，并最终标记 T-FPX-001..012 全部完成 | R10,R11 |
| `context/impl/impl-overview.md` | 更新 active build overview，反映 T-FPX-001..012 已完成和最新 test health | R10,R11 |
| `context/impl/CLAUDE.md` | 指向 `impl-fenicsx-petsc-eit-refactor.md` 作为当前 refactor 的详细 tracking 入口 | R10 |
| `.cavekit/config.json` | Active build site switched to `build-site-fenicsx-petsc-eit-refactor.md`; next task set to T-FPX-011 | R10,R11 |
| `.cavekit/task-status.json` | Initialized via Cavekit runtime and marked T-FPX-011 complete | R10 |
| `context/plans/build-site-fenicsx-petsc-eit-refactor.md` | Marked T-FPX-011 DONE after validation | R10 |
| `.gitignore` | 忽略本地 `test_results/` validation artifacts，避免 sharded runner 产物污染工作树 | R10 |
| `src/pyeidors/forward/eit_forward_model.py` | PETSc forward 诊断增加 RHS 数、KSP solve/matSolve 计数、cache hit、有效 KSP/PC、Mat/Vec type、setup/solve seconds、KSP setup count、preconditioner reuse request/applied flag、KSP iterations、KSP convergence reason、fallback reason、MPI size/rank/support 和 GPU transfer risk；MPI size > 1 显式 fail fast | R2,R3,R8,R9 |
| `tests/unit/test_forward_mat_solve_policy.py` | fake PETSc 增加 matrix/bundle call-count 与诊断断言，覆盖 `mat_solve_mode=off|auto|on` 的 vector-loop/matSolve/fallback 路径和 preconditioner reuse diagnostics | R2,R9 |
| `tests/unit/test_forward_model_3d_cem.py` | 新增 8 电极 3D PETSc `3d_gamg` smoke，验证 finite 输出、GAMG preset、非 LU PC、multi-RHS solve 计数和 KSP setup/reuse diagnostics | R2,R3,R9 |
| `src/eit_app/ui/main_window.py` | 修复 3D interop import 后 radius/height 被交互默认值覆盖的问题 | R10 |
| `tests/unit/test_eit_app_interop_hub.py` | GUI/path picker 断言改用 i18n key，避免 English fallback 环境下误报 | R10 |
| `scripts/benchmarks/benchmark_3d_runtime.py` | 新增 `--forward-only` 与 `--forward-solver-preset`，并输出规范化 `forward_solver_benchmark` artifact，包含 setup count、preconditioner reuse、convergence/fallback、CUDA Mat/Vec/Dense capability/errors、GPU transfer risk 和 MPI support diagnostics | R3,R8,R9 |
| `tests/unit/test_script_entrypoint_acceleration_profiles.py` | 覆盖 forward-only CLI 解析、`forward_solver_benchmark` schema helper、setup/reuse fields 和 probe script MPI section | R3,R8,R9 |
| `tests/unit/test_forward_petsc_helper_branches.py` | 覆盖 forward PETSc backend CUDA/MPI diagnostic helper branch 和 MPI size > 1 fail-fast message | R8 |
| `src/pyeidors/inverse/solvers/gauss_newton_runtime.py` | GN fast linear solver 新增 dense/operator/JacobianLinearization 输入适配、operator-mode Hessian action、callable regularization support、`jacobian_representation`/`linear_iterations` diagnostics；`jacobian_method=linearized|operator|matrix-free` 可进入 runtime operator route | R4,R5 |
| `src/pyeidors/inverse/solvers/gauss_newton_runtime.py` | Matrix-free operator PCG 新增 `diag`/`noser`/`prior` diagonal PC contract：支持 explicit Hessian diag、explicit matrix-free diag、NOSER diag、prior/R diag、identity fallback；记录 PC source/floor/min/max/Pmat availability；`petsc-gamg` 无 Pmat 时降级并记录原因 | R5,R6 |
| `src/pyeidors/inverse/solvers/gauss_newton_engine.py` | GN preconditioner validation 接受 `noser`、`prior` | R6 |
| `src/pyeidors/perf/capabilities.py` | `select_preconditioner()` 将 `noser`、`prior` 作为显式 matrix-free diagonal PC mode 保留；新增 `probe_mpi_runtime()` 作为 MPI size/rank/support/fallback canonical diagnostic | R6,R8 |
| `scripts/diagnostics/probe_petsc_cuda.py` | PETSc CUDA probe 保持 CUDA source of truth，并在同一 JSON payload 中附加 `mpi` diagnostics section | R8 |
| `scripts/run_reconstruction_unified.py` | CLI `--preconditioner` 接受 `noser`、`prior` | R6 |
| `scripts/benchmarks/benchmark_3d_runtime.py` | 3D benchmark `--preconditioner` 接受 `noser`、`prior` | R6 |
| `scripts/common/gn_difference_runner.py` | Difference measurement-space solver/validation 接受 `noser`、`prior` diagonal PC mode | R6 |
| `tests/unit/test_gn_fast_linear_solver.py` | 覆盖 matrix-free NOSER/prior PC metadata、positive floor clamp、`petsc-gamg` no-Pmat fallback | R6 |
| `tests/unit/test_perf_capabilities_selection.py` | 覆盖 `noser`、`prior` resolver 行为和 MPI runtime single-rank limitation diagnostics | R6,R8 |
| `src/pyeidors/inverse/solvers/gauss_newton_runtime.py` | Matrix-free operator PCG 新增 explicit Pmat/coarse/custom inverse-PC contract：`matrix_free_pmat`、`matrix_free_coarse_pmat`/`coarse_hessian_pmat`、`matrix_free_pc_action`；记录 Pmat source/kind/attr/requested preconditioner；`petsc-gamg`+Pmat 使用兼容 Pmat inverse 而非 shell-H 直套 GAMG | R6 |
| `src/pyeidors/inverse/solvers/gauss_newton_engine.py` | GN preconditioner validation 接受 `pmat`、`coarse`、`custom` | R6 |
| `src/pyeidors/perf/capabilities.py` | `select_preconditioner()` 保留 `pmat`、`coarse`、`custom` 显式模式 | R6 |
| `scripts/run_reconstruction_unified.py` | CLI `--preconditioner` 接受 `pmat`、`coarse`、`custom` | R6 |
| `scripts/benchmarks/benchmark_3d_runtime.py` | 3D benchmark `--preconditioner` 接受 `pmat`、`coarse`、`custom` | R6 |
| `scripts/common/gn_difference_runner.py` | Difference measurement-space solver/validation 接受 `pmat`、`coarse`、`custom` diagonal/inverse-action-style mode | R6 |
| `tests/unit/test_gn_fast_linear_solver.py` | 覆盖 sparse Pmat、coarse dense Pmat、custom PCSHELL-like action、`petsc-gamg`+Pmat smoke | R6 |
| `tests/unit/test_perf_capabilities_selection.py` | 覆盖 `pmat`、`coarse`、`custom` resolver 行为 | R6 |
| `src/pyeidors/inverse/__init__.py` | 导出 block metadata/action/update helpers，作为未来 joint inverse 入口 | R7 |
| `context/plans/build-site-fenicsx-petsc-eit-refactor.md` | 标记 T-FPX-009 DONE，记录 block metadata 和 validation command | R7,R10 |
| `context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md` | 记录 T-FPX-009 contract：metadata、block diagonal action、future fieldsplit/Schur attachment boundary | R7,R11 |
| `src/pyeidors/inverse/jacobian/linearized.py` | `JacobianLinearization.normal_matvec()` 的 regularization action 支持 SciPy sparse matrix | R5 |
| `tests/unit/test_gn_fast_linear_solver.py` | 新增 LinearOperator 与 JacobianLinearization parity tests，覆盖 measurement weights、callable regularization、operator diagnostics | R4,R5 |
| `tests/unit/test_gn_runtime_run_reconstruction_branches.py` | 新增 runtime operator route test，确认 `jacobian_method="linearized"` 跳过 dense calculate/projection 并传递 measurement weights | R4,R5 |
| `tests/unit/test_gn_linearized_real_smoke.py` | 新增小型 2D/3D real-ish FEM smoke，专验 `jacobian_method="linearized"` 走 `JacobianLinearization` 且不 materialize dense J | R4,R5 |
| `tests/unit/test_jacobian_linearization.py` | 新增 sparse regularization action test | R5 |

---

## Issues & TODOs

- [x] **Resolved:** `gauss_newton_runtime._solve_linear_system_fast()` 已接收 `JacobianLinearization`/`LinearOperator`，并保留 dense reference path；operator mode 不 materialize dense `J`，dense-only Woodbury/ROM/cholmod debug path 会显式 fallback/记录原因。
- [x] **Resolved:** 增加 forward KSP reuse call-count 测试，验证单次 `_solve_with_petsc` 对所有 RHS 只创建一次 matrix/bundle，并按策略执行一次 matSolve 或 n 次 KSPSolve。
- [x] **Resolved:** Quick 3D forward benchmark 已定义为 `benchmark_3d_runtime.py --forward-only on`，输出 setup/solve/iteration/RHS/device/CUDA/MPI diagnostics；小 3D GAMG smoke 已在 T-FPX-003 完成。
- [x] **Resolved:** Runtime meta 已区分 matrix-free `diag/NOSER/prior/Pmat/coarse/custom` source/kind/floor/min/max/Pmat availability；`Pmat/custom shell` 已由 T-FPX-008 增加小规模 smoke。
- [x] **Resolved:** `sigma + z_contact` joint inverse 已有 block-ready metadata、fieldsplit plan、shape-safe block diagonal inverse action 和 scaled finite `z_contact` update guard；生产 PETSc fieldsplit/Schur 求解仍为后续升级，不阻塞当前 matrix-free sigma baseline。
- [x] **Resolved:** 全量 unit 已拆分为 recoverable shards；`fp-refactor-smoke` 真实运行通过，后续 broad gate 应运行 category shards。
- [x] **Resolved:** 修复 GUI/interop shard 的 4 个断言：3D interop import 保留 radius/height；Interop Hub/path picker 文案断言改用 i18n key。
- [x] **Resolved:** MPI size=1 当前限制已由 `probe_mpi_runtime()`、forward init fail-fast message、probe script `mpi` section 和 benchmark artifact fields 明确记录；解除路线仍要求分布式 Mat/Vec 与 `mpiexec -n 2` smoke。
- [ ] **TODO:** 后续任何 solver default 改动必须同步更新 R11、canonical matrix、build-site plan 和本 tracking 文件。

---

## Dead Ends & Failed Approaches

### DE-1: 裸 `uv run pytest` 作为 FEniCSx 验证入口

**What was attempted:** 在 WSL2 项目目录直接运行 `uv run pytest ...`。  
**Root cause of failure:** 当前裸 `.venv` 触发 NumPy 导入错误：“do not try to import numpy from its source directory”。项目文档也说明完整 FEniCSx 工作流必须进入 Nix dev shell。  
**Verdict:** Do not reattempt for solver validation. 使用 `nix develop -c uv run ...`。

### DE-2: 聚焦测试不加 `--no-cov`

**What was attempted:** 运行少量 targeted pytest 时使用默认 `pyproject.toml` addopts。  
**Root cause of failure:** 全局 coverage fail-under=87 会因为只跑少量测试而失败，即使功能测试全通过。  
**Verdict:** 聚焦验证统一使用 `--no-cov`；coverage 单独作为专门 gate 运行。

### DE-3: 一次性全量 `tests/unit` 不分片

**What was attempted:** `nix develop -c uv run pytest --no-cov tests/unit -q`。  
**Root cause of failure:** 10 分钟超时未返回失败明细，无法形成可靠验收证据。  
**Verdict:** 不要把未分片全量 unit 作为唯一 gate。先按 forward/inverse/cache/gui/env 分片，记录每片结果；必要时再跑夜间完整 gate。

### DE-4: 相对 `--report-dir` 未归一化

**What was attempted:** 运行 `scripts/ci/run_sharded_unit_tests.py --run --shard fp-refactor-smoke --report-dir test_results/sharded_unit/fp_refactor_smoke_check`。  
**Root cause of failure:** 初版脚本能写入相对路径日志，但在生成 `relative_to(REPO_ROOT)` 时没有先把相对路径归一化到仓库根目录，导致收尾阶段抛出 `ValueError`。  
**Verdict:** Do not build report artifact paths directly from user-provided relative paths. 使用 `_normalize_report_dir()` 和 `_relative_to_repo()`；对应测试已加入 `tests/unit/test_ci_sharded_unit_validation.py`。

---

## Test Health

| Test Suite | Command | Result | Notes |
| --- | --- | --- | --- |
| Compile changed solver files | `nix develop -c uv run python -m compileall -q src/pyeidors/forward/eit_forward_model.py src/pyeidors/inverse/jacobian/direct_jacobian.py src/pyeidors/inverse/jacobian/linearized.py src/pyeidors/cache/object_signature.py` | PASS | 首轮实现后通过。 |
| Targeted forward/PETSc/Jacobian/GN | `nix develop -c uv run pytest --no-cov tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_forward_solver_branch_suite.py tests/unit/test_forward_mat_solve_policy.py tests/unit/test_perf_capabilities_selection.py tests/unit/test_gn_fast_linear_solver.py tests/unit/test_forward_vectorized_runtime.py tests/unit/test_forward_solve_view_semantics.py tests/unit/test_adjoint_jacobian_helper_branches.py tests/unit/test_jacobian_linearization.py tests/unit/test_forward_solver_presets.py -q` | PASS | 53 passed in 4.22s。 |
| Full unit unsharded | `nix develop -c uv run pytest --no-cov tests/unit -q` | TIMEOUT | 10 分钟超时；进程已清理。需要 T-FPX-011。 |
| Shard runner compile | `nix develop -c uv run python -m compileall -q scripts/ci/run_sharded_unit_tests.py tests/unit/test_ci_sharded_unit_validation.py` | PASS | T-FPX-011 Gate 1。 |
| Shard runner unit tests | `nix develop -c uv run pytest --no-cov tests/unit/test_ci_sharded_unit_validation.py -q` | PASS | 5 passed in 0.62s after `/ck:check` follow-up。 |
| Shard list/dry-run | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --list`; `... --dry-run --shard fp-refactor-smoke` | PASS | Lists 11 category shards + 1 virtual smoke shard; emitted command includes `nix develop -c uv run pytest --no-cov ... -q`。 |
| Sharded fp-refactor smoke | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard fp-refactor-smoke --timeout 240 --report-dir test_results/sharded_unit/fp_refactor_smoke_check` | PASS | 35 passed in 2.90s; summary JSON records 8.413s wall time and per-shard logs。 |
| T-FPX-011 `/ck:check` follow-up | `nix develop -c uv run pytest --no-cov tests/unit/test_ci_sharded_unit_validation.py -q`; `... --dry-run --shard fp-refactor-smoke --pytest-arg=-k --pytest-arg "solver and not slow"` | PASS | 8 passed in 0.97s；dry-run shell-quotes pytest expressions, `test_results/` is ignored, and GUI/hardware domains are separate。 |
| Full category baseline before GUI/hardware split | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --all --timeout 300` | SOFTWARE PASS / GUI-HARDWARE FAIL | 10 non-GUI/hardware shards passed；combined `gui-hardware` failed with 4 GUI/interoperability assertions while no hardware environment is attached. Runner now splits this into default `gui` and opt-in `hardware` shards. |
| Core non-GUI/hardware category baseline | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --all --timeout 300` | PASS | Historical baseline before split opt-in refinement: 10 default shards passed, 135 files, 480.596s aggregate shard time. Summary: `test_results/sharded_unit/20260420T061445Z/summary.json`。 |
| Split GUI shard baseline | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard gui --timeout 300 --report-dir test_results/sharded_unit/gui_split_review` | FAIL | 99 passed / 4 failed in 282.70s. Failures are GUI/interop assertions only, not missing hardware. Summary: `test_results/sharded_unit/gui_split_review/summary.json`。 |
| GUI targeted fix | `nix develop -c uv run pytest --no-cov tests/unit/test_eit_app_gui_smoke.py::test_interop_imported_3d_geometry_is_not_replaced_by_interactive_defaults tests/unit/test_eit_app_interop_hub.py::test_import_target_path_picker_uses_python_callback tests/unit/test_eit_app_interop_hub.py::test_import_target_path_picker_builds_sidebar_places tests/unit/test_eit_app_interop_hub.py::test_export_target_path_picker_uses_python_callback -q` | PASS | 4 passed in 5.12s；GUI/interop 软件断言已和硬件 shard 解耦。 |
| GUI shard after fix | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard gui --timeout 300 --report-dir test_results/sharded_unit/gui_fixed_check` | PASS | 103 GUI tests passed；summary: `test_results/sharded_unit/gui_fixed_check/summary.json`。 |
| Default sharded unit after GUI fix | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --all --timeout 300 --report-dir test_results/sharded_unit/default_all_gui_fixed` | PASS | 11 default shards passed, hardware shard remains opt-in；summary: `test_results/sharded_unit/default_all_gui_fixed/summary.json`。 |
| T-FPX-003 compile gate | `nix develop -c uv run python -m compileall -q src/pyeidors/forward/eit_forward_model.py tests/unit/test_forward_mat_solve_policy.py tests/unit/test_forward_model_3d_cem.py src/eit_app/ui/main_window.py tests/unit/test_eit_app_interop_hub.py` | PASS | T-FPX-003 + GUI fix changed files compile。 |
| T-FPX-003 targeted forward/PETSc | `nix develop -c uv run pytest --no-cov tests/unit/test_forward_solver_presets.py tests/unit/test_forward_solver_branch_suite.py tests/unit/test_forward_mat_solve_policy.py tests/unit/test_forward_vectorized_runtime.py tests/unit/test_forward_petsc_multirhs.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_forward_model_3d_cem.py -q` | PASS | 36 passed in 6.77s after `/ck:check` follow-up；包含 fake PETSc call-count、`auto|on|off` matSolve policy、CPU matSolve failure fallback、3D GAMG smoke。 |
| T-FPX-003 forward shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard forward --timeout 300 --report-dir test_results/sharded_unit/tfpx003_forward_check` | PASS | `forward` shard passed, 15 files, 15.976s；summary: `test_results/sharded_unit/tfpx003_forward_check/summary.json`。 |
| T-FPX-003 `/ck:check` forward shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard forward --timeout 300 --report-dir test_results/sharded_unit/tfpx003_check_forward` | PASS | `forward` shard passed, 15 files, 16.48s；summary: `test_results/sharded_unit/tfpx003_check_forward/summary.json`。 |
| T-FPX-003 reuse diagnostic follow-up compile gate | `nix develop -c uv run python -m compileall -q src/pyeidors/forward/eit_forward_model.py scripts/benchmarks/benchmark_3d_runtime.py tests/unit/test_forward_mat_solve_policy.py tests/unit/test_forward_solver_branch_suite.py tests/unit/test_forward_petsc_multirhs.py tests/unit/test_forward_model_3d_cem.py tests/unit/test_script_entrypoint_acceleration_profiles.py` | PASS | Forward reuse/setup diagnostic changed files compile。 |
| T-FPX-003 reuse diagnostic targeted forward/PETSc | `nix develop -c uv run pytest --no-cov tests/unit/test_forward_solver_presets.py tests/unit/test_forward_solver_branch_suite.py tests/unit/test_forward_mat_solve_policy.py tests/unit/test_forward_vectorized_runtime.py tests/unit/test_forward_petsc_multirhs.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_forward_model_3d_cem.py tests/unit/test_script_entrypoint_acceleration_profiles.py -q` | PASS | 50 passed in 8.06s；covers KSP setup count、preconditioner reuse request/applied flag、benchmark artifact fields、3D GAMG smoke。 |
| T-FPX-003 reuse diagnostic forward shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard forward --timeout 300 --report-dir test_results/sharded_unit/tfpx003_reuse_diag_followup_final` | PASS | `forward` shard passed, 15 files, 16.844s；summary: `test_results/sharded_unit/tfpx003_reuse_diag_followup_final/summary.json`。 |
| T-FPX-004 compile gate | `nix develop -c uv run python -m compileall -q src/pyeidors/forward/eit_forward_model.py scripts/benchmarks/benchmark_3d_runtime.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_script_entrypoint_acceleration_profiles.py` | PASS | Forward diagnostics and benchmark artifact entrypoint compile。 |
| T-FPX-004 entrypoint/unit gate | `nix develop -c uv run pytest --no-cov tests/unit/test_script_entrypoint_acceleration_profiles.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_forward_mat_solve_policy.py -q` | PASS | 28 passed in 2.21s；covers schema helper, CLI flags, CPU stable Mat/Vec diagnostics, and matSolve policy。 |
| T-FPX-004 benchmark help gate | `nix develop -c uv run python scripts/benchmarks/benchmark_3d_runtime.py --help` | PASS | Help includes `--forward-only` and `--forward-solver-preset`。 |
| T-FPX-004 quick forward benchmark artifact | `nix develop -c uv run python scripts/benchmarks/benchmark_3d_runtime.py --forward-only on --run-diff off --run-absolute off --n-elec 8 --refinement 1 --radius 0.16 --height 0.14 --forward-solver-preset 3d_gamg --forward-mat-solve auto --petsc-device auto --perf-report test_results/benchmarks/tfpx004_forward_solver.json` | PASS | Artifact field check passed；`forward_solver_benchmark` has mesh/RHS/solver/PC/Mat/Vec/timing/iterations/device/fallback/finite-output fields。 |
| T-FPX-004 targeted forward/PETSc/benchmark | `nix develop -c uv run pytest --no-cov tests/unit/test_forward_solver_presets.py tests/unit/test_forward_solver_branch_suite.py tests/unit/test_forward_mat_solve_policy.py tests/unit/test_forward_vectorized_runtime.py tests/unit/test_forward_petsc_multirhs.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_forward_model_3d_cem.py tests/unit/test_script_entrypoint_acceleration_profiles.py -q` | PASS | 46 passed in 6.64s。 |
| T-FPX-004 forward shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard forward --timeout 300 --report-dir test_results/sharded_unit/tfpx004_forward_check` | PASS | `forward` shard passed, 15 files, 14.938s；summary: `test_results/sharded_unit/tfpx004_forward_check/summary.json`。 |
| T-FPX-004 `/ck:check` targeted forward/PETSc/benchmark | `nix develop -c uv run pytest --no-cov tests/unit/test_forward_solver_presets.py tests/unit/test_forward_solver_branch_suite.py tests/unit/test_forward_mat_solve_policy.py tests/unit/test_forward_vectorized_runtime.py tests/unit/test_forward_petsc_multirhs.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_forward_model_3d_cem.py tests/unit/test_script_entrypoint_acceleration_profiles.py -q` | PASS | 47 passed in 7.84s after convergence/fallback diagnostic follow-up。 |
| T-FPX-004 `/ck:check` forward shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard forward --timeout 300 --report-dir test_results/sharded_unit/tfpx004_check_forward_shard` | PASS | `forward` shard passed, 15 files, 16.896s；summary: `test_results/sharded_unit/tfpx004_check_forward_shard/summary.json`。 |
| T-FPX-004 `/ck:check` quick forward benchmark artifact | `nix develop -c uv run python scripts/benchmarks/benchmark_3d_runtime.py --forward-only on --run-diff off --run-absolute off --n-elec 8 --refinement 1 --radius 0.16 --height 0.14 --forward-solver-preset 3d_gamg --forward-mat-solve auto --petsc-device auto --perf-report test_results/benchmarks/tfpx004_check_forward_solver.json` | PASS | Artifact field check passed；artifact now includes `converged_reason=-3`, `converged=false`, and explicit fallback reason when PETSc GAMG fails and SciPy fallback produces finite output。 |
| T-FPX-006 compile gate | `nix develop -c uv run python -m compileall -q src/pyeidors/inverse/solvers/gauss_newton_runtime.py src/pyeidors/inverse/jacobian/linearized.py tests/unit/test_gn_fast_linear_solver.py tests/unit/test_jacobian_linearization.py` | PASS | Fast solver/operator changed files compile。 |
| T-FPX-006 targeted GN operator gate | `nix develop -c uv run pytest --no-cov tests/unit/test_jacobian_linearization.py tests/unit/test_gn_fast_linear_solver.py tests/unit/test_gn_runtime_helper_branches.py -q` | PASS | 26 passed in 2.34s；dense, LinearOperator, JacobianLinearization, measurement weights, callable/sparse regularization, diagnostics covered。 |
| T-FPX-006 inverse GN shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard inverse-gn --timeout 300 --report-dir test_results/sharded_unit/tfpx006_inverse_gn_check` | PASS | `inverse-gn` shard passed, 31 files, 134.55s；summary: `test_results/sharded_unit/tfpx006_inverse_gn_check/summary.json`。 |
| T-FPX-006 `/ck:check` targeted GN operator gate | `nix develop -c uv run pytest --no-cov tests/unit/test_jacobian_linearization.py tests/unit/test_gn_fast_linear_solver.py tests/unit/test_gn_runtime_helper_branches.py tests/unit/test_gn_runtime_run_reconstruction_branches.py -q` | PASS | 29 passed in 3.36s；新增 runtime `jacobian_method="linearized"` operator route test。 |
| T-FPX-006 `/ck:check` inverse GN shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard inverse-gn --timeout 300 --report-dir test_results/sharded_unit/tfpx006_check_inverse_gn` | PASS | `inverse-gn` shard passed, 31 files, 136.451s；summary: `test_results/sharded_unit/tfpx006_check_inverse_gn/summary.json`。 |
| T-FPX-006 2D/3D linearized real-ish smoke | `nix develop -c uv run pytest --no-cov tests/unit/test_gn_linearized_real_smoke.py -q` | PASS | 2 passed in 0.85s；2D tagged unit-square FEM and 3D 4-electrode cylinder FEM both report `jacobian_representation=jacobian_linearization`, `dense_jacobian_materialized=false`, startup dense cache skipped。 |
| T-FPX-006 linearized smoke targeted gate | `nix develop -c uv run pytest --no-cov tests/unit/test_gn_linearized_real_smoke.py tests/unit/test_jacobian_linearization.py tests/unit/test_gn_fast_linear_solver.py tests/unit/test_gn_runtime_run_reconstruction_branches.py -q` | PASS | 13 passed in 1.78s。 |
| T-FPX-006 linearized smoke inverse GN shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard inverse-gn --timeout 300 --report-dir test_results/sharded_unit/tfpx006_linearized_smoke_inverse_gn` | PASS | `inverse-gn` shard passed, 32 files, 138.089s；summary: `test_results/sharded_unit/tfpx006_linearized_smoke_inverse_gn/summary.json`。 |
| T-FPX-007 targeted PC contract gate | `nix develop -c uv run pytest tests/unit/test_gn_fast_linear_solver.py tests/unit/test_perf_capabilities_selection.py -q --no-cov` | PASS | 22 passed in 2.08s；direct run without `--no-cov` also had 22 passed but failed only global coverage threshold, consistent with DE-2。 |
| T-FPX-007 GN runtime helper gate | `nix develop -c uv run pytest tests/unit/test_gn_runtime_helper_branches.py tests/unit/test_gn_runtime_run_reconstruction_branches.py -q --no-cov` | PASS | 21 passed in 1.96s；old fallback/helper branches still pass with new PC contract。 |
| T-FPX-007 inverse GN shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard inverse-gn --timeout 300` | PASS | `inverse-gn` shard passed, 32 files, 133.373s；summary: `test_results/sharded_unit/20260420T083452Z/summary.json`。 |
| T-FPX-008 compile gate | `nix develop -c uv run python -m py_compile src/pyeidors/inverse/solvers/gauss_newton_runtime.py src/pyeidors/inverse/solvers/gauss_newton_engine.py src/pyeidors/perf/capabilities.py scripts/common/gn_difference_runner.py scripts/run_reconstruction_unified.py scripts/benchmarks/benchmark_3d_runtime.py` | PASS | Pmat/coarse/custom PC changed files compile。 |
| T-FPX-008 targeted PC/Pmat gate | `nix develop -c uv run pytest tests/unit/test_gn_fast_linear_solver.py tests/unit/test_perf_capabilities_selection.py -q --no-cov` | PASS | 25 passed in 2.34s；covers sparse Pmat, coarse dense Pmat, custom PCSHELL-like inverse action, and `petsc-gamg`+Pmat metadata。 |
| T-FPX-008 GN runtime helper gate | `nix develop -c uv run pytest tests/unit/test_gn_runtime_helper_branches.py tests/unit/test_gn_runtime_run_reconstruction_branches.py -q --no-cov` | PASS | 21 passed in 2.02s；old helper/runtime branches unchanged。 |
| T-FPX-008 inverse GN shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard inverse-gn --timeout 300 --report-dir test_results/sharded_unit/tfpx008_inverse_gn_check` | PASS | `inverse-gn` shard passed, 32 files, 137.611s；summary: `test_results/sharded_unit/tfpx008_inverse_gn_check/summary.json`。 |
| T-FPX-008 CLI/benchmark gate | `nix develop -c uv run pytest tests/unit/test_recon_cli_validation.py tests/unit/test_script_entrypoint_acceleration_profiles.py -q --no-cov` | PASS | 21 passed in 51.02s；expanded preconditioner choices do not break CLI/benchmark entrypoint tests。 |
| T-FPX-008 final runtime syntax/unit rerun | `nix develop -c uv run python -m py_compile src/pyeidors/inverse/solvers/gauss_newton_runtime.py`; `nix develop -c uv run pytest tests/unit/test_gn_fast_linear_solver.py -q --no-cov` | PASS | Runtime compile passed；11 fast solver tests passed in 1.15s after small dense-shape guard cleanup。 |
| T-FPX-009 compile gate | `nix develop -c uv run python -m py_compile src/pyeidors/inverse/block_system.py src/pyeidors/inverse/__init__.py tests/unit/test_inverse_block_system.py` | PASS | New block metadata/action files compile。 |
| T-FPX-009 block interface unit gate | `nix develop -c uv run pytest tests/unit/test_inverse_block_system.py -q --no-cov` | PASS | 5 passed in 0.57s；covers block sizes/slices, measurement/Hessian coupling shapes, fieldsplit/Schur plan, block diagonal inverse action, scaled finite `z_contact` update guard, and invalid shapes/modes。 |
| T-FPX-009 core-misc shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard core-misc --timeout 300 --report-dir test_results/sharded_unit/tfpx009_core_misc_check` | PASS | `core-misc` shard passed, 12 files, 20.115s；summary: `test_results/sharded_unit/tfpx009_core_misc_check/summary.json`。 |
| T-FPX-009 inverse GN shard | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard inverse-gn --timeout 300 --report-dir test_results/sharded_unit/tfpx009_inverse_gn_check` | PASS | `inverse-gn` shard passed, 32 files, 124.028s；summary: `test_results/sharded_unit/tfpx009_inverse_gn_check/summary.json`。 |
| T-FPX-010 compile gate | `nix develop -c uv run python -m py_compile src/pyeidors/perf/capabilities.py src/pyeidors/forward/eit_forward_model.py scripts/diagnostics/probe_petsc_cuda.py scripts/benchmarks/benchmark_3d_runtime.py tests/unit/test_perf_capabilities_selection.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_script_entrypoint_acceleration_profiles.py` | PASS | CUDA/MPI diagnostic changed files compile。 |
| T-FPX-010 targeted capability/diagnostic gate | `nix develop -c uv run pytest --no-cov tests/unit/test_perf_capabilities_selection.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_script_entrypoint_acceleration_profiles.py -q` | PASS | 37 passed in 3.38s；covers PETSc CUDA probe, MPI single-rank diagnostic, forward MPI fail-fast helper, benchmark artifact fields, and probe script `mpi` section。 |
| T-FPX-010 extra branch gate | `nix develop -c uv run pytest --no-cov tests/unit/test_perf_capabilities_helper_branches.py tests/unit/test_forward_mat_solve_policy.py -q` | PASS | 12 passed in 1.23s；capability helper branches and matSolve policy still green。 |
| T-FPX-010 probe command | `nix develop -c uv run python scripts/diagnostics/probe_petsc_cuda.py --pretty` | PASS | Current runtime reports PETSc CUDA unavailable despite CUDA type names, with explicit Mat/Vec/Dense errors; `mpi` section reports size=1, rank=0, supported。 |
| T-FPX-010 shards | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard forward --timeout 300 --report-dir test_results/sharded_unit/tfpx010_forward_check`; `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard core-misc --timeout 300 --report-dir test_results/sharded_unit/tfpx010_core_misc_check` | PASS | `forward` shard passed, 15 files, 17.942s；`core-misc` shard passed, 12 files, 23.566s。 |
| T-FPX-010 quick forward artifact | `nix develop -c uv run python scripts/benchmarks/benchmark_3d_runtime.py --forward-only on --run-diff off --run-absolute off --n-elec 8 --refinement 1 --radius 0.16 --height 0.14 --forward-solver-preset 3d_gamg --forward-mat-solve auto --petsc-device auto --perf-report test_results/benchmarks/tfpx010_forward_solver.json` | PASS | Artifact field check passed；`forward_solver_benchmark` contains CUDA availability/errors, Mat/Vec/Dense type, transfer risk, MPI size/rank/support/fallback fields and finite output。 |
| T-FPX-012 tracking JSON gate | `nix develop -c uv run python -m json.tool .cavekit/tasks.json`; `nix develop -c uv run python -m json.tool .cavekit/task-status.json` | PASS | Cavekit task registries parse as JSON。 |
| T-FPX-012 status consistency gate | Python status audit over `.cavekit/task-status.json` and `.cavekit/tasks.json` | PASS | Both registries report all 12 `T-FPX-*` tasks complete。 |
| T-FPX-012 diff hygiene | `git diff --check` | PASS | No whitespace errors after final tracking cleanup。 |

### Failing Tests

- 当前没有来自 targeted suite 或 `inverse-gn` shard 的 failing tests。
- Full unit 未产生 failure list，因为超时中断。

---

## Session Log

### Session 2026-04-20

- 阅读 Cavekit `methodology`、`cavekit-writing`、`validation-first`、`impl-tracking` 技能。
- 将用户的 FEniCSx/PETSc/EIT 重构目标转写为 reference、kit、plan、implementation tracking。
- 增补 `fenicsx-petsc-eit-2d-3d-implementation-details.md`，记录 2D/3D forward/inverse 的具体 solver、preconditioner、block split、GPU/MPI 和验证策略。
- 再次核对 DOLFINx/PETSc 官方文档，确认 `LinearProblem/KSP`、`NonlinearProblem/SNES`、`P=`、MUMPS reference、GAMG/Hypre/AMGx、fieldsplit、KSPMatSolve、KSPSetReusePreconditioner、MATSHELL/PCSHELL、CUDA Mat/Vec/PCAMGX 等依据。
- 将用户给出的 2D/3D 正逆问题最终 solver/PC 主线写入 ref 和 implementation detail appendix，并在 kit 中新增 R11 防止后续上下文压缩后遗忘。
- 用户指出官方 Cavekit 流程应使用 `/ck:sketch --from-code`、`/ck:map`、`/ck:make`、`/ck:check`。已查阅 `ck-sketch`、`ck-map`、`ck-make`、`ck-check` prompt，并将本 build site 接入 `.cavekit/tasks.json` / `.cavekit/task-status.json`。
- T-FPX-011 implemented under `/ck:make` inline-equivalent flow：新增 recoverable unit shard runner、docs 和 tests；Cavekit runtime 已标记 T-FPX-011 complete，当前状态 `4/12 complete`。
- T-FPX-011 `/ck:check` follow-up：发现并修复两个 P3 质量缺口：`test_results/` 未忽略、dry-run 对带空格 pytest 参数没有 shell-safe quoting；补充 docs 和测试后验证通过。
- Full category baseline showed all non-GUI/hardware shards passing and combined `gui-hardware` failing. User clarified missing hardware should not mask GUI software failures; runner split the combined shard into default `gui` and opt-in `hardware`。
- Re-ran core non-GUI/hardware `--all` before the split refinement: 10 shards / 135 files passed; baseline summary saved locally under ignored `test_results/sharded_unit/20260420T061445Z/summary.json`。
- Split `gui` shard reviewed separately: 4 failing assertions remain in `test_eit_app_gui_smoke.py` and `test_eit_app_interop_hub.py`; these are tracked as GUI/interop regressions outside the FEniCSx/PETSc solver refactor path。
- GUI/interop regressions fixed before continuing T-FPX-003: 3D import now carries radius/height into simulation/dataset config; path picker tests now assert i18n labels. Targeted GUI tests, full `gui` shard, and default `--all` sharded baseline pass with hardware skipped by default。
- T-FPX-003 implemented under `/ck:make`: PETSc forward diagnostics now expose `forward_rhs_count`、`forward_ksp_solve_count`、`forward_ksp_mat_solve_count`、cache hit、effective KSP/PC and Mat/Vec types; fake PETSc tests assert one matrix/bundle setup per multi-RHS solve; 3D PETSc `3d_gamg` smoke verifies finite output, GAMG PC, non-LU path, and multi-RHS diagnostics。
- T-FPX-003 validation passed: compile gate, 34 targeted forward/PETSc tests, and `forward` shard all green。
- T-FPX-003 `/ck:check` follow-up：发现并修复一个 P3 验收覆盖缝隙，补上 explicit `mat_solve_mode="on"` 强制 matSolve 测试和 CPU matSolve failure fallback 诊断测试；targeted forward/PETSc suite 现为 36 passed，`forward` shard 继续通过。Finding 记录在 `context/impl/impl-review-findings.md`。
- T-FPX-003 reuse diagnostic follow-up：按 Cavekit 主线补强 R9 可观察性，forward diagnostics 和 `forward_solver_benchmark` artifact 现记录 `forward_ksp_setup_count`、`forward_ksp_setup_attempts`、`forward_reuse_preconditioner_requested`、`forward_reuse_preconditioner_applied`、`forward_factor_cache_hit`；targeted forward/PETSc suite 50 passed，`forward` shard 继续通过。
- T-FPX-004 implemented under `/ck:make`: PETSc forward solve now records setup/solve seconds and KSP iteration counts; CPU PETSc diagnostics now include stable Mat/Vec types even when `matSolve` avoids explicit Vec creation; `benchmark_3d_runtime.py --forward-only on` writes a normalized `forward_solver_benchmark` artifact. Quick real artifact saved to ignored local path `test_results/benchmarks/tfpx004_forward_solver.json`。
- T-FPX-004 validation passed: compile gate, entrypoint/unit gate, `benchmark_3d_runtime.py --help`, quick real forward artifact field check, 46 targeted tests, and `forward` shard all green。
- T-FPX-004 `/ck:check` follow-up：发现并修复一个 P2 诊断缺口：真实 quick artifact 只显示 finite output 和 iterations，未暴露 PETSc negative convergence reason。现在 forward diagnostics 和 benchmark artifact 都记录 `converged_reason`/`converged`，negative `matSolve` reason 会进入 fallback/error 路径；targeted suite 47 passed，`forward` shard 继续通过。Finding 记录在 `context/impl/impl-review-findings.md`。
- T-FPX-006 implemented under `/ck:make`: `_solve_linear_system_fast()` now accepts dense ndarray, SciPy `LinearOperator`, and `JacobianLinearization`; Hessian action uses `Jv/J^T r + alpha Rv` in operator mode; callable/sparse regularization actions and solver diagnostics (`jacobian_representation`, `dense_jacobian_materialized`, `linear_iterations`) are covered by unit tests.
- T-FPX-006 validation passed: compile gate, targeted GN operator gate (26 passed), and `inverse-gn` shard (31 files) all green。
- T-FPX-006 `/ck:check` follow-up：发现并修复一个 P2 runtime wiring 缺口。此前 operator 支持只接到 `_solve_linear_system_fast()`，`run_reconstruction()` 仍会走 dense `calculate()`/projection。现在显式 `jacobian_method=linearized|operator|matrix-free` 会调用 `jacobian_calculator.linearize()`，跳过 dense startup cache/projection，并把 measurement weights 传给 fast solver。Finding 记录在 `context/impl/impl-review-findings.md`；targeted 29 passed，`inverse-gn` shard 继续通过。
- T-FPX-006 extra smoke：新增并运行小型 2D/3D real-ish FEM smoke，专验 `jacobian_method="linearized"`。2D tagged unit-square 与 3D 4-electrode cylinder 均通过，backend diagnostics 确认 `jacobian_linearization`、无 dense J materialization、startup dense cache skipped；`inverse-gn` shard 32 files 继续通过。
- T-FPX-007 implemented under `/ck:make`: matrix-free Hessian PCG now has explicit diagonal PC contract for `diag`/`noser`/`prior`; positive finite lower bounds are enforced; metadata records source, floor, min/max and whether explicit Pmat is available; `petsc-gamg` without Pmat falls back to diagonal PC with reason `petsc_gamg_not_supported_in_matrix_free`。
- T-FPX-007 validation passed: targeted PC contract gate 22 passed, GN runtime helper gate 21 passed, and `inverse-gn` shard 32 files passed。
- T-FPX-008 implemented under `/ck:make`: matrix-free PCG can now use explicit sparse/dense Pmat, coarse Pmat, or custom PCSHELL-like inverse action as compatible preconditioners. `petsc-gamg` with explicit Pmat uses the Pmat contract and records requested backend; without Pmat it still falls back with the existing no-Pmat reason。
- T-FPX-008 validation passed: compile gate, targeted PC/Pmat gate 25 passed, GN runtime helper gate 21 passed, CLI/benchmark gate 21 passed, and `inverse-gn` shard 32 files passed。
- T-FPX-009 implemented under `/ck:make`: added `pyeidors.inverse.block_system` with `ParameterBlock`, `BlockCoupling`, `JointInverseBlockMetadata`, `build_sigma_contact_block_metadata()`, `make_block_diagonal_inverse_action()`, and `scale_contact_impedance_update()`。The interface captures `sigma`/`z_contact` sizes, offsets, measurement and Hessian coupling shapes, regularization labels, fieldsplit additive/multiplicative/Schur upgrade plan, and finite scaled contact update guard。
- T-FPX-009 validation passed: compile gate, 5 block interface tests, `core-misc` shard 12 files, and `inverse-gn` shard 32 files all passed。
- T-FPX-010 implemented under `/ck:make`: added `probe_mpi_runtime()` and wired MPI diagnostics into `probe_petsc_cuda.py`, forward backend diagnostics, MPI size > 1 fail-fast message, and `forward_solver_benchmark` artifact. CUDA diagnostics now expose Mat/Vec/Dense availability and PETSc errors; benchmark/runtime output records transfer risk and MPI support fields。
- T-FPX-010 validation passed: compile gate, 37 targeted capability/forward/script tests, 12 extra branch tests, real `probe_petsc_cuda.py --pretty`, `forward` shard, `core-misc` shard, and quick 3D forward artifact field check all passed。
- T-FPX-012 implemented under `/ck:make`: reconciled build-site status, detailed implementation tracking, implementation overview, and impl navigation notes so future agents can resume from durable context rather than chat memory。
- 首轮代码重构已完成：forward solver preset、PC options、cache signature、JacobianLinearization。
- Targeted validation: 53 tests passed。
- Known broad-test note: use sharded validation instead of unsharded full unit as the only gate。

---

## Next Session Start Instructions

新上下文或新代理开始时必须：

1. 读 `context/refs/fenicsx-petsc-eit-refactor-research.md`。
2. 读 `context/kits/cavekit-fenicsx-petsc-eit-refactor.md`。
3. 读 `context/plans/build-site-fenicsx-petsc-eit-refactor.md`。
4. 读 `context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md`。
5. 读本文件的 Task Status、Dead Ends 和 Test Health。
6. Cavekit runtime 当前已完成 T-FPX-001..T-FPX-012；下一步应进入 `/ck:check` 或由用户指定新的 refactor task，不要继续凭旧 TODO 发散。
