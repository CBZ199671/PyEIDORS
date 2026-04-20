---
status: draft
source: user-research-plus-official-doc-check
domain: fenicsx-petsc-eit-refactor
complexity: thorough
---

# Cavekit: FEniCSx/PETSc EIT Solver Refactor

## Scope

本 kit 定义 PyEIDORS 底层 2D/3D EIT 正逆问题重构必须满足的行为、边界和验收标准。它覆盖 forward CEM 求解器、PETSc KSP/PC 预条件策略、多 RHS 复用、matrix-free sensitivity/inverse、GPU/MPI 诊断、缓存安全和跨会话验证纪律。它描述 WHAT，不规定单一实现细节；具体实现顺序见 `context/plans/build-site-fenicsx-petsc-eit-refactor.md`。

## Requirements

### R1: Solver policy is official-aligned and explicit

**Description:** 用户和代理必须能够明确区分 2D/debug direct solver、3D production AMG solver、Hypre/GAMG/AMGx/GPU 候选和 block/fieldsplit 候选；默认行为不得把大 3D 生产路径静默落到 LU。

**Acceptance Criteria:**
- [ ] 3D 默认 forward PETSc 配置解析为 iterative AMG-family preset，而不是 `preonly+lu`。**Gate 2:** `tests/unit/test_forward_solver_presets.py`
- [ ] 2D 或小规模 debug preset 仍能显式选择 direct/MUMPS reference。**Gate 2:** 新增或维护 preset unit test
- [ ] `solver_preset`、`ksp_type`、`pc_type`、`pc_hypre_type`、`pc_gamg_type`、`pc_factor_mat_solver_type` 和 `petsc_options` 都进入 backend signature。**Gate 2:** cache signature unit test
- [ ] unsupported preset 产生清晰错误，不能静默回退。**Gate 2:** preset validation unit test

**Dependencies:** `cavekit-forward-solver.md`, `cavekit-environment-cli.md`

### R2: Forward CEM solve reuses assembly and KSP/PC across RHS

**Description:** 对固定 `sigma,z`，CEM 系统矩阵必须只装配一次，KSP/PC setup 必须只创建一次，然后求解所有 current patterns。多 RHS 行为必须可诊断、可测试，并保留 `KSPSolve` loop 与 `KSPMatSolve` 两条路径。

**Acceptance Criteria:**
- [ ] 一个 forward solve 调用中，所有 stimulation patterns 共享同一个系统矩阵和 KSP bundle。**Gate 2:** fake PETSc call-count test
- [ ] `mat_solve_mode=auto|on|off` 行为稳定，3D aggressive 多 RHS 可自动选择 matSolve。**Gate 2:** `tests/unit/test_forward_mat_solve_policy.py`
- [ ] PETSc matSolve 失败时按设备策略处理：CPU 可回退，显式 CUDA failure 必须报错并给出诊断。**Gate 2:** `tests/unit/test_forward_solver_branch_suite.py`
- [ ] forward diagnostics 记录 effective solver、PC type、Mat/Vec type、multi-RHS mode、fallback reason。**Gate 2:** diagnostics unit test
- [ ] 2D/3D smoke forward 输出 finite voltages，并与 SciPy/direct reference 在小网格上满足容差。**Gate 3:** targeted integration/smoke

**Dependencies:** R1, `cavekit-cache-performance.md`

### R3: Forward production PC strategy avoids large-3D direct factorization

**Description:** 大 3D forward solve 的生产主力必须是 AMG/domain-decomposition/GPU-aware iterative path；direct solver 只能作为 debug/reference/coarse-grid/small-scale path。

**Acceptance Criteria:**
- [ ] 3D runtime diagnostics 对 direct preset 标记 `debug/reference` 或需要显式配置。**Gate 2:** policy diagnostic test
- [ ] 3D GAMG/Hypre preset 至少有 smoke coverage，能完成小 3D CEM solve。**Gate 3:** `nix develop -c uv run pytest --no-cov <3d-forward-smoke> -q`
- [ ] benchmark 输出记录 setup time、solve time、iteration count、RHS count、PC reuse state。**Gate 4:** new or existing 3D benchmark JSON/CSV
- [ ] 如果 GAMG/Hypre 不可用，错误或 fallback reason 必须包含可操作信息。**Gate 2:** capability/fallback test

**Dependencies:** R1, R2

### R4: Sensitivity engine exposes matrix-free `Jv` and `J^T r`

**Description:** Jacobian 层必须提供不 materialize dense `J` 的线性化接口，能计算 `Jv`、`J^T r` 和兼容 dense reference 的结果。dense `J` 只允许作为 2D/debug/reference 或小规模兼容路径。

**Acceptance Criteria:**
- [ ] `JacobianLinearization.matvec(v)` 与 dense `J @ v` 在 synthetic gradients 上一致。**Gate 2:** `tests/unit/test_jacobian_linearization.py`
- [ ] `JacobianLinearization.rmatvec(r)` 与 dense `J.T @ r` 一致。**Gate 2:** `tests/unit/test_jacobian_linearization.py`
- [ ] `DirectJacobianCalculator.linearize()` 返回 shape、measurement count、parameter count 正确的 operator。**Gate 2:** targeted calculator test
- [ ] 3D fast inverse path 可选择 matrix-free operator，而不是强制要求 dense `measurement_jacobian_np`。**Gate 2/3:** GN runtime test
- [ ] 当 dense materialization 被请求时，代码必须显式走 `to_dense()` 或 debug/reference API。**Gate 6:** review

**Dependencies:** `cavekit-inverse-reconstruction.md`, R2

### R5: Inverse linear subproblems use matrix-free Hessian actions

**Description:** 差分和绝对成像的快速/3D 线性子问题必须可用 `H v = J^T W J v + alpha R v` action 求解，而不构造 dense Hessian。

**Acceptance Criteria:**
- [ ] fast GN/差分路径支持 `LinearOperator` 或 `JacobianLinearization` 输入。**Gate 2:** GN fast linear solver unit tests
- [ ] `Hv` action 支持 measurement weights、prior term、regularization action。**Gate 2:** operator unit tests
- [ ] 对小网格，matrix-free 解与 dense reference 解在规定容差内一致。**Gate 3:** parity test
- [ ] 3D fast mode 不调用 `safe_dot(J.T, J)` 形成 dense `J^T J`，除非处于 explicit debug/reference path。**Gate 2/6:** code-path test + review
- [ ] solver meta 记录 selected path、preconditioner、fallback reason、iteration count。**Gate 2:** diagnostics test

**Dependencies:** R4, `cavekit-cache-performance.md`

### R6: Inverse preconditioning is explicit and compatible with matrix-free operators

**Description:** matrix-free Hessian 不能直接假设 ILU/GAMG 可用于 shell matrix；必须提供兼容的 explicit `Pmat`、diagonal/NOSER/prior/coarse approximation 或 custom shell preconditioner。

**Acceptance Criteria:**
- [ ] `petsc-gamg` 不再被误称为 matrix-free Hessian 的直接 PC；若未提供 `Pmat`，必须 fallback 并记录原因。**Gate 2:** existing or updated GN preconditioner tests
- [ ] `diag/NOSER` preconditioner 在 2D/3D shapes 上有限且正下界稳定。**Gate 2:** preconditioner unit test
- [ ] sparse `R`/prior precision 可作为 `LinearOperator` 或 sparse matrix action 注入 `Hv`。**Gate 2:** regularization action test
- [ ] Pmat/coarse Hessian 方案进入 plan，并有至少一个小规模 smoke。**Gate 3:** smoke/parity test

**Dependencies:** R5

### R7: Contact impedance and multi-variable inverse are block-ready

**Description:** 未来 `sigma + z_contact` 或更多参数联合估计必须能进入 block/fieldsplit/Schur 思路，至少先提供清晰数据结构和 out-of-scope 边界，避免后续 agents 把变量混成不可维护 dense monolith。

**Acceptance Criteria:**
- [ ] kit/plan 明确 `sigma` block 和 `z` block 的变量维度、scale、regularization 差异。**Gate 6:** human review
- [ ] 初始实现至少支持 block-diagonal approximation 的接口或 design stub。**Gate 2:** shape/unit test
- [ ] `z` block 可用 small dense/LU/Jacobi reference；`sigma` block 不默认 dense Hessian。**Gate 2/6:** code-path test + review
- [ ] fieldsplit/Schur 是计划中的升级路径，不阻塞当前 matrix-free `sigma` baseline。**Gate 6:** plan review

**Dependencies:** R5, R6

### R8: GPU/MPI behavior is capability-gated and diagnosed

**Description:** GPU/MPI 路径必须检查 PETSc CUDA Mat/Vec/Dense availability，并报告实际 effective device、Mat/Vec type、fallback reason 和 host-device transfer 风险。不能只因用户请求 `cuda` 就假定 GPU 路径有效。

**Acceptance Criteria:**
- [ ] `petsc_device=cuda` 在 PETSc CUDA 不可用时 fail fast，并给出 `nix develop .#cuda` 和 probe 命令提示。**Gate 2:** capability tests
- [ ] `petsc_device=auto` 在 fallback 时记录原因，不报假成功。**Gate 2:** capability tests
- [ ] CUDA matSolve 仅在 dense CUDA Mat 可用时启用。**Gate 2:** matSolve policy tests
- [ ] benchmark/diagnostic 输出包含 effective device、PETSc Mat/Vec type、forward backend、Jacobian backend。**Gate 4:** benchmark artifact check
- [ ] MPI size > 1 当前限制必须明确；解除限制前不得假装并行 3D production 已完成。**Gate 6:** review

**Dependencies:** `cavekit-environment-cli.md`, R2

### R9: Cache and reuse are semantically safe

**Description:** forward static setup、PETSc solver bundle、Jacobian/operator、regularization/preconditioner 等复用都必须由 semantic key 约束，不能因配置、mesh、patterns、`sigma,z` 或 device 变化而误用旧结果。

**Acceptance Criteria:**
- [ ] solver preset/PC/options/device/matSolve policy 参与 forward backend signature。**Gate 2:** cache signature tests
- [ ] `sigma,z,pattern,mesh,drive semantics` 改变会使 forward factor/operator cache invalid。**Gate 2:** cache tests
- [ ] Jacobian linearization 不跨不兼容 `sigma` 复用。**Gate 2:** operator cache test or explicit no-cache test
- [ ] preconditioner reuse 记录 setup count、reuse flag、iteration count；矩阵变化时复用必须可关闭。**Gate 4:** benchmark/diagnostic

**Dependencies:** `cavekit-cache-performance.md`, R1-R6

### R10: Validation is strict, documented, and resumable after context compression

**Description:** 每次实施必须更新 plan/impl tracking，写明做了什么、验证了什么、哪些没有验证、哪些路径不要重试。后续上下文压缩后，代理必须能从 Cavekit 文档恢复总目标和下一步。

**Acceptance Criteria:**
- [ ] 每个 implementation task 映射到至少一个 kit requirement 和 validation gate。**Gate 6:** plan review
- [ ] 每次修改后更新 `context/impl/impl-fenicsx-petsc-eit-refactor.md` 的 task status、files modified、test health、dead ends。**Gate 6:** tracking review
- [ ] 验证命令必须使用 WSL2/Nix 支持路径：`nix develop -c uv run ...`。**Gate 1/2:** actual command logs
- [ ] 聚焦测试使用 `--no-cov`，全量覆盖率测试单独运行并报告耗时/失败原因。**Gate 2:** test log
- [ ] 若全量 `tests/unit` 超时或失败，必须记录超时阈值、是否有残留进程、下一步分片策略。**Gate 6:** tracking review

**Dependencies:** all requirements

### R11: Canonical 2D/3D solver matrix is preserved

**Description:** 2D/3D 正逆 EIT 的 solver/PC 决策矩阵必须作为 durable project memory 保存，后续代理不得只凭聊天上下文或局部测试改动默认路线。任何默认路线变更必须引用官方文档、项目 benchmark 或失败验收证据。

**Acceptance Criteria:**
- [ ] `context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md` 包含 2D forward、2D inverse、3D forward、3D inverse、contact impedance、GPU/MPI 的 canonical solver/PC 策略。**Gate 6:** documentation review
- [ ] 该 canonical matrix 明确 `CG + AMG/Hypre` 只在 SPD 或已消除/处理约束后作为首选；不明 SPD 时使用 `FGMRES/MINRES` 和 block/fieldsplit 路线。**Gate 6:** documentation review
- [ ] 该 canonical matrix 明确 3D inverse 不默认 dense `J`、dense `J^T J` 或 direct inverse。**Gate 6:** documentation review
- [ ] 该 canonical matrix 明确 matrix-free Hessian 不能直接假设 `ILU/GAMG` 可用，必须提供 `Pmat`、diagonal/NOSER/prior/coarse action 或 `PCSHELL`。**Gate 6:** documentation review
- [ ] 后续实现修改 solver defaults 时，必须同时更新本 kit、implementation detail appendix 和 impl tracking。**Gate 6:** tracking review

**Dependencies:** R1-R10

## Out of Scope

- 不在本 kit 中替换 DOLFINx weak form、mesh、facet tag 和 baseline assembly 体系。
- 不在本 kit 中一次性实现完整 GPU FEM assembly。
- 不要求删除 SciPy/direct/MUMPS reference path。
- 不要求一次提交完成 fieldsplit/Schur/GPU/MPI 全部生产化；这些是分阶段目标。
- 不处理 GUI 交互细节，除非 GUI 暴露 solver policy/diagnostics 时需要同步。

## Cross-References

- Depends on: `cavekit-forward-solver.md` — CEM forward behavior和 PETSc backend 基础。
- Depends on: `cavekit-inverse-reconstruction.md` — GN/difference/sparse/reduced inverse workflows。
- Depends on: `cavekit-cache-performance.md` — semantic cache、benchmark、performance gates。
- Depends on: `cavekit-environment-cli.md` — Nix/uv/FEniCSx/CUDA runtime。
- Related reference: `context/refs/fenicsx-petsc-eit-refactor-research.md`。
- Implementation plan: `context/plans/build-site-fenicsx-petsc-eit-refactor.md`。
- Detailed 2D/3D implementation policy: `context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md`。
- Tracking: `context/impl/impl-fenicsx-petsc-eit-refactor.md`。
