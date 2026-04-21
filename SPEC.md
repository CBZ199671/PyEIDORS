# SPEC — PyEIDORS

Source: distilled from code at `acc4281` on `dev/gui-integration`. Flag `?` = inferred, user confirm.

## §G — goal

Python-first EIT framework. FEniCSx (DOLFINx) CEM forward + PyTorch-accel inverse + PETSc KSP/PC. EIDORS parity target; hard tolerance pending. Modern GPU/MPI path. Absolute + difference reconstruction, real-time mesh, benchmark.

**v1 main line (current focus):** EIDORS-style dual-model 3D difference EIT — fine-CEM forward mesh, coarse inverse voxel/tetra mesh, offline one-step GN / NOSER / Laplace / 3D GREIT reconstruction matrix (`RM`), online `x = RM @ normalize(Δv)`. Matrix-free GN-CG / IRGNM / TV / SBL / CNN post-processing are phase-2 / research tiers, not v1 blockers.

## §C — constraints

- Python 3.13.x, `>=3.13,<3.14`
- DOLFINx / FEniCSx: nix devShell owns runtime (`nix develop`)
- PETSc via `petsc4py`; MPI size == 1 enforced ? until distributed Mat/Vec lands
- PyTorch ~2.10, optional CUDA via `nix develop .#cuda`
- NumPy/SciPy/pandas/h5py/pyyaml mandatory; `pyamg`/`scikit-sparse` optional
- Nix + uv dev path, WSL2 supported; GUI launcher `scripts/gui/run_eit_app.sh`
- PySide6 GUI under `src/eit_app/`
- MIT license

## §I — interfaces

### Python API (top-level `pyeidors`)

- `EITForwardModel(n_elec, pattern_config, z, mesh, linear_backend="petsc", backend_config, forward_backend="dolfinx", cache_manager, performance_mode="aggressive")` — CEM forward solve, multi-RHS
- `LinearBackendConfig` fields: `solver_preset`, `ksp_type`, `pc_type`, `rtol`, `atol`, `max_it`, `reuse_preconditioner`, `monitor`, `mat_solve_mode`, `use_mat_solve`, `petsc_device`, `pc_factor_mat_solver_type`, `pc_hypre_type`, `pc_gamg_type`, `petsc_options`, `forward_pc_refresh_policy`, `forward_pc_refresh_iter_threshold`, `forward_pc_refresh_lag`, `forward_mat_solve_min_patterns`
- `DirectJacobianCalculator` — `calculate(sigma, method="efficient"|"traditional")`, `linearize(sigma, method="efficient")`, `block_tuning_info()`
- `JacobianLinearization` — `matvec`, `rmatvec`, `normal_matvec`, `as_linear_operator`, `to_dense`, `hessian_diag(measurement_weights, alpha, regularization_diag, floor)`, `as_petsc_mat`, `assert_compatible(sigma_fingerprint)`; fields `grad_u_all`, `adjoint_gradients`, `cell_areas`, `n_meas_per_stim`, `sign`, `sigma_fingerprint`
- `compute_sigma_fingerprint(sigma) -> str`
- `GaussNewtonReconstructor` + `run_reconstruction(reconstructor, measured_data, jacobian_method="efficient"|"linearized"|"operator"|"matrix-free")`
- `pyeidors.inverse.block_system`: `ParameterBlock`, `BlockCoupling`, `JointInverseBlockMetadata`, `build_sigma_contact_block_metadata(n_sigma, n_contact, n_measurements)`, `make_block_diagonal_inverse_action`, `scale_contact_impedance_update(current_z, delta_z, max_relative_step, floor)`
- Data: `PatternConfig`, `EITMesh`, `EITData`, `EITImage`
- Cache: `build_process_forward_setup_key(*, mesh_file, mesh_content_hash, n_elec, z, pattern_config)`, `backend_signature_from_forward_model`, `model_signature_from_forward_model`, `pattern_signature_from_forward_model`, `rom_signature`, semantic `cache_manager.get_or_compute_semantic`
- Perf: `detect_performance_capabilities`, `select_preconditioner`, `select_fast_linear_path`, `select_fused_strategy`, `probe_mpi_runtime`
- Forward diagnostics: `EITForwardModel.get_backend_diagnostics()` (see §I.diag)

### CLI / scripts

- `scripts/run_synthetic_parity.py` — forward+inverse parity check
- `scripts/run_reconstruction_unified.py` — unified reconstruction runner; `--preconditioner diag|noser|prior|pmat|coarse|custom|petsc-gamg|cholmod|pyamg`
- `scripts/benchmarks/benchmark_3d_runtime.py` — `--forward-only on|off`, `--forward-solver-preset 3d_gamg|3d_hypre|spd_gamg|spd_hypre|direct|mumps|3d_amg|hypre_boomeramg`, `--forward-mat-solve auto|on|off`, `--petsc-device auto|cpu|cuda`
- `scripts/benchmarks/benchmark_difference_runtime.py`
- `scripts/diagnostics/probe_petsc_cuda.py` — PETSc CUDA + MPI probe, `--pretty`, `--require cuda`
- `scripts/ci/run_sharded_unit_tests.py` — `--shard <name>` | `--all`, `--timeout`, `--report-dir`, per-shard JSON summary
- `scripts/gui/run_eit_app.sh` / `.ps1` / `EIT-GUI-CPU.cmd` / `EIT-GUI-GPU.cmd`

### §I.diag — diagnostics surface

- Forward `get_backend_diagnostics()` keys: `forward_rhs_count`, `forward_ksp_solve_count`, `forward_ksp_mat_solve_count`, `forward_ksp_setup_count`, `forward_ksp_setup_attempts`, `forward_reuse_preconditioner_requested`, `forward_reuse_preconditioner_applied`, `forward_pc_session_reused`, `forward_pc_refresh_triggered`, `forward_pc_refresh_reason`, `ksp_type`, `pc_type`, `pc_factor_mat_solver_type`, `petsc_mat_type`, `petsc_vec_type`, `petsc_dense_mat_type`, `petsc_solve_mat_type`, `forward_mat_solve_effective`, `forward_ksp_iterations_per_rhs`, `forward_ksp_iterations_total`, `forward_ksp_converged_reason`, `forward_ksp_converged`, `forward_setup_seconds`, `forward_solve_seconds`, `forward_factor_cache_hit`, `gpu_fallback_reason`, `fallback_reason`, `forward_mat_solve_fallback_reason`, MPI size/rank/support fields
- Inverse `_last_fast_linear_meta` keys: `path`, `resolved_preconditioner`, `fallback_reason`, `fast_linear_path_selected`, `fast_linear_path_reason`, `jacobian_representation`, `jacobian_shape`, `dense_jacobian_materialized`, `linear_iterations`, `matrix_free_pc_source`, `matrix_free_pc_mode`, `matrix_free_pc_floor`, `matrix_free_pc_min`, `matrix_free_pc_max`, `matrix_free_pc_reason`, `matrix_free_pmat_available`, `matrix_free_pmat_kind`, `matrix_free_pmat_attr`, `matrix_free_ksp_backend_requested`, `matrix_free_ksp_backend_effective`, `matrix_free_ksp_backend_fallback_reason`
- Benchmark artifact `forward_solver_benchmark` JSON: mesh/RHS/solver/PC/Mat/Vec/timing/iterations/device/fallback/finite-output/CUDA-errors/MPI fields

### §I.future — planned v1 surfaces (not yet implemented)

Provisional module paths; see §T.T15..T32 for per-feature scope. All marked `?` until the corresponding `x` lands in §T.

**Existing hooks v1 MUST build on (not replace):**
- `src/pyeidors/data/difference.py:66` `build_difference_vector` / `project_measurement_jacobian` — existing normalized/raw difference math. V39 gates parity of the new `normalize_time_difference` against this.
- `src/pyeidors/core_system.py:807` regularization dispatcher (NOSER / Tikhonov / Smoothness / TV) + `eidors_one_step_noser` preset — current "one-step NOSER on the current parameter mesh". v1 extends this to dual-mesh + offline RM form; V38 gates numerical parity with this baseline on small meshes.
- `src/pyeidors/inverse/jacobian/linearized.py:333` lazy adjoint matrix-free path — retained for phase-2 (`T22`, `T23`); not a v1 main-line dependency. Cold-path slowness observed in `lazy_adjoint_matrix_free_experiment_log_20260421.md:1` is part of why v1 prefers offline RM + online matmul.

- `pyeidors.inverse.dual_mesh`: `DualMesh(fine_mesh, coarse_mesh)`, `coarse2fine(mesh_fine, mesh_coarse) -> csr_matrix`
- `pyeidors.inverse.reconstruction_matrix`:
  - `build_one_step_rm(J, regularization, lambda_, mode="tikhonov"|"noser"|"laplace", form="param"|"measurement") -> ndarray`
  - `reconstruct_difference(rm, dv, normalize=True, v_ref=None) -> ndarray`
- `pyeidors.inverse.prior.laplace`: `graph_laplacian(mesh, weight="unit"|"volume") -> csr_matrix`
- `pyeidors.inverse.greit`:
  - `build_3d_greit_rm(fwd_model, targets, noise_figure, regularisation) -> GREITRM`
  - `GREITRM.reconstruct(dv) -> voxel_image`
  - `greit_metrics(voxel_image, target_mask) -> {AR, PE, RES, SD, RNG}`
- `pyeidors.inverse.matrix_free.dual_mesh`: `DualMeshJacobianOperator(fwd_model, coarse2fine)` exposing `Jv`, `JTr`, `normal_matvec`
- `pyeidors.inverse.block_system` (extended): `build_sigma_contact_movement_block_metadata(n_sigma, n_contact, n_electrodes_dofs)`; adds `H_sigma_e`, `H_z_e`, `H_e_e` couplings + `prior_movement` hook
- `pyeidors.data.difference.normalize_time_difference(v_t, v_ref, floor=...) -> dv_norm`
- `pyeidors.data.channels.bad_channel_mask`: apply mask to `J`, `residual`, `W`
- `pyeidors.perf.gpu_kernels.rm_matmul`: batched `RM @ ΔV` on GPU (torch / cupy)

CLI additions (under `scripts/run_reconstruction_unified.py` or new scripts):

- `--algorithm one-step-gn|noser|laplace|greit-3d|matrix-free-gn-cg|tv-pdhg`
- `--dual-mesh on|off`, `--coarse-mesh <path>`
- `--rm-cache <path>`: load / save precomputed RM
- `--normalize-difference on|off`
- `--greit-targets <path>`, `--greit-metrics-out <path>`
- `--bad-channel-mask <csv|json>`

## §V — invariants

| id | invariant | source |
|----|-----------|--------|
| V1 | CEM forward: single matrix assembly + single `ksp.setUp()` per `sigma`; all RHS share KSP bundle | tests/unit/test_forward_petsc_multirhs.py, test_forward_mat_solve_policy.py |
| V2 | `matSolve` auto-mode: true iff `mesh_tdim==3 AND n_patterns>1 AND performance_mode=="aggressive" AND (min_patterns==0 OR n_patterns>=min_patterns)` | tests/unit/test_forward_mat_solve_policy.py |
| V3 | `matSolve` failure → CPU vec-loop fallback with `forward_mat_solve_fallback_reason`; CUDA raises `RuntimeError` w/ `nix develop .#cuda` guidance | tests/unit/test_forward_mat_solve_policy.py |
| V4 | `backend_signature_from_forward_model` includes `solver_preset`, `pc_factor_mat_solver_type`, `pc_hypre_type`, `pc_gamg_type`, `petsc_options`, `petsc_device_effective`, `mat_solve_mode` | src/pyeidors/cache/object_signature.py |
| V5 | Unknown `solver_preset` → `ValueError` listing sorted preset names | tests/unit/test_forward_solver_presets.py |
| V6 | Auto preset: `mesh_tdim<3` → `direct` (`preonly+lu`); `mesh_tdim>=3` → `3d_gamg` (`fgmres+gamg+agg`); explicit `ksp_type`/`pc_type` → `custom` preset | src/pyeidors/forward/eit_forward_model.py `_resolve_linear_backend_config` |
| V7 | `JacobianLinearization.matvec(v)` == `to_dense() @ v`; `rmatvec(r)` == `to_dense().T @ r` | tests/unit/test_jacobian_linearization.py |
| V8 | `hessian_diag(measurement_weights=W, alpha, regularization_diag=R_diag, floor)` == `diag(J^T diag(W) J) + alpha*R_diag` clamped `≥ floor`; sign² applied | tests/unit/test_jacobian_linearization.py |
| V9 | `JacobianLinearization.assert_compatible(fp)` permissive when either stored or provided fingerprint empty; raises `ValueError` on mismatch | tests/unit/test_jacobian_linearization.py |
| V10 | Fast PCG parity with dense GN solution within `rtol 1e-5` for PC modes `{diag, noser, prior, pmat, coarse, custom, cholmod}` (tested); `pyamg` mode code path exists but lacks parity smoke ? — see T13; `petsc-gamg` without Pmat → fallback `diag` + reason `petsc_gamg_not_supported_in_matrix_free` | tests/unit/test_gn_fast_linear_solver.py |
| V11 | `matrix_free_pc_source ∈ {dense-sensitivity, explicit, matrix_free_hessian_diag, hessian_diag, noser, prior, auto_linearization_diag, pmat, coarse-pmat, custom-pcshell, identity}`; `matrix_free_pc_reason` populated on clamp/fallback | tests/unit/test_gn_fast_linear_solver.py |
| V12 | `reconstructor.matrix_free_ksp_backend ∈ {scipy, petsc, auto}`; `petsc4py` missing when `petsc` requested → fallback `scipy` + `matrix_free_ksp_backend_fallback_reason="petsc_backend_unavailable"` | tests/unit/test_gn_fast_linear_solver.py |
| V13 | Forward KSP session reused across `_solve_with_petsc` calls when structural fingerprint stable; `ksp.setOperators(A_new)` + `setReusePreconditioner(True)` skips `PCSetUp` | tests/unit/test_forward_ksp_session_reuse.py |
| V14 | PC refresh under `forward_pc_refresh_policy="auto"` triggers when last iter count > `forward_pc_refresh_iter_threshold`; `lag` rebuilds every `forward_pc_refresh_lag` calls; `never` always reuses; structural change forces rebuild | tests/unit/test_forward_ksp_session_reuse.py |
| V15 | Jacobian reuse honors `jacobian_update_every` + `jacobian_reuse_tol`; operator path (`JacobianLinearization`/`LinearOperator`) respects same gating via `_is_matrix_free_jacobian` | tests/unit/test_gn_runtime_run_reconstruction_branches.py |
| V16 | `build_process_forward_setup_key(*, mesh_file, mesh_content_hash, …)` requires at least one non-empty of `mesh_file`/`mesh_content_hash`; both empty → `ValueError` | tests/unit/test_forward_process_setup_cache.py |
| V17 | `_hash_mesh_content(dolfinx_mesh)` content-addresses `geometry.x` + `topology.connectivity(tdim,0).array`; returns `""` when mesh lacks geometry/topology (unit-fake permissive) | tests/unit/test_forward_process_setup_cache.py |
| V18 | MPI `comm.size > 1` raises `RuntimeError` at `EITForwardModel.__init__`; single-rank-only phase, T2 unlocks distributed Mat/Vec + `mpiexec -n 2` smoke | src/pyeidors/forward/eit_forward_model.py `_assert_supported_mpi_runtime`, src/pyeidors/perf/capabilities.py `probe_mpi_runtime` |
| V19 | `petsc_device="cuda"` w/o PETSc CUDA capability → `RuntimeError` with `nix develop .#cuda` guidance; `auto` falls back to CPU with `gpu_fallback_reason="petsc_cuda_not_available"` | tests/unit/test_forward_mat_solve_policy.py, scripts/diagnostics/probe_petsc_cuda.py |
| V20 | `build_sigma_contact_block_metadata` yields `ParameterBlock(sigma)` + `ParameterBlock(z_contact)` w/ couplings `H_sigma_z`, `H_z_sigma`, measurement jacobian blocks; `fieldsplit_type ∈ {additive, multiplicative, schur}`; `scale_contact_impedance_update` keeps finite + floor-guarded updates | tests/unit/test_inverse_block_system.py |
| V21 | Field-data golden metrics only ? — `correlation≈0.9888`, `RMSE≈4.45e-05`; no synthetic EIDORS hard-tol gate yet | reports/difference_single_step_golden_config.md:31-32, scripts/run_synthetic_parity.py |
| V22 | Sharded unit runner: per-shard JSON summary + recoverable logs; default `gui` shard, opt-in `hardware` shard separate | tests/unit/test_ci_sharded_unit_validation.py, docs/VALIDATION_SHARDS.md |
| V23 | Forward `KSPSetReusePreconditioner(True)` semantics: same KSP, new `setOperators(A_new)`, reuse holds until explicit refresh; iter-count-monitored | src/pyeidors/forward/eit_forward_model.py `ForwardKSPSession`, PETSc `KSPSetReusePreconditioner` manpage |
| V24 | Direct PC (`ksp_type=="preonly"` AND `pc_type ∈ {lu, cholesky, qr}`) never reused across sigma — session forces PC refresh with `forward_pc_refresh_reason="direct_factor_requires_rebuild"`. `preonly` has no Krylov iteration to correct a stale exact factor, unlike iterative+AMG where reuse is a staleness penalty | src/pyeidors/forward/eit_forward_model.py `_decide_pc_reuse_for_session` |
| V25 | Dual-mesh separation: forward CEM assembly on fine 3D mesh; inverse unknowns on coarse voxel / tetra grid; `coarse2fine` linear map projects `Δσ_coarse → Δσ_fine` before fine forward solve (EIDORS dual-model parity) | src/pyeidors/inverse/dual_mesh.py; tests/unit/test_forward_model_3d_cem.py:61-91; tests/unit/test_dual_mesh.py |
| V26 | One-step GN RM (`M ≪ N` form): `RM = P Jᵀ (J P Jᵀ + λ² Rn)⁻¹` with `P ≈ R⁻¹`; inversion happens on `M×M` measurement-space system, not on `N×N` parameter-space system | src/pyeidors/inverse/reconstruction_matrix.py:323-421; tests/unit/test_reconstruction_matrix.py:77-118,152-170 |
| V27 | NOSER RM variant: `R = diag(JᵀJ)`, `RM = (JᵀJ + h² R)⁻¹ Jᵀ`; row-normalized `normalize(Δv)` supported | src/pyeidors/inverse/reconstruction_matrix.py:266-300,323-421; tests/unit/test_reconstruction_matrix.py:40-49,100-115,257-268; tests/unit/test_one_step_rm_parity.py:95-146 |
| V28 | Laplace prior `R_L`: graph-Laplacian over inverse mesh cell neighbours; edge weight from adjacency + optional element-volume scaling | src/pyeidors/inverse/prior/laplace.py; tests/unit/test_laplace_prior.py; tests/unit/test_reconstruction_matrix.py:51-74,118-151 |
| V29 | 3D GREIT RM precomputed offline from synthetic targets; online reconstruction is `x = RM @ dv_norm` single matmul (no KSP solve per frame) | src/pyeidors/inverse/greit.py:207-329; tests/unit/test_greit_rm.py:49-158; tests/unit/test_fenicsx_eit_3d_v1_milestone.py:137-150 |
| V30 | GREIT metrics `{AR, PE, RES, SD, RNG}` computed per reconstruction against a target mask; documented per EIDORS GREIT evaluation protocol | src/pyeidors/inverse/greit.py:20,330-407; tests/unit/test_greit_rm.py:159-193; tests/unit/test_fenicsx_eit_3d_v1_milestone.py:151-159 |
| V31 | Matrix-free Jv/JTr on dual mesh: `Jv(δσ_coarse) = fine_forward(c2f @ δσ_coarse)`, `JTr(r) = c2f.T @ fine_adjoint_grad(r)`; parity against dense reference on small mesh inside tol | src/pyeidors/inverse/matrix_free/dual_mesh.py; tests/unit/test_dual_mesh_matrix_free.py |
| V32 | Joint parameter block `[σ, z_contact, e]` where `e` is electrode pose / motion nuisance; fieldsplit additive→multiplicative→Schur upgrade path extends V20 with `e` block; `prior_movement` regularizes `e` block | extends V20 — future `pyeidors.inverse.block_system` (σ+z exists; +e pending) ? |
| V33 | Normalized time difference: `dv_norm = (v_t - v_ref) / v_ref`; `v_ref` zero-guard (floor or mask); sign orientation consistent with existing `difference_orientation` contract | src/pyeidors/data/difference.py:75-124; tests/unit/test_difference_semantics.py:33-58 |
| V34 | Bad-channel mask `chan_mask` zeroes corresponding rows of `J`, residual entries, and measurement weights `W`; mask survives through offline RM build so precomputed `RM` respects the exact mask used at acquisition | src/pyeidors/data/channels.py:30-215; tests/unit/test_channels.py; tests/unit/test_reconstruction_matrix.py:173-242 |
| V35 | Noise covariance `W` symmetric in Hv contract: `Hv = Jᵀ W J v + α R v`; identical `W` used during offline RM build and online residual weighting; diagonal `W = diag(1/σ²_m)` is the default, full cov ring-fenced for future work | src/pyeidors/data/channels.py:153-215; src/pyeidors/inverse/reconstruction_matrix.py:323-421,482-540; tests/unit/test_reconstruction_matrix.py:173-242; tests/unit/test_rm_v1_artifacts.py:62-87 |
| V36 | RM cache signature includes ALL of: forward-mesh hash, inverse-mesh hash, `coarse2fine` hash, electrode geometry (count + ring layout), stim/meas protocol, background `(σ0, z0)`, difference mode (`raw` / `normalized`), bad-channel mask, noise covariance `W`, regularization type (`tikhonov` / `noser` / `laplace` / `greit`), λ / hyperparameter. Device/backend affect storage path only, NOT the mathematical signature. Any change in the above MUST invalidate the stored RM; device swap MUST NOT | src/pyeidors/inverse/reconstruction_matrix.py:39-95; tests/unit/test_rm_v1_artifacts.py:35-60 |
| V37 | Online reconstruction path executes exactly `RM @ dv` or `RM @ normalize_time_difference(v_t, v_ref)` — no Jacobian rebuild, no KSP solve, no forward/adjoint assembly in the hot path. Test asserts zero forward-solve counter ticks during `N` consecutive online frames | src/pyeidors/inverse/reconstruction_matrix.py:450-540; tests/unit/test_rm_v1_artifacts.py:62-87 |
| V38 | Small-mesh numerical parity: `build_one_step_rm(mode="noser", form="param")` on a small mesh matches the existing dense one-step NOSER baseline (`eidors_one_step_noser`, `src/pyeidors/core_system.py:785`) with `step_size=1`, `line_search_mode="off"`, `difference_step_size_mode="off"`, `jacobian_update_every=1`, and identical normalized/raw difference projection. Parity test must isolate the RM formula from any step-size or line-search optimisation path | tests/unit/test_one_step_rm_parity.py:38-177 |
| V39 | Normalized-difference parity: `normalize_time_difference(v_t, v_ref)` returns the same Δv vector as the existing `build_difference_vector(..., mode="normalized", orientation="target_minus_reference")` at `src/pyeidors/data/difference.py:66` for the same inputs. Default orientation stays `target_minus_reference`; other orientations gated by explicit opt-in | src/pyeidors/data/difference.py:75-124; tests/unit/test_difference_semantics.py:33-58 |
| V40 | Offline (cold) RM-build time and online (warm) RM-apply time are recorded as separate fields in the benchmark artifact; online field dominated by a single dense matmul, cold field allowed arbitrary minutes | src/pyeidors/inverse/reconstruction_matrix.py:98-124; tests/unit/test_rm_v1_artifacts.py:90-105 |
| V41 | 3D GREIT output emits the full metric set `{AR, PE, RES, SD, RNG}` per reconstruction — absence of any single metric fails the GREIT validation gate | src/pyeidors/inverse/greit.py:410-453,762-770; tests/unit/test_greit_rm.py:196-235 |

## §T — tasks

### §T.phase — v1 queue and tiering

Current priority queue. Cavekit `/ck:make` MUST advance along the v1 queue before touching phase-2 / research tiers.

```
v1  (EIDORS-style dual-model offline-RM + online RM@normalize(Δv)):
    T15  →  T18  →  T16  →  T17  →  T26  →  T29  →  T31  →  T19  →  T20  →  T32

phase-2  (after v1 stable, higher-fidelity reconstruction):
    T22, T23, T25, T24, T21, T30

research  (post-phase-2, opt-in enhancement):
    T27, T28, T1, T11

infra / deferred  (hardware / design-heavy, unblock separately):
    T2, T3, T4, T6, T7, T10
```

T32 is the v1 closure milestone and ties in the GREIT metrics, so it
MUST follow T19/T20. Earlier queue drafts put T32 before T19/T20 — do
not reintroduce that ordering; the milestone cannot validate
`{AR, PE, RES, SD, RNG}` before those tasks land.

v1 graduation gate: all rows T15..T20, T26, T29, T31, T32 must be `x` AND V36..V41 must hold before T22+ are eligible.

| id | status | desc | cites |
|----|--------|------|-------|
| T1 | . | Full PETSc `PCFIELDSPLIT` inverse solver for `sigma + z_contact` joint estimation (additive → multiplicative → Schur) | V20 |
| T2 | . | Enable MPI size > 1: distributed Mat/Vec + `mpiexec -n 2` smoke; lift fail-fast guard | V18 |
| T3 | . | Flip `matrix_free_ksp_backend` default to `auto` once 3D benchmark parity vs scipy holds | V12 |
| T4 | . | Real 3D benchmark artifact proving G1 persistent-KSP setup-time saved (iter histogram + cumulative setup seconds) | V13,V14 |
| T5 | x | Wire `JacobianLinearization.assert_compatible(sigma_fp)` at runtime reuse path; stored fingerprint currently inert | V9,V15 |
| T6 | . | Persistent across-iteration Jacobian cache keyed on `sigma_fingerprint` + mesh content hash | V9,V17 |
| T7 | . | CUDA 3D inverse benchmark gated by `probe_petsc_cuda --require cuda` | V19 |
| T8 | x | Guard canonical solver/PC matrix doc against preset-default drift (R11 hard gate) | V4,V5,V6 |
| T9 | x | Explicit startup cache skip for operator Jacobian (avoid `np.asarray(JacobianLinearization, dtype=float)` path) | V15 |
| T10 | . | PETSc AmgX / Hypre CUDA path wiring + capability probe entries in benchmark artifact | V19 |
| T11 | . | Research: PETSc/petsc4py structural reuse hints. `KSPSetOperators(ksp, Amat, Pmat)` has no `SAME_NONZERO_PATTERN` parameter in current API; `petsc4py.KSP.setOperators(A=None, P=None)` likewise. Current main line stays `setOperators(A_new)` + `KSPSetReusePreconditioner(True)` | V13 |
| T12 | x | `forward_pc_session_reused` / `forward_pc_refresh_*` diagnostics covered by `tests/unit/test_forward_ksp_session_reuse.py:189` | V13,V14 |
| T13 | x | Add dense-reference parity smoke for `pyamg` matrix-free PC mode (currently only code path + fallback covered; no PC-output parity assertion) | V10 |
| T14 | x | Guard `_decide_pc_reuse_for_session` against cross-sigma reuse when `ksp_type==preonly` and `pc_type ∈ {lu, cholesky, qr}` | V24,B1 |
| T15 | x | Dual-mesh data structure + `coarse2fine` sparse map builder; fine CEM mesh and coarse inverse voxel / tetra mesh live side-by-side, map is linear projection (piecewise-constant fallback). Prepares V31 but does NOT own it — matrix-free Jv/JTr on dual mesh stays in T22 | V25 |
| T16 | x | One-step GN RM builder: Tikhonov / NOSER / Laplace modes in a single entrypoint returning `RM`, metadata (mode, λ, condition estimate) | V26,V27,V28 |
| T17 | x | Measurement-space RM path `RM = P Jᵀ (J P Jᵀ + λ² Rn)⁻¹` when `M ≪ N`; dense-reference parity test on small dual mesh | V26 |
| T18 | x | Normalized-time-difference front-end: reference-frame capture, `v_ref` zero-guard, `RM @ dv_norm` wiring through the existing difference-mode contract | V33 |
| T19 | x | 3D GREIT RM builder: synthetic training targets (spheres / blobs at grid positions), offline precompute + persistence to disk artifact | V29 |
| T20 | x | GREIT metrics module computing `{AR, PE, RES, SD, RNG}` with a reference target mask, plus CSV / JSON artifact writer | V30 |
| T21 | . | Temporal smoothing + TV postprocess pipeline on RM output (moving-average, exponential decay, 3D TV regulariser on voxel grid) | V28 |
| T22 | x | Matrix-free `Jv` / `JTr` extended over dual mesh (coarse inverse parameter → fine forward) with dense parity test on small mesh | V25,V31 |
| T23 | . | IRGNM / LM wrapper around matrix-free Hv for absolute-ish 3D reconstruction, reusing the existing `_solve_pcg` PETSc or SciPy backend | V12,V31,V35 |
| T24 | . | TV PDHG / PDIPM refinement on ROI after one-step init; seeded by RM output, stops on ROI-restricted residual norm | V26,V28 |
| T25 | . | Electrode-movement Jacobian + `prior_movement`; extends block metadata with `e` block and `H_σe`, `H_ze`, `H_ee` couplings | V20,V32 |
| T26 | x | Bad-channel mask + noise covariance `W` wired into Jacobian rows, residual vector, measurement-weight contract, and RM builder so offline / online weights match | V34,V35 |
| T27 | . | SBL / coarse-basis research enhancement (RBF, sparse-inclusion, low-rank anatomical basis) — tier 3, post-v1 | V31 |
| T28 | . | CNN / U-Net postprocess plug-in interface (coarse 3D image in → enhanced image out), no physics replacement | V29 |
| T29 | x | GPU `RM @ ΔV` online kernel: batched multi-frame matmul on GPU, reuses normalized-difference path | V29,V35 |
| T30 | . | Hypre `BoomerAMG` / NVIDIA AmgX CUDA path wiring for forward CG (extends T10), with capability-probe entries in `forward_solver_benchmark` artifact | V6,V13,T10 |
| T31 | x | Dual-mesh integration smoke: fine CEM + coarse recon + EIDORS-style parity metric on synthetic sphere target | V25,V29,V30 |
| T32 | x | Milestone **FEniCSx-EIT-3D-v1**: ties V25–V35 + GPU online matmul; 10-point checklist (fine CEM, coarse voxel, c2f, reusable KSP/PC, adjoint J on coarse, one-step GN/NOSER/Laplace RM, normalized Δv, GPU RM@Δv, GREIT metrics, bad-channel / W weighting) | V25,V26,V27,V28,V29,V30,V31,V33,V34,V35 |

## §B — bugs

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | 2026-04-20 | `ForwardKSPSession` applied `setReusePreconditioner(True)` uniformly; for `ksp_type==preonly` + `pc_type∈{lu,cholesky,qr}` this silently reuses stale LU/Cholesky/QR factorisation across sigma updates, solving `A(σ_new) x = b` with `A(σ_old)^{-1}`. No Krylov iteration to correct the error (unlike iterative+AMG where reuse is a staleness penalty). | V24,T14 |
