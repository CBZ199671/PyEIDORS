# SPEC — PyEIDORS

Source: distilled from code at `acc4281` on `dev/gui-integration`. Flag `?` = inferred, user confirm.

## §G — goal

Python-first EIT framework. FEniCSx (DOLFINx) CEM forward + PyTorch-accel inverse + PETSc KSP/PC. EIDORS parity target; hard tolerance pending. Modern GPU/MPI path. Absolute + difference reconstruction, real-time mesh, benchmark.

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
| V10 | Fast PCG parity with dense GN solution within `rtol 1e-5` for all PC modes `{diag, noser, prior, pmat, coarse, custom, cholmod, pyamg}`; `petsc-gamg` without Pmat → fallback `diag` + reason `petsc_gamg_not_supported_in_matrix_free` | tests/unit/test_gn_fast_linear_solver.py |
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
| V21 | Field-data golden metrics only ? — `correlation≈0.9888`, `RMSE≈4.45e-05`; no synthetic EIDORS hard-tol gate yet | reports/difference_single_step_golden_config.md:1, scripts/run_synthetic_parity.py |
| V22 | Sharded unit runner: per-shard JSON summary + recoverable logs; default `gui` shard, opt-in `hardware` shard separate | tests/unit/test_ci_sharded_unit_validation.py, docs/VALIDATION_SHARDS.md |
| V23 | Forward `KSPSetReusePreconditioner(True)` semantics: same KSP, new `setOperators(A_new)`, reuse holds until explicit refresh; iter-count-monitored | src/pyeidors/forward/eit_forward_model.py `ForwardKSPSession`, PETSc `KSPSetReusePreconditioner` manpage |

## §T — tasks

| id | status | desc | cites |
|----|--------|------|-------|
| T1 | . | Full PETSc `PCFIELDSPLIT` inverse solver for `sigma + z_contact` joint estimation (additive → multiplicative → Schur) | V20 |
| T2 | . | Enable MPI size > 1: distributed Mat/Vec + `mpiexec -n 2` smoke; lift fail-fast guard | V18 |
| T3 | . | Flip `matrix_free_ksp_backend` default to `auto` once 3D benchmark parity vs scipy holds | V12 |
| T4 | . | Real 3D benchmark artifact proving G1 persistent-KSP setup-time saved (iter histogram + cumulative setup seconds) | V13,V14 |
| T5 | . | Wire `JacobianLinearization.assert_compatible(sigma_fp)` at runtime reuse path; stored fingerprint currently inert | V9,V15 |
| T6 | . | Persistent across-iteration Jacobian cache keyed on `sigma_fingerprint` + mesh content hash | V9,V17 |
| T7 | . | CUDA 3D inverse benchmark gated by `probe_petsc_cuda --require cuda` | V19 |
| T8 | . | Guard canonical solver/PC matrix doc against preset-default drift (R11 hard gate) | V4,V5,V6 |
| T9 | . | Explicit startup cache skip for operator Jacobian (avoid `np.asarray(JacobianLinearization, dtype=float)` path) | V15 |
| T10 | . | PETSc AmgX / Hypre CUDA path wiring + capability probe entries in benchmark artifact | V19 |
| T11 | . | Research: PETSc/petsc4py structural reuse hints. `KSPSetOperators(ksp, Amat, Pmat)` has no `SAME_NONZERO_PATTERN` parameter in current API; `petsc4py.KSP.setOperators(A=None, P=None)` likewise. Current main line stays `setOperators(A_new)` + `KSPSetReusePreconditioner(True)` | V13 |
| T12 | x | `forward_pc_session_reused` / `forward_pc_refresh_*` diagnostics covered by `tests/unit/test_forward_ksp_session_reuse.py:189` | V13,V14 |

## §B — bugs

| id | date | cause | fix |
|----|------|-------|-----|
