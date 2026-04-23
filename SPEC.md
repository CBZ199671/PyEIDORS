# SPEC — PyEIDORS

Source: distilled from code at `acc4281` on `dev/gui-integration`. Flag `?` = inferred, user confirm.

## §G — goal

Python-first EIT framework. FEniCSx (DOLFINx) CEM forward + PyTorch-accel inverse + PETSc KSP/PC. EIDORS parity target; hard tolerance pending. Modern GPU/MPI path. Absolute + difference reconstruction, real-time mesh, benchmark.

**v1 main line (current focus):** EIDORS-style dual-model 3D difference EIT — fine-CEM forward mesh, coarse inverse voxel/tetra mesh, offline one-step GN / NOSER / Laplace / 3D GREIT reconstruction matrix (`RM`), online `x = RM @ normalize(Δv)`. Matrix-free GN-CG / IRGNM / TV / SBL / CNN post-processing are phase-2 / research tiers, not v1 blockers.

**Current 3D repo decision:** production/default path = `spd_gamg + petsc_device=cuda + forward_mat_solve=off` forward, cached dual-model one-step RM/NOSER/Laplace first, 3D GREIT next after RM hot-path optimization. Dense-J GN, direct LU/MUMPS, CUDA `KSPMatSolve`, matrix-free GN-CG, TV, SBL are baseline / debug / phase-2 / research, not realtime 48e/5936 default.

**3D route tiers (final):**
- v1 production: dual-model + one-step GN/NOSER/Laplace RM; online `RM @ normalized ΔV`.
- v1 enhancement: 3D GREIT RM; best for fixed geometry/protocol + multi-frame dynamic EIT.
- forward production default: 3D DOLFINx CEM + PETSc `spd_gamg` CUDA + vec-loop; reuse multi-RHS/KSP/PC/cache.
- phase-2 quality: matrix-free GN-CG/IRGNM + dual mesh + explicit Pmat/NOSER/prior preconditioner.
- postprocess/robustness: TV/PDHG, bad-channel mask, noise covariance, temporal prior.
- research: SBL/sparse Bayesian, self-built PETSc AmgX, full-chain GPU Jacobian.
- not mainline: full 3D dense-J absolute GN, full 3D direct LU/Cholesky, per-frame forward+Jacobian+GN.

**3D GREIT parity note:** current code = linearized 3D GREIT-RM v0 (`Y = T @ J.T`, `D ≈ T`). EIDORS-complete target = `GREIT3D_distribution` + finite-target forward training `vh/vi` + `desired_solution_fn` + `calc_GREIT_RM` + NF weight search + MATLAB parity gates + HDF5 large-cache artifact. No UI/docs may claim "EIDORS同款 / official-equivalent / perfect stable 3D GREIT" until T40..T50 x and V55..V65 hold.

## §C — constraints

- Python 3.13.x, `>=3.13,<3.14`
- DOLFINx / FEniCSx: nix devShell owns runtime (`nix develop`)
- PETSc via `petsc4py`; MPI size == 1 enforced ? until distributed Mat/Vec lands
- PyTorch ~2.10, optional CUDA via `nix develop .#cuda`
- AmgX ? research-only: no FEniCSx version bump required; requires Nix rebuild of PETSc with PCAMGX + same-chain `petsc4py`/SLEPc/DOLFINx/`fenics-dolfinx`
- NumPy/SciPy/pandas/h5py/pyyaml mandatory; `pyamg`/`scikit-sparse` optional
- Large FEM / RM / GREIT cache artifacts target HDF5 `.h5`; DOLFINx mesh-coupled cache uses XDMF + HDF5 sidecar; `.npz` allowed only legacy / small unit-test fixtures
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
- Perf: `detect_performance_capabilities`, `select_preconditioner`, `select_fast_linear_path`, `select_fused_strategy`, `probe_mpi_runtime`, `resolve_3d_cuda_forward_solver_policy`, `resolve_3d_cuda_mat_solve_policy`
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

- Forward `get_backend_diagnostics()` keys: `forward_rhs_count`, `forward_ksp_solve_count`, `forward_ksp_mat_solve_count`, `forward_ksp_setup_count`, `forward_ksp_setup_attempts`, `forward_reuse_preconditioner_requested`, `forward_reuse_preconditioner_applied`, `forward_pc_session_reused`, `forward_pc_refresh_triggered`, `forward_pc_refresh_reason`, `ksp_type`, `pc_type`, `pc_factor_mat_solver_type`, `petsc_mat_type`, `petsc_vec_type`, `petsc_dense_mat_type`, `petsc_solve_mat_type`, `forward_mat_solve_effective`, `forward_mat_solve_requested`, `forward_mat_solve_effective_policy`, `forward_mat_solve_policy_reason`, `forward_mat_solve_policy_warning`, `forward_ksp_iterations_per_rhs`, `forward_ksp_iterations_total`, `forward_ksp_converged_reason`, `forward_ksp_converged`, `forward_setup_seconds`, `forward_solve_seconds`, `forward_factor_cache_hit`, `gpu_fallback_reason`, `fallback_reason`, `forward_mat_solve_fallback_reason`, MPI size/rank/support fields
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
  - planned EIDORS parity: `build_eidors_3d_greit_model(config) -> GREITRM`; config covers `imgsz/xvec/yvec/zvec`, `distr`, `Nsim`, `target_size`, `target_plane`, `target_offset`, `noise_figure|weight`, `noise_covar`, `desired_solution_fn`, `training_mode="forward"|"linearized"`, `keep_model_components`
  - planned EIDORS parity artifact: HDF5 `.h5` stores `RM`, `PJt`, `M`, `noiselev`, `weight_chosen`, `vh`, `vi`, `xyzr`, `D`, `Y`, `rec_model`, cache signature; `.npz` is legacy/small-test only
- `pyeidors.inverse.matrix_free.dual_mesh`: `DualMeshJacobianOperator(fwd_model, coarse2fine)` exposing `Jv`, `JTr`, `normal_matvec`
- `pyeidors.inverse.block_system` (extended): `build_sigma_contact_movement_block_metadata(n_sigma, n_contact, n_electrodes_dofs)`; adds `H_sigma_e`, `H_z_e`, `H_e_e` couplings + `prior_movement` hook
- `pyeidors.data.difference.normalize_time_difference(v_t, v_ref, floor=...) -> dv_norm`
- `pyeidors.data.channels.bad_channel_mask`: apply mask to `J`, `residual`, `W`
- `pyeidors.perf.gpu_kernels.rm_matmul`: batched `RM @ ΔV` on GPU (torch / cupy)
- Nix experimental profile `.#cuda-amgx`: PETSc configured with CUDA + PCAMGX (`--download-amgx=<src>` or `--with-amgx-dir=<path>`, `--with-64-bit-indices=0`), then rebuilds `petsc4py`, SLEPc, DOLFINx, `fenics-dolfinx` against same PETSc; FEniCSx version upgrade alone is neither required nor sufficient

CLI additions (under `scripts/run_reconstruction_unified.py` or new scripts):

- `--algorithm one-step-gn|noser|laplace|greit-3d|matrix-free-gn-cg|tv-pdhg`
- `--dual-mesh on|off`, `--coarse-mesh <path>`
- `--rm-cache <path>`: load / save precomputed RM
- `--normalize-difference on|off`
- `--greit-targets <path>`, `--greit-metrics-out <path>`
- `--bad-channel-mask <csv|json>`
- `scripts/benchmarks/benchmark_mesh_io_formats.py --mesh <path.msh|path.xdmf> --repeats N --output-json <path>` — compare `.msh` import vs XDMF/HDF5 load, verify cells/vertices/facet-tags/cell-tags equal
- `scripts/cache/migrate_artifacts_to_hdf5.py --root <path> --dry-run|--apply` — migrate legacy `.npz/.npy` numeric artifacts to `.h5`, leave source read-only backup, emit manifest
- `pyeidors.io.hdf5_artifacts`: `write_hdf5_artifact(path, arrays, metadata, chunks, compression)`, `read_hdf5_artifact(path, lazy=True)`, `migrate_npz_to_hdf5(src, dst)`

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
| V10 | Fast PCG parity with dense GN solution within `rtol 1e-5` for PC modes `{diag, noser, prior, pmat, coarse, custom, cholmod}` (tested); `pyamg` parity smoke landed in T13; `petsc-gamg` without Pmat → fallback `diag` + reason `petsc_gamg_not_supported_in_matrix_free` | tests/unit/test_gn_fast_linear_solver.py |
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
| V42 | Forward `mat_solve_mode="off"` ! wins over CUDA/dense auto-routing (`forward_mat_solve_effective="vec-loop"`); `pc_type=="amgx"` / `solver_preset=="cuda_amgx"` with `petsc_amgx=false` fails during backend setup with explicit PCAMGX guidance | tests/unit/test_forward_mat_solve_policy.py |
| V43 | AmgX experiment gate: `nix develop .#cuda-amgx` ! yields CUDA Mat/Vec/Dense + `PETSc.PC.Type.AMGX`; `PETSc.PC().setType("amgx")` succeeds; `benchmark_3d_runtime.py --forward-only on --forward-solver-preset cuda_amgx --petsc-device cuda --forward-mat-solve off` completes and reports speed/residual vs `spd_gamg + petsc_device=cuda`. Plain FEniCSx/DOLFINx upgrade does not satisfy this gate | future `.#cuda-amgx` profile + scripts/diagnostics/probe_petsc_cuda.py |
| V44 | GUI/runtime 3D CUDA forward policy: if auto/`cuda_amgx` and `petsc_amgx=false` → `forward_solver_preset="spd_gamg"` + diag `petsc_amgx_available=false`; `spd_hypre`/`cg_hypre`/`3d_hypre`/`hypre_boomeramg` + CUDA blacklisted → warning/downgrade in high-level runtime, fail-fast in low-level `EITForwardModel`, never reaches PETSc solve | tests/unit/test_core_setup_contract.py, tests/unit/test_conductivity_3d_widget_runtime.py, tests/unit/test_forward_mat_solve_policy.py |
| V45 | GUI/runtime 3D CUDA production default: when effective forward solver is `spd_gamg` on PETSc CUDA/DOLFINx and requested `forward_mat_solve="auto"`, high-level runtime resolves `forward_mat_solve="off"` with reason `cuda_spd_gamg_matsolve_disabled_b6`; explicit `forward_mat_solve="on"` remains opt-in for experiments. This prevents unstable/slow PETSc `KSPMatSolve` on current 48-electrode 3D config | tests/unit/test_core_setup_contract.py, tests/unit/test_conductivity_3d_widget_runtime.py |
| V46 | Current 3D production route: online hot path ! cached `RM @ normalize(Δv)` on coarse inverse grid; no `DirectJacobianCalculator`, no forward/adjoint/KSP, no dense-J rebuild per frame after RM artifact exists | V25,V26,V29,V37,V45; reports/runtime_benchmarks/dual_model_rm_v1_20260421/summary.json |
| V47 | Current 3D forward default: PETSc sparse `spd_gamg + petsc_device=cuda + forward_mat_solve=off`; direct LU/MUMPS allowed for 2D, tiny 3D, debug/reference only; Hypre CUDA blacklisted; AmgX research gated by V43 | V6,V42,V43,V44,V45; reports/benchmarks/forward_spd_gamg_cuda_48e_repeat2_20260421.json |
| V48 | Dense-J GN status: implemented and valid as baseline/RM-build/reference; 48e/5936 cold path dominated by Jacobian build, not realtime default. Warm semantic cache may be fast, but cache hit cannot be required for first-frame usability | src/pyeidors/core_system.py:716; src/pyeidors/inverse/jacobian/direct_jacobian.py; reports/runtime_benchmarks/lazy_48e_spd_gamg_cuda_b4_20260421/summary.json |
| V49 | Inverse sparse/matrix-free status: `JacobianLinearization`, lazy adjoint, dual-mesh operator, PETSc/Scipy matrix-free PCG, sparse priors, TV, SBL exist as phase-2/research. They must not replace cached RM as v1 realtime default until end-to-end 48e/5936 benchmark beats RM hot path | src/pyeidors/inverse/jacobian/linearized.py; src/pyeidors/inverse/matrix_free/dual_mesh.py; src/pyeidors/inverse/solvers/gauss_newton_runtime.py; tests/unit/test_gn_fast_linear_solver.py |
| V50 | 3D GREIT current status: linearized RM v0, optimized for 48e/5936 online apply, NOT EIDORS-complete. `training_responses = T @ J.T`, `D≈T`, no finite-target `vh/vi`, no `desired_solution_fn`, no NF weight search, no official MATLAB parity gate, no HDF5 large-cache artifact yet. Production "EIDORS同款" claim ! forbidden until V55..V65 + T40..T50 | src/pyeidors/inverse/greit.py; tests/unit/test_greit_rm.py; reports/runtime_benchmarks/dual_model_rm_48e_5936_t36_20260422/summary.json; B11 |
| V51 | WSLg GUI launch default keeps the main PySide6 surface on Wayland-first Qt (`QT_QPA_PLATFORM=wayland;xcb`) when `WAYLAND_DISPLAY` exists, preserving crisp HiDPI text; XCB is explicit opt-in (`EIT_APP_USE_QT_XCB=1`) or no-Wayland fallback. GUI launch defaults to cached env sync + skipped PETSc CUDA probe; lock/config changes invalidate cache and `--probe-cuda` / `--full-env-check` restore full verification. First paint must not synchronously load the heavy pyeidors/PETSc/Torch/CUQI runtime or block on Windows COM discovery; those run on demand | tests/unit/test_eit_app_bootstrap.py; tests/unit/test_env_sync_script.py; tests/unit/test_script_entrypoint_acceleration_profiles.py; tests/unit/test_eit_app_gui_smoke.py |
| V52 | Benchmark / launch timing harness ! immune to PATH-shadowed coreutils: timing commands use `/usr/bin/env` explicitly OR first assert `command -v env == /usr/bin/env`; artifact records `env_path`. `/home/tom/.local/bin/env` or any user shim must not make benchmark/GUI launch appear as `real 0.00` without executing payload | future benchmark-env guard; B8,T37 |
| V53 | GUI `single_step_cached` difference path ! only call `fwd_solve` with finite `sigma_est = sigma_bg + alpha * delta_sigma` and `min(sigma_est) > sigma_floor`; step-size calibration candidates with nonfinite or floor-violating `sigma_try` return `inf` objective; fallback alpha ! preserve feasible sigma. `cuda_structured` invalid-diagonal guard stays fail-fast | future tests/unit/test_conductivity_3d_widget_runtime.py; B9,T38 |
| V54 | GUI 3D PyVista offscreen drag frames may lower render framebuffer resolution for responsiveness, but the QLabel pixmap logical size MUST stay constant across drag and idle frames; rotating/zooming 3D truth or reconstruction views must not visibly shrink/grow/flicker | tests/unit/test_conductivity_3d_widget_runtime.py |
| V55 | EIDORS-parity 3D GREIT target distribution ! match `GREIT3D_distribution`: voxel volume rec model via `imgsz` or explicit `xvec/yvec/zvec`, one target per valid voxel by default, `downsample` support, point-in-volume mask, 3D `xyz` centers inside rec volume. 2D `xg/yg` raster path ⊥ for parity mode | future `src/pyeidors/inverse/greit.py`; EIDORS `GREIT3D_distribution` |
| V56 | EIDORS-parity training response mode ! use finite target perturbation forward solves: homogeneous `vh` + inhomogeneous `vi` for each `xyzr` target on fine CEM model. Linearized shortcut `Y = T @ J.T` allowed only as `training_mode="linearized"` and artifact labels `eidors_parity=false` | future `src/pyeidors/inverse/greit.py`; EIDORS `mk_GREIT_model` `stim_targets` |
| V57 | Difference normalization parity: if `normalize=1`, `Y = vi ./ vh - 1`; else `Y = vi - vh`. Same mode used for offline RM build and online `GREITRM.reconstruct`; channel ordering and bad-channel deletion/zeroing deterministic | future tests/unit/test_greit_eidors_parity.py; EIDORS `calc_GREIT_RM` |
| V58 | Desired image parity: `D = desired_solution_fn(xyz, radius, options)`; default approximates EIDORS `GREIT_desired_img` over 3D rec model. `D` shape = `n_rec_parameters × n_targets`; `D≈T` only explicit opt-in, never default parity mode | future `src/pyeidors/inverse/greit.py`; EIDORS `calc_GREIT_RM` |
| V59 | RM formula parity: `PJt = D @ Y.T`; `noiselev = weight * mean(abs(Y))`; `Sn = I * noise_covar` or provided covariance; `M = Y @ Y.T + noiselev² * Sn`; `RM = solve(M.T, PJt.T).T`; use transpose, not conjugate transpose; singular fallback emits diagnostic | future tests/unit/test_greit_calc_rm_parity.py; EIDORS `calc_GREIT_RM` |
| V60 | NF / image-SNR weight selection parity: if `noise_figure` or `image_SNR` set and scalar `weight` absent, bounded log10 search chooses scalar weight; objective uses simulated NF/SNR target measurements; chosen weight, achieved metric, tolerance, search bracket stored in artifact | future tests/unit/test_greit_noise_figure.py; EIDORS `mk_GREIT_model` |
| V61 | `keep_model_components` parity: HDF5 artifact stores `RM`, `PJt`, `M`, `noiselev`, `weight`, `vh`, `vi`, `xyzr`, `D`, `Y`, `rec_model`, `fwd_model_signature`; expensive `PJt` cache independent of weight so NF search does not recompute desired-image product | future artifact schema `pyeidors-greit-eidors-hdf5-v1`; EIDORS `mk_GREIT_model` |
| V62 | GREIT RM cache signature ! include V55..V61 math inputs: target distribution grid/downsample, finite target contrast/size/radius/plane/offset, `desired_solution_fn` identity + params, `normalize`, `noise_covar`, scalar weight or target NF/SNR, training mode, forward solver signature. Device/dtype affect storage only | future tests/unit/test_greit_cache_signature.py; V36 |
| V63 | MATLAB EIDORS parity gate: same bridge geometry/protocol/background/targets in MATLAB EIDORS and PyEIDORS; compare `Y`, `D`, `PJt`, `M`, `noiselev`, `RM @ dv`, GREIT metrics. Hard tolerances must be recorded per fixture; unknown tolerance marked `?` until first official fixture | future scripts/diagnostics/compare_greit_eidors_parity.py |
| V64 | After EIDORS-parity RM artifact exists, online path remains one matmul: load HDF5 `RM` + metadata, normalize/weight `dv`, apply `RM @ dv`; no forward solve/Jacobian/KSP per frame. Common hardware configs may be precomputed offline; warm use = `.h5` load + optional GPU handle | future GUI/CLI warmup tests; V37,V40 |
| V65 | Large cache format: DOLFINx/FEniCSx mesh-coupled artifacts use XDMF + HDF5; GREIT/RM/training-response caches use HDF5 `.h5` with chunking/compression/checksum metadata. `.npz` must not be default for large 3D caches; legacy `.npz` loaders remain read-only compatibility | DOLFINx `XDMFFile` HDF5 default; src/pyeidors/geometry/dolfinx_mesh_cache.py; future `pyeidors.cache.hdf5_artifacts` |
| V66 | Mesh first-load policy: if XDMF/HDF5 cache exists, `MeshLoader` loads it first even when source `.msh` missing; metadata/freshness may use explicit mesh content hash or cache manifest, not mandatory `.msh` mtime. If only `.msh` exists, import once → write XDMF/HDF5 → subsequent loads never parse `.msh`. Mesh geometry, cell tags, facet tags, physical group association, structured sidecar fields must round-trip | src/pyeidors/geometry/mesh_loader.py; src/pyeidors/geometry/dolfinx_mesh_cache.py; local benchmark 2026-04-22 |
| V67 | Project binary array persistence default = HDF5 `.h5`: RM/GREIT artifacts, dataset generator outputs, GUI simulation exports, diagnostics bundles, benchmark arrays, MATLAB mesh bridge arrays, reconstruction outputs. `.npz/.npy` creation forbidden outside tests/legacy adapters unless task explicitly marks small fixture. JSON/CSV stay for metadata/tables only | future scan gate; B12 |
| V68 | Mesh IO benchmark gate: for representative 2D + 3D meshes, HDF5 load median ≤ `.msh` import median or regression explained; equality checks: vertices/cells count, topology dim, geometry dim, facet tags, cell tags, association table. Local 2026-04-22 48e 3D samples: `.msh 0.0747s` vs HDF5 `0.0429s`; `.msh 0.0314s` vs HDF5 `0.0276s` | future `scripts/benchmarks/benchmark_mesh_io_formats.py`; B12 |
| V69 | GUI 3D PyVista offscreen interaction defaults to high-performance machines: drag/zoom timer targets 60 fps and full-DPR framebuffer (`EIT_APP_3D_DRAG_RENDER_SCALE=1.0` effective); repeated frames at the same size must not force a PyVista/VTK window resize. Users may explicitly lower `EIT_APP_3D_DRAG_FPS` or `EIT_APP_3D_DRAG_RENDER_SCALE`; logical canvas size remains stable per V54 | tests/unit/test_conductivity_3d_widget_runtime.py; B13 |

## §T — tasks

### §T.phase — v1 queue and tiering

Current priority queue. Cavekit `/ck:make` MUST advance along the v1 queue before touching phase-2 / research tiers.

```
v1  (EIDORS-style dual-model offline-RM + online RM@normalize(Δv)):
    T15  →  T18  →  T16  →  T17  →  T26  →  T29  →  T31  →  T19  →  T20  →  T32

hotfix  (regression / safety):
    T38, T39, T58

greit-parity  (EIDORS-complete 3D GREIT; unlock "EIDORS同款" claim):
    T40  →  T41  →  T42  →  T43  →  T44  →  T45  →  T46  →  T50  →  T47  →  T48  →  T49

hdf5-unification  (project-wide binary cache/save format):
    T51  →  T52  →  T53  →  T54  →  T55  →  T56  →  T57

regularization-foundation  (official baseline + extensible prior core):
    T59  →  T60  →  T61  →  T62

dynamic-foundation  (neural / plant continuous EIT):
    T64  →  T63  →  T69  →  T65

dynamic-quality  (after dynamic foundation metrics stable):
    T66  →  T67

phase-2  (after v1 stable, higher-fidelity reconstruction):
    T22, T23, T25, T24, T21, T30

research  (post-phase-2, opt-in enhancement):
    T27, T70, T68, T28, T1, T11, T33

infra / deferred  (hardware / design-heavy, unblock separately):
    T2, T3, T4, T6, T7, T10, T37
```

T32 is the v1 closure milestone and ties in the GREIT metrics, so it
MUST follow T19/T20. Earlier queue drafts put T32 before T19/T20 — do
not reintroduce that ordering; the milestone cannot validate
`{AR, PE, RES, SD, RNG}` before those tasks land.

v1 graduation gate: all rows T15..T20, T26, T29, T31, T32 must be `x` AND V36..V41 must hold before T22+ are eligible.

3D GREIT official-equivalence gate: all rows T40..T50 must be `x` AND V55..V65 must hold before UI/docs/papers may say "EIDORS同款", "official-equivalent", "perfect stable 3D GREIT".

HDF5 unification gate: all rows T51..T57 must be `x` AND V65..V68 must hold before new binary cache/save code may land with `.npz/.npy` default.

Regularization foundation gate: T59..T62 must be `x` before new inverse solvers claim interchangeable `RtR/R_prior` support across RM, GN runtime, matrix-free, cache signatures.

Dynamic foundation gate: T63..T65 + T69 must be `x` before neural / plant continuous-EIT docs claim 4D prior, propagation-speed preservation, or realtime dynamic reconstruction.

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
| T21 | x | Temporal smoothing + TV postprocess pipeline on RM output (moving-average, exponential decay, 3D TV regulariser on voxel grid) | V28 |
| T22 | x | Matrix-free `Jv` / `JTr` extended over dual mesh (coarse inverse parameter → fine forward) with dense parity test on small mesh | V25,V31 |
| T23 | x | IRGNM / LM wrapper around matrix-free Hv for absolute-ish 3D reconstruction, reusing the existing `_solve_pcg` PETSc or SciPy backend | V12,V31,V35 |
| T24 | x | TV PDHG / PDIPM refinement on ROI after one-step init; seeded by RM output, stops on ROI-restricted residual norm | V26,V28 |
| T25 | x | Electrode-movement Jacobian + `prior_movement`; extends block metadata with `e` block and `H_σe`, `H_ze`, `H_ee` couplings | V20,V32 |
| T26 | x | Bad-channel mask + noise covariance `W` wired into Jacobian rows, residual vector, measurement-weight contract, and RM builder so offline / online weights match | V34,V35 |
| T27 | . | SBL / BSBL / SA-SBL coarse-basis research enhancement (RBF, sparse-inclusion, low-rank anatomical basis) — tier 3, post-v1; no default use until T70 benchmark wins | V31 |
| T28 | . | CNN / U-Net postprocess plug-in interface (coarse 3D image in → enhanced image out), no physics replacement | V29 |
| T29 | x | GPU `RM @ ΔV` online kernel: batched multi-frame matmul on GPU, reuses normalized-difference path | V29,V35 |
| T30 | x | Hypre `BoomerAMG` / NVIDIA AmgX CUDA path wiring for forward CG (extends T10), with capability-probe entries in `forward_solver_benchmark` artifact | V6,V13,T10 |
| T31 | x | Dual-mesh integration smoke: fine CEM + coarse recon + EIDORS-style parity metric on synthetic sphere target | V25,V29,V30 |
| T32 | x | Milestone **FEniCSx-EIT-3D-v1**: ties V25–V35 + GPU online matmul; 10-point checklist (fine CEM, coarse voxel, c2f, reusable KSP/PC, adjoint J on coarse, one-step GN/NOSER/Laplace RM, normalized Δv, GPU RM@Δv, GREIT metrics, bad-channel / W weighting) | V25,V26,V27,V28,V29,V30,V31,V33,V34,V35 |
| T33 | . | Research: `cuda-amgx` Nix profile. Rebuild PETSc with PCAMGX (CUDA + 32-bit `PetscInt` + AmgX external package), then rebuild same-chain `petsc4py`/SLEPc/DOLFINx/`fenics-dolfinx`; benchmark against current safe route `spd_gamg + petsc_device=cuda`. Do not treat FEniCSx version upgrade as AmgX enablement | V19,V42,V43,B5 |
| T34 | x | Switch GUI/default 3D difference route to cached dual-model RM reconstruction when RM artifact exists; cold path may build/load RM, hot path must bypass `DirectJacobianCalculator` | V37,V46,V48 |
| T35 | x | Optimize RM hot path for 48e/5936: persistent RM on device, batched frames, no per-call tensor rebuild, minimal CPU↔GPU copy, float64/float32 policy recorded | V29,V37,V50 |
| T36 | x | Real 48e/5936 dual-model report: compare one-step NOSER/Laplace RM and 3D GREIT RM, split fine-CEM/J build, RM build, artifact load, 1-frame apply, 512-frame apply, GPU/CPU paths | V40,V46,V47,V50 |
| T37 | x | Harden benchmark / GUI timing harness against PATH-shadowed `env`: prefer `/usr/bin/env`, add guard test, record `env_path` in timing artifacts | V52,B8 |
| T38 | x | Guard GUI `single_step_cached`: feasible-step alpha bound by `sigma_floor`, illegal `sigma_try` → `inf`, final `sigma_est` finite & floored before `fwd_solve`; add CUDA/DOLFINx parity regression | V53,B9 |
| T39 | x | Stabilize GUI 3D PyVista offscreen drag display: low-res drag frame is scaled to the same QLabel physical target before DPR assignment, so visual canvas size is invariant | V54,B10 |
| T40 | x | Build EIDORS GREIT source map + golden fixture capture: MATLAB script exports `vh`, `vi`, `xyzr`, `D`, `Y`, `PJt`, `M`, `noiselev`, `RM`, `weight`; include tiny 3D cylinder + 48e/5936 reduced case | V50,V55,V63,B11 |
| T41 | x | Implement `GREIT3D_distribution` parity builder: `imgsz/xvec/yvec/zvec`, `downsample`, point-in-volume, target centers, volume/inside mask, deterministic order | V55 |
| T42 | . | Implement finite-target training response engine: homogeneous `vh`, per-target `vi`, target radius/size/plane/offset, contrast, batching/cache; keep linearized shortcut explicit non-parity mode | V56,V57 |
| T43 | . | Implement desired image stack: default EIDORS-like `GREIT_desired_img` for 3D rec model + custom `desired_solution_fn`; output `D` independent from raw target `T` | V58 |
| T44 | . | Rework `calc_GREIT_RM` parity core: `PJt`, `noiselev` scaling, `Sn`, `M`, transpose solve, diagnostics, singular fallback; unit compare against exported EIDORS components | V57,V58,V59 |
| T45 | . | Implement NF/image-SNR scalar-weight optimizer: target simulation, bounded log10 search, achieved metric/tolerance metadata, failure diagnostics | V60 |
| T46 | . | Add EIDORS-parity GREIT HDF5 artifact/cache schema: store model components in `.h5`, cache `PJt` across weight search, signature includes V55..V61 inputs | V61,V62,V65 |
| T47 | . | Add MATLAB EIDORS parity diagnostics + tests: compare PyEIDORS vs official EIDORS `Y/D/PJt/M/RM/recon/metrics`; record tolerances and drift report | V63 |
| T48 | . | Add common-config offline warmup CLI/GUI path: precompute/load 16/32/48e 3D GREIT `.h5` artifacts; online load+matmul only; no routine cold build for known hardware | V64,V65 |
| T49 | . | Run final 48e/5936 EIDORS-parity 3D GREIT benchmark: cold build, HDF5 artifact load, 1-frame/512-frame online apply, metrics, bad-channel/W cases, GPU/CPU stability | V55,V56,V57,V58,V59,V60,V61,V62,V63,V64,V65 |
| T50 | . | Implement large-cache HDF5 IO layer: chunked/compressed datasets for `RM/Y/D/PJt/M/vh/vi/xyzr`, JSON metadata attrs, checksum, lazy dataset reads, legacy `.npz` read-only import/migration path | V61,V62,V65 |
| T51 | x | Repo persistence inventory + blocklist: classify all `.npz/.npy/.msh/.xdmf/.h5/.mat` writers/readers; mark legacy/test-only exemptions; add CI scan forbidding new production `.npz/.npy` writes | V65,V67,B12 |
| T52 | x | Mesh HDF5-first hardening: support `.xdmf/.h5` cache load without source `.msh`; store source hash/provenance optional; generator writes XDMF/HDF5 from in-memory mesh even when `save_msh=false`; round-trip facet/cell tags + physical groups | V66,V68 |
| T53 | x | Convert RM/GREIT/one-step artifacts from `.npz` to HDF5 `.h5`; keep `.npz` loader read-only + migration helper; update GUI cached-RM loader | V36,V37,V61,V64,V65,V67 |
| T54 | x | Convert dataset generator + GUI simulation export from `mesh_info.npz` / `sample_*.npz` / "NumPy archive" to HDF5 package; update i18n labels and file dialogs | V65,V67 |
| T55 | x | Convert diagnostics/benchmark/reconstruction output bundles (`outputs.npz`, `result_arrays.npz`, `inverse_3d_overview_data.npz`, gallery bundles) to HDF5; JSON/CSV summaries remain | V40,V65,V67 |
| T56 | x | Convert MATLAB/interop mesh bridge arrays from `.npz` default to HDF5/v7.3-compatible `.h5`; retain `.mat`/legacy `.npz` import adapters only | V63,V65,V67 |
| T57 | x | Add mesh IO format benchmark + regression test: compare `.msh` import vs XDMF/HDF5 load on representative meshes; store JSON artifact with speed ratio and tag equality | V66,V68 |
| T58 | x | Raise GUI 3D offscreen drag defaults to 60 fps + full-DPR framebuffer; skip redundant offscreen window resizes; keep env-controlled lower-fps/downsample mode for constrained machines without reintroducing V54 size jitter | V69,B13 |
| T59 | x | Baseline alignment freeze: official-style framewise GN/NOSER/Laplace checklist + tiny fixtures; verify `Jᵀ W J + hp² RtR` formula, `hp²` scaling, normalized/raw difference parity, artifact metadata names | V26,V27,V28,V38,V40 |
| T60 | . | General `RtR/R_prior` prior contract: dense/sparse/`LinearOperator`/callable inputs, `apply(v)`, `diag()`, `as_RtR()`, signature hash, metadata, HDF5 persistence; wire through RM builder + GN runtime without forced dense materialization on large 3D | V26,V28,V35,V36,V49 |
| T61 | . | Curvature prior mode: expose `L = graph_difference_operator(mesh)`, `RtR = L.T @ L` as named `curvature` / `graph_ltl` prior; compare vs Laplace smoothing on same mesh; cache signature distinguishes `laplace` vs `LᵀL` | V28,V35,V36,V49 |
| T62 | . | TV-IRLS inverse prior: iterative `RtR(x)=L.T @ diag(1/sqrt((Lx)^2+β)) @ L`; β/floor finite guard, max outer iterations, stale-RM invalidation, monotone objective smoke; keep TV-PDHG postprocess as separate seeded refinement | V28,V35,V49 |
| T63 | . | Measurement-domain temporal filtering before reconstruction: causal EMA/moving-average + optional bandpass/lock-in hook over `Δv`/raw frames; channel mask + `W` applied deterministically; filter state stored in metadata, no smoothing of timestamps | V33,V34,V35,V37,V40 |
| T64 | . | Dynamic sequence data contract: frames carry `t`, `dt`, sampling rate, frame id, reference policy, stim/meas signature, bad-channel mask, `W`, frequency/context metadata; HDF5 package round-trips multi-frame arrays + metadata; `MeasurementDataset` remains single-frame-compatible | V33,V34,V35,V37,V65,V67 |
| T65 | . | Batch spatiotemporal GN / 4D prior: windowed solve over `X[t,param]` with spatial prior `Rs`, temporal `Dt` first/second difference, block normal operator, λ_s/λ_t metadata, rowwise RM baseline comparison | V25,V31,V35,V49 |
| T66 | . | Spatiotemporal TV / Huber prior: separable spatial graph + temporal difference penalties; preserves abrupt wavefront/onset better than L2 time smoothing; ROI support; compare against T65 on travelling-wave fixture | V28,V31,V35,V49 |
| T67 | . | Online Kalman + fixed-lag smoother prototype: state model `x_t=A x_{t-1}+q`, measurement `y_t=Jx_t+n` or RM-observation shortcut; latency/lag metadata, `Q/R` estimation hooks, no default until T69 metrics pass | V35,V37,V49 |
| T68 | . | Propagation-aware prior research: directed anatomical/vascular graph, velocity-range prior, path-constrained smoothness for nerve / plant conduction; opt-in only, never required for baseline parity | V31,V49 |
| T69 | . | Dynamic validation benchmark: synthetic travelling wave + plant slow-pulse fixtures; report onset-time error, peak-time error, propagation-speed error, amplitude attenuation, SNR gain, spatial metrics; fail if regularization delays peak beyond tolerance ? | V30,V40,V41,V49 |
| T70 | . | SBL/BSBL acceptance benchmark: compare SBL/BSBL/SA-SBL vs GN/NOSER/Laplace/TV-IRLS/4D prior on sparse anomaly + frequency-difference + propagating sparse events; promote only if accuracy/latency win recorded | V31,V35,V49,T27 |

## §B — bugs

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | 2026-04-20 | `ForwardKSPSession` applied `setReusePreconditioner(True)` uniformly; for `ksp_type==preonly` + `pc_type∈{lu,cholesky,qr}` this silently reuses stale LU/Cholesky/QR factorisation across sigma updates, solving `A(σ_new) x = b` with `A(σ_old)^{-1}`. No Krylov iteration to correct the error (unlike iterative+AMG where reuse is a staleness penalty). | V24,T14 |
| B2 | 2026-04-21 | CUDA/dense `matSolve` auto override ignored explicit `mat_solve_mode="off"`; CUDA profile had `petsc_amgx=false` / PCAMGX absent, but `cuda_amgx` proceeded past setup toward solve | V42 |
| B3 | 2026-04-21 | `/check §V` found V2 drift: CUDA auto `matSolve` branch (`effective_device=="cuda"` & `petsc_cuda_dense`) bypasses `performance_mode=="aggressive"` and `forward_mat_solve_min_patterns`; V2 says exact iff formula | V2 |
| B4 | 2026-04-21 | `spd_hypre + petsc_device=cuda` forward benchmark SIGSEGV in PETSc/Hypre CUDA, even with `--forward-mat-solve off`; current measured safe CUDA forward route = `spd_gamg + petsc_device=cuda` | V44 |
| B5 | 2026-04-21 | AmgX cannot enable in current CUDA shell because PETSc has CUDA Mat/Vec/Dense (`aijcusparse`, `cuda`, `densecuda`) but PCAMGX is not compiled/registered: `PETSc.PC.Type.AMGX` is absent and `PCSetType("amgx")` returns error code 86 `Unable to find requested PC type amgx`; `flake.nix` CUDA PETSc override only adds CUDA/cuBLAS/cuSPARSE/cuSOLVER flags, with no AmgX package or `--with-amgx` / `--download-amgx` configure path; FEniCSx/DOLFINx latest release does not ship AmgX by itself | T33 |
| B6 | 2026-04-21 | GUI 3D CUDA policy downgraded unavailable AmgX/Hypre CUDA to `spd_gamg` but left `forward_mat_solve=auto`; PETSc `KSPMatSolve` on current `spd_gamg + cuda` 48-electrode/5936-measurement config failed after a long solve attempt with negative convergence reason `-10`. Stable measured route is `spd_gamg + petsc_device=cuda + forward_mat_solve=off` | V45 |
| B7 | 2026-04-22 | WSLg GUI bootstrap pinned Qt to XCB to protect embedded VTK, so Windows HiDPI scaled the whole app through XWayland blur; launcher also repeated full uv/import sync and PETSc CUDA probe on every GUI start, then imported heavy pyeidors/PETSc/Torch/CUQI paths and synchronously queried Windows COM ports before first paint | V51 |
| B8 | 2026-04-22 | User PATH shadows coreutils `env`: `/home/tom/.local/bin/env` intercepted `env EIT_APP_AUTO_QUIT_MS=5000 bash scripts/gui/run_eit_app.sh --gpu`, returned success without running payload, and produced bogus `real 0.00` GUI timing. `/usr/bin/env ...` executed correctly | V52,T37 |
| B9 | 2026-04-22 | GUI 3D `single_step_cached` accepted unconstrained `sigma_bg + alpha * delta_sigma`; calibration candidate failure swallowed then `alpha=1.0`; `cuda_structured` correctly rejected nonphysical FEM top-left diagonal (`NaN/Inf` or `<=0`) during forward validation | V53,T38 |
| B10 | 2026-04-22 | GUI 3D PyVista offscreen drag path rendered at 60% physical size for responsiveness but set the pixmap DPR to the widget DPR; Qt therefore displayed drag frames as smaller logical images, then snapped back on the idle full-resolution frame | V54,T39 |
| B11 | 2026-04-22 | SPEC/bench text overclaimed 3D GREIT as official-aligned; code is linearized GREIT-RM v0 (`Y=T@J.T`, `D≈T`) and lacks EIDORS finite-target `vh/vi`, `desired_solution_fn`, NF weight search, HDF5 model-component parity artifacts | V50,V55,V56,V57,V58,V59,V60,V61,V62,V63,V64,V65,T40,T41,T42,T43,T44,T45,T46,T47,T48,T49,T50 |
| B12 | 2026-04-22 | Project persistence still mixed: mesh cache already prefers XDMF/HDF5 but freshness tied to source `.msh`; many production writers still emit `.npz/.npy` (`greit_rm.npz`, `one_step_*_rm.npz`, `outputs.npz`, `result_arrays.npz`, dataset `mesh_info.npz`/`sample_*.npz`, GUI "NumPy archive"). This conflicts with FEniCSx-aligned HDF5-unified cache/save target | V65,V66,V67,V68,T51,T52,T53,T54,T55,T56,T57 |
| B13 | 2026-04-22 | After fixing 3D offscreen drag size jitter, the interaction path still defaulted to ~30 fps and 0.6× drag framebuffer scale; on high-resource GPU/WSLg machines that made rotation feel visibly choppy and low-fidelity | V69,T58 |
