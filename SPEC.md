# SPEC — PyEIDORS

Source: distilled from code at `acc4281` on `dev/gui-integration`. Flag `?` = inferred, user confirm.

## §R — registry

`SPEC.md` is the active entrypoint; `docs/spec/registry.json` is the machine-readable authority list. All split files obey the same ID, reference, and status gate.

| registry surface | authority |
|------------------|-----------|
| Machine manifest | `docs/spec/registry.json` |
| Exact ID→file lookup | `docs/spec/id-map.tsv` |
| Invariant domains | `docs/spec/invariants/*.md` |
| Completed tasks / resolved bugs | `docs/spec/history/*.md` |
| Integrity gate | `scripts/ci/spec_integrity_guard.py` |

## §G — goal

Python-first EIT framework. FEniCSx (DOLFINx) CEM forward + PyTorch-accel inverse + PETSc KSP/PC. EIDORS parity target; hard tolerance pending. Modern GPU/MPI path. Absolute + difference reconstruction, real-time mesh, benchmark.

**v1 main line (current focus):** EIDORS-style dual-model 3D difference EIT — fine-CEM forward mesh, coarse inverse voxel/tetra mesh, offline one-step GN / NOSER / Laplace / 3D GREIT reconstruction matrix (`RM`), online `x = RM @ normalize(Δv)`. Matrix-free GN-CG / IRGNM / TV / SBL / CNN post-processing are phase-2 / research tiers, not v1 blockers.

**Current 3D repo decision:** production/default path = `spd_gamg + petsc_device=cuda + forward_mat_solve=off` forward, cached dual-model one-step RM/NOSER/Laplace first, 3D GREIT next after RM hot-path optimization. Dense-J GN, generic direct LU, CUDA `KSPMatSolve`, matrix-free GN-CG, TV, SBL are baseline / debug / phase-2 / research, not realtime 48e/5936 default. MUMPS preset retired: failed current 3D complex CEM consistency gate vs CPU SciPy direct / CUDA dense LU.

**3D route tiers (final):**
- v1 production: dual-model + one-step GN/NOSER/Laplace RM; online `RM @ normalized ΔV`.
- v1 enhancement: 3D GREIT RM; best for fixed geometry/protocol + multi-frame dynamic EIT.
- forward production default: 3D DOLFINx CEM + PETSc `spd_gamg` CUDA + vec-loop; reuse multi-RHS/KSP/PC/cache.
- phase-2 quality: matrix-free GN-CG/IRGNM + dual mesh + explicit Pmat/NOSER/prior preconditioner.
- postprocess/robustness: TV/PDHG, bad-channel mask, noise covariance, temporal prior.
- research: SBL/sparse Bayesian, self-built PETSc AmgX, full-chain GPU Jacobian.
- not mainline: full 3D dense-J absolute GN, full 3D direct LU/Cholesky, per-frame forward+Jacobian+GN.

**3D GREIT parity note:** T40..T50 code path complete: `GREIT3D_distribution` + finite-target `vh/vi` + `desired_solution_fn`/`D` + `calc_GREIT_RM` + NF search + HDF5 `keep_model_components` + parity diagnostics + common-config warmup + T49 surrogate gate + MATLAB/EIDORS 48e official fixture gate. Linearized GREIT-RM v0 (`Y = T @ J.T`, `D ≈ T`) remains explicit non-parity mode. 48e official fixture parity claim allowed only for `reduced_48e_5936` fixture scope: actual EIDORS adjacent/no_meas_current data length `2160`, computed-from-fixture parity ≤ `1e-8`. 5936 measurement protocol official fixture remains separate; UI/docs must not claim "48e/5936 official-equivalent / perfect stable 3D GREIT" until separate 5936 protocol fixture gate passes.

## §C — constraints

- Python 3.13.x, `>=3.13,<3.14`
- DOLFINx / FEniCSx: nix devShell owns runtime (`nix develop`)
- PETSc via `petsc4py`; MPI size == 1 enforced ? until distributed Mat/Vec lands
- PyTorch ~2.10, optional CUDA via `nix develop .#cuda`
- AmgX ? explicit `cuda-amgx` profile: no FEniCSx version bump required; requires Nix rebuild of PETSc with PCAMGX + same-chain `petsc4py`/SLEPc/DOLFINx/`fenics-dolfinx`
- NumPy/SciPy/pandas/h5py/pyyaml mandatory; `pyamg`/`scikit-sparse` optional
- Large FEM / RM / GREIT cache artifacts target HDF5 `.h5`; DOLFINx mesh-coupled cache uses XDMF + HDF5 sidecar; `.npz` allowed only legacy / small unit-test fixtures
- Nix + uv dev path, WSL2 supported; GUI launcher `scripts/gui/run_eit_app.sh`
- PySide6 GUI under `src/eit_app/`
- MIT license
- Frozen queue guard: T2/T3/T7 ⊥ build until explicit env unfreeze + passing CUDA/MPI probes; T11/T27/T28/T68/T70 ⊥ build until explicit research unfreeze + baseline-gain/new-API evidence

## §I — interfaces

### Python API (public package surfaces)

- Top-level `pyeidors`: façade exports `EITSystem`, `check_environment`, `__version__` only; broader solver/forward re-export is not current contract
- `pyeidors.forward`: `EITForwardModel(n_elec, pattern_config, z, mesh, linear_backend="petsc", backend_config, forward_backend="dolfinx", cache_manager, performance_mode="aggressive")` — CEM forward solve, multi-RHS
- `pyeidors.forward.RobinTransconductanceForwardModel` → same `forward_solve`/`fwd_solve` shape as `EITForwardModel`; `EITSystem(..., cem_formulation="classic"|"robin_transconductance")`, default=`classic`
- `pyeidors.forward.LinearBackendConfig` fields: `solver_preset`, `ksp_type`, `pc_type`, `rtol`, `atol`, `max_it`, `reuse_preconditioner`, `monitor`, `mat_solve_mode`, `use_mat_solve`, `petsc_device`, `pc_factor_mat_solver_type`, `pc_hypre_type`, `pc_gamg_type`, `petsc_options`, `forward_pc_refresh_policy`, `forward_pc_refresh_iter_threshold`, `forward_pc_refresh_lag`, `forward_mat_solve_min_patterns`
- `pyeidors.inverse.jacobian.DirectJacobianCalculator` — `calculate(sigma, method="efficient"|"traditional")`, `linearize(sigma, method="efficient")`, `block_tuning_info()`
- `pyeidors.inverse.jacobian.JacobianLinearization` — `matvec`, `rmatvec`, `normal_matvec`, `as_linear_operator`, `to_dense`, `hessian_diag(measurement_weights, alpha, regularization_diag, floor)`, `as_petsc_mat`, `assert_compatible(sigma_fingerprint)`; fields `grad_u_all`, `adjoint_gradients`, `cell_areas`, `n_meas_per_stim`, `sign`, `sigma_fingerprint`
- `pyeidors.inverse.jacobian.compute_sigma_fingerprint(sigma) -> str`
- `pyeidors.inverse.GaussNewtonReconstructor` + `pyeidors.inverse.solvers.gauss_newton_runtime.run_reconstruction(reconstructor, measured_data, jacobian_method="efficient"|"linearized"|"operator"|"matrix-free")`
- `pyeidors.inverse.block_system`: `ParameterBlock`, `BlockCoupling`, `JointInverseBlockMetadata`, `SigmaContactNormalSystem`, `JointFieldSplitSolveResult`, `build_sigma_contact_block_metadata(n_sigma, n_contact, n_measurements, n_movement=None)`, `assemble_sigma_contact_normal_system(j_sigma, j_contact, residual, ...)`, `configure_petsc_fieldsplit_solver(ksp, metadata, ...)`, `solve_sigma_contact_fieldsplit(j_sigma, j_contact, residual, backend="auto"|"petsc"|"scipy", fieldsplit_type="additive"|"multiplicative"|"schur")`, `build_electrode_movement_jacobian`, `prior_movement`, `make_block_diagonal_inverse_action`, `scale_contact_impedance_update(current_z, delta_z, max_relative_step, floor)`
- Data: `PatternConfig`, `EITMesh`, `EITData`, `EITImage`; `pyeidors.data.add_noise(snr, v1, v2=None, options=None, seed=None, rng=None)` — EIDORS `add_noise` parity, array or `EITData`
- `pyeidors.interop`: Bridge/Geometry v3-only contract (`eidors_pyeidors_bridge_v3`, `eidors_pyeidors_geometry_v3`); `BridgeV3Package`, `ElectrodeSpec`, `ProtocolSpec`, `RegisteredModel`, `ModelRegistry`, `ModelContextFactory`; `validate_exchange_payload`, `save_exchange_mat`, `build_mesh_from_exchange_mat`, `build_boundary_facets`; supported cells=`triangle|tetrahedron`, source connectivity=`1-based`; v1/v2/standalone legacy MAT fail actionable
- Cache: `build_process_forward_setup_key(*, mesh_file, mesh_content_hash, n_elec, z, pattern_config)`, `backend_signature_from_forward_model`, `model_signature_from_forward_model`, `pattern_signature_from_forward_model`, `rom_signature`, semantic `cache_manager.get_or_compute_semantic`
- Perf: `detect_performance_capabilities`, `select_preconditioner`, `select_fast_linear_path`, `select_fused_strategy`, `probe_mpi_runtime`, `resolve_3d_cuda_forward_solver_policy`, `resolve_3d_cuda_mat_solve_policy`
- Forward diagnostics: `EITForwardModel.get_backend_diagnostics()` (see §I.diag)

### CLI / scripts

- `scripts/run_synthetic_parity.py` — forward+inverse parity check
- `scripts/run_reconstruction_unified.py` — unified reconstruction runner; `--preconditioner diag|noser|prior|pmat|coarse|custom|petsc-gamg|cholmod|pyamg`
- `scripts/benchmarks/benchmark_3d_runtime.py` — `--forward-only on|off`, `--forward-solver-preset 3d_gamg|3d_hypre|spd_gamg|spd_hypre|direct|3d_amg|hypre_boomeramg`, `--forward-mat-solve auto|on|off`, `--petsc-device auto|cpu|cuda`; MUMPS preset retired
- `scripts/benchmarks/benchmark_difference_runtime.py`
- `scripts/benchmarks/compare_cem_formulations.py` → PyEIDORS/NGSolve/EIDORS classic-vs-Robin CSV+JSON+plot+report
- `scripts/benchmarks/cem_high_precision_reference.py prepare|compare` → canonical coarse P1 mesh + independent multiprecision CEM truth + cross-FEM absolute-error CSV/JSON/plot
- `scripts/benchmarks/cem_continuum_reference_suite.py prepare|compare` → true unit-disk common P1 h-sequence + independent Fourier–Nyström/Richardson continuum CEM reference + cross-FEM convergence CSV/JSON/plot/report; companion `scripts/benchmarks/ngsolve_cem_continuum.py` + `compare_with_Eidors/compare_cem_continuum.m`
- `scripts/diagnostics/probe_petsc_cuda.py` — PETSc CUDA + MPI probe, `--pretty`, `--require cuda`
- `scripts/diagnostics/benchmark_gui_forward_first_load.py` — GUI-style forward setup/solve timing, `--mode setup|solve|both`, `--profile`, JSON output
- `scripts/ci/run_sharded_unit_tests.py` — `--shard <name>` | `--all`, `--timeout`, `--report-dir`, per-shard JSON summary
- `scripts/gui/run_eit_app.sh` / `.ps1`; repository-root `eit-gui`, `EIT-GUI-CPU.cmd` / `EIT-GUI-GPU.cmd` — GUI launchers/wrappers. Default profile is `auto`: choose complex-capable runtime first (`complex64-cuda` when GPU visible, otherwise `complex64`); `--real-cpu` / `--real-gpu` are expert escape hatches
- `scripts/benchmarks/benchmark_mesh_io_formats.py --mesh <path.msh|path.xdmf|path.h5> --repeats N --output-json <path>`
- `scripts/cache/migrate_artifacts_to_hdf5.py --root <path> --dry-run|--apply [--manifest <path>]` — migrate legacy `.npz/.npy` numeric artifacts to `.h5`, emit JSON manifest, leave source files untouched
- `pyeidors-interop capture|validate|inspect|import-geometry|register|verify-numerics` / `python -m pyeidors.interop ...` — novice EIDORS migration CLI; Bridge Package v3-only validation, managed registration, exact 2D/3D model load
- MATLAB `pyeidors_export_v3(source,out_dir,...)` / `pyeidors_import_v3(package_dir)` — explicit standard-object export/import; `fwd_model|inv_model|image`, background/target/data selectors

### §I.diag — diagnostics surface

- Forward `get_backend_diagnostics()` keys: `forward_rhs_count`, `forward_ksp_solve_count`, `forward_ksp_mat_solve_count`, `forward_ksp_setup_count`, `forward_ksp_setup_attempts`, `forward_reuse_preconditioner_requested`, `forward_reuse_preconditioner_applied`, `forward_pc_session_reused`, `forward_pc_refresh_triggered`, `forward_pc_refresh_reason`, `ksp_type`, `pc_type`, `pc_factor_mat_solver_type`, `petsc_mat_type`, `petsc_vec_type`, `petsc_dense_mat_type`, `petsc_solve_mat_type`, `forward_mat_solve_effective`, `forward_mat_solve_requested`, `forward_mat_solve_effective_policy`, `forward_mat_solve_policy_reason`, `forward_mat_solve_policy_warning`, `forward_ksp_iterations_per_rhs`, `forward_ksp_iterations_total`, `forward_ksp_converged_reason`, `forward_ksp_converged`, `forward_setup_seconds`, `forward_solve_seconds`, `forward_factor_cache_hit`, `gpu_fallback_reason`, `fallback_reason`, `forward_mat_solve_fallback_reason`, MPI size/rank/support fields
- Inverse `_last_fast_linear_meta` keys: `path`, `resolved_preconditioner`, `fallback_reason`, `fast_linear_path_selected`, `fast_linear_path_reason`, `jacobian_representation`, `jacobian_shape`, `dense_jacobian_materialized`, `linear_iterations`, `matrix_free_pc_source`, `matrix_free_pc_mode`, `matrix_free_pc_floor`, `matrix_free_pc_min`, `matrix_free_pc_max`, `matrix_free_pc_reason`, `matrix_free_pmat_available`, `matrix_free_pmat_kind`, `matrix_free_pmat_attr`, `matrix_free_ksp_backend_requested`, `matrix_free_ksp_backend_effective`, `matrix_free_ksp_backend_fallback_reason`
- Benchmark artifact `forward_solver_benchmark` JSON: mesh/RHS/solver/PC/Mat/Vec/timing/iterations/device/fallback/finite-output/CUDA-errors/MPI fields

### §I.extended — landed / scope-limited / planned surfaces

Markers describe current code: `[landed]` exists, `[scope-limited]` is intentionally partial, `[planned]` is not contractual yet. See §T for remaining scope.

**Existing hooks v1 MUST build on (not replace):**
- `src/pyeidors/data/difference.py:66` `build_difference_vector` / `project_measurement_jacobian` — existing normalized/raw difference math. V39 gates parity of the new `normalize_time_difference` against this.
- `src/pyeidors/core_system.py:807` regularization dispatcher (NOSER / Tikhonov / Smoothness / TV) + `eidors_one_step_noser` preset — current "one-step NOSER on the current parameter mesh". v1 extends this to dual-mesh + offline RM form; V38 gates numerical parity with this baseline on small meshes.
- `src/pyeidors/inverse/jacobian/linearized.py:333` lazy adjoint matrix-free path — retained for phase-2 (`T22`, `T23`); not a v1 main-line dependency. Cold-path slowness observed in `lazy_adjoint_matrix_free_experiment_log_20260421.md:1` is part of why v1 prefers offline RM + online matmul.

- [landed] `pyeidors.inverse.dual_mesh`: `DualMesh(fine_mesh, coarse_mesh)`, `coarse2fine(mesh_fine, mesh_coarse) -> csr_matrix`
- [landed] `pyeidors.inverse.reconstruction_matrix`:
  - `build_one_step_rm(J, regularization, lambda_, mode="tikhonov"|"noser"|"laplace", form="param"|"measurement") -> ndarray`
  - `reconstruct_difference(rm, dv, normalize=True, v_ref=None) -> ndarray`
- [landed] `pyeidors.inverse.prior.laplace`: `graph_laplacian(mesh, weight="unit"|"volume") -> csr_matrix`
- [landed, scope-limited] `pyeidors.inverse.greit`:
  - `build_3d_greit_rm(fwd_model, targets, noise_figure, regularisation) -> GREITRM`
  - `GREITRM.reconstruct(dv) -> voxel_image`
  - `greit_metrics(voxel_image, target_mask) -> {AR, PE, RES, SD, RNG}`
  - [planned] EIDORS parity: `build_eidors_3d_greit_model(config) -> GREITRM`; config covers `imgsz/xvec/yvec/zvec`, `distr`, `Nsim`, `target_size`, `target_plane`, `target_offset`, `noise_figure|weight`, `noise_covar`, `desired_solution_fn`, `training_mode="forward"|"linearized"`, `keep_model_components`
  - [landed, scope-limited] EIDORS parity artifact: HDF5 `.h5` stores `RM`, `PJt`, `M`, `noiselev`, `weight_chosen`, `vh`, `vi`, `xyzr`, `D`, `Y`, `rec_model`, cache signature; `.npz` is legacy/small-test only
- [landed] `pyeidors.inverse.greit_registry`: `greit_artifact_signature(config) -> sha256`, `resolve_or_build_greit_artifact(config, backend="native"|"matlab-eidors", auto_build=True) -> GREITRM`; registry manifest maps exact config signature → HDF5 artifact + provenance
- [landed] `pyeidors.inverse.matrix_free.dual_mesh`: `DualMeshJacobianOperator(fwd_model, coarse2fine)` exposing `Jv`, `JTr`, `normal_matvec`
- [landed, scope-limited] `pyeidors.inverse.block_system` (extended): movement support currently via `build_sigma_contact_block_metadata(..., n_movement=...)`, `build_electrode_movement_jacobian`, `prior_movement`; no separate `build_sigma_contact_movement_block_metadata` function unless T93/T98 decides to add alias
- [landed] `pyeidors.data.difference.normalize_time_difference(v_t, v_ref, floor=...) -> dv_norm`
- [landed] `pyeidors.data.channels.bad_channel_mask`: apply mask to `J`, `residual`, `W`
- [landed] `pyeidors.perf.gpu_kernels.rm_matmul`: batched `RM @ ΔV` on GPU (torch / cupy)
- [landed, experimental] Nix profile `.#cuda-amgx`: PETSc configured with CUDA + PCAMGX (`--download-amgx=<src>` or `--with-amgx-dir=<path>`, `--with-64-bit-indices=0`), then rebuilds `petsc4py`, SLEPc, DOLFINx, `fenics-dolfinx` against same PETSc; FEniCSx version upgrade alone is neither required nor sufficient

CLI additions still pending unless named above:

- `--algorithm one-step-gn|noser|laplace|greit-3d|matrix-free-gn-cg|tv-pdhg`
- `--dual-mesh on|off`, `--coarse-mesh <path>`
- `--rm-cache <path>`: load / save precomputed RM
- `--normalize-difference on|off`
- `--greit-targets <path>`, `--greit-metrics-out <path>`
- `--bad-channel-mask <csv|json>`
- [landed] `pyeidors.io.hdf5_artifacts`: `write_hdf5_artifact(path, arrays, metadata, chunks, compression)`, `read_hdf5_artifact(path, lazy=True)`, `migrate_npz_to_hdf5(src, dst)`

## §V — invariants

Authoritative invariant rows are split by domain below; exact routing is in `docs/spec/id-map.tsv`. New rows may enter the root inbox, then move to one registered domain without changing ID.

| domain | count | authority |
|--------|------:|-----------|
| Core runtime and public API | 5 | `docs/spec/invariants/core-runtime.md` |
| Forward FEM, CEM, PETSc, and linear solves | 59 | `docs/spec/invariants/forward-fem.md` |
| Inverse solvers, Jacobians, RM, and GREIT | 228 | `docs/spec/invariants/inverse-reconstruction.md` |
| Meshes, geometry, protocols, data, and artifacts | 70 | `docs/spec/invariants/geometry-data.md` |
| Caching, hashing, budgets, and performance | 57 | `docs/spec/invariants/cache-performance.md` |
| EIDORS/MATLAB interoperability and exchange contracts | 27 | `docs/spec/invariants/interop.md` |
| GUI, workers, launchers, and interactive workflows | 321 | `docs/spec/invariants/gui.md` |
| Validation, CI, packaging, docs, and governance | 12 | `docs/spec/invariants/tooling-validation.md` |

### §V.inbox — new invariant intake

| id | invariant | source |
|----|-----------|--------|

## §T — tasks

### §T.phase — active priority queue

Only non-completed work is active here. Frozen environment tasks: T2/T3/T7. Frozen research tasks: T11/T27/T28/T68/T70. Current governance order: T614 → T615. Completed rows are in `docs/spec/history/tasks-completed.md`.

| id | status | task | cites |
|----|--------|------|-------|
| T2 | . | FROZEN env(MPI): enable MPI size > 1 only after explicit unfreeze + distributed Mat/Vec design + `mpiexec -n 2` smoke plan; until then keep fail-fast guard | V18 |
| T3 | . | FROZEN precondition: flip `matrix_free_ksp_backend` default to `auto` only after real 3D PETSc-vs-SciPy parity artifact; no default change before evidence | V12 |
| T7 | . | FROZEN env(CUDA): 3D inverse benchmark only after explicit unfreeze + `probe_petsc_cuda --require cuda` passes in target shell | V19 |
| T11 | . | FROZEN research/closed-form: PETSc/petsc4py structural reuse hints. `KSPSetOperators(ksp, Amat, Pmat)` has no `SAME_NONZERO_PATTERN` parameter in current API; `petsc4py.KSP.setOperators(A=None, P=None)` likewise. Reopen only if new API evidence appears; current main line stays `setOperators(A_new)` + `KSPSetReusePreconditioner(True)` | V13 |
| T27 | . | FROZEN research: SBL / BSBL / SA-SBL coarse-basis enhancement (RBF, sparse-inclusion, low-rank anatomical basis); no implementation/default use until explicit unfreeze + T70-style benchmark case exists | V31 |
| T28 | . | FROZEN research: CNN / U-Net postprocess plug-in interface (coarse 3D image in → enhanced image out), no physics replacement; needs explicit unfreeze + baseline-gain evidence | V29 |
| T68 | . | FROZEN research: propagation-aware prior (directed anatomical/vascular graph, velocity-range prior, path-constrained smoothness for nerve / plant conduction); opt-in only, needs explicit unfreeze + baseline-gain evidence | V31,V49 |
| T70 | . | FROZEN research: SBL/BSBL acceptance benchmark only after explicit unfreeze + T27 candidate exists; compare vs GN/NOSER/Laplace/TV-IRLS/4D prior; promote only if accuracy/latency win recorded | V31,V35,V49,T27 |
| T97 | . | Capture + gate separate MATLAB/EIDORS 5936 measurement-protocol official fixture; only after pass may UI/docs/papers say `48e/5936 official-equivalent`. Until then T49 wording stays 48e official fixture + 5936 surrogate | V50,V63 |
| T585 | . | Restore default full-suite `src/pyeidors` coverage to ≥87% without lowering gate | V637 |
| T614 | . | Delete placeholder `pyeidors.main` hello entrypoint and coverage-only tests | V778,I |
| T615 | . | Audit four oversized modules plus tracked `reports/`/`pictures/`; record impact-ranked decomposition and keep/drop decisions without code/asset mutation | V75,V779 |

## §B — bugs

Resolved history lives in `docs/spec/history/bugs.md`. New bugs enter below with a unique monotonic ID and a recurrence-catching invariant citation.

| id | date | cause | fix |
|----|------|-------|-----|
