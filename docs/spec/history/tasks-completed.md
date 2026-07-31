# Completed task history

Immutable completed-task registry; active work remains in root `SPEC.md`.

| id | status | task | cites |
|----|--------|------|-------|
| T1 | x | Full PETSc `PCFIELDSPLIT` inverse solver for `sigma + z_contact` joint estimation (additive → multiplicative → Schur) | V20 |
| T4 | x | Real 3D benchmark artifact proving G1 persistent-KSP setup-time saved (iter histogram + cumulative setup seconds): `scripts/benchmarks/benchmark_forward_ksp_session_reuse.py` drives N forward solves with same σ sequence across `forward_pc_refresh_policy ∈ {auto,never,always,lag}` and emits HDF5 + JSON + Markdown bundle (per-call iter array, setup_seconds, session_reused, refresh_reason histogram, cumulative_setup_seconds, p50/p95 iter, `sigma_sequence_hash`, command argv, mesh provenance, V13/V14/V52/V67/V80 cites, `env_path`); 3D 16e ref5 hypre artifact `reports/runtime_benchmarks/forward_ksp_session_reuse_t4_20260429_3d/` records controlled σ A/B evidence: `auto.n_reused=9/10`, `never.n_reused=0/10`, G1 cum-setup saved 0.0309s (warm/cold ratio 0.9140), `sigma_sequence_hash=0ceb01d20ba732cd75c41b7f2d2ae5ae60679d61605b231756b9a2fc2ee4f006`, mesh ref5 cells=28248 vertices=6022. Gate: `tests/unit/test_benchmark_forward_ksp_session_reuse.py` smokes 2D `spd_hypre` regimes `auto,never` in-process, asserting V13 (`auto.n_reused>=1`), V14 (`never.n_reused==0`), V80 (equal `sigma_sequence_hash` + mesh provenance), schema, env_path, HDF5 dataset names, summary.md V cites | V13,V14,V80 |
| T5 | x | Wire `JacobianLinearization.assert_compatible(sigma_fp)` at runtime reuse path; stored fingerprint currently inert | V9,V15 |
| T6 | x | Persistent across-iteration Jacobian cache keyed on `sigma_fingerprint` + mesh content hash: opt-in `GaussNewtonReconstructor(persistent_jacobian_cache=True)` ctor flag (default `False` → no behaviour change). New `pyeidors.inverse.jacobian.process_jacobian_cache` (shared `pyeidors.cache.process_lru.ProcessLRUCache`, max_items=4) keys SHA256 over `{sigma_fingerprint, mesh_file or mesh_content_hash, calculator_signature, model/pattern/backend hex signatures, jacobian_method, extra={measurement_space, difference_mode, difference_orientation}}`. V9 guard: empty `sigma_fingerprint` raises `ValueError`; V17 guard: both `mesh_file`+`mesh_content_hash` empty raises `ValueError`; V81 guard: Direct↔`EidorsJacobianAdapter` sign-convention swap misses cache. GN runtime `_calculate_iteration_jacobian` dense branch checks cache before `jacobian_calculator.calculate(...)`, stores post-projection pre-negation array on miss; operator/matrix-free path bypasses cache and reports disabled lookup, not stale dense key. Runtime emits `persistent_jacobian_cache_lookup={hit, stored, key, artifact, reason?}` diagnostic alongside existing `jacobian_cache_lookup` / `startup_cache_lookup`. Gates: `tests/unit/test_process_jacobian_cache.py` (primitive contract: empty-sigma reject, V17 reject, byte-stable hash, distinct-axis keys incl calculator signature, get/put round-trip, LRU evict, clear); `tests/unit/test_gn_runtime_persistent_jacobian_cache.py` (runtime gate: repeat run hits cache + `calculate` count drops to 0, default-off no entry/no behaviour change, distinct mesh_content_hash misses + new entry, Direct→Adapter misses, operator `jacobian_method=linearized` skips cache and resets lookup). `cache` shard 18 files / `inverse-gn` shard 36 files green | V9,V17,V81 |
| T8 | x | Guard canonical solver/PC matrix doc against preset-default drift (R11 hard gate) | V4,V5,V6 |
| T9 | x | Explicit startup cache skip for operator Jacobian (avoid `np.asarray(JacobianLinearization, dtype=float)` path) | V15 |
| T10 | x | PETSc AmgX CUDA path wiring after explicit unfreeze: `cuda_amgx` requires `petsc_amgx_cuda_candidate=true`, Hypre CUDA remains blacklisted, and explicit PCAMGX CEM solve stays on sparse CUDA KSP instead of dense fallback | V19,V43,V111,V661,B5,B560 |
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
| T25 | x | Electrode-movement metadata path landed: `build_sigma_contact_block_metadata(..., n_movement=...)` adds `e` block + `H_σe`/`H_ze`/`H_ee`; `build_electrode_movement_jacobian` finite-diff helper + `prior_movement` diagonal prior. Scope = metadata/preconditioner contract, not production Schur solve | V20,V32 |
| T26 | x | Bad-channel mask + noise covariance `W` wired into Jacobian rows, residual vector, measurement-weight contract, and RM builder so offline / online weights match | V34,V35 |
| T29 | x | GPU `RM @ ΔV` online kernel: batched multi-frame matmul on GPU, reuses normalized-difference path | V29,V35 |
| T30 | x | Forward CUDA AMG policy/report surface: presets + capability fields for Hypre/AmgX, `forward_solver_benchmark` reports `petsc_hypre_available` / `petsc_amgx_available` / `petsc_amgx_cuda_candidate`, high-level runtime downgrades unavailable AmgX and blacklisted Hypre CUDA to `3d_gamg`; actual `.#cuda-amgx` PETSc PCAMGX build landed in T33/T10 | V6,V13,V43,V44,T10,T33 |
| T31 | x | Dual-mesh integration smoke: fine CEM + coarse recon + EIDORS-style parity metric on synthetic sphere target | V25,V29,V30 |
| T32 | x | Milestone **FEniCSx-EIT-3D-v1**: ties V25–V35 + GPU online matmul; 10-point checklist (fine CEM, coarse voxel, c2f, reusable KSP/PC, adjoint J on coarse, one-step GN/NOSER/Laplace RM, normalized Δv, GPU RM@Δv, GREIT metrics, bad-channel / W weighting) | V25,V26,V27,V28,V29,V30,V31,V33,V34,V35 |
| T33 | x | `cuda-amgx` Nix profile after explicit unfreeze + PCAMGX source/package path: PETSc rebuilt with CUDA + 32-bit `PetscInt` + AmgX external, then same-chain `petsc4py`/SLEPc/DOLFINx/`fenics-dolfinx`; `probe_petsc_cuda.py --require cuda` reports `petsc_amgx_cuda_candidate=true`, and 3D tetra `cuda_amgx` benchmark reports real `fgmres+amgx` with no dense fallback. FEniCSx upgrade alone ⊥ AmgX enablement | V19,V42,V43,V661,B5,B560 |
| T34 | x | Switch GUI/default 3D difference route to cached dual-model RM reconstruction when RM artifact exists; cold path may build/load RM, hot path must bypass `DirectJacobianCalculator` | V37,V46,V48 |
| T35 | x | Optimize RM hot path for 48e/5936: persistent RM on device, batched frames, no per-call tensor rebuild, minimal CPU↔GPU copy, float64/float32 policy recorded | V29,V37,V50 |
| T36 | x | Real 48e/5936 dual-model report: compare one-step NOSER/Laplace RM and 3D GREIT RM, split fine-CEM/J build, RM build, artifact load, 1-frame apply, 512-frame apply, GPU/CPU paths | V40,V46,V47,V50 |
| T37 | x | Harden benchmark / GUI timing harness against PATH-shadowed `env`: prefer `/usr/bin/env`, add guard test, record `env_path` in timing artifacts | V52,B8 |
| T38 | x | Guard GUI `single_step_cached`: feasible-step alpha bound by `sigma_floor`, illegal `sigma_try` → `inf`, final `sigma_est` finite & floored before `fwd_solve`; add CUDA/DOLFINx parity regression | V53,B9 |
| T39 | x | Stabilize GUI 3D PyVista offscreen drag display: low-res drag frame is scaled to the same QLabel physical target before DPR assignment, so visual canvas size is invariant | V54,B10 |
| T40 | x | Build EIDORS GREIT source map + golden fixture capture: MATLAB script exports `vh`, `vi`, `xyzr`, `D`, `Y`, `PJt`, `M`, `noiselev`, `RM`, `weight`; include tiny 3D cylinder + 48e/5936 reduced case | V50,V55,V63,B11 |
| T41 | x | Implement `GREIT3D_distribution` parity builder: `imgsz/xvec/yvec/zvec`, `downsample`, point-in-volume, target centers, volume/inside mask, deterministic order | V55 |
| T42 | x | Implement finite-target training response engine: homogeneous `vh`, per-target `vi`, target radius/size/plane/offset, contrast, batching/cache; keep linearized shortcut explicit non-parity mode | V56,V57 |
| T43 | x | Implement desired image stack: default EIDORS-like `GREIT_desired_img` for 3D rec model + custom `desired_solution_fn`; output `D` independent from raw target `T` | V58 |
| T44 | x | Rework `calc_GREIT_RM` parity core per design review: consume V57 `Y` + V58 `D`; emit/compare `PJt`, `noiselev_eff`, `Sn`, `M`, `RM`; use `solve(M.T, PJt.T).T` non-conjugate solve; scalar weight supported, matrix weight explicit unsupported; diagnostics include shapes/rank/condition/fallback; unit compare against exported EIDORS components | V57,V58,V59 |
| T45 | x | Add tiny bracket-search fixture first: injectable scalar objective, known log10 optimum, bracket/tolerance/call-budget asserts, no RM-core coupling. Then implement NF/image-SNR scalar-weight optimizer: target simulation, bounded log10 search, achieved metric/tolerance metadata, failure diagnostics | V60 |
| T46 | x | Add EIDORS-parity GREIT HDF5 artifact/cache schema: store model components in `.h5`, cache `PJt` across weight search, signature includes V55..V61 inputs | V61,V62,V65 |
| T47 | x | Add MATLAB EIDORS parity diagnostics + tests: compare PyEIDORS vs official EIDORS `Y/D/PJt/M/RM/recon/metrics`; record tolerances and drift report | V63 |
| T48 | x | Add common-config offline warmup CLI/GUI path: precompute/load 16/32/48e 3D GREIT `.h5` artifacts; online load+matmul only; no routine cold build for known hardware | V64,V65 |
| T49 | x | T49 runtime gate landed as surrogate 48e/5936 benchmark + MATLAB/EIDORS 48e official fixture gate. Cold build/HDF5 load/1-frame/512-frame online apply/metrics/bad-channel/W/GPU-CPU stability recorded; official-equivalence claim scope = 48e fixture actual `n_measurements=2160`. 5936 protocol official fixture remains pending T97; ⊥ claim `48e/5936 official-equivalent` | V50,V55,V56,V57,V58,V59,V60,V61,V62,V63,V64,V65 |
| T50 | x | Implement large-cache HDF5 IO layer: chunked/compressed datasets for `RM/Y/D/PJt/M/vh/vi/xyzr`, JSON metadata attrs, checksum, lazy dataset reads, legacy `.npz` read-only import/migration path | V61,V62,V65 |
| T51 | x | Repo persistence inventory + blocklist: classify all `.npz/.npy/.msh/.xdmf/.h5/.mat` writers/readers; mark legacy/test-only exemptions; add CI scan forbidding new production `.npz/.npy` writes | V65,V67,B12 |
| T52 | x | Mesh HDF5-first hardening: support `.xdmf/.h5` cache load without source `.msh`; store source hash/provenance optional; generator writes XDMF/HDF5 from in-memory mesh even when `save_msh=false`; round-trip facet/cell tags + physical groups | V66,V68 |
| T53 | x | Convert RM/GREIT/one-step artifacts from `.npz` to HDF5 `.h5`; keep `.npz` loader read-only + migration helper; update GUI cached-RM loader | V36,V37,V61,V64,V65,V67 |
| T54 | x | Convert dataset generator + GUI simulation export from `mesh_info.npz` / `sample_*.npz` / "NumPy archive" to HDF5 package; update i18n labels and file dialogs | V65,V67 |
| T55 | x | Convert diagnostics/benchmark/reconstruction output bundles (`outputs.npz`, `result_arrays.npz`, `inverse_3d_overview_data.npz`, gallery bundles) to HDF5; JSON/CSV summaries remain | V40,V65,V67 |
| T56 | x | Convert MATLAB/interop mesh bridge arrays from `.npz` default to HDF5/v7.3-compatible `.h5`; retain `.mat`/legacy `.npz` import adapters only | V63,V65,V67 |
| T57 | x | Add mesh IO format benchmark + regression test: compare `.msh` import vs XDMF/HDF5 load on representative meshes; store JSON artifact with speed ratio and tag equality | V66,V68 |
| T58 | x | Raise GUI 3D offscreen drag defaults to 60 fps + full-DPR framebuffer; skip redundant offscreen window resizes; keep env-controlled lower-fps/downsample mode for constrained machines without reintroducing V54 size jitter | V69,B13 |
| T59 | x | Baseline alignment freeze: official-style framewise GN/NOSER/Laplace checklist + tiny fixtures; verify `Jᵀ W J + hp² RtR` formula, `hp²` scaling, normalized/raw difference parity, artifact metadata names | V26,V27,V28,V38,V40 |
| T60 | x | General `RtR/R_prior` prior contract: dense/sparse/`LinearOperator`/callable inputs, `apply(v)`, `diag()`, `as_RtR()`, signature hash, metadata, HDF5 persistence; wire through RM builder + GN runtime without forced dense materialization on large 3D | V26,V28,V35,V36,V49 |
| T61 | x | Curvature prior mode: expose squared EIDORS-Laplace `RtR = prior_laplace(mesh).T @ prior_laplace(mesh)` as named `curvature` / `graph_ltl` prior; compare vs Laplace smoothing on same mesh; cache signature distinguishes `laplace` vs `graph_ltl` | V28,V35,V36,V49,V91 |
| T62 | x | TV-IRLS inverse prior: iterative `RtR(x)=L.T @ diag(1/sqrt((Lx)^2+β)) @ L`; β/floor finite guard, max outer iterations, stale-RM invalidation, monotone objective smoke; keep TV-PDHG postprocess as separate seeded refinement | V28,V35,V49 |
| T63 | x | Measurement-domain temporal filtering before reconstruction: causal EMA/moving-average + optional bandpass/lock-in hook over `Δv`/raw frames; channel mask + `W` applied deterministically; filter state stored in metadata, no smoothing of timestamps | V33,V34,V35,V37,V40 |
| T64 | x | Dynamic sequence data contract: frames carry `t`, `dt`, sampling rate, frame id, reference policy, stim/meas signature, bad-channel mask, `W`, frequency/context metadata; HDF5 package round-trips multi-frame arrays + metadata; `MeasurementDataset` remains single-frame-compatible | V33,V34,V35,V37,V65,V67 |
| T65 | x | Batch spatiotemporal GN / 4D prior: windowed solve over `X[t,param]` with spatial prior `Rs`, temporal `Dt` first/second difference, block normal operator, λ_s/λ_t metadata, rowwise RM baseline comparison | V25,V31,V35,V49 |
| T66 | x | Spatiotemporal TV / Huber prior: separable spatial graph + temporal difference penalties; preserves abrupt wavefront/onset better than L2 time smoothing; ROI support; compare against T65 on travelling-wave fixture | V28,V31,V35,V49 |
| T67 | x | Online Kalman + fixed-lag smoother prototype: state model `x_t=A x_{t-1}+q`, measurement `y_t=Jx_t+n` or RM-observation shortcut; latency/lag metadata, `Q/R` estimation hooks, no default until T69 metrics pass | V35,V37,V49 |
| T69 | x | Dynamic validation benchmark: synthetic travelling wave + plant slow-pulse fixtures; report onset-time error, peak-time error, propagation-speed error, amplitude attenuation, SNR gain, spatial metrics; fail if regularization delays peak beyond tolerance ? | V30,V40,V41,V49 |
| T71 | x | Add EIDORS-style `add_noise`: SNR-exact Gaussian scaling for `v1`, `v1-v2`, normalized-difference signal; support arrays and `EITData`; export via `pyeidors.data` | V70 |
| T72 | x | Add bucket all-modes noise gradient experiment: full256/full208/far3-drop-near3-keep/raw160/poly2/poly3/spline across SNR ladder; CSV+plots+Word in `仿真各情况加噪声梯度测试` | V70,V71 |
| T73 | x | Jacobian sign/convention contract freeze (Path A): 加 characterization tests (`Direct == -Adjoint` signed parity + `abs(Direct) == abs(Adjoint)` 辅助 + `linearize().to_dense()` 同) + class docstring 明确两 sign 约定 (Direct=+∂V/∂σ runtime, Adjoint=-∂V/∂σ EIDORS canonical) + `sign_convention` metadata 属性. **不删** Adjoint，**不动** GN/RM/cache. 融合 → Path C 未来任务 (抽 shared core, Direct 主面，Adjoint 退为 sign adapter, 最后视情况删) | V73,B15 |
| T74 | x | 修 `prior/tv_irls.py:358` `_effective_beta(beta, beta)` → `_effective_beta(beta, beta_floor)`；扩展测试覆盖 β<beta_floor 路径（断言 objective 与外层 IRLS 一致） | V74,B16 |
| T75 | x | Path C — Jacobian 体系渐进融合 (T73 后续). 步骤已全落地：(1) 抽 `pyeidors.inverse.jacobian._core`：几何 setup (`mesh/V/V_sigma/Q_DG/DG0/cell_areas`)、`compute_field_gradients`、`measurement_to_current_patterns`、`assemble_jacobian_efficient_numpy`、`assemble_jacobian_traditional`、`convert_electrode_to_measurement_jacobian`、`calibrate_block_size_once` (commit `cfa2976`; dead `compute_adjoint_fields_efficient` helper later deleted after zero caller audit). (2) `DirectJacobianCalculator` 薄 façade 调 shared core，保留 `+∂V/∂σ` runtime sign + cache manager + CUDA assembly orchestration；下游 9 caller 不变 (commit `0f849cf`). (3) `EidorsJacobianAdapter._assemble_numpy` 改 sign-flip adapter 调同一 shared core (`return -assemble_jacobian_efficient_numpy(...)`); torch GPU path + `linearize_lazy` 保留 (commit `bb27df0`). (4) 完整回归全 `test_jacobian_*` + `test_one_step_rm_parity` + `test_modular` + V21/V38/V63 parity gate 仍过. (5) 改名 `EidorsStyleAdjointJacobian` → `EidorsJacobianAdapter` + 10 caller (scripts/src/tests) 同步迁移；曾短暂保留过渡别名 + 兼容 contract test (commit `3f8d6d6`). 旧 API 别名随后于 commit `2945bcf` 有意删除：production caller 全清零, 外部 import 旧名报 ImportError. GN runtime / RM 公式 / cache 签名 (V73 物理 δσ pairing) 不动. 后续约束: 见 V75 (新 sibling impl 必入 `_core` 或子类 base, 否则 §B 解释 dup) | V73,V75,T73,B15 |
| T76 | x | Path C → sparse solver 群. **实际完成范围 (commit `011151b`)**: 7-file tier → 5-file. 删 `sparse_bayesian.py` (8-line import alias, 全 caller migrate 至 `sparse_bayesian_engine`) + `sparse_bayesian_backends.py` (190 行 `SparseBayesianBackendMixin`). 14 wrapper method 折入 `SparseBayesianReconstructor` class body: cuqi adapter (`_linear_model`/`_sparse_prior`/`_gaussian_likelihood`/`_bayesian_problem`/`_solve_with_cuqi_map`) 用 module-level static binding 保 monkeypatch 注入点; pure forwarder (`_solve_fista`/`_solve_irls`/`_compute_projection`/`_estimate_lipschitz_constant`/`_get_coarse_matrix`/`_solve_sparse_map`/`_coarse_initialization`/`_multilevel_correction`/`_block_refinement`) 留 instance method form 保 `sparse_map_solver` 之 `reconstructor._foo(...)` 调点. **未做 (intentionally NOT scoped)**: 抽独立 `sparse._core` 模块 — kernel functions (FISTA/IRLS/projection/coarse hierarchy/GPU context) 已分别 live 在 `sparse_optimizers.py` / `sparse_projection.py` 等 dedicated module, 进一步 `_core` 封装 = 重 indirection 无收益. **Entrance gate**: `tests/unit/test_sparse_consolidation.py` 7 case 鎖架构 (alias 不可 import / mixin 不存 / MRO 单 / 14 wrapper method on reconstructor / kernel re-export 仍 reachable / `_solve_sparse_map` signature 稳). **邊界守 (V73-style)**: V49 phase-2 status 不退; SBL/BSBL 容差不松; cache signature 字段不變; `monkeypatch.setattr(engine_module, "LinearModel", None)` 仍触 cuqi import-guard branch (static binding 保留 late-lookup 行为) | V49,V73,V75,T75 |
| T77 | x | Path C → GN engine+runtime (`gauss_newton.py` entry + `gauss_newton_engine.py` 591 行 `GaussNewtonReconstructor` + `gauss_newton_runtime.py` 3694 行 50+ helpers + `matrix_free_gn.py` + `gauss_newton_line_search.py` + `gauss_newton_weights.py` + `gauss_newton_device.py`). 目标：抽 `pyeidors.inverse.solvers.gn._core` 或同级 companion：PETSc matrix-free Hv/PC builder (`_PETScMatrixFreeHessianContext` + `_PETScMatrixFreePCContext` + `_build_matrix_free_*`), `_JacobianActionBundle`, residual/objective/step/rollback/iteration_log/startup_cache shared helpers. `gauss_newton_runtime.py` 拆按职责至 sub-modules (`linear_system.py` / `startup_cache.py` / `measurement_space.py` / `step_size.py` / `iteration_log.py`); `GaussNewtonReconstructor` 改 thin reconstructor wrapper 调 sub-module helpers. **Gate 已落地**: phase 1 string/signature gate (`50ba5aa`, `tests/unit/test_gn_runtime_contract_freeze.py`) 锁 V73 `rhs=-jtr` literal + V11/V12/V14 string + `_IterationLog`/`_JacobianActionBundle`/`run_reconstruction`/PETSc-context import surface; phase 1.5 behavioural golden gate (`17a190a`, `tests/unit/test_gn_runtime_golden_gate.py`) 锁 fixed dense-vs-fast delta, `matrix_free_pc_source`, PETSc fallback reason, startup payload, `_IterationLog.to_payload`; startup payload canonical SHA256 已 pin 常量 (`b14f188`); fast-PCG / woodbury `linear_iterations` golden 已补 (`linear_iterations == 3` / `0`). **commit #4/#5/#6/#7 gate**: `test_gn_runtime_golden_gate.py` + `test_gn_runtime_contract_freeze.py` + `test_gn_fast_linear_solver.py` 需全绿; source-string contract 已扩展为 runtime + `gauss_newton_linear_system.py` companion, runtime import/monkeypatch surface 必保. **已完成 phase 2**: commit #1 `iteration_log.py` (`e7668ec`), #2 `startup_cache.py` (`87b7e98`), #3 `measurement_space.py` (`8375ba2`), #4 `linear_system.py` (`bdfa41e`; PETSc matrix-free context, `_JacobianActionBundle`, PC builders, fast linear solver, Jv/JTr action helpers moved; runtime 留 re-export + patch-sync wrapper), #5 `step_size.py` (`1e92557`; `_difference_step_size_objective` / `_apply_difference_step_size` / `_select_step_size` moved; runtime 留 wrapper + `minimize_scalar` patch-sync), #6 `regularization.py` (`ee5222d`; `GaussNewtonReconstructor.ensure_regularization_ready` 改 wrapper, regularization prep body + `_is_rtr_prior_contract` moved, engine dead imports 清理), #7 runtime compat final audit (`eaf5efe`; no safe runtime wrapper delete found; `T77_RUNTIME_COMPAT_SURFACE` manifest locks retained wrappers/re-exports). **禁止边界**: 不动 V73 `rhs=-jtr` sign pairing; 不修 V10 fast PCG rtol `1e-5` 容差; 不改 V11 `matrix_free_pc_source` 枚举值; 不改 V12 `matrix_free_ksp_backend` fallback reason 字符串; 不动 V15 `jacobian_update_every` / `jacobian_reuse_tol` 语义; 不删 `_PETScMatrixFreeHessianContext` 等 context 类除非 sub-module 内 import-equivalent; cache signature `forward_pc_session_reused` 等 diagnostic key 一字不改 (V13/V14). **完成边界**: runtime 剩余 wrappers 是 legacy import/monkeypatch surface, 非 dead helper; 真 body 已拆至 companions, 剩余 runtime-owned helpers 仍属 `run_reconstruction` loop local state. | V10,V11,V12,V13,V14,V15,V73,V75,T75 |
| T78 | x | Path C → mesh generator 群. **实际完成范围 (2 commit)**: phase 1 (`4917016`) — 抽 `format_float_compact` / `build_mesh_cache_name` / `build_mesh_cache_name_3d` 至 `geometry/_helpers.py` (cache filename byte-stable; 旧 underscore-prefixed name 留 module-level alias `_format_float`/`_build_cache_name`/`_build_cache_name_3d` for in-tree caller); `MeshConverter` 加 keyword-only `radius_provider` hook, `OptimizedMeshConverter` 改薄子类仅 `__init__` 注 `estimate_radius`. phase 2 (`456321c`) — 抽 `finalize_3d_cylinder_mesh` 至 `_helpers.py` (electrode_names + facet_names + validate_mesh_data_tags + write_association_table + write_dolfinx_mesh_cache + build_eit_mesh 全 30+ 行 tail). 3 cylinder generator (`_LegacyTetraCylinder3DMeshGenerator` / `_GeomV2TetraCylinder3DMeshGenerator` / `_GeomV2HexCylinder3DMeshGenerator`) 之 `generate` tail 縮為 1 helper call. **未做 (intentionally NOT scoped)**: 3 cylinder generator 抽 base class — `_create_geometry`/`_set_physical_groups`/`_structured_geometry`/`_boundary_quads` body 各 generator legitimately 分歧 (extrusion / staged extrusion / tensor-product hex), 强合属 premature abstraction. `SimpleEITMeshGenerator` 已 thin wrapper, 不动. **Entrance gate**: `tests/unit/test_mesh_helpers_consolidation.py` 18 case 鎖 cache-name byte stability + `OptimizedMeshConverter` 仅覆 `__init__` + `radius_provider` keyword + 3 generator `.generate` source 必含 `finalize_3d_cylinder_mesh(`. **邊界守**: v1/v2 generator API + `create_*_eit_mesh` 签名向后兼容; `Cylinder3DMeshConfig`/`ElectrodeArcConfig` field 名 不变; structured sidecar JSON schema 不动 (V66); `_LegacyTetraCylinder3DMeshGenerator` 留; `monkeypatch.setattr(mesh3d_module, "estimate_radius", ...)` 等 helper-branch test 已迁至 `helpers_module` 注入点 | V66,V73,V75,T75 |
| T79 | x | Path C → process-local cache 群. **实际完成范围 (commit `4a18629`)**: 抽 `pyeidors.cache.process_lru.ProcessLRUCache` + `hash_json_payload` + `path_signature`; `geometry.process_mesh_cache` 与 `forward.process_setup_cache` 改 thin wrapper, 保留 public fn 名 (`build_process_mesh_cache_key` / `get_process_cached_mesh` / `put_process_cached_mesh` / `clear_process_mesh_cache`, `build_process_forward_setup_key` / `get_process_forward_setup_bundle` / `put_process_forward_setup_bundle` / `clear_process_forward_setup_cache`) 与 max-items/LRU/thread-lock 语义. **Byte contract**: JSON hash 仍 `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=True`; process mesh key / forward setup key 64-hex deterministic, V16/V17/V36 不退. **Entrance gate**: `tests/unit/test_process_lru_consolidation.py` 锁 shared primitive, key byte-stability, eviction, stats, wrapper surfaces. **未做 (intentionally NOT scoped)**: `pyeidors.geometry.cache._core`, `MeshCacheLayer`, XDMF/HDF5/ADIOS2 disk-cache 统一, cross-layer disk artifact key prefix, historical `.h5` / `.xdmf` / ADIOS2 fixture compat; 这些拆到 T82 pending. **邊界守**: disk cache modules (`dolfinx_mesh_cache.py`, `adios4dolfinx_checkpoint.py`) 继续按 format 分离; `process_setup_cache.py` forward-side key 不删; cache signature 字段不改; ADIOS2 仍 optional. | V16,V17,V36,V62,V65,V66,V67,V73,V75,T75 |
| T80 | x | Path C → `pyeidors.inverse.jacobian.linearized` eager+lazy fusion. **实际完成范围 (commit `868771f`)**: 抽 `_LinearizationBase` ABC 持 8 共 method — `n_parameters`/`n_measurements`/`shape` (从 `cell_areas.size` + `sum(n_meas_per_stim)` 派生)、`assert_compatible` (V9 permissive empty + 错误信息经 `type(self).__name__` 区分 eager/lazy)、`as_linear_operator` (scipy `LinearOperator` 包装)、`normal_matvec` + `as_normal_operator` (`J^T W J v + alpha R v`)、`as_petsc_mat` (PETSc Python mat 包装)、`_apply_regularization` static (callable / sparse / dense / LinearOperator). `JacobianLinearization` (eager, sign=+1.0) + `LazyAdjointJacobianLinearization` (lazy, sign=-1.0) 各保 dataclass form, 仅覆 `_validate_shapes` / `matvec` / `rmatvec` / `hessian_diag` (lazy 加 multi-mode `diag_mode`) + 自身 storage init. **设计偏离 (vs 原 spec hook 名)**: 原 plan 之 `_resolve_adjoint_block(measurement_indices) -> ndarray` hook 未实现 — eager + lazy 之 matvec/rmatvec compute path 实质太分歧 (eager einsum on stored `adjoint_gradients`, lazy 跑 `forward_solve` + `solve_full_rhs` 之 sensitivity-RHS PETSc 解), 共抽 hook 反成 leaky abstraction. 故只共 8 method 之 surface, matvec/rmatvec 各自实现. **Entrance gate**: `tests/unit/test_linearization_base_parity.py` 9 case 鎖 cross-class API + V7 (`matvec == to_dense @ v` / `rmatvec == to_dense.T @ r`) + V8 (`hessian_diag` 公式 + `sign²` + floor) + V9 (permissive empty fp). **邊界守**: V73 sign default 不变 (eager +1.0 / lazy -1.0); `LazyAdjointJacobianLinearization.fwd_model` 等公字段保; PETSc Python mat shape/dtype/context wiring 不动; sparse 路径 (T76) 不触 | V7,V8,V9,V73,V75,T75 |
| T81 | x | Path C → data sweep/audit 群 (`bucket_dense_experiments.py` 1792 + `bucket_domain_audit.py` 625 + `holdout_fit_diff.py` 1249 + `holdout_point_audit.py` 357 + `factor_sweep.py` 845 + `voltage_digit_sweep.py` 410 + `visual_audit.py` 618 + `eit_digit_metrics.py` 759 + `dynamic_sequence.py` 410 + `temporal_filtering.py` 398). 模式重: 多 `*Case` / `*Row` / `*Summary` dataclass + `run` loop + JSON/HDF5/CSV dump. CSV 此处 = metadata/report table/golden table fixture only, 不是 binary cache/data package (V67). **已落地**: phase 1 audit doc + presence gate; phase 2a shared row/table primitives (`5f91113`, `_sweep_core` CSV cell/order writer/table fmt); phase 2b CSV golden fixtures (`4ae00d4`, `tests/fixtures/sweep_csv_columns/*.csv`); phase 2c HDF5 sheet-name fixture gate (`a3b643c`, `write_hdf5_row_tables` + `tests/fixtures/sweep_hdf5_tables/phase2a_table_names.h5`); phase 2d schema base consolidation (`f20eb2d`, zero-field `SweepRow`/`ReconMetricRow`/`StructureMetricRow` mixins + shared `StructureMetrics`; migrated `VoltageDigitSweepSummary`, `BucketDenseSummaryRow`, `BucketFull256CompareSummaryRow`, `HoldoutFitDiffSummary`, `HoldoutStructureMetricRow`, field rows); phase 2e `run_sweep(cases, compute_row, dump_target)` + real caller HDF5/JSON shared dump landed (`_sweep_core.write_json_row_tables` / `write_sweep_table_artifacts`; `eit_voltage_digit_sweep.py`, `eit_factor_sweep.py`, `eit_bucket_dense_experiments.py`, `eit_bucket_full256_compare.py`, `eit_bucket_all_modes_noise_sweep.py`; gates `test_sweep_core_primitives.py` + voltage/factor/bucket CLI artifact assertions); phase 2f `holdout_point_audit` soft-merge landed (`HoldoutPointAuditRow` → zero-field `SweepRow`; `HoldoutPointAuditSummary` stays bespoke); phase 2g V70/V71 gate rerun passed (`tests/unit/test_eidors_noise.py`; real `eit_bucket_all_modes_noise_sweep.py` 7 modes × SNR `inf,10`, 14 summary rows, 5432 field rows, opt-in HDF5/JSON table names stable). **重要设计**: zero-field mixin only; ⊥ dataclass field inheritance that reorders CSV columns; HDF5/JSON 输出 opt-in only, default CSV/Markdown/PNG 行为不变. Gates: `tests/unit/test_sweep_schema_base_consolidation.py` + CSV/HDF5 golden + audit presence + phase 2e JSON/HDF5 caller tests + phase 2f holdout contract + phase 2g V70/V71 numeric gate. **收口判定**: T81 closed; optional `ReconMetricRow` adoption = future only after new fixture need, no blocker. **禁止边界**: V70/V71 add_noise + bucket sweep 数值结果一字节不变 (CSV/HDF5 输出 fixture 字节级稳定); V72 test perf budget 不退 (in-process; phase 2d full unit 1351 passed/10 skipped in 590.13s); 历史 report artifact (HDF5/CSV/Word 文件名 + sheet 名) 向后兼容; ⊥ 新 production CSV cache/data writer; 不删 `*Row` dataclass field; 不动 dataset generator (`mesh_info` / `sample_*`) 输出 schema (V67 HDF5 default). | V67,V70,V71,V72,V73,V75,T75 |
| T82 | x | T79 phase 2 → persistent disk cache unification. **已落地 phase 1..5**: #1 shared `pyeidors.cache.disk_artifacts` manifest/key core (`95c6ccd`); #2 HDF5 writer + DOLFINx mesh-cache writer embed `artifact_key`/`artifact_manifest`, legacy HDF5/metadata reader in-memory backfill (`f6220e6`); #3 `mesh_provenance` cross-layer subkey + HDF5 `subkey_payloads` / DOLFINx auto subkey (`ff0b14f`); #4 schema audit/report `docs/code-fusion/T82_disk_artifact_manifest_schema_audit.md` + gate test, records integrated `hdf5-artifact` / `dolfinx-mesh-cache` vs future-scope candidates (`22d926f`); #5 governance registry landed: `DISK_ARTIFACT_KIND_POLICIES` marks `hdf5-artifact` / `dolfinx-mesh-cache` integrated, `adios4dolfinx-checkpoint` / `adios2-vtx-side-artifact` / `cache-manager-disk-object` / `mesh-cache-layer` future-scope, `legacy-npz-artifact` read-only; `build_disk_artifact_key` / `build_disk_artifact_manifest` reject future/unknown/read-only kinds until new task changes policy + gates. **收口判定**: T82 closed; optional integrations become future tasks, not current HDF5/DOLFINx blocker. **禁止边界**: V36/V62/V65/V66/V67 字段定义不动; 旧 artifact 仍可读; device/backend 不进 math signature; ADIOS2 optional import; 不强行统一 format-divergent code if abstraction leaks; legacy `.npz` remains read-only compatibility only. | V16,V17,V36,V62,V65,V66,V67,V75,T79 |
| T83 | x | Path C → graph/cell-difference operator fusion. **已落地**: 新 `pyeidors.inverse.prior._graph_core` 持 weight validation, voxel edge, shared-facet edge, DOLFINx facet-adjacent edge, `difference_from_edges`, `laplacian_from_edges`, cell volume helpers. `prior/laplace.py` 改 thin façade 调 core; legacy `regularization/smoothness.py::_cell_difference_operator` 改 wrapper 调 `dolfinx_cell_difference_operator`. 保 `graph_laplacian`/`graph_difference_operator`/`graph_ltl`/`_cell_difference_operator` public/test import surface. Gate: `tests/unit/test_graph_operator_core_consolidation.py` 锁 Laplace/graph_ltl 数值、fake-DOLFINx topology parity、empty-edge fallback identity、wrapper source no duplicate row assembly; related prior/RM/regularization tests passed. | V28,V35,V49,V75,T61,T62 |
| T84 | x | Path C → workflow result assembly + difference projection. **phase 1 已落地**: `workflows.base.merge_workflow_metadata(*maps)` 顺序覆盖 + `build_reconstruction_result(...)` 统一 `compute_residuals`/`ReconstructionResult` 装配. `absolute.py` / `difference.py` / `sparse_bayesian.py` 尾部装配迁 helper；public workflow fn 签名不变；metadata precedence 保持 (GN: base→user; sparse: base→user→solver_output.metadata)；module-level `compute_residuals` monkeypatch 面由 `residual_fn` 注入保留. **phase 2 已落地**: 抽 `resolve_difference_vectors(..., simulated_measurement_space='difference'|'raw')`；GN `simulated_measurement` 已是 difference-space → 不再投影，sparse `simulated_measurement` 仍 raw forward measurement → helper 内投影. Gate: `tests/unit/test_workflow_result_assembly.py` + workflow wrapper/sparse workflow tests 锁 precedence、residual injection、fallback simulated path、difference-space no-project、raw-project. | V33,V35,V49,V75,T76,T77 |
| T85 | x | Path C → workflow guard/fallback. **phase 1 已落地**: 抽 `workflows.base.require_initialized(message=...)`, `require_solver_output(owner=...)`, `resolve_simulated_or_forward(...)`. `absolute.py`/`difference.py` init guard 迁 helper 但保原 `"not initialized"` 文案；sparse init guard 迁 helper 但保 `"must be initialised"` 文案；sparse `SolverOutput` type guard + sparse-only simulated fallback 迁共享 helper. **phase 2 已落地**: 抽 `forward_measurement_vector(fwd_model, conductivity_image)`；absolute/difference 裸 `fwd_solve(...).meas` 迁 helper；`resolve_simulated_or_forward` fallback 复用同 helper. **final audit**: `docs/code-fusion/T85_workflow_final_audit.md` 判定剩余尾巴 leave: sparse-only factory, baseline/initial policy, metadata key/precedence, module-local monkeypatch surfaces (`compute_residuals`/`difference_measurement`/`project_measurement_vector`), generic solver-output diagnostic wording. **边界守**: ordinary absolute/difference eager `fwd_solve` 行为不变；difference 即使有 preprojected solver simulated 仍保原 eager forward call；未合并 T84 difference projection；public workflow fn 签名/metadata precedence 不变. Gate: helper tests 锁错误文案、type message、provided simulated 不调用 `fwd_solve`、fallback 调用 `fwd_solve`、forward helper 返回 `.meas`; workflow wrapper test 锁 difference eager call + no reproject; focused workflow/sparse tests + full ruff/full unit 过. | V33,V35,V49,V75,T84 |
| T86 | x | Path C → temporal array validation core. **已落地**: 抽 `pyeidors.data._temporal_core`: `as_frame_batch` / `positive_int` / `unit_interval`; `data.temporal_filtering`, `inverse.postprocess.temporal`, `inverse.postprocess.tv` 迁 import alias (`_as_frame_batch`/`_positive_int`/`_unit_interval`) 保 private name + error wording. **边界守**: public temporal/postprocess API 不变；causal MA/EMA 数值不变；TV PDHG ROI/metadata 不变；RM online hot path 不引入 forward/Jacobian. Gate: `tests/unit/test_temporal_core_consolidation.py` 锁 alias identity, vector/batch behavior, error messages; existing measurement temporal + temporal TV + TV postprocess tests pass. | V28,V33,V37,V40,V49,V75,T21,T63 |
| T87 | x | Path C → HDF5/report JSON-ready core. **已落地**: 抽 `pyeidors.io._json.json_ready`; `io.hdf5_artifacts._json_ready`, `scripts/mesh_tools/matlab_mesh_hdf5._json_ready`, benchmark/report script `_json_ready`/`_jsonable`/`jsonable` 迁 alias (`benchmark_dual_model_rm_v1`, `benchmark_dynamic_validation`, `benchmark_dynamic_tv_huber_sweep`, `benchmark_lazy_48e_cuda_runtime`, `review_dynamic_eidors_metrics`, `gallery_shared`). **边界守**: HDF5 `metadata_json` 字节行为不变 (`json.dumps(..., sort_keys=True)` 调用点不动); Path/Mapping/list/tuple/`np.ndarray`/`np.generic` 递归转换不变; MATLAB mesh bridge CLI 仍可从 repo script path import. Gate: `tests/unit/test_hdf5_json_ready_consolidation.py` 锁 alias identity + payload parity; existing HDF5 artifact + MATLAB mesh bridge tests pass. | V65,V67,V75,T50,T55,T56,T82 |
| T88 | x | Hot-path efficiency sweep #1. **已落地**: `inverse.postprocess.temporal.moving_average_frames` cumsum vectorize; `data.temporal_filtering._moving_average` cumsum + prior-tail state preserved; `JacobianLinearization.__post_init__` pre-stack `_adjoint_blocks` for `matvec`/`rmatvec`/`to_dense`/`hessian_diag`; `refine_tv_pdhg` prealloc `previous`/`x_new` + `np.clip(out=)`/`np.copyto`; `data.difference.build_difference_frames` shared batch core + `reconstruction_matrix._normalize_time_difference_frames` uses it. **rejected**: `object_signature` id-only cache (stale under in-place mutation); added V76 regression gate instead. Gates: focused temporal/Jacobian/RM/TV/signature tests + full unit pass (`1380 passed, 10 skipped`, commit `cf8b681`; prior temporal/Jacobian commit `5e5deca`). | V7,V8,V33,V37,V39,V49,V72,V75,V76,T86 |
| T89 | x | Low-risk Path C / GUI interop JSON helper: `src/eit_app/interop/bridge_package.py::_json_default` delegate to `pyeidors.io._json.json_ready` while preserving unknown-object `TypeError`; scan GUI JSON writers (`frame_database.py`, services/env payloads) but only migrate semantically identical Path/ndarray cases. Gate: bridge package JSON byte/parity fixture + T87 helper tests; no app DB schema drift. | V67,V75,T87 |
| T90 | x | Hash helper audit before schema-touching changes: inventory `hashlib.sha256(arr.tobytes())` sites; classify into cache-key/schema/golden/report-only. Do **not** mechanical-replace with `cache.keys.hash_array` unless output hash contract intentionally changes + schema/version bump/golden update. Include V76 check for any semantic cache memoization. Deliver audit report + candidate split. | V36,V62,V65,V67,V76,T79,T82 |
| T91 | x | Mesh cache cold-start preflight / lazy heavy import. Goal: `load_or_create_mesh` returns disk/process cached mesh with minimal avoidable work; explore moving cache-name/preflight before expensive generation path and lazying optional `gmsh`/DOLFINx imports only if cache miss. Gate: cached 2D/3D mesh load tests + import/cold-start benchmark; preserve V66/V68 cache naming, sidecar validation, CEM completeness checks. | V62,V66,V68,V72,T78,T79,T82 |
| T92 | x | PETSc matrix template reuse / `SUBSET_NONZERO_PATTERN` experiment. Single task + benchmark gate only: prove conductivity block sparsity pattern stable across sigma for target CEM routes; reuse template/preallocation with `Mat.axpy(..., SUBSET_NONZERO_PATTERN)` (M and K are subsets of the union template, not equal to it; `SAME_NONZERO_PATTERN` was tried first but routes values incorrectly when source pattern is a strict subset of dest) only where safe. Gate: numeric parity, PETSc diagnostics, CUDA/CPU forward benchmark, fallback to current `DIFFERENT_NONZERO_PATTERN` when gauge/CEM pattern uncertain. | V1,V13,V14,V23,V45,V47,V72 |
| T93 | x | Public API sync: keep top-level `pyeidors` façade limited to `EITSystem`/`check_environment`/`__version__`; add import/`__all__` gate for top-level vs subpackage exports; fix `pyeidors.inverse.jacobian.compute_sigma_fingerprint` re-export | V77,B17 |
| T94 | x | GUI launcher inventory sync: repository-root `EIT-GUI-CPU.cmd` / `EIT-GUI-GPU.cmd` wrappers verified against `scripts/gui/run_eit_app.ps1`; §I lists actual `.sh`/`.ps1`/root `.cmd` surfaces only | V79 |
| T95 | x | Add `scripts/cache/migrate_artifacts_to_hdf5.py --root <path> --dry-run|--apply [--manifest <path>]`; wraps `migrate_npz_to_hdf5`, supports legacy `.npz/.npy`, emits JSON manifest, leaves source files untouched | V65,V67,V78,V79 |
| T96 | x | Update `docs/MEASUREMENT_DATA_SPEC.md` to recommend HDF5 `.h5/.hdf5` package/default; `.npz/.npy` described only as legacy/test/read/migration adapters; add doc guard | V65,V67,V78 |
| T98 | x | Unified reconstruction CLI surface synced: planned v1 flags (`--algorithm`, `--dual-mesh`, `--coarse-mesh`, `--rm-cache`, `--normalize-difference`, `--greit-targets`, `--greit-metrics-out`, `--bad-channel-mask`) remain explicitly pending in §I.future and are not accepted by current runner; parser guard locks actual `--method` + acceleration/cache surface | V77,V79 |
| T99 | x | GUI simulation inverse inventory/rename: replace stale method list (`eidors_one_step_noser`, `eidors_abs_gn`, `eidors_demo3d_tv`) with route labels matching SPEC: `noser_rm`, `laplace_rm`, `greit3d_rm`, `absolute_gn`, `debug_fine_mesh_noser`, `debug_full_gn`; legacy `eidors_abs_gn` normalizes to `absolute_gn`; i18n/tooltips state cold-build vs hot-path cost, absolute-vs-difference semantics, and artifact need | V26,V27,V28,V37,V50,V64,V84,V116,B22,B50 |
| T100 | x | Wire 2D/3D simulation NOSER RM default: build/load one-step RM artifact (HDF5) from current forward protocol, coarse inverse mesh/grid, bad-channel/W/difference signature; online reconstruction uses `RM @ Δv` or `RM @ normalize(Δv)`, not fine-mesh dense `JᵀJ` solve; keep fine-mesh path only under debug method | V26,V27,V33,V34,V35,V36,V37,V38,V65,V84 |
| T101 | x | Add Laplace/curvature RM simulation route: EIDORS 2× graph-Laplacian / squared-Laplacian prior on inverse mesh; UI exposes smooth RM option; cache signature invalidates prior semantic drift; regression asserts Laplace and curvature do not collapse to identical RM | V28,V35,V36,V37,V61,V83,V84,V91 |
| T102 | x | Wire 3D GREIT artifact route in GUI simulation: choose/resolve only EIDORS-parity non-fixture artifact matching current 3D geometry/protocol, load HDF5, run online `rm_matmul`; deterministic common-config warm fixtures remain test-only; use model-component `Y/D/rec_model` for masked geometry + boundary-voltage fit when present | V50,V55,V56,V57,V58,V59,V60,V61,V62,V63,V64,V65,V84,V91,T97 |
| T103 | x | Fix inverse UI hyperparameter semantics: rename α control to `hp` or `λ_eff` by selected route; one-step EIDORS formulas use `hp² RtR`; if fast path uses canonical `λ_eff=1e-2`, UI shows locked value/diagnostic and does not pretend user α applied | V26,V27,V38,V59,V84 |
| T104 | x | GUI route parity tests: simulation method selection asserts correct metadata (`rm_artifact_path`/`greit_common_config`/debug flag), RM routes report `online_hot_path=rm_matmul` and zero forward/Jacobian/KSP rebuild; dense fine-mesh route labelled debug; screenshot/numeric smoke catches fragmented default NOSER regression | V37,V46,V50,V64,V84,B22 |
| T105 | x | GREIT artifact registry + config-driven auto builder: implement `pyeidors.inverse.greit_registry` manifest/signature resolver; build on existing `src/pyeidors/inverse/greit.py` primitives (`build_greit3d_distribution`, `build_greit_finite_target_responses`, `build_greit_desired_images`, `calc_greit_rm`, `GREITRM.save/load`). GUI `greit3d_rm` computes V92 signature from `ForwardModelConfig` + actual stim/meas/channel vectors; exact HDF5 hit loads, miss queues/execs native builder to generate finite-target `vh/vi/Y/D/RM/rec_model` artifact then registers. Add optional Windows MATLAB/EIDORS backend script calling `GREIT3D_distribution` + `mk_GREIT_model` and importing/exporting official HDF5 parity artifact. Gates: mutate each V92 signature axis → miss; warm hit → same artifact; no UI-thread long build; no deterministic fixture fallback; stored `Y/D/rec_model` drives fit + geometry | V50,V55,V56,V57,V58,V59,V60,V61,V62,V63,V65,V87,V91,V92,T97 |
| T106 | x | Fix GUI 3D non-NOSER RM/GREIT visibility regression: Laplace/curvature auto-build use param-form official RtR solve/signature; GREIT native builder scales `target_size` by tank radius, masks rec volume cylindrically, bumps builder version, and renders rec-center hexa geometry from axis spacing | V91,V92,V93,B30 |
| T107 | x | Add simulation Step4 advanced custom `λ_eff` entry for production RM routes: unchecked state keeps canonical locked `λ_eff=1e-2`; checked state enables input, records custom lambda semantics, and forces a distinct RM cache/artifact signature so first run cold-builds the requested RM | V26,V27,V37,V84,V117 |
| T108 | x | Expose GUI GREIT advanced params: desired image mode (`center`/`gauss`/`adaptive_gauss`/`sobol_qmc`), training target count, target-size fraction, weight/NF, cache/rebuild toggles; feed exact registry config/metadata and registry cache policy; warn cold-build cost | V92,V121,V123,V124 |
| T109 | x | Add first-stage cache ops CLI: package `pyeidors.cache.ops/cli`, expose `eit-cache` + repo-local `./eit-cache`, keep `scripts/cache/cache_ctl.py` compatibility wrapper, add JSON `doctor/stats/gc/warm` reports over `CacheManager` + GUI backend worker/FFCx caches, and document operational commands | V149 |
| T110 | x | Add mesh-derived HDF5/process artifact layer: `pyeidors.geometry.derived_cache` builds/loads content-addressed `node_coords/cell_connectivity/cell_centers/cell_measures`, `EITMesh` reuses one in-process derived bundle for repeated cells/centers/measures calls, `eit-cache doctor` reports artifact counts, and docs describe the cache role | V150 |
| T111 | x | Add GUI NumPy array geometry process cache: `eit_app.ui.array_geometry_cache` derives `cell_centers` from raw result arrays once per content signature; 3D conductivity widget and simulation metrics panel reuse it; tests cover hit/miss/mutation/invalid-connectivity behavior | V151 |
| T112 | x | Make `pyeidors.inverse` / `pyeidors.inverse.jacobian` public export surfaces lazy: replace eager package imports with module map + `__getattr__`, preserve `__all__`, keep lightweight dual-mesh / fingerprint imports usable without loading GN/GREIT/workflow/Jacobian calculator stacks, and add import-surface regression test | V152 |
| T113 | x | Make geometry import surface lazy: replace `pyeidors.geometry` eager exports with module map + `__getattr__`, add `_runtime` lazy MPI/XDMF helpers, defer `mpi4py` in mesh generator/converter/cache paths until actual mesh IO, and extend lazy-import regression tests | V153 |
| T114 | x | Make top-level `pyeidors` façade lazy: move DOLFINx/Torch/CUQI probes from import time to cached `check_environment()` / compatibility private flag access, preserve `EITSystem` lazy import and top-level API contract, and add subprocess regression for no top-level heavy imports | V154 |
| T115 | x | Add cache/import observability: `pyeidors.cache.ops.summarize_import_health` and `eit-cache doctor/stats` expose lightweight-import health/timings/heavy-module list so startup regressions appear in operational cache reports | V155 |
| T116 | x | Add GUI array-geometry cache observability: expose JSON-safe stats/entry metadata from `eit_app.ui.array_geometry_cache`, include process-local snapshot in `eit-cache doctor/stats`, and document process-local semantics | V156 |
| T117 | x | Harden forward package import surface: convert `pyeidors.forward` to shared lazy export map with global caching, extend import-health targets/sentinels to include forward/PETSc modules, and add subprocess regression for no eager forward runtime imports | V157 |
| T118 | x | Make forward scalar support PETSc-lazy: move `petsc4py` import behind cached `_petsc_module()`, keep `PETSc` monkeypatch compatibility, and add subprocess regression that `import pyeidors.forward.complex_support` leaves `petsc4py` unloaded | V158 |
| T119 | x | Make data package import surface lazy: replace eager `pyeidors.data` imports with export/submodule maps, preserve public `__all__` and submodule import behavior, extend import-health sentinels to data-heavy modules, and add subprocess regression | V159 |
| T120 | x | Make perf package import surface lazy: replace eager `pyeidors.perf` imports with export/submodule maps, keep policy constants and capability submodule compatibility, extend import-health to perf/scipy/gpu-kernel sentinels, and add subprocess regression | V160 |
| T121 | x | Make visualization package import surface lazy: defer `eit_plots`/Matplotlib/DOLFINx/UFL/MPI until `EITVisualizer` or `create_visualizer` access, extend import-health visualization sentinels, and add subprocess regression | V161 |
| T122 | x | Make inverse solver package import surface lazy: replace eager `pyeidors.inverse.solvers` imports with export/submodule maps, keep GN/matrix-free/sparse Bayesian symbol compatibility and submodule imports, extend import-health solver sentinels, and add subprocess regression | V162 |
| T123 | x | Make io package import surface lazy: defer `hdf5_artifacts`/`h5py` until HDF5 API symbol or submodule access, extend import-health HDF5 sentinels, and add subprocess regression | V163 |
| T124 | x | Make regularization package import surface lazy: defer base/smoothness implementations until class/submodule access, keep public `__all__`, extend import-health DOLFINx/SciPy/Jacobian regularization sentinels, and add subprocess regression | V164 |
| T125 | x | Make prior package import surface lazy: defer Laplace/RtR/TV-IRLS implementations until function/class/submodule access, keep public `__all__` and submodule imports, extend import-health prior sentinels, and add subprocess regression | V165 |
| T126 | x | Make postprocess package import surface lazy: defer temporal/TV implementations until function/class/submodule access, keep public `__all__` and submodule imports, extend import-health postprocess sentinels, and add subprocess regression | V166 |
| T127 | x | Make reduced package import surface lazy: defer lowrank/POD/reduced-step/snapshot implementations until function/class/submodule access, keep public `__all__`, extend import-health reduced sentinels, and add subprocess regression | V167 |
| T128 | x | Make matrix-free package import surface lazy: defer dual-mesh operator implementation until class/submodule access, keep public `__all__`, extend import-health matrix-free sentinels, and add subprocess regression | V168 |
| T129 | x | Make workflows package import surface lazy: defer absolute/difference/sparse workflow implementations until function/class/submodule access, keep public `__all__`, extend import-health workflow/sparse sentinels, and add subprocess regression | V169 |
| T130 | x | Make femx package import surface lazy: defer DOLFINx/UFL helper implementation until helper/submodule access, keep public `__all__`, extend import-health FEM sentinels, and add subprocess regression | V170 |
| T131 | x | Make interop package import surface lazy: defer SciPy/FEM exchange implementation until helper/submodule access, keep public `__all__`, extend import-health interop sentinels, and add subprocess regression | V171 |
| T132 | x | Make cache package import surface lazy: defer NumPy-backed keys/types, cache manager, object signatures, and store backends until symbol/submodule access; extend import-health cache sentinels and add subprocess regression | V172 |
| T133 | x | Make physics package import surface lazy: defer NumPy current-drive/unit-consistency implementations until symbol/submodule access, keep unit-consistency type access from loading current-drive, extend import-health physics sentinels, and add subprocess regression | V173 |
| T134 | x | Normalize electrodes package lazy surface: map pattern manager/submodules through `__getattr__`, expose `layout`/`patterns` in `dir()`, extend import-health electrodes sentinels, and add subprocess regression | V174 |
| T135 | x | Add byte-budgeted RM artifact process cache and fit-Jacobian restore: skip retaining oversize RM artifacts, skip oversize persisted fit J without rebuilding, expose diagnostics, and cover both branches in cache regression tests | V175 |
| T136 | x | Add HDF5 streaming RM matmul for oversize artifacts: inspect lazy RM shape/dtype, stream CPU/auto requests in chunks under process-cache budget, skip full RM load/handle prep, expose streaming diagnostics, and add regression proving full loader unused | V176 |
| T137 | x | Make RM HDF5 artifacts streaming-friendly at write time: write `rm` with row-block/full-width chunks, persist chunk-layout metadata, document cache behavior, and add HDF5 roundtrip/layout tests | V177 |
| T138 | x | Align HDF5 RM streaming reads to dataset chunk rows when they fit the runtime byte budget, expose read-chunk diagnostics, and extend the streaming regression to prove chunk-row batching | V178 |
| T139 | x | Remove large HDF5 checksum byte-copy spike: stream `_array_digest` payload bytes via memoryview while preserving legacy digests, and add regression covering numeric/string parity plus no `tobytes` in digest body | V179 |
| T140 | x | Switch RM HDF5 artifact default compression to fast `lzf`, persist compression metadata, and assert RM-specific compression without changing GREIT/general gzip large-cache tests | V180 |
| T141 | x | Reuse HDF5 array digest payloads across manifest and dataset attributes so large artifacts hash each array once per write, and add a regression with counted digest calls | V181 |
| T142 | x | Apply fit-Jacobian byte budget during RM artifact writes: skip oversize J persistence, mark metadata, avoid immediate/warm voltage-fit overlay, and prevent rebuild loops for intentionally fitless artifacts | V182 |
| T143 | x | Keep GREIT `Y/D` auxiliary matrices lazy in the HDF5 RM lightweight loader, so streaming RM apply reads only the RM row chunks unless boundary-fit projection asks for training matrices | V183 |
| T144 | x | Avoid eager `rec_model` loads in the HDF5 RM lightweight loader when explicit node/cell geometry exists, and extend the lazy auxiliary regression to cover rec-model fallback avoidance | V184 |
| T145 | x | Extend `eit-cache warm` into a practical 3D preflight: accept `--repair-jit`, repair stale FFCx locks before worker warm, and include profile/full worker cache summaries in warm JSON output | V185 |
| T146 | x | Keep one HDF5 file handle open across RM streaming row-block reads, expose `rm_hdf5_file_open_mode`, and add regression that fails if streaming falls back to per-chunk `HDF5LazyDataset.__getitem__` | V186 |
| T147 | x | Add RSS observability and byte-budget recycling for persistent backend workers so large 3D solves can free process-resident DOLFINx/PETSc heaps after result write while preserving warm workers below budget | V187 |
| T148 | x | Add visualization-only point-cloud sampling budget for 3D viewers: cap PyVista rendered points, keep anomaly points first, retain deterministic background coverage, and expose sampled/original counts on the widget | V188 |
| T149 | x | Keep large auto-point-cloud 3D payloads on PyVista offscreen when embedded VTK is unavailable, using sampling to control render load and an unavailable caption when offscreen cannot render | V189 |
| T150 | x | Keep large point-cloud sampling O(n): preserve value-based anomaly candidates before sampling but defer spatial coherence / `cKDTree` work to the sampled display set | V190 |
| T151 | x | Avoid eager dtype promotion in 3D display entrypoint: preserve `float32` coordinate/sigma arrays and `int32` connectivity through render dispatch, with conversions only where a backend explicitly needs them | V191 |
| T152 | x | Avoid hidden int32→int64 expansion in GUI array-geometry cache: preserve integer connectivity dtype through signature/center derivation and cover cache entry metadata | V192 |
| T153 | x | Replace `coords[cells].mean(axis=1)` in GUI array-geometry cache with low-peak per-vertex accumulation, keeping float32 centers and avoiding `(cells,verts,dims)` temporary | V193 |
| T154 | x | Add import-only persistent backend runtime prime to `warm`: move DOLFINx/PETSc/PyEIDORS forward-stack imports into idle worker warmup, expose prime metadata in `eit-cache warm`, and keep 3D prewarm from running a full solve | V194 |
| T155 | x | Make backend worker entrypoint lazy-light so process start does not import protocol/controllers/solver modules until a request or runtime prime explicitly needs them | V195 |
| T156 | x | Switch backend worker HDF5 IPC array compression from gzip to configurable fast `lzf`, preserving HDF5 protocol semantics while reducing large 3D result write/read latency | V196 |
| T157 | x | Reduce simulation metrics post-solve memory: bypass nearest-neighbor index for same-geometry comparisons and preserve float32/int32 arrays while deriving metric samples | V197 |
| T158 | x | Replace display `cell_to_node_average` expanded-repeat scatter with low-peak per-vertex scatter-add, preserving float32 outputs for large 3D display meshes | V198 |
| T159 | x | Make backend worker HDF5 protocol import controller-lazy so forward-only workers do not import reconstruction controller/solver stacks during protocol load | V199 |
| T160 | x | Preserve int32 connectivity through GUI boundary/projection triangle extraction and avoid redundant boolean-filter copies for already-valid triangle meshes | V200 |
| T161 | x | Replace 3D display `sigma[cells].mean(axis=1)` with low-peak streamed cell-mean computation for point-scalar render paths | V201 |
| T162 | x | Remove full-range/background index allocations from capped 3D point-cloud sampling by generating budget-sized evenly spaced range/background-rank indices directly | V202 |
| T163 | x | Preserve float32 sigma dtype through 3D point-cloud sampling/anomaly detection so display-only thresholding does not allocate a full float64 copy | V203 |
| T164 | x | Avoid redundant full anomaly-index allocation in point-cloud sampling by making spatial coherence return before `flatnonzero` when centers are absent | V204 |
| T165 | x | Replace eager finite-score subset copies in `_cell_anomaly_mask` with mask+where peak/count stats and crowded-only percentile materialization | V205 |
| T166 | x | Remove eager float64/finite-subset copies from all-finite 3D conductivity color-limit calculation | V206 |
| T167 | x | Make PyVista point-cloud display arrays sample before backend preparation and preserve float32 sampled subsets | V207 |
| T168 | x | Route `_cell_center_sigma` point-scalar aggregation through low-peak streamed cell means instead of `values[cells]` expansion | V208 |
| T169 | x | Slice spatial-highlight candidate centers before float64 contiguous preparation for `cKDTree` | V209 |
| T170 | x | Preserve float32 RM artifact/display geometry through RM artifact load, auto-build context, and reconstruction-controller geometry extraction | V210 |
| T171 | x | Replace `CellMesh.cell_centers()` expanded coordinate indexing with streamed per-vertex accumulation | V211 |
| T172 | x | Replace mesh-derived cache center/measure full cell-point expansion with low-peak per-cell/per-vertex derivation | V212 |
| T173 | x | Stream dual-mesh generic `_cell_centers(mesh)` fallback instead of expanding `coords[cells]` | V213 |
| T174 | x | Stream dual-mesh cell locator bounding boxes and gather only candidate simplex vertices | V214 |
| T175 | x | Replace `VoxelGrid.cell_centers()` meshgrid/stack allocation with direct column-fill output generation | V215 |
| T176 | x | Replace `build_greit3d_distribution` meshgrid/stack candidate generation with direct x-fastest center matrix fill | V216 |
| T177 | x | Replace GREIT `_metric_centers` meshgrid/stack fallback with direct C-order center matrix fill | V217 |
| T178 | x | Replace GREIT `_gauss_reference_offsets` meshgrid/stack quadrature helper with direct Cartesian offset/weight generation | V218 |
| T179 | x | Replace GREIT `_default_radius` all-pairs distance matrix with cKDTree nearest-neighbor radius estimation | V219 |
| T180 | x | Replace GREIT `_nearest_center_distance` all-pairs spacing fallback with the shared cKDTree nearest-neighbor helper | V220 |
| T181 | x | Stream GREIT adaptive-gauss boundary distances per target instead of holding an all-target center-distance matrix | V221 |
| T182 | x | Stream GREIT desired-image sample distances through reusable 1D work buffers instead of sample-target coordinate matrices | V222 |
| T183 | x | Preserve float32 and vectorize/direct-fill reconstruction-controller GREIT center-cloud hexa/quad display geometry | V223 |
| T184 | x | Replace GREIT Gauss/Sobol/adaptive desired-image samples tensor construction with offset-streamed weighted sigmoid averaging | V224 |
| T185 | x | Stream GREIT center desired-image target distances per output column and remove the unused production samples-tensor helper | V225 |
| T186 | x | Replace GeomV2 hex O-grid core `np.meshgrid` allocation with broadcast views while preserving structured geometry output | V226 |
| T187 | x | Replace GUI metrics nearest-resample brute-force fallback all-pairs distance tensor with a streamed one-target search | V227 |
| T188 | x | Replace lazy linearized Hessian diag chunk broadcast multiplications with direct weighted `einsum` vector reduction | V228 |
| T189 | x | Make eager `JacobianLinearization.to_dense()` fill dense block slices in-place instead of assigning a multiplied temporary block | V229 |
| T190 | x | Route fast GN dense Jacobian measurement weighting through a weighted action bundle instead of copying a full weighted dense J | V230 |
| T191 | x | Replace Woodbury dense `J * inv_diag[None,:]` temporary with column-block small-system accumulation | V231 |
| T192 | x | Replace hardware reconstruction widget interpolation-grid `np.meshgrid` with direct `sample_points` column fill | V232 |
| T193 | x | Stream deterministic GREIT common-config fixture RM generation by row blocks instead of full broadcast temporaries | V233 |
| T194 | x | Project normalized difference Jacobians with one output buffer and in-place ufuncs | V234 |
| T195 | x | Build batch difference frames in one output buffer and avoid safe-reference copies when no clamp is needed | V235 |
| T196 | x | Share strided axis-column fill for VoxelGrid and GREIT cartesian center generation | V236 |
| T197 | x | Replace GREIT weighted-centroid broadcast matrix with vector-matrix reduction | V237 |
| T198 | x | Build cuda_structured Jacobi diagonal inverse on torch device instead of NumPy column temporary | V238 |
| T199 | x | Route phantom/dof squared-distance masks through shared one-work-vector helper | V239 |
| T200 | x | Replace digit/holdout fallback parameter-grid meshgrid with shared direct output fill | V240 |
| T201 | x | Share holdout/bucket weighted centroid+covariance helper without broadcast matrix temporaries | V241 |
| T202 | x | Build bucket source gradients in final output buffer without `diff` matrix and `r2[:,None]` broadcast | V242 |
| T203 | x | Route bucket source-potential squared distances through shared helper | V243 |
| T204 | x | Fill EIDORS noise row-reference broadcast output directly instead of `v2[:,None]` broadcast path | V244 |
| T205 | x | Add CacheManager per-key single-flight lock for concurrent miss coalescing | V245 |
| T206 | x | Stream cache key file/array payload digests without full `read_bytes()` / `tobytes()` copies | V246 |
| T207 | x | Sync T90 SHA256 audit inventory after real/complex sigma fingerprint split | V247 |
| T208 | x | Route real/complex sigma fingerprints through byte-stable streaming array payload digest helper | V248 |
| T209 | x | Sync GN runtime contract-freeze test with `_JacobianActionBundle.hessian_diag` action field | V249 |
| T210 | x | Route GN linear-system sparse/dense/ROM cache hashes through byte-stable streaming payload helper | V250 |
| T211 | x | Route remaining CUDA/reduced/sparse/GREIT array digest sites through byte-stable streaming payload helper | V251 |
| T212 | x | Remove WSLg/headless PyVista offscreen bypass for auto point-cloud 3D payloads while preserving display sampling | V252 |
| T213 | x | Add byte-budget eviction to GUI array-geometry process cache | V253 |
| T214 | x | Stream disk-cache pickle/gzip payload IO without whole-object byte buffers | V254 |
| T215 | x | Route GUI reconstruction RM mesh-signature hashing through shared streaming array digest updater | V255 |
| T216 | x | Route GN difference CLI background-sigma cache hash through streaming payload helper | V256 |
| T217 | x | Avoid pickle serialization in process-cache size estimates for array-backed objects | V257 |
| T218 | x | Route forward model mesh/scalar cache hash helpers through streaming array digest updater | V258 |
| T219 | x | Route remaining inverse/RM/GREIT signature digest helpers through shared streaming array payload updater | V259 |
| T220 | x | Copy acquisition ring-buffer frames through shared-memory views without intermediate bytes buffers | V260 |
| T221 | x | Route benchmark/diagnostic script large-array hash payloads through shared streaming helpers | V261 |
| T222 | x | Route GUI array-geometry cache signature shape/payload hashing through streaming updater | V262 |
| T223 | x | Direct-fill vector-to-frame broadcast helpers without `broadcast_to(...).copy()` temporaries | V263 |
| T224 | x | Direct-fill GREIT repeated cell-extent matrices without `broadcast_to(...).copy()` temporaries | V264 |
| T225 | x | Direct-fill dynamic sequence bad-channel mask vector expansion without broadcast view copy | V265 |
| T226 | x | Stream GUI/dataset 3D volume-fraction painting vertices in chunks instead of expanding all cell vertices | V266 |
| T227 | x | Chunk script measurement-space `J diag(scale) J.T` builds instead of allocating full scaled Jacobian | V267 |
| T228 | x | Direct-fill hardware reconstruction sample grid columns without tile/repeat temporaries | V268 |
| T229 | x | Replace script full weighted-J temporaries with row-weighted actions and streaming weighted-J hashes | V269 |
| T230 | x | Remove unused whole-object pickle helpers from disk cache store | V270 |
| T231 | x | Chunk GN difference ROM `U.T @ diag(R) @ U` builds instead of allocating full scaled basis | V271 |
| T232 | x | Route 3D benchmark/diagnostic anomaly distance masks through one-work-vector helper | V272 |
| T233 | x | Direct-fill GUI planar quad projection source indices without repeat expansion | V273 |
| T234 | x | Direct-fill hardware reconstruction barycentric weight columns without column_stack temporary | V274 |
| T235 | x | Direct-fill native complex LinearOperator regularization dense matrix without eye/column_stack temporaries | V275 |
| T236 | x | Direct-fill GN ROM synthetic snapshot matrices without column_stack temporaries | V276 |
| T237 | x | Direct-fill reduced snapshot bank stack/dedupe matrices without column_stack temporaries | V277 |
| T238 | x | Direct-fill reduced POD basis merge blocks without column_stack temporaries | V278 |
| T239 | x | Direct-fill reduced GN regularized basis projection without column_stack temporaries | V279 |
| T240 | x | Direct-fill sparse Bayesian grouped coarse matrix without column_stack temporaries | V280 |
| T241 | x | Direct-fill dual-mesh dense materialization without eye/column_stack temporaries | V281 |
| T242 | x | Direct-fill GREIT finite-target and contracted training response matrices without column_stack temporaries | V282 |
| T243 | x | Stream GREIT finite-target conductivity distance masks through reusable work vectors | V283 |
| T244 | x | Stream GREIT target generation and equivalent-ball masks through squared-distance work vectors | V284 |
| T245 | x | Precompute GUI 3D hex volume sample weights and direct-fill 3D electrode patch geometry without column_stack | V285 |
| T246 | x | Direct-fill GREIT 2D-to-XYZ point padding without column_stack temporaries | V286 |
| T247 | x | Direct-fill traditional Jacobian affine/assembly/projection matrices without stack temporaries | V287 |
| T248 | x | Direct-fill RtR prior dense materialization without eye/column_stack temporaries | V288 |
| T249 | x | Stream FEMx cell midpoint and radius helpers without expanded coordinate temporaries | V289 |
| T250 | x | Direct-fill matrix-free Jacobian adjoint gradient blocks without stack temporaries | V290 |
| T251 | x | Direct-fill adjoint Jacobian Torch assembly gradient blocks without stack temporaries | V291 |
| T252 | x | Direct-fill dynamic Kalman and rowwise RM frame-row matrices without vstack temporaries | V292 |
| T253 | x | Direct-fill temporal TV postprocess refined frame rows without vstack temporaries | V293 |
| T254 | x | Direct-fill TV-IRLS batch frame rows without vstack temporaries | V294 |
| T255 | x | Direct-fill GREIT finite-target xyzr and distribution bounds without vstack temporaries | V295 |
| T256 | x | Direct-fill mesh-derived tetra determinant matrices without vstack temporaries | V296 |
| T257 | x | Direct-fill cross-layer measurement matrices without vstack temporaries | V297 |
| T258 | x | Direct-fill bucket-domain boundary coordinates and boundary/interior nodes without stack temporaries | V298 |
| T259 | x | Direct-fill bucket electrode centers and preallocate dense sensitivity rows without stack temporaries | V299 |
| T260 | x | Direct-fill geometry exchange boundary edge matrices without vstack temporaries | V300 |
| T261 | x | Stream electrode label centroids without vstacking segment points | V301 |
| T262 | x | Direct-fill frame CSV real/imag columns without column_stack temporaries | V302 |
| T263 | x | Direct-fill GUI electrode overlay arc/patch point arrays without stack temporaries | V303 |
| T264 | x | Direct-fill 3D pattern length expansion and measurement selector arrays without tile/concatenate temporaries | V304 |
| T265 | x | Direct-fill sigma/contact normal-system RHS without concatenate temporaries | V305 |
| T266 | x | Direct-fill dynamic spatiotemporal GN and TV/Huber RHS vectors without concatenate temporaries | V306 |
| T267 | x | Direct-fill GUI contact-impedance repeated vectors without tile temporaries | V307 |
| T268 | x | Direct-fill GUI point-cloud selected indices without concatenate temporaries | V308 |
| T269 | x | Direct-fill measurement moving-average resume cumulative input and history tail without concatenate temporaries | V309 |
| T270 | x | Stream GUI boundary-voltage y-range min/max without concatenate temporaries | V310 |
| T271 | x | Direct-fill core inverse small vectors without concatenate/repeat helper temporaries | V311 |
| T272 | x | Stream holdout/bucket sweep plot sigma ranges without concatenate temporaries | V312 |
| T273 | x | Stream cache hash fallback, avoid process-cache pickle size estimation, direct-fill hex core grid | V313 |
| T274 | x | Stream GUI 3D cell-center fallback without materializing coords[cells] | V314 |
| T275 | x | Stream 3D volume-fraction samples one weight row at a time | V315 |
| T276 | x | Enforce RSS budget on backend warm and propagate prime metadata into GUI results | V316 |
| T277 | x | Surface GUI 3D backend worker warm status and per-profile warm reports | V317 |
| T278 | x | Add process-cache admission gate and rejection telemetry for oversized/low-rank L1 entries | V318 |
| T279 | x | Add forward/backend/GUI phase timing metadata so slow 3D first loads can be attributed to import, JIT/setup, solve, HDF5 transport, or visualization update | V319 |
| T280 | x | Add opt-in 3D setup-prime backend prewarm command that warms mesh/static setup/JIT caches without running full forward solves | V320 |
| T281 | x | Expose setup-prime through `eit-cache warm --forward-request` and distinct GUI status messages | V321 |
| T282 | x | Key GUI setup-prime warm reports by stable forward-setup signature instead of profile-only or full simulation-input signature | V322 |
| T283 | x | Run setup-prime under the profile-scoped backend cache lock to avoid concurrent FFCx cache compilation races | V323 |
| T284 | x | Add reusable GUI-style 3D forward first-load benchmark CLI for setup-prime/full-solve phase timing evidence | V324 |
| T285 | x | Cache PETSc CUDA capability probe per backend profile/runtime to reduce repeated setup-prime configure time | V325 |
| T286 | x | Split forward configure timing into subphases and build GUI forward mesh geometry arrays from one connectivity pass | V326 |
| T287 | x | Warm PETSc CUDA capability probe during backend worker `prime_runtime` so GUI worker prewarm removes that runtime cost from later 3D setup/solve requests | V327 |
| T288 | x | Add `--prewarm-worker` benchmark mode to measure setup/solve after GUI-style worker import/capability warm in one persistent worker pool | V328 |
| T289 | x | Surface backend worker PETSc CUDA probe/cache status in GUI warm reports and status-bar messages | V329 |
| T290 | x | Add profile-local PETSc CUDA capability probe cache summaries to `eit-cache doctor/stats/warm` backend worker reports | V330 |
| T291 | x | Bound backend warm and first-load benchmark progress message retention while preserving total/truncated telemetry fields | V331 |
| T292 | x | Add process-local negative capability cache for failed PyVista offscreen 3D rendering, show unavailable caption on cached failure, and remove duplicate 3D display-array work | V332 |
| T293 | x | Enable HDF5 shuffle filter for numeric backend worker IPC arrays while preserving configurable fast `lzf` compression | V333 |
| T294 | x | Omit absent optional backend worker IPC datasets while keeping legacy placeholder reads compatible | V334 |
| T295 | x | Add explicit row-major HDF5 chunk sizing for compressed backend worker IPC numeric datasets | V335 |
| T296 | x | Read backend worker HDF5 numeric datasets directly into final C-order arrays | V336 |
| T297 | x | Preserve single-precision GUI display channel arrays instead of widening 3D result payloads to float64 | V337 |
| T298 | x | Scan GUI complex-mode detection in bounded chunks without copying full finite imaginary subsets | V338 |
| T299 | x | Compute GUI composite display channel in bounded chunks without full magnitude and phase temporaries | V339 |
| T300 | x | Reuse 3D anomaly residual buffer for negative and absolute anomaly score modes | V340 |
| T301 | x | Sample large 3D point-cloud anomaly masks by rank without materializing full anomaly index arrays | V341 |
| T302 | x | Build 3D point-cloud highlight arrays through one helper instead of repeated boolean indexing | V342 |
| T303 | x | Avoid full finite-mask allocation for all-finite 3D conductivity color/anomaly calculations | V343 |
| T304 | x | Reuse candidate anomaly bool mask as the final 3D anomaly mask and reuse existing finite mask for non-finite filtering | V344 |
| T305 | x | Preserve direct score-peak helper 2-tuple compatibility while making anomaly-mask bool reuse opt-in | V344 |
| T306 | x | Replace PyVista 3D grid/electrode `.flatten()` buffer copies with C-order `.ravel()` views | V345 |
| T307 | x | Make shared mesh cell-to-node averaging fill orphan/NaN nodes in-place without finite-subset or whole-array `np.where` copies | V346 |
| T308 | x | Remove equipotential PyVista face-buffer flatten copy and chunk finite warp-span scan | V347 |
| T309 | x | Stream boundary-voltage y-axis min/max with finite-mask reductions instead of finite-subset copies | V348 |
| T310 | x | Replace PyVista volume highlight `np.where(inhom_mask)[0]` extraction with direct `np.flatnonzero` indices | V349 |
| T311 | x | Reuse sorted `np.unique` outputs for GREIT center-cloud axis spacing without re-sort or finite-positive diff subsets | V350 |
| T312 | x | Reuse 3D spatial anomaly nearest-distance buffer instead of copying finite-positive nearest subsets | V351 |
| T313 | x | Preserve boundary-voltage y-range input dtype instead of widening each series to float64 | V352 |
| T314 | x | Preserve 2D conductivity image and projection coordinate dtype instead of widening display arrays to float64 | V353 |
| T315 | x | Preserve boundary-voltage reconstructed overlay dtype before pyqtgraph `setData` | V354 |
| T316 | x | Preserve 3D display face/highlight value dtype instead of widening cell sigma to float64 | V355 |
| T317 | x | Preserve hardware equipotential float32 coords/sigma through widget entry and render dispatch | V356 |
| T318 | x | Preserve hardware reconstruction image float32 coords/sigma through widget entry and cell-to-node averaging | V357 |
| T319 | x | Stream simulation metrics finite-pair statistics without copying finite value subsets | V358 |
| T320 | x | Add all-finite fast path for simulation metrics nearest-neighbor resampling | V359 |
| T321 | x | Preserve batch reconstruction PNG/voltage-fit display arrays without widening float32 payloads | V360 |
| T322 | x | Preserve single-result PNG/voltage-fit export arrays without widening float32 payloads | V361 |
| T323 | x | Preserve live/manual hardware voltage plot and recording export arrays without widening float32 payloads | V362 |
| T324 | x | Cast non-floating mesh/3D display helper values to float32 instead of float64 | V363 |
| T325 | x | Preserve equipotential camera coordinate axes without widening float32 payloads | V364 |
| T326 | x | Preserve 3D cell-center cache-miss fallback coords without widening float32 payloads | V365 |
| T327 | x | Preserve 3D spatial anomaly candidate score dtype during component ranking | V366 |
| T328 | x | Reuse the 3D anomaly score buffer for signed-mode residual MAD thresholding | V367 |
| T329 | x | Direct-fill 3D electrode overlay triangle indices without tuple-list staging | V368 |
| T330 | x | Remove Matplotlib 3D facecolor buffer/cache pressure from 3D rendering | V369 |
| T331 | x | Build 3D electrode overlays through PyVista direct-filled buffers, not Matplotlib polygons | V370 |
| T332 | x | Compute 3D point-data face means without per-face indexed subsets | V371 |
| T333 | x | Build 3D surface-helper face vertices without per-face index arrays | V372 |
| T334 | x | Compute spatial anomaly component masses without candidate-score subset arrays | V373 |
| T335 | x | Preserve float32 coordinates through streaming 3D volume-fraction painting buffers | V374 |
| T336 | x | Reuse forward geometry extraction in dataset generator instead of duplicate midpoint/connectivity traversal | V375 |
| T337 | x | Direct-fill all-finite metrics nearest-resample output without mapped-values copy | V376 |
| T338 | x | Preserve float32 work buffers in metrics brute-force nearest fallback | V377 |
| T339 | x | Stream conductivity image square-axis finite bounds without coordinate subset copies | V378 |
| T340 | x | Preserve float32 candidate centers in 3D spatial anomaly KDTree filtering | V379 |
| T341 | x | Preserve float32 legacy cell-vertices samples in 3D volume-fraction fallback | V380 |
| T342 | x | Preserve float32 inside-count and fraction buffers in 3D volume-fraction blending | V381 |
| T343 | x | Direct-fill tetra projection boundary faces without kept-list staging | V382 |
| T344 | x | Direct-fill shared tetra boundary triangles without kept-list staging | V383 |
| T345 | x | Direct-fill 3D boundary-face sources without kept-list staging | V384 |
| T346 | x | Reuse all-valid 3D boundary faces without payload staging | V385 |
| T347 | x | Batch-fill 3D face vertices for boundary/highlight helper outputs | V386 |
| T348 | x | Direct-fill 3D anomaly highlight vertices and values | V387 |
| T349 | x | Reuse 3D anomaly mask without flatnonzero index array | V388 |
| T350 | x | Direct-fill point-cloud highlight arrays without flatnonzero index vector | V389 |
| T351 | x | Direct-fill spatial anomaly candidate indices and centers | V390 |
| T352 | x | Apply spatial anomaly keep mask without candidate index subset | V391 |
| T353 | x | Direct-fill all-retained point-cloud background indices without flatnonzero | V392 |
| T354 | x | Direct-fill all-retained point-cloud true/anomaly indices without flatnonzero | V393 |
| T355 | x | Direct-fill mesh-derived cell measures without Python list staging | V394 |
| T356 | x | Compute mesh-derived tetra measures without per-cell point gather | V395 |
| T357 | x | Fast-path axis-aligned hexa mesh-derived measures without ConvexHull | V396 |
| T358 | x | Reuse one mesh extraction for mesh-derived cold-build signature and arrays | V397 |
| T359 | x | Direct-fill FEMx cell/facet connectivity extraction without links list staging | V398 |
| T360 | x | Reuse one-dimensional dual-mesh bbox candidate mask during cell lookup | V399 |
| T361 | x | Iterate dual-mesh bbox candidate mask without flatnonzero candidate indices | V400 |
| T362 | x | Reuse dual-mesh candidate simplex vertex buffer without per-candidate coords gather | V401 |
| T363 | x | Reuse GREIT finite-target radius mask and apply contrast with where-add | V402 |
| T364 | x | Use where-divide Sparse MAP warm-start singular filtering without masked subsets | V403 |
| T365 | x | Direct-fill simulation metrics masked-target resample output without mapped_values vector | V404 |
| T366 | x | Sanitize and normalize GN measurement weights in-place without replacement arrays | V405 |
| T367 | x | Clamp GN difference-mode measurement weights in-place without np.where replacement | V406 |
| T368 | x | Sanitize matrix-free preconditioner diag in-place on private copy | V407 |
| T369 | x | Scan GN line-search finite metrics without valid-index or objective-subset arrays | V408 |
| T370 | x | Build dynamic TV-Huber robust weights/penalties without np.where replacement arrays | V409 |
| T371 | x | Avoid dynamic temporal ROI submatrix copies in robust normal/objective paths | V410 |
| T372 | x | Reuse TV regularization gradient buffer for weight prep and in-place normalization | V411 |
| T373 | x | Scan GN finite diagnostic summaries without finite-subset copies | V412 |
| T374 | x | Scan shared numeric finite summaries without finite-subset copies | V413 |
| T375 | x | Scan electrode pattern finite summaries without finite-subset copies | V414 |
| T376 | x | Scan GN regularization validation finite min/max without finite-subset copies | V415 |
| T377 | x | Downsample 3D point-cloud true indices without per-chunk flatnonzero arrays | V416 |
| T378 | x | Direct-fill PETSc electrode coupling nonzero indices and values without flatnonzero | V417 |
| T379 | x | Apply GREIT blob target masks without inverse-mask slice assignment | V418 |
| T380 | x | Zero measurement bad-channel rows/vectors/weights without boolean lhs indexing | V419 |
| T381 | x | Zero RM frame-batch bad-channel columns without boolean lhs indexing | V420 |
| T382 | x | Reuse temporal RM frame-contract metadata without dense diagonal pre-prepare | V421 |
| T383 | x | Zero dynamic temporal non-ROI weight columns without inverse-mask indexing | V422 |
| T384 | x | Store diagonal measurement contracts without dense diagonal matrices | V423 |
| T385 | x | Build full measurement sqrt transform without intermediate dense diagonal | V424 |
| T386 | x | Apply RM frame-batch diagonal sqrt weights in-place on private weight copy | V425 |
| T387 | x | Gather 3D boundary face values into preallocated output | V426 |
| T388 | x | Gather 3D point-cloud display arrays into preallocated outputs | V427 |
| T389 | x | Sort 3D point-cloud sampled indices in-place | V428 |
| T390 | x | Reuse finite-scan chunk buffer for 3D color/anomaly helpers | V429 |
| T391 | x | Fill invalid 3D spatial nearest distances with copyto where mask | V430 |
| T392 | x | Reuse mesh-derived fallback cell-vertex buffer | V431 |
| T393 | x | Reuse graph-prior simplex volume work buffers | V432 |
| T394 | x | Reuse 3D crowded anomaly finite mask for percentile thresholding | V433 |
| T395 | x | Direct-fill metrics nearest-resample masked source/query compaction | V434 |
| T396 | x | Reuse metrics finite-row scan chunk bool buffers | V435 |
| T397 | x | Reuse metrics finite-pair chunk bool buffers | V436 |
| T398 | x | Apply forward shape-paint masks via copyto where | V437 |
| T399 | x | Fill cell-to-node orphan/NaN nodes via copyto where | V438 |
| T400 | x | Reuse GN line-search perturb-limit mask writes | V439 |
| T401 | x | Compute GREIT metric masked weighted sums without subset copies | V440 |
| T402 | x | Reuse hardware reconstruction interpolation buffers | V441 |
| T403 | x | Direct-fill hardware reconstruction grid-cache barycentric arrays | V442 |
| T404 | x | Avoid GREIT metrics default target-mask float target copy | V443 |
| T405 | x | Reuse GREIT metrics image buffer for positive signed target path | V444 |
| T406 | x | Direct-fill VoxelGrid locate-points inside/outside mapping | V445 |
| T407 | x | Reuse VoxelGrid locate-points scaled-coordinate work buffer | V446 |
| T408 | x | Avoid GUI 3D color-limit finite subset median copy | V447 |
| T409 | x | Direct-fill GREIT3D inside target centers without boolean row subset | V448 |
| T410 | x | Avoid TV regularization non-finite weight median subset copy | V449 |
| T411 | x | Avoid GUI 3D point-cloud background mask recount pass | V450 |
| T412 | x | Reuse VoxelGrid outside-mask count for nearest compaction | V451 |
| T413 | x | Reuse GREIT native center-spacing unique order without diff subsets | V452 |
| T414 | x | Reuse GREIT3D inside-mask count for target-center compaction | V453 |
| T415 | x | Avoid GREIT finite-target positivity bool vector allocation | V454 |
| T416 | x | Avoid GREIT background conductivity positivity bool vector allocation | V455 |
| T417 | x | Avoid GREIT metric cell-volume positivity bool vector allocation | V456 |
| T418 | x | Avoid GREIT measurement-order range bool vector allocation | V457 |
| T419 | x | Avoid GREIT vh normalization full abs/bool allocations | V458 |
| T420 | x | Avoid GREIT measurement-order unique sort and identity arange allocation | V459 |
| T421 | x | Avoid GREIT desired extent active-axis bool matrix allocation | V460 |
| T422 | x | Avoid GREIT domain fallback bounds bool matrix allocation | V461 |
| T423 | x | Avoid dynamic ROI index range bool vector allocation | V462 |
| T424 | x | Avoid dynamic ROI sparse row index-subset allocation | V463 |
| T425 | x | Avoid GREIT desired cell-extent negative bool matrix allocation | V464 |
| T426 | x | Avoid GREIT XYZ point finite full-bool allocation | V465 |
| T427 | x | Avoid GREIT measurement/Y finite full-bool allocation | V466 |
| T428 | x | Avoid dynamic frame/state finite full-bool allocation | V467 |
| T429 | x | Avoid dynamic RM/Kalman matrix finite full-bool allocation | V468 |
| T430 | x | Avoid dynamic temporal weighted-normal positive bool vectors | V469 |
| T431 | x | Avoid GREIT artifact/NF finite full-bool allocation | V470 |
| T432 | x | Avoid GUI 3D spatial candidate finite full-bool allocation | V471 |
| T433 | x | Reuse GUI mesh helper finite-scan chunk buffer | V472 |
| T434 | x | Avoid GUI PyVista highlight duplicate mask pass | V473 |
| T435 | x | Avoid GUI simulation voltage-fit finite full-bool allocation | V474 |
| T436 | x | Share bounded imaginary-component scans across routing/forward/GN guards | V475 |
| T437 | x | Use shared bounded finite scans in forward scalar/CUDA diagonal guards | V476 |
| T438 | x | Use shared bounded finite/imag scans in reconstruction controller guards | V477 |
| T439 | x | Reuse cell-to-node touched mask for NaN fill and finite mean scan | V478 |
| T440 | x | Reuse bounded bool buffers for conductivity image xy finite bounds | V479 |
| T441 | x | Use bounded finite scans across reconstruction-matrix helpers | V480 |
| T442 | x | Use bounded finite scans across sigma/contact block-system helpers | V481 |
| T443 | x | Use bounded finite scans across matrix-free GN and dual-mesh helpers | V482 |
| T444 | x | Use bounded finite scans in measurement-channel contracts | V483 |
| T445 | x | Use bounded finite scans in temporal measurement filtering | V484 |
| T446 | x | Use bounded finite scans in RM matmul kernels | V485 |
| T447 | x | Use bounded finite scans in inverse postprocess temporal/TV guards | V486 |
| T448 | x | Use bounded numeric scans in dynamic sequence and EIDORS noise ingress | V487 |
| T449 | x | Use bounded finite scans in RtR and TV-IRLS prior guards | V488 |
| T450 | x | Use bounded finite scans in GN regularization readiness guards | V489 |
| T451 | x | Use bounded finite scans in GN linear-system guards | V490 |
| T452 | x | Use bounded finite scans in reduced snapshot bank add | V491 |
| T453 | x | Use bounded scans in dual-mesh array validators | V492 |
| T454 | x | Use bounded finite scans in GUI single-step sigma update guards | V493 |
| T455 | x | Use bounded finite scans in dynamic inverse guards | V494 |
| T456 | x | Use bounded finite scans across remaining GREIT helpers | V495 |
| T457 | x | Use bounded finite scans in data experiment validators | V496 |
| T458 | x | Use bounded finite scans in sweep and bucket experiment builders | V497 |
| T459 | x | Finish residual bounded finite scan cleanup in physics/mesh/electrode/regularization helpers | V498 |
| T460 | x | Replace selected comparison bool payloads with min/max reductions | V499 |
| T461 | x | Replace GREIT comparison bool payloads with min/max reductions | V500 |
| T462 | x | Use reduction/threshold work buffers in interop, graph, and GUI complex checks | V501 |
| T463 | x | Reuse electrode measurement hash comparison hits | V502 |
| T464 | x | Clamp normalized-difference references with bounded work buffers | V503 |
| T465 | x | Add measurement-form diagonal regularisation/noise terms in place | V504 |
| T466 | x | Reduce GREIT nearest-distance positive min without subset copies | V505 |
| T467 | x | Avoid dense GN diagonal/offdiag and jitter identity temporaries | V506 |
| T468 | x | Avoid dense matrix-free GN weight diagonal comparison payloads | V507 |
| T469 | x | Use sparse diagonal identity fallbacks for regularization matrices | V508 |
| T470 | x | Prefer boolean mask extraction for PyVista volume highlight cells | V509 |
| T471 | x | Add native complex GN diagonal regularization in place | V510 |
| T472 | x | Add dense PMAT diagonal shift in place | V511 |
| T473 | x | Keep GUI native-complex identity regularization lazy | V512 |
| T474 | x | Add sparse Bayesian dense-system diagonal terms in place | V513 |
| T475 | x | Direct-fill and reuse dynamic Kalman identity matrices | V514 |
| T476 | x | Direct-fill TV nonlinear dense diagonal output | V515 |
| T477 | x | Add digit-metric ridge identity terms in place | V516 |
| T478 | x | Direct-fill diagonal-to-dense compatibility matrices | V517 |
| T479 | x | Use reshape views for GN initial/prior vectors | V518 |
| T480 | x | Use diagonal views for dense diagonal extraction | V519 |
| T481 | x | Direct-fill GN linear-system identity/diagonal dense builds | V520 |
| T482 | x | Add GN difference runner diagonal system terms in place | V521 |
| T483 | x | Reduce 3D diagnostic overview mask/metric temporaries | V522 |
| T484 | x | Share script Pearson correlation helper without corrcoef stack | V523 |
| T485 | x | Reduce gallery diagnostic correlation and ROI mean temporaries | V524 |
| T486 | x | Route diagnostic finite correlations through common reducers | V525 |
| T487 | x | Route small-domain diagnostic ROI means through common reducer | V526 |
| T488 | x | Clean benchmark difference measurement weights in place | V527 |
| T489 | x | Compute holdout indexed voltage RMSE in chunks | V528 |
| T490 | x | Direct-fill dynamic validation truth/Jacobian benchmark matrices | V529 |
| T491 | x | Direct-fill real gallery slice interpolation query matrices | V530 |
| T492 | x | Stream real tank holdout script metrics and output matrices | V531 |
| T493 | x | Direct-fill small-domain diagnostic grid queries and circle masks | V532 |
| T494 | x | Remove residual dense temporaries from synthetic parity script | V533 |
| T495 | x | Add benchmark difference dense diagonals in place | V534 |
| T496 | x | Direct-fill prior travelling-wave frames and Jacobian rows | V535 |
| T497 | x | Direct-fill dual-model RM fine centers and Jacobian rows | V536 |
| T498 | x | Direct-fill common method-runner measurement frame stacks | V537 |
| T499 | x | Direct-fill GREIT parity benchmark dense helper matrices | V538 |
| T500 | x | Direct-fill 3D overview wireframe and electrode point matrices | V539 |
| T501 | x | Direct-fill fair EIDORS export boundary and measurement matrices | V540 |
| T502 | x | Direct-fill EIDORS forward parity-gate measurement concat | V541 |
| T503 | x | Direct-fill mesh IO tag-pair hash matrices | V542 |
| T504 | x | Stream bucket noise-sweep plot value ranges | V543 |
| T505 | x | Remove small diagnostic stack/concat builders | V544 |
| T506 | x | Direct-fill GN difference LSMR augmented vectors | V545 |
| T507 | x | Direct-fill direct-Jacobian traditional electrode identity | V546 |
| T508 | x | Bound remaining script/UI finite scans | V547 |
| T509 | x | Bound remaining comparison-mask scans | V548 |
| T510 | x | Replace remaining `np.all(np.isfinite(...))` guards | V549 |
| T511 | x | Chunk single-step sigma-floor alpha limit scans | V550 |
| T512 | x | Stream masked holdout/bucket structure reductions | V551 |
| T513 | x | Remove TV-PDHG ROI subset copies | V552 |
| T514 | x | Direct-fill measurement-pattern filtered rows | V553 |
| T515 | x | Preserve float32 and direct-fill temporal smoothing/TV postprocess | V554 |
| T516 | x | Preserve float32 RM online preprojected frame payloads | V555 |
| T517 | x | Preserve float32 GUI single-step sigma floor updates | V556 |
| T518 | x | Preserve float32 measurement-channel contracts | V557 |
| T519 | x | Preserve float32 difference projections | V558 |
| T520 | x | Preserve hardware equipotential PyVista point dtype | V559 |
| T521 | x | Remove redundant acquisition frame widening/copy in ring-buffer polling | V560 |
| T522 | x | Preserve float32 in GUI abs-threshold scan and GREIT rec-model padding | V561 |
| T523 | x | Preserve float32 reference-floor work buffers in normalized difference helpers | V562 |
| T524 | x | Reduce FrameData complex/magnitude extraction temporaries | V563 |
| T525 | x | Stream HDF5 numeric checksum verification without full dataset materialization | V564 |
| T526 | x | Stream HDF5 legacy manifest fallback digests without full dataset materialization | V565 |
| T527 | x | Defer hardware reconstruction grid work buffers and preserve float32 display dtype | V566 |
| T528 | x | Downcast 3D display-only float64 payloads to float32 before scene construction | V567 |
| T529 | x | Prefer PyVista offscreen for WSLg/Wayland first-view 3D rendering and reserve the failure cache for unavailable captions | V568 |
| T530 | x | Reduce forward-result geometry center temporary from n_cells×dim to n_cells | V569 |
| T531 | x | Stream noncontiguous HDF5 artifact array digests in bounded C-order chunks | V570 |
| T532 | x | Stream noncontiguous cache-key array digests in bounded C-order chunks | V571 |
| T533 | x | Remove local contiguous copies from GUI RM mesh-signature hashing | V572 |
| T534 | x | Remove local contiguous copies from GREIT cache-signature hashing | V573 |
| T535 | x | Remove local contiguous copies from forward/TV/GN hash callers | V574 |
| T536 | x | Remove local contiguous copies from KSP benchmark hash helpers | V575 |
| T537 | x | Use flat DOLFINx connectivity arrays for GUI forward-result geometry extraction | V576 |
| T538 | x | Avoid forward/dataset measurement-vector copies when no noise is added | V577 |
| T539 | x | Avoid EIDORS noise input copies before final noisy output allocation | V578 |
| T540 | x | Avoid Sparse-Bayes baseline/reference metadata snapshots | V579 |
| T541 | x | Avoid difference-measurement raw target/reference entry snapshots | V580 |
| T542 | x | Reuse single-frame normalized difference buffers when reference floor is not needed | V581 |
| T543 | x | Avoid normalized Jacobian safe-reference copies when reference floor is not needed | V582 |
| T544 | x | Collapse online RM frame-contract private buffer creation to one allocation | V583 |
| T545 | x | Broadcast 1D batch difference references without expanding reference frames | V584 |
| T546 | x | Remove local contiguous copies from semantic object-signature ndarray hashing | V585 |
| T547 | x | Avoid GREIT target-center copies when target plane/offset is not used | V586 |
| T548 | x | Reuse read-only persistent Jacobian cache hits without full private copy | V587 |
| T549 | x | Remove local contiguous copies from RM signature ndarray hashing | V588 |
| T550 | x | Reuse owned GN final sigma array for the final forward-fit image | V589 |
| T551 | x | Index compiled FFCx modules once during backend worker JIT-cache cleanup | V590 |
| T552 | x | Remove local contiguous copies from GUI array-geometry cache signature inputs | V591 |
| T553 | x | Use flat DOLFINx connectivity arrays for mesh-derived cell extraction | V592 |
| T554 | x | Remove local contiguous copies from CUDA structured sigma hashing | V593 |
| T555 | x | Remove local contiguous copies from GN and direct-Jacobian sigma cache hashing | V594 |
| T556 | x | Remove local contiguous copies from linearized sigma fingerprints and ROM snapshot hashes | V595 |
| T557 | x | Remove local contiguous copies from GREIT registry ndarray signature hashing | V596 |
| T558 | x | Remove local contiguous copies from Sparse-Bayesian baseline cache hashing | V597 |
| T559 | x | Remove local contiguous copies from GN linear-system cache-signature hashing | V598 |
| T560 | x | Remove local contiguous copies from RtR prior signature hashing | V599 |
| T561 | x | Remove full nearest-valid bool buffer from GUI 3D spatial anomaly radius estimation | V600 |
| T562 | x | Reuse point-cloud background rank array instead of copying initial candidates | V601 |
| T563 | x | Stream boundary-voltage y-range finite min/max without full masks | V602 |
| T564 | x | Reuse hardware equipotential finite-range chunk work buffer | V603 |
| T565 | x | Reuse complex-channel imaginary scan finite and abs work buffers | V604 |
| T566 | x | Add resident-byte budget to persistent dense-Jacobian process cache | V605 |
| T567 | x | Add resident-byte budget to generated/loaded EITMesh process cache | V606 |
| T568 | x | Add resident-byte budget to forward static setup process cache | V607 |
| T569 | x | Add resident-byte budgets to GUI reconstruction system and single-step context caches | V608 |
| T570 | x | Enforce total-byte LRU for RM fit-Jacobian process cache | V609 |
| T571 | x | Make 3D GUI prewarm default to setup-prime while retaining explicit worker mode | V610 |
| T572 | x | Bound TV regularization finite-median mask allocation | V611 |
| T573 | x | Stream rectangle/cuboid center-paint masks with bounded axis buffers | V612 |
| T574 | x | Avoid full complex magnitude copy in GN nonfinite summaries | V613 |
| T575 | x | Reuse real GN difference-weight subtraction buffer for absolute values | V614 |
| T576 | x | Bound GN line-search lower-alpha guard scan | V615 |
| T577 | x | Bound GN preconditioner diagonal clamp detection mask | V616 |
| T578 | x | Bound GN line-search upper-alpha overflow limit scans | V617 |
| T579 | x | Bundle + verify FEniCSx JIT compiler in packaged backend apps | V620,V638,V670,V673 |
| T580 | x | Add candidate-constrained NIS reject/variance-inflation to fixed-lag dynamic Kalman | V674 |
| T581 | x | Add persistent diagonal Kalman registry and backend-worker realtime postprocess | V674,V675 |
| T582 | x | Add persistent measurement-space diagonal Kalman + auto/fast fallback for realtime worker | V674,V675,V676,V677,V678,V679 |
| T583 | x | Bundle core runtime commands for host-independent pure Nix backend wrapper initialization | V673,V680 |
| T584 | x | Add deterministic dynamic Kalman sequence acceptance report | V674,V675,V676,V681 |
| T586 | x | Anchor measurement Kalman to NOSER + guard/reset divergent state; make auto safe-image | V675,V676,V677,V681,V682 |
| T587 | x | Add Robin-transconductance CEM + PyEIDORS/NGSolve/EIDORS parity benchmark | V1,V683,V684,V685,V686 |
| T588 | x | Backprop strict same-mesh/float64 CEM parity + fair cold/warm timing benchmark | V683,V684,V685,V686,V687,V688,V689,V690,V691,V692 |
| T589 | x | Add independent 80/128-dps CEM truth + absolute PyEIDORS/NGSolve/EIDORS accuracy ranking | V67,V683,V687,V693,V694,V695,V696,V697,V698,V699,V700 |
| T590 | x | Add rational-circular multi-case exact CEM truth + cross-FEM robustness report | V683,V686,V687,V689,V691,V692,V697,V698,V699,V700,V701,V702,V703,V704,V705,V706 |
| T591 | x | Backprop unambiguous paired CEM cold/setup/warm timing + absolute speedup report | V688,V691,V697,V706,V708,V709 |
| T592 | x | Add true-circle h-refinement + independent continuum CEM reference + cross-FEM convergence report | V683,V684,V686,V687,V691,V692,V697,V698,V699,V705,V706,V708,V710,V711,V712,V713,V714,V715,V716 |
| T593 | x | Expand exact rational CEM to nested mesh-refinement sequence + solver-accuracy report | V683,V686,V687,V691,V692,V697,V698,V699,V700,V701,V702,V703,V704,V705,V706,V707,V708,V709,V717,V718,V719,V720 |
| T594 | x | Reconcile true-circle total error with rational discrete accuracy + shared-reference sensitivity | V705,V706,V710,V711,V712,V713,V714,V715,V716,V720,V721,V722 |
| T595 | x | Expand exact rational CEM to balanced mesh/physics/drive factorial + cached QQ basis truth | V683,V686,V687,V691,V692,V697,V698,V699,V700,V701,V702,V703,V704,V705,V706,V707,V708,V709,V717,V718,V720,V723,V724,V725 |
| T596 | x | Run preregistered rational extension for sigma/z, heterogeneous cells, 8e and Q4 with per-case QQ truth | V683,V686,V687,V691,V692,V697,V698,V699,V700,V701,V702,V704,V705,V706,V707,V708,V709,V717,V718,V724,V726,V727,V728,V730,V731,V732 |
| T597 | x | Cross low-z assembly matrices with SciPy/MATLAB backends and isolate controlled accumulation-order sensitivity | V683,V687,V691,V697,V702,V704,V705,V708,V727,V728,V729,V730 |
| T598 | x | Decouple EIDORS interop CLIs from benchmark internals and restore current API/scalar-compatible bidirectional roundtrip | V129,V171,V698,V733 |
| T599 | x | Productize Bridge/Geometry v2 for novice exact 2D/3D EIDORS migration: protocol/schema/CLI, custom stim-meas capture, real mesh GUI+worker+dataset path, examples/docs, real MATLAB bidirectional 3D gate | V129,V171,V698,V733,V734,V735,V736,V737,V738,V739,V740 |
| T600 | x | Replace heuristic EIDORS capture with source-verified object discovery, provenance, electrode/current/conductivity semantics, strict imported-forward readiness, and real CEM/PEM/default acceptance | V741,V742,V743,V744,V745,V746 |
| T601 | x | Add native point-electrode forward model and preserve PEM identity across EIDORS↔PyEIDORS import/export without facet projection | V743,V746,V747,V748,V749,V750,V751,V752 |
| T602 | x | Specify + implement Bridge/Geometry v3-only schema, fingerprints, integrity, typed package API and CLI contract | V753 |
| T603 | x | Replace global electrode route with exact weighted/mixed PEM+CEM operator incl internal CEM | V754 |
| T604 | x | Capture/export/import EIDORS runtime operators, fields and common inverse semantics in v3 MATLAB/Python bridge | V753,V755,V760 |
| T605 | x | Add immutable managed ModelRegistry + ModelContextFactory + persistent flow bindings | V756,V757,V760,V765 |
| T606 | x | Route simulation and dataset through bound ModelContext exact background/target/protocol | V757 |
| T607 | x | Migrate FrameDatabase v3 + bind historical sessions + exact database reconstruction context | V758,V760 |
| T608 | x | Add provable channel remap and proportional actual-current realtime model override | V759 |
| T609 | x | Add one-click v3 package/script GUI, asset manager, selectors, preview and beginner examples/docs | V755,V761 |
| T610 | x | Run bidirectional identity, voltage/Jacobian parity, full Nix/Ruff/build/installed-CLI delivery gates | V762 |
| T611 | x | Harden/rebuild 3 Linux one-click packages against novice PATH/Nix/Python/CUDA conflicts; expand beginner docs + clean-host acceptance | V132,V671,V673,V680,V770,V771 |
| T612 | x | Backprop GUI smoke drift, legacy runner, complex cache scalar, lock-free worker exit, exact Bridge rewrite, dead interop helper | V146,V245,V317,V734,V753,V756,V757,V764,V772,V773,V774,V775,V776 |
| T613 | x | Add registry-wide SPEC integrity gate; fix duplicate B IDs; archive completed T/B; split V by domain with root index + ID map; refresh §T.phase, §I landed markers, V21 scope | V21,V777 |
| T614 | x | Delete placeholder `pyeidors.main` hello entrypoint and coverage-only tests | V778,I |
