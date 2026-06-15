# Lazy Adjoint / Matrix-Free 3D EIT Experiment Log

Date: 2026-04-21

This is a tracked design/experiment note. Raw benchmark artifacts remain under `reports/runtime_benchmarks/`, which is ignored by Git.

## Purpose

Record the current exploratory attempts around replacing the 3D dense Jacobian path with a lazy adjoint / matrix-free path. This is not a final optimization plan; it is a snapshot of what was tried, measured, and deliberately not promoted to default behavior.

## Baseline Case

GUI-like 3D simulation configuration:

- total electrodes: 48
- rings: 3
- electrodes per ring: 16
- mesh family: tetra
- radius: 0.18
- height: 0.16
- refinement: 2
- electrode levels: 0.15, 0.5, 0.85
- protocol: `hybrid_full_3d`
- measurements: 5936
- parameters/elements: 5166
- forward backend: DOLFINx + PETSc
- CUDA path: PETSc `aijcusparse`, PETSc `cuda` Vec, PETSc `densecuda`, Torch CUDA

Earlier dense/single-step CUDA observation:

- cold wall time: 119.70 s
- dense/direct Jacobian build: 84.98 s
- warm same-process/cache time: 0.45 s
- CPU cold semantic context exceeded 1204 s timeout and was killed

Interpretation: dense Jacobian materialization was the main cold-path bottleneck.

## Attempt 1: GUI 3D Tetra GPU Runtime Policy

Implemented before this note:

- 3D tetra + `--gpu` now resolves to PETSc/Torch CUDA instead of silently staying on CPU.
- GUI diagnostics expose `mesh_family`, `forward_backend`, `petsc_device_effective`, `torch_device`, `cache_hit`, and `jacobian_representation`.
- GUI status text now makes the CPU/CUDA/cache path visible.

Decision: keep. This makes future runtime experiments interpretable.

## Attempt 2: Lazy Adjoint Linearization

Implemented:

- `EITForwardModel.solve_full_rhs(...)` for arbitrary full-system RHS blocks.
- `LazyAdjointJacobianLinearization`.
- Lazy `Jv` without materializing dense `J`.
- Lazy `J^T r` that combines residual weights per stimulation, instead of building one full adjoint gradient per measurement.
- Lazy `hessian_diag(...)` without dense `J`.
- `jacobian_representation="lazy"` in `scripts/common/gn_difference_runner.py`.
- Dense operator artifacts `Jt`, `A`, and `LU` are disabled for lazy mode.
- GUI/controller accept explicit `jacobian_representation="lazy"` / `"matrix-free"`.

Current limitation:

- `Jv` still assembles sensitivity RHS through DOLFINx forms inside each Krylov action.
- The bottleneck has moved from dense Jacobian build to repeated Krylov actions.

## Attempt 3: Linear Solver Strategy Control

Problem:

- Lazy CG with a small iteration count did not converge.
- Old behavior could automatically fall back to LSMR.
- In lazy mode, this fallback is expensive because LSMR repeatedly calls both `Jv` and `J^T r`.

Implemented:

- `linearized_solver_strategy="cg_only"`: CG only, no LSMR fallback. Lazy `auto` resolves here.
- `linearized_solver_strategy="cg_lsmr"`: CG first, LSMR fallback on CG failure.
- `linearized_solver_strategy="lsmr"`: direct LSMR.
- `linearized_solver_strategy="cgls"`: augmented least-squares CGLS.

Diagnostics added:

- method
- strategy
- `cg_info`
- converged
- maxiter
- action info
- preconditioner info

Decision: keep. It prevents hidden expensive fallback and makes benchmark runs comparable.

## Attempt 4: Lazy Preconditioner Modes

Implemented:

- `approx`: cheap block-level approximate diagonal; no extra adjoint solves; current lazy default.
- `batch_noser`: sampled NOSER-like diagonal using batched adjoint solves.
- `coarse`: smaller sampled Hessian/diagonal approximation.
- `prior`: regularization/prior diagonal only.

Current default:

- `lazy_preconditioner_mode="auto"` resolves to `approx`.

Reason:

- `batch_noser` was structurally useful but expensive on cold context and did not improve 20-iteration CG convergence in the tested case.

## Reproducible Benchmark Script

Added:

- `scripts/benchmarks/benchmark_lazy_48e_cuda_runtime.py`

Example:

```bash
nix develop .#complex64-cuda -c python scripts/benchmarks/benchmark_lazy_48e_cuda_runtime.py \
  --linearized-maxiter 20 \
  --linearized-solver-strategy cg_only \
  --lazy-preconditioner-mode auto
```

## Benchmark Results

### Lazy + Approx PC, CG Only, maxiter=3

Terminal smoke result:

- cold context: 3.31 s
- cold process: 4.36 s
- warm context: 0.10 s
- warm process: 3.73 s
- cold `linear_solve`: 3.61 s
- warm `linear_solve`: 2.83 s
- `cg_info`: 3
- converged: false

Interpretation: very fast preview/smoke path, not a converged reconstruction solve.

### Lazy + Approx PC, CG Only, maxiter=20

Artifact:

- `reports/runtime_benchmarks/lazy_48e_cuda_20260421_120747/summary.json`

Measured:

- cold context: 2.64 s
- warm context: 0.09 s
- cold process: 13.47 s
- warm process: 13.77 s
- lazy Jacobian scaffold: about 1.01 s
- operator preconditioner setup: about 0.007 s
- cold `linear_solve`: 12.39 s
- warm `linear_solve`: 12.68 s
- `cg_info`: 20
- converged: false

Interpretation:

- Dense Jacobian cold build is effectively removed.
- Runtime is now dominated by matrix-free Krylov actions.
- 20 iterations still does not report convergence.

### Lazy + Batch NOSER PC, CG Only, maxiter=20

Artifact:

- `reports/runtime_benchmarks/lazy_48e_cuda_batch_noser_20260421/summary.json`

Measured:

- cold context: 17.03 s
- warm context: 0.10 s
- cold process: 13.91 s
- warm process: 14.05 s
- cold `operator_precond`: 14.31 s
- sampled measurements: 560
- sampled diag solve count: 560
- cold `linear_solve`: 12.97 s
- warm `linear_solve`: 12.94 s
- `cg_info`: 20
- converged: false

Interpretation:

- Batch NOSER is cacheable and warm context is cheap.
- Cold setup cost is significant.
- It did not improve 20-iteration CG convergence enough to become default.

## Current Decision

Do not switch GUI `auto` to lazy yet.

Reason:

- Lazy removes the dense 3D Jacobian cold-build bottleneck.
- The actual linear solve still does not report convergence at maxiter=20 in the tested 48-electrode case.
- Switching GUI `auto` now would risk presenting a fast but not-yet-converged result as the default reconstruction path.

Keep:

- dense/default path for high-confidence GUI auto behavior
- explicit lazy/matrix-free path for experiments
- diagnostics and benchmark script for future optimization work

## Verification

Targeted tests:

- `tests/unit/test_jacobian_linearization.py`
- `tests/unit/test_adjoint_jacobian_helper_branches.py`
- `tests/unit/test_gn_diff_operator_cache.py`
- `tests/unit/test_gn_linearized_real_smoke.py`
- `tests/unit/test_reconstruction_cache_regressions.py`
- `tests/unit/test_conductivity_3d_widget_runtime.py`
- `tests/unit/test_gn_diff_3d_operator_cache.py`

Result: 69 passed.

Full default software sharded unit baseline:

```bash
nix develop .#complex64-cuda -c python scripts/ci/run_sharded_unit_tests.py --run --all --timeout 300
```

Result:

- forward: passed
- inverse-gn: passed
- cache: passed
- sparse: passed
- mesh-femx: passed
- perf-cuda: passed
- gui: passed
- env-cli: passed
- coverage-gap: passed
- data-visualization: passed
- core-misc: passed

Summary artifact:

- `test_results/sharded_unit/20260421T041527Z/summary.json`

## Redesign Notes

1. `Jv` still pays DOLFINx sensitivity RHS assembly cost per Krylov action.
2. `J^T r` is combined by stimulation, but each Krylov iteration still performs many PDE solves.
3. Need residual history and reconstruction-quality metrics, not just `cg_info`.
4. Batch NOSER needs a better policy before it can be useful by default.
5. Separate preview policy from production policy:
   - preview: low maxiter, explicitly marked not converged
   - production: convergence-gated
6. GUI auto should stay conservative until lazy has a hard acceptance gate.
7. Future fair comparisons should separately time mesh, base forward, lazy scaffold, PC construction, Krylov actions, forward validation, and warm cache effects.
8. Future design should consider PETSc MatShell/KSP for inverse Krylov, coarse inverse mesh, multiresolution reconstruction, stronger prior/preconditioner design, and reduced sensitivity RHS assembly overhead.

## Bottom Line

The experiment proves that lazy adjoint / matrix-free can remove the dense 3D Jacobian cold-build bottleneck. The next optimization plan should focus on matrix-free action cost, convergence policy, and stronger preconditioning rather than simply toggling GUI auto to lazy.
