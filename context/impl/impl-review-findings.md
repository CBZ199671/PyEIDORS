---
created: "2026-04-20"
last_edited: "2026-04-20"
---

# Review Findings

| Finding | Severity | File | Status |
| --- | --- | --- | --- |
| F-001: T-FPX-003 matSolve policy gap | P3 | `tests/unit/test_forward_mat_solve_policy.py` | RESOLVED |
| F-002: T-FPX-004 benchmark hid KSP non-convergence | P2 | `src/pyeidors/forward/eit_forward_model.py`; `scripts/benchmarks/benchmark_3d_runtime.py` | RESOLVED |
| F-003: T-FPX-006 operator solve was only wired below runtime | P2 | `src/pyeidors/inverse/solvers/gauss_newton_runtime.py` | RESOLVED |
| F-004: bare shard run selected hardware by default | P2 | `scripts/ci/run_sharded_unit_tests.py` | RESOLVED |

## Finding Details

### F-001: T-FPX-003 matSolve policy gap

`T-FPX-003` was marked complete after `mat_solve_mode=auto|off` call-count tests and 3D GAMG smoke passed, but `/ck:check` found two small acceptance-coverage gaps: explicit `mat_solve_mode="on"` was not pinned in the policy test file, and CPU matSolve failure fallback did not have a dedicated diagnostic assertion. Fixed in-place by adding `test_forward_mat_solve_mode_on_forces_mat_solve` and `test_forward_mat_solve_cpu_failure_falls_back_to_vector_loop`.

### F-002: T-FPX-004 benchmark hid KSP non-convergence

`T-FPX-004` produced the required `forward_solver_benchmark` fields, but `/ck:check` found the real quick artifact reported finite output while the PETSc GAMG `matSolve` had reached a negative convergence reason. The artifact did not expose `converged_reason`/`converged`, so readers could mistake a SciPy fallback result for a converged PETSc solve. Fixed in-place by recording KSP convergence reason in forward diagnostics, treating negative `matSolve` reason as fallback/error, adding `converged_reason` and `converged` to the benchmark artifact, and updating tests to accept explicit early fallback diagnostics.

### F-003: T-FPX-006 operator solve was only wired below runtime

`T-FPX-006` added operator support to `_solve_linear_system_fast()`, but `/ck:check` found `run_reconstruction()` still called dense `jacobian_calculator.calculate()` and `_project_measurement_jacobian()` before the fast solver. That meant real GN runtime could not select the new `JacobianLinearization` route. Fixed in-place by adding explicit `jacobian_method=linearized|operator|matrix-free` routing to `jacobian_calculator.linearize()`, skipping dense startup cache/projection for operator mode, passing measurement weights into the fast solver, and adding a runtime test that asserts dense `calculate()` is not called.

### F-004: bare shard run selected hardware by default

Final `/ck:check` found `run_sharded_unit_tests.py --run` and `--dry-run` still selected the opt-in `hardware` shard because their broad fallback used `build_all_shards()` with optional shards included. Fixed in-place by making broad run/dry-run selection pass `include_optional=args.include_hardware`, preserving explicit `--shard hardware`, updating validation docs, and adding tests for bare default exclusion plus opt-in inclusion.
