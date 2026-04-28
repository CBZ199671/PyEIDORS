# T85 Workflow Final Audit

Date: 2026-04-28

Scope: `src/pyeidors/inverse/workflows/` after T84 and T85 phases 1-2.
This audit records the remaining workflow-layer seams and closes T85 unless a
new workflow family creates fresh duplication.

## Current Shape

Workflow package:

| file | role | public function count | note |
|---|---:|---:|---|
| `absolute.py` | GN absolute facade | 1 | 66-line wrapper around `EITSystem.inverse_solve`. |
| `difference.py` | GN difference facade | 1 | 74-line wrapper; preserves eager forward solve before difference projection. |
| `sparse_bayesian.py` | sparse absolute/difference facades | 2 | 1 sparse-only factory + 2 wrappers. |
| `base.py` | shared workflow primitives | 9 helpers + `ReconstructionResult` | Result assembly, guard/type checks, forward-vector helper, and difference-vector resolution. |

Already consolidated:

| task | shared primitive | migrated behavior |
|---|---|---|
| T84 phase 1 | `merge_workflow_metadata`, `build_reconstruction_result` | Result object assembly and residual injection. |
| T84 phase 2 | `resolve_difference_vectors` | GN preprojected difference-space vector vs sparse raw-vector projection. |
| T85 phase 1 | `require_initialized`, `require_solver_output`, `resolve_simulated_or_forward` | Init guards, sparse `SolverOutput` guard, sparse simulated fallback. |
| T85 phase 2 | `forward_measurement_vector` | Raw `fwd_solve(...).meas` extraction shared by absolute/difference and sparse fallback. |

## Remaining Tails

| candidate | leave / future | rationale |
|---|---|---|
| `resolve_reconstruction_output` vs `require_solver_output` | leave | Error wording differs by layer: generic inverse solver payload vs sparse reconstructor owner-specific type guard. Merging would blur diagnostics. |
| baseline setup in `absolute.py` and `sparse_bayesian.py` | leave | Absolute copies `baseline_elem` and builds `initial_guess`; sparse passes baseline into SBL and records likelihood/prior metadata. Shared helper would be mostly argument plumbing. |
| `initial_guess` handling in absolute/difference | leave | Absolute defaults to baseline conductivity; difference defaults to `None` unless caller supplies an image. Not the same policy. |
| sparse `_ensure_reconstructor` | leave | Sparse-only factory hook. It preserves `SparseBayesianReconstructor` monkeypatch injection used by tests. No second workflow owns this shape. |
| mode-specific metadata dicts | leave | `baseline_used`, `reference_measured`, `solver_diagnostics`, sparse likelihood/prior fields, and user/solver metadata precedence are workflow contracts. `merge_workflow_metadata` is the right abstraction boundary. |
| `difference_measurement` / `project_measurement_vector` module imports in difference and sparse wrappers | leave | They intentionally remain module-level injection points; tests monkeypatch them before passing into `resolve_difference_vectors`. Hiding them in base defaults would weaken branch tests and make debugging harder. |
| `compute_residuals` module imports in wrappers | leave | Same injection pattern as above. Wrappers pass the module-local symbol into `build_reconstruction_result` so existing monkeypatch tests stay local to the facade module. |
| public `workflows.__init__` exports | leave | Four reconstruction entrypoints plus `ReconstructionResult`; no duplicate implementation. |

## Risk Notes

Do not merge the GN difference and sparse difference wrappers into a generic
`perform_difference_like_reconstruction` helper. Their simulated-measurement
contracts differ:

- GN runtime may emit `solver_output.simulated_measurement` already in
  difference measurement space.
- Sparse workflows emit raw forward measurements and must still project them
  through `resolve_difference_vectors(..., simulated_measurement_space="raw")`.
- The GN difference wrapper deliberately keeps the eager forward solve even
  when solver output already has a preprojected simulated vector. This is a
  compatibility behavior, locked by `test_difference_workflow_keeps_eager_forward_solve_with_solver_simulated`.

Also do not remove the module-local monkeypatch surfaces listed above unless
the corresponding tests move to a new explicit injection API.

## Closure Decision

T85 can close here:

- No remaining same-responsibility duplicate implementation is large enough to
  justify another shared helper.
- Remaining common-looking snippets encode different workflow policies or
  stable test injection surfaces.
- Further extraction would mostly create a generic orchestration helper with
  many mode flags, which would be less readable than the current thin facades.

Future extraction should wait for a new workflow family, for example dynamic
Kalman, propagation-aware, or batch spatiotemporal workflows. If those wrappers
repeat baseline resolution, metadata assembly, or simulated-measurement policy
for a third time, add a new task and gate it with explicit behaviour tests.

## Gates

Current T85 gates:

- `tests/unit/test_workflow_result_assembly.py`
- `tests/unit/test_workflow_wrapper_branches.py`
- `tests/unit/test_sparse_workflow_branches.py`
- `tests/unit/test_sparse_workflows.py`
- `tests/unit/test_simplified_eit_system.py`

Most recent full unit confirmation before this audit:

```text
1372 passed, 10 skipped in 598.89s
```
