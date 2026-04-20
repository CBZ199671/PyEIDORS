---
status: in_progress
source: map
---

# Implementation Overview

## Current Phase

FEniCSx/PETSc EIT refactor build site reached final continuity cleanup after
T-FPX-010. Production code and tests have changed; use
`impl-fenicsx-petsc-eit-refactor.md` as the detailed lab notebook before any
new solver/default changes.

## Active Build Site

- `context/plans/build-site-brownfield-cavekit.md`
- `context/plans/build-site-fenicsx-petsc-eit-refactor.md`

## Planned Work

| Task | Status | Notes |
| --- | --- | --- |
| T-CK-001 | DONE | Collect-only run completed; 96 tests collected before one unknown `fenics` marker error in interop test. |
| T-CK-002 | TODO | Add source-to-kit mappings. |
| T-CORE-001 | TODO | Validate core kit. |
| T-DATA-001 | TODO | Validate data/unit kit. |
| T-GEOM-001 | TODO | Validate geometry/electrode kit. |
| T-FWD-001 | TODO | Validate forward solver kit. |
| T-INV-001 | TODO | Validate inverse reconstruction kit. |
| T-CACHE-001 | TODO | Validate cache/performance kit. |
| T-INTEROP-001 | TODO | Validate interop kit. |
| T-GUI-001 | TODO | Validate workstation GUI kit. |
| T-ENV-001 | TODO | Validate environment/CLI kit. |
| T-CK-003 | TODO | Stabilize kits from validation evidence. |
| T-CK-004 | TODO | Final tracking update. |
| T-FPX-001 | DONE | Forward solver preset体系完成；3D auto 路径转向 AMG-family。 |
| T-FPX-002 | DONE | Solver/PC/options 已加入 backend cache signature。 |
| T-FPX-003 | DONE | Forward KSP/multi-RHS reuse tests, 3D GAMG smoke, KSP setup count, and PC reuse diagnostics complete。 |
| T-FPX-004 | DONE | Forward solver benchmark artifact emits setup/solve/iteration/device diagnostics。 |
| T-FPX-005 | DONE | Matrix-free Jacobian action object 已新增。 |
| T-FPX-006 | DONE | GN fast linear solver 已接入 `JacobianLinearization`/operator input。 |
| T-FPX-007 | DONE | Matrix-free Hessian diagonal/NOSER/prior PC contract complete。 |
| T-FPX-008 | DONE | Pmat/coarse/custom inverse PC smoke complete。 |
| T-FPX-009 | DONE | `sigma + z_contact` block-ready metadata/action/update helpers complete。 |
| T-FPX-010 | DONE | CUDA/MPI capability diagnostics and fallback reporting complete。 |
| T-FPX-011 | DONE | Full unit category shards, GUI/hardware split, and shard docs complete。 |
| T-FPX-012 | DONE | Continuity docs and implementation tracking reconciled after execution waves。 |

## Known Gaps

- Hardware-in-loop and MATLAB/EIDORS checks require external dependencies.
- WSLg XCB/Wayland clarity/stability lacks a dedicated automated acceptance
  test.
- `tests/unit/test_interop_geometry_exchange.py` uses `@pytest.mark.fenics`,
  but `pyproject.toml` registers `fenicsx`; warnings-as-errors makes collection
  fail.
- Large 3D MPI production remains future work; current forward model records
  MPI size/rank and fails fast for size > 1.

## Test Health

| Check | Command | Result |
| --- | --- | --- |
| T-CK-001 collect-only | `nix --option warn-dirty false develop --command bash -lc "python -m pytest -q --collect-only -o addopts= ..."` | FAIL: 96 tests collected, 1 collection error from unknown `fenics` marker. |
| T-FPX-010 capability/diagnostic gate | `nix develop -c uv run pytest --no-cov tests/unit/test_perf_capabilities_selection.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_script_entrypoint_acceleration_profiles.py -q` | PASS: 37 passed. |
| T-FPX-010 forward/core shards | `nix develop -c uv run python scripts/ci/run_sharded_unit_tests.py --run --shard forward --timeout 300`; `... --shard core-misc --timeout 300` | PASS: forward 15 files, core-misc 12 files. |
| T-FPX-003 reuse diagnostic follow-up | `nix develop -c uv run pytest --no-cov tests/unit/test_forward_solver_presets.py tests/unit/test_forward_solver_branch_suite.py tests/unit/test_forward_mat_solve_policy.py tests/unit/test_forward_vectorized_runtime.py tests/unit/test_forward_petsc_multirhs.py tests/unit/test_forward_petsc_helper_branches.py tests/unit/test_forward_model_3d_cem.py tests/unit/test_script_entrypoint_acceleration_profiles.py -q`; `... run_sharded_unit_tests.py --run --shard forward --timeout 300 --report-dir test_results/sharded_unit/tfpx003_reuse_diag_followup_final` | PASS: 50 passed; forward shard passed. |

## Session Log

### 2026-04-19

- Started `/ck:make`.
- Completed `T-CK-001` collection pass.
- Updated `context/kits/validation-report.md` with per-file collection status.
- Left production code unchanged; marker fix deferred to a later task or human
  decision.

### 2026-04-20

- Completed FEniCSx/PETSc EIT refactor task chain T-FPX-001 through T-FPX-012.
- Revisited T-FPX-003 and added explicit KSP setup/preconditioner reuse diagnostics to runtime and benchmark artifacts.
- Canonical 2D/3D forward/inverse solver and PC matrix lives in
  `context/plans/fenicsx-petsc-eit-2d-3d-implementation-details.md`.
- Detailed validation history, dead ends, and next-step rules live in
  `context/impl/impl-fenicsx-petsc-eit-refactor.md`.
