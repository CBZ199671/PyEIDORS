# Brownfield Validation Report

## Status

Initial sketch generated from the existing codebase. `T-CK-001` collect-only
verification was run on 2026-04-19 in the default Nix shell. Collection is
partially verified: pytest collected 96 tests but stopped with one collection
error in `tests/unit/test_interop_geometry_exchange.py`.

Command:

```bash
nix --option warn-dirty false develop --command bash -lc \
  "python -m pytest -q --collect-only -o addopts= \
   tests/unit/test_core_setup_contract.py \
   tests/unit/test_measurement_dataset.py \
   tests/unit/test_electrode_position_y_axis.py \
   tests/unit/test_forward_model_3d_cem.py \
   tests/unit/test_difference_semantics.py \
   tests/unit/test_cache_manager_extended.py \
   tests/unit/test_interop_geometry_exchange.py \
   tests/unit/test_eit_app_gui_smoke.py \
   tests/unit/test_env_manifest_verify.py"
```

Observed result:

- Exit code: 1.
- Collection before interruption: 96 tests.
- Error: `PytestUnknownMarkWarning: Unknown pytest.mark.fenics` at
  `tests/unit/test_interop_geometry_exchange.py:82`, promoted to error by
  `filterwarnings = ["error"]`.
- Root cause: `pyproject.toml` registers `fenicsx`, not `fenics`; the test file
  uses `@pytest.mark.fenics`.

## Covered By Existing Test Inventory

| Kit | Evidence |
| --- | --- |
| Core System | `test_core_setup_contract.py`, `test_workflow_wrapper_branches.py`, unified CLI integration tests |
| Data and Units | measurement dataset tests, frame I/O compatibility tests, unit scale invariance integration test |
| Geometry and Electrodes | mesh generation tests, electrode Y-axis convention test, 3D cylinder tests |
| Forward Solver | 3D CEM tests, PETSc helper tests, CUDA structured backend tests |
| Inverse Reconstruction | GN tests, difference semantics tests, sparse Bayesian tests, reduced GN tests |
| Cache and Performance | cache manager/signature/store tests, perf policy tests, perf gate integration tests |
| Interop | geometry exchange tests, interop environment and hub tests |
| Workstation GUI | GUI smoke tests, acquisition/database/hardware transport tests |
| Environment and CLI | env sync/manifest tests, WSL2/CUDA docs and probes, CLI smoke integration tests |

## Gaps To Verify In Next Pass

- Execute representative smoke tests for each domain before marking kits stable.
- Add explicit source-to-kit `CLAUDE.md` mappings under major source directories
  if Cavekit adoption continues beyond sketch.
- Add plan files after a human reviews whether these kits describe desired
  behavior or merely current behavior.
- Hardware-in-loop criteria remain partially covered because physical devices
  are not always available in CI.
- Wayland-vs-XCB clarity/stability behavior is documented but needs a dedicated
  reproducible GUI smoke if it becomes product-critical.
- Interop collect gap: register the `fenics` pytest marker, rename the marker to
  existing `fenicsx`, or otherwise document the legacy-FEniCS test skip policy.

## T-CK-001 Collection Status

| File | Result | Notes |
| --- | --- | --- |
| `tests/unit/test_core_setup_contract.py` | COLLECTED | Listed by pytest before interruption. |
| `tests/unit/test_measurement_dataset.py` | COLLECTED | Listed by pytest before interruption. |
| `tests/unit/test_electrode_position_y_axis.py` | COLLECTED | Listed by pytest before interruption. |
| `tests/unit/test_forward_model_3d_cem.py` | COLLECTED | Listed by pytest before interruption. |
| `tests/unit/test_difference_semantics.py` | COLLECTED | Listed by pytest before interruption. |
| `tests/unit/test_cache_manager_extended.py` | COLLECTED | Listed by pytest before interruption. |
| `tests/unit/test_interop_geometry_exchange.py` | ERROR | Unknown marker `fenics` at line 82 under warnings-as-errors. |
| `tests/unit/test_eit_app_gui_smoke.py` | COLLECTED | Listed by pytest before interruption. |
| `tests/unit/test_env_manifest_verify.py` | COLLECTED | Listed by pytest before interruption. |

## Recommended Smoke Set

```bash
pytest tests/unit/test_core_setup_contract.py
pytest tests/unit/test_measurement_dataset.py
pytest tests/unit/test_electrode_position_y_axis.py
pytest tests/unit/test_forward_model_3d_cem.py
pytest tests/unit/test_difference_semantics.py
pytest tests/unit/test_cache_manager_extended.py
pytest tests/unit/test_interop_geometry_exchange.py
pytest tests/unit/test_eit_app_gui_smoke.py
pytest tests/unit/test_env_manifest_verify.py
```
