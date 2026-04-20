---
status: draft
source: from-code
domain: forward-solver
---

# Cavekit: Forward Solver

## Scope

This kit covers CEM forward solve behavior, backend selection, PETSc/SciPy
fallbacks, CUDA PETSc routing, and forward-solver diagnostics.

## Requirements

### R1: Complete Electrode Model forward solves are deterministic

**Description:** Given the same mesh, patterns, conductivity, contact impedance,
and drive semantics, the forward solver returns stable voltage predictions.

**Acceptance Criteria:**
- [ ] 2D and 3D CEM smoke tests produce finite voltages.
- [ ] Contact impedance and drive configuration changes affect cache keys and
  outputs.
- [ ] Forward solve views do not mutate source conductivity unexpectedly.

**Dependencies:** `cavekit-geometry-electrodes.md`, `cavekit-data-and-units.md`

### R2: Backend policy is explicit

**Description:** Users can select or auto-resolve PETSc, SciPy, and structured
CUDA forward paths without silent unsupported behavior.

**Acceptance Criteria:**
- [ ] `linear_backend="petsc"` uses PETSc when available.
- [ ] SciPy fallback is used only when policy allows it.
- [ ] Unsupported backend names fail with a clear error.
- [ ] Structured CUDA backend is gated by mesh/backend compatibility.

**Dependencies:** `cavekit-environment-cli.md`

### R3: PETSc CUDA routing is probe-gated

**Description:** CUDA routing is enabled only when PETSc can actually create CUDA
matrix/vector types in the active shell.

**Acceptance Criteria:**
- [ ] `petsc_device="cuda"` fails fast if PETSc CUDA is unavailable.
- [ ] `petsc_device="auto"` records fallback reason when CUDA is unavailable.
- [ ] Probe diagnostics report matrix, vector, dense matrix, and error details.

**Dependencies:** `cavekit-environment-cli.md`

### R4: Forward setup caches are semantically safe

**Description:** Repeated forward solves can reuse static setup artifacts only
when semantic inputs match.

**Acceptance Criteria:**
- [ ] Static setup cache hits do not change numerical output.
- [ ] Background conductivity and backend configuration participate in
  invalidation.
- [ ] Cache stats expose hit/miss behavior for diagnostics.

**Dependencies:** `cavekit-cache-performance.md`

## Brownfield Evidence

- Source: `src/pyeidors/forward/eit_forward_model.py`
- Source: `src/pyeidors/forward/cuda_structured_backend.py`
- Source: `src/pyeidors/forward/process_setup_cache.py`
- Source: `scripts/diagnostics/probe_petsc_cuda.py`
- Tests: `tests/unit/test_forward_model_3d_cem.py`
- Tests: `tests/unit/test_forward_mat_solve_policy.py`
- Tests: `tests/unit/test_forward_petsc_multirhs.py`
- Tests: `tests/integration/test_cuda_structured_pipeline_parity.py`

## Out of Scope

- Inverse solver iteration policy; see Inverse Reconstruction.
- Environment provisioning; see Environment and CLI.

## Cross-References

- Depends on: `cavekit-geometry-electrodes.md`
- Depends on: `cavekit-data-and-units.md`
- Related: `cavekit-cache-performance.md`
- Related: `cavekit-environment-cli.md`

