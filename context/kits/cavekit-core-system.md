---
status: draft
source: from-code
domain: core-system
---

# Cavekit: Core System

## Scope

This kit covers the public orchestration behavior exposed by PyEIDORS for setting
up EIT geometry, running forward solves, running inverse solves, and controlling
runtime caches. It describes observable behavior, not class layout.

## Requirements

### R1: Explicit setup paths

**Description:** Users can initialize an EIT workflow from an existing mesh,
from a named cache entry, or from generated geometry without hidden fallback that
masks setup failure.

**Acceptance Criteria:**
- [ ] Given an explicit mesh object, setup produces a system ready for forward
  and inverse workflows.
- [ ] Given `mesh_source="cache"` and a missing cache entry, setup reports a
  setup failure rather than silently generating a replacement mesh.
- [ ] Given `mesh_source="generated"` with valid geometry options, setup
  produces a mesh compatible with configured electrode and pattern counts.

**Dependencies:** `cavekit-geometry-electrodes.md`, `cavekit-forward-solver.md`

### R2: Typed solver outputs

**Description:** Solver APIs return structured outputs whose fields can be
validated without relying on ad-hoc dictionaries.

**Acceptance Criteria:**
- [ ] Forward workflows expose voltage data and metadata needed by inverse
  workflows.
- [ ] Inverse workflows expose reconstructed conductivity/image data and
  diagnostics.
- [ ] Invalid solver configuration raises a deterministic error naming the bad
  option.

**Dependencies:** `cavekit-forward-solver.md`, `cavekit-inverse-reconstruction.md`

### R3: Cache controls on public workflows

**Description:** Users can inspect and clear process and disk caches through the
public workflow surface.

**Acceptance Criteria:**
- [ ] Cache stats include hit/miss counters and footprint information.
- [ ] Clearing `process`, `disk`, or `both` removes the requested layer without
  corrupting the other layer.
- [ ] Corrupted disk cache payloads are removed and recomputed rather than
  crashing the workflow.

**Dependencies:** `cavekit-cache-performance.md`

### R4: Backward-compatible facade workflows

**Description:** Higher-level absolute, difference, and sparse workflows remain
available through a stable public entrypoint while internal modules evolve.

**Acceptance Criteria:**
- [ ] Existing unit tests for workflow wrapper branches pass.
- [ ] Existing scripts can call the unified reconstruction workflow without
  importing private modules.
- [ ] Unsupported workflow names fail with a clear supported-option list.

**Dependencies:** `cavekit-inverse-reconstruction.md`, `cavekit-environment-cli.md`

## Brownfield Evidence

- Source: `src/pyeidors/core_system.py`
- Source: `src/pyeidors/core_system_facade.py`
- Tests: `tests/unit/test_core_setup_contract.py`
- Tests: `tests/unit/test_workflow_wrapper_branches.py`
- Tests: `tests/integration/test_recon_unified_cli_smoke.py`

## Out of Scope

- Numerical details of CEM assembly; see Forward Solver.
- Solver algorithm internals; see Inverse Reconstruction.
- GUI orchestration; see Workstation GUI.

## Cross-References

- Depends on: `cavekit-geometry-electrodes.md`
- Depends on: `cavekit-forward-solver.md`
- Depends on: `cavekit-inverse-reconstruction.md`
- Depends on: `cavekit-cache-performance.md`

