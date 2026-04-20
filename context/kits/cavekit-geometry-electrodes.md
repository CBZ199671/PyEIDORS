---
status: draft
source: from-code
domain: geometry-electrodes
---

# Cavekit: Geometry and Electrodes

## Scope

This kit covers mesh generation, mesh loading, electrode placement conventions,
and stimulation/measurement pattern behavior needed by forward and inverse
workflows.

## Requirements

### R1: Mesh generation supports 2D and 3D EIT geometry

**Description:** The system can create or load EIT meshes with electrode tags and
metadata sufficient for CEM workflows.

**Acceptance Criteria:**
- [ ] 2D generated meshes include boundary/electrode information required by
  CEM setup.
- [ ] 3D cylindrical generated meshes support configured radius, height,
  electrode count, and ring layout.
- [ ] Mesh loaders preserve node, element, boundary, and electrode associations.

**Dependencies:** None

### R2: Electrode layout has stable conventions

**Description:** Electrode positions follow documented conventions so generated
meshes, imported meshes, and tests agree.

**Acceptance Criteria:**
- [ ] Y-axis start convention tests pass for current electrode placement.
- [ ] Electrode ring fractions for 3D layouts are deterministic for a given
  configuration.
- [ ] Invalid electrode counts or geometry values fail before mesh generation
  produces unusable output.

**Dependencies:** None

### R3: Pattern generation validates measurement ordering

**Description:** Stimulation and measurement pattern expansion produces a stable
flattened measurement order for simulation, real data, and GUI frame handling.

**Acceptance Criteria:**
- [ ] Adjacent stimulation/measurement patterns produce the expected measurement
  count.
- [ ] Measurement-current exclusion and rotation flags affect ordering
  deterministically.
- [ ] The dataset contract and GUI measurement layout helpers agree on counts.

**Dependencies:** `cavekit-data-and-units.md`

### R4: Process caches can reuse mesh artifacts safely

**Description:** Repeated workflows can reuse loaded/generated mesh artifacts
without stale geometry crossing into incompatible configurations.

**Acceptance Criteria:**
- [ ] Mesh cache keys include geometry, electrode, and code-relevant metadata.
- [ ] Clearing mesh caches causes later workflows to recompute or reload.
- [ ] Cache warm-start tests show repeated compatible setups reuse artifacts.

**Dependencies:** `cavekit-cache-performance.md`

## Brownfield Evidence

- Source: `src/pyeidors/geometry/optimized_mesh_generator.py`
- Source: `src/pyeidors/geometry/mesh3d_generator.py`
- Source: `src/pyeidors/geometry/mesh_loader.py`
- Source: `src/pyeidors/electrodes/layout.py`
- Source: `src/pyeidors/electrodes/patterns.py`
- Tests: `tests/unit/test_electrode_position_y_axis.py`
- Tests: `tests/unit/test_mesh3d_cylinder_generator.py`
- Tests: `tests/unit/test_measurement_projection_finite_guard.py`
- Tests: `tests/integration/test_3d_cache_warmup_speed.py`

## Out of Scope

- PETSc matrix assembly; see Forward Solver.
- EIDORS bridge payload format; see Interop.

## Cross-References

- Depended on by: `cavekit-forward-solver.md`
- Depended on by: `cavekit-data-and-units.md`
- Related: `cavekit-cache-performance.md`

