---
status: draft
source: from-code
domain: data-and-units
---

# Cavekit: Data and Units

## Scope

This kit covers EIT data structures, real measurement dataset ingestion, per-frame
recording format, drive semantics, and unit consistency checks.

## Requirements

### R1: Standard measurement dataset contract

**Description:** Real measurement data can be supplied as a measurement matrix
plus metadata that fully defines pattern and drive semantics.

**Acceptance Criteria:**
- [ ] Measurements accept one frame or multiple frames with shape consistent
  with the configured stimulation and measurement pattern.
- [ ] Metadata includes electrode count, stimulation pattern, measurement
  pattern, drive mode, drive value, geometry scale, measurement-current flags,
  rotation behavior, and frame count.
- [ ] Shape or metadata mismatches fail with actionable context.

**Dependencies:** `cavekit-geometry-electrodes.md`

### R2: Drive semantics are explicit

**Description:** Current drive configuration distinguishes line current density,
total current, and normalized drive to avoid ambiguous amplitude use.

**Acceptance Criteria:**
- [ ] `line_current_density` is interpreted as A/m with electrode length and
  geometry scale considered.
- [ ] `total_current` is interpreted as A.
- [ ] `normalized` is accepted for algorithmic comparisons.
- [ ] Obsolete ambiguous amplitude fields are rejected or ignored according to
  the documented contract.

**Dependencies:** `cavekit-forward-solver.md`

### R3: Frame I/O preserves acquisition traceability

**Description:** Recorded GUI frames preserve voltage arrays and metadata needed
for later reconstruction and database indexing.

**Acceptance Criteria:**
- [ ] Per-frame CSV and YAML records round-trip into frame models.
- [ ] Multi-frame recordings retain frame index and timestamp metadata.
- [ ] Legacy frame formats remain readable where compatibility tests exist.

**Dependencies:** `cavekit-workstation-gui.md`

### R4: Unit consistency checks guard physical setups

**Description:** Physical 2D setups are checked for incompatible drive mode,
geometry scale, electrode length, and model assumptions before reconstruction
or diagnostics proceed.

**Acceptance Criteria:**
- [ ] Unit check reports expose pass/fail status and human-readable findings.
- [ ] Invalid 2D physical setups are detected before producing misleading
  results.
- [ ] Scale-invariance integration tests remain stable for mm, cm, and m inputs.

**Dependencies:** `cavekit-forward-solver.md`

## Brownfield Evidence

- Source: `src/pyeidors/data/measurement_dataset.py`
- Source: `src/pyeidors/data/frame_io.py`
- Source: `src/pyeidors/physics/current_drive.py`
- Source: `src/pyeidors/physics/unit_consistency.py`
- Docs: `docs/MEASUREMENT_DATA_SPEC.md`
- Tests: `tests/unit/test_measurement_dataset.py`
- Tests: `tests/unit/test_measurement_dataset_copy_policy.py`
- Tests: `tests/unit/test_frame_io_legacy_compat.py`
- Tests: `tests/integration/test_unit_scale_invariance_mm_cm_m.py`

## Out of Scope

- Hardware transport protocol details; see Workstation GUI.
- Mesh generation details; see Geometry and Electrodes.

## Cross-References

- Depends on: `cavekit-geometry-electrodes.md`
- Depends on: `cavekit-forward-solver.md`
- Depended on by: `cavekit-workstation-gui.md`
- Depended on by: `cavekit-inverse-reconstruction.md`

