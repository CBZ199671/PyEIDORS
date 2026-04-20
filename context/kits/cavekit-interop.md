---
status: draft
source: from-code
domain: interop
---

# Cavekit: Interop

## Scope

This kit covers PyEIDORS/EIDORS exchange formats, MATLAB/EIDORS environment
detection, bridge packages, and same-geometry validation workflows.

## Requirements

### R1: Geometry exchange format is explicit

**Description:** PyEIDORS and EIDORS exchange geometry through a versioned
payload containing nodes, elements, boundary edges, electrode nodes, counts, and
scenario metadata.

**Acceptance Criteria:**
- [ ] Required bridge fields are present when exporting geometry.
- [ ] Imported geometry validates indexing conventions before reconstruction.
- [ ] Optional metadata can be ignored without breaking required behavior.

**Dependencies:** `cavekit-geometry-electrodes.md`

### R2: Same-geometry validation compares both frameworks fairly

**Description:** Bridge validation removes geometry mismatch as a confounder and
checks voltage and conductivity residuals against source/self results.

**Acceptance Criteria:**
- [ ] PyEIDORS-to-EIDORS and EIDORS-to-PyEIDORS directions can be run
  independently.
- [ ] Validation reports include voltage RMSE and conductivity RMSE.
- [ ] Large asymmetry after geometry alignment is treated as a validation
  failure or investigation item.

**Dependencies:** `cavekit-forward-solver.md`, `cavekit-inverse-reconstruction.md`

### R3: Desktop Interop Hub exposes bridge workflow safely

**Description:** GUI users can locate MATLAB/EIDORS, generate bridge assets,
import/export bundles, and run smoke validation without blocking the main UI
indefinitely.

**Acceptance Criteria:**
- [ ] Environment detection reports missing MATLAB or EIDORS paths clearly.
- [ ] Bridge package import/export preserves geometry and measurement metadata.
- [ ] GUI smoke tests cover Interop Hub actions without requiring MATLAB.

**Dependencies:** `cavekit-workstation-gui.md`

## Brownfield Evidence

- Source: `src/pyeidors/interop/geometry_exchange.py`
- Source: `src/eit_app/interop/`
- Source: `src/eit_app/ui/dialogs/interop_hub_dialog.py`
- Docs: `docs/EIDORS_PYEIDORS_INTEROP.md`
- Tests: `tests/unit/test_interop_geometry_exchange.py`
- Tests: `tests/unit/test_eit_app_interop_environment.py`
- Tests: `tests/unit/test_eit_app_interop_hub.py`
- Tests: `tests/integration/test_compare_3d_eidors_alignment.py`

## Out of Scope

- Installing MATLAB or EIDORS.
- Redesigning EIDORS algorithms.

## Cross-References

- Depends on: `cavekit-geometry-electrodes.md`
- Depends on: `cavekit-forward-solver.md`
- Depends on: `cavekit-inverse-reconstruction.md`
- Related: `cavekit-workstation-gui.md`

