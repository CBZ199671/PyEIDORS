# EIDORS <-> PyEIDORS Interop

This document standardizes the same-geometry bridge between EIDORS and PyEIDORS so the two frameworks can exchange native meshes and run cross-framework validation with minimal manual setup.

Python-side reusable helpers live in [geometry_exchange.py](/D:/workspace/PyEIDORS/src/pyeidors/interop/geometry_exchange.py), exposed as the public module `pyeidors.interop`.

## Scope

- 2D triangular meshes
- Complete electrode model (CEM)
- Adjacent stimulation / adjacent measurement (`{ad}` / `{ad}`)
- Difference imaging benchmark workflow

## Standard Exchange Format

The bridge uses a MATLAB `.mat` payload with `exchange_format = 'eidors_pyeidors_bridge_v1'`.

Required fields:

| Field | Meaning |
| --- | --- |
| `exchange_format` | Format version identifier. |
| `source_framework` | `eidors` or `pyeidors`. |
| `nodes` | Node coordinates, shape `N x 2`, MATLAB 1-based convention is not needed for coordinates. |
| `elems` | Triangle connectivity, shape `M x 3`, **1-based** indexing. |
| `boundary_edges` | Boundary edges, shape `K x 2`, **1-based** indexing. |
| `electrode_nodes` | Padded matrix of electrode node ids, shape `L x P`, **1-based** indexing. |
| `electrode_node_counts` | Number of active nodes in each `electrode_nodes` row. |
| `n_elec` | Number of electrodes. |
| `background` | Background conductivity. |
| `truth_elem_data` | Element-wise conductivity truth on the exported mesh. |
| `contact_impedance` | Scalar contact impedance used for all electrodes in the benchmark case. |
| `mesh_name` | Human-readable mesh identifier. |
| `mesh_level` | `coarse`, `medium`, or `fine`. |
| `scenario_name` | `low_z` or `high_z`. |

Optional metadata may be added by either side, but consumers should not rely on it unless documented.

## Standard Scripts

### PyEIDORS -> EIDORS

- Export PyEIDORS native geometry: [export_geometry_from_pyeidors.py](/D:/workspace/PyEIDORS/scripts/interop/export_geometry_from_pyeidors.py)
- Import into EIDORS and reconstruct: [import_geometry_from_pyeidors.m](/D:/workspace/PyEIDORS/compare_with_Eidors/import_geometry_from_pyeidors.m)

### EIDORS -> PyEIDORS

- Export EIDORS native geometry: [export_geometry_from_eidors.m](/D:/workspace/PyEIDORS/compare_with_Eidors/export_geometry_from_eidors.m)
- Import into PyEIDORS and reconstruct: [import_geometry_from_eidors.py](/D:/workspace/PyEIDORS/scripts/interop/import_geometry_from_eidors.py)

## One-Command Workflow

Use [run_same_geometry_bridge.ps1](/D:/workspace/PyEIDORS/scripts/interop/run_same_geometry_bridge.ps1) on the Windows host side.

Examples:

```powershell
.\scripts\interop\run_same_geometry_bridge.ps1 -Direction both -MeshLevel medium -Scenario low_z
```

```powershell
.\scripts\interop\run_same_geometry_bridge.ps1 -Direction both -MeshLevel medium -Scenario low_z -RenderPlots
```

```powershell
.\scripts\interop\run_same_geometry_bridge.ps1 -Direction pyeidors_to_eidors -MeshLevel fine -Scenario high_z -ElectrodeCoverage 0.5 -RenderPlots
```

Default outputs are written to:

```text
docs/benchmarks/interop/
```

When `-RenderPlots` is enabled, the bridge also emits:

- `*_details.mat` for framework-specific reconstruction details
- `*_conductivity.png` for conductivity-distribution comparison
- `*_voltage.png` for boundary-voltage fitting

## Environment Split

- Python / PyEIDORS side: executed in WSL2 Docker container `pyeidors`
- MATLAB / EIDORS side: executed on the Windows host
- File exchange side: repository-local `.mat`, `.csv`, and `.json` artifacts

This split mirrors the current development setup and keeps both toolchains reproducible.

## Validation Rule

For a successful same-geometry bridge, the imported-framework result should match the source-framework self result up to numerical tolerance in both:

- `voltage_rmse`
- `conductivity_rmse`

The bridge is considered validated when any previous large asymmetry disappears after geometry alignment.
