# EIDORS ↔ PyEIDORS source-semantics acceptance

Date: 2026-07-30
Result: **PASS (24/24 checks)**

Runtime:

- MATLAB R2023b
- EIDORS 3.12
- PyEIDORS `complex64-cuda` pure-Nix development shell

Reproduction:

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#complex64-cuda --command \
  python scripts/interop/run_eidors_source_semantics_acceptance.py \
  --output <new-output-directory> \
  --matlab '/mnt/d/Program Files/MATLAB/R2023b/bin/matlab.exe' \
  --eidors-startup \
  '/mnt/d/Program Files/MATLAB/R2023b/toolbox/eidors-v3.12-ng/eidors/startup.m'
```

## EIDORS → PyEIDORS

| Case | Source model | Imported geometry | Electrodes | Protocol | Result |
| --- | --- | --- | --- | --- | --- |
| 2D CEM | `mk_common_model('c2C2',16)` | 313 nodes, 576 triangles | 16 exact surface CEM | 208 measurements | finite forward PASS |
| 3D CEM | `ng_mk_cyl_models` | 1117 nodes, 4506 tetrahedra | 8 exact surface CEM | 40 measurements | finite forward PASS |
| 3D PEM | `mk_common_model('a3cr',16)` | 205 nodes, 768 tetrahedra | 16 one-node PEM | 208 measurements | default BLOCK; explicit incident-facet projection PASS |
| Missing fields | `mk_common_model('c2C2',8)` | 313 nodes, 576 triangles | 8 surface CEM | 40 measurements | geometry PASS; equivalent forward BLOCK |

Verified source semantics:

- PEM raw stimulation maximum: `0.02 A`.
- EIDORS `fwd_model.current_density`: `2.0`.
- Effective PyEIDORS stimulation maximum: `0.01 A`.
- PEM background resistivity `0.5 Ω·source-unit` mapped by EIDORS to
  conductivity `2.0`.
- PEM target first-element resistivity `0.25` mapped to conductivity `4.0`.
- PEM forward without opt-in returned the expected
  `point_or_distributed_point_electrode_requires_explicit_projection_opt_in`
  blocker.
- Missing-field case preserved all eight absent `z_contact` values as
  `NaN` with `contact_impedance_present=false`.
- Missing `gnd_node` recorded the standard first-order center-node runtime
  derivation.
- Missing normalization recorded the live EIDORS `mdl_normalize` runtime
  source.
- No contact impedance, background conductivity, target conductivity, or
  stimulation amplitude was fabricated.
- A workspace with multiple unrelated forward-model candidates failed with an
  actionable `fwd_model_var` selector message.
- Re-running the same script with `fwd_model_var=fmdl_b` deterministically
  selected the 16-electrode model; because that example intentionally has no
  image, background remained missing and equivalent forward stayed blocked.

## PyEIDORS → EIDORS

A native PyEIDORS 3D package was rebuilt and solved by the same real
MATLAB/EIDORS installation:

| Check | Result |
| --- | --- |
| Geometry | 1425 nodes, 6215 tetrahedra |
| Boundary | 320 facets, exact |
| Electrodes | 8, exact |
| Stimulation/measurement matrices | exact |
| Measurements | 40 |
| Forward values finite | PASS |
| Forward measurement count exact | PASS |

The reverse acceptance report returned
`eidors_bridge_import_acceptance_v1.status="passed"`.

## Repository gates

- Ruff format and lint: PASS (717 Python files checked).
- Full unit suite without coverage enforcement: PASS
  (2099 passed, 394 skipped).
- The default coverage run had the same 2099 passed and 394 skipped with no
  test failures, but the repository-wide total was `85.88%`, below the
  existing `87%` gate. The uncovered lines are distributed across existing
  CUDA, CLI, realtime, and experimental modules and were not disguised by
  adding out-of-scope tests.

## Claim boundary

This gate proves deterministic model discovery, field provenance, geometry,
electrode, conductivity, stimulation/measurement transfer, and executable
forward paths for the declared cases. It does not claim arbitrary custom
MATLAB solver portability or inverse-reconstruction numerical parity.
