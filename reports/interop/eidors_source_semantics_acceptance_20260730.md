# EIDORS ↔ PyEIDORS source-semantics acceptance

Date: 2026-07-30
Result: **PASS (28/28 checks)**

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
| 3D PEM | `mk_common_model('a3cr',16)` | 205 nodes, 768 tetrahedra | 16 exact one-node PEM, no projection | 208 measurements | native PEM forward PASS |
| Missing fields | `mk_common_model('c2C2',8)` | 313 nodes, 576 triangles | 8 surface CEM | 40 measurements | geometry PASS; equivalent forward BLOCK |

Verified source semantics:

- PEM raw stimulation maximum: `0.02 A`.
- EIDORS `fwd_model.current_density`: `2.0`.
- Effective PyEIDORS stimulation maximum: `0.01 A`.
- PEM background resistivity `0.5 Ω·source-unit` mapped by EIDORS to
  conductivity `2.0`.
- PEM target first-element resistivity `0.25` mapped to conductivity `4.0`.
- Native PyEIDORS PEM uses the exact EIDORS source vertices for injection and
  measurement (`Q=N2EᵀI`, `U=N2E·u`) without incident-facet projection.
- EIDORS versus PyEIDORS relative L2 was `4.9174e-7` for the homogeneous field
  and `4.7870e-7` for the heterogeneous target.
- DOLFINx source-cell reordering was applied automatically through
  `topology.original_cell_index`; source element 0 was local element 182 in
  this real fixture.
- Changing every PEM `z_contact` to approximately `1e9` changed both PyEIDORS
  forward vectors by exactly zero. Contact impedance is therefore preserved
  only as provenance/required EIDORS structure, not used in PEM physics.
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

A native PyEIDORS PEM package, using a labeled finite structural
`z_contact` placeholder, was rebuilt and solved by the same real
MATLAB/EIDORS installation:

| Check | Result |
| --- | --- |
| Geometry | 205 nodes, 768 tetrahedra |
| Boundary | 256 facets, exact |
| Electrodes | 16 exact singleton-node PEM electrodes |
| Stimulation/measurement matrices | exact |
| Measurements | 208 |
| Electrode projection | none |
| Contact impedance applicability | false |
| EIDORS z-invariance check | exact (`max_abs=0`) |
| Forward values finite | PASS |
| Forward measurement count exact | PASS |

The reverse acceptance report returned
`eidors_bridge_import_acceptance_v1.status="passed"`.

## Repository gates

- Ruff format and lint: PASS (716 Python files checked).
- Full unit suite without coverage enforcement: PASS
  (2103 passed, 450 skipped).
- The first default coverage run exposed 32 legacy bare/mock CEM fixtures that
  lacked the new model attributes; after restoring the CEM defaults, all 70
  affected PETSc/KSP branch tests passed. That first run also reported the
  pre-existing repository-wide coverage debt (`85.24% < 87%`); no out-of-scope
  tests were added to disguise it.

## Claim boundary

This gate proves deterministic model discovery, field provenance, exact PEM
source-node and source-cell transfer, geometry, conductivity,
stimulation/measurement identity, native forward numerical parity, and
bidirectional PEM execution for the declared cases. It does not claim
arbitrary custom MATLAB solver portability or inverse-reconstruction
numerical parity.
