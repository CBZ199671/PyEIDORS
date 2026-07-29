# EIDORS migration examples

These scripts are ordinary EIDORS scripts and intentionally contain no
PyEIDORS-specific export calls.

- `eidors_2d_quickstart.m`: 2D surface CEM, adjacent protocol, one inclusion.
- `eidors_3d_quickstart.m`: 3D surface CEM, adjacent protocol, one sphere.
- `eidors_3d_point_electrode_quickstart.m`: 3D PEM with an exact
  `0.02 / current_density(2) = 0.01 A` effective drive and
  resistivity-to-conductivity mapping.
- `eidors_missing_fields_semantics.m`: deliberately missing contact,
  ground-node, and normalization source fields; verifies that capture does
  not invent values.
- `eidors_multiple_models_requires_selector.m`: two unrelated standard
  forward models; automatic capture must stop until `--fwd-model-var` is
  supplied.
- `pyeidors_3d_export.py`: native PyEIDORS 3D model → EIDORS Bridge Package.
- `validate_bridge_in_eidors.m`: rebuild and forward-solve an exported package
  in real MATLAB/EIDORS, then write a machine-readable acceptance report.

Capture either script from the repository Nix shell:

```bash
pyeidors-interop capture examples/interop/eidors_3d_quickstart.m \
  --output output/eidors_3d_quickstart_bridge \
  --matlab '<MATLAB executable>' \
  --eidors-startup '<EIDORS startup.m>'
```

Then run:

```bash
pyeidors-interop validate output/eidors_3d_quickstart_bridge
pyeidors-interop import-geometry output/eidors_3d_quickstart_bridge --forward-smoke
```

For the reverse direction, generate a package with `pyeidors_3d_export.py`,
then run this in MATLAB:

```matlab
addpath('examples/interop');
validate_bridge_in_eidors('output/pyeidors_3d_bridge/run_in_eidors.m');
```

Maintainers can combine the real 2D/3D packages and the MATLAB/EIDORS report
with `scripts/interop/build_acceptance_report.py`.

The complete real source-semantics matrix is:

```bash
python scripts/interop/run_eidors_source_semantics_acceptance.py \
  --output output/eidors_source_semantics_acceptance \
  --matlab '<MATLAB executable>' \
  --eidors-startup '<EIDORS startup.m>'
```

See `docs/EIDORS_PYEIDORS_INTEROP.md` for the full Chinese quickstart and the
Bridge/Geometry v2 contract.
