# Current Drive Migration Guide

This document describes the hard cutover from legacy `amplitude` to explicit current-drive semantics.

## Breaking Changes

1. `PatternConfig.amplitude` was removed.
2. Metadata field `amplitude` is no longer accepted by `MeasurementDataset.from_metadata`.
3. Stimulation configuration must now use:
   - `drive_mode`: `"line_current_density" | "total_current" | "normalized"`
   - `drive_value`: numeric magnitude
   - `geometry_scale_to_m`: geometry scale to meters
   - `electrode_length_m_override` (optional)

## Why This Change

Using a single scalar amplitude hides physical units and can produce order-of-magnitude drift when geometry units change (`m` vs `cm`). Explicit drive semantics avoid these silent modeling errors.

## Old vs New

Old:

```python
PatternConfig(
    n_elec=16,
    stim_pattern="{ad}",
    meas_pattern="{ad}",
    amplitude=5e-5,
)
```

New (physical 2D):

```python
PatternConfig(
    n_elec=16,
    stim_pattern="{ad}",
    meas_pattern="{ad}",
    drive_mode="line_current_density",
    drive_value=5e-5,
    geometry_scale_to_m=1.0,
)
```

New (dimensionless benchmark):

```python
PatternConfig(
    n_elec=16,
    stim_pattern="{ad}",
    meas_pattern="{ad}",
    drive_mode="normalized",
    drive_value=1.0,
    geometry_scale_to_m=1.0,
)
```

## Metadata Update

Old metadata:

```yaml
n_elec: 16
stim_pattern: "{ad}"
meas_pattern: "{ad}"
amplitude: 0.02
```

New metadata:

```yaml
n_elec: 16
stim_pattern: "{ad}"
meas_pattern: "{ad}"
drive_mode: line_current_density
drive_value: 0.02
geometry_scale_to_m: 1.0
# electrode_length_m_override: [0.01, ...]  # optional
```

## Validation Rules

- `geometry_scale_to_m` must be positive.
- In `line_current_density` mode, 2D topology is required.
- Per-electrode lengths must be positive.
- `electrode_length_m_override` must be scalar positive or a list with `n_elec` positive values.

## Pre-Experiment Consistency Check

Run the unit guard before experiments:

```bash
nix --extra-experimental-features "nix-command flakes" develop -c \
  /Users/tom/workspace/PyEIDORS/.venv/bin/python \
  scripts/diagnostics/check_unit_consistency.py \
  --mesh-source cache \
  --mesh-dir eit_meshes \
  --drive-mode line_current_density \
  --drive-value 5e-5 \
  --geometry-scale-to-m 1.0 \
  --strict
```

Checklist details and interpretation:
- `docs/UNIT_CONSISTENCY_CHECKLIST.md`
