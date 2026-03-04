# Unit Consistency Checklist (Pre-Experiment Guard)

Use this checklist before running absolute or difference experiments with physical interpretation.

## Quick command

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

The command exits non-zero if any blocking item fails.

## Five automatic checks

1. **Drive config validity**
- Validates `drive_mode`, `drive_value`, `geometry_scale_to_m`, and 2D restrictions.

2. **Geometry scale consistency**
- Converts mesh extents to meters and checks positivity/finite values.
- If `--expected-domain-size-m` is provided, checks relative deviation (default threshold 5%).

3. **Electrode physical length consistency**
- Ensures all per-electrode physical lengths are present, finite, and positive.

4. **Current conservation**
- Verifies each stimulation pattern has near-zero net injected current.
- Threshold: `abs(sum(I_pattern)) <= 1e-12 * max(1, max_abs(I_pattern))`.

5. **Current density closure (`line_current_density` only)**
- Verifies `I_e / L_e` recovers `drive_value * weight_e` within tolerance.
- Default relative tolerance: `1e-8`.

## Expected outcomes

- `INFO`: passed check.
- `WARN`: near-threshold behavior (typically numerical edge noise).
- `ERROR`: blocking mismatch, experiment should not proceed.

## Scale invariance regression

Cross-unit invariance is also tested in CI:

- [test_unit_scale_invariance_mm_cm_m.py](/Users/tom/workspace/PyEIDORS/tests/integration/test_unit_scale_invariance_mm_cm_m.py)

This verifies the same physical object represented as `m`, `cm`, `mm` yields consistent boundary voltages.
