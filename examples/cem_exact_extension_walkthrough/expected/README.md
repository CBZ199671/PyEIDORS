# Frozen report evidence

These files are byte-for-byte copies of the evidence used by
`docs/benchmarks/cem_exact_extension_report.md`:

- `cem_exact_extension_metrics.csv`: 228 rows =
  38 cases × 3 FEM solvers × 2 CEM formulations.
- `cem_exact_extension_timing.csv`: 228 rows with matched cold, setup, and
  retained-state timing summaries.

Evidence snapshot date: 2026-07-20.

SHA-256:

```text
16ce692bfebdef17e866a00e279375a11ebdffac2ab9faf993a114b3c198d1ce  cem_exact_extension_metrics.csv
799003b1ed5663364a833e7341115678c026ba73f96dee0aabb22df9ce776dfe  cem_exact_extension_timing.csv
```

Run `export_walkthrough_assets.py` after a new complete experiment to replace
these snapshots deliberately.
