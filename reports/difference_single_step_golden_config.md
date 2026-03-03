# Difference Single-Step – “Golden” Configuration (Field Data)

This records the settings that yielded the high-correlation fit on complex field measurements (non‑tank).

## Script
- `scripts/run_difference_single_step.py`

## Example command (EIT_DEV field data)
```bash
python scripts/run_difference_single_step.py \
  --csv data/measurements/EIT_DEV_Test/2025-09-23-00-01-56_10_10.00_100uA_2000Hz.csv \
  --metadata data/measurements/EIT_DEV_Test/2025-09-23-00-01-56_10_10.00_100uA_2000Hz.yaml \
  --difference-hyperparameter 0.1 \
  --step-size-calibration \
  --diff-orientation reference_minus_target \
  --output-dir results/old_diff_refminus_try
```

## Key settings
- Jacobian: `DirectJacobian` (`efficient` pathway)
- Difference orientation: `reference_minus_target`
- λ (`difference-hyperparameter`): `0.1`
- Step-size search: enabled (step_size ≈ 10)
- Measurement weights: baseline
- NOSER exponent: `0.5`
- Conductivity clamp: `[1e-6, 10]`
- Post-calibration: scale ≈ `0.889`, bias ≈ `-3.46e-05`

## Outputs (for the above run)
- Directory: `results/old_diff_refminus_try/2025-09-23-00-01-56_10-10.00_100uA_2000Hz/`
- Correlation ≈ `0.9888`
- RMSE(diff) ≈ `4.45e-05` (MAE ≈ `2.27e-05`)
- Files: `diff_comparison.png`, `voltage_comparison.png`, `reconstruction.png`, `metrics.json`, `outputs.npz`

## Notes
- This configuration was tuned for field data (complex environment), not the simpler tank cases.
- The deleted helper script `scripts/run_difference_single_step_parity.py` is no longer needed; use the settings above with `run_difference_single_step.py`.
