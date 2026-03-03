# Corn dataset (2025-09-22-07-38-23_3_10.00_20uA_2000Hz) — four combinations

Common settings: `scripts/run_difference_single_step.py`, lambda=0.7, NOSER exponent=0.8, step-size calibration on, baseline weights, DirectJacobian (efficient), GPU.

## swap-reference-target = True
- diff_orientation = reference_minus_target
  - Dir: results/old_diff_refminus_lam0.7_noser0.8_corn/...
  - corr ≈ 0.337, RMSE(diff) ≈ 8.46e-05, step_size ≈ 1.5e-05, scale ≈ 0.945
- diff_orientation = target_minus_reference
  - Dir: results/old_diff_tgtminus_lam0.7_noser0.8_corn/...
  - corr ≈ 0.90, RMSE(diff) ≈ 3.92e-05, step_size ≈ 1.5e-05, scale ≈ 0.986

## swap-reference-target = False
- diff_orientation = target_minus_reference
  - Dir: results/combos_corn_swapFalse_tgtminus/...
  - corr ≈ 0.337, RMSE(diff) ≈ 8.46e-05, step_size ≈ 1.5e-05, scale ≈ 0.945
- diff_orientation = reference_minus_target
  - Dir: results/combos_corn_swapFalse_refminus/...
  - corr ≈ 0.90, RMSE(diff) ≈ 3.92e-05, step_size ≈ 1.5e-05, scale ≈ 0.986

Observation: swapping or changing orientation toggles sign; two cases give high corr (~0.90) with smaller RMSE (~3.9e-05), two cases give low corr (~0.34) with larger RMSE (~8.5e-05). Step_size stays tiny (~1.5e-05), indicating strong regularization / small update.
