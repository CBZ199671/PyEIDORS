# Dynamic Validation: 4D GN vs Rowwise RM

- schema: `pyeidors-dynamic-validation-benchmark-v1`
- created_utc: `2026-04-25T15:04:55.224346+00:00`
- n_cells/n_frames/n_measurements: `32/32/20`
- lambda_s/lambda_t: `0.08/0.08`
- temporal_order: `2`
- noise_std: `0.002`
- peak_delay_gate: `passed=True` max_delay=`0.0322581` tolerance=`0.16`

## Fixture Summary

| fixture | rmse rowwise RM | rmse 4D GN | delta rowwise-4D | speed err rowwise | speed err 4D | peak MAE rowwise | peak MAE 4D | rel L2 vs rowwise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| travelling_wave | 0.0427355 | 0.0437737 | -0.00103823 | 0.0186226 | 0.017762 | 0.00230415 | 0.00115207 | 0.0104779 |
| plant_slow_pulse | 0.00683518 | 0.00651256 | 0.000322615 | 0.0570454 | 0.0441677 | 0.00868486 | 0.00372208 | 0.011324 |

## Mean Deltas

- mean_rmse_delta_rowwise_minus_4d: `-0.000357807`
- mean_speed_error_delta_rowwise_minus_4d: `0.00686913`
- mean_peak_time_mae_delta_rowwise_minus_4d: `0.00305743`

## Method Notes

- `rowwise RM`: independent framewise Laplace RM solve with the same spatial prior and measurement contract.
- `4D GN`: one windowed block normal solve with spatial `Rs` and temporal `Dt.T @ Dt` prior.
- Positive delta means 4D GN improved that error metric relative to rowwise RM.
