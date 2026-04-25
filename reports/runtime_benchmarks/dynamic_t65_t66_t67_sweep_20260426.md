# Dynamic Sweep: T65 4D GN vs T66 TV/Huber vs T67 Kalman

- schema: `pyeidors-dynamic-t65-t66-t67-sweep-v1`
- created_utc: `2026-04-25T16:57:05.248394+00:00`
- fixture: `travelling_wave` / `neural_fast_conduction`
- n_cells/n_frames/n_measurements: `32/32/20`
- lambda_s: `0.08`
- temporal_order: `2`
- noise_std: `0.002`
- gate: peak_delay<=`0.05`, rmse_ratio<=`1.08`

## Recommended Fast-Conduction Regions

- T66 lambda_t range: `0.2..0.2`
- T66 huber_delta range: `0.01..0.01`
- T66 gate-passing rows: `29/36`
- T67 fixed_lag range: `0..3`
- T67 process_noise Q range: `0.02..0.08`
- T67 measurement_noise R range: `0.01..0.04`
- T67 gate-passing rows: `0/80`; if zero, the range is the best-scored fallback region.
- best overall score: `t66_spatiotemporal_tv_huber`

## Best Points

| method | params | score | speed err | peak MAE | onset MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| T65 4D GN | lambda_t=0.02 | 0.0987728 | 0.018295 | 0.00115207 | 0.0230415 | 0.0429929 |
| T66 TV/Huber | lambda_t=0.2, delta=0.01 | 0.0345572 | 0.00581881 | 0 | 0.0126728 | 0.0247595 |
| T67 Kalman | lag=0, Q=0.08, R=0.01 | 0.100045 | 0.018574 | 0.00230415 | 0.0184332 | 0.0440071 |

## Top T67 Kalman Lag/Q/R Rows

| lag | Q | R | latency | score | speed err | peak MAE | onset MAE | RMSE ratio vs T65 | speed delta vs T66 | pass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.08 | 0.01 | 0 | 0.100045 | 0.018574 | 0.00230415 | 0.0184332 | 1.02359 | -0.0127552 | no |
| 2 | 0.08 | 0.01 | 0.0645161 | 0.103684 | 0.0187135 | 0.00115207 | 0.0230415 | 1.02002 | -0.0128947 | no |
| 3 | 0.08 | 0.01 | 0.0967742 | 0.103762 | 0.0187188 | 0.00115207 | 0.0230415 | 1.02036 | -0.0129 | no |
| 1 | 0.08 | 0.01 | 0.0322581 | 0.104924 | 0.0186828 | 0.00230415 | 0.0230415 | 1.01762 | -0.012864 | no |
| 1 | 0.04 | 0.01 | 0.0322581 | 0.10745 | 0.0190271 | 0.00115207 | 0.0230415 | 1.03547 | -0.0132083 | no |
| 1 | 0.08 | 0.02 | 0.0322581 | 0.107488 | 0.019032 | 0.00115207 | 0.0230415 | 1.03573 | -0.0132132 | no |
| 0 | 0.04 | 0.01 | 0 | 0.108603 | 0.0187893 | 0.00460829 | 0.0172811 | 1.06591 | -0.0129705 | no |
| 0 | 0.08 | 0.02 | 0 | 0.108646 | 0.0187941 | 0.00460829 | 0.0172811 | 1.06622 | -0.0129753 | no |
| 2 | 0.04 | 0.01 | 0.0645161 | 0.108836 | 0.0191289 | 0.00115207 | 0.0230415 | 1.04215 | -0.0133101 | no |
| 2 | 0.08 | 0.02 | 0.0645161 | 0.108875 | 0.019134 | 0.00115207 | 0.0230415 | 1.04241 | -0.0133151 | no |
| 3 | 0.04 | 0.01 | 0.0967742 | 0.109233 | 0.0191562 | 0.00115207 | 0.0230415 | 1.04404 | -0.0133374 | no |
| 3 | 0.08 | 0.02 | 0.0967742 | 0.109272 | 0.0191612 | 0.00115207 | 0.0230415 | 1.04431 | -0.0133424 | no |

## Top T66 TV/Huber Rows

| lambda_t | huber_delta | score | speed err | peak MAE | onset MAE | RMSE ratio | pass |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.2 | 0.01 | 0.0345572 | 0.00581881 | 0 | 0.0126728 | 0.512446 | yes |
| 0.12 | 0.01 | 0.0449141 | 0.00426154 | 0.00576037 | 0.0103687 | 0.543677 | yes |
| 0.2 | 0.02 | 0.05149 | 0.00932605 | 0 | 0.014977 | 0.589541 | yes |
| 0.35 | 0.01 | 0.0521831 | 0.00959941 | 0.00345622 | 0.016129 | 0.550911 | yes |
| 0.12 | 0.02 | 0.0589628 | 0.00636265 | 0.00691244 | 0.0138249 | 0.601743 | yes |
| 0.12 | 0.03 | 0.065877 | 0.00831252 | 0.00576037 | 0.014977 | 0.655494 | yes |
| 0.08 | 0.01 | 0.067059 | 0.0040757 | 0.014977 | 0.0115207 | 0.659842 | yes |
| 0.2 | 0.03 | 0.0683624 | 0.0124782 | 0.00230415 | 0.016129 | 0.658272 | yes |
| 0.08 | 0.02 | 0.0684238 | 0.0049662 | 0.0115207 | 0.0138249 | 0.688069 | yes |
| 0.08 | 0.03 | 0.0698228 | 0.00634501 | 0.00921659 | 0.0138249 | 0.71882 | yes |
| 0.04 | 0.05 | 0.0713914 | 0.00680481 | 0.00576037 | 0.0126728 | 0.946535 | yes |
| 0.08 | 0.05 | 0.0726195 | 0.00889126 | 0.00576037 | 0.0138249 | 0.787944 | yes |

## Notes

- T65 baseline is L2 spatiotemporal GN at each `lambda_t`.
- T66 uses Huber IRLS over spatial graph differences and temporal differences.
- T67 uses the cached Laplace RM as a state observation, then applies online Kalman filtering plus optional fixed-lag RTS smoothing.
- Lower score favours propagation-speed, peak-time, onset-time fidelity first; RMSE ratio is a guard, not the main objective.
