# Prior Travelling-Wave EIDORS Metric Review

- schema: `pyeidors-prior-travelling-wave-benchmark-v1`
- source_json: `reports/runtime_benchmarks/prior_travelling_wave_eidors_review_20260426.json`
- old best RMSE: `laplace`
- old best center RMSE: `tv_irls`

## Per-Metric Winners

| metric | winner | value |
|---|---|---:|
| rmse | curvature | 0.0461301 |
| center_rmse | tv_irls | 0.00157012 |
| speed_abs_error | tv_irls | 0.00439478 |
| peak_time_mean_abs_error | curvature | 0.00465839 |
| AR_error | tv_irls | 0.000109886 |
| PE | tv_irls | 0.00133208 |
| RES | tv_irls | 0.266927 |
| SD | curvature | 0 |
| RNG | tv_irls | 0.0140579 |
| NF | curvature | 1.86051e+14 |
| solution_error | curvature | 0.00270878 |

## Method Values

| method | RMSE | center RMSE | AR err | PE | RES | SD | RNG | NF | solution error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| curvature | 0.0461301 | 0.00570699 | 0.0002413 | 0.00499486 | 0.273438 | 0 | 0.045498 | 1.86051e+14 | 0.00270878 |
| graph_ltl | 0.0461301 | 0.00570699 | 0.0002413 | 0.00499486 | 0.273438 | 0 | 0.045498 | 1.86051e+14 | 0.00270878 |
| laplace | 0.0461301 | 0.00570699 | 0.0002413 | 0.00499486 | 0.273438 | 0 | 0.045498 | 1.86051e+14 | 0.00270878 |
| tv_irls | 0.0658625 | 0.00157012 | 0.000109886 | 0.00133208 | 0.266927 | 0 | 0.0140579 | 2.65635e+14 | 0.00545313 |

## Recheck

- Laplace/graph_ltl/curvature remain numerically identical in this fixture; their official metric values tie.
- TV-IRLS still improves center tracking, but Laplace-family methods keep better RMSE/solution-error in this zero-noise fixture.
- Therefore the earlier conclusion needs nuance: TV-IRLS helps spatiotemporal localization, not every EIDORS-style image/data metric.
