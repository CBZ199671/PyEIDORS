# Dynamic EIDORS Metric Review

- schema: `pyeidors-dynamic-eidors-metric-review-v1`
- created_utc: `2026-04-26T06:16:43.374927+00:00`
- scenarios: `6`
- statement: Previous dynamic-score conclusions are broadly supported by the EIDORS-aligned metric majority, but individual metric trade-offs remain visible.

## Scenario Summary

| source | noise | seed | legacy best | official winner counts | official majority supports legacy |
|---|---:|---:|---|---|:---:|
| dynamic_t65_t66_t67_eidors_review_default_20260426.json | 0.002 | 20260425 | T66 | T65:1, T66:6 | yes |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260426_20260426.json | 0.01 | 20260426 | T66 | T66:5, T67:2 | yes |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260427_20260426.json | 0.01 | 20260427 | T66 | T66:4, T67:3 | yes |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260428_20260426.json | 0.01 | 20260428 | T66 | T66:5, T67:2 | yes |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260429_20260426.json | 0.01 | 20260429 | T66 | T66:5, T67:2 | yes |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260430_20260426.json | 0.01 | 20260430 | T66 | T66:4, T67:3 | yes |

## Per-Metric Winners

| source | AR err | PE | RES | SD | RNG | NF | solution error |
|---|---|---|---|---|---|---|---|
| dynamic_t65_t66_t67_eidors_review_default_20260426.json | T65 4.14701e-06 | T66 0.0011031 | T66 0.225586 | T66 0 | T66 0.00330288 | T66 52.7921 | T66 0.00171867 |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260426_20260426.json | T66 0.000177302 | T66 0.00982749 | T66 0.226562 | T67 0.0192026 | T67 0.00787811 | T66 15.4512 | T66 0.0081795 |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260427_20260426.json | T67 4.64298e-05 | T66 0.00730706 | T66 0.21875 | T67 0.00694444 | T67 0.00676686 | T66 15.8477 | T66 0.00846612 |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260428_20260426.json | T66 0.00156866 | T66 0.00488263 | T66 0.220703 | T67 0.00347222 | T67 0.0112274 | T66 16.2535 | T66 0.00825598 |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260429_20260426.json | T67 2.5154e-06 | T66 0.0073701 | T66 0.212891 | T66 0.00837054 | T67 0.0111121 | T66 15.5678 | T66 0.00861567 |
| dynamic_t65_t66_t67_eidors_review_high_noise_seed_20260430_20260426.json | T67 3.88829e-05 | T66 0.00604219 | T66 0.22168 | T67 0.00347222 | T67 0.00607879 | T66 16.0061 | T66 0.00883105 |

## Notes

- EIDORS does not define a single scalar score for AR/PE/RES/SD/RNG/NF/solution-error, so this report reviews per-metric winners instead of replacing them with RMSE.
- `AR err` is `abs(AR - 1)`. Lower is better for every column in this review.
- The previous dynamic score is still useful for propagation timing, but it is no longer treated as the whole quality story.
