# 48e/5936 EIDORS-Parity GREIT Runtime Gate

- schema: `pyeidors-greit-eidors-parity-48e-v1-benchmark`
- generated: 2026-04-26T15:48:30.410775+00:00
- git: `934b0c6`
- official EIDORS fixture: `True`
- parity components passed: `True`
- official-equivalence claim allowed: `True`

## Cases

| case | fixture | parity | bad ch | W | load s | metric PE | metric RES |
|---|---|---:|---:|---|---:|---:|---:|
| reduced_48e_5936 | reports/eidors_greit_fixtures/reduced_48e_5936_eidors_greit_fixture.mat | True | 0 | identity | 0.220094 | 1.736945890683578e-08 | 0.020833333333333332 |

## Online Apply

| case | device | effective | resident | 1 frame s | 512 frame s | forward solves | KSP solves |
|---|---|---|---|---:|---:|---:|---:|
| reduced_48e_5936 | cpu | cpu | cpu | 0.000456 | 0.025483 | 0 | 0 |
| reduced_48e_5936 | auto | cuda | device | 0.067558 | 0.025785 | 0 | 0 |
| reduced_48e_5936 | cuda | cuda | device | 0.000913 | 0.021373 | 0 | 0 |

## Cold Build

| case | finite responses s | desired D s | RM build s | artifact write s | parity compare s |
|---|---:|---:|---:|---:|---:|
| reduced_48e_5936 | n/a | n/a | n/a | 1.427543 | 0.586510 |
