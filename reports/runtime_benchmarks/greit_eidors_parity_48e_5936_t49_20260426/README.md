# 48e/5936 EIDORS-Parity GREIT Runtime Gate

- schema: `pyeidors-greit-eidors-parity-48e-v1-benchmark`
- generated: 2026-04-26T14:36:07.854295+00:00
- git: `d374424`
- official EIDORS fixture: `False`
- parity components passed: `True`
- official-equivalence claim allowed: `False`

## Fixture Note

No external MATLAB/EIDORS 48e fixture was supplied; this run uses a deterministic EIDORS-compatible surrogate to exercise the full PyEIDORS benchmark path.

## Cases

| case | fixture | parity | bad ch | W | load s | metric PE | metric RES |
|---|---|---:|---:|---|---:|---:|---:|
| bad_weighted | generated_synthetic_eidors_surrogate | True | 192 | diagonal | 2.265816 | 0.16025356321290934 | 0.3262389700974053 |

## Online Apply

| case | device | effective | resident | 1 frame s | 512 frame s | forward solves | KSP solves |
|---|---|---|---|---:|---:|---:|---:|
| bad_weighted | cpu | cpu | cpu | 0.003116 | 0.051371 | 0 | 0 |
| bad_weighted | auto | cuda | device | 0.120942 | 0.126497 | 0 | 0 |
| bad_weighted | cuda | cuda | device | 0.004829 | 0.133575 | 0 | 0 |

## Cold Build

| case | finite responses s | desired D s | RM build s | artifact write s | parity compare s |
|---|---:|---:|---:|---:|---:|
| bad_weighted | 9.923667 | 0.000959 | 0.267523 | 12.087613 | 2.918235 |
