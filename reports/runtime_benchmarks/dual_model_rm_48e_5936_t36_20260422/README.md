# 48e/5936 Dual-Model RM Runtime Report

- schema: `pyeidors-dual-model-rm-v1-benchmark`
- scope: 48e/5936 RM-layer benchmark; forward/J cold path cited from real spd_gamg CUDA reports
- generated: 2026-04-22T03:32:03.775333+00:00
- git: `1709304`

## Forward Reference

- solver: `spd_gamg` / `cg` + `gamg`
- PETSc device: `cuda`; matSolve: `vec-loop`
- setup seconds: 0.111076; solve seconds: 4.768245
- lazy context seconds: 3.384371
- lazy jacobian seconds: 1.233017

## RM Build And Load

| algorithm | rm build s | artifact load s |
|---|---:|---:|
| noser | 84.925926 | 0.030400 |
| laplace | 73.075975 | 0.028112 |
| greit | 2.606761 | 0.057517 |

## Online Apply

| algorithm | device | prepare s | 1 frame s | batch frames | batch s | effective device | resident |
|---|---|---:|---:|---:|---:|---|---|
| noser | cpu | 0.003997 | 0.030261 | 512 | 0.178917 | cpu | cpu |
| noser | cuda | 0.071544 | 0.103497 | 512 | 0.051057 | cuda | device |
| laplace | cpu | 0.003861 | 0.004875 | 512 | 0.069919 | cpu | cpu |
| laplace | cuda | 0.002885 | 0.001757 | 512 | 0.036958 | cuda | device |
| greit | cpu | 0.013482 | 0.013912 | 512 | 0.085797 | cpu | cpu |
| greit | cuda | 0.002770 | 0.001352 | 512 | 0.033325 | cuda | device |

## Previous GREIT Baseline

| device | previous 512-frame s | current 512-frame s | speedup |
|---|---:|---:|---:|
| cpu | 36.752693 | 0.085797 | 428.37x |
| cuda | 35.927631 | 0.033325 | 1078.09x |

## GREIT Metrics

| metric | value |
|---|---:|
| AR | 0.03999997046219719 |
| PE | 0.03705085140112853 |
| RES | 0.5263632998046983 |
| SD | 0.047619047619047616 |
| RNG | 0.2351136505301192 |
