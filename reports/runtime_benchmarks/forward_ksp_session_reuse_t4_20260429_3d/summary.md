# T4 — Forward KSP session reuse benchmark (G1 evidence)

- generated: `2026-04-29T08:52:54Z`
- env_path: `/nix/store/xs8scz9w9jp4hpqycx3n3bah5y07ymgj-coreutils-9.8/bin/env`
- mesh_dim: 3, n_elec: 16, n_iter: 10, sigma_noise_scale: 0.03
- solver_preset: `3d_hypre`, ksp_type: `auto`, pc_type: `auto`, petsc_device: `cpu`

| regime | calls | reused | refresh | cum_setup_s | first_setup_s | subseq_mean_s | iter_max_mean | iter_max_p95 | total_setups |
|--------|------:|-------:|--------:|------------:|--------------:|--------------:|--------------:|-------------:|-------------:|
| auto | 10 | 9 | 1 | 0.346307 | 0.038396 | 0.034212 | 1.00 | 1.0 | 1 |
| never | 10 | 0 | 10 | 0.369072 | 0.037652 | 0.036824 | 1.00 | 1.0 | 1 |

**G1 cumulative setup saved (never − auto)**: `0.022766s`  
**warm/cold setup ratio (auto / never)**: `0.9383`

## refresh_reasons
- `auto`: {"initial_setup": 1, "reused": 9}
- `never`: {"initial_setup": 10}

V cites: V13, V14, V52, V67
