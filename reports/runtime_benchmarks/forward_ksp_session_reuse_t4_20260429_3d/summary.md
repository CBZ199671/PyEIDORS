# T4 — Forward KSP session reuse benchmark (G1 evidence)

- generated: `2026-04-29T09:30:07Z`
- env_path: `/usr/bin/env`
- mesh_dim: 3, n_elec: 16, n_iter: 10, sigma_noise_scale: 0.03
- solver_preset: `3d_hypre`, ksp_type: `auto`, pc_type: `auto`, petsc_device: `cpu`

| regime | calls | reused | refresh | cum_setup_s | first_setup_s | subseq_mean_s | iter_max_mean | iter_max_p95 | total_setups |
|--------|------:|-------:|--------:|------------:|--------------:|--------------:|--------------:|-------------:|-------------:|
| auto | 10 | 9 | 1 | 0.328849 | 0.036067 | 0.032531 | 1.00 | 1.0 | 1 |
| never | 10 | 0 | 10 | 0.359796 | 0.035427 | 0.036041 | 1.00 | 1.0 | 1 |

**G1 cumulative setup saved (never − auto)**: `0.030947s`
**warm/cold setup ratio (auto / never)**: `0.9140`

## refresh_reasons
- `auto`: {"initial_setup": 1, "reused": 9}
- `never`: {"initial_setup": 10}

V cites: V13, V14, V52, V67, V80
