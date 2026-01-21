# Sparse Bayesian Difference Imaging (tank, physical amplitude, bg=0.008) - Command

This directory contains a sparse Bayesian (difference) reconstruction on the tank dataset, using the same physical-amplitude modeling settings as:
`results/tank_final_results/difference_imaging_physical_bg0008_lam0.9/`.

## Command

```bash
cd /root/shared && python scripts/run_sparse_bayesian_reconstruction.py \
  --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
  --metadata data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.yaml \
  --mode difference \
  --reference-col 0 \
  --target-col 2 \
  --difference-calibration after \
  --measurement-gain 10 \
  --background-sigma 0.008 \
  --mesh-dir eit_meshes \
  --mesh-name mesh_16e_r0p025_ref10_cov0p5 \
  --mesh-radius 0.025 \
  --electrode-coverage 0.5 \
  --contact-impedance 1e-5 \
  --subspace-rank 64 \
  --linear-warm-start \
  --coarse-group-size 40 \
  --coarse-levels 96 48 \
  --coarse-iterations 1 \
  --block-iterations 2 \
  --block-size 64 \
  --solver fista \
  --use-gpu \
  --gpu-dtype float32 \
  --output-root results/tank_final_results/sparse_bayesian_physical_bg0008_v1_0
```

## Notes

- Physical drive current is read from the metadata (`amplitude: 5e-05` for 50 µA).
- The CSV is assumed to be amplifier-scaled; `--measurement-gain 10` converts it back to physical units.
- This run uses `difference-calibration=after`, which calibrates the difference vector after subtraction.

