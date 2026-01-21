# Tank Difference Imaging (physical amplitude, bg=0.008, λ=0.9) - Command

This directory contains one of the paper's tank difference-imaging results using a physical-amplitude forward model.

## Command
```bash
cd /root/shared && python scripts/run_single_step_diff_realdata.py \
  --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
  --background-sigma 0.008 \
  --lambda 0.9 \
  --pattern-amplitude 5e-05 \
  --output results/tank_final_results/difference_imaging_physical_bg0008_lam0.9
```

## Notes
- `--pattern-amplitude 5e-05` corresponds to 50 µA (physical drive current), used to match the voltage scale.
