# Tank Difference Imaging (λ=1.5) - Command

This directory contains one of the paper's tank difference-imaging results (EIDORS-style single-step GN).

## Command
```bash
cd /root/shared && python scripts/run_single_step_diff_realdata.py \
  --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
  --background-sigma 0.008 \
  --lambda 1.5 \
  --output results/tank_final_results/difference_imaging_lam1.5
```

## Input file
- `data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv`
  - Frame 0: background (homogeneous medium)
  - Frame 1: target (with inclusion)
