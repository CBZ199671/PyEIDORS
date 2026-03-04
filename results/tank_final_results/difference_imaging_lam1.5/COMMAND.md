# Tank Difference Imaging (λ=1.5) - Command

This directory contains one of the paper's tank difference-imaging results (EIDORS-style single-step GN).

## Command
```bash
cd /root/shared && python scripts/run_reconstruction_unified.py \
  --method gn-difference \
  --input-mode paired \
  --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
  --reference-col 0 \
  --target-col 2 \
  --background-sigma 0.008 \
  --lambda 1.5 \
  --output-root results/tank_final_results
```

## Input file
- `data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv`
  - Frame 0: background (homogeneous medium)
  - Frame 1: target (with inclusion)
