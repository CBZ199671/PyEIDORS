# Tank Difference Imaging (physical amplitude, bg=0.008, λ=0.9) - Command

This directory contains one of the paper's tank difference-imaging results using a physical-amplitude forward model.

## Command
```bash
cd /root/shared && python scripts/run_reconstruction_unified.py \
  --method gn-difference \
  --input-mode paired \
  --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
  --reference-col 0 \
  --target-col 2 \
  --background-sigma 0.008 \
  --lambda 0.9 \
  --output-root results/tank_final_results
```

## Notes
- Drive mode/current comes from workflow defaults and input metadata handling in the unified runner.
