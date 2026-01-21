# Corn Stem Difference Imaging - Command

## Date
2025-12-11

## Command
```bash
cd /root/shared && python scripts/run_single_step_diff_realdata.py \
  --csv data/measurements/EIT_DEV_Test/2025-09-23-00-01-56_10_10.00_100uA_2000Hz.csv \
  --background-sigma 0.15 \
  --lambda 0.5 \
  --transparent \
  --step-size-calibration \
  --output results/corn_stem_imaging/difference
```

## Parameters
| Flag | Value | Description |
|------|-----|------|
| `--csv` | ... | Measurement CSV |
| `--background-sigma` | 0.15 | Background conductivity (S/m) used for the Jacobian |
| `--lambda` | 0.5 | Regularization parameter |
| `--output` | ... | Output directory |

## Metric
| Metric | Value |
|------|-----|
| RMSE (abs) | 0.27193 |

## Data
- File: `2025-09-23-00-01-56_10_10.00_100uA_2000Hz.csv`
- Drive current: 100 µA
- Frequency: 2000 Hz
- Electrodes: 16
- CSV format: 4 columns = 2 frames × (real + imag)
  - Col 0: frame 0 real (background)
  - Col 1: frame 0 imag
  - Col 2: frame 1 real (target)
  - Col 3: frame 1 imag

## EIDORS-style difference imaging
- Uses normalized stimulation amplitude (1.0 A)
- Output is relative Δσ (dimensionless)
