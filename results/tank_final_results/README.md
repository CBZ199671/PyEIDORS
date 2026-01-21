# Tank EIT Imaging Results (Paper Demo)

## Date
2025-12-11

## Data source
- **Device**: EIT measurement system
- **Tank diameter**: 6 cm (radius 0.03 m)
- **Electrodes**: 16
- **Drive current**: 50 µA @ 3 kHz
- **Measurement gain**: 10.0
- **Input file**: `data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv`

---

## Directory structure

```
tank_final_results/
├── README.md                    # This file
├── absolute_imaging/            # Absolute imaging result
│   ├── COMMAND.md              # Command log
│   ├── conductivity.png        # Conductivity map
│   ├── prediction_vs_measurement.png
│   ├── result_arrays.npz       # NumPy arrays
│   └── run_summary.json        # Run summary
├── difference_imaging/          # Difference imaging (EIDORS-style, λ=0.9)
│   ├── COMMAND.md              # Command log
│   ├── diff_comparison.png
│   ├── voltage_comparison.png
│   └── outputs.npz             # NumPy arrays
├── difference_imaging_lam1.5/   # Difference imaging (EIDORS-style, λ=1.5, paper result)
│   ├── COMMAND.md
│   ├── diff_comparison.png
│   ├── voltage_comparison.png
│   ├── reconstruction.png
│   └── outputs.npz
└── difference_imaging_physical_bg0008_lam0.9/  # Difference imaging (physical amplitude, paper result)
    ├── COMMAND.md
    ├── diff_comparison.png
    ├── voltage_comparison.png
    ├── reconstruction.png
    └── outputs.npz
```

---

## Absolute imaging

### Command
```bash
python scripts/run_gn_absolute_eidors_style.py \
  --background-sigma 0.002 \
  --lambda 0.5 \
  --output-dir results/tank_final_results/absolute_imaging \
  --max-iter 20
```

### Key parameters
| Parameter | Value |
|------|-----|
| Background conductivity | 0.002 S/m |
| Regularization λ | 0.5 |
| Iterations | 20 |

### Metrics
| Metric | Value |
|------|-----|
| **RMSE** | 6.48e-04 |
| **Correlation** | 0.992 |
| **Conductivity range** | [0.0002, 0.0023] S/m |

---

## Difference imaging

### Command
```bash
python scripts/run_single_step_diff_realdata.py \
  --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv \
  --background-sigma 0.008 \
  --lambda 0.9 \
  --output results/tank_final_results/difference_imaging
```

### Key parameters
| Parameter | Value |
|------|-----|
| Background conductivity | 0.008 S/m |
| Regularization λ | 0.9 |

### Metric
| Metric | Value |
|------|-----|
| RMSE (abs) | 5.07 |

---

## Algorithm notes

### EIDORS-style Gauss-Newton
- **Update**: `dx = -(J'WJ + λ²RtR)⁻¹(J'W·dv + λ²RtR·de)`
- **NOSER prior**: `R = diag(J'J)^0.5`
- **Line search**: multi-point sampling [0, 1/16, 1/8, 1/4, 1/2, 1] × max_step

### Notes (implementation details)
1. NOSER prior uses an exponent parameter (default 0.5, like EIDORS)
2. GN includes the prior error term `de = σ - σ_prior`
3. Objective includes measurement residual + prior residual
4. Removed the internal scaling_factor logic

---

## Notes

1. **Choosing background conductivity**
   - Absolute imaging: tune to make the Meas/Model ratio close to 1
   - Difference imaging: more flexible since it focuses on changes

2. **Regularization**
   - λ=0.5 is often a reasonable trade-off
   - Larger λ: smoother, may lose details
   - Smaller λ: sharper, may introduce artifacts

3. **Measurement gain**
   - Must match the instrument setting (10.0 in this dataset)
   - A wrong gain will cause large model/data mismatch
