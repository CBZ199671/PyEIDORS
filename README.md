# PyEIDORS

<p align="center">
  <img src="pictures/Fig.%204.%20fig_absolute_vs_difference.png" alt="PyEIDORS banner" width="900" />
</p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
![License](https://img.shields.io/badge/license-MIT-green)
![Backend](https://img.shields.io/badge/backend-FEniCSx%20(DOLFINx)-orange)
![Accel](https://img.shields.io/badge/accel-PyTorch-red)

A Python-first EIT framework with a FEniCSx (DOLFINx) Complete Electrode Model (CEM) backend and PyTorch acceleration, designed to be familiar to EIDORS users while enabling modern GPU/differentiable workflows.

SoftwareX manuscript in preparation; citation info will be added after acceptance.

## Why PyEIDORS

- **Numerical Consistency**: Matches EIDORS-style workflows with verified simulation parity.
- **Modern Architecture**: Hybrid FEniCSx (FEM) + PyTorch (Inverse/Accel) design.
- **Modular & Extensible**: `EITSystem` coordinator makes it easy to replace geometry, forward models, or solvers.
- **Research Ready**: End-to-end scripts for absolute & difference reconstruction, real-time mesh generation, and benchmarking.

---

## Quick Start

PyEIDORS uses **Nix + uv** as the primary development path for FEniCSx:

```bash
git clone https://github.com/CBZ199671/PyEIDORS.git
cd PyEIDORS
nix develop
uv pip install --python .venv/bin/python --no-deps -e .
python -c "import dolfinx, basix, ufl; print(dolfinx.__version__)"
```

Then run a quick workflow check:

```bash
python scripts/run_synthetic_parity.py --output-root results/simulation_parity/run03 --mode both --difference-solver single-step --gn-regularization 1e-11
```

For full setup, validation, and troubleshooting, see `docs/NIX_FENICSX.md`.

Legacy Docker notes are archived in `docs/archive/DOCKER_LEGACY.md`.

> Hard-cut note: the runtime is now **FEniCSx-only** in `src/pyeidors/**`. Legacy DOLFIN compatibility aliases are removed.

### Phase-2 API Notes (Breaking)

- `EITSystem.setup()` no longer auto-falls back from cache loading to generation.
- Use explicit setup paths:
  - `system.setup(mesh=eit_mesh)`
  - `system.setup(mesh_source="cache", mesh_dir="eit_meshes", mesh_name="mesh_...")`
  - `system.setup(mesh_source="generated", radius=1.0, mesh_size=0.1)`
- Solver APIs now return typed `SolverOutput` objects (not ad-hoc dictionaries).

---

## Gallery & Validation

### Modern Architecture: FEniCSx + PyTorch

<p align="center">
  <img src="pictures/Fig.%201.%20pyeidors_architecture.png" alt="Architecture: FEniCSx + PyTorch" width="900" />
</p>

### Gauss-Newton Absolute Reconstruction

<p align="center">
  <img src="pictures/reconstruction_iterations.gif" alt="Gauss-Newton absolute reconstruction iterations" width="600" />
</p>

*Absolute-mode voltage RMSE reaches **8.23×10⁻⁸ V**, supporting numerical consistency of the underlying FEM implementation and EIDORS-style reconstruction workflow.*

Numerical performance metrics for absolute and difference reconstruction modes:

| Metric | Absolute mode | Difference mode |
|---|---:|---:|
| RMSE (V) | 8.23×10⁻⁸ | 7.68×10⁻⁵ |
| MAE (V) | 5.59×10⁻⁸ | 6.43×10⁻⁵ |
| Max. absolute error (V) | 2.60×10⁻⁷ | 1.47×10⁻⁴ |
| Pearson correlation | >0.9999 | 0.991 |
| Measurements | 208 | 208 |

### Simulation Parity with EIDORS

<p align="center">
  <img src="pictures/Fig.%203.%20Simulation%20parity_combined.png" alt="Simulation parity with EIDORS" width="900" />
</p>

Comparison of conductivity reconstructions and voltage predictions between PyEIDORS and MATLAB/EIDORS. (a) Ground-truth. (b) PyEIDORS single-step difference. (c) MATLAB/EIDORS raw. (d) EIDORS (aligned). (e) Differential voltage traces.

*Note: The larger residuals observed in EIDORS' voltage predictions compared to PyEIDORS do not imply superior reconstruction performance by PyEIDORS. This discrepancy arises primarily from numerical implementation differences in forward modeling. Since the synthetic measurement data in this experiment was generated using PyEIDORS' forward model, the PyEIDORS inverse solver benefits from inherent modeling consistency. Conversely, EIDORS incurs inevitable modeling bias when reconstructing from this data due to subtle differences in mesh discretization, finite element interpolation orders, and Complete Electrode Model (CEM) boundary handling.*

### Experimental Validation: Tank Data

<p align="center">
  <img src="pictures/Fig.%205.%20compare_tank.png" alt="Tank data validation" width="900" />
</p>

Verification using tank measurement data. PyEIDORS (b, c) demonstrates consistent performance with MATLAB/EIDORS benchmarks (d, e). Panels (f, g) show the original EIDORS reconstructions before affine alignment.

Parameter settings for the forward modeling and inverse solution corresponding to the tank comparison panels:

| Category | Parameter | b | c | d | e |
|---|---|---:|---:|---:|---:|
| Fwd. | Background conductivity (S/m) | 0.008 | 0.008 | 0.008 | 0.008 |
| Fwd. | Stimulation amplitude | 1.0 | 5e-5 | 1.0 | 5e-5 |
| Fwd. | Measurement gain | 10 | 10 | 10 | 10 |
| Fwd. | Contact impedance | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| Fwd. | Mesh radius | 0.025 | 0.025 | 0.025 | 0.025 |
| Fwd. | Number of electrodes | 16 | 16 | 16 | 16 |
| Fwd. | Number of nodes & elements | 1976 & 3806 | 1976 & 3806 | 1122 & 2130 | 1122 & 2130 |
| Inv. | Regularization lambda | 1.5 | 0.9 | 1.5 | 0.9 |
| Inv. | Regularization type | NOSER | NOSER | NOSER | NOSER |

### Experimental Validation: Bio-impedance (Corn Stalk)

<p align="center">
  <img src="pictures/Fig.%206.%20corn_stem_reconstruction.png" alt="Corn stalk reconstruction" width="900" />
</p>

Reconstruction of a corn stalk sample, capturing electrical heterogeneity in biological tissue.

---

## System Architecture

PyEIDORS is designed as a modular pipeline:

```
Mesh Loading/Generation ──► Forward Model (Complete Electrode Model)
                                  │
                                  ▼
                      Jacobian Computation & Regularization
                                  │
                                  ▼
                      Modular Gauss-Newton Reconstruction
                                  │
                                  ▼
            Visualization · Synthetic Data · Result Analysis
```

This architecture is intentionally modular so you can swap meshes, forward models, Jacobian calculators, or priors without rewriting the full pipeline.

Highlights:

- Designed for research and engineering practice, covering the complete pipeline of mesh generation, forward modeling, Jacobian computation, regularization, and Gauss-Newton reconstruction.
- Modular design with `EITSystem` as the core coordinator for geometry, forward, and inverse problem components.
- Supports Gmsh + DOLFINx native mesh workflow, with built-in stimulation/measurement pattern manager, synthetic data generation, and visualization tools.
- Provides examples, tests, and reports to help verify electrode layouts, mesh quality, and end-to-end reconstruction pipelines.

### Key Components

- **Geometry**: GMsh-based mesh generation (`mesh_generator.py`) and cached loading (`mesh_loader.py`).
- **Forward Model**: Complete Electrode Model (CEM) implemented in FEniCSx (`eit_forward_model.py`).
- **Inverse Solver**:
    - Adaptive Gauss-Newton (PyTorch-accelerated).
    - EIDORS-style single-step difference imaging.
    - Sparse Bayesian learning workflows.
- **Visualization**: `EITVisualizer` for meshes, conductivity maps, and measurement error plots.

For a file-level map of the codebase, see `FILE_ORGANIZATION.md`.

---

## Performance Benchmarks

End-to-end **single-step difference reconstruction** timing (Warm Start).
PyEIDORS' **measurement-space solve** is derived via Woodbury to avoid forming the large n&times;n system (n = mesh elements). The standard form solves
(J<sup>T</sup>J + &lambda;R) &delta; = J<sup>T</sup>d (parameter space), while the measurement-space form solves
(J R<sup>-1</sup> J<sup>T</sup> + &lambda;I) y = d and then &delta; = R<sup>-1</sup> J<sup>T</sup> y.
Here m = number of measurements, so the inner solve is m&times;m (typically 208), which is much smaller than n&times;n for dense meshes. With NOSER, R is diagonal, so R<sup>-1</sup> is just elementwise inversion, keeping the solution algebraically equivalent but far cheaper in memory and runtime.

<p align="center">
  <img src="pictures/benchmark_difference_runtime.png" alt="PyEIDORS baseline difference benchmark" width="900" />
</p>
<p align="center">
  <img src="pictures/benchmark_difference_runtime_measurement_6_24.png" alt="PyEIDORS measurement-space difference benchmark" width="900" />
</p>

For reference, we include the EIDORS timing curve (cold vs cached). EIDORS caching is extremely strong; the cold (no cache) curve is the fairer comparison point. PyEIDORS caching is currently limited and does not yet match EIDORS' cache behavior.

<p align="center">
  <img src="pictures/benchmark_difference_runtime_eidors.png" alt="EIDORS difference benchmark" width="900" />
</p>

Bench scripts:
- PyEIDORS: `python scripts/benchmarks/benchmark_difference_runtime.py`
- EIDORS: `compare_with_Eidors/benchmark_jacobian_runtime.m` (set `benchmark_mode = 'difference'`)

Accuracy check (parameter-space vs measurement-space, refinement=12):
- `delta_rel=8.66e-09`, `rmse_param=4.503e-01`, `rmse_meas=4.503e-01`, `pred_rel=5.10e-10`
- Reproduce with: `python scripts/benchmarks/benchmark_difference_runtime.py --refinements 12 --compare-solvers --single-step-space measurement`

CI perf gating compares:
- baseline profile: parameter-space / iterative options
- optimized profile: measurement-space / single-step options
- thresholds: median improvement `>=10%`, worst-case regression `<=5%`

| Elements | Baseline (s) | Measurement-Space (s) | Speedup |
|---:|---:|---:|---:|
| 2,650 | 1.873 | 1.336 | **1.40×** |
| 5,702 | 5.848 | 2.356 | **2.48×** |
| 10,584 | 20.722 | 4.003 | **5.18×** |
| 14,650 | 46.465 | 5.313 | **8.74×** |
| 18,474 | 85.128 | 7.082 | **12.02×** |

---

## Advanced Usage

### 1. Synthetic Simulation Comparison
Automatically generate simulation data and compute error statistics vs. EIDORS:

```bash
python scripts/run_synthetic_parity.py \
  --output-root results/simulation_parity/run01 \
  --mode both --save-forward-csv \
  --difference-solver single-step \
  --eidors-csv path/to/eidors_voltages.csv
```

The script will:

- Run forward simulation with given circular phantom parameters, saving baseline/anomaly boundary voltages.
- Execute absolute and difference imaging reconstruction sequentially, outputting residuals, RMSE, correlation coefficients, and other metrics (`metrics.json`).
- Optionally load EIDORS-generated voltage vectors via `--eidors-csv` to automatically compute differences with PyEIDORS.

### 2. Real Measurement Data
After data normalization (see `docs/MEASUREMENT_DATA_SPEC.md`), run reconstruction:

```bash
python scripts/run_real_measurement_reconstruction.py \
  --csv data/measurements/sample.csv \
  --metadata data/measurements/sample.yaml \
  --use-cols 0 2
```

The script validates the measurement matrix, builds `EITSystem`, and performs difference inverse problem reconstruction. Output measurement curves and conductivity images are saved in `results/real_measurements/`.

### 3. Sparse Bayesian Learning
Run the advanced sparse Bayesian solver (supports GPU):

```bash
python scripts/run_sparse_bayesian_reconstruction.py \
  --csv data/measurements/sample.csv \
  --mode both --solver fista --use-gpu
```

The repository also includes a pre-generated tank sparse Bayesian demo under:
`results/tank_final_results/sparse_bayesian_physical_bg0008_v1_0/` (see `COMMAND.md` inside).

Results are written to `results/sparse_bayesian/` by default. For a full list of options, run `python scripts/run_sparse_bayesian_reconstruction.py --help`.

## Data, Visualization, and Testing

- Synthetic data: `create_synthetic_data` supports setting noise level, anomaly position and conductivity, returning clean/noisy data with SNR metrics.
- Real measurement data: `MeasurementDataset` helper class builds `EITData` from normalized measurement matrices and metadata, see `docs/MEASUREMENT_DATA_SPEC.md`.
- Visualization: `EITVisualizer` includes built-in plotting for mesh, conductivity, measurements, reconstruction comparison, and convergence curves, with PNG report output.
- Testing: `tests/unit/test_complete_eit_system.py` provides end-to-end pipeline validation, `tests/unit/test_optimized_mesh_generator.py` covers geometry and electrode layout.
- Examples: `examples/basic_usage.py` demonstrates module structure, environment checking, and system initialization steps.

---

## Documentation

- **File Structure**: `FILE_ORGANIZATION.md`
- **Nix + uv (FEniCSx) Setup**: `docs/NIX_FENICSX.md`
- **Phase-2 Migration Guide**: `docs/MIGRATION_PHASE2.md`
- **Data Specs**: `docs/MEASUREMENT_DATA_SPEC.md`
- **Electrode Setup**: `docs/ELECTRODE_Y_AXIS_POSITIONING.md`
- **Docker Notes (Legacy archive)**: `docs/archive/DOCKER_LEGACY.md`

## Environment Note

The primary maintained developer workflow is **Nix + uv** with FEniCSx (DOLFINx), documented in `docs/NIX_FENICSX.md`.

Docker content is archived for historical reproducibility in `docs/archive/DOCKER_LEGACY.md`.
