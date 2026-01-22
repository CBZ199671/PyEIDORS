# PyEIDORS

<p align="center">
  <img src="pictures/Fig.%204.%20fig_absolute_vs_difference.png" alt="PyEIDORS banner" width="900" />
</p>

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
![License](https://img.shields.io/badge/license-GPL--3.0--only-green)
![Backend](https://img.shields.io/badge/backend-FEniCS-orange)
![Accel](https://img.shields.io/badge/accel-PyTorch-red)

A Python-first EIT framework with a FEniCS Complete Electrode Model (CEM) backend and PyTorch acceleration, designed to be familiar to EIDORS users while enabling modern GPU/differentiable workflows.

SoftwareX manuscript in preparation; citation info will be added after acceptance.

## Why PyEIDORS

- **Numerical Consistency**: Matches EIDORS-style workflows with verified simulation parity.
- **Modern Architecture**: Hybrid FEniCS (FEM) + PyTorch (Inverse/Accel) design.
- **Modular & Extensible**: `EITSystem` coordinator makes it easy to replace geometry, forward models, or solvers.
- **Research Ready**: End-to-end scripts for absolute & difference reconstruction, real-time mesh generation, and benchmarking.

---

## Quick Start

1. **Start the Docker Environment** (see [Docker Setup](#docker-environment-setup) for details):
   ```bash
   docker run -ti -v "$(pwd):/root/shared" -w /root/shared --name pyeidors ghcr.io/cbz199671/pyeidors-env:latest
   ```

2. **Install the Package**:
   ```bash
   pip install -e .
   ```

3. **Run a Synthetic Demo** (Paper parity example):
   ```bash
   python scripts/run_synthetic_parity.py --output-root results/simulation_parity/run03 --mode both --difference-solver single-step --gn-regularization 1e-11
   ```

4. **Run a Real-Data Demo** (Tank difference imaging):
   ```bash
   python scripts/run_single_step_diff_realdata.py --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv --background-sigma 0.008 --lambda 0.9
   ```

---

## Gallery & Validation

### Modern Architecture: FEniCS + PyTorch

<p align="center">
  <img src="pictures/Fig.%201.%20pyeidors_architecture.png" alt="Architecture: FEniCS + PyTorch" width="900" />
</p>

### Gauss-Newton Absolute Reconstruction

<p align="center">
  <img src="pictures/reconstruction_iterations.gif" alt="Gauss-Newton absolute reconstruction iterations" width="600" />
</p>

*Absolute-mode voltage RMSE reaches **8.23×10⁻⁸ V**, supporting numerical consistency of the underlying FEM implementation.*

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

Verification using tank measurement data. PyEIDORS (b, c) demonstrates consistent performance with MATLAB/EIDORS benchmarks (d, e).

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

### Key Components

- **Geometry**: GMsh-based mesh generation (`mesh_generator.py`) and cached loading (`mesh_loader.py`).
- **Forward Model**: Complete Electrode Model (CEM) implemented in FEniCS (`eit_forward_model.py`).
- **Inverse Solver**:
    - Adaptive Gauss-Newton (PyTorch-accelerated).
    - EIDORS-style single-step difference imaging.
    - Sparse Bayesian learning workflows.
- **Visualization**: `EITVisualizer` for meshes, conductivity maps, and measurement error plots.

---

## Performance Benchmarks

End-to-end **single-step difference reconstruction** timing (Warm Start).
PyEIDORS' **measurement-space solve** ($J R^{-1} J^T$) significantly outperforms the standard parameter-space solve ($J^T J$) on dense meshes.

<p align="center">
  <img src="pictures/benchmark_difference_runtime_measurement_6_24.png" alt="PyEIDORS measurement-space difference benchmark" width="900" />
</p>

| Elements | Baseline (s) | Measurement-Space (s) | Speedup |
|---:|---:|---:|---:|
| 5,702 | 5.848 | 2.356 | **2.48×** |
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

### 2. Real Measurement Data
After data normalization (see `docs/MEASUREMENT_DATA_SPEC.md`), run reconstruction:

```bash
python scripts/run_real_measurement_reconstruction.py \
  --csv data/measurements/sample.csv \
  --metadata data/measurements/sample.yaml \
  --use-cols 0 2
```

### 3. Sparse Bayesian Learning
Run the advanced sparse Bayesian solver (supports GPU):

```bash
python scripts/run_sparse_bayesian_reconstruction.py \
  --csv data/measurements/sample.csv \
  --mode both --solver fista --use-gpu
```

---

## Documentation

- **File Structure**: `FILE_ORGANIZATION.md`
- **Data Specs**: `docs/MEASUREMENT_DATA_SPEC.md`
- **Electrode Setup**: `docs/ELECTRODE_Y_AXIS_POSITIONING.md`
- **Docker Notes**: `docs/DOCKER.md`

## Docker Environment Setup

### Option A: Prebuilt Image (Recommended)

```bash
```docker pull ghcr.io/cbz199671/pyeidors-env:latest

docker run -ti \
  -v "$(pwd):/root/shared" \
  -w /root/shared \
  --name pyeidors \
  ghcr.io/cbz199671/pyeidors-env:latest
```
*Note: Add `--gpus all` for GPU support.*

### Option B: Manual Build
See `docs/DOCKER.md` or use the provided `Dockerfile`.

## Environment Components
- **FEniCS**: 2024-05-30 (Official Image)
- **CUQIpy**: 1.3.0
- **PyTorch**: 2.7.1+cu128