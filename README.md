# PyEidors

A FEniCS-based Electrical Impedance Tomography (EIT) forward and inverse problem solving system, providing a Pythonic implementation similar to EIDORS with PyTorch acceleration.

## Project Overview

- Designed for research and engineering practice, covering the complete pipeline of mesh generation, forward modeling, Jacobian computation, regularization, and Gauss-Newton reconstruction.
- Modular design with `EITSystem` as the core coordinator for geometry, forward, and inverse problem components, making it easy to replace or extend any part.
- Supports GMsh+meshio+FEniCS mesh workflow, with built-in stimulation/measurement pattern manager, synthetic data generation, and visualization tools.
- Provides examples, tests, and reports to help verify electrode layouts, mesh quality, and end-to-end reconstruction pipelines.

## Documentation

- Repository layout and code map: `FILE_ORGANIZATION.md`
- Measurement data format: `docs/MEASUREMENT_DATA_SPEC.md`
- Electrode positioning convention: `docs/ELECTRODE_Y_AXIS_POSITIONING.md`
- Docker usage and publishing notes: `docs/DOCKER.md`

## System Architecture Overview

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

- Geometry module prepares FEniCS-compatible meshes (can load existing H5/XDMF or generate in real-time using GMsh).
- `EITForwardModel` builds finite element discretization, applies stimulation/measurement patterns, and outputs electrode voltages and measurements.
- `DirectJacobianCalculator` and regularization modules provide flexible sensitivity matrices and penalty terms.
- `ModularGaussNewtonReconstructor` implements GPU/CPU adaptive Gauss-Newton iteration using PyTorch for inverse problem solving.

## Key Components

- `EITSystem`: end-to-end coordinator for geometry, forward model, and inverse solvers.
- Forward: Complete Electrode Model implemented with FEniCS.
- Inverse: Gauss-Newton reconstruction and EIDORS-style single-step difference imaging; optional sparse Bayesian workflows.
- Geometry: GMsh-based mesh generation plus cached mesh loading.
- Visualization: utilities for plotting meshes, voltages, and reconstructions.

For a file-level map of the codebase, see `FILE_ORGANIZATION.md`.

## Quickstart

1. Start the Docker container (see below).
2. Install the package inside the container:
   - `pip install -e .`
3. Run a synthetic demo (paper parity example):
   - `python scripts/run_synthetic_parity.py --output-root results/simulation_parity/run03_single_step --mode both --save-forward-csv --difference-solver single-step`
4. Run a real-data demo (tank difference imaging):
   - `python scripts/run_single_step_diff_realdata.py --csv data/measurements/tank/2025-11-14-22-18-02_1_10.00_50uA_3000Hz.csv --background-sigma 0.008 --lambda 0.9 --output results/tank_final_results/difference_imaging`

The repository includes pre-generated demo outputs under `results/` (see the `COMMAND.md` files inside each demo directory).

### Synthetic Simulation Comparison

When comparing PyEidors with existing MATLAB/EIDORS workflows, use the new script to automatically generate simulation data and compute error statistics:

```bash
python scripts/run_synthetic_parity.py \
  --output-root results/simulation_parity/run01 \
  --mode both --save-forward-csv
```

The script will:

- Run forward simulation with given circular phantom parameters, saving baseline/anomaly boundary voltages;
- Execute absolute and difference imaging reconstruction sequentially, outputting residuals, RMSE, correlation coefficients, and other metrics (`metrics.json`);
- Optionally load EIDORS-generated voltage vectors via `--eidors-csv path/to/voltages.csv` to automatically compute differences with PyEidors.

The output JSON, CSV, and generated reconstruction images (`results/simulation_parity/run01/...`) can be directly used for simulation experiment placeholder figures in papers.

### Real Measurement Data Reconstruction Example

After data normalization (see `docs/MEASUREMENT_DATA_SPEC.md`), use the script for a quick difference reconstruction:

```bash
python scripts/run_real_measurement_reconstruction.py \
  --csv data/measurements/2025-06-29-20-00-52_1_10.00_20uA_1000Hz.csv \
  --metadata data/measurements/2025-06-29-20-00-52_1_10.00_20uA_1000Hz.yaml \
  --use-cols 0 2
```

The script validates the measurement matrix, builds `EITSystem`, and performs difference inverse problem reconstruction. Output measurement curves and conductivity images are saved in `results/real_measurements/`.

For sparse Bayesian reconstruction on real measurement data, use the new `scripts/run_sparse_bayesian_reconstruction.py`, supporting absolute/difference imaging and automatic comparison with Gauss-Newton results:

```bash
python scripts/run_sparse_bayesian_reconstruction.py \
  --csv data/measurements/EIT_DEV_Test/...csv \
  --mode both --absolute-col 2 --reference-col 0 --target-col 2 \
  --subspace-rank 64 --linear-warm-start --coarse-group-size 40 \
  --coarse-levels 96 48 --coarse-iterations 1 --block-iterations 2 \
  --block-size 64 --solver fista --use-gpu --gpu-dtype float32 \
  --contact-impedance 1e-5 --difference-calibration after
```

Results are written to `results/sparse_bayesian/` by default. For a full list of options, run `python scripts/run_sparse_bayesian_reconstruction.py --help`.

## Data, Visualization, and Testing

- Synthetic data: `create_synthetic_data` supports setting noise level, anomaly position and conductivity, returning clean/noisy data with SNR metrics.
- Real measurement data: `MeasurementDataset` helper class builds `EITData` from normalized measurement matrices and metadata, see `docs/MEASUREMENT_DATA_SPEC.md`.
- Visualization: `EITVisualizer` includes built-in plotting for mesh, conductivity, measurements, reconstruction comparison, and convergence curves, with PNG report output.
- Testing: `tests/unit/test_complete_eit_system.py` provides end-to-end pipeline validation, `tests/unit/test_optimized_mesh_generator.py` covers geometry and electrode layout.
- Examples: `examples/basic_usage.py` demonstrates module structure, environment checking, and system initialization steps.

## Environment Setup

This project is developed in a Docker environment using the following core components:

- **FEniCS**: ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
- **CUQIpy**: 1.3.0
- **CUQIpy-FEniCS**: 0.8.0
- **PyTorch**: 2.7.1+cu128 (GPU support)
- **Python**: 3.10+ (provided via Docker)

The following installation is based on the official FEniCS image. You can also use the `Dockerfile` in the repository to build a full environment or directly pull a pre-built image.

## Docker Environment Setup

### Option A (recommended): use the prebuilt image

The recommended workflow is to use a prebuilt Docker image that already contains FEniCS, CUQIpy, and PyTorch.
See `docs/DOCKER.md` for the most up-to-date commands.

```bash
docker pull ghcr.io/cbz199671/pyeidors-env:latest

docker run -ti \
  --gpus all \
  --shm-size=24g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network=host \
  --cpus=20 \
  --memory=28g \
  -v "$(pwd):/root/shared" \
  -w /root/shared \
  --name pyeidors \
  ghcr.io/cbz199671/pyeidors-env:latest
```

### Option B: start from the official FEniCS image (manual installation)

```bash
# Start container
docker run -ti \
  --gpus all \
  --shm-size=24g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network=host \
  --cpus=20 \
  --memory=28g \
  -v "D:\workspace\PyEIDORS:/root/shared" \
  -w /root/shared \
  --name pyeidors \
  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30

# (Optional) Install CJK font dependencies
apt-get update && apt-get install -y fonts-wqy-zenhei

# Install CUQIpy and CUQIpy-FEniCS
pip install cuqipy cuqipy-fenics

# Create virtual environment
python3 -m venv /opt/final_venv --system-site-packages
source /opt/final_venv/bin/activate

# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
