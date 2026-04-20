# Architecture Overview

## System Description

PyEIDORS is a Python EIT framework and desktop workstation. The core package
implements mesh generation/loading, stimulation and measurement patterns,
FEniCSx/DOLFINx Complete Electrode Model forward solves, inverse reconstruction,
semantic caches, real measurement data ingestion, visualization, EIDORS
interoperability, and performance policy. The desktop application wraps the core
package with PySide6 workflows for hardware acquisition, simulation, database
browsing, dataset generation, and reconstruction.

## Technology Stack

- Language: Python 3.13.
- Core numerical runtime: FEniCSx/DOLFINx, PETSc, MPI, NumPy, SciPy.
- Acceleration: PyTorch, optional PETSc CUDA shell, optional structured CUDA
  backend.
- GUI: PySide6, pyqtgraph, matplotlib, PyVista/VTK.
- Packaging and environment: Nix dev shells plus uv-locked virtual
  environments.
- Tests: pytest with unit, integration, GUI, hardware, gpu, and fenicsx markers.

## Build and Test Commands

- CPU shell: `nix develop`
- CUDA shell: `nix develop .#cuda`
- GUI CPU: `bash scripts/gui/run_eit_app.sh --cpu`
- GUI GPU: `bash scripts/gui/run_eit_app.sh --gpu`
- PETSc CUDA probe: `python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty`
- Test entrypoint: `pytest`
- Unified reconstruction CLI smoke: `python scripts/run_reconstruction_unified.py --help`

## Directory Structure

- `src/pyeidors/`: reusable EIT framework.
- `src/eit_app/`: EIT Workstation desktop application.
- `scripts/`: diagnostics, benchmarks, environment tools, interop tools, GUI launchers.
- `tests/unit/`: focused unit and widget smoke tests.
- `tests/integration/`: end-to-end solver, CUDA, cache, and CLI checks.
- `docs/`: architecture, environment, cache, measurement data, and interop notes.
- `data/`, `eit_meshes/`, `results/`: datasets, generated meshes, and outputs.

## Key Domains

- Core orchestration and public API.
- Data, measurement frames, and physics units.
- Geometry, electrodes, and stimulation/measurement patterns.
- Forward CEM solver and PETSc/CUDA backend policy.
- Inverse reconstruction workflows.
- Cache and performance policy.
- EIDORS/PyEIDORS interoperability.
- EIT Workstation GUI, hardware acquisition, and recording.
- Environment, CLI, diagnostics, and reproducibility.

## External Dependencies

- Nix-provided FEniCSx/DOLFINx, PETSc, MPI, and optional CUDA PETSc runtime.
- Python packages from `pyproject.toml` extras: PySide6, pyqtgraph, pyvista,
  pyvistaqt, pyserial, torch, CUQIpy, gmsh, meshio, pandas, h5py, pyyaml.
- Optional MATLAB/EIDORS on the Windows host for bridge validation.
- Optional physical EIT hardware via serial, relay, or Windows-hosted serial
  bridge.

## Known Issues / Tech Debt

- WSLg Qt defaults favor XCB for stability, but XWayland can look blurry on
  high-DPI Windows displays.
- GUI launchers route Nix through the main worktree while source imports may
  come from a linked worktree; dependency changes must be synchronized between
  worktrees.
- 3D GPU, fused, and reduced-order paths have explicit policy switches and are
  guarded by diagnostics and benchmark tests.
- Hardware-backed workflows require simulator fallbacks and preflight checks so
  tests can run without physical devices.

