# PyEidors Project Layout

This document is intended for code readers and contributors. For end-user setup and quick demos, start with `README.md`.

## Directory Overview

### Core source code
```
src/
└── pyeidors/                   # Main package
    ├── core_system.py          # High-level EITSystem entry point
    ├── data/                   # Data structures + synthetic/real data helpers
    ├── electrodes/             # Electrodes and stimulation/measurement patterns
    ├── forward/                # Forward solver (Complete Electrode Model)
    ├── inverse/                # Inverse solvers / workflows
    ├── geometry/               # Mesh generation + loading
    ├── visualization/          # Plotting and reporting utilities
    └── utils/                  # Misc utilities
```

### Tests
```
tests/
├── unit/                       # Unit tests
└── run_all_tests.py            # Test runner
```

### Examples and demos
```
examples/                       # Example scripts / notebooks
demos/                          # Demo scripts and generated figures
scripts/                        # CLI entry scripts and utilities
```

### Data, meshes, and results (mostly generated)
```
data/                           # Input datasets (measurements)
eit_meshes/                     # Cached meshes (generated)
results/                        # Generated outputs (figures, npz, csv, metrics)
```

These folders are large and/or private by default, so the repository uses a whitelist
in `.gitignore` to keep only the small demo artifacts needed for the paper results.

## Code Map (file-level)

This section lists the main implementation files and where to look when modifying the project.

- `src/pyeidors/core_system.py`: `EITSystem` orchestration (`setup`, `forward_solve`, `inverse_solve`).
- `src/pyeidors/forward/eit_forward_model.py`: Complete Electrode Model forward solver.
- `src/pyeidors/electrodes/patterns.py`: stimulation/measurement pattern generation and filtering.
- `src/pyeidors/inverse/jacobian/direct_jacobian.py`: direct/adjoint Jacobian computation.
- `src/pyeidors/inverse/jacobian/adjoint_jacobian.py`: EIDORS-style adjoint Jacobian utilities.
- `src/pyeidors/inverse/regularization/smoothness.py`: regularization operators (including NOSER-style options).
- `src/pyeidors/inverse/solvers/gauss_newton.py`: modular Gauss-Newton solver.
- `src/pyeidors/inverse/solvers/sparse_bayesian.py`: sparse Bayesian solver implementation (CUQIpy-based).
- `src/pyeidors/inverse/workflows/`: higher-level workflows (absolute, difference, sparse Bayesian).
- `src/pyeidors/data/structures.py`: shared data structures (`PatternConfig`, `EITData`, `EITImage`, ...).
- `src/pyeidors/data/measurement_dataset.py`: loading/validation helpers for real measurement datasets.
- `src/pyeidors/data/synthetic_data.py`: synthetic data generation and phantoms.
- `src/pyeidors/geometry/optimized_mesh_generator.py`: GMsh mesh generation + electrode placement convention.
- `src/pyeidors/geometry/mesh_loader.py`: loading cached meshes and conversion helpers.
- `src/pyeidors/visualization/eit_plots.py`: plotting utilities used by demos and scripts.

## Recent change highlight

### Electrode position convention (Y-axis start)
- Code: `src/pyeidors/geometry/optimized_mesh_generator.py`
- Tests: `tests/unit/test_electrode_position_y_axis.py`
- Demo: `demos/demo_y_axis_electrodes.py`

## Contributor workflow

### Run tests
```bash
python tests/unit/test_electrode_position_y_axis.py
python tests/run_all_tests.py
```

### Add or update paper/demo artifacts

This repository keeps most generated outputs out of Git, and only whitelists the paper/demo artifacts in `.gitignore`.
If you add a new demo, update the allowlist and include a `COMMAND.md` file in the output directory documenting how it was generated.

---

Updated: 2025-07-04
