---
status: draft
source: from-code
complexity: thorough
complexity_score: 16
complexity_axes:
  files: 4
  type: 4
  judgment: 3
  cross_component: 3
  novelty: 2
---

# Cavekit Overview

## Domains

| Domain | File | Summary | Status |
| --- | --- | --- | --- |
| Core System | `cavekit-core-system.md` | Public orchestration API for setup, forward solve, inverse solve, and cache controls. | DRAFT |
| Data and Units | `cavekit-data-and-units.md` | Measurement data contracts, frame I/O, drive semantics, and unit checks. | DRAFT |
| Geometry and Electrodes | `cavekit-geometry-electrodes.md` | Mesh generation/loading, electrode layout, and pattern generation. | DRAFT |
| Forward Solver | `cavekit-forward-solver.md` | CEM forward model behavior, PETSc/SciPy backends, CUDA policy, and diagnostics. | DRAFT |
| Inverse Reconstruction | `cavekit-inverse-reconstruction.md` | Absolute, difference, sparse Bayesian, and reduced reconstruction workflows. | DRAFT |
| FEniCSx/PETSc EIT Refactor | `cavekit-fenicsx-petsc-eit-refactor.md` | Official-aligned solver refactor: 3D AMG forward, multi-RHS reuse, matrix-free inverse, strict validation. | DRAFT |
| Cache and Performance | `cavekit-cache-performance.md` | Semantic cache layers, invalidation, performance policy, and benchmark gates. | DRAFT |
| Interop | `cavekit-interop.md` | EIDORS/PyEIDORS geometry exchange and validation bridge. | DRAFT |
| Workstation GUI | `cavekit-workstation-gui.md` | PySide6 application, hardware acquisition, simulation, database, and visualization workflows. | DRAFT |
| Environment and CLI | `cavekit-environment-cli.md` | Nix/uv environments, launch scripts, diagnostics, and reproducible CLI paths. | DRAFT |

## Cross-Cutting Concerns

- Numerical reproducibility spans Core System, Forward Solver, Inverse
  Reconstruction, Cache and Performance, and Environment and CLI.
- Real measurement traceability spans Data and Units, Workstation GUI, Cache and
  Performance, and Interop.
- CUDA behavior spans Forward Solver, Inverse Reconstruction, Cache and
  Performance, and Environment and CLI.
- FEniCSx/PETSc EIT refactor spans Forward Solver, Inverse Reconstruction,
  Cache and Performance, and Environment and CLI.
- WSL2/Windows behavior spans Workstation GUI, Interop, hardware transports, and
  Environment and CLI.

## Validation Entry

See `validation-report.md` for current brownfield coverage notes.
