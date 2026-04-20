---
status: draft
source: map
---

# Plan Overview

## Build Sites

| Build Site | File | Purpose | Status |
| --- | --- | --- | --- |
| Brownfield Cavekit Stabilization | `build-site-brownfield-cavekit.md` | Validate generated kits against the existing PyEIDORS codebase, add source-to-kit traceability, and close actionable coverage gaps. | READY |
| FEniCSx/PETSc EIT Solver Refactor | `build-site-fenicsx-petsc-eit-refactor.md` | Refactor forward/inverse solver internals toward official PETSc KSP/PC, multi-RHS reuse, and matrix-free 3D inverse workflows. | READY |

## Implementation Detail Appendices

| Appendix | File | Purpose |
| --- | --- | --- |
| 2D/3D EIT Solver Details | `fenicsx-petsc-eit-2d-3d-implementation-details.md` | Concrete forward/inverse solver and preconditioner choices for 2D/3D EIT, including GPU/MPI and contact impedance block policy. |

## Dependency Tiers

| Tier | Tasks | Purpose |
| --- | --- | --- |
| 0 | `T-CK-001` | Establish cited test inventory. |
| 1 | `T-CK-002`, `T-CORE-001`, `T-DATA-001`, `T-GEOM-001`, `T-ENV-001` | Add traceability and validate independent foundations. |
| 2 | `T-FWD-001` | Validate forward solver after data, geometry, and environment foundations. |
| 3 | `T-INV-001`, `T-INTEROP-001` | Validate inverse and bridge behavior after forward evidence. |
| 4 | `T-CACHE-001`, `T-GUI-001` | Validate cache/performance and GUI workflows after solver evidence. |
| 5 | `T-CK-003`, `T-CK-004` | Convert validation results into stable kit updates and implementation tracking. |

## Refactor Build Site Dependency Tiers

| Tier | Tasks | Purpose |
| --- | --- | --- |
| 0 | `T-FPX-001`, `T-FPX-002`, `T-FPX-005`, `T-FPX-012` | Preserve first-pass solver preset and matrix-free operator work. |
| 1 | `T-FPX-003`, `T-FPX-004` | Harden forward KSP/PC reuse and benchmark diagnostics. |
| 2 | `T-FPX-006`, `T-FPX-007` | Move inverse fast linear solves onto matrix-free operator actions. |
| 3 | `T-FPX-008`, `T-FPX-009` | Add compatible inverse preconditioning and block-ready contact impedance design. |
| 4 | `T-FPX-010`, `T-FPX-011` | Strengthen GPU/MPI diagnostics and sharded validation discipline. |

## Global Validation Gates

- Gate 1: Python import/collection succeeds for targeted tests.
- Gate 2: Targeted unit tests pass for each domain touched.
- Gate 3: Integration smoke tests pass for cross-domain behavior when feasible.
- Gate 4: Benchmark/performance gates run only for cache, CUDA, and solver tasks.
- Gate 5: GUI startup smoke runs only when GUI/runtime task changes behavior.
- Gate 6: Human review confirms the kits describe intended behavior, not merely
  accidental current behavior.

## File Ownership

| File/Pattern | Owner |
| --- | --- |
| `context/kits/**` | Brownfield Cavekit Stabilization |
| `context/plans/**` | Brownfield Cavekit Stabilization |
| `context/impl/**` | Brownfield Cavekit Stabilization |
| `src/pyeidors/**/CLAUDE.md` | Domain task owning that source subtree |
| `src/eit_app/**/CLAUDE.md` | Workstation GUI task |
| `tests/**` | Domain task owning the related behavior |

## Next Prompt

Run implementation through `context/prompts/002-implement-brownfield-cavekit-plan.md`
once this plan is reviewed.
