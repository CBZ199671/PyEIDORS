---
status: draft
source: from-code
domain: inverse-reconstruction
---

# Cavekit: Inverse Reconstruction

## Scope

This kit covers absolute Gauss-Newton, difference reconstruction, sparse
Bayesian workflows, Jacobian computation, regularization, reduced-order helpers,
and reconstruction diagnostics.

## Requirements

### R1: Difference reconstruction supports strict and fast paths

**Description:** Difference imaging can run exact reference paths and optimized
paths while preserving documented equivalence and diagnostics.

**Acceptance Criteria:**
- [ ] Strict 3D difference reconstruction uses an exact backend or records the
  exact fallback reason.
- [ ] Fast single-step difference reconstruction reuses compatible operator
  caches.
- [ ] Voltage orientation and difference mode are normalized before solving.

**Dependencies:** `cavekit-forward-solver.md`, `cavekit-cache-performance.md`

### R2: Absolute reconstruction exposes controlled GN behavior

**Description:** Absolute imaging runs Gauss-Newton iterations with validated
regularization, line search, device policy, and diagnostics.

**Acceptance Criteria:**
- [ ] Invalid regularization or solver options fail deterministically.
- [ ] GN iteration outputs include convergence-relevant diagnostics.
- [ ] Device selection follows `cpu`, `auto`, and `cuda` semantics.

**Dependencies:** `cavekit-forward-solver.md`, `cavekit-environment-cli.md`

### R3: Jacobian and regularization are modular and testable

**Description:** Direct/adjoint Jacobian paths and regularization operators can
be validated independently of end-to-end scripts.

**Acceptance Criteria:**
- [ ] Jacobian shape and cache-key stability tests pass.
- [ ] NOSER, smoothness, Tikhonov, and total-variation regularization operators
  produce finite compatible systems.
- [ ] Block tuning and vectorized assembly preserve finite results.

**Dependencies:** `cavekit-geometry-electrodes.md`

### R4: Sparse and reduced workflows stay opt-in and diagnosable

**Description:** Sparse Bayesian and reduced-order helpers are available without
polluting default strict behavior.

**Acceptance Criteria:**
- [ ] Sparse Bayesian workflows validate inputs and expose backend diagnostics.
- [ ] Reduced-order GN helpers pass equivalence tests where such tests exist.
- [ ] Experimental accelerators are gated by explicit policy switches.

**Dependencies:** `cavekit-cache-performance.md`

## Brownfield Evidence

- Source: `src/pyeidors/inverse/solvers/gauss_newton.py`
- Source: `src/pyeidors/inverse/solvers/gauss_newton_engine.py`
- Source: `src/pyeidors/inverse/workflows/difference.py`
- Source: `src/pyeidors/inverse/workflows/absolute.py`
- Source: `src/pyeidors/inverse/solvers/sparse_bayesian.py`
- Source: `src/pyeidors/inverse/reduced/`
- Tests: `tests/unit/test_difference_semantics.py`
- Tests: `tests/unit/test_gauss_newton_solver_extended.py`
- Tests: `tests/unit/test_sparse_bayesian_solver_extended.py`
- Tests: `tests/unit/test_reduced_gn_step_equivalence.py`
- Tests: `tests/integration/test_3d_diff_fast_vs_strict.py`

## Out of Scope

- GUI controls for reconstruction; see Workstation GUI.
- Raw hardware acquisition; see Workstation GUI.

## Cross-References

- Depends on: `cavekit-forward-solver.md`
- Depends on: `cavekit-data-and-units.md`
- Related: `cavekit-cache-performance.md`

