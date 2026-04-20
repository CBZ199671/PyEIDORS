---
status: draft
source: from-code
domain: environment-cli
---

# Cavekit: Environment and CLI

## Scope

This kit covers reproducible runtime environments, Nix/uv integration, CUDA and
CPU profile split, diagnostics, launch scripts, and command-line workflows.

## Requirements

### R1: Nix and uv define the supported Python runtime

**Description:** Development commands run inside concrete project shells that
provide FEniCSx, PETSc, MPI, and locked Python package sets.

**Acceptance Criteria:**
- [ ] CPU workflow enters `nix develop` and verifies environment manifest.
- [ ] CUDA workflow enters `nix develop .#cuda` and uses `.venv-cuda`.
- [ ] `uv.lock` remains the Python dependency source of truth.
- [ ] Global `pip install` is not required for supported workflows.

**Dependencies:** None

### R2: GUI launchers preserve runtime and source paths

**Description:** GUI launch scripts enter the correct Nix profile, synchronize
the locked Python environment, prepend repository/source paths, and run a
preflight before launching.

**Acceptance Criteria:**
- [ ] `run_eit_app.sh --cpu` uses the CPU dev shell.
- [ ] `run_eit_app.sh --gpu` uses the CUDA dev shell and PETSc CUDA probe unless
  explicitly skipped.
- [ ] Linked worktree source can be used while the Nix flake comes from the main
  checkout.
- [ ] Missing required modules fail before the Qt event loop starts.

**Dependencies:** `cavekit-workstation-gui.md`

### R3: Diagnostics distinguish availability from actual backend creation

**Description:** Probes and manifests validate real runtime capability, not just
symbol presence.

**Acceptance Criteria:**
- [ ] PETSc CUDA probe attempts to create CUDA sparse matrix, CUDA vector, and
  dense CUDA matrix.
- [ ] Environment manifests can be exported and verified per profile.
- [ ] Diagnostic failures include enough detail to decide whether to use CPU,
  auto, or CUDA policy.

**Dependencies:** `cavekit-forward-solver.md`

### R4: CLI workflows expose reproducible reconstruction paths

**Description:** Scripts provide stable entrypoints for synthetic parity,
unified reconstruction, benchmarks, and reports.

**Acceptance Criteria:**
- [ ] Unified reconstruction CLI validates input mode, method, mesh dimension,
  and device policy before running.
- [ ] Benchmark scripts can write reports consumed by comparison/guard scripts.
- [ ] Demo and diagnostic scripts can run from the project root with documented
  output locations.

**Dependencies:** `cavekit-core-system.md`, `cavekit-cache-performance.md`

## Brownfield Evidence

- Source: `flake.nix`
- Source: `pyproject.toml`
- Source: `uv.lock`
- Source: `scripts/env/`
- Source: `scripts/gui/run_eit_app.sh`
- Source: `scripts/run_reconstruction_unified.py`
- Docs: `docs/NIX_FENICSX.md`
- Docs: `docs/WSL2_CUDA.md`
- Tests: `tests/unit/test_env_sync_script.py`
- Tests: `tests/unit/test_env_manifest_verify.py`
- Tests: `tests/integration/test_env_repro_mac_linux_contract.py`

## Out of Scope

- System-wide Nix installation.
- Windows host GPU driver installation.

## Cross-References

- Depended on by: `cavekit-workstation-gui.md`
- Depended on by: `cavekit-forward-solver.md`
- Depended on by: `cavekit-cache-performance.md`

