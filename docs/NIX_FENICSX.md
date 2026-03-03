# PyEIDORS: Nix + uv FEniCSx environment

This document defines the maintained environment setup for running PyEIDORS with FEniCSx on macOS (including Apple Silicon) and Linux, without Docker and without Conda.

## Strategy

- Nix provides the system and scientific stack (DOLFINx, Basix, UFL, FFCx, MPI).
- uv manages the project virtual environment and editable install.
- We avoid pip-based DOLFINx installation.

## Facts checked on 2026-03-02

- `fenics-dolfinx` is not available on PyPI (`https://pypi.org/project/fenics-dolfinx/` is 404).
- `fenics-dolfinx` exists in `nixpkgs-unstable`.
- Therefore this repository uses Nix to provide DOLFINx.

## Darwin stability pin (checked on 2026-03-03)

To avoid a known upstream regression on macOS (Apple Silicon), this repository
pins `nixpkgs` to:

- `b665e98e4cb439b95f8faee197c76c6578197e55` (2025-11-08)

Why this pin exists:

- On newer revisions we reproduced an upstream crash in
  `test_symmetry_interior_facet_assembly[mesh1]` with PETSc TRAP/MPI_ABORT
  (`errorcode: 59`) on Darwin.
- On the pinned revision, the same upstream test passes.

Reference logs (local reproducibility artifacts):

- Probe summary:
  `.codex_logs/nixpkgs_probe/20260303-143011/summary.tsv`
- Lock update evidence:
  `.codex_logs/nixpkgs_probe/20260303-143011/lock-update.log`
- Passing upstream target after pin:
  `.codex_logs/upstream_mesh1_after_lock.log`

Before changing `flake.lock`, run a probe first and only upgrade if the
critical upstream test still passes on Darwin:

```bash
python scripts/diagnostics/probe_nixpkgs_dolfinx.py \
  --window 2 \
  --max-candidates 5 \
  --stop-on-pass \
  --update-lock
```

For targeted probing of known FEniCSx/PETSc touch points, provide explicit
revisions:

```bash
python scripts/diagnostics/probe_nixpkgs_dolfinx.py \
  --revisions-file .codex_logs/nixpkgs_probe/path_candidates_near_locked.txt \
  --stop-on-pass \
  --update-lock
```

### Darwin linker warning mitigation (added 2026-03-03)

Observed symptom on macOS during FFCx/JIT compilation:

- Repeated linker noise:
  `ld: warning: directory not found for option '-L/nix/store/eeee.../lib'`
- In some noisy runs we also saw:
  `warning: unhandled Platform key FamilyDisplayName`

Root cause:

- Python `sysconfig` can expose `LDFLAGS`/`LDSHARED` values containing placeholder
  `-L` paths that do not exist on disk (for example `/nix/store/eeee...`).
- Those invalid search paths are inherited by extension/JIT builds and produce
  massive warning spam, even when tests still pass.

Mitigation implemented in `flake.nix` `shellHook` (Darwin only):

- Read Python `sysconfig` `LDFLAGS` and `LDSHARED`.
- Strip only invalid `-L` entries while keeping argument order.
- Export sanitized `LDFLAGS` and `LDSHARED` for the dev shell.
- Print a short diagnostic line with removed entry count.

Validation artifacts for this mitigation:

- Single-node before/after check:
  - `.codex_logs/upstream_warn_single_prefix_20260303-155327.log`
  - `.codex_logs/upstream_warn_single_postfix_20260303-155327.log`
  - `.codex_logs/upstream_warn_single_20260303-155327.counts.txt`
- Full upstream target file check (`test/unit/fem/test_assembler.py`):
  - `.codex_logs/upstream_test_assembler_full_after_lock_20260303-155501.log`
  - `.codex_logs/upstream_test_assembler_full_after_lock_20260303-155501.summary.tsv`
  - `.codex_logs/upstream_test_assembler_full_after_lock_20260303-155501.counts.txt`

Acceptance results from the full run above:

- `ld: warning: directory not found for option ...` count: `0`
- `unhandled Platform key FamilyDisplayName` count: `0`
- pytest summary: `252 passed, 6 warnings in 405.69s (0:06:45)`

Important:

- Do not treat linker warning spam by itself as evidence that the pinned
  `nixpkgs` revision is unstable.
- Keep using the upstream crash guard test (`mesh1`) as the pin/upgrade gate.

## Prerequisites

1. Install Nix.
2. Enable flakes.

```bash
mkdir -p ~/.config/nix
cat > ~/.config/nix/nix.conf <<'CONF'
experimental-features = nix-command flakes
CONF
```

## First-time setup

From the repository root:

```bash
nix develop
```

The `shellHook` in `flake.nix` will:

1. Set uv to use Nix-provided Python.
2. Create `.venv` on first run.
3. Use `--system-site-packages` so `.venv` can import Nix-provided DOLFINx packages.
4. Activate `.venv` automatically.

Then install PyEIDORS in editable mode:

```bash
uv pip install --python .venv/bin/python --no-deps -e .
```

Why `--no-deps`:

- Nix already pins the scientific stack.
- This prevents pip from replacing Nix-managed FEniCSx dependencies.

## Validation commands

```bash
python -c "import dolfinx, basix, ufl; print('dolfinx', dolfinx.__version__)"
python -c "from mpi4py import MPI; print('mpi4py size=', MPI.COMM_WORLD.size)"
python -c "import gmsh; print('gmsh', gmsh.__version__)"
pytest tests/unit/test_cache.py -v
```

Upstream FEniCSx regression guard (Darwin):

```bash
nix develop -c bash -lc 'cd .codex_logs/upstream_python_20260303-010524 \
  && python -P -m pytest -c pyproject.toml -s -vv \
  test/unit/fem/test_assembler.py::TestPETScAssemblers::test_symmetry_interior_facet_assembly[mesh1]'
```

## Troubleshooting

### 1) `attribute 'fenics-dolfinx' missing`

Possible causes:

- `flake.lock` points to a nixpkgs revision without FEniCSx packages.
- Local cache does not match the lock state.

Check and refresh:

```bash
cat flake.lock
nix flake update
```

### 2) Corrupted `.venv` or import mismatch

```bash
rm -rf .venv
nix develop
uv pip install --python .venv/bin/python --no-deps -e .
```

### 3) `import pyeidors` fails while `import dolfinx` works

This is not expected after the Phase-2 hard cutover. Check for stale local editable installs and reinstall:

```bash
uv pip install --python .venv/bin/python --no-deps -e .
```

## Scope boundary

- This document covers the supported runtime path: **FEniCSx-only** + Nix + uv.
- Legacy Docker notes are archived under `docs/archive/DOCKER_LEGACY.md`.
- Breaking API/name changes from the hard cutover are listed in `docs/MIGRATION_PHASE2.md`.
