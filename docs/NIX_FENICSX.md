# PyEIDORS: Nix + uv FEniCSx environment

This document defines the maintained environment setup for running PyEIDORS with FEniCSx on macOS (including Apple Silicon) and Linux, without Docker and without Conda.

## Strategy

PyEIDORS uses a two-layer lock strategy bound to this repository:

- System layer: `flake.lock` pins Nix packages (including DOLFINx/FEniCSx stack).
- Python layer: `uv.lock` pins Python packages, synchronized by `uv`.
- Python major/minor is fixed to `3.13` in the dev shell contract.
- Standard profile extras are fixed as: `torch`, `cuqi`, `dev`.
- Official entrypoint is `nix develop` (non-Nix path is not guaranteed 1:1 reproducible).

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
2. Create `.venv` on first run (or rebuild when Python major/minor changes).
3. Use `--system-site-packages` so `.venv` can import Nix-provided DOLFINx packages.
4. Activate `.venv` automatically.
5. Run `scripts/env/sync_locked_env.sh --check`.
6. If drift is detected, auto-run `scripts/env/sync_locked_env.sh --repair`.
7. Fail fast if repair still fails, with explicit manual recovery command.

## Validation commands

```bash
scripts/env/sync_locked_env.sh --print-profile
scripts/env/sync_locked_env.sh --check
python scripts/env/verify_env_manifest.py
python -c "import dolfinx, torch, cuqi, numpy, scipy, pyeidors"
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
scripts/env/sync_locked_env.sh --repair
```

### 3) `import pyeidors` fails while `import dolfinx` works

This is not expected after lock sync. Re-run:

```bash
scripts/env/sync_locked_env.sh --repair
python scripts/env/verify_env_manifest.py
```

### 4) Lock drift (`uv.lock` / profile mismatch)

Symptoms:

- `scripts/env/sync_locked_env.sh --check` fails.
- `python scripts/env/verify_env_manifest.py` reports mismatch keys.

Fix:

```bash
scripts/env/sync_locked_env.sh --repair
python scripts/env/verify_env_manifest.py
```

### 5) Network/index issue during sync

`--repair` may fail due to network/index transient failures. Keep the original error and retry once with your local proxy wrapper for that failing command only (do not enable global proxy permanently).

## Environment Upgrade Flow (Mandatory)

When changing environment inputs, keep this order:

1. Update `flake.lock` (if needed).
2. Update `uv.lock` in Nix shell (`uv lock --python .venv/bin/python --upgrade`).
3. Re-export manifests:
   - `python scripts/env/export_env_manifest.py --output env/manifests/macos-aarch64.lock.json --platform-id macos-aarch64`
   - `python scripts/env/export_env_manifest.py --output env/manifests/linux-x86_64.lock.json --platform-id linux-x86_64`
4. Verify:
   - `scripts/env/sync_locked_env.sh --check`
   - `python scripts/env/verify_env_manifest.py`

Rules:

- Do not do ad-hoc `pip install` without writing back to lock files.
- Any PR touching `pyproject.toml`, `flake.nix`, or `uv.lock` must update manifests and pass CI env guard.

## Scope boundary

- This document covers the supported runtime path: **FEniCSx-only** + Nix + uv.
- Legacy Docker notes are archived under `docs/archive/DOCKER_LEGACY.md`.
- Breaking API/name changes from the hard cutover are listed in `docs/MIGRATION_PHASE2.md`.
