# PyEIDORS: pure Nix FEniCSx environment

This document defines the maintained environment setup for running PyEIDORS with FEniCSx on macOS (including Apple Silicon) and Linux, without Docker and without Conda.

## Strategy

PyEIDORS uses a Nix-first lock strategy bound to this repository:

- Runtime layer: `flake.nix` and `flake.lock` pin the Nix Python package closure, including DOLFINx/FEniCSx, Torch, CUQI, Qt/PySide6, pyqtgraph, PETSc/SLEPc, MPI, HDF5, and CUDA variants.
- Manifest layer: `env/manifests/<platform>-<profile>.lock.json` records the active Nix profile, lock hashes, Python version, and package versions verified in that profile.
- Python major/minor is fixed to `3.13` in the dev shell contract.
- Official entrypoint is `nix develop` (non-Nix path is not guaranteed 1:1 reproducible).
- `uv` is an opt-in legacy/local maintenance path only; it is not the default runtime route.

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
nix develop .#complex64-cuda
```

The dev shell itself is the runtime bootstrap. It must be able to verify the core stack immediately:

```bash
python scripts/env/verify_env_manifest.py
python -c "import dolfinx, torch, cuqi, pyeidors, pyqtgraph; from PySide6.QtCore import Qt"
```

Legacy `uv` environment repair is opt-in only:

```bash
PYEIDORS_ENABLE_UV_SYNC=1 scripts/env/sync_locked_env.sh --repair
```

Important for WSL2 and other fresh shells:

- `python` inside `nix develop .#complex64-cuda` is the supported runtime interpreter.
- `.venv*` and `uv run` are not default runtime entrypoints.
- If `nix` itself is missing on WSL2, install Nix first; the repository does not support a 1:1 reproducible non-Nix bootstrap for DOLFINx.
- If a plain WSL2 shell can `import pyeidors` but fails on `pyeidors.EITSystem` with NumPy/Torch/shared-library errors, that still counts as an unsupported runtime state; re-enter with `nix develop` before debugging deeper.
- When the Linux manifest is exported from WSL2, it may record `platform.runtime_context.kind = wsl2` as informational provenance only; `verify_env_manifest.py` does not treat that field as a hard compatibility gate.
- For CUDA on WSL2/NVIDIA, supported dev shells are `.#cuda` for real-only CUDA and `.#complex64-cuda` / `.#complex-cuda` for complex-capable CUDA; do not treat the default CPU shell as a GPU runtime.
- After entering a CUDA shell, run `python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty` before enabling `--petsc-device auto|cuda` in benchmarks or CLI runs; use `--device auto|cuda` as the matching Torch/GN inverse runtime switch.
- For the GUI, the supported default launcher is `bash scripts/gui/run_eit_app.sh` (`--auto`). It chooses `complex64-cuda` when GPU is visible and `complex64` otherwise; use `--precision complex128` for the complex128 profiles. Do not launch the GUI with `PYTHONPATH=src python -m eit_app.app`; that drops nix-provided FEniCSx runtime paths and can break realtime reconstruction imports.
- On the Windows host, the supported launcher is `powershell -File .\scripts\gui\run_eit_app.ps1 -Profile auto`, or the repository-root one-click wrapper `EIT-GUI.cmd`. `EIT-GUI-CPU.cmd` / `EIT-GUI-GPU.cmd` remain compatibility wrappers for capability CPU/GPU launches.
- For a lightweight preflight before the full stack is present, you can still inspect package detection with:

```bash
PYTHONPATH=src python -c "import pyeidors; print(pyeidors.check_environment())"
```

That `PYTHONPATH=src` form is only for lightweight import probing. It is not the
supported way to run the GUI or any full reconstruction workflow.

The `shellHook` in `flake.nix` will:

1. Set `UV_PYTHON` to the Nix-provided Python for legacy tooling.
2. Unset `VIRTUAL_ENV` / `VIRTUAL_ENV_PROMPT` so `.venv*` cannot shadow the Nix profile.
3. Set `PYEIDORS_ACTIVE_ENV=nix` and add repository `src/` to `PYTHONPATH`.
4. Import-check `dolfinx, torch, cuqi, numpy, scipy, pyeidors, pyqtgraph, PySide6`.
5. Initialize a terminal-scoped cache session under `.pyeidors_cache/v2/.sessions/<session-id>` and clear that shell-owned cache automatically on shell exit, signal termination, or `deactivate`.

CPU and CUDA shells follow the same cache-lifecycle rule: each terminal gets its own runtime disk cache, terminals do not share session cache, and closing one shell only cleans that shell's cache. Bare `.venv/bin/python` outside the shell hook is not the supported long-running workflow.

### Optional performance extras

PyEIDORS keeps the default user runtime in Nix. Optional acceleration extras can
still be explored through the legacy uv maintenance route:

- `pyamg>=5.2`
- `scikit-sparse>=0.4.12`

If you only want to add them to an existing shell environment, enable them explicitly:

```bash
PYEIDORS_ENABLE_UV_SYNC=1 \
ENABLE_PERFORMANCE_EXTRAS=1 scripts/env/sync_locked_env.sh --repair
```

Verification:

```bash
python -c "import pyamg; print(pyamg.__version__)"
python -c "import sksparse"
python -c "from sksparse import cholmod; print('cholmod ok')"
```

For fair 3D benchmark comparisons, also pin the thread count so BLAS / OMP do not skew medians:

```bash
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
ENABLE_PERFORMANCE_EXTRAS=1 \
/nix/var/nix/profiles/default/bin/nix --extra-experimental-features 'nix-command flakes' develop -c \
python scripts/benchmarks/benchmark_3d_fair_compare.py --benchmark-phase quick
```

Recommended interpretation for the fused 3D profiles:

- `quick` now compares `A_baseline` against `D_combined`, because `D_combined` is the delivery profile for 3D fast mode.
- `full` should be read together with `check_perf_gate.py`; strict gate now focuses on `B_cholmod_only`, `C_autotune_only`, and `D_combined`.
- `E_fused` remains available for research runs (`rom_mode=on`, `inexact_mode=auto/on`, `lowrank_mode=auto/on`) but is no longer the primary performance gate target.
- For the mac CPU封版 summary and historical migration handoff, see `docs/WSL2_CUDA_HANDOFF.md`.
- For the active CUDA profile workflow, see `docs/WSL2_CUDA.md`.

`sksparse` availability depends on SuiteSparse support in the current platform
toolchain; absence is non-fatal and PyEIDORS will fall back to PETSc/SciPy paths.

Nix note:

- `flake.nix` now includes `pkgs.suitesparse` in the dev shell.
- `shellHook` prints a one-line status for `pyamg/sksparse/cholmod` so you can
  confirm acceleration capabilities immediately after `nix develop`.

## Validation commands

```bash
python scripts/env/verify_env_manifest.py
python -c "import dolfinx, torch, cuqi, numpy, scipy, pyeidors, pyqtgraph; from PySide6.QtCore import Qt"
```

CUDA profile validation:

```bash
nix develop .#cuda
python scripts/diagnostics/probe_petsc_cuda.py --require cuda --pretty
python scripts/env/verify_env_manifest.py --profile cuda
```

## Mesh cache policy

PyEIDORS follows the DOLFINx Gmsh I/O guidance for large problems: use
`gmshio.model_to_mesh()` or `gmshio.read_from_msh()` only for the first import,
then persist the DOLFINx mesh and `MeshTags` for repeat access.

Runtime behavior:

- The source mesh remains `<name>.msh` for Gmsh provenance and old-cache
  compatibility.
- The fast reusable cache is `<name>.xdmf` plus its HDF5 sidecar `<name>.h5`.
- `<name>_dolfinx_cache.json` stores the association table, physical-group
  dimensions, generator metadata, and source `.msh` size/mtime signature.
- `MeshLoader` and `load_or_create_mesh()` prefer the XDMF/HDF5 cache when it is
  fresh, and fall back to `.msh` import only when the native cache is missing or
  stale.
- The process-local mesh cache keys use cheap path metadata signatures rather
  than hashing full mesh payloads, so large HDF5 caches do not get re-read just
  to compute a Python-process cache key.

ADIOS2/VTX note:

- DOLFINx exposes `VTXWriter` for ADIOS2/BP output, but its Python API is a
  writer path, not the authoritative mesh/tag reload path used here.
- To emit an optional ADIOS2 mesh snapshot next to the XDMF cache, set:

```bash
PYEIDORS_WRITE_ADIOS2_MESH_CACHE=1 python <your-script>.py
```

That `.bp` artifact is useful for downstream ADIOS2/visualization workflows, but
PyEIDORS reloads CEM meshes from XDMF/HDF5 because that path preserves the mesh
and named `MeshTags` through official DOLFINx read APIs.

ADIOS4DOLFINx checkpoint option:

- `adios4dolfinx` is a third-party checkpoint layer for DOLFINx mesh,
  `MeshTags`, and `Function` data. It is appropriate for very large MPI runs
  where scalable checkpoint/restart matters more than staying inside the core
  DOLFINx API surface.
- It is intentionally optional here. The maintained default remains XDMF/HDF5.
- The current Nix dev shell already provides Python `adios2`, but does not
  currently provide `adios4dolfinx`. Install it only in a compatible environment
  where the ADIOS2 MPI build matches the DOLFINx/MPI runtime.
- When `adios4dolfinx` is installed, emit an additional checkpoint with:

```bash
PYEIDORS_WRITE_ADIOS4DOLFINX_CHECKPOINT=1 python <your-script>.py
```

Optional engine override:

```bash
PYEIDORS_ADIOS4DOLFINX_ENGINE=BP4 \
PYEIDORS_WRITE_ADIOS4DOLFINX_CHECKPOINT=1 \
python <your-script>.py
```

The generated `<mesh>_adios4dolfinx.bp` is recorded in
`<mesh>_dolfinx_cache.json`. It is not used by default GUI reloads unless we
later promote the optional checkpoint layer into the runtime selection policy.

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

### 2) Stale `.venv` shadows the Nix runtime

```bash
unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT
nix develop .#complex64-cuda
python scripts/env/verify_env_manifest.py
```

### 3) `import pyeidors` fails while `import dolfinx` works

This is not expected inside the Nix profile. Re-run:

```bash
python scripts/env/verify_env_manifest.py
```

### 3b) Plain WSL2 shell hits `libstdc++.so.6` / NumPy / Torch import errors

Symptoms:

- plain-shell `python -c "import pyeidors; print(pyeidors.check_environment())"` works, but
- full-stack imports fail with a shared-library or runtime import error.

Fix:

```bash
nix develop .#complex64-cuda
python scripts/env/verify_env_manifest.py
```

This repository only treats the `nix develop` shell as the supported full-runtime entrypoint for WSL2/Linux.

### 4) Manifest/profile drift

Symptoms:

- `python scripts/env/verify_env_manifest.py` reports mismatch keys.

Fix:

```bash
python scripts/env/export_env_manifest.py --output env/manifests/linux-x86_64-complex64-cuda.lock.json
python scripts/env/verify_env_manifest.py
```

### 5) Network/index issue during optional uv sync

`--repair` may fail due to network/index transient failures. Keep the original error and retry once with your local proxy wrapper for that failing command only (do not enable global proxy permanently).

## Environment Upgrade Flow (Mandatory)

When changing environment inputs, keep this order:

1. Update `flake.lock` (if needed).
2. Update `flake.nix` package closure if the runtime package set changed.
3. Re-export manifests from the matching Nix profile:
   - `python scripts/env/export_env_manifest.py --output env/manifests/macos-aarch64.lock.json --platform-id macos-aarch64`
   - `python scripts/env/export_env_manifest.py --output env/manifests/linux-x86_64.lock.json --platform-id linux-x86_64`
   - `python scripts/env/export_env_manifest.py --output env/manifests/linux-x86_64-complex64-cuda.lock.json --profile complex64-cuda`
4. Verify:
   - `python scripts/env/verify_env_manifest.py`

Rules:

- Do not do ad-hoc `pip install` for the supported runtime.
- Any PR touching `pyproject.toml`, `flake.nix`, or `flake.lock` must update manifests and pass CI env guard.

## Scope boundary

- This document covers the supported runtime path: **FEniCSx-only** + pure Nix.
- Docker content from the old runtime has been removed; `docs/DOCKER.md` records the current status.
