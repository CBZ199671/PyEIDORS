# Reference: Existing Codebase

The existing source tree is the primary source of truth for this brownfield
sketch.

## Primary Sources

- `README.md`: user-facing system summary, launch commands, and API notes.
- `FILE_ORGANIZATION.md`: file-level map of core package domains.
- `pyproject.toml`: runtime dependencies, optional extras, package data, pytest
  configuration, and `eit-app` entrypoint.
- `docs/MEASUREMENT_DATA_SPEC.md`: standardized real-measurement interface.
- `docs/CACHE_ARCHITECTURE.md`: semantic cache design and invalidation rules.
- `docs/EIDORS_PYEIDORS_INTEROP.md`: bridge format and validation rule.
- `docs/WSL2_CUDA.md`: CPU/CUDA shell split, probe policy, and device switches.
- `src/pyeidors/`: reusable EIT framework implementation.
- `src/eit_app/`: desktop application implementation.
- `tests/unit/` and `tests/integration/`: current executable behavior.

## How To Use This Reference

1. Read `context/refs/architecture-overview.md` for orientation.
2. Read `context/kits/cavekit-overview.md` to select a domain.
3. For a selected domain, inspect the source and tests named in the kit.
4. When behavior and kit disagree, treat current tests and code as brownfield
   evidence, then decide whether the kit captures intended behavior or a bug.

## What The Codebase Tells Us

- Current public workflow shape.
- Current command-line and GUI launch paths.
- Current metadata and file format contracts.
- Current solver and cache behavior under supported test cases.
- Current environment constraints for WSL2, CUDA, Qt, and MATLAB/EIDORS bridge.

## What The Codebase Does Not Tell Us

- Which historical behaviors are accidental rather than desired.
- Product priority between research-only paths and production GUI paths.
- Full physical hardware acceptance thresholds without device-in-loop runs.
- Long-term API compatibility guarantees beyond current tests and docs.

