# Docker Status

PyEIDORS Phase-2 is maintained on a **FEniCSx-only** runtime and the primary workflow is:

- Nix + uv (`docs/NIX_FENICSX.md`)

Docker is not the primary maintained execution path in current CI. Historical Docker instructions were moved to:

- `docs/archive/DOCKER_LEGACY.md`

If you need containerized reproducibility, use the archived instructions as a reference and validate behavior against the current test suite (`pytest --cov=src/pyeidors --cov-fail-under=80`) before relying on it for production/research runs.
