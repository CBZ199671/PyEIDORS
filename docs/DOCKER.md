# Docker Status

PyEIDORS Phase-2 is maintained on a **FEniCSx-only** runtime and the primary workflow is:

- Pure Nix (`docs/NIX_FENICSX.md`)

Docker is not a maintained execution path in current CI. The previous Docker image and GHCR publish workflow depended on the removed DOLFIN-era stack, so they were deleted during the FEniCSx hard cutover.

If containerized reproducibility is needed again, rebuild it from the supported Nix contract and validate behavior against the current test suite before relying on it for production or research runs.
