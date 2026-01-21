# Docker usage

PyEidors is developed and tested inside Docker. The recommended workflow is to use a prebuilt image that already contains:

- FEniCS (from `ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30`)
- CUQIpy + CUQIpy-FEniCS
- PyTorch (GPU wheels by default)

The image is large (multi-GB). Do not commit `docker save` archives into the Git repository.

## Option A (recommended): pull a prebuilt image

1. Pull the image (replace the tag with a published version):

```bash
docker pull ghcr.io/cbz199671/pyeidors-env:latest
```

2. Run it and mount your working directory:

```bash
docker run -ti \
  -v "$(pwd):/root/shared" \
  -w /root/shared \
  --name pyeidors \
  ghcr.io/cbz199671/pyeidors-env:latest
```

Inside the container you can run scripts directly. For editable installs, run `pip install -e .`.

### Runtime flags (tuning)

Docker has no fully automatic “fit to my hardware” mode for these flags, but the default behavior is already adaptive:
if you omit `--cpus` and `--memory`, Docker can use the available host resources (subject to Docker Desktop limits).

Use the following flags only when needed:

- **GPU**: `--gpus all` (requires NVIDIA Container Toolkit / Docker Desktop GPU support).
- **Shared memory**: if you see `/dev/shm` errors, add `--shm-size=2g` (or larger). On Linux, `--ipc=host` can also help.
- **Limit resources** (optional): use `--cpus=<n>` and/or `--memory=<size>` to cap usage on smaller machines.

### Mount path (Windows / Linux)

The `-v <host_path>:/root/shared` host path should point to your local clone of this repository.

- Linux/macOS (bash): `-v "$(pwd):/root/shared"`
- Windows PowerShell: `-v "${PWD}:/root/shared"`
- Windows cmd.exe: `-v "%cd%:/root/shared"`

## Option B: build the image locally from `Dockerfile`

```bash
docker build -t pyeidors-env:local .
```

CPU-only build (no GPU wheels):

```bash
docker build -t pyeidors-env:cpu --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
```

## Publishing the image (maintainers)

This repository includes a GitHub Actions workflow that can build and publish the image to GitHub Container Registry (GHCR).

Notes:

- You must make the resulting package public in GitHub Packages if you want unauthenticated pulls.
- A typical naming scheme is `ghcr.io/cbz199671/pyeidors-env:<git-tag>` plus `:latest`.

## Notes

- The image is large. Do not upload exported tarballs (from `docker save`) to Git.
- FEniCS is provided by the base image. This Dockerfile focuses on freezing the Python stack (CUQIpy, uv, PyTorch) and convenience defaults.
