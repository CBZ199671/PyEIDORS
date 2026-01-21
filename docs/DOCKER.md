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
  --gpus all \
  --shm-size=24g \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cpus=20 \
  --memory=28g \
  -v "$(pwd):/workspace" \
  -w /workspace \
  --name pyeidors \
  ghcr.io/cbz199671/pyeidors-env:latest
```

Inside the container you can run scripts directly. `PYTHONPATH` is set to `/workspace/src` so you do not need `pip install -e .` for basic usage.

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

