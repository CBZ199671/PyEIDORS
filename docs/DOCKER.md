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

## Option B: use an offline image archive (`docker load`)

If you distribute a “frozen” image as a compressed archive (for example `pyeidors-ultra22.tar.zst`),
users can load it locally without building or pulling from a registry.

Do not commit image archives into this Git repository. Publish them as Release assets or host them externally.

### Windows prerequisites (recommended)

1. Install WSL2 and an Ubuntu distribution (Ubuntu 22.04 LTS recommended).
2. Install Docker Desktop and enable the WSL2 backend + WSL integration.
3. (GPU only) Ensure your NVIDIA driver supports WSL2 + Docker GPU.

### Load the image (recommended inside WSL2)

Install `zstd` in WSL2:

```bash
sudo apt-get update && sudo apt-get install -y zstd
```

From the directory that contains `pyeidors-ultra22.tar.zst`, load the image:

```bash
zstd -d -c pyeidors-ultra22.tar.zst | docker load
```

If your machine has limited RAM, use a two-step load:

```bash
zstd -d pyeidors-ultra22.tar.zst -o pyeidors-ultra22.tar
docker load -i pyeidors-ultra22.tar
rm -f pyeidors-ultra22.tar
```

### Run the container

Minimal (portable) command:

```bash
docker run -it --rm \
  -v "$(pwd):/root/shared" \
  -w /root/shared \
  --name pyeidors \
  pyeidors:latest \
  bash
```

Keep a container running (WSL2/Linux):

```bash
docker run -d \
  --name pyeidors \
  --restart unless-stopped \
  -v "$(pwd):/root/shared" \
  -w /root/shared \
  pyeidors:latest \
  sleep infinity
```

Windows PowerShell example (equivalent to the command above):

```powershell
docker run -d `
  --name pyeidors `
  --restart unless-stopped `
  -v "${PWD}:/root/shared" `
  -w /root/shared `
  pyeidors:latest `
  sleep infinity
```

Optional performance flags (add only if needed / supported on your platform):

- GPU: `--gpus all`
- Shared memory: `--shm-size=2g` (or larger), and on Linux optionally `--ipc=host`
- Resource limits: `--cpus=<n> --memory=<size>`
- Host networking: `--network=host` (Linux-only; not supported on all platforms)

## Option C: build the image locally from `Dockerfile`

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

### Publishing an offline archive (maintainers)

If you need to publish a `docker load` archive, create it on your machine:

```bash
docker save pyeidors:latest | zstd -19 -T0 -o pyeidors-ultra22.tar.zst
sha256sum pyeidors-ultra22.tar.zst > pyeidors-ultra22.tar.zst.sha256
```

If the archive is larger than the hosting per-file limit, split it into parts:

```bash
split -b 1900M pyeidors-ultra22.tar.zst pyeidors-ultra22.tar.zst.part-
```

Users can reassemble it with:

```bash
cat pyeidors-ultra22.tar.zst.part-* > pyeidors-ultra22.tar.zst
```

## Notes

- The image is large. Do not upload exported tarballs (from `docker save`) to Git.
- FEniCS is provided by the base image. This Dockerfile focuses on freezing the Python stack (CUQIpy, uv, PyTorch) and convenience defaults.
