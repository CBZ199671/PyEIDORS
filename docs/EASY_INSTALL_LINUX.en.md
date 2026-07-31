# PyEIDORS @VERSION@ One-Click Linux Installation

This guide is written for users who may be new to Linux, Nix, CUDA, or PyEIDORS. You do not need to configure Python, PyTorch, CUDA, PETSc, SLEPc, FEniCSx, or DOLFINx manually.

PyEIDORS @VERSION@ is currently a private pre-publication preview. Share it only with users authorized by the author. Do not publish or redistribute it.

## 1. Choose exactly one installer

| Installer | Computer | Included runtime |
|---|---|---|
| `PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run` | No NVIDIA GPU; Professor Andy should use this package | real, complex64, and complex128 on CPU |
| `PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-SM61-LINUX.run` | GTX 10xx, compute capability 6.1 | real and complex64 on GPU; complex128 automatically on CPU; all CPU modes retained |
| `PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-MODERN-LINUX.run` | GTX 16xx and RTX 20/30/40/50 | real, complex64, and complex128 on GPU; CPU fallback for all modes |

All editions install under:

```text
~/apps/PyEIDORS-@VERSION@
```

Do not install several editions on top of each other.

## 2. How the installer prevents version conflicts

### Archive tools

The installer prefers absolute system tools under `/usr/bin` or `/bin`. It does not trust a command merely because it appears first on PATH. It performs a real tar+zstd create, test, and extract probe.

Broken or incompatible `tar`, `zstd`, `unzip`, or `curl` shims under a home directory, Conda, or another custom PATH therefore cannot silently take control.

If system tools are genuinely missing or broken, the installer can repair them through `apt`, `dnf`, or `pacman`, then repeats the functional probe.

### Nix

Each existing Nix candidate must have:

- a working `nix --version`;
- Nix version 2.4 or newer;
- a working `nix-store` in the same `bin` directory.

The cache importer receives the already validated absolute Nix path. It does not search PATH again. A stale alias, shell shim, or unrelated old Nix cannot replace it halfway through installation.

If an old or damaged Nix installation already exists, the installer stops with an upgrade/repair message instead of creating a conflicting second Nix installation.

### Python, PyTorch, and CUDA

The packaged Python, NumPy, SciPy, PyTorch, CUDA Toolkit, PETSc, SLEPc, FEniCSx, DOLFINx, and Qt come from the prebuilt Nix store closure.

The installed launcher removes host overrides including:

- `PYTHONHOME`, `PYTHONPATH`, `PYTHONSTARTUP`, `PYTHONUSERBASE`;
- `VIRTUAL_ENV` and common `CONDA_*` activation variables;
- `CUDA_HOME`, `CUDA_PATH`, `CUDA_ROOT`, `CUDACXX`;
- `PETSC_DIR`, `SLEPC_DIR`, `CMAKE_PREFIX_PATH`;
- `LD_PRELOAD` and host Qt plugin paths.

GPU wrappers force their own packaged `CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `PETSC_DIR`, and `SLEPC_DIR`.

You do not need to uninstall an existing CUDA, PyTorch, venv, or Conda environment. They are ignored when PyEIDORS is started through its installed launcher.

The NVIDIA Linux driver is the intentional host dependency because it must match the real GPU and host kernel. The package does not install or modify that driver.

## 3. Requirements

- x86_64 64-bit Linux.
- Ubuntu 22.04/24.04 is recommended. Other common Debian, Fedora, and Arch-family distributions are supported.
- Native Linux is preferred. WSL2 Ubuntu is supported; WSL1 is not.
- Run as an ordinary user. Do not put `sudo` before the `.run` file.
- A GPU edition needs a working `nvidia-smi`.
- A Linux desktop, WSLg, X11, or remote desktop session is required for the GUI.

Recommended free space:

| Edition | Nix-store filesystem | Temporary filesystem |
|---|---:|---:|
| CPU Universal | at least 30 GiB | at least 10 GiB |
| NVIDIA SM61 | at least 50 GiB | at least 14 GiB |
| NVIDIA Modern | at least 70 GiB | at least 20 GiB |

## 4. Verify the download

Download only the selected `.run` file and:

```text
PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt
```

Put both in the same directory and run:

```bash
sha256sum -c --ignore-missing PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt
```

The selected file should report `OK`. The `.run` file also verifies its embedded SHA-256 before extraction.

## 5. Install

Professor Andy should use:

```bash
cd ~/Downloads
chmod +x PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
./PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

If executable permission was removed by a browser, cloud drive, USB disk, or Windows filesystem, run:

```bash
bash PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

Do not run:

```bash
sudo ./PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

The installer requests the Linux password itself only when a system package or multi-user Nix operation needs it.

## 6. What the installer does

1. Checks Linux, x86_64, and ordinary-user execution.
2. Resolves and functionally tests system archive tools.
3. Checks temporary free space.
4. Verifies the embedded payload.
5. Validates or installs Nix.
6. Imports the complete prebuilt local Nix cache.
7. Builds a new installation in a staging directory.
8. Clears host Python/CUDA overrides and runs real, complex64, and complex128 CPU doctors.
9. Replaces the live installation only after all doctors pass.
10. Creates Nix garbage-collection roots and verifies hardware routing.

The local cache is unsigned and imported with `--no-check-sigs`. There is no cache key, no public/private key file, no `extra-trusted-public-keys`, no `/etc/nix/nix.conf` edit, and no key-related daemon restart.

## 7. Start

Default complex64:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh
```

Other modes:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --real
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --complex64
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --complex128
```

Show the hardware decision without opening the GUI:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --show-selection
```

Force a GPU package to use CPU:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --cpu
```

The CPU package always uses CPU. SM61 uses a compatible GTX 10xx only for real and complex64; complex128 uses CPU without reducing precision. Modern NVIDIA uses GPU for all three modes on supported hardware. A missing driver, missing GPU, or mismatched package safely falls back to CPU.

## 8. Troubleshooting

### Old or broken tar/zstd

The installer runs an actual archive probe. Ask the administrator to repair the distribution packages for:

```text
tar zstd unzip curl ca-certificates xz
```

Do not replace `/usr/bin` tools with binaries from an unknown website.

### Old or damaged Nix

Check:

```bash
/nix/var/nix/profiles/default/bin/nix --version
/nix/var/nix/profiles/default/bin/nix-store --version
```

Ask the administrator to upgrade or repair it if it is absent, broken, or older than Nix 2.4. Do not install a second Nix merely to bypass the check.

### Nix was installed but not yet found

Close the terminal, open a new terminal, and rerun the same `.run` file. Re-running is safe.

### CUDA or PyTorch is already installed

Leave it installed. Do not alter Conda or venv. Start PyEIDORS only through:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh
```

Running a random system `python` from the extracted source directory intentionally bypasses the one-click runtime isolation.

### GPU is not selected

Run:

```bash
nvidia-smi
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --show-selection
```

If `nvidia-smi` fails, repair the NVIDIA Linux driver first. GTX 10xx needs SM61; GTX 16xx and RTX 20/30/40/50 need Modern.

### Not enough temporary space

Use a larger Linux filesystem:

```bash
mkdir -p "$HOME/pyeidors-tmp"
TMPDIR="$HOME/pyeidors-tmp" bash PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

Replace the filename for a GPU edition.

### Path contains spaces

Quote the full filename:

```bash
bash "/path/with spaces/PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run"
```

### Installation was interrupted

Rerun the same `.run` file. Temporary files are cleaned automatically. The live directory changes only after all doctors pass.

If the final switch fails, the new directory is retained as `.failed-TIMESTAMP` and the previous working installation is restored automatically.

### Reinstallation

The previous directory is backed up as:

```text
~/apps/PyEIDORS-@VERSION@.backup-TIMESTAMP-PID
```

Remove old backups manually only after confirming the new installation.

### GUI does not open

Use a Linux desktop, WSLg, configured X11 forwarding, or a remote desktop. A plain SSH terminal has no graphical display unless one is configured.

### Logs

Installation logs are under:

```text
~/.cache/pyeidors-installer/
```

For support, send the newest `PyEIDORS-@VERSION@-*.log` with the Linux distribution, hardware, selected edition, and `nvidia-smi` result.

## 9. Uninstall

The user installation is:

```text
~/apps/PyEIDORS-@VERSION@
```

Removing it removes the launcher, private source snapshot, and associated Nix GC roots. Nix store data is reclaimed only by a later Nix garbage collection.

Do not remove the entire `/nix` directory just to uninstall PyEIDORS.

## 10. Files to send to Professor Andy

Send only:

1. `PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run`
2. `PyEIDORS-@VERSION@-EASY-INSTALL-README-EN.md`
3. `PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt`

The GPU installers, source zip, manual tar archives, and Nix keys are not needed.
