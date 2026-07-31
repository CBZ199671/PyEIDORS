# PyEIDORS @VERSION@ One-Click Linux Installation Guide

This guide is for all Linux users, including people who are new to Linux, Nix, CUDA, or PyEIDORS. In a normal installation, you only need to choose one `.run` file for your hardware and run it. You do not need to configure Python, PyTorch, the CUDA Toolkit, PETSc, SLEPc, FEniCSx, or DOLFINx manually.

PyEIDORS @VERSION@ is currently a private pre-publication preview. Use it only if you have permission from the author. Do not upload it to a public file-sharing service or GitHub Release, and do not redistribute it without permission.

## 1. Choose the installer for your computer

Download and install exactly one of the three editions:

| Installer | Hardware | Runtime capability |
|---|---|---|
| `PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run` | No NVIDIA GPU, CPU-only use, or uncertain GPU compatibility | real, complex64, and complex128 on CPU |
| `PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-SM61-LINUX.run` | NVIDIA GTX 10xx, compute capability 6.1 | real and complex64 on GPU; complex128 automatically on CPU; all three CPU environments retained |
| `PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-MODERN-LINUX.run` | NVIDIA GTX 16xx and RTX 20/30/40/50 | real, complex64, and complex128 on GPU; all modes can be forced to CPU |

If you do not know whether the computer has a usable NVIDIA GPU, run:

```bash
nvidia-smi
```

- If the command is missing, fails, or lists no GPU, choose CPU Universal.
- If it shows a GTX 10xx, choose NVIDIA SM61.
- If it shows a GTX 16xx or RTX 20/30/40/50, choose NVIDIA Modern.
- If the model is not listed above, choose CPU Universal first and ask the software maintainer about compatibility.

All editions install under:

```text
~/apps/PyEIDORS-@VERSION@
```

Do not install several editions consecutively into the same live directory. To change editions later, run the new installer. It verifies the new environment first, then backs up the old directory and switches over.

## 2. What the one-click package provides

The installer contains a prebuilt Nix runtime and its complete dependency closure. Its purpose is to keep PyEIDORS independent from development environments already installed on the computer.

### 2.1 System archive tools are functionally tested

The installer prefers `tar`, `zstd`, `unzip`, and `curl` from absolute system locations such as `/usr/bin` and `/bin`. It does not trust a command merely because it appears first on PATH.

Before installation, it creates, tests, and extracts a small tar+zstd archive. An old, broken, or disguised command under a home directory, Conda environment, or custom PATH therefore cannot silently take control.

If the real system tools are missing or unusable, the installer can try to repair them using a common distribution package manager:

- Ubuntu/Debian: `apt`
- Fedora/RHEL families: `dnf`
- Arch families: `pacman`

The script asks for `sudo` only at the step that needs system access and repeats the functional probe afterward. If validation still fails, it stops with a clear error instead of continuing with broken tools.

### 2.2 Existing Nix is validated before reuse

The installer checks Nix in this order:

1. `/nix/var/nix/profiles/default/bin/nix`
2. `~/.nix-profile/bin/nix`
3. other candidates from the original PATH used to start the installer

A candidate must have all of the following:

- a working `nix --version`;
- Nix version 2.4 or newer;
- a working `nix-store` in the same `bin` directory;
- a working `nix-store --version`.

The cache importer uses only the validated absolute path and does not search PATH again. A stale script, shell alias, or broken `nix` shim cannot replace the correct Nix halfway through installation.

- If the existing Nix is valid, it is reused without changing its version.
- If Nix is completely absent, the official Nix installer is used: multi-user with systemd, otherwise single-user.
- If an existing Nix is old or damaged, installation stops with repair instructions instead of installing a conflicting second copy.

### 2.3 Python, PyTorch, and CUDA are isolated from the host

The packaged Python, NumPy, SciPy, PyTorch, CUDA Toolkit, PETSc, SLEPc, FEniCSx, DOLFINx, and Qt come from the prebuilt Nix closure.

When you use `start-pyeidors.sh`, it removes common host overrides, including:

- `PYTHONHOME`, `PYTHONPATH`, `PYTHONSTARTUP`, and `PYTHONUSERBASE`;
- `VIRTUAL_ENV` and common `CONDA_*` activation variables;
- `CUDA_HOME`, `CUDA_PATH`, `CUDA_ROOT`, and `CUDACXX`;
- `PETSC_DIR`, `SLEPC_DIR`, and `CMAKE_PREFIX_PATH`;
- `LD_PRELOAD` and host Qt plugin paths.

GPU environments explicitly select the package's own `CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `PETSC_DIR`, and `SLEPC_DIR`.

You do not need to uninstall an existing CUDA, PyTorch, Conda, or venv environment, and you do not need to change it to the versions used by this package. Those environments do not override PyEIDORS when you use the installed launcher.

The NVIDIA Linux driver is the only intentional host hardware dependency because it must match the real GPU and Linux kernel. The package does not install, upgrade, or remove the GPU driver.

### 2.4 The local cache needs no key

The bundled Nix cache is imported locally with `--no-check-sigs`. Therefore:

- no cache key is required;
- no public or private key file is required;
- no `extra-trusted-public-keys` setting is required;
- `/etc/nix/nix.conf` does not need to be edited;
- `nix-daemon` does not need a key-related restart.

If older instructions ask you to add a cache public key, you have an outdated installer or outdated documentation. Use the current release instead.

## 3. Before installation

### 3.1 System requirements

- x86_64 64-bit Linux.
- Ubuntu 22.04/24.04 is recommended. Other common Debian, Fedora, and Arch-family distributions can also be used.
- Native Linux is recommended. WSL2 Ubuntu is supported; WSL1 is not.
- Install as an ordinary user. Do not put `sudo` before the `.run` file.
- A GPU edition requires `nvidia-smi` to display the GPU and driver correctly.
- The GUI requires a Linux desktop, WSLg, X11 forwarding, or a remote desktop session.
- If Nix is completely absent, internet access is needed while the official Nix installer is downloaded.

If a system tool or multi-user Nix operation needs administrator access, the script invokes `sudo` at that step and asks for the current Linux user's password.

### 3.2 Disk space

Keep at least the following space free:

| Edition | Nix-store filesystem | Temporary filesystem |
|---|---:|---:|
| CPU Universal | 30 GiB | 10 GiB |
| NVIDIA SM61 | 50 GiB | 14 GiB |
| NVIDIA Modern | 70 GiB | 20 GiB |

Temporary space is used to verify and extract the large payload. The installer checks free space before writing large amounts of data. If `/tmp` is small, set `TMPDIR` as shown in Section 9.

### 3.3 Verify the download

Download the selected `.run` file and:

```text
PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt
```

Put them in the same directory and run:

```bash
sha256sum -c --ignore-missing PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt
```

The selected file should report `OK`. You do not need to download the other two large installers just to verify one file.

Even if you skip this manual check, the `.run` file verifies the embedded payload's SHA-256 before extraction. An incomplete or damaged download stops safely.

## 4. Install with one command

Open a terminal and enter the download directory. Most browsers use `~/Downloads`:

```bash
cd ~/Downloads
```

Run one of the following commands for the edition you downloaded.

CPU Universal:

```bash
bash PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

NVIDIA SM61:

```bash
bash PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-SM61-LINUX.run
```

NVIDIA Modern:

```bash
bash PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-MODERN-LINUX.run
```

Using `bash filename.run` does not depend on executable permission. It works after downloading through a browser, cloud drive, USB disk, Windows filesystem, or network share.

If the file is not in `~/Downloads`, enter its real directory or quote the full path:

```bash
bash "/actual/path/PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run"
```

Do not run:

```bash
sudo bash PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

Starting the whole installer as root gives root ownership of the user directory, logs, and launcher. Individual steps that need administrator access request it themselves.

## 5. What you will see during installation

The installer performs these steps:

1. Checks Linux, x86_64, and ordinary-user execution.
2. Resolves and functionally tests archive tools from absolute system paths.
3. Checks free space on the temporary and Nix-store filesystems.
4. Verifies the embedded payload's SHA-256.
5. Extracts the local Nix binary cache and private source snapshot.
6. Validates existing Nix or installs Nix if it is completely absent.
7. Imports the complete prebuilt Nix store closure.
8. Builds a new installation in a staging directory.
9. Clears host environment overrides and runs the real, complex64, and complex128 CPU doctors.
10. Switches the live directory only after every check succeeds.
11. Creates Nix garbage-collection protection and checks CPU/GPU routing.
12. Prints the installation path, log path, and launch command.

Verifying, extracting, and importing the large cache can take time. If the terminal is still printing stage information, let it continue. Do not close the terminal or suspend the computer.

## 6. Start PyEIDORS

Start the default complex64 mode:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh
```

Select a numeric mode:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --real
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --complex64
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --complex128
```

Show the hardware decision without opening the GUI:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --show-selection
```

Force a GPU edition to use CPU:

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --cpu
```

Always use this launcher. Running a system `python` directly in the source directory bypasses the one-click Nix runtime and its conflict isolation.

## 7. Automatic hardware selection

- CPU Universal always uses CPU, even if an NVIDIA GPU is present.
- SM61 uses a compatible GTX 10xx / compute capability 6.1 GPU for real and complex64.
- SM61 always runs complex128 on CPU without reducing it to complex64 precision.
- NVIDIA Modern runs real, complex64, and complex128 on supported GTX 16xx and RTX 20/30/40/50 GPUs.
- A GPU edition safely falls back to CPU if no GPU is present, the driver is unavailable, or the hardware does not match the package.
- `--show-selection` reports whether the current mode selected CPU or GPU.

## 8. How existing software and version conflicts are handled

### 8.1 Old tar, zstd, or other archive tools

Old commands on the user's PATH do not take priority over validated absolute system tools. If the system tools themselves are old or damaged, the installer tries to repair them through the distribution package manager. If repair is not possible, it stops and lists the tools that need administrator attention.

### 8.2 Old Nix

A working Nix version 2.4 or newer is reused. An old Nix, a Nix missing its matching `nix-store`, or a Nix that cannot run causes a clear error. The installer does not silently create a second Nix because inconsistent installations can cause PATH and daemon conflicts.

### 8.3 User-installed CUDA, PyTorch, Conda, or venv

Leave these installed. The launcher clears `PYTHONPATH`, `CUDA_HOME`, Conda/venv, and related overrides before entering the packaged Nix environment. Your environments are neither changed nor removed, and they do not select PyEIDORS dependency versions.

### 8.4 NVIDIA driver

The package uses the existing Linux NVIDIA driver. The system administrator maintains this driver because it is outside the user-level Nix runtime. If `nvidia-smi` fails, fix the driver first; installing a CUDA Toolkit or PyTorch does not replace a GPU driver.

## 9. Troubleshooting

### 9.1 tar, zstd, unzip, curl, or xz is reported unusable

The installer has already performed a real functional probe. Ask the system administrator to install or repair these distribution packages:

```text
tar zstd unzip curl ca-certificates xz
```

Do not overwrite `/usr/bin` with individual binaries from an unknown website.

### 9.2 Nix is reported old or damaged

Check the system Nix:

```bash
/nix/var/nix/profiles/default/bin/nix --version
/nix/var/nix/profiles/default/bin/nix-store --version
```

If either command is missing or fails, or Nix is older than 2.4, ask the administrator to repair or upgrade the existing installation. Do not bypass the check by putting another `nix` at the front of PATH.

### 9.3 Nix was just installed but is not visible in the current terminal

Close the current terminal, open a new terminal, and run the same `.run` file again. Re-running the installer is safe.

### 9.4 Not enough temporary space

Create a temporary directory on a larger Linux filesystem:

```bash
mkdir -p "$HOME/pyeidors-tmp"
TMPDIR="$HOME/pyeidors-tmp" bash PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

Replace the filename when using a GPU edition. Do not use a full `/tmp`, an unreliable network share, or a FAT/exFAT USB disk as `TMPDIR`.

### 9.5 The path contains spaces or special characters

Quote the full path:

```bash
bash "/path/with spaces/PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run"
```

### 9.6 Installation was interrupted

Run the same `.run` file again. Temporary files are cleaned automatically, and the live directory changes only after all doctors pass.

If a final step fails after the switch, the new directory is retained as `.failed-TIMESTAMP`, and the previous working installation is restored automatically.

### 9.7 Reinstallation or changing hardware editions

The previous directory is backed up as:

```text
~/apps/PyEIDORS-@VERSION@.backup-TIMESTAMP-PID
```

Delete old backups manually only after confirming that the new installation works.

### 9.8 GPU is not selected

Run:

```bash
nvidia-smi
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --show-selection
```

If `nvidia-smi` fails, repair the NVIDIA Linux driver first. If it succeeds, check that the GPU model matches the selected installer.

### 9.9 GUI does not open

Make sure the session has a graphical display: a Linux desktop, WSLg, configured X11 forwarding, or remote desktop. A plain SSH command-line session cannot open the GUI without display forwarding.

### 9.10 The system has no sudo or is centrally managed

Ask the system administrator to install the basic archive tools and a compatible Nix in advance. Then run the `.run` file as an ordinary user.

### 9.11 Installation logs

Logs are stored under:

```text
~/.cache/pyeidors-installer/
```

When requesting support, provide the newest `PyEIDORS-@VERSION@-*.log` together with:

- the complete installer filename;
- the Linux distribution and version;
- the output of `uname -m`;
- the CPU and GPU model;
- whether `nvidia-smi` succeeds;
- whether the error occurs during verification, extraction, Nix setup, cache import, a doctor, or GUI startup.

## 10. Upgrade, backup, and uninstall

Running an installer again does not overwrite the current environment immediately. The new environment is validated in a staging directory first. Only then is the old directory backed up and the new one activated.

The user installation is located at:

```text
~/apps/PyEIDORS-@VERSION@
```

Removing this directory removes the launcher, private source snapshot, and associated Nix garbage-collection roots. Unused Nix store data is reclaimed only by a later Nix garbage collection.

Do not remove the entire `/nix` directory just to uninstall PyEIDORS. It may contain other software or other users' environments.

## 11. Download checklist

A typical user needs only:

1. one `.run` file matching the computer's hardware;
2. the installation guide in the preferred language;
3. `PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt`.

You do not need the other two hardware editions, a Nix key, a source zip, or an old manual-install tar archive.
