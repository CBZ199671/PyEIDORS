# PyEIDORS 2.0.0 纯 Nix 用户安装指南

本文给 PyEIDORS 2.0.0 的最终使用者阅读。你收到的通常是一个私有压缩包，而不是公开 GitHub 仓库：

```text
PyEIDORS-2.0.0-pure-nix-source.zip
PyEIDORS-2.0.0-pure-nix-source.SHA256SUMS.txt
```

这套分发方式的目标是：用户只需要安装 Nix，就能运行 PyEIDORS 的 GUI 和核心计算环境；不需要自己安装 FEniCSx、DOLFINx、PETSc、MPI、Gmsh、Qt、PyTorch、CUQI 或 CUDA Toolkit，也不需要执行 `uv sync`。

如果发布者同时提供快速安装二进制缓存包：

```text
PyEIDORS-2.0.0-fast-install-x86_64-linux.tar.zst
PyEIDORS-2.0.0-fast-install-x86_64-linux.SHA256SUMS.txt
```

请优先使用这个快速安装包。它包含 source zip、发布者已经编译好的 Nix binary cache，以及用于校验这个 cache 的公开签名 key，能避免用户第一次运行时长时间编译 CUDA/PETSc/PyTorch/VTK 等大依赖。

## 先选运行路线

推荐先按有没有 NVIDIA GPU 选择入口：

| 使用场景 | 推荐启动命令 | 说明 |
|---|---|---|
| 没有 NVIDIA GPU，或暂时只想稳定运行 | `nix run .#eit-app-complex64` | 推荐 CPU 通用入口。可以处理复值问题；如果任务输出是纯实值，GUI 后端会转到实值 worker 以节省内存。 |
| 有 NVIDIA GPU，并且 `nvidia-smi` 可用 | `nix run .#eit-app-complex64-cuda` | 推荐 GPU 通用入口。可以处理复值 CUDA 路线；纯实值任务会尽量走实值 worker。 |
| 只做实值 CPU | `nix run .#eit-app-real-cpu` | 运行闭包更小，但不适合复导纳问题。 |
| 只做实值 GPU | `nix run .#eit-app-real-gpu` | 需要可用 NVIDIA 驱动。 |
| 需要更高精度复值 CPU | `nix run .#eit-app-complex128-cpu` | complex128，内存占用更高。 |
| 需要更高精度复值 GPU | `nix run .#eit-app-complex128-gpu` | complex128 CUDA，显存和内存占用更高。 |

如果不确定，普通 CPU 用户用：

```bash
nix run .#eit-app-complex64
```

普通 GPU 用户用：

```bash
nix run .#eit-app-complex64-cuda
```

## 支持的平台

当前推荐平台：

| 平台 | CPU 版本 | GPU/CUDA 版本 | 备注 |
|---|---:|---:|---|
| Windows 11 + WSL2 Ubuntu | 支持 | 支持 | Windows 用户推荐路线。GUI 依赖 WSLg。 |
| Linux 原生 x86_64 | 支持 | 支持 | 推荐 Ubuntu/Debian/Fedora/Arch 等常见发行版。 |
| Windows 原生 `.exe` | 不作为当前主路线 | 不作为当前主路线 | 当前纯 Nix 包是 Linux 运行环境；Windows 用户建议走 WSL2。 |
| macOS | 暂不推荐 | 不支持 CUDA | 当前项目依赖的数值栈和 CUDA 路线主要面向 Linux。 |

GPU 版本要求：

- NVIDIA 显卡。
- 系统里 `nvidia-smi` 可以正常运行。
- WSL2 用户需要 Windows 侧安装支持 WSL 的 NVIDIA 驱动。
- 不要求用户在 Linux/WSL 里手动安装 CUDA Toolkit；Nix 包会提供项目需要的 CUDA 用户态运行库，但宿主机驱动仍然必须存在。

## 第一步：安装 Nix

官方参考：

- Nix 安装说明：https://nix.dev/manual/nix/stable/installation/
- Nix 入门页：https://nix.dev/install-nix

Nix 官方文档当前推荐 Linux 优先使用 multi-user 安装；没有 systemd 的环境可以使用 single-user 安装。

### WSL2 Ubuntu 用户

在 Windows PowerShell 中安装 WSL2 Ubuntu：

```powershell
wsl --install -d Ubuntu-22.04
```

安装完成后重启 Windows 或重新打开终端，然后进入 Ubuntu：

```powershell
wsl -d Ubuntu-22.04
```

在 Ubuntu 中安装基础工具：

```bash
sudo apt update
sudo apt install -y curl ca-certificates xz-utils unzip
```

优先安装 multi-user Nix：

```bash
curl -L https://nixos.org/nix/install | sh -s -- --daemon
```

如果安装器提示当前 WSL 环境没有 systemd，改用 single-user Nix：

```bash
curl -L https://nixos.org/nix/install | sh -s -- --no-daemon
```

安装完成后关闭当前 Ubuntu 终端，重新打开，再检查：

```bash
nix --version
```

启用 flakes：

```bash
mkdir -p ~/.config/nix
grep -q "experimental-features" ~/.config/nix/nix.conf 2>/dev/null || \
  printf "experimental-features = nix-command flakes\n" >> ~/.config/nix/nix.conf
```

检查 GUI 显示环境：

```bash
echo "$DISPLAY"
echo "$WAYLAND_DISPLAY"
```

至少有一个变量不为空时，WSLg 通常可以显示 GUI。如果两个都为空，先检查 Windows 是否启用了 WSLg，或者改用带桌面的原生 Linux。

### 原生 Linux 用户

Debian/Ubuntu：

```bash
sudo apt update
sudo apt install -y curl ca-certificates xz-utils unzip
```

Fedora：

```bash
sudo dnf install -y curl ca-certificates xz unzip
```

Arch Linux：

```bash
sudo pacman -S --needed curl ca-certificates xz unzip
```

优先安装 multi-user Nix：

```bash
curl -L https://nixos.org/nix/install | sh -s -- --daemon
```

如果系统没有 systemd，改用 single-user Nix：

```bash
curl -L https://nixos.org/nix/install | sh -s -- --no-daemon
```

重新打开终端后检查：

```bash
nix --version
```

启用 flakes：

```bash
mkdir -p ~/.config/nix
grep -q "experimental-features" ~/.config/nix/nix.conf 2>/dev/null || \
  printf "experimental-features = nix-command flakes\n" >> ~/.config/nix/nix.conf
```

## 第二步：校验并解压 PyEIDORS

### 使用快速安装包

如果你收到了 `PyEIDORS-2.0.0-fast-install-x86_64-linux.tar.zst`，优先按本节操作。

先校验：

```bash
cd ~/Downloads
sha256sum -c PyEIDORS-2.0.0-fast-install-x86_64-linux.SHA256SUMS.txt
```

解压：

```bash
mkdir -p ~/apps
cd ~/apps
tar -I zstd -xf ~/Downloads/PyEIDORS-2.0.0-fast-install-x86_64-linux.tar.zst
cd PyEIDORS-2.0.0-fast-install-x86_64-linux
```

导入本地 Nix binary cache：

```bash
bash install-from-local-cache.sh
```

这一步会把发布者已经编译好的 Nix store path 导入到当前机器的 `/nix/store`。如果你的 Nix daemon 不允许普通用户临时信任新的 binary cache key，脚本会提示把 `binary-cache-public-key.txt` 中的公钥加入 Nix 配置。常见 multi-user Nix 需要管理员在 `/etc/nix/nix.conf` 中加入：

```bash
extra-trusted-public-keys = <binary-cache-public-key.txt 中的完整一行>
```

然后重启 `nix-daemon`，再重新运行：

```bash
bash install-from-local-cache.sh
```

导入完成后，再解压源码包：

```bash
cd ~/apps
unzip ~/apps/PyEIDORS-2.0.0-fast-install-x86_64-linux/PyEIDORS-2.0.0-pure-nix-source.zip
cd PyEIDORS-2.0.0
```

然后进入“第三步：启动 GUI”。

### 只使用 source zip

假设发布者把两个文件放在 `~/Downloads`：

```bash
cd ~/Downloads
sha256sum -c PyEIDORS-2.0.0-pure-nix-source.SHA256SUMS.txt
```

看到 `OK` 后再解压：

```bash
mkdir -p ~/apps
cd ~/apps
unzip ~/Downloads/PyEIDORS-2.0.0-pure-nix-source.zip
cd PyEIDORS-2.0.0
```

WSL2 用户建议把项目解压到 Linux 文件系统，例如 `~/apps/PyEIDORS-2.0.0`。不要放在 `/mnt/c/...` 下长期运行，否则文件访问会慢，也更容易遇到权限和 GUI 缓存问题。

## 第三步：启动 GUI

CPU 通用入口：

```bash
cd ~/apps/PyEIDORS-2.0.0
nix run .#eit-app-complex64
```

GPU 通用入口：

```bash
cd ~/apps/PyEIDORS-2.0.0
nix run .#eit-app-complex64-cuda
```

第一次运行可能会下载或构建较大的 Nix 依赖。CPU 路线通常比 GPU 路线轻；GPU 路线会包含 CUDA、PETSc/DOLFINx、PyTorch/VTK 等大闭包。如果发布者提供了私有 Nix binary cache，首次运行会快很多。

后续再次运行会复用 `/nix/store`，通常会明显变快。

## 可选：预热缓存

预热不是必须的，但可以提前构建后端 worker、网格和运行时缓存，减少第一次在 GUI 里点击计算时的等待。

CPU 通用预热：

```bash
cd ~/apps/PyEIDORS-2.0.0
nix run .#eit-cache-complex64 -- warm --profile complex64
```

GPU 通用预热：

```bash
cd ~/apps/PyEIDORS-2.0.0
nix run .#eit-cache-complex64-cuda -- warm --profile complex64-cuda
```

查看缓存状态：

```bash
nix run .#eit-cache-complex64 -- status --include-worker-cache
```

运行时缓存默认写入用户目录，不会写回源码目录：

```text
${XDG_CACHE_HOME:-$HOME/.cache}/pyeidors
${XDG_DATA_HOME:-$HOME/.local/share}/pyeidors
```

## 可选：安装成系统命令

如果希望以后直接输入 `eit-app`，可以安装到当前用户的 Nix profile。

CPU 通用版本：

```bash
cd ~/apps/PyEIDORS-2.0.0
nix profile install .#pyeidors-complex64
eit-app
```

GPU 通用版本：

```bash
cd ~/apps/PyEIDORS-2.0.0
nix profile install .#pyeidors-complex64-cuda
eit-app
```

注意：不同 package 都提供同名命令 `eit-app` 和 `eit-cache`。如果同时安装多个 PyEIDORS profile，命令名可能冲突。多数用户建议直接使用 `nix run .#入口名`，不要同时安装多个 profile。

卸载当前 profile 中的 PyEIDORS：

```bash
nix profile list
nix profile remove <编号或包名>
```

## 完整入口表

| 类型 | package | GUI app | cache app | profile 名 | PETSc 标量 |
|---|---|---|---|---|---|
| 实值 CPU | `.#pyeidors` | `.#eit-app-real-cpu` | `.#eit-cache-real-cpu` | `default` | `float64` |
| 复值 CPU | `.#pyeidors-complex64` | `.#eit-app-complex64-cpu` | `.#eit-cache-complex64-cpu` | `complex64` | `complex64` |
| 复值 CPU | `.#pyeidors-complex` | `.#eit-app-complex128-cpu` | `.#eit-cache-complex128-cpu` | `complex` | `complex128` |
| 实值 GPU/CUDA | `.#pyeidors-cuda` | `.#eit-app-real-gpu` | `.#eit-cache-real-gpu` | `cuda` | `float64` |
| 复值 GPU/CUDA | `.#pyeidors-complex64-cuda` | `.#eit-app-complex64-gpu` | `.#eit-cache-complex64-gpu` | `complex64-cuda` | `complex64` |
| 复值 GPU/CUDA | `.#pyeidors-complex-cuda` | `.#eit-app-complex128-gpu` | `.#eit-cache-complex128-gpu` | `complex-cuda` | `complex128` |

兼容入口也可用：

```text
.#eit-app
.#eit-app-default
.#eit-app-complex64
.#eit-app-complex
.#eit-app-cuda
.#eit-app-complex64-cuda
.#eit-app-complex-cuda
```

其中 `.#eit-app` 和 `.#eit-app-default` 是实值 CPU 入口。最终用户如果不确定任务是否包含复导纳，优先用 `.#eit-app-complex64` 或 `.#eit-app-complex64-cuda`。

## GPU 用户检查清单

原生 Linux：

```bash
nvidia-smi
```

WSL2：

```bash
nvidia-smi
```

如果 WSL2 里找不到 `nvidia-smi`，再试：

```bash
/usr/lib/wsl/lib/nvidia-smi
```

如果仍然不可用，请先更新 Windows 侧 NVIDIA 驱动，并确认使用的是 WSL2 而不是 WSL1：

```powershell
wsl -l -v
```

GPU 版本不是按显卡型号固定打包的。它依赖 NVIDIA 驱动与 CUDA 用户态库的兼容性。通常只要驱动足够新、`nvidia-smi` 在 Linux/WSL 中可见，就可以运行对应 CUDA 路线。不同显卡的性能、显存容量和可承受的网格规模会不同。

## 常见问题

### 提示 `experimental Nix feature 'nix-command' is disabled`

说明 flakes 没启用。执行：

```bash
mkdir -p ~/.config/nix
printf "experimental-features = nix-command flakes\n" >> ~/.config/nix/nix.conf
```

然后重新打开终端。

### 提示找不到 `nix`

重新打开终端后再试：

```bash
nix --version
```

如果仍然找不到，检查 Nix 安装是否完成。single-user 安装有时需要手动加载 profile：

```bash
. "$HOME/.nix-profile/etc/profile.d/nix.sh"
```

### WSL2 能启动命令但看不到 GUI

先检查：

```bash
echo "$DISPLAY"
echo "$WAYLAND_DISPLAY"
```

如果都为空，通常是 WSLg 不可用。建议升级 WSL、使用 Windows 11，或在原生 Linux 桌面环境中运行。

### GPU 入口失败，但 CPU 入口能运行

先确认：

```bash
nvidia-smi
```

如果 `nvidia-smi` 不可用，GPU 入口无法正常工作。可以先使用 CPU 通用入口：

```bash
nix run .#eit-app-complex64
```

### 首次运行特别慢

这是 Nix 首次下载或构建运行闭包。GPU 版本尤其大。后续会复用 `/nix/store`。如果发布者提供 binary cache，请按发布者给出的 cache 配置启用。

### 磁盘占用变大

Nix 会保留构建结果和依赖闭包。清理未使用 store 路径：

```bash
nix store gc
```

如果安装到了 profile，先卸载不需要的 profile 项：

```bash
nix profile list
nix profile remove <编号或包名>
nix store gc
```

### 不要使用 `sudo nix run`

普通运行不需要 `sudo`。用 `sudo` 反而可能让 GUI、缓存目录和用户权限变乱。

### 不要执行 `uv sync`

最终用户的纯 Nix 运行路线不需要 `uv`。`uv` 是开发和发布验证工具，不是最终用户安装步骤。

## 给发布者：如何重新打包

在发布者自己的 WSL2 仓库根目录运行：

```bash
cd /home/tom/workspace/PyEidors_wsl2
scripts/release/build_private_distribution.sh
```

快速只重建私有 source zip，跳过较慢验证：

```bash
RUN_TESTS=0 RUN_WARM=0 RUN_CUDA_BUILD=0 scripts/release/build_private_distribution.sh
```

如果要在发布机器上实际构建 CUDA 三个 package：

```bash
RUN_CUDA_BUILD=1 scripts/release/build_private_distribution.sh
```

正式分发前建议至少确认：

```bash
nix flake check --option warn-dirty false --no-build
nix build --no-link --print-out-paths .#pyeidors .#pyeidors-complex .#pyeidors-complex64
nix build --max-jobs 1 --no-link --print-out-paths \
  .#pyeidors-cuda \
  .#pyeidors-complex-cuda \
  .#pyeidors-complex64-cuda
```

发布脚本会把本文件复制成压缩包根目录的 `INSTALL.zh.md`，所以用户解压后直接读 `INSTALL.zh.md` 即可。

## 给发布者：如何打包二进制缓存

如果希望用户避免首次长时间编译，除了 source zip，还应提供快速安装二进制缓存包：

```bash
cd /home/tom/workspace/PyEidors_wsl2
scripts/release/build_binary_cache_bundle.sh
```

输出文件：

```text
dist/PyEIDORS-2.0.0-fast-install-x86_64-linux.tar.zst
dist/PyEIDORS-2.0.0-fast-install-x86_64-linux.SHA256SUMS.txt
```

这个包内包含：

```text
nix-cache/
install-from-local-cache.sh
binary-cache-public-key.txt
README_FAST_INSTALL.zh.md
manifest.json
top-level-store-paths.txt
closure-store-paths.txt
PyEIDORS-2.0.0-pure-nix-source.zip
PyEIDORS-2.0.0-pure-nix-source.SHA256SUMS.txt
```

默认会确认并导出六个 package：

```text
.#pyeidors
.#pyeidors-complex
.#pyeidors-complex64
.#pyeidors-cuda
.#pyeidors-complex-cuda
.#pyeidors-complex64-cuda
```

脚本会自动为本地 binary cache 生成签名 key：

```text
dist/binary-cache-keys/pyeidors-2.0.0-x86_64-linux.sec
dist/binary-cache-keys/pyeidors-2.0.0-x86_64-linux.pub
```

`.pub` 会复制进快速安装包；`.sec` 是发布者私钥，绝对不要发给用户，也不要提交到 Git。后续重新打包同一版本时保留这对 key，可以让用户继续信任同一个发布者 key。

如果只想重新生成 source zip 并一起打二进制缓存：

```bash
BUILD_SOURCE_ZIP=1 scripts/release/build_binary_cache_bundle.sh
```

如果确信六个 package 都已经构建过，只想导出现有闭包：

```bash
BUILD_PACKAGES=0 scripts/release/build_binary_cache_bundle.sh
```

快速安装包是 `x86_64-linux` 专用。WSL2 Ubuntu 和原生 Linux x86_64 都可以使用；macOS、ARM Linux 或 Windows 原生不能使用这个二进制缓存包。

## 给发布者：分发包应包含什么

纯 Nix source package 至少需要：

```text
flake.nix
flake.lock
pyproject.toml
README.md
LICENSE
INSTALL.zh.md
docs/PRIVATE_RUNTIME_INSTALL.md
src/pyeidors/
src/eit_app/
```

不要手动挑单个 Python 文件。GUI、worker、缓存、正问题、逆问题、仿真页面和运行时 profile 之间有大量导入关系，直接提供完整的 `src/pyeidors/` 和 `src/eit_app/` 最稳。

不要放入最终用户分发包：

```text
.venv/
.venv-*/
dist/
data/
results/
outputs/
eit_meshes/
.pyeidors_cache/
tests/
notes/
reports/
archived/
compare_with_Eidors/
SoftwareX-PyEidors-Paper/
temp_abs_result/
Software_patent/
scripts/benchmarks/
scripts/diagnostics/
scripts/demos/
docs/benchmarks/
docs/screenshots/
```

## 给发布者：关于私有 Git 和 binary cache

如果用户有私有仓库访问权限，也可以通过私有 Git 运行：

```bash
nix run git+ssh://git@github.com/你的账号/你的私有仓库.git?ref=v2.0.0#eit-app-complex64
```

GPU 用户：

```bash
nix run git+ssh://git@github.com/你的账号/你的私有仓库.git?ref=v2.0.0#eit-app-complex64-cuda
```

注意：私有 Git 仍然会把源码交给有权限的用户，只是不公开到公网。

如果希望用户首次运行更快，推荐同时提供私有 Nix binary cache，例如 Cachix、私有 S3/HTTP cache 或内网 Nix cache。发布者预先构建 CPU/GPU 六个 package 并推送闭包后，用户运行 `nix run` 会优先下载预构建结果，而不是在本机长时间编译。

## 收集报错信息

如果需要向发布者反馈问题，请提供：

```bash
uname -a
nix --version
nix flake show
```

GPU 用户再提供：

```bash
nvidia-smi
```

同时提供实际执行的命令，以及终端最后 50 行日志。WSL2 用户请注明 Windows 版本、WSL 发行版和 `wsl -l -v` 输出。
