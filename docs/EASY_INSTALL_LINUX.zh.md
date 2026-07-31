# PyEIDORS @VERSION@ Linux 一键安装指南

本指南面向所有 Linux 用户，包括第一次使用 Linux、Nix、CUDA 或 PyEIDORS 的用户。正常情况下，您只需选择一个适合硬件的 `.run` 文件并运行它；不需要手动配置 Python、PyTorch、CUDA Toolkit、PETSc、SLEPc、FEniCSx 或 DOLFINx。

PyEIDORS @VERSION@ 当前是论文发表前的私有预览版。请仅供获得作者许可的用户使用，不要上传到公开网盘、公开 GitHub Release 或未经许可二次分发。

## 1. 先选择适合您的安装包

三个安装包只需下载和安装其中一个：

| 安装包 | 适用硬件 | 运行能力 |
|---|---|---|
| `PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run` | 没有 NVIDIA 显卡、只想使用 CPU，或不确定显卡兼容性 | real、complex64、complex128 全部使用 CPU |
| `PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-SM61-LINUX.run` | NVIDIA GTX 10xx 系列，compute capability 6.1 | real、complex64 使用 GPU；complex128 自动回退 CPU；同时保留三种 CPU 环境 |
| `PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-MODERN-LINUX.run` | NVIDIA GTX 16xx、RTX 20/30/40/50 系列 | real、complex64、complex128 使用 GPU；三种模式都可强制使用 CPU |

如果不知道电脑是否有可用的 NVIDIA 显卡，请先运行：

```bash
nvidia-smi
```

- 命令不存在、报错或没有列出显卡：选择 CPU 通用版。
- 显示 GTX 10xx：选择 NVIDIA SM61 版。
- 显示 GTX 16xx 或 RTX 20/30/40/50：选择 NVIDIA 现代版。
- 显卡型号不在表中：优先选择 CPU 通用版，并向软件维护者确认兼容性。

三个版本都会安装到：

```text
~/apps/PyEIDORS-@VERSION@
```

不要把多个版本连续安装到同一目录。需要更换版本时，直接运行新的安装包即可；安装器会先完成新环境验证，再备份旧目录并切换。

## 2. 一键包为您准备了什么

安装包内含已经编译好的 Nix 运行环境和完整依赖闭包。它的目标是让 PyEIDORS 不受电脑上已有开发环境的干扰。

### 2.1 系统压缩工具经过功能验证

安装器优先使用 `/usr/bin`、`/bin` 等系统绝对路径中的 `tar`、`zstd`、`unzip` 和 `curl`，不会仅因为某个同名命令排在 PATH 前面就使用它。

安装前会实际创建、校验并解压一个小型 tar+zstd 文件。因此，用户目录、Conda 或自定义 PATH 中过旧、损坏或伪装的同名工具不能静默接管安装。

如果系统工具确实缺失或不可用，安装器可通过常见发行版的包管理器尝试补齐：

- Ubuntu/Debian：`apt`
- Fedora/RHEL 系：`dnf`
- Arch 系：`pacman`

需要系统权限时，脚本会在对应步骤请求 `sudo`，随后重新做功能验证。验证仍不通过时，安装器会停止并给出明确错误，不会带着损坏的工具继续安装。

### 2.2 现有 Nix 会先验证再复用

安装器按以下顺序检查 Nix：

1. `/nix/var/nix/profiles/default/bin/nix`
2. `~/.nix-profile/bin/nix`
3. 启动安装器时原始 PATH 中的其他候选

候选必须同时满足：

- `nix --version` 可以正常执行；
- Nix 版本不低于 2.4；
- 同一 `bin` 目录中存在可用的 `nix-store`；
- `nix-store --version` 可以正常执行。

缓存导入阶段只使用已经验证过的绝对路径，不会再次从 PATH 搜索 Nix。因此，旧脚本、shell alias 或损坏的 `nix` shim 不会在安装中途替换正确的 Nix。

- 已有 Nix 合格：直接复用，不修改它的版本。
- 完全没有 Nix：使用 Nix 官方安装器；有 systemd 时使用 multi-user 模式，否则使用 single-user 模式。
- 已有 Nix 过旧或损坏：停止并提示修复，不额外安装一套相互冲突的 Nix。

### 2.3 Python、PyTorch 和 CUDA 与主机环境隔离

本包使用的 Python、NumPy、SciPy、PyTorch、CUDA Toolkit、PETSc、SLEPc、FEniCSx、DOLFINx 和 Qt 都来自预编译 Nix 闭包。

通过 `start-pyeidors.sh` 启动时，会清除常见的主机环境覆盖项，包括：

- `PYTHONHOME`、`PYTHONPATH`、`PYTHONSTARTUP`、`PYTHONUSERBASE`
- `VIRTUAL_ENV` 和常见 `CONDA_*` 激活变量
- `CUDA_HOME`、`CUDA_PATH`、`CUDA_ROOT`、`CUDACXX`
- `PETSC_DIR`、`SLEPC_DIR`、`CMAKE_PREFIX_PATH`
- `LD_PRELOAD` 和主机 Qt 插件路径

GPU 环境还会显式使用本包自己的 `CUDA_HOME`、`CUDA_PATH`、`CUDACXX`、`PETSC_DIR` 和 `SLEPC_DIR`。

因此，您不需要卸载电脑上已有的 CUDA、PyTorch、Conda 或 venv，也不需要把它们改成与本包相同的版本。只要通过已安装的启动器运行，这些环境不会覆盖 PyEIDORS。

NVIDIA Linux 驱动是唯一有意保留的主机硬件依赖，因为驱动必须匹配实际显卡和 Linux 内核。安装包不会安装、升级或删除显卡驱动。

### 2.4 本地缓存不需要密钥

内置 Nix 缓存使用 `--no-check-sigs` 从本地导入，因此：

- 不需要 cache key；
- 不需要公钥或私钥文件；
- 不需要配置 `extra-trusted-public-keys`；
- 不需要编辑 `/etc/nix/nix.conf`；
- 不需要为密钥问题重启 `nix-daemon`。

如果您看到要求添加缓存公钥的旧说明，说明正在使用过期安装包或过期文档，请改用当前版本。

## 3. 安装前检查

### 3.1 系统要求

- x86_64（Intel/AMD 64 位）Linux。
- 推荐 Ubuntu 22.04/24.04；Debian、Fedora、Arch 等常见发行版也可使用。
- 推荐原生 Linux；支持 WSL2 Ubuntu，不支持 WSL1。
- 使用普通用户安装，不要把 `sudo` 放在 `.run` 文件前面。
- GPU 版本要求 `nvidia-smi` 能正常显示显卡和驱动。
- GUI 需要 Linux 桌面、WSLg、X11 转发或远程桌面会话。
- 如果电脑完全没有 Nix，自动安装 Nix 时需要联网。

安装过程中若需补齐系统工具或安装 multi-user Nix，脚本会在需要时自行调用 `sudo` 并询问当前 Linux 用户的密码。

### 3.2 磁盘空间

建议至少预留：

| 安装包 | Nix 存储所在分区 | 临时目录所在分区 |
|---|---:|---:|
| CPU 通用版 | 30 GiB | 10 GiB |
| NVIDIA SM61 版 | 50 GiB | 14 GiB |
| NVIDIA 现代版 | 70 GiB | 20 GiB |

临时空间用于校验和解压大型 payload。安装器会在写入大量数据前检查空间。如果 `/tmp` 较小，可按第 9 节设置 `TMPDIR`。

### 3.3 校验下载文件

下载所选 `.run` 文件和：

```text
PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt
```

把它们放在同一目录，然后运行：

```bash
sha256sum -c --ignore-missing PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt
```

所选文件应显示 `OK`。不需要为了校验而下载另外两个大型安装包。

即使跳过手工校验，`.run` 文件也会在解压前验证内置 payload 的 SHA-256；下载不完整或文件损坏时会安全停止。

## 4. 安装：复制一条命令即可

打开终端，进入下载目录。多数浏览器默认下载到 `~/Downloads`：

```bash
cd ~/Downloads
```

然后根据您下载的版本，运行下面三条命令中的一条。

CPU 通用版：

```bash
bash PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

NVIDIA SM61 版：

```bash
bash PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-SM61-LINUX.run
```

NVIDIA 现代版：

```bash
bash PyEIDORS-@VERSION@-EASY-INSTALL-NVIDIA-MODERN-LINUX.run
```

使用 `bash 文件名.run` 不依赖文件的可执行权限，因此从浏览器、网盘、U 盘、Windows 分区或网络盘下载后也能直接使用。

如果文件不在 `~/Downloads`，请进入实际下载目录，或者给出带引号的完整路径：

```bash
bash "/实际路径/PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run"
```

请不要这样运行：

```bash
sudo bash PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

以 root 身份启动会把用户目录、日志和启动器安装给 root，普通用户之后反而无法正常使用。需要管理员权限的个别步骤会由安装器自行请求。

## 5. 安装时会看到什么

安装器依次执行：

1. 检查 Linux、x86_64 和普通用户身份。
2. 从系统绝对路径选择并功能测试压缩/解压工具。
3. 检查临时空间和 Nix store 所在分区空间。
4. 校验 `.run` 内置 payload 的 SHA-256。
5. 解压内置 Nix 二进制缓存和私有源码快照。
6. 验证现有 Nix；完全没有 Nix 时自动安装。
7. 导入完整的预编译 Nix store 闭包。
8. 在 staging 目录构造全新的安装。
9. 清除主机环境覆盖项，并运行 real、complex64、complex128 三套 CPU doctor。
10. 全部验证成功后才切换正式目录。
11. 建立 Nix GC 保护并检查默认 CPU/GPU 路由。
12. 显示安装位置、日志位置和启动命令。

大型缓存的校验、解压和导入需要一些时间。只要终端仍在输出阶段信息，就让它继续运行，不要关闭终端或让电脑休眠。

## 6. 启动 PyEIDORS

默认启动 complex64 模式：

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh
```

选择数值模式：

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --real
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --complex64
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --complex128
```

只查看硬件选择而不打开 GUI：

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --show-selection
```

让 GPU 安装包强制使用 CPU：

```bash
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --cpu
```

请始终使用这个启动器。直接在源码目录运行系统 `python` 会绕过一键包的 Nix 运行环境和冲突隔离。

## 7. 硬件自动选择规则

- CPU 通用版始终使用 CPU，即使电脑上存在 NVIDIA 显卡。
- SM61 版只在兼容的 GTX 10xx / compute capability 6.1 显卡上让 real、complex64 使用 GPU。
- SM61 版的 complex128 始终回退 CPU，不会把精度降低为 complex64。
- 现代版在支持的 GTX 16xx、RTX 20/30/40/50 上让 real、complex64、complex128 使用 GPU。
- GPU 不存在、驱动不可用或安装包与硬件不匹配时，GPU 包会安全回退 CPU。
- `--show-selection` 可显示当前模式最终选择了 CPU 还是 GPU。

## 8. 旧软件和版本冲突会怎样处理

### 8.1 旧 tar、zstd 或其他压缩工具

用户 PATH 中的旧版本不会优先于已验证的系统绝对路径。若系统工具本身过旧或损坏，安装器尝试通过发行版包管理器修复；无法修复时会停止并列出需要管理员处理的工具。

### 8.2 旧 Nix

可用且版本不低于 2.4 的 Nix 会被复用。过旧、缺少配套 `nix-store` 或无法运行的 Nix 会触发明确错误。安装器不会偷偷安装第二套 Nix，因为两套不一致的 Nix 更容易造成路径和 daemon 冲突。

### 8.3 自己安装的 CUDA、PyTorch、Conda 或 venv

这些软件可以保留。启动器会清除 `PYTHONPATH`、`CUDA_HOME`、Conda/venv 等覆盖项，然后进入本包的 Nix 环境。用户自己的环境不会被删除或修改，也不会决定 PyEIDORS 使用的依赖版本。

### 8.4 NVIDIA 驱动

安装包使用主机现有的 Linux NVIDIA 驱动。驱动由系统管理员维护，不属于 Nix 用户运行时。如果 `nvidia-smi` 失败，应先修复驱动；安装 CUDA Toolkit 或 PyTorch 不能替代显卡驱动。

## 9. 常见问题与解决办法

### 9.1 提示 tar、zstd、unzip、curl 或 xz 不可用

安装器已经做过真实功能测试。请让系统管理员通过发行版包管理器安装或修复：

```text
tar zstd unzip curl ca-certificates xz
```

不要从不明网站下载单个二进制文件覆盖 `/usr/bin`。

### 9.2 提示 Nix 过旧或损坏

可检查系统 Nix：

```bash
/nix/var/nix/profiles/default/bin/nix --version
/nix/var/nix/profiles/default/bin/nix-store --version
```

如果命令不存在、报错或 Nix 低于 2.4，请让管理员修复或升级原有 Nix。不要通过在 PATH 前面再放一个 `nix` 来绕过检查。

### 9.3 Nix 刚安装完成，但当前终端仍找不到

关闭当前终端，重新打开一个终端，再运行同一个 `.run` 文件。重复运行安装器是安全的。

### 9.4 临时目录空间不足

在空间较大的 Linux 分区创建临时目录：

```bash
mkdir -p "$HOME/pyeidors-tmp"
TMPDIR="$HOME/pyeidors-tmp" bash PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run
```

使用 GPU 版本时替换文件名。不要把空间不足的 `/tmp`、不稳定的网络盘或 FAT/exFAT U 盘用作 `TMPDIR`。

### 9.5 下载路径包含空格、中文或特殊字符

给完整路径加双引号：

```bash
bash "/路径/带 空格/PyEIDORS-@VERSION@-EASY-INSTALL-CPU-UNIVERSAL-LINUX.run"
```

### 9.6 安装被中断

重新运行同一个 `.run` 文件即可。临时目录会自动清理，正式目录只会在新环境全部 doctor 通过后切换。

如果最终切换后的收尾步骤失败，新目录会保留为 `.failed-日期时间`，安装器会自动恢复之前的可用版本。

### 9.7 重复安装或更换硬件版本

安装器会把旧目录备份为：

```text
~/apps/PyEIDORS-@VERSION@.backup-日期时间-进程号
```

确认新安装正常后，可手动删除旧备份释放空间。

### 9.8 GPU 没有被使用

运行：

```bash
nvidia-smi
~/apps/PyEIDORS-@VERSION@/start-pyeidors.sh --show-selection
```

如果 `nvidia-smi` 失败，先修复 NVIDIA Linux 驱动。如果命令成功，再检查显卡型号与安装包是否匹配。

### 9.9 GUI 无法显示

确认当前会话具有图形显示：Linux 桌面、WSLg、配置好的 X11 转发或远程桌面均可。普通 SSH 命令行会话在未配置显示转发时不能打开 GUI。

### 9.10 系统没有 sudo 或由机构统一管理

请让系统管理员预先安装基础压缩工具和兼容的 Nix。完成后，再由普通用户运行 `.run` 文件。

### 9.11 安装日志在哪里

日志目录：

```text
~/.cache/pyeidors-installer/
```

需要技术支持时，请提供最新的 `PyEIDORS-@VERSION@-*.log`，并附上：

- 安装包的完整文件名；
- Linux 发行版和版本；
- `uname -m` 的输出；
- CPU 和显卡型号；
- `nvidia-smi` 是否成功；
- 错误发生在校验、解压、Nix、缓存导入、doctor 还是 GUI 启动阶段。

## 10. 升级、备份和卸载

重新运行安装包不会直接覆盖当前环境。安装器先在 staging 目录验证新环境，成功后才备份旧目录并切换。

用户安装目录是：

```text
~/apps/PyEIDORS-@VERSION@
```

删除该目录会移除启动器、私有源码快照和对应的 Nix GC root。Nix store 中不再使用的数据只有在以后执行 Nix 垃圾回收时才会释放。

不要为了卸载 PyEIDORS 删除整个 `/nix`，其中可能还有其他软件或其他用户的环境。

## 11. 下载文件清单

一般用户只需要：

1. 与自己硬件匹配的一个 `.run` 文件；
2. 当前语言的安装说明；
3. `PyEIDORS-@VERSION@-EASY-INSTALL.SHA256SUMS.txt`。

不需要同时下载另外两个硬件版本，也不需要 Nix key、源码 zip 或旧式手动安装 tar 包。
