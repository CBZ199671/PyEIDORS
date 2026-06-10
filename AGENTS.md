# Local Agent Instructions

## Language
- 输出始终使用中文。

## Execution Policy
- 本仓库位于 WSL2 Ubuntu：`/home/tom/workspace/PyEidors_wsl2`。
- 代码任务默认在 WSL2 Ubuntu 中执行，默认 shell 为 `bash`。
- 任何项目操作都必须先进入仓库根目录：
  `cd /home/tom/workspace/PyEidors_wsl2`。
- Git 操作默认在 WSL2 中执行；除非用户明确要求，否则不要使用 Windows 原生 Git。
- 不要回滚用户已有改动；如工作区已有无关修改，保持不动。

## Runtime Routes
- 本项目运行环境固定为 **Nix + uv** 开发路线，以及纯 Nix 分发路线。
- 为了计算速度和广泛兼容性，默认开发/测试/脚本/GUI/CLI 验证路线使用 `complex64-cuda` dev shell。
- 除非用户明确指定 CPU、real、complex128、complex64 CPU 或其他路线，否则不要改用 `default`、`cuda`、`complex`、`complex64` 等 shell。
- 默认开发命令模板：

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#complex64-cuda --command bash -lc "uv run <command>"
```

- 多条验证命令优先放在同一个 Nix shell 中执行，例如：

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#complex64-cuda --command bash -lc "uv run ruff format --check . && uv run pytest tests/unit/test_x.py -q --no-cov"
```

- 不要在未进入 `nix develop .#complex64-cuda` 或用户指定 shell 的情况下直接运行 `uv run pytest`、`uv run python`、`uv run ruff` 或项目脚本；这会绕过 Nix 提供的 Python/系统库运行时，可能造成动态链接错误或依赖漂移。
- 若只检查锁文件或静态文本，可直接用普通 shell 工具；只要涉及 Python 导入、测试、脚本、GUI、FEniCSx/DOLFINx、Torch、CUQI、PySide6、NumPy/SciPy、CUDA/PETSc/SLEPc 等运行时，就必须走 Nix dev shell。
- 如果 `complex64-cuda` shell 因硬件、驱动或 CUDA 环境异常不可用，先报告原因，不要私自切换到 CPU 或其他 shell。

## Environment Stability
- 本项目环境通常已固化；日常开发默认只修改项目代码，不主动改依赖或重建环境。
- 进入既有开发环境运行命令是允许的；但不要主动执行会改变依赖状态的命令，例如 `uv add`、`uv sync --upgrade`、`uv lock`、`nix flake update`。
- 仅在首次装配、依赖缺失、锁文件/环境文件刚被用户或任务修改，或用户明确要求时，才执行 `uv sync` 或环境修复命令。
- 不要主动修改 `flake.nix`、`flake.lock`、`pyproject.toml`、`uv.lock`、`.venv*` 或 Nix 配置，除非任务明确涉及依赖/环境/分发配置。
- 如果遇到缺包或环境异常，先诊断并说明是否属于环境问题；需要改环境或引入新包时再执行。

## Python Rules
- Python 项目依赖事实来源优先级：`uv.lock` > `pyproject.toml`。
- 日常一次性 Python 命令优先使用：

```bash
nix develop .#complex64-cuda --command bash -lc "uv run <command>"
```

- 需要首次装配或确需补齐依赖时，才使用：

```bash
nix develop .#complex64-cuda --command bash -lc "uv sync"
```

- 不要使用全局 `pip install`，除非用户明确要求。
- 不要在 `~/workspace` 根目录创建 `.venv`、缓存、临时文件或构建产物。

## Formatting Rules
- 写完 Python 代码后，必须使用 Ruff 做格式一致性检查。
- 本项目默认格式检查命令：

```bash
nix develop .#complex64-cuda --command bash -lc "uv run ruff format --check ."
```

- 如只改少量文件，可额外运行：

```bash
nix develop .#complex64-cuda --command bash -lc "uv run ruff check <files>"
```

## Test Rules
- 默认测试命令必须走 `complex64-cuda` 的 Nix + uv 路线：

```bash
nix develop .#complex64-cuda --command bash -lc "uv run pytest <tests> -q"
```

- 项目 `pyproject.toml` 默认启用全仓覆盖率门槛；聚焦跑单个测试文件或局部验证时，优先使用：

```bash
nix develop .#complex64-cuda --command bash -lc "uv run pytest <tests> -q --no-cov"
```

## Distribution Rules
- 分发、复现、打包验证默认走纯 Nix，不额外执行 `uv sync` 或创建/更新 `.venv*`。
- 默认纯 Nix 运行/验收应用时，优先使用 complex64 CUDA app，例如 `nix run .#eit-app-complex64-cuda`；该命令会按当前 flake 自动构建所需包后再运行。
- 默认纯 Nix 构建/打包/产物检查时，使用 complex64 CUDA package，例如 `nix build .#pyeidors-complex64-cuda`；该命令只构建并在 `result` 链接暴露产物，不直接运行 GUI/CLI。
- 只有用户明确指定 CPU、real、complex128 或其他精度/后端时，才切换对应 Nix package/app。

## WSL2 Delivery Links
- 在 Windows 端 Codex App 中交付本仓库 WSL2 原生文件或目录时，必须使用可点击的 `file://wsl.localhost/Ubuntu-22.04/<path>` URI 格式，而不是裸 UNC 路径或 POSIX 路径。
- 示例：WSL2 路径 `/home/tom/workspace/PyEidors_wsl2/output/doc/report.docx` 应交付为 `[report.docx](file://wsl.localhost/Ubuntu-22.04/home/tom/workspace/PyEidors_wsl2/output/doc/report.docx)`。
- 对 Word、PDF、图片、压缩包、分发包、诊断输出目录等交付产物，优先使用上述 `file://wsl.localhost/...` Markdown 链接格式；必要时可同时附上 POSIX 路径用于终端复制。
- 不要为了可点击而自动复制到 Windows 侧目录，也不要自动创建 `.lnk` 快捷方式，除非用户明确要求。
- 如果路径包含空格、中文或特殊字符，交付链接前必须做 URI 编码；推荐流程：先用 `wslpath -w <path>` 得到 UNC，再用 PowerShell `[uri]$unc` 生成 `file://wsl.localhost/...`。

## Search Scope
- 代码搜索默认避开 `.venv*`、`.pyeidors_cache`、`.pytest_cache`、`.ruff_cache`、`htmlcov`、`build`、`dist`、`results`、`output`、`outputs`、`eit_meshes`、`tmp`。
- 只有任务明确涉及生成结果、缓存、报告、网格或临时产物时，才读取这些目录。
