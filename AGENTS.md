# Local Agent Instructions

## Language
- 输出始终使用中文。

## Execution Policy
- 本仓库位于 WSL2 Ubuntu：`/home/tom/workspace/PyEidors_wsl2`。
- 代码任务默认在 WSL2 Ubuntu 中执行，默认 shell 为 `bash`。
- 任何项目操作都必须先进入仓库根目录：
  `cd /home/tom/workspace/PyEidors_wsl2`。

## Project Runtime
- 本项目运行环境固定为 **Nix + uv**。
- Python、测试、格式化、构建、脚本执行、GUI/CLI 验证等项目命令，默认必须通过 `nix develop` 进入项目 dev shell 后再运行。
- 优先使用以下模板：

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop --command bash -lc "uv run <command>"
```

- 多条验证命令优先放在同一个 Nix shell 中执行，例如：

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop --command bash -lc "uv run ruff format --check . && uv run pytest tests/unit/test_x.py -q"
```

- 不要在未进入 `nix develop` 的情况下直接运行 `uv run pytest`、`uv run python`、`uv run ruff` 或项目脚本；这会绕过 Nix 提供的 Python/系统库运行时，可能造成动态链接错误或依赖漂移。
- 若只检查锁文件或静态文本，可直接用普通 shell 工具；只要涉及 Python 导入、测试、脚本、GUI、FEniCSx/DOLFINx、Torch、CUQI、PySide6、NumPy/SciPy 等运行时，就必须走 `nix develop`。

## Python Rules
- Python 项目依赖事实来源优先级：`uv.lock` > `pyproject.toml`。
- 需要完整项目环境或重复执行命令时，优先使用：

```bash
nix develop --command bash -lc "uv sync"
```

- 一次性 Python 命令优先使用：

```bash
nix develop --command bash -lc "uv run <command>"
```

- 不要使用全局 `pip install`，除非用户明确要求。
- 不要在 `~/workspace` 根目录创建 `.venv`、缓存、临时文件或构建产物。

## Formatting Rules
- 写完 Python 代码后，必须使用 Ruff 做格式一致性检查。
- 本项目默认格式检查命令：

```bash
nix develop --command bash -lc "uv run ruff format --check ."
```

- 如只改少量文件，可额外运行：

```bash
nix develop --command bash -lc "uv run ruff check <files>"
```

## Test Rules
- 默认测试命令必须走 Nix + uv：

```bash
nix develop --command bash -lc "uv run pytest <tests> -q"
```

- 项目 `pyproject.toml` 默认启用全仓覆盖率门槛；聚焦跑单个测试文件时，如只是验证局部行为，可使用：

```bash
nix develop --command bash -lc "uv run pytest <tests> -q --no-cov"
```

## Git Rules
- Git 操作默认在 WSL2 中执行。
- 除非用户明确要求，否则不要使用 Windows 原生 Git。
- 不要回滚用户已有改动；如工作区已有无关修改，保持不动。
