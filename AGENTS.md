# Local Agent Instructions

## Language
- 输出始终使用中文。

## Execution Policy
- 本仓库位于 WSL2 Ubuntu：`/home/tom/workspace/PyEidors_wsl2`。
- 代码任务默认在 WSL2 Ubuntu 中执行，默认 shell 为 `bash`。
- 任何项目操作都必须先进入仓库根目录：
  `cd /home/tom/workspace/PyEidors_wsl2`。
- Git 操作默认在 WSL2 中执行；除非用户明确要求，否则不要使用 Windows 原生 Git。
- 本仓库本地约定 `git config core.filemode false`，用于避免 Windows/WSL 9P 权限位映射产生大批幻影改动。
- 如需新增真正可执行脚本，提交前显式执行 `git update-index --chmod=+x <file>`，不要依赖 filemode 自动探测。
- 不要回滚用户已有改动；如工作区已有无关修改，保持不动。

## WSL2 Stability
- 从 Windows/Codex 侧进入本仓库时，避免并发启动多条 `wsl.exe -d Ubuntu-22.04 ...` 命令；此前并发启动多条 WSL 命令曾短暂触发 `Wsl/Service/E_UNEXPECTED`，串行执行后恢复正常。
- 涉及 WSL2 的项目命令默认串行执行；需要多步验证时，优先合并到同一个 `bash -lc` 或同一个 `nix develop .#complex64-cuda --command bash -lc "..."` 会话中，而不是并行发起多个 `wsl.exe` 进程。
- 如遇 `Wsl/Service/E_UNEXPECTED`、WSL 命令卡住或 GitNexus MCP/CLI 连接中断，先暂停并诊断 WSL 会话状态，例如 `wsl.exe -l -v`、检查残留 `wsl.exe`/`wslhost.exe`/`vmmemWSL` 进程；不要直接绕过卡死继续做代码判断。温和恢复优先使用串行重试，必要时再说明并执行 `wsl.exe --shutdown`。

## Runtime Routes
- 本项目默认运行环境固定为 **纯 Nix**：开发、测试、脚本、GUI、CLI 验证和分发都优先使用 Nix profile。
- `uv` 只作为显式 opt-in 的本地/遗留环境维护工具，不是默认入口，也不是用户运行时依赖路线。
- 为了计算速度和广泛兼容性，默认开发/测试/脚本/GUI/CLI 验证路线使用 `complex64-cuda` dev shell。
- 除非用户明确指定 CPU、real、complex128、complex64 CPU 或其他路线，否则不要改用 `default`、`cuda`、`complex`、`complex64` 等 shell。
- 默认开发命令模板：

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#complex64-cuda --command <command>
```

- 多条验证命令优先放在同一个 Nix shell 中执行，例如：

```bash
cd /home/tom/workspace/PyEidors_wsl2
nix develop .#complex64-cuda --command bash -lc "python -m pytest tests/unit/test_x.py -q --no-cov"
```

- 不要在未进入 `nix develop .#complex64-cuda` 或用户指定 shell 的情况下直接运行 `python -m pytest`、`python` 或项目脚本；这会绕过 Nix 提供的 Python/系统库运行时，可能造成动态链接错误或依赖漂移。
- 不要把 `uv run` 作为默认入口；它会创建/使用独立 `.venv*`，容易和 Nix profile 分叉。
- 若只检查锁文件或静态文本，可直接用普通 shell 工具；只要涉及 Python 导入、测试、脚本、GUI、FEniCSx/DOLFINx、Torch、CUQI、PySide6、NumPy/SciPy、CUDA/PETSc/SLEPc 等运行时，就必须走 Nix dev shell。
- 如果 `complex64-cuda` shell 因硬件、驱动或 CUDA 环境异常不可用，先报告原因，不要私自切换到 CPU 或其他 shell。

## Environment Stability
- 本项目环境通常已固化；日常开发默认只修改项目代码，不主动改依赖或重建环境。
- 进入既有开发环境运行命令是允许的；但不要主动执行会改变依赖状态的命令，例如 `uv add`、`uv sync --upgrade`、`uv lock`、`nix flake update`。
- 仅在用户明确要求旧 `uv` 环境、依赖缺失诊断确认需要、或设置 `PYEIDORS_ENABLE_UV_SYNC=1` 时，才执行 `uv sync` 或 `scripts/env/sync_locked_env.sh --repair`。
- 不要主动修改 `flake.nix`、`flake.lock`、`pyproject.toml`、`uv.lock`、`.venv*` 或 Nix 配置，除非任务明确涉及依赖/环境/分发配置。
- 如果遇到缺包或环境异常，先诊断并说明是否属于环境问题；需要改环境或引入新包时再执行。

## Python Rules
- 默认运行时依赖事实来源是 `flake.nix` / `flake.lock` 及对应 Nix package closure；`pyproject.toml` 是项目元数据；`uv.lock` 只服务显式 legacy/local uv 路线。
- 日常一次性 Python 命令优先使用：

```bash
nix develop .#complex64-cuda --command python -m <module>
```

- 只有明确维护旧 `uv` 环境时，才使用：

```bash
PYEIDORS_ENABLE_UV_SYNC=1 nix develop .#complex64-cuda --command scripts/env/sync_locked_env.sh --repair
```

- 不要使用全局 `pip install`，除非用户明确要求。
- 不要在 `~/workspace` 根目录创建 `.venv`、缓存、临时文件或构建产物。

## Formatting Rules
- 写完 Python 代码后，必须使用 Ruff 做格式一致性检查。
- Ruff 是维护者开发检查工具，不进入用户 runtime 依赖；当前开发机/Nix shell 的 `PATH` 中可用时执行：

```bash
nix develop .#complex64-cuda --command ruff format --check .
```

- 如只改少量文件，可额外运行：

```bash
nix develop .#complex64-cuda --command ruff check <files>
```

## Test Rules
- 默认测试命令必须走 `complex64-cuda` 的纯 Nix 路线：

```bash
nix develop .#complex64-cuda --command python -m pytest <tests> -q
```

- 项目 `pyproject.toml` 默认启用全仓覆盖率门槛；聚焦跑单个测试文件或局部验证时，优先使用：

```bash
nix develop .#complex64-cuda --command python -m pytest <tests> -q --no-cov
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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **PyEIDORS** (17109 symbols, 34166 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/PyEIDORS/context` | Codebase overview, check index freshness |
| `gitnexus://repo/PyEIDORS/clusters` | All functional areas |
| `gitnexus://repo/PyEIDORS/processes` | All execution flows |
| `gitnexus://repo/PyEIDORS/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
