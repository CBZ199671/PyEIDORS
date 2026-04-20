# Memory

## Me
<!-- 姓名、角色、团队 — 后续补充 -->

## People
<!-- 同事、合作者、联系人 -->
| Who | Role |
|-----|------|

## Terms
<!-- 缩写、专有词汇、内部术语 -->
| Term | Meaning |
|------|---------|

## Projects
<!-- 当前项目、代号 -->
| Name | What |
|------|------|
| **PyEidors** | Python 版 EIDORS，跨平台 EIT 仿真 / 重建工具箱（本仓库） |
| **EIT Workstation** | 新的跨平台 EIT GUI，替代旧 C# 上位机，位于 `src/eit_app/` |

## Preferences
<!-- 协作偏好、沟通风格、工具习惯 -->

## Worktree 约定

本仓库由两路并行的 Claude Code 会话同时使用，按 git worktree + 分支分流以避免冲突：

| Worktree 路径 | 分支 | 归属 |
|---------------|------|------|
| 仓库根目录 (`PyEidors_wsl2/`) | `dev/gui-integration` | 另一路 AI 会话 |
| `.claude/worktrees/gui-polish/` | `dev/gui-polish` | **本会话** |

**会话启动行为（默认）**：在做任何编辑、Bash 改动、或 git 操作之前，先检查当前工作目录。如果当前位于仓库根目录（即在 `dev/gui-integration` 主 worktree），**立即调用** `EnterWorktree` 工具，传 `path=".claude/worktrees/gui-polish"`，把会话切到 `dev/gui-polish`。如果启动时已经在 `.claude/worktrees/gui-polish/` 内，跳过。

**例外**：如果用户在首条消息里明确要求在 `dev/gui-integration` 或其它分支上工作（例如"帮我修一下 gui-integration 上的 X"），不要自动切，按用户指示走。

**禁止**：不要主动 merge / rebase 这两个分支，**禁止 push 到 `main`**。`main` 只在两路 AI 全部开发完毕后由用户做大版本合并，开发期内不要碰。

## 新增 Python 依赖的正确姿势（重要）

GUI 启动器 `scripts/gui/run_eit_app.sh` 为绕开 nix ≥ 2.17 在 worktree 下静默退出的 bug，**强行 `cd` 到主仓库根目录跑 `nix develop`**（见 `run_eit_app.sh` 末尾的 `cd "$NIX_REPO_ROOT"`）。这意味着：

- **运行时 Python 解释器**：来自主仓库 (`gui-integration` worktree) 的 `.venv`
- **加载的源码**：来自当前 worktree（通过 `PYTHONPATH` 注入，见 `run_eit_app_inner.sh`）

所以**只在某个 worktree 里 `uv pip install <pkg>` 是无效的** —— 主仓库 `.venv` 还是没有这个包，GUI 启动后 `import` 立刻崩。

正确流程：
1. 在当前 worktree 改 `pyproject.toml`，把新依赖加到合适的 extras（GUI 相关放 `eit-app`）
2. 在当前 worktree 跑 `uv lock --python .venv/bin/python` 刷新 `uv.lock`
3. `git add pyproject.toml uv.lock && git commit`
4. **同步到另一个 worktree**：在另一个 worktree `git cherry-pick <hash>`
5. 在主仓库（`gui-integration`）跑一次 `nix develop --command bash -c true`，让 env-sync 把新依赖装进主 `.venv`

第 4 步是**绕不开的**：不 cherry-pick 过去，主仓库的 pyproject 没动，env-sync 不会装新包，启动器仍然崩。两路 AI 各自的 worktree 都需要保持 pyproject + uv.lock 同步，否则启动器永远拿主仓库那份。

新用户拿到仓库后只需 `nix develop` 一次，所有 dep 由 lockfile 全自动装好，无需任何手动 pip 操作。
