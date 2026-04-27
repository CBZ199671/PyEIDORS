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

## 当前开发分支

`dev/gui-integration` 是当前主开发分支，所有改动直接在仓库根目录 (`PyEidors_wsl2/`) 进行。

**禁止**：未经用户允许不要主动 merge / rebase 到其他分支，**禁止 push 到 `main`**。`main` 只在用户做正式版本合并时使用，开发期内不要碰。

## 新增 Python 依赖的正确姿势

GUI 启动器 `scripts/gui/run_eit_app.sh` 为绕开 nix ≥ 2.17 在 worktree 下静默退出的 bug，**强行 `cd` 到仓库根目录跑 `nix develop`**（见 `run_eit_app.sh` 末尾的 `cd "$NIX_REPO_ROOT"`）。所以运行时 Python 解释器来自仓库根目录的 `.venv`。

只在临时目录里 `uv pip install <pkg>` 是无效的 —— 必须走 lockfile：

1. 改 `pyproject.toml`，把新依赖加到合适的 extras（GUI 相关放 `eit-app`）
2. `uv lock --python .venv/bin/python` 刷新 `uv.lock`
3. `git add pyproject.toml uv.lock && git commit`
4. 跑一次 `nix develop --command bash -c true`，让 env-sync 把新依赖装进 `.venv`

新用户拿到仓库后只需 `nix develop` 一次，所有 dep 由 lockfile 全自动装好，无需任何手动 pip 操作。
