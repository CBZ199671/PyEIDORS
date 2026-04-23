"""Unit tests for scripts/env/sync_locked_env.sh."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_SOURCE = REPO_ROOT / "scripts" / "env" / "sync_locked_env.sh"


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _build_fake_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    script_path = repo / "scripts" / "env" / "sync_locked_env.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT_SOURCE, script_path)
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    fake_python = """#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "-c" ]; then
  echo "${PY_MM:-3.13}"
  exit 0
fi
if [ "${1:-}" = "-" ]; then
  if [ "${PY_IMPORT_FAIL:-0}" = "1" ]; then
    echo "import check failed" >&2
    exit 1
  fi
  cat >/dev/null || true
  exit 0
fi
exit 0
"""
    _write_executable(repo / ".venv" / "bin" / "python", fake_python)

    fake_uv = """#!/usr/bin/env bash
set -euo pipefail
echo "$@" >> "${UV_LOG:?}"

if [ "${1:-}" = "lock" ] && [ "${2:-}" = "--check" ]; then
  [ "${UV_LOCK_FAIL:-0}" = "1" ] && exit 1
  exit 0
fi

if [ "${1:-}" = "sync" ]; then
  case " $* " in
    *" --check "*) [ "${UV_SYNC_CHECK_FAIL:-0}" = "1" ] && exit 1 ;;
    *) [ "${UV_SYNC_REPAIR_FAIL:-0}" = "1" ] && exit 1 ;;
  esac
  exit 0
fi

exit 0
"""
    _write_executable(repo / "bin" / "uv", fake_uv)

    env = os.environ.copy()
    env["PATH"] = f"{repo / 'bin'}:{env['PATH']}"
    env["PYTHON_BIN"] = str(repo / ".venv" / "bin" / "python")
    env["UV_LOG"] = str(repo / "uv.log")
    return repo, env


def _run(
    repo: Path, env: dict[str, str], mode: str
) -> subprocess.CompletedProcess[str]:
    cmd = [str(repo / "scripts" / "env" / "sync_locked_env.sh"), mode]
    return subprocess.run(
        cmd, cwd=repo, env=env, text=True, capture_output=True, check=False
    )


def test_print_profile_outputs_expected_contract(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    out = _run(repo, env, "--print-profile")
    assert out.returncode == 0
    assert "Profile extras: torch, cuqi, dev" in out.stdout
    assert "Lock freshness gate: uv lock --check" in out.stdout


def test_check_mode_runs_lock_and_sync_with_profile(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    out = _run(repo, env, "--check")
    assert out.returncode == 0

    uv_log = (repo / "uv.log").read_text(encoding="utf-8")
    assert "lock --check" in uv_log
    assert "sync" in uv_log
    assert "--check" in uv_log
    assert "--extra torch" in uv_log
    assert "--extra cuqi" in uv_log
    assert "--extra dev" in uv_log


def test_check_mode_uses_inexact_when_requested(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    env["PYEIDORS_ENV_SYNC_INEXACT"] = "1"

    out = _run(repo, env, "--check")

    assert out.returncode == 0
    uv_log = (repo / "uv.log").read_text(encoding="utf-8")
    assert "--inexact" in uv_log


def test_check_mode_reports_lock_drift(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    env["UV_LOCK_FAIL"] = "1"
    out = _run(repo, env, "--check")
    assert out.returncode != 0
    assert "uv.lock is outdated relative to pyproject metadata" in out.stderr
    assert "uv lock --python" in out.stderr


def test_missing_project_python_emits_bootstrap_hint(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    shutil.rmtree(repo / ".venv")

    out = _run(repo, env, "--check")
    assert out.returncode != 0
    assert "python interpreter not found" in out.stderr
    assert "Official bootstrap command: nix develop" in out.stderr
    assert "docs/NIX_FENICSX.md" in out.stderr


def test_repair_mode_fails_on_import_check_error(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    env["PY_IMPORT_FAIL"] = "1"
    out = _run(repo, env, "--repair")
    assert out.returncode != 0
    assert "import check failed" in out.stderr


def test_invalid_mode_returns_usage_error(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    out = _run(repo, env, "--unknown")
    assert out.returncode == 2
    assert "Usage:" in out.stderr


def test_check_mode_defaults_to_active_profile_venv(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    cuda_python = repo / ".venv-cuda" / "bin" / "python"
    cuda_python.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / ".venv" / "bin" / "python", cuda_python)
    cuda_python.chmod(cuda_python.stat().st_mode | stat.S_IXUSR)

    env.pop("PYTHON_BIN", None)
    env["PYEIDORS_ACTIVE_VENV"] = ".venv-cuda"
    env["VIRTUAL_ENV"] = str(repo / ".venv-cuda")

    out = _run(repo, env, "--check")
    assert out.returncode == 0
    uv_log = (repo / "uv.log").read_text(encoding="utf-8")
    assert "--python .venv-cuda/bin/python" in uv_log
    assert "--active" in uv_log


def test_check_mode_uses_fresh_cache_when_enabled(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    env["PYEIDORS_ENV_SYNC_CACHE"] = "1"

    first = _run(repo, env, "--check")
    assert first.returncode == 0
    assert (repo / "uv.log").exists()

    (repo / "uv.log").unlink()
    second = _run(repo, env, "--check")

    assert second.returncode == 0
    assert "cached locked environment check is fresh" in second.stdout
    assert not (repo / "uv.log").exists()


def test_check_mode_cache_invalidates_when_lock_changes(tmp_path: Path):
    repo, env = _build_fake_repo(tmp_path)
    env["PYEIDORS_ENV_SYNC_CACHE"] = "1"
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    first = _run(repo, env, "--check")
    assert first.returncode == 0

    (repo / "uv.log").unlink()
    (repo / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    second = _run(repo, env, "--check")

    assert second.returncode == 0
    uv_log = (repo / "uv.log").read_text(encoding="utf-8")
    assert "lock --check" in uv_log
    assert "sync" in uv_log
