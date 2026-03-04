"""Smoke contract for locked environment reproducibility in Nix shell."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_env_repro_contract_guard():
    if not (REPO_ROOT / ".venv" / "bin" / "python").exists():
        pytest.skip("requires nix develop to create project .venv")

    check = _run(["scripts/env/sync_locked_env.sh", "--check"])
    assert check.returncode == 0, check.stderr

    verify = _run([sys.executable, "scripts/env/verify_env_manifest.py"])
    assert verify.returncode == 0, verify.stderr

    imports = _run([sys.executable, "-c", "import dolfinx, torch, cuqi, pyeidors"])
    assert imports.returncode == 0, imports.stderr
