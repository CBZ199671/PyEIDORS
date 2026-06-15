"""Smoke contract for locked environment reproducibility in Nix shell."""

from __future__ import annotations

import os
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
    if os.environ.get("PYEIDORS_ACTIVE_ENV") != "nix":
        pytest.skip("requires nix develop pure-Nix profile")
    if not os.environ.get("PYEIDORS_ENV_PROFILE"):
        pytest.skip("requires PYEIDORS_ENV_PROFILE from nix develop")

    verify = _run([sys.executable, "scripts/env/verify_env_manifest.py"])
    assert verify.returncode == 0, verify.stderr

    imports = _run(
        [
            sys.executable,
            "-c",
            "import dolfinx, torch, cuqi, pyeidors, pyqtgraph; "
            "from PySide6.QtCore import Qt",
        ]
    )
    assert imports.returncode == 0, imports.stderr
