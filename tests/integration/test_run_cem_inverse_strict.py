"""Integration checks for strict CEM script behavior."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_cem_16e_square_test.py"


def _run(
    args: list[str], *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _require_dolfinx() -> None:
    if "DOLFINX_SKIP_TESTS" in os.environ:
        pytest.skip("dolfinx tests are disabled by DOLFINX_SKIP_TESTS")
    try:
        import dolfinx  # noqa: F401
    except Exception:
        pytest.skip("requires dolfinx runtime")


def test_run_cem_default_runs_inverse():
    _require_dolfinx()
    result = _run([])
    assert result.returncode == 0, result.stderr
    assert "Reconstruction range:" in result.stdout
    assert "Relative error (L2):" in result.stdout


def test_run_cem_skip_inverse_mode():
    _require_dolfinx()
    result = _run(["--skip-inverse"])
    assert result.returncode == 0, result.stderr
    assert "Inverse reconstruction is skipped" in result.stdout


def test_run_cem_strict_failure_exit_nonzero():
    _require_dolfinx()
    env = os.environ.copy()
    env["PYEIDORS_TEST_FORCE_CEM_FAIL"] = "1"
    result = _run([], env=env)
    assert result.returncode != 0
    assert "[ERROR] CEM square test failed:" in result.stderr
