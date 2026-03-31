"""Integration tests for the 3D CEM cylinder smoke script."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_cem_16e_cylinder_3d_test.py"


def _run(args: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = dict(os.environ)
    merged_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    merged_env.setdefault("OMP_NUM_THREADS", "1")
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=merged_env,
    )


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_run_cem_3d_cylinder_default_and_skip_and_strict_failure():
    ok = _run([])
    assert ok.returncode == 0, ok.stderr

    skip = _run(["--skip-inverse"])
    assert skip.returncode == 0, skip.stderr

    fail = _run([], env={"PYEIDORS_TEST_FORCE_CEM_FAIL": "1"})
    assert fail.returncode != 0
    combined = f"{fail.stdout}\n{fail.stderr}"
    assert "Forced 3D CEM failure" in combined
