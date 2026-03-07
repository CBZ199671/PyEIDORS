"""Optional heavy regression for 3D strict difference OOM fallback."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "benchmarks" / "benchmark_3d_runtime.py"
RUN_HEAVY = os.environ.get("PYEIDORS_RUN_3D_STRICT_OOM_REGRESSION") == "1"


@pytest.mark.skipif(
    (not GMSH_AVAILABLE) or (not RUN_HEAVY),
    reason="requires gmsh and PYEIDORS_RUN_3D_STRICT_OOM_REGRESSION=1",
)
def test_benchmark_3d_strict_diff_uses_measurement_exact_and_completes(tmp_path: Path):
    report = tmp_path / "strict_diff_report.json"
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--solver-mode",
            "strict",
            "--run-diff",
            "on",
            "--run-absolute",
            "off",
            "--refinement",
            "2",
            "--repeat",
            "1",
            "--perf-report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    diff = dict(payload.get("difference_solver", {}))
    assert diff.get("strict_solver_backend_effective") == "measurement-exact"
    assert diff.get("strict_memory_guard_triggered") is True
    assert diff.get("strict_measurement_system_shape") == [208, 208]
