"""Integration test for the 3D EIDORS alignment comparison suite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "diagnostics" / "compare_3d_eidors_alignment.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_compare_3d_alignment_exports_all_preset_cases(tmp_path: Path):
    output_dir = tmp_path / "alignment_suite"
    run = _run(
        [
            "--output-dir",
            str(output_dir),
            "--refinement",
            "1",
            "--max-iterations",
            "1",
        ]
    )
    assert run.returncode == 0, run.stderr

    summary = json.loads((output_dir / "alignment_summary.json").read_text(encoding="utf-8"))
    case_names = {entry["case"] for entry in summary}
    assert case_names == {
        "difference_eidors_one_step_noser",
        "difference_sphere_multistep_noser",
        "difference_eidors_demo3d_tv",
        "absolute_eidors_abs_gn",
    }
