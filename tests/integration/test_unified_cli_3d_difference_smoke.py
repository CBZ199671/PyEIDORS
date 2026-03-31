"""Integration smoke tests for unified 3D difference CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyeidors.data.structures import PatternConfig
from pyeidors.electrodes.patterns import StimMeasPatternManager
from pyeidors.geometry.mesh3d_generator import GMSH_AVAILABLE

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_reconstruction_unified.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _write_paired_csv(path: Path) -> None:
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
    )
    n_meas = StimMeasPatternManager(pattern).n_meas_total
    data = np.zeros((n_meas, 4), dtype=float)
    data[:, 0] = 1.0
    data[:, 1] = 0.0
    data[:, 2] = 1.001
    data[:, 3] = 0.0
    np.savetxt(path, data, delimiter=",")


@pytest.mark.skipif(not GMSH_AVAILABLE, reason="gmsh python bindings not available")
def test_unified_cli_3d_difference_dry_run_and_small_execution(tmp_path: Path):
    paired = tmp_path / "paired.csv"
    _write_paired_csv(paired)

    dry = _run(
        [
            "--method",
            "gn-difference",
            "--csv",
            str(paired),
            "--output-root",
            str(tmp_path / "out_dry"),
            "--mesh-dim",
            "3",
            "--dry-run",
        ]
    )
    assert dry.returncode == 0, dry.stderr

    run = _run(
        [
            "--method",
            "gn-difference",
            "--csv",
            str(paired),
            "--output-root",
            str(tmp_path / "out_run"),
            "--mesh-dir",
            str(tmp_path / "mesh_cache"),
            "--mesh-dim",
            "3",
            "--radius",
            "0.1",
            "--mesh-height",
            "0.08",
            "--refinement",
            "1",
            "--no-plots",
        ]
    )
    assert run.returncode == 0, run.stderr
    summary = tmp_path / "out_run" / "gn-difference" / "batch_summary.json"
    assert summary.exists()
