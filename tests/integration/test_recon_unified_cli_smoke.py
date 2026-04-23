"""Lightweight smoke tests for unified reconstruction CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_reconstruction_unified.py"


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _write_paired_csv(path: Path) -> None:
    rows = np.array(
        [
            [1.0, 0.1, 1.2, 0.2],
            [1.1, 0.1, 1.3, 0.2],
            [1.2, 0.1, 1.4, 0.2],
        ]
    )
    np.savetxt(path, rows, delimiter=",")


def test_unified_cli_dry_run_all_methods(tmp_path: Path):
    paired = tmp_path / "paired.csv"
    ref = tmp_path / "ref.csv"
    target = tmp_path / "target.csv"
    metadata = tmp_path / "meta.yaml"

    _write_paired_csv(paired)
    np.savetxt(ref, np.array([1.0, 2.0, 3.0]), delimiter=",")
    np.savetxt(target, np.array([1.5, 2.5, 3.5]), delimiter=",")
    metadata.write_text(
        "\n".join(
            [
                "n_elec: 16",
                'stim_pattern: "{ad}"',
                'meas_pattern: "{ad}"',
            ]
        ),
        encoding="utf-8",
    )

    out_abs = _run(
        [
            "--method",
            "gn-absolute",
            "--csv",
            str(paired),
            "--metadata",
            str(metadata),
            "--output-root",
            str(tmp_path / "out_abs"),
            "--dry-run",
        ]
    )
    assert out_abs.returncode == 0

    out_diff = _run(
        [
            "--method",
            "gn-difference",
            "--csv",
            str(paired),
            "--output-root",
            str(tmp_path / "out_diff"),
            "--dry-run",
        ]
    )
    assert out_diff.returncode == 0

    out_sparse = _run(
        [
            "--method",
            "sparse-bayes",
            "--input-mode",
            "frame",
            "--csv",
            str(ref),
            "--csv",
            str(target),
            "--reference-index",
            "0",
            "--output-root",
            str(tmp_path / "out_sparse"),
            "--dry-run",
        ]
    )
    assert out_sparse.returncode == 0
