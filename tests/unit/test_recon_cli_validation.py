"""Validation tests for unified reconstruction CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_reconstruction_unified.py"
    )


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(_script_path()), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_help_is_available_without_full_runtime_stack():
    out = _run_cli(["--help"])
    assert out.returncode == 0
    assert "Unified PyEIDORS reconstruction CLI" in out.stdout


def test_requires_input_source(tmp_path: Path):
    out = _run_cli(
        [
            "--method",
            "gn-absolute",
            "--output-root",
            str(tmp_path / "out"),
            "--metadata",
            str(tmp_path / "meta.yaml"),
        ]
    )
    assert out.returncode != 0
    assert "Provide --input-dir or --csv" in out.stderr


def test_absolute_requires_metadata(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("1,2,3,4\n", encoding="utf-8")

    out = _run_cli(
        [
            "--method",
            "gn-absolute",
            "--csv",
            str(csv_path),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert out.returncode != 0
    assert "gn-absolute requires --metadata" in out.stderr


def test_difference_frame_requires_reference(tmp_path: Path):
    frame = tmp_path / "frame.csv"
    frame.write_text("1\n2\n3\n", encoding="utf-8")

    out = _run_cli(
        [
            "--method",
            "gn-difference",
            "--input-mode",
            "frame",
            "--csv",
            str(frame),
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert out.returncode != 0
    assert "requires --reference-csv or --reference-index" in out.stderr


def test_sparse_frame_allows_reference_index(tmp_path: Path):
    ref = tmp_path / "ref.csv"
    tar = tmp_path / "tar.csv"
    ref.write_text("1\n2\n3\n", encoding="utf-8")
    tar.write_text("3\n4\n5\n", encoding="utf-8")

    out = _run_cli(
        [
            "--method",
            "sparse-bayes",
            "--input-mode",
            "frame",
            "--csv",
            str(ref),
            "--csv",
            str(tar),
            "--reference-index",
            "0",
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert out.returncode == 0
    assert '"case_name": "tar"' in out.stdout


def test_cli_accepts_cache_controls_in_dry_run(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("1,2,3,4\n", encoding="utf-8")

    out = _run_cli(
        [
            "--method",
            "gn-difference",
            "--csv",
            str(csv_path),
            "--output-root",
            str(tmp_path / "out"),
            "--cache-scope",
            "both",
            "--cache-dir",
            str(tmp_path / ".pyeidors_cache"),
            "--cache-clear-name",
            "calc_jacobian",
            "--cache-clear-name",
            "inv_solve_diff_GN_one_step",
            "--dry-run",
        ]
    )
    assert out.returncode == 0


def test_cli_accepts_fast_solver_controls_in_dry_run(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("1,2,3,4\n", encoding="utf-8")

    out = _run_cli(
        [
            "--method",
            "gn-difference",
            "--csv",
            str(csv_path),
            "--output-root",
            str(tmp_path / "out"),
            "--mesh-dim",
            "3",
            "--solver-mode",
            "fast",
            "--linear-solver",
            "scipy-lsmr",
            "--jacobian-update-every",
            "2",
            "--jacobian-reuse-tol",
            "1e-3",
            "--line-search-mode",
            "fast",
            "--dry-run",
        ]
    )
    assert out.returncode == 0


def test_cli_accepts_perf_and_preconditioner_controls_in_dry_run(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("1,2,3,4\n", encoding="utf-8")

    out = _run_cli(
        [
            "--method",
            "gn-difference",
            "--csv",
            str(csv_path),
            "--output-root",
            str(tmp_path / "out"),
            "--mesh-dim",
            "3",
            "--preconditioner",
            "auto",
            "--fast-linear-path",
            "auto",
            "--cholmod-max-n",
            "12000",
            "--cholmod-max-memory-gib",
            "4.0",
            "--absolute-startup-cache",
            "on",
            "--forward-mat-solve",
            "auto",
            "--rom-mode",
            "auto",
            "--rom-rank-global",
            "32",
            "--rom-rank-adaptive",
            "16",
            "--rom-refresh-every",
            "2",
            "--rom-snapshot-source",
            "hybrid",
            "--inexact-mode",
            "auto",
            "--inexact-forcing",
            "eisenstat-walker",
            "--inexact-eta0",
            "0.2",
            "--inexact-eta-min",
            "1e-3",
            "--inexact-eta-max",
            "0.5",
            "--lowrank-mode",
            "auto",
            "--lowrank-rank",
            "16",
            "--lowrank-method",
            "tsvd",
            "--lowrank-energy",
            "0.995",
            "--jacobian-block-tune",
            "auto",
            "--jacobian-block-size",
            "0",
            "--jacobian-block-candidates",
            "64,128,256,512",
            "--perf-report",
            str(tmp_path / "perf.json"),
            "--perf-gate",
            "warn",
            "--dry-run",
        ]
    )
    assert out.returncode == 0


def test_cli_accepts_petsc_device_in_dry_run(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("1,2,3,4\n", encoding="utf-8")

    out = _run_cli(
        [
            "--method",
            "gn-difference",
            "--csv",
            str(csv_path),
            "--output-root",
            str(tmp_path / "out"),
            "--mesh-dim",
            "3",
            "--petsc-device",
            "cuda",
            "--dry-run",
        ]
    )
    assert out.returncode == 0


def test_cli_accepts_inverse_device_in_dry_run(tmp_path: Path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("1,2,3,4\n", encoding="utf-8")

    out = _run_cli(
        [
            "--method",
            "gn-difference",
            "--csv",
            str(csv_path),
            "--output-root",
            str(tmp_path / "out"),
            "--mesh-dim",
            "3",
            "--device",
            "cuda",
            "--dry-run",
        ]
    )
    assert out.returncode == 0
