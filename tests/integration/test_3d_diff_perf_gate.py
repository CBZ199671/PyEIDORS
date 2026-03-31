"""Integration tests for diff-side performance gate checks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _script_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmarks"
        / "check_perf_gate.py"
    )


def test_perf_gate_warn_does_not_fail_process(tmp_path: Path):
    payload = {
        "config": {"absolute_iters": 2},
        "stages": [
            {"stage": "diff_context_cold", "elapsed_sec": 40.0, "peak_mib": 100.0},
            {"stage": "diff_context_warm", "elapsed_sec": 2.0, "peak_mib": 90.0},
            {"stage": "absolute_reconstruct", "elapsed_sec": 80.0, "peak_mib": 300.0},
        ],
    }
    report = tmp_path / "perf_bad.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(_script_path()), "--input", str(report), "--mode", "warn"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "performance gate failed" in proc.stdout


def test_perf_gate_strict_fails_for_bad_report(tmp_path: Path):
    payload = {
        "config": {"absolute_iters": 2},
        "stages": [
            {"stage": "diff_context_cold", "elapsed_sec": 50.0, "peak_mib": 100.0},
            {"stage": "diff_context_warm", "elapsed_sec": 10.0, "peak_mib": 100.0},
            {"stage": "absolute_reconstruct", "elapsed_sec": 100.0, "peak_mib": 500.0},
        ],
    }
    report = tmp_path / "perf_bad_strict.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(_script_path()), "--input", str(report), "--mode", "strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "performance gate failed" in proc.stderr
