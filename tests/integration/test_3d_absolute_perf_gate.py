"""Integration tests for absolute-side performance gate checks."""

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


def test_perf_gate_strict_passes_for_good_report(tmp_path: Path):
    payload = {
        "config": {"absolute_iters": 2},
        "stages": [
            {"stage": "diff_context_cold", "elapsed_sec": 10.0, "peak_mib": 100.0},
            {"stage": "diff_context_warm", "elapsed_sec": 0.05, "peak_mib": 50.0},
            {"stage": "absolute_reconstruct", "elapsed_sec": 15.0, "peak_mib": 120.0},
        ],
    }
    report = tmp_path / "perf.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(_script_path()), "--input", str(report), "--mode", "strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "[OK] performance gate passed" in proc.stdout
