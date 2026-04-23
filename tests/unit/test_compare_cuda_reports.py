"""Tests for CUDA report comparison helper."""

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
        / "compare_cuda_reports.py"
    )


def test_compare_cuda_reports_handles_2d_payload(tmp_path: Path):
    cpu = tmp_path / "cpu_2d.json"
    gpu = tmp_path / "gpu_2d.json"
    cpu.write_text(
        json.dumps(
            {
                "stage_timings": {
                    "forward_homogeneous": 2.0,
                    "forward_phantom": 2.0,
                    "difference_reconstruct": 4.0,
                },
                "perf_summary": {"execution_profile": "cpu"},
            }
        ),
        encoding="utf-8",
    )
    gpu.write_text(
        json.dumps(
            {
                "stage_timings": {
                    "forward_homogeneous": 1.0,
                    "forward_phantom": 1.0,
                    "difference_reconstruct": 3.0,
                },
                "perf_summary": {"execution_profile": "cuda"},
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--cpu",
            str(cpu),
            "--gpu",
            str(gpu),
            "--kind",
            "2d",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["gpu_faster_total"] is True
    assert payload["gpu_faster_forward_heavy"] is True


def test_compare_cuda_reports_handles_3d_payload(tmp_path: Path):
    cpu = tmp_path / "cpu_3d.json"
    gpu = tmp_path / "gpu_3d.json"
    cpu.write_text(
        json.dumps(
            {
                "stages": [
                    {"stage": "diff_context_cold", "elapsed_sec": 5.0},
                    {"stage": "absolute_reconstruct", "elapsed_sec": 10.0},
                ],
                "stage_breakdown": {
                    "absolute": {"forward": 4.0},
                    "difference": {"forward_validate": 2.0},
                },
                "absolute_solver": {
                    "execution_profile": "cpu",
                    "first_forward_elapsed_sec": 2.0,
                    "warm_forward_avg_sec": 0.8,
                    "warm_forward_total_sec": 4.0,
                    "absolute_reconstruct_elapsed_sec": 10.0,
                },
                "difference_solver": {
                    "difference_context_cold_elapsed_sec": 5.0,
                    "difference_context_warm_elapsed_sec": 1.0,
                    "difference_reconstruct_elapsed_sec": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )
    gpu.write_text(
        json.dumps(
            {
                "stages": [
                    {"stage": "diff_context_cold", "elapsed_sec": 4.0},
                    {"stage": "absolute_reconstruct", "elapsed_sec": 8.0},
                ],
                "stage_breakdown": {
                    "absolute": {"forward": 2.0},
                    "difference": {"forward_validate": 1.0},
                },
                "absolute_solver": {
                    "execution_profile": "cuda",
                    "first_forward_elapsed_sec": 1.0,
                    "warm_forward_avg_sec": 0.4,
                    "warm_forward_total_sec": 2.0,
                    "absolute_reconstruct_elapsed_sec": 8.0,
                },
                "difference_solver": {
                    "difference_context_cold_elapsed_sec": 4.0,
                    "difference_context_warm_elapsed_sec": 0.5,
                    "difference_reconstruct_elapsed_sec": 3.0,
                },
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--cpu",
            str(cpu),
            "--gpu",
            str(gpu),
            "--kind",
            "3d",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["gpu_faster_total"] is True
    assert payload["gpu_faster_forward_heavy"] is True
    assert payload["gpu_faster_first_forward"] is True
    assert payload["gpu_faster_warm_forward_avg"] is True
    assert payload["forward_heavy_metric"] == "absolute_warm_forward_avg"
    assert payload["gpu_faster_absolute_reconstruct"] is True
    assert payload["gpu_faster_difference_total"] is True
