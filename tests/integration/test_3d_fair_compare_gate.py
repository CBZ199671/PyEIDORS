"""Integration tests for fair-compare performance gate checks."""

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


def _full_payload(pass_case: bool) -> dict:
    if pass_case:
        return {
            "benchmark_phase": "full",
            "results": {
                "ref_1": {
                    "profiles": {
                        "A_baseline": {"median": {"absolute_peak_mib": 100.0}},
                        "B_cholmod_only": {
                            "median": {
                                "fast_solver_path": "pcg-cholmod-precond",
                                "fallback_reason": "",
                                "absolute_peak_mib": 105.0,
                            }
                        },
                        "D_combined": {"median": {"absolute_peak_mib": 108.0}},
                        "E_fused": {
                            "median": {
                                "absolute_peak_mib": 108.0,
                                "rom_enabled_effective": True,
                                "fast_solver_path": "fused-rom+inexact+lowrank",
                            }
                        },
                    },
                    "speedup_vs_A": {
                        "B_cholmod_only": {"absolute_linear_speedup_x": 1.25},
                        "D_combined": {"absolute_total_speedup_x": 1.17},
                        "E_fused": {"absolute_total_speedup_x": 1.01},
                    },
                },
                "ref_2": {
                    "profiles": {
                        "A_baseline": {"median": {"absolute_peak_mib": 120.0}},
                        "B_cholmod_only": {
                            "median": {
                                "fast_solver_path": "pcg-diag-precond",
                                "fallback_reason": "cholmod_memory_limit",
                                "absolute_peak_mib": 118.0,
                            }
                        },
                        "D_combined": {"median": {"absolute_peak_mib": 125.0}},
                        "E_fused": {
                            "median": {
                                "absolute_peak_mib": 126.0,
                                "rom_enabled_effective": True,
                                "fast_solver_path": "fused-rom+inexact",
                            }
                        },
                    },
                    "speedup_vs_A": {
                        "B_cholmod_only": {"absolute_linear_speedup_x": 1.10},
                        "C_autotune_only": {
                            "absolute_jacobian_assembly_speedup_x": 1.20
                        },
                        "D_combined": {"absolute_total_speedup_x": 1.12},
                        "E_fused": {"absolute_total_speedup_x": 1.00},
                    },
                },
            },
        }
    return {
        "benchmark_phase": "full",
        "results": {
            "ref_1": {
                "profiles": {
                    "A_baseline": {"median": {"absolute_peak_mib": 100.0}},
                    "B_cholmod_only": {
                        "median": {
                            "fast_solver_path": "pcg-diag-precond",
                            "fallback_reason": "",
                            "absolute_peak_mib": 130.0,
                        }
                    },
                    "D_combined": {"median": {"absolute_peak_mib": 140.0}},
                    "E_fused": {
                        "median": {
                            "absolute_peak_mib": 140.0,
                            "rom_enabled_effective": False,
                            "fast_solver_path": "pcg-diag-precond",
                        }
                    },
                },
                "speedup_vs_A": {
                    "B_cholmod_only": {"absolute_linear_speedup_x": 1.01},
                    "D_combined": {"absolute_total_speedup_x": 0.95},
                    "E_fused": {"absolute_total_speedup_x": 1.03},
                },
            }
        },
    }


def _quick_payload(pass_case: bool) -> dict:
    if pass_case:
        return {
            "benchmark_phase": "quick",
            "quick_pass": True,
            "quick_thresholds": {
                "total": 0.05,
                "linear": 0.10,
                "peak_overhead_limit": 0.10,
            },
            "quick_eval": {
                "total_improvement_ratio": 0.06,
                "linear_improvement_ratio": 0.02,
                "peak_memory_delta_ratio": 0.05,
            },
            "results": {"ref_1": {"profiles": {"A_baseline": {}, "D_combined": {}}}},
        }
    return {
        "benchmark_phase": "quick",
        "quick_pass": False,
        "quick_thresholds": {
            "total": 0.05,
            "linear": 0.10,
            "peak_overhead_limit": 0.10,
        },
        "quick_eval": {
            "total_improvement_ratio": 0.01,
            "linear_improvement_ratio": 0.02,
            "peak_memory_delta_ratio": 0.20,
        },
        "results": {"ref_1": {"profiles": {"A_baseline": {}, "D_combined": {}}}},
    }


def _run_gate(report: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_script_path()),
            "--input",
            str(report),
            "--mode",
            "strict",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_fair_compare_full_gate_strict_passes_for_good_payload(tmp_path: Path):
    report = tmp_path / "fair_pass.json"
    report.write_text(json.dumps(_full_payload(pass_case=True)), encoding="utf-8")
    proc = _run_gate(report)
    assert proc.returncode == 0, proc.stderr
    assert "[OK] performance gate passed" in proc.stdout


def test_fair_compare_full_gate_strict_fails_for_bad_payload(tmp_path: Path):
    report = tmp_path / "fair_fail.json"
    report.write_text(json.dumps(_full_payload(pass_case=False)), encoding="utf-8")
    proc = _run_gate(report)
    assert proc.returncode != 0
    assert "performance gate failed" in proc.stderr


def test_fair_compare_quick_gate_strict_passes_for_good_payload(tmp_path: Path):
    report = tmp_path / "quick_pass.json"
    report.write_text(json.dumps(_quick_payload(pass_case=True)), encoding="utf-8")
    proc = _run_gate(report)
    assert proc.returncode == 0, proc.stderr


def test_fair_compare_quick_gate_strict_fails_for_bad_payload(tmp_path: Path):
    report = tmp_path / "quick_fail.json"
    report.write_text(json.dumps(_quick_payload(pass_case=False)), encoding="utf-8")
    proc = _run_gate(report)
    assert proc.returncode != 0
    assert "performance gate failed" in proc.stderr
