"""Unit coverage for forward report numeric parity helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_compare_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmarks"
        / "compare_cuda_reports.py"
    )
    spec = importlib.util.spec_from_file_location(
        "compare_reports_for_parity_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_forward_output_parity_decodes_complex_electrode_payloads() -> None:
    module = _load_compare_module()
    cpu_report = {
        "forward_solver_benchmark": {
            "electrode_voltages": {
                "shape": [1, 2],
                "dtype": "complex128",
                "real": [1.0, 2.0],
                "imag": [0.5, -0.5],
            }
        }
    }
    gpu_report = {
        "forward_solver_benchmark": {
            "electrode_voltages": {
                "shape": [1, 2],
                "dtype": "complex128",
                "real": [1.0, 2.1],
                "imag": [0.5, -0.45],
            }
        }
    }

    parity = module._forward_output_parity(cpu_report, gpu_report)

    assert parity["available"] is True
    assert parity["shape"] == [1, 2]
    assert parity["relative_l2"] > 0.0
    assert parity["max_abs"] > 0.0


def test_forward_output_parity_reports_missing_values() -> None:
    module = _load_compare_module()

    parity = module._forward_output_parity({}, {})

    assert parity == {
        "available": False,
        "reason": "missing_forward_electrode_voltages",
    }
