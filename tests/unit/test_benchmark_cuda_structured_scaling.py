"""Lightweight contract tests for the cuda_structured scaling benchmark script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmarks"
        / "benchmark_cuda_structured_scaling.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_cuda_structured_scaling", script)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError("failed to load benchmark_cuda_structured_scaling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_refinements_normalizes_and_sorts():
    module = _load_module()
    assert module._parse_refinements("8,3,4,8") == [3, 4, 8]


def test_ref8_gate_requires_first_and_warm_speedups():
    module = _load_module()
    passed = module._evaluate_gate(8, first_forward_speedup=3.1, warm_forward_speedup=5.1)
    failed_first = module._evaluate_gate(8, first_forward_speedup=2.9, warm_forward_speedup=5.1)
    failed_warm = module._evaluate_gate(8, first_forward_speedup=3.1, warm_forward_speedup=4.9)

    assert passed["passed"] is True
    assert failed_first["passed"] is False
    assert failed_warm["passed"] is False
