"""Lightweight contract tests for the cuda_structured scaling benchmark script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "benchmarks"
        / "benchmark_cuda_structured_scaling.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_cuda_structured_scaling", script
    )
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
    passed = module._evaluate_gate(
        8, first_forward_speedup=3.1, warm_forward_speedup=5.1
    )
    failed_first = module._evaluate_gate(
        8, first_forward_speedup=2.9, warm_forward_speedup=5.1
    )
    failed_warm = module._evaluate_gate(
        8, first_forward_speedup=3.1, warm_forward_speedup=4.9
    )

    assert passed["passed"] is True
    assert failed_first["passed"] is False
    assert failed_warm["passed"] is False


def test_main_uses_resolved_mesh_contract_in_output(monkeypatch, tmp_path: Path):
    module = _load_module()
    output_json = tmp_path / "scaling.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_cuda_structured_scaling.py",
            "--output-json",
            str(output_json),
            "--refinements",
            "3",
            "--gate",
            "off",
        ],
    )
    monkeypatch.setattr(
        module,
        "resolve_3d_mesh_contract",
        lambda *, acceleration_profile: ("hex_custom", "geom_custom", "g3d9"),
    )
    monkeypatch.setattr(
        module,
        "_run_refinement",
        lambda args, *, refinement, mesh_root, cache_root: {  # noqa: ARG005
            "refinement": int(refinement),
            "gate": {"passed": True},
        },
    )

    module.main()

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["config"]["mesh_family"] == "hex_custom"
    assert payload["config"]["geometry_version"] == "geom_custom"
    assert payload["config"]["generator_revision"] == "g3d9"
