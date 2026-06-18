"""Unit tests for complex route speed/accuracy comparison orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_compare_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "diagnostics"
        / "complex_route_speed_accuracy_compare.py"
    )
    spec = importlib.util.spec_from_file_location(
        "complex_route_speed_accuracy_compare_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_cases_accepts_electrode_refinement_pairs() -> None:
    module = _load_compare_module()

    assert module._parse_cases("8:1, 16:2") == [(8, 1), (16, 2)]


def test_case_row_contains_cpu_dense_native_gpu_and_amgx_routes(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    case_dir = tmp_path / "16e_ref2"
    case_dir.mkdir()
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "reference_route": "3d_gamg",
                "petsc_scalar_type": "complex64",
                "petsc_device": "cuda",
                "mesh": {"elements": 3962},
                "n_dofs": 990,
                "n_patterns": 16,
                "reference_solution": {
                    "kind": "scipy_splu_direct",
                    "solve_seconds": 0.8,
                },
                "dense_reference_solution": {
                    "kind": "numpy_dense_direct",
                    "solve_seconds": 0.7,
                },
                "dense_direct_vs_sparse_direct": {"relative_l2": 1.0e-12},
                "dense_direct_electrode_voltage_vs_sparse_direct": {
                    "relative_l2": 1.0e-12
                },
                "runtime_reference_solution": {
                    "solve_seconds": 1.2,
                    "backend_diagnostics": {"fallback_reason": "cuda_dense_lu"},
                },
                "runtime_vs_direct": {"relative_l2": 8.0e-6},
                "runtime_electrode_voltage_vs_direct": {"relative_l2": 6.0e-6},
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "block_real_amgx.json").write_text(
        json.dumps(
            {
                "reference_kind": "scipy_splu_direct",
                "solver": {
                    "solve_seconds": 0.3,
                    "iterations_per_rhs": [29, 30],
                },
                "solution_error_vs_reference": {"relative_l2": 1.0e-8},
                "electrode_voltage_error_vs_reference": {"relative_l2": 8.0e-9},
            }
        ),
        encoding="utf-8",
    )

    row = module._case_row(
        case_dir=case_dir,
        n_elec=16,
        refinement=2,
        max_solution_rel_l2=1.0e-4,
        max_electrode_rel_l2=1.0e-4,
    )

    assert row["pass"] is True
    assert [route["family"] for route in row["routes"]] == [
        "cpu_direct_reference",
        "cpu_dense_reference_check",
        "native_complex_cuda",
        "block_real_amgx",
    ]
    assert row["routes"][0]["solution_relative_l2"] == 0.0
    assert row["routes"][1]["solution_relative_l2"] == 1.0e-12
    assert row["routes"][2]["solution_relative_l2"] == 8.0e-6
    assert row["routes"][3]["solver"]["iterations_per_rhs"] == [29, 30]


def test_speedup_uses_cpu_direct_as_baseline() -> None:
    module = _load_compare_module()

    assert module._speedup({"solve_seconds": 0.25}, {"solve_seconds": 1.0}) == 4.0


def test_case_row_gpu_only_mode_uses_native_cuda_as_reference(tmp_path: Path) -> None:
    module = _load_compare_module()
    case_dir = tmp_path / "16e_ref3"
    case_dir.mkdir()
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "reference_route": "3d_gamg",
                "petsc_scalar_type": "complex64",
                "petsc_device": "cuda",
                "mesh": {"elements": 31208},
                "n_dofs": 6536,
                "n_patterns": 16,
                "reference_solution": {
                    "kind": "3d_gamg",
                    "solve_seconds": 2.4,
                    "cpu_direct_skipped": True,
                },
                "runtime_reference_solution": {
                    "solve_seconds": 2.4,
                    "backend_diagnostics": {"fallback_reason": "cuda_dense_lu"},
                },
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "block_real_amgx.json").write_text(
        json.dumps(
            {
                "reference_kind": "runtime_reference",
                "solver": {
                    "solve_seconds": 0.6,
                    "iterations_per_rhs": [42, 44],
                },
                "solution_error_vs_reference": {"relative_l2": 9.0e-6},
                "electrode_voltage_error_vs_reference": {"relative_l2": 5.0e-6},
            }
        ),
        encoding="utf-8",
    )

    row = module._case_row(
        case_dir=case_dir,
        n_elec=16,
        refinement=3,
        max_solution_rel_l2=1.0e-4,
        max_electrode_rel_l2=1.0e-4,
    )

    assert row["pass"] is True
    assert row["cpu_direct_skipped"] is True
    assert [route["family"] for route in row["routes"]] == [
        "native_complex_cuda_reference",
        "block_real_amgx",
    ]
    assert row["routes"][0]["solution_relative_l2"] == 0.0
    assert row["routes"][1]["electrode_relative_l2"] == 5.0e-6
