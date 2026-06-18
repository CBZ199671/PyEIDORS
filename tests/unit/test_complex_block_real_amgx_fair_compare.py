"""Unit tests for complex block-real AmgX fair-compare orchestration."""

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
        / "complex_block_real_amgx_fair_compare.py"
    )
    spec = importlib.util.spec_from_file_location(
        "complex_block_real_fair_compare_test", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_cases_accepts_electrode_refinement_pairs() -> None:
    module = _load_compare_module()

    assert module._parse_cases("8:1, 16:2") == [(8, 1), (16, 2)]


def test_case_row_passes_when_solution_and_electrode_errors_under_threshold(
    tmp_path: Path,
) -> None:
    module = _load_compare_module()
    case_dir = tmp_path / "8e_ref1"
    case_dir.mkdir()
    (case_dir / "metadata.json").write_text(
        json.dumps(
            {
                "mesh": {"elements": 1138},
                "n_dofs": 306,
                "n_patterns": 8,
                "runtime_vs_direct": {"relative_l2": 1.0e-7},
                "runtime_electrode_voltage_vs_direct": {"relative_l2": 1.0e-7},
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "block_real_amgx.json").write_text(
        json.dumps(
            {
                "reference_kind": "scipy_splu_direct",
                "solver": {"iterations_per_rhs": [24, 24]},
                "solution_error_vs_reference": {"relative_l2": 7.0e-6},
                "electrode_voltage_error_vs_reference": {"relative_l2": 6.0e-6},
            }
        ),
        encoding="utf-8",
    )

    row = module._case_row(
        case_dir=case_dir,
        n_elec=8,
        refinement=1,
        max_solution_rel_l2=1.0e-4,
        max_electrode_rel_l2=1.0e-4,
    )

    assert row["pass"] is True
    assert row["reference_kind"] == "scipy_splu_direct"
    assert row["solver"]["iterations_per_rhs"] == [24, 24]
