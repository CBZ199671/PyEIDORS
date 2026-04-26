"""Batched GN engine validation tests without per-case subprocess cold starts."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest


_TEST_IMPORT_ERROR = None
try:
    from pyeidors.inverse.solvers.gauss_newton_engine import GaussNewtonReconstructor
except Exception as exc:
    _TEST_IMPORT_ERROR = exc


def _skip_if_unavailable() -> None:
    if _TEST_IMPORT_ERROR is not None:
        pytest.skip(f"GN engine unavailable: {_TEST_IMPORT_ERROR}")


def _make_fwd_model():
    fwd = mock.MagicMock()
    fwd.n_elec = 4
    fwd.mesh.topology.dim = 2
    return fwd


@pytest.mark.parametrize(
    ("kwargs", "expected_match"),
    [
        ({"performance_mode": "bad"}, "performance_mode"),
        ({"solver_mode": "bad"}, "solver_mode"),
        ({"linear_solver": "bad"}, "linear_solver"),
        ({"line_search_mode": "bad"}, "line_search_mode"),
        ({"preconditioner": "bad"}, "preconditioner"),
        ({"fast_linear_path": "bad"}, "fast_linear_path"),
        ({"rom_mode": "bad"}, "rom_mode"),
        ({"rom_snapshot_source": "bad"}, "rom_snapshot_source"),
        ({"inexact_mode": "bad"}, "inexact_mode"),
        ({"inexact_forcing": "bad"}, "inexact_forcing"),
        ({"lowrank_mode": "bad"}, "lowrank_mode"),
        ({"lowrank_method": "bad"}, "lowrank_method"),
        ({"lowrank_energy": 0.0}, "lowrank_energy"),
        ({"lowrank_energy": 1.5}, "lowrank_energy"),
        ({"inexact_eta_min": -0.1}, "inexact eta bounds"),
        ({"inexact_eta_min": 0.9, "inexact_eta_max": 0.1}, "inexact_eta_min"),
    ],
)
def test_gn_engine_validation_errors_batched_in_process(kwargs, expected_match) -> None:
    _skip_if_unavailable()
    with pytest.raises(ValueError, match=expected_match):
        GaussNewtonReconstructor(fwd_model=_make_fwd_model(), **kwargs)


def test_gn_validation_unit_tests_do_not_use_per_case_subprocesses() -> None:
    run_python_call = "run_" + "python("
    unit_dir = Path(__file__).parent
    guarded_paths = [
        *unit_dir.glob("test_coverage_gn_engine_validation*.py"),
        unit_dir / "test_gn_engine_validation_edges.py",
        unit_dir / "test_recon_cli_validation.py",
    ]
    for path in guarded_paths:
        text = path.read_text(encoding="utf-8")
        assert run_python_call not in text, path
        assert ("subprocess" + ".run(") not in text, path
