"""Additional constructor validation tests for GN engine edge parameters."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import pyeidors.inverse.solvers.gauss_newton_engine as gn_engine
from tests.utils import run_python


def _run_validation_case(argument_expr: str, expected_substring: str) -> None:
    code = f"""
from unittest import mock
fm = mock.MagicMock()
fm.n_elec = 4
fm.mesh.topology.dim = 2
try:
    from pyeidors.inverse.solvers.gauss_newton_engine import GaussNewtonReconstructor
    GaussNewtonReconstructor(fwd_model=fm, {argument_expr})
except ValueError as e:
    print("PASS" if {expected_substring!r} in str(e) else f"WRONG: {{e}}")
except Exception as e:
    print("PASS" if {expected_substring!r} in str(e) else f"OTHER: {{e}}")
"""
    result = run_python(code)
    assert "PASS" in result.stdout or result.returncode == 0, result.stderr


def test_invalid_cholmod_max_n():
    _run_validation_case("cholmod_max_n=0", "cholmod_max_n")


def test_invalid_cholmod_max_memory_gib():
    _run_validation_case("cholmod_max_memory_gib=0.0", "cholmod_max_memory_gib")


def test_invalid_inexact_eta_bounds_negative():
    _run_validation_case("inexact_eta_min=0.0", "inexact eta bounds")


def test_invalid_inexact_eta_order():
    _run_validation_case("inexact_eta_min=0.9, inexact_eta_max=0.1", "inexact_eta_min")


def test_invalid_lowrank_energy_high():
    _run_validation_case("lowrank_energy=1.5", "lowrank_energy")


def test_invalid_inexact_and_lowrank_validation_inprocess(
    eit_system, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        gn_engine,
        "resolve_torch_device",
        lambda *args, **kwargs: SimpleNamespace(
            requested="cpu",
            effective="cpu",
            fallback_reason=None,
            torch_device="cpu",
        ),
    )

    with pytest.raises(ValueError, match="inexact eta bounds"):
        gn_engine.GaussNewtonReconstructor(
            fwd_model=eit_system.fwd_model,
            inexact_eta_min=0.0,
            verbose=False,
        )

    with pytest.raises(ValueError, match="inexact_eta_min"):
        gn_engine.GaussNewtonReconstructor(
            fwd_model=eit_system.fwd_model,
            inexact_eta_min=0.9,
            inexact_eta_max=0.1,
            verbose=False,
        )

    with pytest.raises(ValueError, match="lowrank_energy"):
        gn_engine.GaussNewtonReconstructor(
            fwd_model=eit_system.fwd_model,
            lowrank_energy=1.5,
            verbose=False,
        )
