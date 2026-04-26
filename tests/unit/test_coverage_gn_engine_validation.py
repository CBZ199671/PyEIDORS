"""Tests for gauss_newton_engine parameter validation to achieve coverage."""

from __future__ import annotations

from unittest import mock

import pytest


_TEST_IMPORT_ERROR = None
try:
    from pyeidors.inverse.solvers.gauss_newton_engine import GaussNewtonReconstructor
except Exception as exc:
    _TEST_IMPORT_ERROR = exc


def _skip():
    if _TEST_IMPORT_ERROR is not None:
        pytest.skip(f"GN engine unavailable: {_TEST_IMPORT_ERROR}")


def _make_fwd_model():
    _skip()
    fwd = mock.MagicMock()
    fwd.n_elec = 4
    fwd.mesh.topology.dim = 2
    return fwd


def _assert_validation_error(expected_match: str, **kwargs) -> None:
    _skip()
    with pytest.raises(ValueError, match=expected_match):
        GaussNewtonReconstructor(fwd_model=_make_fwd_model(), **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "expected_match"),
    [
        ({"performance_mode": "invalid"}, "performance_mode"),
        ({"solver_mode": "invalid"}, "solver_mode"),
        ({"linear_solver": "invalid"}, "linear_solver"),
        ({"rom_mode": "invalid"}, "rom_mode"),
        ({"lowrank_energy": 0.0}, "lowrank_energy"),
    ],
)
def test_gn_engine_validation_errors_in_process(kwargs, expected_match) -> None:
    _assert_validation_error(expected_match, **kwargs)
