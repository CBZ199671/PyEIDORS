"""Top-level package API checks."""

from __future__ import annotations

import pytest


def test_environment_schema_is_dolfinx_only():
    import pyeidors

    env = pyeidors.check_environment()
    assert env["dolfinx_available"] is True
    assert "dolfinx_available" in env
    assert "torch_available" in env
    assert "mps_available" in env


def test_legacy_solver_alias_removed():
    legacy_name = "Standard" + "GaussNewtonReconstructor"
    with pytest.raises((ImportError, KeyError, AttributeError)):
        __import__("pyeidors.inverse.solvers", fromlist=[legacy_name]).__dict__[legacy_name]


def test_public_inverse_solver_name():
    from pyeidors.inverse.solvers import GaussNewtonReconstructor

    assert GaussNewtonReconstructor.__name__ == "GaussNewtonReconstructor"
