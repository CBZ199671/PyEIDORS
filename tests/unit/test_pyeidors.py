"""Top-level package API checks."""

from __future__ import annotations

import builtins
import importlib.util

import pytest


def test_environment_schema_is_reported_without_eager_runtime_imports():
    import pyeidors

    env = pyeidors.check_environment()
    assert "dolfinx_available" in env
    assert "torch_available" in env
    assert "mps_available" in env
    assert isinstance(env["dolfinx_available"], bool)

    if importlib.util.find_spec("dolfinx") is not None:
        assert env["dolfinx_available"] is True


def test_eitsystem_access_reports_missing_runtime_cleanly():
    import pyeidors

    if importlib.util.find_spec("dolfinx") is not None:
        assert pyeidors.EITSystem.__name__ == "EITSystem"
        return

    with pytest.raises(ImportError, match="nix develop"):
        _ = pyeidors.EITSystem


def test_eitsystem_access_normalizes_modulenotfound_runtime_errors(monkeypatch):
    import pyeidors

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("core_system"):
            exc = ModuleNotFoundError("No module named 'dolfinx'")
            exc.name = "dolfinx"
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="nix develop"):
        _ = pyeidors.EITSystem


def test_eitsystem_access_normalizes_import_runtime_errors(monkeypatch):
    import pyeidors

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("core_system"):
            raise ImportError("libstdc++.so.6: cannot open shared object file")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="nix develop"):
        _ = pyeidors.EITSystem


def test_removed_solver_alias_is_unavailable():
    removed_name = "Standard" + "GaussNewtonReconstructor"
    with pytest.raises((ImportError, KeyError, AttributeError)):
        __import__("pyeidors.inverse.solvers", fromlist=[removed_name]).__dict__[
            removed_name
        ]


def test_public_inverse_solver_name():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("requires torch runtime")

    from pyeidors.inverse.solvers import GaussNewtonReconstructor

    assert GaussNewtonReconstructor.__name__ == "GaussNewtonReconstructor"
