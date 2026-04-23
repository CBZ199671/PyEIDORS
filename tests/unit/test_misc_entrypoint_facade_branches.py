"""Small branch coverage tests for entrypoints and facade helpers."""

from __future__ import annotations

import runpy

import numpy as np
import pytest

import pyeidors
import pyeidors.core_system_facade as facade_module
import pyeidors.forward as forward_pkg
import pyeidors.main as main_module
from pyeidors.core_system_facade import CoreSystemFacadeMixin


def test_package_dir_and_invalid_lazy_attributes():
    assert "EITSystem" in pyeidors.__dir__()
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = getattr(pyeidors, "not_exported")
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = getattr(forward_pkg, "not_exported")


def test_main_module_runs_script_entrypoint(capsys):
    runpy.run_path(main_module.__file__, run_name="__main__")
    captured = capsys.readouterr()
    assert "Hello from pyeidors!" in captured.out


def test_core_system_facade_uses_default_baseline_and_helpers(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {}

    class _ReconFacade(CoreSystemFacadeMixin):
        def __init__(self):
            self._is_initialized = True

        def create_homogeneous_image(self, conductivity=None):
            return f"baseline:{conductivity}"

        def _require_initialized(self):
            return None

    system = _ReconFacade()
    monkeypatch.setattr(
        facade_module,
        "perform_absolute_reconstruction",
        lambda **kwargs: calls.setdefault("absolute", kwargs) or "absolute-out",
    )
    monkeypatch.setattr(
        facade_module,
        "perform_difference_reconstruction",
        lambda **kwargs: calls.setdefault("difference", kwargs) or "difference-out",
    )
    monkeypatch.setattr(
        facade_module,
        "create_homogeneous_image",
        lambda eit_system, conductivity=None: np.array(
            [conductivity or 1.0], dtype=float
        ),
    )
    monkeypatch.setattr(
        facade_module,
        "add_circular_phantom",
        lambda eit_system, **kwargs: ("phantom", kwargs),
    )
    monkeypatch.setattr(
        facade_module, "collect_system_info", lambda eit_system: {"ok": True}
    )

    abs_out = system.absolute_reconstruct(measurement_data="meas", baseline_image=None)
    assert calls["absolute"]["baseline_image"] == "baseline:None"
    assert abs_out == calls["absolute"]

    diff_out = system.difference_reconstruct(
        measurement_data="meas", reference_data="ref"
    )
    assert diff_out == calls["difference"]

    class _HelperFacade(CoreSystemFacadeMixin):
        def __init__(self):
            self._is_initialized = True

        def _require_initialized(self):
            return None

    helper = _HelperFacade()
    np.testing.assert_allclose(
        helper.create_homogeneous_image(2.0), np.array([2.0], dtype=float)
    )
    phantom = helper.add_phantom(phantom_conductivity=3.0)
    assert phantom[0] == "phantom"
    assert helper.get_system_info() == {"ok": True}
