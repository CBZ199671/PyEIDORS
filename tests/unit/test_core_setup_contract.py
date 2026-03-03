"""Contract tests for explicit EITSystem setup flows."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig


def _new_system() -> EITSystem:
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        amplitude=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    return EITSystem(
        n_elec=16,
        pattern_config=pattern,
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
    )


def test_setup_requires_explicit_source():
    system = _new_system()
    with pytest.raises(ValueError, match="requires an explicit mesh source"):
        system.setup()


def test_setup_rejects_unknown_source():
    system = _new_system()
    with pytest.raises(ValueError, match="requires an explicit mesh source"):
        system.setup(mesh_source="unknown")


def test_setup_dispatches_cache(monkeypatch):
    system = _new_system()
    calls = {}

    def _fake_setup_from_cache(*, mesh_dir: str, mesh_name: str | None):
        calls["mesh_dir"] = mesh_dir
        calls["mesh_name"] = mesh_name

    monkeypatch.setattr(system, "setup_from_cache", _fake_setup_from_cache)
    system.setup(mesh_source="cache", mesh_dir="my_meshes", mesh_name="mesh_001")

    assert calls["mesh_dir"] == "my_meshes"
    assert calls["mesh_name"] == "mesh_001"


def test_setup_dispatches_generated(monkeypatch):
    system = _new_system()
    calls = {}

    def _fake_setup_generated_mesh(*, radius: float | None, mesh_size: float | None):
        calls["radius"] = radius
        calls["mesh_size"] = mesh_size

    monkeypatch.setattr(system, "setup_generated_mesh", _fake_setup_generated_mesh)
    system.setup(mesh_source="generated", radius=1.5, mesh_size=0.08)

    assert calls["radius"] == 1.5
    assert calls["mesh_size"] == 0.08


def test_setup_with_mesh_type_guard():
    system = _new_system()
    with pytest.raises(TypeError, match="expects an EITMesh"):
        system.setup_with_mesh(object())  # type: ignore[arg-type]
