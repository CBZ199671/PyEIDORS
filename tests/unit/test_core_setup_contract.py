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
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
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

    def _fake_setup_from_cache(
        *,
        mesh_dir: str,
        mesh_name: str | None,
        gdim: int = 2,
    ):
        calls["mesh_dir"] = mesh_dir
        calls["mesh_name"] = mesh_name
        calls["gdim"] = gdim

    monkeypatch.setattr(system, "setup_from_cache", _fake_setup_from_cache)
    system.setup(mesh_source="cache", mesh_dir="my_meshes", mesh_name="mesh_001")

    assert calls["mesh_dir"] == "my_meshes"
    assert calls["mesh_name"] == "mesh_001"
    assert calls["gdim"] == 2


def test_setup_dispatches_generated(monkeypatch):
    system = _new_system()
    calls = {}

    def _fake_setup_generated_mesh(
        *,
        radius: float | None,
        mesh_size: float | None,
        dimension: int = 2,
        height: float | None = None,
        electrode_height_ratio: float | None = None,
        z_center: float | None = None,
    ):
        calls["radius"] = radius
        calls["mesh_size"] = mesh_size
        calls["dimension"] = dimension
        calls["height"] = height
        calls["electrode_height_ratio"] = electrode_height_ratio
        calls["z_center"] = z_center

    monkeypatch.setattr(system, "setup_generated_mesh", _fake_setup_generated_mesh)
    system.setup(mesh_source="generated", radius=1.5, mesh_size=0.08)

    assert calls["radius"] == 1.5
    assert calls["mesh_size"] == 0.08
    assert calls["dimension"] == 2


def test_setup_with_mesh_type_guard():
    system = _new_system()
    with pytest.raises(TypeError, match="expects an EITMesh"):
        system.setup_with_mesh(object())  # type: ignore[arg-type]


def test_setup_from_cache_calls_loader_paths(monkeypatch):
    system = _new_system()
    captured = {}

    class _FakeLoader:
        def __init__(self, mesh_dir: str, gdim: int = 2):
            captured["mesh_dir"] = mesh_dir
            captured["gdim"] = gdim

        def load_mesh(self, mesh_name: str):
            return f"mesh:{mesh_name}"

        def get_default_mesh(self):
            return "mesh:default"

    monkeypatch.setattr("pyeidors.core_system.MeshLoader", _FakeLoader)
    monkeypatch.setattr(system, "setup_with_mesh", lambda mesh: captured.__setitem__("mesh", mesh))

    system.setup_from_cache(mesh_dir="cache_dir", mesh_name="foo")
    assert captured["mesh_dir"] == "cache_dir"
    assert captured["gdim"] == 2
    assert captured["mesh"] == "mesh:foo"

    system.setup_from_cache(mesh_dir="cache_dir", mesh_name=None)
    assert captured["mesh"] == "mesh:default"


def test_setup_generated_mesh_uses_defaults_and_overrides(monkeypatch):
    system = _new_system()
    system.mesh_config.radius = 1.23
    system.mesh_config.mesh_size = 0.07

    generated_calls = []
    monkeypatch.setattr(
        "pyeidors.core_system.create_simple_eit_mesh",
        lambda **kwargs: generated_calls.append(kwargs) or "generated-mesh",
    )
    monkeypatch.setattr(system, "setup_with_mesh", lambda mesh: generated_calls.append({"mesh": mesh}))

    system.setup_generated_mesh()
    system.setup_generated_mesh(radius=2.0, mesh_size=0.05)

    assert generated_calls[0]["radius"] == 1.23
    assert generated_calls[0]["mesh_size"] == 0.07
    assert generated_calls[1]["mesh"] == "generated-mesh"
    assert generated_calls[2]["radius"] == 2.0
    assert generated_calls[2]["mesh_size"] == 0.05


def test_system_stores_public_device_policy():
    system = _new_system()
    assert system.device == "auto"

    system_cpu = EITSystem(
        n_elec=16,
        pattern_config=system.pattern_config,
        contact_impedance=np.full(16, 1e-5, dtype=float),
        regularization_type="noser",
        regularization_alpha=1.0,
        device="cpu",
    )
    assert system_cpu.device == "cpu"
