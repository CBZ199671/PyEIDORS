from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from eit_app.controllers.forward_solver_controller import _resolve_forward_runtime  # noqa: E402
from eit_app.controllers.reconstruction_controller import (  # noqa: E402
    _resolve_reconstruction_runtime,
)
from eit_app.models.forward_model_config import ForwardModelConfig  # noqa: E402
from eit_app.ui.conductivity_3d_widget import (  # noqa: E402
    Conductivity3DWidget,
    SUPPORTED_3D_CELL_VERTEX_COUNTS,
    embedded_vtk_enabled,
    embedded_vtk_status,
)
from eit_app.ui.simulation.simulation_results_widget import (  # noqa: E402
    _ConductivityViewSlot,
)
from eit_app.ui.simulation.mesh_setup_panel import MeshSetupPanel  # noqa: E402


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _tetra_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int64)
    sigma = np.array([1.25], dtype=float)
    return sigma, coords, cells


def _hex_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    sigma = np.array([1.75], dtype=float)
    return sigma, coords, cells


def test_supported_3d_cell_types_cover_tetra_and_hex():
    assert {4, 8}.issubset(SUPPORTED_3D_CELL_VERTEX_COUNTS)


def test_mesh_setup_panel_exposes_tetra_and_hex_3d_families():
    _get_app()
    panel = MeshSetupPanel()
    try:
        panel.set_config({"mesh_dimension": 3, "mesh_family": "tetra"})
        assert panel.get_config()["mesh_family"] == "tetra"

        panel.set_config({"mesh_dimension": 3, "mesh_family": "hex"})
        assert panel.get_config()["mesh_family"] == "hex"

        panel.set_config({"mesh_dimension": 2, "mesh_family": "hex"})
        assert panel.get_config()["mesh_family"] == "tetra"
    finally:
        panel.close()


def test_gpu_forward_runtime_keeps_tetra_and_hex_distinct(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")

    tetra = _resolve_forward_runtime(
        ForwardModelConfig(mesh_dimension=3, mesh_family="tetra")
    )
    assert tetra["mesh_family"] == "tetra"
    assert tetra["forward_backend"] == "dolfinx"
    assert tetra["petsc_device"] == "cuda"

    hex_cfg = _resolve_forward_runtime(
        ForwardModelConfig(mesh_dimension=3, mesh_family="hex")
    )
    assert hex_cfg["mesh_family"] == "hex"
    assert hex_cfg["forward_backend"] == "cuda_structured"
    assert hex_cfg["petsc_device"] == "cuda"


def test_gpu_reconstruction_runtime_keeps_tetra_and_hex_distinct(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")

    tetra = _resolve_reconstruction_runtime(
        {"mesh_family": "tetra", "forward_backend": "cuda_structured"},
        mesh_dim=3,
    )
    assert tetra["mesh_family"] == "tetra"
    assert tetra["forward_backend"] == "dolfinx"

    hex_cfg = _resolve_reconstruction_runtime({"mesh_family": "hex"}, mesh_dim=3)
    assert hex_cfg["mesh_family"] == "hex"
    assert hex_cfg["forward_backend"] == "cuda_structured"


def test_embedded_vtk_disabled_for_offscreen_qt(monkeypatch):
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    enabled, reason = embedded_vtk_status()

    assert enabled is False
    assert embedded_vtk_enabled() is False
    assert "offscreen" in reason


def test_embedded_vtk_can_be_forced(monkeypatch):
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("EIT_APP_ENABLE_EMBEDDED_VTK", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    enabled, reason = embedded_vtk_status()

    assert enabled is True
    assert embedded_vtk_enabled() is True
    assert "forced" in reason


def test_embedded_vtk_disabled_on_wsl_even_when_display_is_available(monkeypatch):
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("DISPLAY", ":0")

    enabled, reason = embedded_vtk_status()

    assert enabled is False
    assert embedded_vtk_enabled() is False
    assert "WSLg" in reason or "unsafe" in reason


def test_3d_payload_stays_in_3d_widget_when_vtk_disabled(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")
    calls: list[tuple[str, str | None]] = []

    def unexpected_mpl_update(_sigma, _coords, _cells, title=None):
        raise AssertionError("3D volume data must not fall back to the 2D plot")

    def fake_3d_update(_sigma, _coords, _cells, title=None):
        calls.append(("3d", title))

    monkeypatch.setattr(slot._mpl, "update_image", unexpected_mpl_update)
    monkeypatch.setattr(slot._three_d, "update_image", fake_3d_update)

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert calls == [("3d", "Truth")]
    assert slot._stack.currentWidget() is slot._three_d
    slot.close()


def test_safe_3d_backend_renders_small_tetra(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert slot._stack.currentWidget() is slot._three_d
    assert slot._three_d._stack.currentWidget() is slot._three_d._mpl3d_host
    assert slot._three_d._last_image is not None
    assert slot._three_d._render_backend == "mpl3d"
    slot.close()


def test_safe_3d_backend_renders_hex_when_vtk_disabled(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")

    sigma, coords, cells = _hex_payload()
    slot.update_image(sigma, coords, cells, title="Hex Truth")

    assert slot._stack.currentWidget() is slot._three_d
    assert slot._three_d._stack.currentWidget() is slot._three_d._mpl3d_host
    assert slot._three_d._last_image is not None
    assert slot._three_d._last_image[3] == "Hex Truth"
    assert slot._three_d._render_backend == "mpl3d"
    slot.close()


def test_3d_payload_uses_vtk_widget_when_forced(monkeypatch):
    _get_app()
    monkeypatch.setenv("EIT_APP_ENABLE_EMBEDDED_VTK", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")
    calls: list[tuple[str, str | None]] = []

    def unexpected_mpl_update(_sigma, _coords, _cells, title=None):
        raise AssertionError("Matplotlib fallback should not run when VTK is forced")

    def fake_vtk_update(_sigma, _coords, _cells, title=None):
        calls.append(("vtk", title))

    monkeypatch.setattr(slot._mpl, "update_image", unexpected_mpl_update)
    monkeypatch.setattr(slot._three_d, "update_image", fake_vtk_update)

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert calls == [("vtk", "Truth")]
    assert slot._stack.currentWidget() is slot._three_d
    slot.close()


def test_hex_3d_payload_uses_vtk_widget_when_forced(monkeypatch):
    _get_app()
    monkeypatch.setenv("EIT_APP_ENABLE_EMBEDDED_VTK", "1")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")
    calls: list[tuple[str, tuple[int, int], str | None]] = []

    def unexpected_mpl_update(_sigma, _coords, _cells, title=None):
        raise AssertionError("Hex volume data must use the 3D VTK widget")

    def fake_vtk_update(_sigma, _coords, cells, title=None):
        calls.append(("vtk", tuple(cells.shape), title))

    monkeypatch.setattr(slot._mpl, "update_image", unexpected_mpl_update)
    monkeypatch.setattr(slot._three_d, "update_image", fake_vtk_update)

    sigma, coords, cells = _hex_payload()
    slot.update_image(sigma, coords, cells, title="Hex Truth")

    assert calls == [("vtk", (1, 8), "Hex Truth")]
    assert slot._stack.currentWidget() is slot._three_d
    slot.close()


def test_3d_widget_builds_pyvista_hex_grid():
    pv = pytest.importorskip("pyvista")
    _get_app()

    class _FakeActor:
        def __init__(self) -> None:
            self.visible = True

        def SetVisibility(self, visible):  # noqa: N802 (VTK API)
            self.visible = bool(visible)

        def GetProperty(self):  # noqa: N802 (VTK API)
            return self

        def SetOpacity(self, _opacity):  # noqa: N802 (VTK API)
            pass

    class _FakePlotter:
        def __init__(self) -> None:
            self.meshes = []
            self.render_count = 0

        def add_mesh(self, mesh, *args, **kwargs):
            self.meshes.append((mesh, kwargs))
            return _FakeActor()

        def remove_actor(self, _actor, render=False):
            pass

        def reset_camera(self):
            pass

        def render(self):
            self.render_count += 1

    widget = Conductivity3DWidget("Hex")
    fake_plotter = _FakePlotter()
    widget._plotter = fake_plotter

    sigma, coords, cells = _hex_payload()
    widget._build_scene(sigma, coords, cells)

    grid, kwargs = fake_plotter.meshes[0]
    assert grid.n_cells == 1
    assert int(grid.celltypes[0]) == int(pv.CellType.HEXAHEDRON)
    assert kwargs["preference"] == "cell"
    assert fake_plotter.render_count == 1
    widget.close()
