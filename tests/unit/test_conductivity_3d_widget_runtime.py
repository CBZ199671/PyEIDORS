from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from eit_app.controllers.forward_solver_controller import (  # noqa: E402
    _paint_shape,
    _resolve_forward_runtime,
)
from eit_app.controllers import reconstruction_controller as rc  # noqa: E402
from eit_app.controllers.reconstruction_controller import (  # noqa: E402
    _resolve_reconstruction_runtime,
)
from eit_app.models.frame_model import FrameData  # noqa: E402
from eit_app.models.forward_model_config import ForwardModelConfig  # noqa: E402
from eit_app.models.simulation_state import InhomogeneitySpec  # noqa: E402
from pyeidors.electrodes.layout import (  # noqa: E402
    effective_pattern_layout_for_3d_mesh,
    effective_pattern_layout_for_zigzag_3d_mesh,
)
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


def _inhomogeneous_tetra_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    sigma = np.array([1.0, 2.0], dtype=float)
    return sigma, coords, cells


def _frame(index: int) -> FrameData:
    return FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=index,
    )


def test_supported_3d_cell_types_cover_tetra_and_hex():
    assert {4, 8}.issubset(SUPPORTED_3D_CELL_VERTEX_COUNTS)


def test_zigzag_3d_mesh_uses_total_electrode_pattern_layout():
    assert effective_pattern_layout_for_3d_mesh(
        mesh_tdim=3,
        n_elec=8,
        n_rings=2,
        electrode_layout="ring_major",
    ) == (8, 2)
    assert effective_pattern_layout_for_zigzag_3d_mesh(
        mesh_tdim=3,
        n_elec=8,
        n_rings=2,
    ) == (16, 1)
    assert effective_pattern_layout_for_zigzag_3d_mesh(
        mesh_tdim=2,
        n_elec=8,
        n_rings=2,
    ) == (8, 2)


def test_paint_circle_is_area_in_2d_even_with_2d_centers():
    centers = np.array([[0.0, 0.0], [0.24, 0.0], [0.3, 0.0]], dtype=float)
    values = np.ones(centers.shape[0], dtype=float)
    spec = InhomogeneitySpec(shape="circle", size_x=0.25, conductivity=2.0)

    _paint_shape(values, centers, spec, mesh_dimension=2)

    assert values.tolist() == [2.0, 2.0, 1.0]


def test_paint_circle_is_sphere_in_3d_not_vertical_cylinder():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3],
            [0.1, 0.0, 0.1],
        ],
        dtype=float,
    )
    values = np.ones(centers.shape[0], dtype=float)
    spec = InhomogeneitySpec(shape="circle", size_x=0.2, conductivity=2.0)

    _paint_shape(values, centers, spec, mesh_dimension=3)

    assert values.tolist() == [2.0, 1.0, 2.0]


def test_paint_ellipsoid_and_box_use_z_extent_in_3d():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.16],
            [0.19, 0.0, 0.0],
        ],
        dtype=float,
    )

    ellipsoid_values = np.ones(centers.shape[0], dtype=float)
    ellipsoid = InhomogeneitySpec(
        shape="ellipse",
        size_x=0.2,
        size_y=0.2,
        size_z=0.1,
        conductivity=2.0,
    )
    _paint_shape(ellipsoid_values, centers, ellipsoid, mesh_dimension=3)
    assert ellipsoid_values.tolist() == [2.0, 1.0, 2.0]

    box_values = np.ones(centers.shape[0], dtype=float)
    box = InhomogeneitySpec(
        shape="rectangle",
        size_x=0.2,
        size_y=0.2,
        size_z=0.1,
        conductivity=3.0,
    )
    _paint_shape(box_values, centers, box, mesh_dimension=3)
    assert box_values.tolist() == [3.0, 1.0, 3.0]


def test_single_step_cached_promotes_3d_line_current_density_to_total_current():
    request = rc.ReconstructionRequest(
        reference_frame=_frame(0),
        target_frame=_frame(1),
        mesh_dimension=3,
        mesh_refinement=0.1,
        metadata={
            "mesh_dimension": 3,
            "n_elec": 8,
            "n_rings": 2,
            "drive_mode": "line_current_density",
            "drive_value": 1.0,
            "mesh_size": 0.1,
            "radius": 0.18,
            "height": 0.16,
            "mesh_family": "hex",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.meta["drive_mode"] == "total_current"
    assert "line_current_density" not in runtime.cache_key


def test_single_step_cached_uses_measurement_space_when_operator_shape_matches(
    monkeypatch: pytest.MonkeyPatch,
):
    reference = FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    target = FrameData(
        real=np.array([1.5, 2.5, 4.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    expected_dv = target.real - reference.real
    delta_sigma = np.array([0.25, -0.5], dtype=float)
    base_meas = np.array([10.0, 20.0, 30.0], dtype=float)
    pred_diff = np.array([0.1, 0.2, 0.3], dtype=float)
    calls = {"measurement": 0, "parameter": 0}

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=base_meas + pred_diff), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "mode": "strict",
            "strict_solver_backend_effective": "dense-param",
            "A": np.eye(expected_dv.size, dtype=float),
            "Jt": np.ones((delta_sigma.size, expected_dv.size), dtype=float),
            "inv_reg_diag": np.ones(delta_sigma.size, dtype=float),
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    def _measurement_space_delta(*, operator_bundle, rhs):
        calls["measurement"] += 1
        assert operator_bundle is ctx["operator_bundle"]
        assert np.allclose(rhs, expected_dv)
        return delta_sigma

    def _solve_linear_from_bundle(_operator_bundle, _rhs):
        calls["parameter"] += 1
        raise AssertionError("parameter-space solve should not be used")

    fake_diff_runner = SimpleNamespace(
        STRICT_SOLVER_BACKEND_MEASUREMENT="measurement-exact",
        _calibrate_step_size=lambda **_kwargs: 1.0,
        _measurement_space_delta=_measurement_space_delta,
        _solve_linear_from_bundle=_solve_linear_from_bundle,
        build_shared_context=lambda **_kwargs: None,
    )
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", lambda: fake_diff_runner)
    monkeypatch.setattr(
        rc,
        "_ensure_single_step_cached_context",
        lambda _runtime, *, emit, build_shared_context: ctx,
    )

    result = rc._run_single_step_cached_request(
        rc.ReconstructionRequest(
            reference_frame=reference,
            target_frame=target,
            mesh_dimension=3,
            metadata={
                "reconstruction_runtime": "single_step_cached",
                "step_size_calib": False,
                "n_elec": 8,
                "n_rings": 2,
            },
        )
    )

    assert calls == {"measurement": 1, "parameter": 0}
    assert np.allclose(result.conductivity, delta_sigma)
    assert np.allclose(result.measured, expected_dv)
    assert np.allclose(result.simulated, pred_diff)
    assert result.metadata["single_step_operator_space"] == "measurement"


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


def test_embedded_vtk_enabled_on_wsl_when_qt_uses_xcb(monkeypatch):
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "xcb")
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("DISPLAY", ":0")

    enabled, reason = embedded_vtk_status()

    assert enabled is True
    assert embedded_vtk_enabled() is True
    assert "XCB" in reason or "compatible" in reason


def test_embedded_vtk_disabled_on_wsl_without_xcb(monkeypatch):
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("EIT_APP_DISABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    monkeypatch.setenv("DISPLAY", ":0")

    enabled, reason = embedded_vtk_status()

    assert enabled is False
    assert embedded_vtk_enabled() is False
    assert "xcb" in reason


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


def test_pyvista_offscreen_backend_renders_small_tetra(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")

    sigma, coords, cells = _tetra_payload()
    slot.update_image(sigma, coords, cells, title="Truth")

    assert slot._stack.currentWidget() is slot._three_d
    assert slot._three_d._stack.currentWidget() is slot._three_d._offscreen_host
    assert slot._three_d._last_image is not None
    assert slot._three_d._render_backend == "pyvista_offscreen"
    assert slot._three_d._offscreen_label.pixmap() is not None
    slot.close()


def test_pyvista_offscreen_backend_renders_hex_when_vtk_disabled(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    slot = _ConductivityViewSlot("Conductivity")

    sigma, coords, cells = _hex_payload()
    slot.update_image(sigma, coords, cells, title="Hex Truth")

    assert slot._stack.currentWidget() is slot._three_d
    assert slot._three_d._stack.currentWidget() is slot._three_d._offscreen_host
    assert slot._three_d._last_image is not None
    assert slot._three_d._last_image[3] == "Hex Truth"
    assert slot._three_d._render_backend == "pyvista_offscreen"
    assert slot._three_d._offscreen_label.pixmap() is not None
    slot.close()


def test_pyvista_offscreen_controls_keep_rendered_canvas(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget = Conductivity3DWidget("Conductivity")

    sigma, coords, cells = _inhomogeneous_tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")
    initial_pixmap = widget._offscreen_label.pixmap()
    assert widget._render_backend == "pyvista_offscreen"
    assert initial_pixmap is not None

    widget._opacity_slider.setValue(30)
    QApplication.processEvents()
    assert widget._offscreen_label.pixmap() is not None
    assert widget._offscreen_mesh_actor is not None
    assert widget._offscreen_mesh_actor.GetProperty().GetOpacity() == pytest.approx(0.30)

    assert widget._offscreen_highlight_actor is not None
    widget._highlight_check.setChecked(False)
    QApplication.processEvents()
    assert widget._offscreen_label.pixmap() is not None
    assert widget._offscreen_highlight_actor.GetVisibility() == 0

    widget._wire_check.setChecked(False)
    QApplication.processEvents()
    assert widget._offscreen_label.pixmap() is not None
    assert widget._offscreen_wire_actor is not None
    assert widget._offscreen_wire_actor.GetVisibility() == 0

    widget._reset_btn.click()
    QApplication.processEvents()
    assert widget._offscreen_label.pixmap() is not None
    assert widget._stack.currentWidget() is widget._offscreen_host
    widget.close()


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
