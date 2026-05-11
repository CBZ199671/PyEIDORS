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
from eit_app.models.forward_model_config import (  # noqa: E402
    ForwardModelConfig,
    electrode_level_fractions_for_rings,
    max_electrode_height_ratio_for_rings,
)
from eit_app.models.simulation_state import InhomogeneitySpec  # noqa: E402
from pyeidors.electrodes.layout import (  # noqa: E402
    effective_pattern_layout_for_3d_mesh,
    effective_pattern_layout_for_zigzag_3d_mesh,
)
from eit_app.ui.conductivity_3d_widget import (  # noqa: E402
    Conductivity3DWidget,
    DISPLAY_MODE_POINTS,
    DISPLAY_MODE_VOLUME,
    SUPPORTED_3D_CELL_VERTEX_COUNTS,
    _cell_inhomogeneity_mask,
    _conductivity_color_limits,
    embedded_vtk_enabled,
    embedded_vtk_status,
)
from eit_app.ui.simulation.simulation_results_widget import (  # noqa: E402
    _ConductivityViewSlot,
)
from eit_app.ui.simulation.mesh_setup_panel import MeshSetupPanel  # noqa: E402
from pyeidors.geometry.mesh3d_generator import Cylinder3DMeshConfig  # noqa: E402


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_v104_3d_highlight_ignores_near_constant_absolute_sigma_noise() -> None:
    near_constant = np.array([1.0, 1.003, 0.997, 1.004, 0.998], dtype=np.float64)
    visible_anomaly = np.array([1.0, 1.0, 1.0, 1.12, 1.0], dtype=np.float64)

    assert not np.any(_cell_inhomogeneity_mask(near_constant))
    assert np.flatnonzero(_cell_inhomogeneity_mask(visible_anomaly)).tolist() == [3]


def test_v107_3d_color_limits_do_not_amplify_near_constant_sigma_noise() -> None:
    near_constant = np.array([1.0, 1.003, 0.997, 1.004, 0.998], dtype=np.float64)
    visible_anomaly = np.array([1.0, 1.0, 1.0, 1.12, 1.0], dtype=np.float64)

    sigma_min, sigma_max = _conductivity_color_limits(near_constant)
    assert sigma_min == pytest.approx(0.98)
    assert sigma_max == pytest.approx(1.02)
    assert _conductivity_color_limits(visible_anomaly) == pytest.approx((1.0, 1.12))


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


def test_v109_paint_3d_sphere_respects_z_radius_when_sizes_disagree():
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.07],
            [0.04, 0.0, 0.0],
        ],
        dtype=float,
    )
    values = np.ones(centers.shape[0], dtype=float)
    spec = InhomogeneitySpec(
        shape="circle",
        size_x=0.2,
        size_y=0.2,
        size_z=0.05,
        conductivity=2.0,
    )

    _paint_shape(values, centers, spec, mesh_dimension=3)

    assert values.tolist() == [2.0, 1.0, 2.0]


def test_v101_paint_3d_sphere_uses_volume_fraction_for_coarse_hex_layers():
    x0, x1 = -0.04, 0.04
    y0, y1 = -0.04, 0.04
    z_levels = np.array([-0.08, -0.056, -0.024, 0.0, 0.024, 0.056, 0.08])
    cell_vertices = []
    for z0, z1 in zip(z_levels[:-1], z_levels[1:], strict=True):
        cell_vertices.append(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
    cell_vertices_arr = np.asarray(cell_vertices, dtype=float)
    centers = cell_vertices_arr.mean(axis=1)
    values = np.ones(centers.shape[0], dtype=float)
    spec = InhomogeneitySpec(shape="circle", size_x=0.063, conductivity=2.0)

    _paint_shape(
        values,
        centers,
        spec,
        mesh_dimension=3,
        cell_vertices=cell_vertices_arr,
    )

    assert np.all(values > 1.0)
    assert values[0] == pytest.approx(values[-1])
    assert values[1] == pytest.approx(values[-2])
    assert values[2] == pytest.approx(values[-3])
    assert values[0] < values[1]
    assert values[0] < values[2]
    assert values[2] > 1.5


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
    monkeypatch.setattr(
        rc, "_load_gn_difference_runner_module", lambda: fake_diff_runner
    )
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
    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + delta_sigma)
    assert np.allclose(result.measured, expected_dv)
    assert np.allclose(result.simulated, pred_diff)
    assert result.metadata["single_step_operator_space"] == "measurement"


def test_single_step_cached_limits_alpha_before_forward_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    reference = FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    target = FrameData(
        real=np.array([2.0, 4.0, 6.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    delta_sigma = np.array([-2.0, 0.1], dtype=float)
    sigma_bg = np.ones_like(delta_sigma)
    sigma_floor = 0.2
    base_meas = np.array([10.0, 20.0, 30.0], dtype=float)
    pred_diff = np.array([0.05, 0.1, 0.15], dtype=float)
    captured_sigma: list[np.ndarray] = []

    class _StubForwardModel:
        def fwd_solve(self, image):
            sigma = np.asarray(image.elem_data, dtype=float)
            captured_sigma.append(sigma.copy())
            assert np.all(np.isfinite(sigma))
            assert float(np.min(sigma)) > sigma_floor
            return SimpleNamespace(meas=base_meas + pred_diff), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": "measurement-exact",
        },
        "sigma_bg": sigma_bg,
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    def _calibration_failed(**_kwargs):
        raise RuntimeError("candidate sigma was infeasible")

    fake_diff_runner = SimpleNamespace(
        STRICT_SOLVER_BACKEND_MEASUREMENT="measurement-exact",
        _calibrate_step_size=_calibration_failed,
        _measurement_space_delta=lambda *, operator_bundle, rhs: delta_sigma,
        _solve_linear_from_bundle=lambda *_args, **_kwargs: delta_sigma,
        build_shared_context=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        rc, "_load_gn_difference_runner_module", lambda: fake_diff_runner
    )
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
                "step_size_calib": True,
                "sigma_floor": sigma_floor,
                "n_elec": 8,
                "n_rings": 2,
            },
        )
    )

    assert captured_sigma
    assert float(np.min(captured_sigma[-1])) > sigma_floor
    assert 0.0 < result.metadata["step_size_alpha"] < 0.4
    assert result.metadata["step_size_alpha_requested"] == pytest.approx(1.0)
    assert result.metadata["step_size_alpha_limited"] is True
    np.testing.assert_allclose(result.conductivity, captured_sigma[-1])


def test_single_step_cached_uses_linearized_operator_solver(
    monkeypatch: pytest.MonkeyPatch,
):
    reference = FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    target = FrameData(
        real=np.array([1.5, 2.25, 2.75], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    expected_dv = target.real - reference.real
    delta_sigma = np.array([0.2, -0.1], dtype=float)
    base_meas = np.array([10.0, 20.0, 30.0], dtype=float)
    pred_diff = np.array([0.05, -0.1, 0.15], dtype=float)
    calls = {"linearized": 0, "measurement": 0, "parameter": 0}

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
            "jacobian_representation": "linearized",
            "strict_solver_backend_effective": "dense-param",
        },
        "jacobian_representation": "linearized",
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    def _solve_linearized_delta(*, operator_bundle, rhs):
        calls["linearized"] += 1
        assert operator_bundle is ctx["operator_bundle"]
        assert np.allclose(rhs, expected_dv)
        return delta_sigma

    fake_diff_runner = SimpleNamespace(
        STRICT_SOLVER_BACKEND_MEASUREMENT="measurement-exact",
        _calibrate_step_size=lambda **_kwargs: 1.0,
        _measurement_space_delta=lambda **_kwargs: calls.__setitem__("measurement", 1),
        _solve_linear_from_bundle=lambda *_args, **_kwargs: calls.__setitem__(
            "parameter", 1
        ),
        _solve_linearized_delta=_solve_linearized_delta,
        build_shared_context=lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        rc, "_load_gn_difference_runner_module", lambda: fake_diff_runner
    )
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

    assert calls == {"linearized": 1, "measurement": 0, "parameter": 0}
    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + delta_sigma)
    assert result.metadata["single_step_operator_space"] == "linearized"


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


def test_mesh_setup_panel_exposes_2d_length_and_3d_area_geometry():
    _get_app()
    panel = MeshSetupPanel()
    try:
        panel.set_config({"mesh_dimension": 2, "radius": 1.0, "n_electrodes": 16})
        cfg_default_2d = panel.get_config()
        assert cfg_default_2d["electrode_length_m_override"] is None
        assert cfg_default_2d["electrode_coverage"] == pytest.approx(0.5)
        assert panel._electrode_length_spin.value() == pytest.approx(
            2.0 * np.pi * 0.5 / 16.0,
            abs=1.0e-6,
        )

        panel.set_config(
            {
                "mesh_dimension": 2,
                "radius": 1.0,
                "n_electrodes": 16,
                "electrode_length_m_override": 0.125,
            }
        )
        cfg_2d = panel.get_config()
        assert cfg_2d["electrode_length_m_override"] == pytest.approx(0.125)
        assert cfg_2d["electrode_coverage"] == pytest.approx(
            0.125 / (2.0 * np.pi / 16.0)
        )
        assert cfg_2d["electrode_area_m2_override"] is None
        assert panel._electrode_length_spin.isEnabled()
        assert not panel._electrode_area_spin.isEnabled()

        panel.set_config(
            {
                "mesh_dimension": 3,
                "radius": 0.18,
                "height": 0.16,
                "n_electrodes": 8,
                "n_rings": 2,
                "electrode_area_m2_override": 0.003,
            }
        )
        cfg_3d = panel.get_config()
        expected_length = 2.0 * np.pi * 0.18 * 0.5 / 8.0
        assert cfg_3d["electrode_length_m_override"] == pytest.approx(expected_length)
        assert cfg_3d["electrode_area_m2_override"] == pytest.approx(0.003)
        assert cfg_3d["electrode_height_ratio"] == pytest.approx(
            0.003 / (expected_length * 0.16)
        )
        assert not panel._electrode_length_spin.isEnabled()
        assert panel._electrode_area_spin.isEnabled()
    finally:
        panel.close()


def test_mesh_setup_panel_clamps_3d_area_to_non_overlapping_ring_windows():
    _get_app()
    panel = MeshSetupPanel()
    try:
        panel.set_config(
            {
                "mesh_dimension": 3,
                "radius": 0.18,
                "height": 0.16,
                "n_electrodes": 8,
                "n_rings": 8,
                "electrode_layout": "ring_major",
                "electrode_area_m2_override": 0.003,
            }
        )

        cfg = panel.get_config()

        levels = electrode_level_fractions_for_rings(8)
        max_ratio = max_electrode_height_ratio_for_rings(8)
        expected_length = 2.0 * np.pi * 0.18 * 0.5 / 8.0
        assert cfg["electrode_height_ratio"] <= max_ratio
        assert cfg["electrode_height_ratio"] == pytest.approx(max_ratio, rel=1.0e-5)
        assert cfg["electrode_area_m2_override"] == pytest.approx(
            expected_length * 0.16 * cfg["electrode_height_ratio"]
        )
        assert panel._electrode_area_spin.value() == pytest.approx(
            cfg["electrode_area_m2_override"]
        )
        Cylinder3DMeshConfig(
            radius=cfg["radius"],
            height=cfg["height"],
            electrode_height_ratio=cfg["electrode_height_ratio"],
            electrode_level_fractions=levels,
        )
    finally:
        panel.close()


def test_gpu_forward_runtime_keeps_tetra_and_hex_distinct(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setattr(
        "eit_app.controllers.forward_solver_controller.probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": False,
        },
    )

    tetra = _resolve_forward_runtime(
        ForwardModelConfig(mesh_dimension=3, mesh_family="tetra")
    )
    assert tetra["mesh_family"] == "tetra"
    assert tetra["forward_backend"] == "dolfinx"
    assert tetra["petsc_device"] == "cuda"
    assert tetra["device"] == "cuda"
    assert tetra["acceleration_profile"] == "gpu3d"
    assert tetra["forward_solver_preset"] == "spd_gamg"
    assert tetra["petsc_amgx_available"] is False
    assert tetra["forward_mat_solve"] == "off"
    assert (
        tetra["forward_mat_solve_policy_reason"] == "cuda_spd_gamg_matsolve_disabled_b6"
    )

    hex_cfg = _resolve_forward_runtime(
        ForwardModelConfig(mesh_dimension=3, mesh_family="hex")
    )
    assert hex_cfg["mesh_family"] == "hex"
    assert hex_cfg["forward_backend"] == "cuda_structured"
    assert hex_cfg["petsc_device"] == "cuda"


def test_v83_gpu_forward_runtime_keeps_2d_auto_petsc_on_cpu(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")

    runtime = _resolve_forward_runtime(
        ForwardModelConfig(mesh_dimension=2, petsc_device="auto")
    )

    assert runtime["acceleration_profile"] == "default"
    assert runtime["forward_backend"] == "dolfinx"
    assert runtime["petsc_device"] == "cpu"
    assert runtime["device"] == "auto"
    assert runtime["forward_mat_solve"] == "off"


def test_gpu_reconstruction_runtime_keeps_tetra_and_hex_distinct(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")
    monkeypatch.setattr(
        "eit_app.controllers.reconstruction_controller.probe_petsc_cuda_runtime",
        lambda: {
            "petsc_cuda": True,
            "petsc_hypre": True,
            "petsc_amgx": False,
        },
    )

    tetra = _resolve_reconstruction_runtime(
        {"mesh_family": "tetra", "forward_backend": "cuda_structured"},
        mesh_dim=3,
    )
    assert tetra["mesh_family"] == "tetra"
    assert tetra["forward_backend"] == "dolfinx"
    assert tetra["petsc_device"] == "cuda"
    assert tetra["device"] == "cuda"
    assert tetra["acceleration_profile"] == "gpu3d"
    assert tetra["forward_solver_preset"] == "spd_gamg"
    assert (
        tetra["forward_solver_policy_reason"]
        == "amgx_unavailable_downgraded_to_spd_gamg"
    )
    assert tetra["petsc_amgx_available"] is False
    assert tetra["forward_mat_solve"] == "off"
    assert (
        tetra["forward_mat_solve_policy_reason"] == "cuda_spd_gamg_matsolve_disabled_b6"
    )

    requested_amgx = _resolve_reconstruction_runtime(
        {"mesh_family": "tetra", "forward_solver_preset": "cuda_amgx"},
        mesh_dim=3,
    )
    assert requested_amgx["forward_solver_preset_requested"] == "cuda_amgx"
    assert requested_amgx["forward_solver_preset"] == "spd_gamg"
    assert requested_amgx["forward_mat_solve"] == "off"

    explicit_matsolve = _resolve_reconstruction_runtime(
        {
            "mesh_family": "tetra",
            "forward_solver_preset": "spd_gamg",
            "forward_mat_solve": "on",
        },
        mesh_dim=3,
    )
    assert explicit_matsolve["forward_mat_solve_requested"] == "on"
    assert explicit_matsolve["forward_mat_solve"] == "on"
    assert explicit_matsolve["forward_mat_solve_policy_reason"] == ""

    hex_cfg = _resolve_reconstruction_runtime({"mesh_family": "hex"}, mesh_dim=3)
    assert hex_cfg["mesh_family"] == "hex"
    assert hex_cfg["forward_backend"] == "cuda_structured"


def test_v83_gpu_reconstruction_runtime_keeps_2d_auto_petsc_on_cpu(monkeypatch):
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")

    runtime = _resolve_reconstruction_runtime({"petsc_device": "auto"}, mesh_dim=2)

    assert runtime["acceleration_profile"] == "default"
    assert runtime["forward_backend"] == "dolfinx"
    assert runtime["petsc_device"] == "cpu"
    assert runtime["device"] == "auto"
    assert runtime["forward_mat_solve"] == "off"


def test_single_step_solver_diagnostics_exposes_runtime_summary():
    diagnostics = rc._single_step_cached_solver_diagnostics(
        {
            "mesh_family": "tetra",
            "forward_backend": "dolfinx",
            "petsc_device": "cuda",
            "petsc_backend_info": {
                "forward_backend_effective": "dolfinx",
                "solver_preset": "spd_gamg",
                "petsc_amgx_available": False,
                "petsc_device_requested": "cuda",
                "petsc_device_effective": "cuda",
            },
            "device_requested": "cuda",
            "device_effective": "cuda",
            "torch_device": "cuda",
            "jacobian_representation": "linearized",
            "mesh_cache_hit": True,
            "mesh_cache_layer": "disk",
            "mesh_cache_name": "mesh3d_demo",
            "cache_lookups": {
                "base_meas": {"hit": True, "layer": "disk"},
                "operator_A": {"hit": False, "layer": "process"},
                "operator_rom_reduced_rm": {"hit": False, "layer": "disabled"},
            },
            "cache_build_seconds": {},
            "cache_miss_reasons": {},
            "cache_manager": None,
        },
        strict_backend="measurement-exact",
    )

    runtime = diagnostics["runtime"]
    assert runtime["mesh_family"] == "tetra"
    assert runtime["forward_backend_effective"] == "dolfinx"
    assert runtime["forward_solver_preset"] == "spd_gamg"
    assert runtime["petsc_amgx_available"] is False
    assert runtime["petsc_device_effective"] == "cuda"
    assert runtime["torch_device"] == "cuda"
    assert runtime["jacobian_representation"] == "linearized"
    assert runtime["mesh_cache_hit"] is True
    assert runtime["cache_hit"] is False
    assert runtime["cache_hits"] == {"base_meas": True, "operator_A": False}


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
    assert widget._offscreen_mesh_actor.GetProperty().GetOpacity() == pytest.approx(
        0.30
    )

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


def test_pyvista_offscreen_backend_renders_point_cloud_mode(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget = Conductivity3DWidget("Conductivity")
    widget.set_display_mode(DISPLAY_MODE_POINTS)

    sigma, coords, cells = _inhomogeneous_tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")

    assert widget._render_backend == "pyvista_offscreen"
    assert widget._stack.currentWidget() is widget._offscreen_host
    assert widget._offscreen_label.pixmap() is not None
    assert widget._offscreen_mesh_actor is not None
    assert widget._offscreen_highlight_actor is not None
    widget.close()


def test_pyvista_offscreen_drag_defaults_to_full_resolution_60fps(monkeypatch):
    _get_app()
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("EIT_APP_3D_DRAG_FPS", raising=False)
    monkeypatch.delenv("EIT_APP_3D_DRAG_RENDER_SCALE", raising=False)

    class FakePlotter:
        def __init__(self) -> None:
            self._window_size = (0, 0)
            self.window_size_sets = 0
            self.screenshot_sizes: list[tuple[int, int]] = []

        @property
        def window_size(self) -> tuple[int, int]:
            return self._window_size

        @window_size.setter
        def window_size(self, value: tuple[int, int]) -> None:
            self.window_size_sets += 1
            self._window_size = value

        def render(self) -> None:
            pass

        def screenshot(self, *, return_img: bool):  # noqa: ANN001
            assert return_img is True
            width, height = self.window_size
            self.screenshot_sizes.append((width, height))
            return np.zeros((height, width, 3), dtype=np.uint8)

    widget = Conductivity3DWidget("Conductivity")
    widget._offscreen_label.resize(800, 600)
    plotter = FakePlotter()
    widget._offscreen_plotter = plotter
    widget._render_backend = "pyvista_offscreen"

    widget._is_dragging_offscreen = False
    widget._refresh_offscreen_pixmap()
    idle_pixmap = widget._offscreen_label.pixmap()
    assert idle_pixmap is not None
    idle_logical = (
        idle_pixmap.width() / idle_pixmap.devicePixelRatioF(),
        idle_pixmap.height() / idle_pixmap.devicePixelRatioF(),
    )

    widget._is_dragging_offscreen = True
    widget._refresh_offscreen_pixmap()
    drag_pixmap = widget._offscreen_label.pixmap()
    assert drag_pixmap is not None
    drag_logical = (
        drag_pixmap.width() / drag_pixmap.devicePixelRatioF(),
        drag_pixmap.height() / drag_pixmap.devicePixelRatioF(),
    )

    assert widget._offscreen_render_timer.interval() == 17
    assert widget._offscreen_drag_render_scale == pytest.approx(1.0)
    assert plotter.screenshot_sizes[1] == plotter.screenshot_sizes[0]
    assert plotter.window_size_sets == 1
    assert drag_pixmap.width() == idle_pixmap.width()
    assert drag_pixmap.height() == idle_pixmap.height()
    assert drag_logical == pytest.approx(idle_logical)
    assert drag_logical == pytest.approx((800.0, 600.0))

    widget.close()


def test_pyvista_offscreen_drag_scale_can_be_reduced_without_size_jitter(monkeypatch):
    _get_app()
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("EIT_APP_3D_DRAG_FPS", "30")
    monkeypatch.setenv("EIT_APP_3D_DRAG_RENDER_SCALE", "0.5")

    class FakePlotter:
        def __init__(self) -> None:
            self._window_size = (0, 0)
            self.window_size_sets = 0
            self.screenshot_sizes: list[tuple[int, int]] = []

        @property
        def window_size(self) -> tuple[int, int]:
            return self._window_size

        @window_size.setter
        def window_size(self, value: tuple[int, int]) -> None:
            self.window_size_sets += 1
            self._window_size = value

        def render(self) -> None:
            pass

        def screenshot(self, *, return_img: bool):  # noqa: ANN001
            assert return_img is True
            width, height = self.window_size
            self.screenshot_sizes.append((width, height))
            return np.zeros((height, width, 3), dtype=np.uint8)

    widget = Conductivity3DWidget("Conductivity")
    widget._offscreen_label.resize(800, 600)
    plotter = FakePlotter()
    widget._offscreen_plotter = plotter
    widget._render_backend = "pyvista_offscreen"

    widget._is_dragging_offscreen = False
    widget._refresh_offscreen_pixmap()
    idle_pixmap = widget._offscreen_label.pixmap()
    assert idle_pixmap is not None
    idle_logical = (
        idle_pixmap.width() / idle_pixmap.devicePixelRatioF(),
        idle_pixmap.height() / idle_pixmap.devicePixelRatioF(),
    )

    widget._is_dragging_offscreen = True
    widget._refresh_offscreen_pixmap()
    drag_pixmap = widget._offscreen_label.pixmap()
    assert drag_pixmap is not None
    drag_logical = (
        drag_pixmap.width() / drag_pixmap.devicePixelRatioF(),
        drag_pixmap.height() / drag_pixmap.devicePixelRatioF(),
    )

    assert widget._offscreen_render_timer.interval() == 33
    assert widget._offscreen_drag_render_scale == pytest.approx(0.5)
    assert plotter.screenshot_sizes[1][0] < plotter.screenshot_sizes[0][0]
    assert plotter.window_size_sets == 2
    assert drag_pixmap.width() == idle_pixmap.width()
    assert drag_pixmap.height() == idle_pixmap.height()
    assert drag_logical == pytest.approx(idle_logical)

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


def test_3d_widget_point_cloud_mode_builds_cell_center_polydata():
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
    widget.set_display_mode(DISPLAY_MODE_POINTS)
    fake_plotter = _FakePlotter()
    widget._plotter = fake_plotter

    sigma, coords, cells = _hex_payload()
    widget._build_scene(sigma, coords, cells)

    cloud, kwargs = fake_plotter.meshes[0]
    assert isinstance(cloud, pv.PolyData)
    assert cloud.n_points == cells.shape[0]
    np.testing.assert_allclose(cloud.points[0], coords[cells[0], :3].mean(axis=0))
    assert kwargs["scalars"] == "sigma"
    assert kwargs["render_points_as_spheres"] is True
    assert kwargs["point_size"] >= 4.0
    assert fake_plotter.render_count == 1
    widget.close()


def test_3d_display_mode_switch_rerenders_cached_payload(monkeypatch):
    _get_app()
    monkeypatch.delenv("EIT_APP_ENABLE_EMBEDDED_VTK", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    widget = Conductivity3DWidget("Conductivity")
    calls: list[str] = []

    def fake_offscreen_render(_sigma, _coords, _cells):
        calls.append(widget.display_mode())
        widget._render_backend = "pyvista_offscreen"
        return True

    monkeypatch.setattr(
        widget,
        "_render_pyvista_offscreen_scene",
        fake_offscreen_render,
    )

    sigma, coords, cells = _tetra_payload()
    widget.update_image(sigma, coords, cells, title="Truth")
    widget.set_display_mode(DISPLAY_MODE_POINTS)

    assert calls == [DISPLAY_MODE_VOLUME, DISPLAY_MODE_POINTS]
    assert widget._points_mode_btn.isChecked()
    assert not widget._volume_mode_btn.isChecked()
    widget.close()


def test_matplotlib_point_cloud_mode_renders_cell_center_collection():
    _get_app()
    widget = Conductivity3DWidget("Conductivity")
    widget.set_display_mode(DISPLAY_MODE_POINTS)

    sigma, coords, cells = _inhomogeneous_tetra_payload()
    widget._render_matplotlib_scene(sigma, coords, cells)

    assert widget._render_backend == "mpl3d"
    assert widget._stack.currentWidget() is widget._mpl3d_host
    assert widget._mpl3d_mesh_collection is not None
    assert widget._mpl3d_mesh_facecolors is None
    assert widget._mpl3d_highlight_collection is not None
    widget.close()
