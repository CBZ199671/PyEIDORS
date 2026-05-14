from __future__ import annotations

import os
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from eit_app.controllers import reconstruction_controller as rc
from eit_app.models.frame_model import FrameData
from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget
from eit_app.ui.simulation.metrics_panel import MetricsPanel

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_reconstruction_widget_can_replace_colorbar_repeatedly() -> None:
    _get_app()
    widget = ReconstructionWidget()
    result = SimpleNamespace(
        error_msg=None,
        conductivity=np.array([1.0], dtype=float),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=int),
        measured=None,
        simulated=None,
    )

    widget.update_reconstruction(result)
    widget.update_reconstruction(result)
    widget.clear()
    widget.clear()


def test_metrics_panel_compares_values_by_geometry_not_cell_order() -> None:
    _get_app()
    panel = MetricsPanel()
    node_coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    truth_cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    recon_cells = np.array([[1, 3, 2], [0, 1, 2]], dtype=int)

    panel.update_metrics(
        np.array([1.0, 2.0], dtype=float),
        np.array([2.0, 1.0], dtype=float),
        ground_truth_node_coords=node_coords,
        ground_truth_cell_connectivity=truth_cells,
        reconstructed_node_coords=node_coords,
        reconstructed_cell_connectivity=recon_cells,
    )

    assert panel._l2_label.text() == "0.0000"
    assert panel._corr_label.text() == "1.0000"
    assert panel._rmse_label.text() == "0.000000"


def test_mesh_loader_default_mesh_skips_incompatible_3d_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.geometry.mesh_loader import MeshLoader

    (tmp_path / "mesh3d_first.msh").write_text("3d", encoding="utf-8")
    (tmp_path / "mesh_2d_second.msh").write_text("2d", encoding="utf-8")

    loader = MeshLoader(mesh_dir=str(tmp_path), gdim=2)
    sentinel = object()

    def _fake_load_mesh(name: str):
        if name == "mesh3d_first":
            raise ValueError(
                "Topological dimension cannot be larger than geometric dimension."
            )
        if name == "mesh_2d_second":
            return sentinel
        raise AssertionError(f"Unexpected mesh candidate: {name}")

    monkeypatch.setattr(loader, "load_mesh", _fake_load_mesh)

    assert loader.get_default_mesh() is sentinel


def _make_frame(index: int) -> FrameData:
    return FrameData(
        real=np.array([1.0, 2.0, 3.0], dtype=float),
        imag=np.array([0.1, 0.2, 0.3], dtype=float),
        timestamp=0.0,
        frame_index=index,
    )


def test_effective_refinement_accepts_simulation_mesh_size_without_inflation() -> None:
    assert rc._compute_effective_refinement(1.0, 0.1) == 5
    assert rc._compute_effective_refinement(1.0, 10.0) == 20
    assert rc._compute_effective_refinement(1.0, 10.0, mesh_size=0.1) == 5


def test_v106_default_3d_rm_inverse_mesh_size_stays_coarse() -> None:
    size = rc.default_rm_inverse_mesh_size(0.1, 0.18, mesh_dimension=3)

    assert size == pytest.approx(0.1)
    assert rc._compute_effective_refinement(0.18, 0.1, mesh_size=size) == 2
    assert rc.default_rm_inverse_mesh_size(
        0.02, 0.18, mesh_dimension=3
    ) == pytest.approx(0.06)


def test_single_step_cached_runtime_uses_3d_multiring_fast_defaults() -> None:
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        mesh_dimension=3,
        mesh_refinement=0.1,
        metadata={
            "mesh_dimension": 3,
            "mesh_size": 0.1,
            "n_elec": 8,
            "n_rings": 2,
            "drive_mode": "line_current_density",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert rc._total_electrodes_from_meta(runtime.meta) == 16
    assert runtime.meta["drive_mode"] == "total_current"
    assert runtime.meta["solver_mode"] == "fast"
    assert runtime.meta["forward_mat_solve"] == "auto"
    assert runtime.meta["mesh_family"] == "tetra"
    assert runtime.meta["jacobian_representation"] == "linearized"
    assert runtime.refinement == 5


def test_single_step_cached_runtime_keeps_large_3d_auto_on_dense_jacobian() -> None:
    large_ref = FrameData(
        real=np.ones(5936, dtype=float),
        imag=np.zeros(5936, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    large_tgt = FrameData(
        real=np.ones(5936, dtype=float) * 1.001,
        imag=np.zeros(5936, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    request = rc.ReconstructionRequest(
        reference_frame=large_ref,
        target_frame=large_tgt,
        mesh_dimension=3,
        mesh_refinement=0.1,
        metadata={
            "mesh_dimension": 3,
            "mesh_size": 0.1,
            "n_elec": 16,
            "n_rings": 3,
            "jacobian_representation": "auto",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.meta["solver_mode"] == "fast"
    assert runtime.meta["jacobian_representation"] == "dense"
    assert runtime.meta["jacobian_representation_reason"] == "auto_dense_large_or_non3d"


def test_single_step_cached_runtime_uses_request_alpha_when_lambda_is_absent() -> None:
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        regularization_alpha=0.75,
        metadata={"reconstruction_runtime": "single_step_cached"},
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.lam == pytest.approx(0.75)
    assert runtime.meta["difference_lambda"] == pytest.approx(0.75)
    assert runtime.meta["jacobian_representation"] == "dense"


def test_single_step_cached_runtime_prefers_explicit_difference_lambda() -> None:
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        regularization_alpha=0.75,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_lambda": 0.02,
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)

    assert runtime.lam == pytest.approx(0.02)


def test_single_step_cached_runtime_keys_semantics_with_version_fallback() -> None:
    default_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )
    semantic_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "single_step_projection_math_convention": "test-projection-v2",
        },
    )
    version_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "single_step_algorithm_version": "test-local-version",
        },
    )

    default_runtime = rc._prepare_single_step_cached_runtime(default_request)
    semantic_runtime = rc._prepare_single_step_cached_runtime(semantic_request)
    version_runtime = rc._prepare_single_step_cached_runtime(version_request)
    semantic_signature = default_runtime.cache_key[0]

    assert isinstance(semantic_signature, tuple)
    assert default_runtime.meta["single_step_jacobian_calculator"] in semantic_signature
    assert (
        default_runtime.meta["single_step_jacobian_math_convention"]
        in semantic_signature
    )
    assert (
        default_runtime.meta["single_step_projection_math_convention"]
        in semantic_signature
    )
    assert (
        default_runtime.meta["single_step_operator_math_convention"]
        in semantic_signature
    )
    assert (
        default_runtime.meta["single_step_algorithm_version"]
        in default_runtime.cache_key
    )
    assert default_runtime.cache_key != semantic_runtime.cache_key
    assert default_runtime.cache_key != version_runtime.cache_key


def test_one_step_rm_signature_rejects_stale_normalized_jacobian_semantics() -> None:
    from scripts.common import gn_difference_runner

    assert (
        gn_difference_runner.SINGLE_STEP_JACOBIAN_MATH_CONVENTION
        == rc._SINGLE_STEP_JACOBIAN_MATH_CONVENTION
    )
    assert (
        gn_difference_runner.SINGLE_STEP_PROJECTION_MATH_CONVENTION
        == rc._SINGLE_STEP_PROJECTION_MATH_CONVENTION
    )
    assert (
        gn_difference_runner.SINGLE_STEP_ALGORITHM_VERSION
        == rc._SINGLE_STEP_CACHED_ALGORITHM_VERSION
    )

    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "simulation_inverse_route": "noser_rm",
        "rm_auto_build": True,
        "mesh_size": 0.1,
        "rm_inverse_mesh_size": 0.1,
        "difference_mode": "normalized",
        "difference_orientation": "target_minus_reference",
        "n_elec": 16,
        "radius": 1.0,
    }
    current_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata=base_meta,
    )
    stale_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "single_step_jacobian_math_convention": (
                "eidors_adapter_difference_dv_dsigma_v3"
            ),
            "single_step_projection_math_convention": (
                "difference_projection_weights_v2"
            ),
            "single_step_algorithm_version": "eidors_noser_single_step_v4",
            "one_step_rm_jacobian_build_convention": (
                "dense_eidors_adapter_jacobian_v2"
            ),
            "one_step_rm_algorithm_version": (
                "one_step_rm_auto_build_dense_jacobian_v6"
            ),
            "one_step_rm_content_contract": (
                "one_step_rm_hdf5_dense_fit_jacobian_contract_v0"
            ),
            "single_step_context_cache_scope": "both",
        },
    )

    current_runtime = rc._prepare_single_step_cached_runtime(current_request)
    stale_runtime = rc._prepare_single_step_cached_runtime(stale_request)
    current_signature, current_payload = rc._planned_one_step_rm_signature(
        current_request,
        current_runtime,
    )
    stale_signature, stale_payload = rc._planned_one_step_rm_signature(
        stale_request,
        stale_runtime,
    )

    assert current_runtime.cache_key != stale_runtime.cache_key
    assert current_signature != stale_signature
    assert current_runtime.meta["single_step_context_cache_scope"] == "process"
    assert (
        current_payload["hyperparameters"]["rm_jacobian_math_convention"]
        == rc._SINGLE_STEP_JACOBIAN_MATH_CONVENTION
    )
    assert (
        current_payload["hyperparameters"]["rm_projection_math_convention"]
        == rc._SINGLE_STEP_PROJECTION_MATH_CONVENTION
    )
    assert (
        current_payload["hyperparameters"]["rm_jacobian_build_convention"]
        == rc._ONE_STEP_RM_JACOBIAN_BUILD_CONVENTION
    )
    assert (
        current_payload["hyperparameters"]["rm_algorithm_version"]
        == rc._ONE_STEP_RM_ALGORITHM_VERSION
    )
    assert (
        current_payload["hyperparameters"]["rm_content_contract"]
        == rc._ONE_STEP_RM_CONTENT_CONTRACT
    )
    assert (
        current_payload["hyperparameters"]["rm_jacobian_source_cache_scope"]
        == "process"
    )
    assert (
        stale_payload["hyperparameters"]["rm_content_contract"]
        != current_payload["hyperparameters"]["rm_content_contract"]
    )
    assert stale_payload["hyperparameters"]["rm_jacobian_source_cache_scope"] == "both"
    assert stale_payload["difference_mode"] == "normalized"


def test_noser_rm_signature_ignores_device_backend_storage_axes() -> None:
    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "simulation_inverse_route": "noser_rm",
        "rm_auto_build": True,
        "mesh_size": 0.25,
        "rm_inverse_mesh_size": 0.25,
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "n_elec": 16,
        "radius": 1.0,
    }
    cpu_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "device": "cpu",
            "petsc_device": "cpu",
            "forward_backend": "dolfinx",
        },
    )
    cuda_request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "device": "cuda",
            "petsc_device": "cuda",
            "forward_backend": "cuda_structured",
        },
    )

    cpu_runtime = rc._prepare_single_step_cached_runtime(cpu_request)
    cuda_runtime = rc._prepare_single_step_cached_runtime(cuda_request)
    assert cpu_runtime.cache_key != cuda_runtime.cache_key

    cpu_signature, _ = rc._planned_noser_rm_signature(cpu_request, cpu_runtime)
    cuda_signature, _ = rc._planned_noser_rm_signature(cuda_request, cuda_runtime)
    assert cpu_signature == cuda_signature


def test_one_step_rm_signature_tracks_effective_measurement_count() -> None:
    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "simulation_inverse_route": "laplace_rm",
        "rm_auto_build": True,
        "mesh_size": 0.25,
        "rm_inverse_mesh_size": 0.25,
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "n_elec": 16,
        "n_rings": 3,
        "radius": 0.18,
        "height": 0.16,
        "rm_regularization": "laplace",
    }
    ref_2160 = FrameData(
        real=np.ones(2160, dtype=float),
        imag=np.zeros(2160, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    tgt_2160 = FrameData(
        real=np.ones(2160, dtype=float) * 1.01,
        imag=np.zeros(2160, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    ref_5936 = FrameData(
        real=np.ones(5936, dtype=float),
        imag=np.zeros(5936, dtype=float),
        timestamp=0.0,
        frame_index=0,
    )
    tgt_5936 = FrameData(
        real=np.ones(5936, dtype=float) * 1.01,
        imag=np.zeros(5936, dtype=float),
        timestamp=0.0,
        frame_index=1,
    )
    request_2160 = rc.ReconstructionRequest(
        reference_frame=ref_2160,
        target_frame=tgt_2160,
        mesh_dimension=3,
        metadata=base_meta,
    )
    request_5936 = rc.ReconstructionRequest(
        reference_frame=ref_5936,
        target_frame=tgt_5936,
        mesh_dimension=3,
        metadata=base_meta,
    )

    sig_2160, payload_2160 = rc._planned_one_step_rm_signature(
        request_2160,
        rc._prepare_single_step_cached_runtime(request_2160),
    )
    sig_5936, payload_5936 = rc._planned_one_step_rm_signature(
        request_5936,
        rc._prepare_single_step_cached_runtime(request_5936),
    )

    assert sig_2160 != sig_5936
    assert payload_2160["stim_meas_protocol"]["n_measurements"] == 2160
    assert payload_5936["stim_meas_protocol"]["n_measurements"] == 5936


def test_smooth_rm_signature_tracks_graph_prior_semantics_not_storage_axes() -> None:
    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "rm_auto_build": True,
        "mesh_size": 0.25,
        "rm_inverse_mesh_size": 0.25,
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "n_elec": 16,
        "radius": 1.0,
        "rm_graph_weight": "unit",
    }
    laplace_cpu = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "simulation_inverse_route": "laplace_rm",
            "rm_regularization": "laplace",
            "device": "cpu",
            "petsc_device": "cpu",
        },
    )
    laplace_cuda = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "simulation_inverse_route": "laplace_rm",
            "rm_regularization": "laplace",
            "device": "cuda",
            "petsc_device": "cuda",
        },
    )
    curvature = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            **base_meta,
            "simulation_inverse_route": "curvature_rm",
            "rm_regularization": "curvature",
            "device": "cpu",
            "petsc_device": "cpu",
        },
    )

    laplace_cpu_runtime = rc._prepare_single_step_cached_runtime(laplace_cpu)
    laplace_cuda_runtime = rc._prepare_single_step_cached_runtime(laplace_cuda)
    curvature_runtime = rc._prepare_single_step_cached_runtime(curvature)
    laplace_cpu_signature, laplace_payload = rc._planned_one_step_rm_signature(
        laplace_cpu,
        laplace_cpu_runtime,
    )
    laplace_cuda_signature, _ = rc._planned_one_step_rm_signature(
        laplace_cuda,
        laplace_cuda_runtime,
    )
    curvature_signature, curvature_payload = rc._planned_one_step_rm_signature(
        curvature,
        curvature_runtime,
    )

    assert laplace_cpu_runtime.cache_key != laplace_cuda_runtime.cache_key
    assert laplace_cpu_signature == laplace_cuda_signature
    assert laplace_cpu_signature != curvature_signature
    assert laplace_payload["regularization_type"] == "laplace"
    assert (
        laplace_payload["hyperparameters"]["prior_operator"]
        == "eidors_prior_laplace_graph_x2"
    )
    assert (
        laplace_payload["hyperparameters"]["rm_jacobian_build_representation"]
        == "dense"
    )
    assert laplace_payload["hyperparameters"]["form"] == "param"
    assert (
        laplace_payload["hyperparameters"]["singular_prior_form_policy"]
        == "param_for_graph_laplace_curvature_v1"
    )
    assert (
        laplace_payload["hyperparameters"]["rm_algorithm_version"]
        == rc._ONE_STEP_RM_ALGORITHM_VERSION
    )
    assert curvature_payload["regularization_type"] == "curvature"
    assert curvature_payload["hyperparameters"]["form"] == "param"
    assert (
        curvature_payload["hyperparameters"]["prior_operator"]
        == "eidors_prior_laplace_squared"
    )


def test_greit_center_cloud_geometry_uses_axis_spacing_not_cloud_median() -> None:
    centers = np.asarray(
        [
            [-0.75, -0.75, 0.0],
            [-0.25, -0.75, 0.0],
            [0.25, -0.75, 0.0],
            [0.75, -0.75, 0.0],
            [-0.25, -0.25, 0.0],
            [0.25, -0.25, 0.0],
            [-0.25, 0.25, 0.0],
            [0.25, 0.25, 0.0],
        ],
        dtype=float,
    )

    coords, cells = rc._center_cloud_hexa_geometry(centers, {"radius": 1.0})

    assert coords.shape == (centers.shape[0] * 8, 3)
    assert cells.shape == (centers.shape[0], 8)
    first_cell = coords[cells[0]]
    assert np.ptp(first_cell[:, 0]) == pytest.approx(0.45)
    assert np.ptp(first_cell[:, 1]) == pytest.approx(0.45)
    assert np.ptp(first_cell[:, 2]) == pytest.approx(0.45)


def test_v125_greit_2d_rec_model_geometry_uses_planar_quads() -> None:
    centers = np.asarray(
        [
            [-0.75, -0.75, 0.0],
            [-0.25, -0.75, 0.0],
            [-0.75, -0.25, 0.0],
            [-0.25, -0.25, 0.0],
        ],
        dtype=float,
    )

    coords, cells = rc._greit_rec_model_geometry(
        centers,
        n_parameters=centers.shape[0],
        meta={"mesh_dimension": 2, "radius": 1.0},
    )

    assert coords.shape == (centers.shape[0] * 4, 2)
    assert cells.shape == (centers.shape[0], 4)
    assert np.ptp(coords[cells[0]][:, 0]) == pytest.approx(0.45)
    assert np.ptp(coords[cells[0]][:, 1]) == pytest.approx(0.45)


def test_run_reconstruction_request_dispatches_to_single_step_cached_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )
    sentinel = rc.ReconstructionResult(
        conductivity=np.array([1.0], dtype=float),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=int),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )

    monkeypatch.setattr(
        rc,
        "_run_single_step_cached_request",
        lambda req, progress_cb=None: sentinel,
    )

    def _unexpected_full(*_args, **_kwargs):
        raise AssertionError(
            "full GN path should not be used for realtime single-step requests"
        )

    monkeypatch.setattr(rc, "_run_full_gn_request", _unexpected_full)

    result = rc.run_reconstruction_request(request)

    assert result is sentinel


def test_single_step_cached_request_returns_absolute_sigma_for_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.array([1.0, 2.0, 3.0], dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.array([4.0, 6.0, 9.0], dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "step_size_calib": False,
        },
    )

    delta_sigma = np.array([0.25, -0.5], dtype=float)
    base_meas = np.array([10.0, 20.0, 30.0], dtype=float)
    pred_diff = np.array([0.5, 1.0, 1.5], dtype=float)

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=base_meas + pred_diff), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: ctx)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda operator_bundle, rhs: delta_sigma,
    )

    result = rc._run_single_step_cached_request(request)

    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + delta_sigma)
    assert result.metadata["conductivity_display_mode"] == "absolute_sigma"
    assert np.allclose(
        result.measured,
        request.target_frame.real - request.reference_frame.real,
    )
    assert np.allclose(result.simulated, pred_diff)


def test_single_step_cached_request_uses_normalized_difference_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.array([2.0, 4.0, -8.0], dtype=float)
    target = np.array([3.0, 2.0, -4.0], dtype=float)
    base_meas = reference.copy()
    pred_target = np.array([2.5, 5.0, -12.0], dtype=float)
    delta_sigma = np.array([0.2, 0.3], dtype=float)
    captured_rhs: list[np.ndarray] = []
    measurement_backend = "measurement-exact"

    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": "normalized",
            "difference_orientation": "target_minus_reference",
            "step_size_calib": False,
        },
    )

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=pred_target), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": measurement_backend,
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }

    def _fake_delta(*, operator_bundle, rhs):
        captured_rhs.append(np.asarray(rhs, dtype=float))
        return delta_sigma

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: ctx)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(
            STRICT_SOLVER_BACKEND_MEASUREMENT=measurement_backend,
            _measurement_space_delta=_fake_delta,
            _solve_linear_from_bundle=lambda operator_bundle, rhs: delta_sigma,
            _calibrate_step_size=lambda **kwargs: 1.0,
            build_shared_context=lambda **kwargs: ctx,
        ),
    )

    result = rc._run_single_step_cached_request(request)

    expected_measured = (target - reference) / reference
    expected_simulated = (pred_target - base_meas) / base_meas
    assert np.allclose(captured_rhs[0], expected_measured)
    assert np.allclose(result.measured, expected_measured)
    assert np.allclose(result.simulated, expected_simulated)


def test_single_step_cached_request_uses_rm_artifact_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.array([2.0, 4.0, -8.0], dtype=float)
    target = np.array([3.0, 8.0, -4.0], dtype=float)
    rm = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    from pyeidors.inverse import write_rm_artifact

    artifact = tmp_path / "one_step_rm.h5"
    write_rm_artifact(
        artifact,
        rm=rm,
        voxel_shape=np.asarray([2, 1, 1], dtype=np.int64),
        metadata={"algorithm": "one-step-noser"},
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": "normalized",
            "difference_orientation": "target_minus_reference",
            "dual_model_rm_path": str(artifact),
            "device": "cpu",
            "n_elec": 8,
            "n_rings": 2,
            "radius": 0.18,
            "height": 0.16,
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("RM hot path must not build GN context/Jacobian.")

    def _unexpected_runner():
        raise AssertionError("RM hot path must not import the GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    expected_dv = (target - reference) / reference
    expected_sigma = rm @ expected_dv
    assert np.allclose(result.conductivity, expected_sigma)
    assert np.allclose(result.measured, expected_dv)
    assert result.simulated is None
    assert result.node_coords.shape[1] == 3
    assert result.cell_connectivity.shape == (2, 8)
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["adjoint_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["runtime"]["ksp_solve_count"] == 0
    assert diagnostics["runtime"]["rm_persistent"] is True
    assert diagnostics["runtime"]["rm_prepare_mode"] == "reused_handle"
    assert diagnostics["runtime"]["rm_dtype"] == "float64"
    assert diagnostics["runtime"]["rm_artifact_cache_hit"] is False
    assert diagnostics["cache_lookups"]["rm_artifact"]["layer"] == "artifact"
    assert diagnostics["rm_metadata"]["algorithm"] == "one-step-noser"

    result_warm = rc._run_single_step_cached_request(request)
    warm_diagnostics = result_warm.metadata["solver_diagnostics"]
    assert np.allclose(result_warm.conductivity, expected_sigma)
    assert warm_diagnostics["runtime"]["rm_artifact_cache_hit"] is True
    assert warm_diagnostics["cache_lookups"]["rm_artifact"]["layer"] == "process"
    assert warm_diagnostics["rm_matmul"]["rm_prepare_mode"] == "reused_handle"


def test_single_step_cached_rm_artifact_rejects_measurement_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import write_rm_artifact

    artifact = tmp_path / "wrong_measurement_count_rm.h5"
    write_rm_artifact(
        artifact,
        rm=np.ones((2, 5), dtype=np.float64),
        voxel_shape=np.asarray([2, 1, 1], dtype=np.int64),
        metadata={"algorithm": "one-step-noser", "n_measurements": 5},
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=np.ones(3, dtype=float),
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=np.ones(3, dtype=float) * 1.01,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "dual_model_rm_path": str(artifact),
            "device": "cpu",
            "n_elec": 8,
            "n_rings": 2,
            "radius": 0.18,
            "height": 0.16,
        },
    )

    def _unexpected_runner():
        raise AssertionError("mismatched RM artifact must fail before GN fallback.")

    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    with pytest.raises(
        ValueError,
        match="RM artifact measurement dimension 5 does not match request measurement dimension 3",
    ):
        rc._run_single_step_cached_request(request)


def test_single_step_cached_noser_rm_route_auto_builds_hdf5_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import build_one_step_rm

    reference = np.array([2.0, 4.0, 8.0], dtype=float)
    target = np.array([3.0, 5.0, 10.0], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    context_calls = {"count": 0}

    def _fake_context(*_args, **_kwargs):
        context_calls["count"] += 1
        return fake_ctx

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _fake_context)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.04,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
        },
    )

    result = rc._run_single_step_cached_request(request)

    expected_dv = target - reference
    expected_rm = build_one_step_rm(
        jacobian,
        lambda_=0.2,
        mode="noser",
        form="measurement",
    )
    expected_sigma = 1.0 + expected_rm @ expected_dv
    np.testing.assert_allclose(result.conductivity, expected_sigma)
    np.testing.assert_allclose(result.simulated, jacobian @ (expected_sigma - 1.0))
    assert result.error_msg is None
    assert context_calls["count"] == 1
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    assert result.metadata["rm_output_display_mode"] == "absolute_sigma"
    artifact_path = Path(result.metadata["rm_artifact_path"])
    assert artifact_path.suffix == ".h5"
    assert artifact_path.exists()
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["rm_metadata"]["rm_build_route"] == "noser_rm"
    assert diagnostics["rm_metadata"]["rm_signature"]
    assert diagnostics["rm_metadata"]["form"] == "measurement"

    warm = rc._run_single_step_cached_request(request)
    np.testing.assert_allclose(warm.conductivity, expected_sigma)
    np.testing.assert_allclose(warm.simulated, jacobian @ (expected_sigma - 1.0))
    assert context_calls["count"] == 1
    assert (
        warm.metadata["solver_diagnostics"]["cache_lookups"]["rm_artifact"]["layer"]
        == "process"
    )


def test_single_step_cached_auto_built_rm_rebuilds_stale_fitless_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import build_one_step_rm, write_rm_artifact

    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()
    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    reference = np.array([2.0, 4.0, 8.0], dtype=float)
    target = np.array([3.0, 5.0, 10.0], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    context_calls = {"count": 0}

    def _fake_context(*_args, **_kwargs):
        context_calls["count"] += 1
        return fake_ctx

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _fake_context)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.04,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
        },
    )
    runtime = rc._prepare_single_step_cached_runtime(request)
    stale_path, _signature, _payload = rc._planned_one_step_rm_artifact_path(
        request, runtime
    )
    stale_rm = build_one_step_rm(
        jacobian,
        lambda_=0.2,
        mode="noser",
        form="measurement",
    )
    write_rm_artifact(
        stale_path,
        stale_rm,
        metadata={"algorithm": "one-step-noser", "rm_build_route": "noser_rm"},
        node_coords=node_coords,
        cell_connectivity=cells,
    )

    result = rc._run_single_step_cached_request(request)

    expected_dv = target - reference
    expected_sigma = 1.0 + stale_rm @ expected_dv
    np.testing.assert_allclose(result.conductivity, expected_sigma)
    np.testing.assert_allclose(result.simulated, jacobian @ (expected_sigma - 1.0))
    assert result.metadata["rm_artifact_cache_status"] == "built"
    assert result.metadata["rm_fit_jacobian_cache_status"].startswith("built_")
    assert context_calls["count"] == 1


@pytest.mark.parametrize(
    ("route", "mode", "prior_builder", "expected_source"),
    [
        ("laplace_rm", "laplace", "graph_laplacian", "provided_laplace"),
        ("curvature_rm", "curvature", "graph_ltl_prior", "provided_graph_ltl"),
    ],
)
def test_single_step_cached_smooth_rm_routes_auto_build_graph_prior_hdf5_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    mode: str,
    prior_builder: str,
    expected_source: str,
) -> None:
    from pyeidors.inverse import CellMesh
    from pyeidors.inverse.prior import graph_laplacian, graph_ltl_prior
    from pyeidors.inverse.reconstruction_matrix import build_one_step_rm

    reference = np.ones(4, dtype=float)
    target = reference + np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    jacobian = np.eye(4, dtype=float)
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        dtype=float,
    )
    cells = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(4, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    context_calls = {"count": 0}

    def _fake_context(*_args, **_kwargs):
        context_calls["count"] += 1
        return fake_ctx

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _fake_context)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": route,
            "rm_regularization": mode,
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.25,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
        },
    )

    result = rc._run_single_step_cached_request(request)

    inverse_mesh = CellMesh(node_coords, cells, name=f"{route}-expected")
    regularization = (
        graph_laplacian(inverse_mesh)
        if prior_builder == "graph_laplacian"
        else graph_ltl_prior(inverse_mesh)
    )
    expected_rm = build_one_step_rm(
        jacobian,
        regularization=regularization,
        lambda_=0.5,
        mode=mode,
        form="param",
    )
    expected_delta = expected_rm @ (target - reference)
    np.testing.assert_allclose(result.conductivity, 1.0 + expected_delta)
    assert result.error_msg is None
    assert context_calls["count"] == 1
    assert Path(result.metadata["rm_artifact_path"]).suffix == ".h5"
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    delta = result.conductivity - 1.0
    assert delta[1:3].mean() > delta[[0, 3]].mean()
    if route == "laplace_rm":
        assert abs(delta[1] - delta[2]) <= 1.0e-12
        assert abs(delta[0] - delta[3]) <= 1.0e-12
    else:
        assert np.isfinite(delta).all()

    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["rm_metadata"]["rm_build_route"] == route
    assert diagnostics["rm_metadata"]["regularization_type"] == mode
    assert diagnostics["rm_metadata"]["regularization_source"] == expected_source
    assert diagnostics["rm_metadata"]["RtR_signature_hash"]
    assert diagnostics["rm_metadata"]["form"] == "param"

    warm = rc._run_single_step_cached_request(request)
    np.testing.assert_allclose(warm.conductivity, result.conductivity)
    assert context_calls["count"] == 1
    assert (
        warm.metadata["solver_diagnostics"]["cache_lookups"]["rm_artifact"]["layer"]
        == "process"
    )


def test_single_step_cached_auto_built_rm_honors_float32_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse.reconstruction_matrix import load_rm_artifact

    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()
    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()

    reference = np.ones(4, dtype=np.float32)
    target = reference + np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32)
    jacobian = np.eye(4, dtype=np.float64)
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(4, dtype=np.float64),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }

    monkeypatch.setattr(
        rc,
        "_ensure_single_step_cached_context",
        lambda *_args, **_kwargs: fake_ctx,
    )
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=lambda **_kwargs: fake_ctx),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(4, dtype=np.float32),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(4, dtype=np.float32),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "rm_regularization": "noser",
            "rm_form": "measurement",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "delta_sigma",
            "difference_lambda": 0.25,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "compute_precision": "float32",
            "compute_dtype": "float32",
            "rm_dtype": "float32",
            "rm_matmul_dtype": "float32",
            "device": "cpu",
        },
    )

    result = rc._run_single_step_cached_request(request)

    artifact_path = Path(result.metadata["rm_artifact_path"])
    artifact = load_rm_artifact(artifact_path)
    diagnostics = result.metadata["solver_diagnostics"]

    assert result.error_msg is None
    assert artifact.rm.dtype == np.float32
    assert artifact.metadata["rm_dtype"] == "float32"
    assert artifact.metadata["build_dtype"] == "float32"
    assert artifact.metadata["prior_inverse_solver"] == "diagonal"
    assert diagnostics["runtime"]["rm_dtype"] == "float32"
    assert diagnostics["rm_metadata"]["rm_dtype"] == "float32"
    assert diagnostics["rm_matmul"]["rm_dtype"] == "float32"
    assert result.metadata["rm_signature_payload"]["hyperparameters"]["rm_dtype"] == (
        "float32"
    )


def test_single_step_cached_3d_rm_auto_build_forces_dense_jacobian_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()

    from pyeidors.inverse import CellMesh
    from pyeidors.inverse.prior import graph_laplacian
    from pyeidors.inverse.reconstruction_matrix import build_one_step_rm

    reference = np.ones(4, dtype=float)
    target = reference + np.array([0.0, 1.0, 0.5, -0.25], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    build_calls: list[dict[str, object]] = []

    def _fake_build_shared_context(**kwargs):
        build_calls.append(dict(kwargs))
        return fake_ctx

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: None)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=_fake_build_shared_context),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "mesh_dimension": 3,
            "simulation_inverse_route": "laplace_rm",
            "rm_regularization": "laplace",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 0.04,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "solver_mode": "fast",
            "jacobian_representation": "auto",
            "device": "cpu",
        },
    )

    runtime = rc._prepare_single_step_cached_runtime(request)
    assert runtime.meta["jacobian_representation"] == "dense"
    assert runtime.meta["rm_build_jacobian_representation_requested"] == "linearized"
    assert runtime.meta["single_step_context_cache_scope"] == "process"

    result = rc._run_single_step_cached_request(request)

    inverse_mesh = CellMesh(node_coords, cells, name="expected-3d-laplace-rm")
    expected_rm = build_one_step_rm(
        jacobian,
        regularization=graph_laplacian(inverse_mesh),
        lambda_=0.2,
        mode="laplace",
        form="param",
    )
    expected_delta = expected_rm @ (target - reference)
    np.testing.assert_allclose(result.conductivity, 1.0 + expected_delta)
    np.testing.assert_allclose(result.simulated, jacobian @ expected_delta)
    assert result.error_msg is None
    assert len(build_calls) == 1
    assert build_calls[0]["jacobian_representation"] == "dense"
    assert build_calls[0]["cache_scope"] == "process"
    assert "_inmem_jacobian" not in result.metadata
    assert result.metadata["rm_build_jacobian_representation"] == "dense"
    assert result.metadata["rm_build_jacobian_representation_requested"] == "linearized"
    assert (
        result.metadata["solver_diagnostics"]["rm_metadata"][
            "rm_jacobian_source_cache_scope"
        ]
        == "process"
    )
    assert (
        result.metadata["solver_diagnostics"]["rm_metadata"]["rm_build_route"]
        == "laplace_rm"
    )

    warm_result = rc._run_single_step_cached_request(request)

    np.testing.assert_allclose(warm_result.conductivity, 1.0 + expected_delta)
    np.testing.assert_allclose(warm_result.simulated, jacobian @ expected_delta)
    assert len(build_calls) == 1
    assert "_inmem_jacobian" not in warm_result.metadata
    assert warm_result.metadata["rm_artifact_cache_status"] == "disk_hit"
    assert warm_result.metadata["rm_fit_jacobian_cache_status"] == "process_hit"

    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()

    artifact_fit_result = rc._run_single_step_cached_request(request)

    np.testing.assert_allclose(artifact_fit_result.conductivity, 1.0 + expected_delta)
    np.testing.assert_allclose(artifact_fit_result.simulated, jacobian @ expected_delta)
    assert len(build_calls) == 1
    assert "_inmem_jacobian" not in artifact_fit_result.metadata
    assert artifact_fit_result.metadata["rm_artifact_cache_status"] == "disk_hit"
    assert artifact_fit_result.metadata["rm_fit_jacobian_cache_status"].startswith(
        "artifact_hit_"
    )


def test_single_step_cached_2d_cuda_rm_auto_build_uses_cuda_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with rc._RM_FIT_JACOBIAN_CACHE_LOCK:
        rc._RM_FIT_JACOBIAN_CACHE.clear()
    with rc._RM_ARTIFACT_CACHE_LOCK:
        rc._RM_ARTIFACT_CACHE.clear()

    reference = np.ones(4, dtype=float)
    target = reference + np.array([0.0, 0.20, -0.10, 0.05], dtype=float)
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.1, 0.8],
            [0.4, 0.3],
            [0.2, 0.5],
        ],
        dtype=float,
    )
    node_coords = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    fake_ctx = {
        "J": jacobian,
        "display_node_coords": node_coords,
        "display_cell_connectivity": cells,
        "sigma_bg": np.ones(2, dtype=float),
        "mesh": SimpleNamespace(coordinates=lambda: node_coords, cells=lambda: cells),
    }
    build_calls: list[dict[str, object]] = []

    def _fake_build_shared_context(**kwargs):
        build_calls.append(dict(kwargs))
        return fake_ctx

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: None)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        rc,
        "_load_gn_difference_runner_module",
        lambda: SimpleNamespace(build_shared_context=_fake_build_shared_context),
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(4, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        regularization_alpha=0.1,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "mesh_dimension": 2,
            "mesh_size": 0.1,
            "simulation_inverse_route": "noser_rm",
            "rm_regularization": "noser",
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_artifact_dir": str(tmp_path),
            "rm_device": "cpu",
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": 1.0e-2,
            "difference_mode": "normalized",
            "difference_orientation": "target_minus_reference",
            "solver_mode": "strict",
            "jacobian_representation": "auto",
            "device": "cuda",
            "petsc_device": "cuda",
            "forward_backend": "dolfinx",
            "mesh_family": "tetra",
            "potential_order": 1,
        },
    )

    result = rc._run_single_step_cached_request(request)

    assert result.error_msg is None
    assert np.isfinite(result.conductivity).all()
    assert len(build_calls) == 1
    call = build_calls[0]
    assert call["mesh_dim"] == 2
    assert call["petsc_device"] == "cuda"
    assert call["device"] == "cuda"
    assert call["forward_backend"] == "dolfinx"
    assert call["jacobian_representation"] == "dense"
    assert call["cache_scope"] == "process"
    assert call["difference_mode"] == "normalized"
    assert (
        result.metadata["solver_diagnostics"]["rm_metadata"][
            "rm_jacobian_source_cache_scope"
        ]
        == "process"
    )


def test_single_step_cached_non_noser_rm_route_requires_artifact_before_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.array([2.0, 4.0, -8.0], dtype=float)
    target = np.array([3.0, 8.0, -4.0], dtype=float)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros(3, dtype=float),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=2,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "laplace_rm",
            "rm_route_requires_artifact": True,
            "rm_route_pending_task": "T101",
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("RM route without artifact must not build dense context.")

    def _unexpected_runner():
        raise AssertionError("RM route without artifact must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    assert result.error_msg is not None
    assert "laplace_rm requires a precomputed RM/GREIT artifact" in result.error_msg
    assert result.conductivity.size == 0
    assert result.metadata["rm_artifact_missing"] is True
    assert result.metadata["rm_route_pending_task"] == "T101"
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm_missing_artifact"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0


def test_single_step_cached_request_resolves_greit_common_config_hot_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import precompute_greit_common_config

    warmup = precompute_greit_common_config("16e", artifact_dir=tmp_path)
    reference = np.linspace(1.0, 2.0, warmup.config.n_measurements, dtype=float)
    target = reference + np.linspace(
        0.01,
        0.02,
        warmup.config.n_measurements,
        dtype=float,
    )
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros_like(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros_like(target),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "greit_common_config": "16e",
            "greit_common_config_dir": str(tmp_path),
            "device": "cpu",
            "n_elec": 16,
            "n_rings": 1,
            "radius": 0.18,
            "height": 0.16,
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("common-config RM hot path must not build context.")

    def _unexpected_runner():
        raise AssertionError("common-config RM hot path must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    expected_dv = target - reference
    expected_sigma = warmup.greit.rm @ expected_dv
    np.testing.assert_allclose(result.conductivity, expected_sigma)
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    assert result.metadata["rm_artifact_path"] == str(warmup.artifact_path)
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["adjoint_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["runtime"]["ksp_solve_count"] == 0
    assert diagnostics["rm_metadata"]["common_config_id"] == "16e"


def test_single_step_cached_greit_production_route_rejects_fixture_auto_warm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import greit_common_config

    cfg = greit_common_config("16e")
    reference = np.linspace(1.0, 2.0, cfg.n_measurements, dtype=float)
    target = reference + np.linspace(0.01, 0.02, cfg.n_measurements, dtype=float)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros_like(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros_like(target),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "greit3d_rm",
            "rm_route_requires_artifact": True,
            "greit_common_config": "16e",
            "greit_common_config_dir": str(tmp_path),
            "greit_common_config_auto_warm": True,
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
            "n_elec": 16,
            "n_rings": 1,
            "radius": 0.18,
            "height": 0.16,
            "greit_official_fixture_scope": "requires registered EIDORS parity artifact",
            "greit_5936_protocol_scope": "production route rejects deterministic fixtures",
            "greit_official_equivalence_claim_allowed": False,
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("GREIT common-config route must not build context.")

    def _unexpected_runner():
        raise AssertionError("GREIT common-config route must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    assert result.error_msg
    assert result.metadata["rm_artifact_missing"] is True
    assert result.metadata["rm_artifact_required"] is True
    assert "registered EIDORS-parity artifact" in result.error_msg
    assert not (tmp_path / "greit3d_common_16e.h5").exists()


def test_single_step_cached_greit_official_artifact_uses_rec_geometry_and_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import GREIT_EIDORS_HDF5_SCHEMA, GREITRM

    artifact_path = tmp_path / "official_greit.h5"
    rm = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
        ],
        dtype=float,
    )
    d = np.eye(2, dtype=float)
    rec_model = np.array(
        [
            [-0.2, -0.2, 0.0],
            [0.0, -0.2, 0.0],
            [-0.2, 0.0, 0.0],
            [0.2, 0.2, 0.1],
            [0.4, 0.2, 0.1],
            [0.2, 0.4, 0.1],
        ],
        dtype=float,
    )
    GREITRM(
        rm=rm,
        metadata=MappingProxyType(
            {
                "algorithm": "greit-3d",
                "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
                "artifact_format": "hdf5",
                "eidors_parity": True,
                "fixture_only": False,
                "keep_model_components": True,
                "online_hot_path": "rm_matmul",
            }
        ),
        voxel_shape=(2, 1, 1),
        y=y,
        d=d,
        rec_model=rec_model,
    ).save(artifact_path)

    reference = np.array([2.0, 3.0, 4.0], dtype=float)
    target = reference + np.array([0.2, -0.1, 0.05], dtype=float)
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros_like(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros_like(target),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "greit3d_rm",
            "rm_route_requires_artifact": True,
            "greit_rm_path": str(artifact_path),
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            "device": "cpu",
            "n_elec": 3,
            "n_rings": 1,
            "radius": 1.0,
            "height": 1.0,
        },
    )

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("official GREIT artifact route must not build context.")

    def _unexpected_runner():
        raise AssertionError("official GREIT artifact route must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    dv = target - reference
    expected = rm @ dv
    np.testing.assert_allclose(result.conductivity, expected)
    assert result.node_coords.shape == (16, 3)
    assert result.cell_connectivity.shape == (2, 8)
    assert result.metadata["rm_geometry_source"] == "greit_rec_model_centers"
    assert result.metadata["rm_fit_source"] == "greit_training_space_projection"
    np.testing.assert_allclose(result.simulated, y @ expected)
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["path"] == "single_step_cached_rm"
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["adjoint_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["runtime"]["ksp_solve_count"] == 0
    assert diagnostics["rm_metadata"]["eidors_parity"] is True


def test_single_step_cached_greit_registry_hot_path_requires_exact_signature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import (
        GREIT_EIDORS_HDF5_SCHEMA,
        GREITRM,
        greit_artifact_signature,
        greit_artifact_signature_payload,
        register_greit_artifact,
    )

    reference = np.array([2.0, 3.0, 4.0], dtype=float)
    target = reference + np.array([0.2, -0.1, 0.05], dtype=float)
    base_meta = {
        "reconstruction_runtime": "single_step_cached",
        "simulation_inverse_route": "greit3d_rm",
        "rm_route_requires_artifact": True,
        "rm_auto_build": False,
        "greit_registry_auto_resolve": True,
        "greit_registry_dir": str(tmp_path),
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
        "device": "cpu",
        "n_elec": 3,
        "n_rings": 1,
        "radius": 1.0,
        "height": 1.0,
        "electrode_height_ratio": 0.2,
        "electrode_level_fractions": (0.25, 0.75),
        "electrode_layout": "ring_major",
        "measurement_protocol": "eidors_full_3d",
        "stim_pattern": "{ad}",
        "meas_pattern": "{ad}",
        "background_conductivity": 1.0,
        "contact_impedance": 0.0,
        "imgsz": (2, 1, 1),
        "target_radius": 0.2,
        "target_contrast": 1.0,
        "weight": 0.5,
        "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
        "builder_backend": "native",
        "builder_semantic_version": "native-greit-finite-target-v1",
    }
    request = rc.ReconstructionRequest(
        reference_frame=FrameData(
            real=reference,
            imag=np.zeros_like(reference),
            timestamp=0.0,
            frame_index=0,
        ),
        target_frame=FrameData(
            real=target,
            imag=np.zeros_like(target),
            timestamp=0.0,
            frame_index=1,
        ),
        mesh_dimension=3,
        metadata=dict(base_meta),
    )
    runtime = rc._prepare_single_step_cached_runtime(request)
    config = rc._greit_registry_config_from_runtime(request, runtime)
    signature = greit_artifact_signature(config)
    payload = greit_artifact_signature_payload(config)
    artifact_path = tmp_path / "registered_greit.h5"
    rm = np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.25],
        ],
        dtype=float,
    )
    y = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.25],
        ],
        dtype=float,
    )
    d = np.eye(2, dtype=float)
    rec_model = np.array([[-0.2, -0.2, 0.0], [0.2, -0.2, 0.0]], dtype=float)
    GREITRM(
        rm=rm,
        metadata=MappingProxyType(
            {
                "algorithm": "greit-3d",
                "artifact_schema": GREIT_EIDORS_HDF5_SCHEMA,
                "artifact_format": "hdf5",
                "eidors_parity": True,
                "fixture_only": False,
                "keep_model_components": True,
                "online_hot_path": "rm_matmul",
                "greit_registry_signature": signature,
                "greit_registry_signature_payload": payload,
            }
        ),
        voxel_shape=(2, 1, 1),
        y=y,
        d=d,
        rec_model=rec_model,
    ).save(artifact_path)
    registered = register_greit_artifact(config, artifact_path, registry_dir=tmp_path)
    assert registered.signature == signature

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("GREIT registry hit must not build context.")

    def _unexpected_runner():
        raise AssertionError("GREIT registry hit must not import GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)

    result = rc._run_single_step_cached_request(request)

    expected = rm @ (target - reference)
    np.testing.assert_allclose(result.conductivity, expected)
    assert result.metadata["greit_registry_signature"] == signature
    assert result.metadata["greit_registry_cache_status"] == "disk_hit"
    assert result.metadata["rm_artifact_path"] == str(artifact_path)

    bad_request = rc.ReconstructionRequest(
        reference_frame=request.reference_frame,
        target_frame=request.target_frame,
        mesh_dimension=3,
        metadata={**base_meta, "n_rings": 2},
    )
    bad_result = rc._run_single_step_cached_request(bad_request)
    assert bad_result.error_msg
    assert bad_result.metadata["rm_artifact_missing"] is True


def test_single_step_cached_request_uses_hardware_drive_metadata_for_context_and_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    pred_diff = np.array([0.1, 0.2, 0.3], dtype=float)
    delta_sigma = np.array([0.4], dtype=float)
    cache_keys: list[tuple[object, ...]] = []
    build_kwargs: list[dict[str, object]] = []

    class _StubForwardModel:
        def fwd_solve(self, image):
            assert image.elem_data is not None
            return SimpleNamespace(meas=base_meas + pred_diff), None

    def _fake_build_shared_context(**kwargs):
        build_kwargs.append(dict(kwargs))
        return {
            "mesh": object(),
            "display_node_coords": np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=float,
            ),
            "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
            "operator_bundle": {
                "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
            },
            "sigma_bg": np.ones_like(delta_sigma),
            "fwd_model": _StubForwardModel(),
            "base_meas": base_meas,
            "cache_build_seconds": {},
            "cache_miss_reasons": {},
            "cache_manager": None,
        }

    monkeypatch.setattr(
        rc,
        "_get_cached_fast_context",
        lambda cache_key: cache_keys.append(cache_key) or None,
    )
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(diff_runner, "build_shared_context", _fake_build_shared_context)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda operator_bundle, rhs: delta_sigma,
    )

    request_100 = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "drive_mode": "total_current",
            "stim_amp_uA": 100,
            "step_size_calib": False,
        },
    )
    request_200 = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "drive_mode": "total_current",
            "stim_amp_uA": 200,
            "step_size_calib": False,
        },
    )

    result_100 = rc._run_single_step_cached_request(request_100)
    result_200 = rc._run_single_step_cached_request(request_200)

    assert len(cache_keys) == 2
    assert cache_keys[0] != cache_keys[1]
    assert build_kwargs[0]["drive_mode"] == "total_current"
    assert build_kwargs[0]["drive_value"] == pytest.approx(100e-6)
    assert build_kwargs[1]["drive_mode"] == "total_current"
    assert build_kwargs[1]["drive_value"] == pytest.approx(200e-6)
    assert result_100.metadata["drive_mode"] == "total_current"
    assert result_100.metadata["drive_value"] == pytest.approx(100e-6)
    assert result_200.metadata["drive_value"] == pytest.approx(200e-6)
    assert (
        build_kwargs[0]["single_step_algorithm_version"]
        == result_100.metadata["single_step_algorithm_version"]
    )
    assert (
        build_kwargs[0]["single_step_jacobian_math_convention"]
        == result_100.metadata["single_step_jacobian_math_convention"]
    )
    assert (
        build_kwargs[0]["single_step_projection_math_convention"]
        == result_100.metadata["single_step_projection_math_convention"]
    )
    assert (
        build_kwargs[0]["single_step_operator_math_convention"]
        == result_100.metadata["single_step_operator_math_convention"]
    )


def test_single_step_cached_request_scales_absolute_display_by_calibrated_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    delta_sigma = np.array([2.0, -2.0], dtype=float)
    base_meas = np.array([1.0, 2.0, 3.0], dtype=float)
    captured_sigmas: list[np.ndarray] = []

    class _StubForwardModel:
        def fwd_solve(self, image):
            sigma = np.asarray(image.elem_data, dtype=float)
            captured_sigmas.append(sigma.copy())
            pred = np.array([sigma[0], sigma[1], sigma.mean()], dtype=float)
            return SimpleNamespace(meas=base_meas + pred), None

    ctx = {
        "mesh": object(),
        "display_node_coords": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=float,
        ),
        "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
        "operator_bundle": {
            "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
        },
        "sigma_bg": np.ones_like(delta_sigma),
        "fwd_model": _StubForwardModel(),
        "base_meas": base_meas,
        "cache_build_seconds": {},
        "cache_miss_reasons": {},
        "cache_manager": None,
    }
    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={"reconstruction_runtime": "single_step_cached"},
    )

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: ctx)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda operator_bundle, rhs: delta_sigma,
    )
    monkeypatch.setattr(
        diff_runner,
        "_calibrate_step_size",
        lambda **kwargs: 0.25,
    )

    result = rc._run_single_step_cached_request(request)

    expected_update = delta_sigma * 0.25
    assert np.allclose(result.conductivity, np.ones_like(delta_sigma) + expected_update)
    assert np.allclose(captured_sigmas[-1], np.ones_like(delta_sigma) + expected_update)
    assert result.metadata["step_size_alpha"] == pytest.approx(0.25)


def test_single_step_cached_request_warmup_only_primes_context_without_solving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    build_calls: list[dict[str, object]] = []

    def _fake_build_shared_context(**kwargs):
        build_calls.append(dict(kwargs))
        return {
            "mesh": object(),
            "display_node_coords": np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=float,
            ),
            "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
            "operator_bundle": {
                "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
            },
            "sigma_bg": np.array([1.0], dtype=float),
            "fwd_model": object(),
            "base_meas": np.array([0.0, 0.0, 0.0], dtype=float),
            "cache_build_seconds": {"mesh": 0.1},
            "cache_miss_reasons": {},
            "cache_manager": None,
        }

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: None)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(diff_runner, "build_shared_context", _fake_build_shared_context)
    monkeypatch.setattr(
        diff_runner,
        "_measurement_space_delta",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("warmup should not solve")
        ),
    )

    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "warmup_only": True,
        },
    )

    result = rc._run_single_step_cached_request(request)

    assert len(build_calls) == 1
    assert result.conductivity.size == 0
    assert result.metadata["cache_warmup_only"] is True
    assert (
        result.metadata["solver_diagnostics"]["strict_solver_backend_effective"]
        == "warmup_only"
    )


def test_single_step_cached_3d_context_uses_total_current_multiring_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    build_calls: list[dict[str, object]] = []

    def _fake_build_shared_context(**kwargs):
        build_calls.append(dict(kwargs))
        return {
            "mesh": object(),
            "display_node_coords": np.array(
                [[0.0, 0.0, -0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
                dtype=float,
            ),
            "display_cell_connectivity": np.array([[0, 1, 2]], dtype=int),
            "operator_bundle": {
                "strict_solver_backend_effective": diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT,
            },
            "sigma_bg": np.array([1.0], dtype=float),
            "fwd_model": object(),
            "base_meas": np.array([0.0, 0.0, 0.0], dtype=float),
            "cache_build_seconds": {},
            "cache_miss_reasons": {},
            "cache_manager": None,
        }

    monkeypatch.setattr(rc, "_get_cached_fast_context", lambda _cache_key: None)
    monkeypatch.setattr(rc, "_put_cached_fast_context", lambda _cache_key, _ctx: None)
    monkeypatch.setattr(diff_runner, "build_shared_context", _fake_build_shared_context)

    request = rc.ReconstructionRequest(
        reference_frame=_make_frame(0),
        target_frame=_make_frame(1),
        mesh_dimension=3,
        metadata={
            "reconstruction_runtime": "single_step_cached",
            "warmup_only": True,
            "mesh_dimension": 3,
            "n_elec": 8,
            "n_rings": 2,
            "drive_mode": "line_current_density",
        },
    )

    result = rc._run_single_step_cached_request(request)

    assert result.metadata["cache_warmup_only"] is True
    assert len(build_calls) == 1
    assert build_calls[0]["mesh_dim"] == 3
    assert build_calls[0]["n_elec"] == 8
    assert build_calls[0]["n_rings"] == 2
    assert build_calls[0]["drive_mode"] == "total_current"
    assert build_calls[0]["jacobian_representation"] == "linearized"


def test_gn_difference_runner_3d_multiring_loads_ring_ordered_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.common.gn_difference_runner as diff_runner

    captured: dict[str, object] = {}

    def _fake_load_or_create_mesh(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after mesh kwargs")

    monkeypatch.setattr(diff_runner, "load_or_create_mesh", _fake_load_or_create_mesh)

    with pytest.raises(RuntimeError, match="stop after mesh kwargs"):
        diff_runner.build_shared_context(
            mesh_dir=str(tmp_path),
            mesh_name=None,
            mesh_dim=3,
            mesh_height=0.16,
            electrode_height_ratio=0.2,
            z_center=0.0,
            electrode_level_fractions=(0.25, 0.75),
            refinement=2,
            n_elec=8,
            n_rings=2,
            radius=0.18,
            drive_mode="line_current_density",
            drive_value=1.0e-5,
            solver_mode="fast",
        )

    assert captured["n_elec"] == 16
    assert captured["dimension"] == 3
    assert captured["electrode_layout"] == "ring_major"


def test_load_gn_difference_runner_module_falls_back_to_repo_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rc._load_gn_difference_runner_module.cache_clear()
    module_name = "scripts.common.gn_difference_runner"
    repo_root = Path(rc.__file__).resolve().parents[3]
    sentinel = SimpleNamespace(name="gn-diff-runner")
    calls: list[tuple[str, list[str]]] = []
    original_sys_path = list(sys.path)

    def _fake_import(name: str):
        calls.append((name, list(sys.path)))
        if name != module_name:
            raise AssertionError(f"Unexpected import: {name}")
        if len(calls) == 1:
            exc = ModuleNotFoundError("No module named 'scripts'")
            exc.name = "scripts"
            raise exc
        return sentinel

    monkeypatch.setattr(rc.importlib, "import_module", _fake_import)
    sys.path[:] = [
        entry
        for entry in original_sys_path
        if Path(entry or ".").resolve() != repo_root
    ]
    try:
        loaded = rc._load_gn_difference_runner_module()
    finally:
        sys.path[:] = original_sys_path
        rc._load_gn_difference_runner_module.cache_clear()

    assert loaded is sentinel
    assert [name for name, _ in calls] == [module_name, module_name]
    assert str(repo_root) in calls[1][1]


def test_recover_nix_runtime_site_packages_restores_missing_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_paths = [
        "/nix/store/a-python3.13-fenics-dolfinx/lib/python3.13/site-packages",
        "/nix/store/b-python3.13-fenics-ufl/lib/python3.13/site-packages",
        "/nix/store/c-petsc/lib/python3.13/site-packages",
    ]
    original_sys_path = list(sys.path)
    original_pythonpath = os.environ.get("PYTHONPATH")

    def _fake_glob(pattern: str) -> list[str]:
        if "fenics-dolfinx" in pattern:
            return [fake_paths[0]]
        if "fenics-ufl" in pattern:
            return [fake_paths[1]]
        if "petsc" in pattern:
            return [fake_paths[2]]
        return []

    monkeypatch.setattr(rc.Path, "exists", lambda self: str(self) == "/nix/store")
    monkeypatch.setattr(rc.glob, "glob", _fake_glob)
    monkeypatch.setattr(rc.os.path, "isdir", lambda path: path in fake_paths)
    sys.path[:] = [entry for entry in original_sys_path if entry not in fake_paths]
    os.environ["PYTHONPATH"] = "/tmp/original-pythonpath"
    captured_sys_path: list[str] = []
    captured_pythonpath = ""
    try:
        added = rc._recover_nix_runtime_site_packages("ufl")
        captured_sys_path = list(sys.path)
        captured_pythonpath = os.environ["PYTHONPATH"]
    finally:
        sys.path[:] = original_sys_path
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath

    assert added == tuple(reversed(fake_paths))
    assert captured_sys_path[: len(fake_paths)] == fake_paths
    assert captured_pythonpath.startswith(os.pathsep.join(reversed(fake_paths)))


def test_clear_reconstruction_system_cache_clears_both_runtime_caches() -> None:
    rc._SYSTEM_CACHE[("system",)] = object()
    rc._FAST_CONTEXT_CACHE[("fast",)] = object()

    rc.clear_reconstruction_system_cache()

    assert not rc._SYSTEM_CACHE
    assert not rc._FAST_CONTEXT_CACHE


def test_boundary_voltage_plot_keeps_recon_overlay_visible_for_tiny_fit() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="hardware")
    measured = np.linspace(-1.0, 1.0, 208, dtype=float)
    reconstructed = 1.0e-6 * np.sin(np.linspace(0.0, 6.0 * np.pi, 208, dtype=float))

    widget.update_hardware_voltages(measured, reconstructed)

    assert widget._curve_primary.isVisible() is True
    assert widget._curve_reconstructed_outline.isVisible() is True
    assert widget._curve_reconstructed.isVisible() is True
    assert widget._curve_reconstructed_markers.isVisible() is True
    marker_x, marker_y = widget._curve_reconstructed_markers.getData()
    assert marker_x is not None and marker_y is not None
    assert len(marker_x) >= 2
    assert float(marker_x[0]) == pytest.approx(1.0)
    assert float(marker_x[-1]) == pytest.approx(208.0)


def test_boundary_voltage_plot_truth_is_not_hidden_by_recon_outline() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="simulation")

    widget.update_simulation_voltages(
        np.linspace(-1.0, 1.0, 16, dtype=float),
        np.linspace(-1.0, 1.0, 16, dtype=float),
    )

    assert widget._curve_primary.isVisible() is True
    assert widget._curve_reconstructed_outline.isVisible() is True
    assert widget._curve_reconstructed_outline.zValue() < widget._curve_primary.zValue()
    assert widget._curve_primary.zValue() < widget._curve_reconstructed.zValue()


def test_boundary_voltage_plot_rescales_y_range_for_new_simulation_data() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="simulation")
    widget._plot_widget.setYRange(1000.0, 2000.0, padding=0.0)

    truth = np.array([1.0e-6, 2.0e-6, 3.0e-6], dtype=float)
    reconstructed = np.array([1.5e-6, -4.0e-6, 2.5e-6], dtype=float)
    widget.update_simulation_voltages(truth, reconstructed)

    _x_range, y_range = widget._plot_widget.getPlotItem().getViewBox().viewRange()
    assert y_range[0] < -4.0e-6
    assert y_range[1] > 3.0e-6
    primary_x, primary_y = widget._curve_primary.getData()
    recon_x, recon_y = widget._curve_reconstructed.getData()
    assert primary_x is not None and primary_y is not None
    assert recon_x is not None and recon_y is not None
    np.testing.assert_allclose(primary_y, truth)
    np.testing.assert_allclose(recon_y, reconstructed)


def test_boundary_voltage_plot_hides_recon_overlay_without_fit_data() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="hardware")

    widget.update_hardware_voltages(np.linspace(-1.0, 1.0, 16, dtype=float), None)

    assert widget._curve_reconstructed_outline.isVisible() is False
    assert widget._curve_reconstructed.isVisible() is False
    assert widget._curve_reconstructed_markers.isVisible() is False
