from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from eit_app.controllers import reconstruction_controller as rc
from eit_app.models.frame_model import FrameData
from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget

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


def test_single_step_cached_request_returns_delta_conductivity_for_display(
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

    assert np.allclose(result.conductivity, delta_sigma)
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


def test_single_step_cached_request_scales_display_by_calibrated_alpha(
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

    expected_display = delta_sigma * 0.25
    assert np.allclose(result.conductivity, expected_display)
    assert np.allclose(
        captured_sigmas[-1], np.ones_like(delta_sigma) + expected_display
    )
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


def test_boundary_voltage_plot_hides_recon_overlay_without_fit_data() -> None:
    _get_app()
    widget = BoundaryVoltagePlotWidget(mode="hardware")

    widget.update_hardware_voltages(np.linspace(-1.0, 1.0, 16, dtype=float), None)

    assert widget._curve_reconstructed_outline.isVisible() is False
    assert widget._curve_reconstructed.isVisible() is False
    assert widget._curve_reconstructed_markers.isVisible() is False
