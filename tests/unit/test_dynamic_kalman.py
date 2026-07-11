"""Tests for online Kalman and fixed-lag dynamic reconstruction prototypes."""

from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

import pyeidors.inverse.dynamic as dynamic_module
from pyeidors.inverse import (
    DYNAMIC_KALMAN_SCHEMA,
    build_one_step_rm,
    graph_laplacian,
    run_dynamic_kalman_filter,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "benchmark_dynamic_validation.py"
)


def _load_dynamic_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_dynamic_validation_t67", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_kalman_jacobian_mode_applies_measurement_weights_and_mask() -> None:
    truth = np.full((8, 1), 0.5, dtype=np.float64)
    noise = np.array([0.05, -0.03, 0.02, -0.04, 0.03, -0.02, 0.01, -0.01])
    observations = np.column_stack(
        [truth[:, 0] + noise, np.full(truth.shape[0], 1.0e6)]
    )
    channel_mask = np.zeros_like(observations, dtype=bool)
    channel_mask[:, 1] = True
    weights = np.column_stack(
        [np.linspace(1.0, 4.0, truth.shape[0]), np.ones(truth.shape[0])]
    )
    jacobian = np.array([[1.0], [1.0]], dtype=np.float64)

    result = run_dynamic_kalman_filter(
        jacobian,
        observations,
        process_noise=0.05,
        measurement_noise=0.02,
        initial_covariance=1.0,
        channel_mask=channel_mask,
        measurement_weights=weights,
        timestamps=np.arange(truth.shape[0], dtype=np.float64) * 0.01,
    )

    raw_rmse = np.sqrt(np.mean((observations[:, :1] - truth) ** 2))
    filtered_rmse = np.sqrt(np.mean((result.filtered - truth) ** 2))
    assert result.metadata["schema"] == DYNAMIC_KALMAN_SCHEMA
    assert result.metadata["measurement_contract_applied"] is True
    assert result.metadata["measurement_weight_kinds"] == ("diagonal",) * truth.shape[0]
    assert result.metadata["bad_channel_counts"] == (1,) * truth.shape[0]
    assert filtered_rmse < raw_rmse
    assert result.metadata["default_enabled"] is False


def test_fixed_lag_smoother_records_latency_and_improves_constant_velocity_state() -> (
    None
):
    times = np.arange(10, dtype=np.float64)
    true_position = times.copy()
    observations = (
        true_position
        + np.array([0.2, -0.4, 0.3, -0.2, 0.1, -0.3, 0.25, -0.2, 0.1, -0.1])
    ).reshape(-1, 1)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    jacobian = np.array([[1.0, 0.0]], dtype=np.float64)

    result = run_dynamic_kalman_filter(
        jacobian,
        observations,
        transition=transition,
        process_noise=np.diag([1.0e-3, 1.0e-4]),
        measurement_noise=0.08,
        initial_state=np.array([0.0, 1.0]),
        initial_covariance=np.diag([1.0, 0.5]),
        fixed_lag=3,
        timestamps=times * 0.02,
    )

    filtered_rmse = np.sqrt(np.mean((result.filtered[:, 0] - true_position) ** 2))
    smoothed_rmse = np.sqrt(np.mean((result.smoothed[:, 0] - true_position) ** 2))
    assert result.metadata["smoother"]["enabled"] is True
    assert result.metadata["latency_frames"] == 3
    assert result.metadata["latency_seconds"] == 0.06
    assert smoothed_rmse <= filtered_rmse


def test_v674_candidate_constrained_reject_removes_isolated_spike() -> None:
    observations = np.array([[0.0], [0.0], [0.0], [20.0], [0.0], [0.0]])
    candidates = np.array([False, False, False, True, False, False])

    result = run_dynamic_kalman_filter(
        np.array([[1.0]]),
        observations,
        process_noise=0.01,
        measurement_noise=0.01,
        initial_state=np.array([0.0]),
        initial_covariance=0.1,
        innovation_gate="reject",
        innovation_gate_candidates=candidates,
        innovation_nis_threshold=9.0,
    )

    assert result.filtered[3, 0] == pytest.approx(result.predicted[3, 0])
    assert result.metadata["innovation_gate_actions"][3] == "reject"
    assert result.metadata["innovation_gate_count"] == 1
    assert result.metadata["innovation_reject_count"] == 1
    assert abs(result.filtered[4, 0]) < 1.0e-12


def test_v674_noncandidate_persistent_step_matches_legacy_filter() -> None:
    observations = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]])
    kwargs = {
        "process_noise": 0.05,
        "measurement_noise": 0.02,
        "initial_state": np.array([0.0]),
        "initial_covariance": 0.2,
    }
    legacy = run_dynamic_kalman_filter(np.array([[1.0]]), observations, **kwargs)
    guarded = run_dynamic_kalman_filter(
        np.array([[1.0]]),
        observations,
        innovation_gate="reject",
        innovation_gate_candidates=np.zeros(observations.shape[0], dtype=bool),
        innovation_nis_threshold=1.0,
        **kwargs,
    )

    np.testing.assert_array_equal(guarded.filtered, legacy.filtered)
    np.testing.assert_array_equal(guarded.predicted, legacy.predicted)
    assert guarded.filtered[-1, 0] > 0.9
    assert guarded.metadata["innovation_gate_count"] == 0


def test_v674_variance_inflation_softens_candidate_update() -> None:
    observations = np.array([[0.0], [0.0], [5.0], [0.0]])
    candidates = np.array([False, False, True, False])
    common = {
        "process_noise": 0.01,
        "measurement_noise": 0.01,
        "initial_state": np.array([0.0]),
        "initial_covariance": 0.1,
    }
    legacy = run_dynamic_kalman_filter(np.array([[1.0]]), observations, **common)
    robust = run_dynamic_kalman_filter(
        np.array([[1.0]]),
        observations,
        innovation_gate="inflate",
        innovation_gate_candidates=candidates,
        innovation_nis_threshold=4.0,
        innovation_max_variance_inflation=100.0,
        **common,
    )

    assert abs(robust.filtered[2, 0]) < abs(legacy.filtered[2, 0])
    assert robust.metadata["innovation_gate_actions"][2] == "inflate"
    assert robust.metadata["innovation_variance_inflations"][2] > 1.0
    assert robust.metadata["innovation_inflate_count"] == 1


def test_v674_enabled_gate_requires_threshold_and_matching_candidates() -> None:
    observations = np.zeros((3, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="innovation_nis_threshold is required"):
        run_dynamic_kalman_filter(
            np.array([[1.0]]),
            observations,
            innovation_gate="reject",
            innovation_gate_candidates=[False, True, False],
        )
    with pytest.raises(ValueError, match="length must match"):
        run_dynamic_kalman_filter(
            np.array([[1.0]]),
            observations,
            innovation_gate="inflate",
            innovation_gate_candidates=[True],
            innovation_nis_threshold=3.0,
        )


def test_rm_observation_shortcut_keeps_online_hot_path_counters_zero() -> None:
    truth = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.4, 0.3], [0.5, 0.4]],
        dtype=np.float64,
    )
    observations = truth + np.array(
        [[0.05, -0.03], [-0.02, 0.04], [0.03, -0.02], [-0.01, 0.02]],
        dtype=np.float64,
    )
    rm = np.eye(2, dtype=np.float64)

    result = run_dynamic_kalman_filter(
        rm,
        observations,
        observation_mode="rm_observation",
        process_noise=0.04,
        measurement_noise=np.array([0.02, 0.02]),
        initial_covariance=1.0,
        fixed_lag=1,
        sampling_rate_hz=100.0,
    )

    assert result.shape == truth.shape
    assert result.metadata["online_hot_path"] == "rm_observation_plus_kalman"
    assert result.metadata["forward_solve_count"] == 0
    assert result.metadata["adjoint_solve_count"] == 0
    assert result.metadata["ksp_solve_count"] == 0
    assert result.metadata["jacobian_rebuild_count"] == 0
    assert result.metadata["online_hot_path_replaced"] is False
    assert result.metadata["latency_seconds"] == 0.01


def test_v292_dynamic_frame_results_direct_fill_without_vstack(monkeypatch) -> None:
    def fail_vstack(*_args, **_kwargs):
        raise AssertionError("dynamic frame-row assembly must not call np.vstack")

    monkeypatch.setattr(dynamic_module.np, "vstack", fail_vstack)

    observations = np.array(
        [[0.0, 0.1], [0.2, 0.2], [0.4, 0.3], [0.6, 0.4]],
        dtype=np.float64,
    )
    jacobian = np.eye(2, dtype=np.float64)
    jacobian_result = dynamic_module.run_dynamic_kalman_filter(
        jacobian,
        observations,
        process_noise=0.03,
        measurement_noise=0.04,
        initial_covariance=1.0,
        fixed_lag=1,
    )
    rm_result = dynamic_module.run_dynamic_kalman_filter(
        np.eye(2, dtype=np.float64),
        observations,
        observation_mode="rm_observation",
        process_noise=0.03,
        measurement_noise=0.04,
        initial_covariance=1.0,
        fixed_lag=0,
    )
    gn_result = dynamic_module.solve_batch_spatiotemporal_gn(
        jacobian,
        observations,
        spatial_prior=sparse.eye(2, format="csr"),
        lambda_s=0.05,
        lambda_t=0.0,
        rowwise_rm_baseline=True,
    )

    assert jacobian_result.filtered.shape == observations.shape
    assert jacobian_result.smoothed.shape == observations.shape
    assert rm_result.predicted.shape == observations.shape
    assert gn_result.rowwise_baseline is not None
    assert gn_result.rowwise_baseline.shape == observations.shape
    for obj in (
        dynamic_module.run_dynamic_kalman_filter,
        dynamic_module._kalman_observations,
        dynamic_module._fixed_lag_smoother,
        dynamic_module._rowwise_rm_baseline,
    ):
        assert "np.vstack" not in inspect.getsource(obj)


def test_v306_dynamic_rhs_vectors_direct_fill_without_concatenate() -> None:
    observations = np.array(
        [[0.1, -0.1], [0.2, 0.0], [0.3, 0.1]],
        dtype=np.float64,
    )
    jacobian = np.eye(2, dtype=np.float64)

    gn_result = dynamic_module.solve_batch_spatiotemporal_gn(
        jacobian,
        observations,
        spatial_prior=sparse.eye(2, format="csr"),
        lambda_s=0.0,
        lambda_t=0.0,
    )
    tv_result = dynamic_module.solve_spatiotemporal_tv_huber(
        jacobian,
        observations,
        lambda_s=0.0,
        lambda_t=0.0,
        max_outer_iterations=1,
    )

    assert gn_result.values.shape == observations.shape
    assert tv_result.values.shape == observations.shape
    assert gn_result.values.flags.c_contiguous
    assert tv_result.values.flags.c_contiguous
    for obj in (
        dynamic_module.solve_batch_spatiotemporal_gn,
        dynamic_module.solve_spatiotemporal_tv_huber,
    ):
        assert "np.concatenate" not in inspect.getsource(obj)


def test_v494_dynamic_inverse_finite_guards_use_bounded_scanner() -> None:
    checked = (
        dynamic_module._as_sparse_difference,
        dynamic_module._optional_timestamps,
        dynamic_module._jacobian_matrix,
        dynamic_module._resolve_spatial_prior,
        dynamic_module._solve_block_system,
    )
    old_patterns = (
        "np.isfinite(matrix.data).all()",
        "np.isfinite(arr).all()",
        "np.isfinite(array).all()",
        "np.isfinite(out).all()",
    )

    for func in checked:
        source = inspect.getsource(func)
        assert "_all_finite_values(" in source
        for old_pattern in old_patterns:
            assert old_pattern not in source


def test_v409_dynamic_robust_huber_helpers_avoid_where_replacement_arrays() -> None:
    values = np.array([[-2.0, -0.5, 0.0], [0.5, 2.0, 4.0]], dtype=np.float64)
    huber_delta = 1.25
    epsilon = 0.2
    abs_values = np.sqrt(values * values + epsilon * epsilon)

    weights = dynamic_module._robust_irls_weights(
        values,
        penalty="huber",
        huber_delta=huber_delta,
        epsilon=epsilon,
    )
    penalties = dynamic_module._robust_penalty_values(
        values,
        penalty="huber",
        huber_delta=huber_delta,
        epsilon=epsilon,
    )

    np.testing.assert_allclose(weights, np.minimum(1.0, huber_delta / abs_values))
    np.testing.assert_allclose(
        penalties,
        np.where(
            abs_values <= huber_delta,
            0.5 * abs_values * abs_values,
            huber_delta * (abs_values - 0.5 * huber_delta),
        ),
    )
    np.testing.assert_allclose(
        dynamic_module._robust_irls_weights(
            values,
            penalty="tv",
            huber_delta=huber_delta,
            epsilon=epsilon,
        ),
        1.0 / abs_values,
    )

    for obj in (
        dynamic_module._robust_irls_weights,
        dynamic_module._robust_penalty_values,
    ):
        source = inspect.getsource(obj)
        assert "np.where" not in source
        assert "arr * arr" not in source
        assert "np.square" in source


def test_v410_dynamic_temporal_roi_paths_avoid_roi_submatrix_copies() -> None:
    values = np.array(
        [[0.0, 0.2, 0.4], [0.3, 0.1, 0.5], [0.6, -0.2, 0.9]],
        dtype=np.float64,
    )
    dt = sparse.csr_matrix(
        np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]], dtype=np.float64)
    )
    roi_mask = np.array([True, False, True], dtype=bool)
    huber_delta = 0.35
    epsilon = 0.05
    temporal_diffs = np.asarray(dt @ values)

    actual = dynamic_module._robust_spatiotemporal_objective(
        [sparse.csr_matrix((3, 3), dtype=np.float64) for _ in range(values.shape[0])],
        [np.zeros(3, dtype=np.float64) for _ in range(values.shape[0])],
        values,
        spatial_difference_matrix=sparse.csr_matrix((0, values.shape[1])),
        dt=dt,
        roi_mask=roi_mask,
        lambda_s=0.0,
        lambda_t=1.0,
        penalty="huber",
        huber_delta=huber_delta,
        epsilon=epsilon,
    )
    expected = float(
        np.sum(
            dynamic_module._robust_penalty_values(
                temporal_diffs[:, roi_mask],
                penalty="huber",
                huber_delta=huber_delta,
                epsilon=epsilon,
            )
        )
    )
    assert actual == pytest.approx(expected)

    weights = np.array([[0.0, 4.0, 2.0], [3.0, 0.0, 5.0]], dtype=np.float64)
    assert dynamic_module._weight_range(weights, column_mask=roi_mask) == (2.0, 5.0)
    masked_weights = weights.copy()
    dynamic_module._zero_non_roi_columns_in_place(masked_weights, roi_mask)
    np.testing.assert_allclose(
        masked_weights,
        np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 5.0]], dtype=np.float64),
    )

    for obj in (
        dynamic_module._temporal_robust_normal,
        dynamic_module._robust_spatiotemporal_objective,
    ):
        source = inspect.getsource(obj)
        assert "weights[:, roi_mask]" not in source
        assert "weights[:, ~roi_mask]" not in source
        assert "temporal_diffs[:, roi_mask]" not in source
    helper_source = inspect.getsource(dynamic_module._zero_non_roi_columns_in_place)
    assert "for col_idx, enabled in enumerate" in helper_source
    assert "arr[arr > 0.0]" not in inspect.getsource(dynamic_module._weight_range)


def test_v462_dynamic_roi_index_range_uses_minmax_reduction() -> None:
    source = inspect.getsource(dynamic_module._roi_mask)
    helper_source = inspect.getsource(dynamic_module._indices_within_range)

    assert "(indices < 0) | (indices >=" not in source
    assert "_indices_within_range(indices" in source
    assert "np.min(values)" in helper_source
    assert "np.max(values)" in helper_source

    mask = dynamic_module._roi_mask([0, 2, 4], n_parameters=5)
    assert mask.tolist() == [True, False, True, False, True]

    empty = dynamic_module._roi_mask([], n_parameters=3)
    assert empty.tolist() == [False, False, False]

    with pytest.raises(ValueError, match="out of range"):
        dynamic_module._roi_mask([0, 3], n_parameters=3)
    with pytest.raises(ValueError, match="out of range"):
        dynamic_module._roi_mask([-1, 1], n_parameters=3)


def test_v463_dynamic_roi_difference_rows_avoid_index_subset() -> None:
    source = inspect.getsource(dynamic_module._restrict_difference_rows_to_roi)
    helper_source = inspect.getsource(dynamic_module._all_mask_indices_enabled)

    assert "roi_mask[cols]" not in source
    assert "roi[cols]" not in source
    assert "_all_mask_indices_enabled(roi, cols)" in source
    assert "for raw_idx in np.asarray(indices).reshape(-1)" in helper_source

    matrix = sparse.csr_matrix(
        np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    restricted = dynamic_module._restrict_difference_rows_to_roi(
        matrix,
        np.array([True, True, False, True], dtype=bool),
    )

    expected = sparse.csr_matrix(np.array([[-1.0, 1.0, 0.0, 0.0]], dtype=np.float64))
    np.testing.assert_allclose(restricted.toarray(), expected.toarray())


def test_v467_dynamic_frame_finite_checks_use_bounded_scan() -> None:
    frame_source = inspect.getsource(dynamic_module._frame_batch)
    initial_source = inspect.getsource(dynamic_module._initial_dynamic_state)
    helper_source = inspect.getsource(dynamic_module._all_finite_values)

    assert "np.isfinite(arr).all()" not in frame_source
    assert "np.isfinite(arr).all()" not in initial_source
    assert "_all_finite_values(arr)" in frame_source
    assert "_all_finite_values(arr)" in initial_source
    assert "out=work_view" in helper_source

    frames = dynamic_module._frame_batch(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        name="frames",
    )
    np.testing.assert_allclose(frames, [[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(FloatingPointError, match="non-finite"):
        dynamic_module._frame_batch(
            np.array([[1.0, np.nan]], dtype=np.float64),
            name="frames",
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        dynamic_module._initial_dynamic_state(
            np.array([[np.inf, 1.0]], dtype=np.float64),
            fallback=np.zeros((1, 2), dtype=np.float64),
            n_frames=1,
            n_parameters=2,
        )


def test_v468_dynamic_matrix_finite_checks_use_bounded_scan() -> None:
    for obj in (
        dynamic_module._rm_matrix,
        dynamic_module._transition_matrix,
        dynamic_module._kalman_initial_state,
        dynamic_module._covariance_matrix,
    ):
        source = inspect.getsource(obj)
        assert "np.isfinite(matrix).all()" not in source
        assert "np.isfinite(arr).all()" not in source
        assert "_all_finite_values(" in source

    rm = dynamic_module._rm_matrix(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        n_measurements=2,
    )
    np.testing.assert_allclose(rm, np.eye(2))

    with pytest.raises(FloatingPointError, match="non-finite"):
        dynamic_module._rm_matrix(
            np.array([[1.0, np.nan]], dtype=np.float64),
            n_measurements=2,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        dynamic_module._transition_matrix(
            np.array([[1.0, np.inf], [0.0, 1.0]], dtype=np.float64),
            n_state=2,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        dynamic_module._kalman_initial_state(
            np.array([0.0, np.nan], dtype=np.float64),
            n_state=2,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        dynamic_module._covariance_matrix(
            np.array([[1.0, 0.0], [0.0, np.nan]], dtype=np.float64),
            n=2,
            name="covariance",
        )


def test_v514_dynamic_kalman_identity_payloads_are_direct_filled_and_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked = (
        dynamic_module.run_dynamic_kalman_filter,
        dynamic_module._kalman_observations,
        dynamic_module._transition_matrix,
        dynamic_module._kalman_covariance,
        dynamic_module._resolve_kalman_noise,
        dynamic_module._resolve_measurement_noise,
        dynamic_module._covariance_matrix,
    )
    for func in checked:
        source = inspect.getsource(func)
        assert "np.eye" not in source
        assert "_dense_identity(" in source
    assert "identity_state - kh" not in inspect.getsource(
        dynamic_module.run_dynamic_kalman_filter
    )

    observations = np.array(
        [[0.0, 0.1], [0.2, 0.2], [0.4, 0.3]],
        dtype=np.float64,
    )
    identity_model = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    def _fail_eye(*_args, **_kwargs):
        raise AssertionError("dynamic Kalman path must not call np.eye")

    monkeypatch.setattr(dynamic_module.np, "eye", _fail_eye)

    result = dynamic_module.run_dynamic_kalman_filter(
        identity_model,
        observations,
        process_noise=0.03,
        measurement_noise=0.04,
        initial_covariance=1.0,
        fixed_lag=1,
    )
    assert result.filtered.shape == observations.shape
    assert np.isfinite(result.filtered).all()

    projected, h_stack, _, _ = dynamic_module._kalman_observations(
        identity_model,
        observations,
        mode="rm_observation",
        channel_mask=None,
        measurement_weights=None,
    )
    assert projected.shape == observations.shape
    assert h_stack[0] is h_stack[1]
    np.testing.assert_allclose(h_stack[0], identity_model)


def test_v517_dynamic_vector_covariance_direct_fills_dense_diagonal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    covariance_source = inspect.getsource(dynamic_module._covariance_matrix)
    helper_source = inspect.getsource(dynamic_module._dense_diagonal)
    assert "np.diag(arr)" not in covariance_source
    assert "_dense_diagonal(arr)" in covariance_source
    assert "matrix.reshape(-1)[:: size + 1] = diagonal" in helper_source

    def _fail_diag(*_args, **_kwargs):
        raise AssertionError("vector covariance must not call np.diag")

    monkeypatch.setattr(dynamic_module.np, "diag", _fail_diag)
    matrix = dynamic_module._covariance_matrix(
        np.array([0.2, 0.4], dtype=np.float64),
        n=2,
        name="covariance",
    )

    expected = np.zeros((2, 2), dtype=np.float64)
    expected.reshape(-1)[::3] = np.array([0.2, 0.4], dtype=np.float64)
    np.testing.assert_allclose(matrix, expected)


def test_v469_dynamic_temporal_weighted_normal_uses_positive_reduction() -> None:
    source = inspect.getsource(dynamic_module._temporal_weighted_normal)
    helper_source = inspect.getsource(dynamic_module._has_positive_value)

    assert "np.any(column_weights > 0.0)" not in source
    assert "_has_positive_value(column_weights)" in source
    assert "np.max(arr)" in helper_source

    dt = sparse.csr_matrix(
        np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]], dtype=np.float64)
    )
    weights = np.array([[0.0, 2.0], [0.0, 3.0]], dtype=np.float64)

    normal = dynamic_module._temporal_weighted_normal(dt, weights).toarray()

    expected_param_1 = dt.T @ sparse.diags(weights[:, 1], 0, format="csr") @ dt
    expected = np.zeros((6, 6), dtype=np.float64)
    expected[np.ix_([1, 3, 5], [1, 3, 5])] = expected_param_1.toarray()
    np.testing.assert_allclose(normal, expected)
    assert not dynamic_module._has_positive_value(np.zeros(0, dtype=np.float64))


def test_kalman_accepts_q_and_r_estimation_hooks() -> None:
    q_frames: list[int] = []
    r_frames: list[int] = []

    def q_hook(context):
        q_frames.append(int(context["frame_index"]))
        return 0.01 + 0.001 * int(context["frame_index"])

    def r_hook(context):
        r_frames.append(int(context["frame_index"]))
        return np.array([[0.02 + 0.001 * int(context["frame_index"])]])

    observations = np.array([[0.0], [0.1], [0.2]], dtype=np.float64)
    result = run_dynamic_kalman_filter(
        np.array([[1.0]], dtype=np.float64),
        observations,
        process_noise_hook=q_hook,
        measurement_noise_hook=r_hook,
        initial_covariance=1.0,
    )

    assert q_frames == [0, 1, 2]
    assert r_frames == [0, 1, 2]
    assert result.metadata["process_noise_hook_used"] is True
    assert result.metadata["measurement_noise_hook_used"] is True
    assert result.metadata["process_noise_sources"] == ("hook", "hook", "hook")
    assert result.metadata["measurement_noise_sources"] == ("hook", "hook", "hook")


def test_kalman_rm_shortcut_runs_on_travelling_wave_fixture_with_t69_metrics() -> None:
    module = _load_dynamic_benchmark_module()
    fixture = module.build_travelling_wave_fixture(
        n_cells=8,
        n_frames=6,
        n_measurements=5,
        noise_std=1.0e-4,
        seed=20260426,
    )
    mesh = fixture["mesh"]
    prior = graph_laplacian(mesh) + 1.0e-8 * sparse.eye(mesh.num_cells(), format="csr")
    rm = build_one_step_rm(
        fixture["jacobian"],
        regularization=prior,
        lambda_=0.08,
        mode="laplace",
        return_metadata=True,
    )

    result = run_dynamic_kalman_filter(
        rm.rm,
        fixture["measurements"],
        observation_mode="rm_observation",
        process_noise=0.03,
        measurement_noise=0.04,
        initial_covariance=1.0,
        fixed_lag=2,
        timestamps=fixture["times"],
    )
    metrics = module.dynamic_fidelity_metrics(
        fixture["truth"],
        result.values,
        clean_measurements=fixture["clean_measurements"],
        noisy_measurements=fixture["measurements"],
        positions=fixture["positions"],
        times=fixture["times"],
        onset_fraction=0.30,
    )

    assert result.metadata["requires_t69_gate_before_default"] is True
    assert result.metadata["default_enabled"] is False
    assert result.metadata["smoother"]["enabled"] is True
    assert np.isfinite(metrics["propagation_speed_abs_error"])
    assert np.isfinite(metrics["peak_time_mean_abs_error"])
