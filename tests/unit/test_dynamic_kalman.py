"""Tests for online Kalman and fixed-lag dynamic reconstruction prototypes."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

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
