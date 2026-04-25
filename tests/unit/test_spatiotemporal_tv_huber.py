"""Tests for spatiotemporal TV / Huber dynamic reconstruction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from pyeidors.inverse import (
    SPATIOTEMPORAL_TV_HUBER_SCHEMA,
    graph_laplacian,
    solve_spatiotemporal_tv_huber,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "benchmark_dynamic_validation.py"
)


def _load_dynamic_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_dynamic_validation_t66", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_huber_temporal_prior_preserves_abrupt_onset_better_than_l2() -> None:
    jacobian = np.ones((1, 1), dtype=np.float64)
    truth = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64).reshape(
        -1, 1
    )

    result = solve_spatiotemporal_tv_huber(
        jacobian,
        truth,
        lambda_s=0.0,
        lambda_t=3.0,
        huber_delta=0.05,
        temporal_order=1,
        max_outer_iterations=8,
        tolerance=1.0e-8,
    )

    huber_error = np.linalg.norm(result.values - truth)
    l2_error = np.linalg.norm(result.l2_baseline - truth)
    huber_jump = float(result.values[3, 0] - result.values[2, 0])
    l2_jump = float(result.l2_baseline[3, 0] - result.l2_baseline[2, 0])
    assert result.metadata["schema"] == SPATIOTEMPORAL_TV_HUBER_SCHEMA
    assert huber_error < l2_error
    assert huber_jump > l2_jump
    assert result.metadata["t65_l2_comparison"]["enabled"] is True
    assert result.metadata["online_hot_path_replaced"] is False


def test_spatiotemporal_tv_huber_roi_restricts_temporal_penalty() -> None:
    jacobian = np.eye(2, dtype=np.float64)
    residuals = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = solve_spatiotemporal_tv_huber(
        jacobian,
        residuals,
        lambda_s=0.0,
        lambda_t=4.0,
        huber_delta=0.05,
        roi_mask=np.array([True, False]),
        temporal_order=1,
    )

    assert result.metadata["roi_enabled"] is True
    assert result.metadata["roi_parameter_count"] == 1
    np.testing.assert_allclose(result.values[:, 1], residuals[:, 1], atol=1.0e-10)
    assert np.linalg.norm(result.l2_baseline[:, 1] - residuals[:, 1]) > 1.0e-3


def test_spatiotemporal_tv_huber_applies_measurement_weights_and_mask() -> None:
    jacobian = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    truth = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.4]], dtype=np.float64)
    residuals = truth @ jacobian.T
    residuals[:, 2] = 1.0e6
    channel_mask = np.zeros_like(residuals, dtype=bool)
    channel_mask[:, 2] = True
    weights = np.vstack(
        [
            np.array([1.0, 2.0, 0.1]),
            np.array([2.0, 3.0, 0.1]),
            np.array([3.0, 4.0, 0.1]),
        ]
    )

    result = solve_spatiotemporal_tv_huber(
        jacobian,
        residuals,
        lambda_s=0.0,
        lambda_t=0.1,
        channel_mask=channel_mask,
        measurement_weights=weights,
        penalty="tv",
        temporal_order=2,
        max_outer_iterations=3,
    )

    assert result.metadata["measurement_contract_applied"] is True
    assert result.metadata["measurement_weight_kinds"] == (
        "diagonal",
        "diagonal",
        "diagonal",
    )
    assert result.metadata["bad_channel_counts"] == (1, 1, 1)
    assert result.metadata["temporal_order"] == 2
    assert np.isfinite(result.values).all()


def test_spatiotemporal_tv_huber_compares_against_t65_on_travelling_wave_fixture() -> (
    None
):
    module = _load_dynamic_benchmark_module()
    fixture = module.build_travelling_wave_fixture(
        n_cells=10,
        n_frames=8,
        n_measurements=6,
        noise_std=1.0e-4,
        seed=20260425,
    )
    mesh = fixture["mesh"]
    spatial_prior = graph_laplacian(mesh) + 1.0e-8 * sparse.eye(
        mesh.num_cells(),
        format="csr",
    )

    result = solve_spatiotemporal_tv_huber(
        fixture["jacobian"],
        fixture["measurements"],
        spatial_graph=mesh,
        lambda_s=0.05,
        lambda_t=0.15,
        huber_delta=0.03,
        temporal_order=2,
        max_outer_iterations=5,
    )
    l2_result = module.solve_batch_spatiotemporal_gn(
        fixture["jacobian"],
        fixture["measurements"],
        spatial_prior=spatial_prior,
        lambda_s=0.05,
        lambda_t=0.15,
        temporal_order=2,
        rowwise_rm_baseline=False,
    )
    tv_fidelity = module.dynamic_fidelity_metrics(
        fixture["truth"],
        result.values,
        clean_measurements=fixture["clean_measurements"],
        noisy_measurements=fixture["measurements"],
        positions=fixture["positions"],
        times=fixture["times"],
        onset_fraction=0.30,
    )
    l2_fidelity = module.dynamic_fidelity_metrics(
        fixture["truth"],
        l2_result.values,
        clean_measurements=fixture["clean_measurements"],
        noisy_measurements=fixture["measurements"],
        positions=fixture["positions"],
        times=fixture["times"],
        onset_fraction=0.30,
    )

    assert result.shape == fixture["truth"].shape
    assert result.metadata["t65_l2_baseline"]["enabled"] is True
    assert result.metadata["t65_l2_baseline"]["schema"].endswith("gn-v1")
    assert result.metadata["t65_l2_comparison"]["enabled"] is True
    assert np.isfinite(result.metadata["t65_l2_comparison"]["relative_l2_delta"])
    assert np.isfinite(tv_fidelity["peak_time_mean_abs_error"])
    assert np.isfinite(l2_fidelity["peak_time_mean_abs_error"])
