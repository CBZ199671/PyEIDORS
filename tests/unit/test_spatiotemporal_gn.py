"""Tests for batch spatiotemporal GN / 4D prior reconstruction."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy import sparse

from pyeidors.data.dynamic_sequence import DynamicMeasurementSequence
from pyeidors.inverse import (
    SPATIOTEMPORAL_GN_SCHEMA,
    graph_laplacian,
    solve_batch_spatiotemporal_gn,
    temporal_difference_operator,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "benchmark_dynamic_validation.py"
)


def _load_dynamic_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_dynamic_validation_t65", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_temporal_difference_operator_builds_first_and_second_order_dt() -> None:
    first = temporal_difference_operator(4, order=1)
    np.testing.assert_allclose(
        first.toarray(),
        np.array(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ]
        ),
    )

    second = temporal_difference_operator(4, order=2)
    np.testing.assert_allclose(
        second.toarray(),
        np.array(
            [
                [1.0, -2.0, 1.0, 0.0],
                [0.0, 1.0, -2.0, 1.0],
            ]
        ),
    )
    assert temporal_difference_operator(1, order=1).shape == (0, 1)


def test_spatiotemporal_gn_matches_rowwise_rm_when_temporal_lambda_is_zero() -> None:
    jacobian = np.array(
        [
            [1.0, 0.2],
            [0.3, 1.0],
            [0.7, -0.1],
        ],
        dtype=np.float64,
    )
    truth = np.array(
        [
            [0.2, 0.0],
            [0.3, 0.1],
            [0.1, 0.4],
            [-0.2, 0.2],
        ],
        dtype=np.float64,
    )
    residuals = truth @ jacobian.T
    residuals[:, 2] += 1.0e6
    mask = np.zeros_like(residuals, dtype=bool)
    mask[:, 2] = True
    weights = np.vstack(
        [
            np.array([1.0, 4.0, 0.25]),
            np.array([2.0, 3.0, 0.25]),
            np.array([3.0, 2.0, 0.25]),
            np.array([4.0, 1.0, 0.25]),
        ]
    )
    spatial_prior = sparse.diags([0.5, 1.5], 0, format="csr")

    result = solve_batch_spatiotemporal_gn(
        jacobian,
        residuals,
        spatial_prior=spatial_prior,
        lambda_s=0.3,
        lambda_t=0.0,
        temporal_order=1,
        channel_mask=mask,
        measurement_weights=weights,
        return_normal_operator=True,
    )

    assert result.metadata["schema"] == SPATIOTEMPORAL_GN_SCHEMA
    assert result.metadata["normal_equation_formula"].endswith("DtTDt_kron_I")
    assert result.metadata["lambda_s_squared"] == 0.09
    assert result.metadata["lambda_t_squared"] == 0.0
    assert result.metadata["measurement_weight_kinds"] == (
        "diagonal",
        "diagonal",
        "diagonal",
        "diagonal",
    )
    assert result.metadata["bad_channel_counts"] == (1, 1, 1, 1)
    assert result.normal_operator is not None
    assert result.normal_operator.shape == (truth.size, truth.size)
    assert result.rowwise_baseline is not None
    np.testing.assert_allclose(result.values, result.rowwise_baseline, atol=1.0e-10)
    assert result.metadata["rowwise_rm_comparison"]["l2_delta"] < 1.0e-10


def test_spatiotemporal_gn_temporal_prior_reduces_rowwise_jitter() -> None:
    jacobian = np.eye(3, dtype=np.float64)
    smooth_truth = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float64,
    )
    alternating_noise = np.array([0.2, -0.2, 0.2, -0.2, 0.2]).reshape(-1, 1)
    residuals = smooth_truth + alternating_noise

    result = solve_batch_spatiotemporal_gn(
        jacobian,
        residuals,
        spatial_prior=sparse.eye(3, format="csr"),
        lambda_s=0.0,
        lambda_t=1.0,
        temporal_order=1,
    )

    assert result.rowwise_baseline is not None
    smoothed_variation = np.linalg.norm(np.diff(result.values, axis=0))
    rowwise_variation = np.linalg.norm(np.diff(result.rowwise_baseline, axis=0))
    assert smoothed_variation < rowwise_variation
    assert result.metadata["rowwise_rm_comparison"]["l2_delta"] > 0.0
    assert result.metadata["online_hot_path_replaced"] is False


def test_spatiotemporal_gn_accepts_dynamic_sequence_and_second_order_dt() -> None:
    jacobian = np.eye(2, dtype=np.float64)
    residuals = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.4, 0.2],
            [0.6, 0.3],
        ],
        dtype=np.float64,
    )
    sequence = DynamicMeasurementSequence.from_arrays(
        residuals,
        t=np.array([0.0, 0.1, 0.2, 0.3]),
        sampling_rate_hz=10.0,
        reference_policy="difference_measurements_preprojected",
        stim_meas_signature="unit:identity",
        bad_channel_mask=np.zeros_like(residuals, dtype=bool),
        measurement_weights=np.ones_like(residuals),
        data_type="difference",
    )

    result = solve_batch_spatiotemporal_gn(
        jacobian,
        sequence,
        spatial_prior=sparse.eye(2, format="csr"),
        lambda_s=0.0,
        lambda_t=0.5,
        temporal_order=2,
    )

    assert result.shape == residuals.shape
    assert result.metadata["sequence_metadata"]["schema"].endswith("sequence-v1")
    assert result.metadata["temporal_order"] == 2
    assert result.metadata["temporal_operator_shape"] == (2, 4)
    assert np.isfinite(result.values).all()


def test_spatiotemporal_gn_runs_on_travelling_wave_fixture_with_rm_comparison() -> None:
    module = _load_dynamic_benchmark_module()
    fixture = module.build_travelling_wave_fixture(
        n_cells=8,
        n_frames=6,
        n_measurements=5,
        noise_std=0.0,
        seed=20260424,
    )
    mesh = fixture["mesh"]
    spatial_prior = graph_laplacian(mesh) + 1.0e-8 * sparse.eye(
        mesh.num_cells(),
        format="csr",
    )

    result = solve_batch_spatiotemporal_gn(
        fixture["jacobian"],
        fixture["measurements"],
        spatial_prior=spatial_prior,
        lambda_s=0.08,
        lambda_t=0.20,
        temporal_order=2,
    )

    assert result.shape == fixture["truth"].shape
    assert result.metadata["rowwise_rm_baseline"]["enabled"] is True
    assert result.metadata["rowwise_rm_comparison"]["enabled"] is True
    assert result.metadata["temporal_operator_shape"] == (4, 6)
    assert np.isfinite(result.metadata["rowwise_rm_comparison"]["relative_l2_delta"])
