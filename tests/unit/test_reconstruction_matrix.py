"""Tests for online reconstruction-matrix helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pyeidors.data.difference import normalize_time_difference
from pyeidors.inverse.reconstruction_matrix import (
    build_one_step_rm,
    reconstruct_difference,
)


def test_build_one_step_rm_tikhonov_matches_dense_formula() -> None:
    jacobian = np.array([[1.0, 0.5], [0.0, 2.0], [1.0, -1.0]], dtype=float)
    lam = 0.25

    result = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        return_metadata=True,
    )
    expected = np.linalg.solve(
        jacobian.T @ jacobian + lam**2 * np.eye(2),
        jacobian.T,
    )

    np.testing.assert_allclose(result.rm, expected)
    assert result.shape == (2, 3)
    assert result.metadata["mode"] == "tikhonov"
    assert result.metadata["form"] == "param"
    assert result.metadata["regularization_source"] == "identity"
    assert result.metadata["condition_estimate"] >= 1.0


def test_build_one_step_rm_noser_uses_diag_jtj_regularization() -> None:
    jacobian = np.array([[1.0, 2.0], [3.0, 0.5], [0.0, 1.0]], dtype=float)
    lam = 0.1

    rm = build_one_step_rm(jacobian, lambda_=lam, mode="noser")
    noser = np.diag(np.sum(jacobian * jacobian, axis=0))
    expected = np.linalg.solve(jacobian.T @ jacobian + lam**2 * noser, jacobian.T)

    np.testing.assert_allclose(rm, expected)


def test_build_one_step_rm_laplace_accepts_sparse_regularization() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25], [1.0, 1.0, 0.0]],
        dtype=float,
    )
    laplace = sparse.csr_matrix(
        np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    )
    lam = 0.2

    result = build_one_step_rm(
        jacobian,
        regularization=laplace,
        lambda_=lam,
        mode="laplace",
        return_metadata=True,
    )
    expected = np.linalg.solve(
        jacobian.T @ jacobian + lam**2 * laplace.toarray(),
        jacobian.T,
    )

    np.testing.assert_allclose(result.rm, expected)
    assert result.metadata["regularization_source"] == "provided_laplace"


def test_build_one_step_rm_measurement_form_matches_param_for_tikhonov() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5, -0.25], [0.5, 2.0, -1.0, 0.75]],
        dtype=float,
    )
    lam = 0.3

    param_rm = build_one_step_rm(jacobian, lambda_=lam, mode="tikhonov")
    measurement = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        form="measurement",
        return_metadata=True,
    )

    np.testing.assert_allclose(measurement.rm, param_rm, rtol=1e-10, atol=1e-12)
    assert measurement.metadata["form"] == "measurement"
    assert measurement.metadata["inversion_dimension"] == "measurement"
    assert measurement.metadata["system_shape"] == (2, 2)
    assert measurement.metadata["prior_inverse_solver"] == "solve"


def test_build_one_step_rm_measurement_form_matches_param_for_noser() -> None:
    jacobian = np.array(
        [[1.0, 2.0, 0.5], [3.0, 0.5, -1.0]],
        dtype=float,
    )
    lam = 0.15

    param_rm = build_one_step_rm(jacobian, lambda_=lam, mode="noser")
    measurement_rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="noser",
        form="measurement",
    )

    np.testing.assert_allclose(measurement_rm, param_rm, rtol=1e-10, atol=1e-12)


def test_build_one_step_rm_measurement_form_matches_param_for_spd_laplace() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.25]],
        dtype=float,
    )
    laplace = sparse.csr_matrix(
        np.array(
            [
                [1.5, -1.0, 0.0],
                [-1.0, 2.5, -1.0],
                [0.0, -1.0, 1.5],
            ],
            dtype=float,
        )
    )
    lam = 0.2

    param_rm = build_one_step_rm(
        jacobian,
        regularization=laplace,
        lambda_=lam,
        mode="laplace",
    )
    measurement_rm = build_one_step_rm(
        jacobian,
        regularization=laplace,
        lambda_=lam,
        mode="laplace",
        form="measurement",
    )

    np.testing.assert_allclose(measurement_rm, param_rm, rtol=1e-10, atol=1e-12)


def test_build_one_step_rm_measurement_form_accepts_measurement_regularization() -> (
    None
):
    jacobian = np.array([[1.0, 0.5, 0.0], [0.25, -1.0, 2.0]], dtype=float)
    rn = np.diag([2.0, 3.0])
    lam = 0.4

    rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        form="measurement",
        measurement_regularization=rn,
        return_metadata=True,
    )
    expected = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + lam**2 * rn)

    np.testing.assert_allclose(rm.rm, expected)
    assert rm.metadata["measurement_regularization_source"] == "provided"


def test_build_one_step_rm_applies_bad_channels_and_weights_consistently() -> None:
    jacobian = np.array(
        [[1.0, 0.0], [5.0, 5.0], [0.0, 2.0]],
        dtype=float,
    )
    dv = np.array([2.0, 100.0, -1.0], dtype=float)
    mask = np.array([False, True, False], dtype=bool)
    weights = np.array([4.0, 9.0, 1.0], dtype=float)
    lam = 0.2

    result = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        channel_mask=mask,
        measurement_weights=weights,
        return_metadata=True,
    )
    recon = reconstruct_difference(
        result.rm,
        dv,
        normalize=False,
        channel_mask=mask,
        measurement_weights=weights,
    )

    sqrt_w = np.diag(np.sqrt([4.0, 0.0, 1.0]))
    masked_j = jacobian.copy()
    masked_j[1, :] = 0.0
    masked_dv = dv.copy()
    masked_dv[1] = 0.0
    weighted_j = sqrt_w @ masked_j
    weighted_dv = sqrt_w @ masked_dv
    expected_rm = np.linalg.solve(
        weighted_j.T @ weighted_j + lam**2 * np.eye(2),
        weighted_j.T,
    )

    np.testing.assert_allclose(result.rm, expected_rm)
    np.testing.assert_allclose(recon, expected_rm @ weighted_dv)
    assert result.metadata["bad_channel_count"] == 1
    assert result.metadata["measurement_weight_kind"] == "diagonal"


def test_build_one_step_rm_measurement_form_honors_same_weight_contract() -> None:
    jacobian = np.array(
        [[1.0, 0.0, 0.5], [0.0, 2.0, -1.0], [1.0, -0.5, 0.25]],
        dtype=float,
    )
    mask = np.array([False, False, True], dtype=bool)
    weights = np.array([2.0, 0.5, 7.0], dtype=float)
    lam = 0.3

    param_rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        channel_mask=mask,
        measurement_weights=weights,
    )
    measurement_rm = build_one_step_rm(
        jacobian,
        lambda_=lam,
        mode="tikhonov",
        form="measurement",
        channel_mask=mask,
        measurement_weights=weights,
    )

    np.testing.assert_allclose(measurement_rm, param_rm, rtol=1e-10, atol=1e-12)


def test_build_one_step_rm_rejects_invalid_form_and_measurement_rn_shape() -> None:
    with pytest.raises(ValueError, match="form must be"):
        build_one_step_rm(np.eye(2), lambda_=0.1, form="bad")
    with pytest.raises(ValueError, match="measurement_regularization"):
        build_one_step_rm(
            np.eye(2),
            lambda_=0.1,
            form="measurement",
            measurement_regularization=np.eye(3),
        )


def test_reconstruct_difference_applies_rm_to_normalized_time_difference() -> None:
    rm = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 4.0]], dtype=float)
    reference = np.array([2.0, 4.0, -2.0], dtype=float)
    target = np.array([3.0, 8.0, -1.0], dtype=float)

    expected_dv = normalize_time_difference(target, reference)
    expected = rm @ expected_dv

    np.testing.assert_allclose(
        reconstruct_difference(rm, target, normalize=True, v_ref=reference),
        expected,
    )


def test_reconstruct_difference_accepts_preprojected_dv_and_sparse_rm() -> None:
    rm = sparse.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]], dtype=float))
    dv = np.array([0.25, -0.5, 2.0], dtype=float)

    np.testing.assert_allclose(
        reconstruct_difference(rm, dv, normalize=False),
        np.array([4.25, 2.5], dtype=float),
    )


def test_reconstruct_difference_validates_shapes_and_finite_values() -> None:
    rm = np.ones((2, 3), dtype=float)
    with pytest.raises(ValueError, match="RM column count"):
        reconstruct_difference(rm, np.ones(2), normalize=False)
    with pytest.raises(FloatingPointError, match="dv contains non-finite"):
        reconstruct_difference(rm, np.array([1.0, np.nan, 2.0]), normalize=False)
    with pytest.raises(ValueError, match="rm must be a 2D"):
        reconstruct_difference(np.ones(3), np.ones(3), normalize=False)
