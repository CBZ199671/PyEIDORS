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


def test_build_one_step_rm_rejects_measurement_form_until_t17() -> None:
    with pytest.raises(NotImplementedError, match="reserved for T17"):
        build_one_step_rm(np.eye(2), lambda_=0.1, form="measurement")


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
