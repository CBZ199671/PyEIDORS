"""Tests for matrix-free Jacobian linearization helpers."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from pyeidors.inverse.jacobian.linearized import JacobianLinearization


def _make_linearization() -> JacobianLinearization:
    grad_u_all = (
        np.array([[1.0, 0.5], [0.25, 2.0], [1.5, -0.5]], dtype=float),
        np.array([[0.2, 1.0], [1.0, -1.0], [0.75, 0.25]], dtype=float),
    )
    adjoint_gradients = (
        np.array([[0.5, 1.0], [1.0, 0.25], [-0.5, 0.75]], dtype=float),
        np.array([[1.5, -0.5], [0.1, 0.2], [0.3, 0.4]], dtype=float),
        np.array([[-0.2, 0.6], [0.7, 0.8], [1.1, -0.3]], dtype=float),
    )
    return JacobianLinearization(
        grad_u_all=grad_u_all,
        adjoint_gradients=adjoint_gradients,
        cell_areas=np.array([2.0, 1.5, 0.5], dtype=float),
        n_meas_per_stim=(2, 1),
        sign=1.0,
    )


def test_linearized_actions_match_dense_jacobian() -> None:
    linearization = _make_linearization()
    dense = linearization.to_dense(block_size=2)

    vector = np.array([0.5, -1.0, 2.0], dtype=float)
    residual = np.array([1.0, -0.25, 0.75], dtype=float)

    np.testing.assert_allclose(linearization.matvec(vector), dense @ vector)
    np.testing.assert_allclose(linearization.rmatvec(residual), dense.T @ residual)

    op = linearization.as_linear_operator()
    np.testing.assert_allclose(op.matvec(vector), dense @ vector)
    np.testing.assert_allclose(op.rmatvec(residual), dense.T @ residual)


def test_normal_operator_applies_weighting_and_regularization() -> None:
    linearization = _make_linearization()
    dense = linearization.to_dense()
    weights = np.array([1.0, 0.5, 2.0], dtype=float)
    regularization = np.diag([2.0, 3.0, 4.0])
    vector = np.array([0.25, -0.5, 1.5], dtype=float)

    expected = dense.T @ (weights * (dense @ vector)) + 0.1 * (regularization @ vector)
    actual = linearization.normal_matvec(
        vector,
        measurement_weights=weights,
        alpha=0.1,
        regularization=regularization,
    )
    np.testing.assert_allclose(actual, expected)

    normal_op = linearization.as_normal_operator(
        measurement_weights=weights,
        alpha=0.1,
        regularization=regularization,
    )
    np.testing.assert_allclose(normal_op.matvec(vector), expected)


def test_normal_operator_accepts_sparse_regularization_action() -> None:
    linearization = _make_linearization()
    dense = linearization.to_dense()
    regularization = sparse.diags([2.0, 3.0, 4.0], 0, format="csr")
    vector = np.array([0.25, -0.5, 1.5], dtype=float)

    expected = dense.T @ (dense @ vector) + 0.2 * regularization.dot(vector)
    actual = linearization.normal_matvec(
        vector,
        alpha=0.2,
        regularization=regularization,
    )
    np.testing.assert_allclose(actual, expected)
