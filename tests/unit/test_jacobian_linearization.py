"""Tests for matrix-free Jacobian linearization helpers."""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
from scipy import sparse

import pyeidors.inverse.jacobian.linearized as linearized_module
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


def test_v290_jacobian_linearization_direct_fills_adjoint_blocks(monkeypatch) -> None:
    def _fail_stack(*_args, **_kwargs):
        raise AssertionError("JacobianLinearization must direct-fill adjoint blocks")

    monkeypatch.setattr(linearized_module.np, "stack", _fail_stack)

    linearization = _make_linearization()
    assert "np.stack" not in inspect.getsource(
        linearized_module._stack_adjoint_gradient_block
    )
    assert "np.stack" not in inspect.getsource(
        linearized_module.JacobianLinearization.__post_init__
    )
    assert len(linearization._adjoint_blocks) == 2
    np.testing.assert_allclose(
        linearization._adjoint_blocks[0],
        np.asarray(linearization.adjoint_gradients[:2]),
    )
    np.testing.assert_allclose(
        linearization._adjoint_blocks[1],
        np.asarray(linearization.adjoint_gradients[2:3]),
    )


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


def test_hessian_diag_matches_dense_reference() -> None:
    linearization = _make_linearization()
    dense = linearization.to_dense()

    expected = np.sum(dense * dense, axis=0)
    np.testing.assert_allclose(linearization.hessian_diag(), expected)


def test_hessian_diag_applies_measurement_weights_and_regularization() -> None:
    linearization = _make_linearization()
    dense = linearization.to_dense()
    weights = np.array([1.0, 0.5, 2.0], dtype=float)
    reg_diag = np.array([1.25, 0.5, 3.0], dtype=float)

    expected = (weights[:, None] * dense * dense).sum(axis=0) + 0.1 * reg_diag
    actual = linearization.hessian_diag(
        measurement_weights=weights,
        alpha=0.1,
        regularization_diag=reg_diag,
    )
    np.testing.assert_allclose(actual, expected)


def test_hessian_diag_handles_negative_sign_and_floor() -> None:
    linearization = _make_linearization()
    signed = JacobianLinearization(
        grad_u_all=linearization.grad_u_all,
        adjoint_gradients=linearization.adjoint_gradients,
        cell_areas=linearization.cell_areas,
        n_meas_per_stim=linearization.n_meas_per_stim,
        sign=-1.0,
    )
    dense_signed = signed.to_dense()

    expected = np.sum(dense_signed * dense_signed, axis=0)
    np.testing.assert_allclose(signed.hessian_diag(), expected)

    floor = float(expected.min()) * 10.0
    floored = signed.hessian_diag(floor=floor)
    assert np.all(floored >= floor)


def test_hessian_diag_rejects_shape_mismatch() -> None:
    linearization = _make_linearization()
    with np.testing.assert_raises(ValueError):
        linearization.hessian_diag(measurement_weights=np.ones(2))
    with np.testing.assert_raises(ValueError):
        linearization.hessian_diag(alpha=0.1, regularization_diag=np.ones(2))


def test_sigma_fingerprint_defaults_to_empty_and_skips_guard() -> None:
    linearization = _make_linearization()
    assert linearization.sigma_fingerprint == ""
    linearization.assert_compatible(None)
    linearization.assert_compatible("")
    linearization.assert_compatible("not-empty-but-stored-is-empty")


def test_sigma_fingerprint_matching_call_succeeds() -> None:
    linearization = JacobianLinearization(
        grad_u_all=(np.ones((3, 2), dtype=float),),
        adjoint_gradients=(np.ones((3, 2), dtype=float),),
        cell_areas=np.ones(3, dtype=float),
        n_meas_per_stim=(1,),
        sigma_fingerprint="fingerprint-abc",
    )
    linearization.assert_compatible("fingerprint-abc")


def test_sigma_fingerprint_mismatch_raises() -> None:
    linearization = JacobianLinearization(
        grad_u_all=(np.ones((3, 2), dtype=float),),
        adjoint_gradients=(np.ones((3, 2), dtype=float),),
        cell_areas=np.ones(3, dtype=float),
        n_meas_per_stim=(1,),
        sigma_fingerprint="fingerprint-abc",
    )
    with np.testing.assert_raises(ValueError):
        linearization.assert_compatible("fingerprint-xyz")


def test_compute_sigma_fingerprint_is_stable_and_detects_change() -> None:
    from pyeidors.inverse.jacobian.linearized import compute_sigma_fingerprint

    sigma_a = np.array([0.5, 0.6, 0.7], dtype=float)
    sigma_b = np.array([0.5, 0.6, 0.7001], dtype=float)

    fp_a = compute_sigma_fingerprint(sigma_a)
    fp_a_again = compute_sigma_fingerprint(np.array([0.5, 0.6, 0.7], dtype=float))
    fp_b = compute_sigma_fingerprint(sigma_b)
    complex_sigma = np.array([0.5 + 0.1j, 0.6 + 0.2j], dtype=np.complex64)
    complex_fp = compute_sigma_fingerprint(complex_sigma)
    complex_payload = (
        b"complex64\0"
        + np.ascontiguousarray(complex_sigma, dtype=np.complex64).tobytes()
    )

    assert fp_a == fp_a_again
    assert (
        fp_a
        == hashlib.sha256(
            np.ascontiguousarray(sigma_a, dtype=np.float64).tobytes()
        ).hexdigest()
    )
    assert fp_a != fp_b
    assert complex_fp == hashlib.sha256(complex_payload).hexdigest()
    assert len(fp_a) == 64
    assert ".tobytes(" not in inspect.getsource(compute_sigma_fingerprint)


def test_v595_compute_sigma_fingerprint_streams_noncontiguous_views(
    monkeypatch,
) -> None:
    from pyeidors.inverse.jacobian.linearized import compute_sigma_fingerprint

    sigma_view = np.arange(24, dtype=np.float64).reshape(8, 3)[:, 1]
    assert not sigma_view.flags.c_contiguous
    expected = linearized_module.hash_array_payload(
        np.ascontiguousarray(sigma_view, dtype=np.float64)
    )
    original_hash = linearized_module.hash_array_payload
    captured: dict[str, np.ndarray] = {}

    def _capture_hash(arr: np.ndarray, *, prefix: bytes = b"") -> str:
        captured["arr"] = arr
        return original_hash(arr, prefix=prefix)

    monkeypatch.setattr(linearized_module, "hash_array_payload", _capture_hash)

    actual = compute_sigma_fingerprint(sigma_view)

    assert actual == expected
    assert captured["arr"] is sigma_view
    assert captured["arr"].dtype == np.float64
    assert not captured["arr"].flags.c_contiguous
