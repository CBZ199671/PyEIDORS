"""Additional branch coverage for reduced Gauss-Newton step helpers."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pyeidors.inverse.reduced import reduced_gn_step as reduced_step_module
from pyeidors.inverse.reduced.reduced_gn_step import (
    build_reduced_operator,
    solve_reduced_step,
)


def test_build_reduced_operator_validates_input_shapes():
    with pytest.raises(ValueError, match="jacobian and basis must be 2D"):
        build_reduced_operator(
            jacobian=np.array([1.0, 2.0], dtype=float),
            basis=np.eye(2),
            regularization_apply=lambda vec: vec,
            lambda_eff=1.0,
        )

    with pytest.raises(ValueError, match="basis dimension mismatch"):
        build_reduced_operator(
            jacobian=np.ones((3, 2), dtype=float),
            basis=np.ones((3, 1), dtype=float),
            regularization_apply=lambda vec: vec,
            lambda_eff=1.0,
        )


def test_v279_build_reduced_operator_direct_fills_regularized_basis(
    monkeypatch: pytest.MonkeyPatch,
):
    jacobian = np.array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 1.0, 3.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    basis = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    reg_diag = np.array([2.0, 3.0, 4.0], dtype=float)

    def _apply_regularization(vec: np.ndarray) -> np.ndarray:
        return reg_diag * vec

    expected_r_u = np.empty((3, 2), dtype=np.float64)
    expected_r_u[:, 0] = reg_diag * basis[:, 0]
    expected_r_u[:, 1] = reg_diag * basis[:, 1]
    actual_r_u = reduced_step_module._apply_regularization_to_basis(
        basis, _apply_regularization
    )
    np.testing.assert_allclose(actual_r_u, expected_r_u)

    def _fail_column_stack(*_args, **_kwargs):
        raise AssertionError("reduced regularization columns must direct-fill")

    monkeypatch.setattr(reduced_step_module.np, "column_stack", _fail_column_stack)

    result = build_reduced_operator(
        jacobian=jacobian,
        basis=basis,
        regularization_apply=_apply_regularization,
        lambda_eff=0.25,
    )
    expected_ju = jacobian @ basis
    expected_h = expected_ju.T @ expected_ju + 0.25 * (basis.T @ expected_r_u)
    np.testing.assert_allclose(result["JU"], expected_ju)
    np.testing.assert_allclose(result["H"], 0.5 * (expected_h + expected_h.T))
    assert "np.column_stack" not in inspect.getsource(
        reduced_step_module._apply_regularization_to_basis
    )
    assert "np.column_stack" not in inspect.getsource(
        reduced_step_module.build_reduced_operator
    )


def test_solve_reduced_step_uses_cg_when_inexact_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        reduced_step_module,
        "cg",
        lambda _h, _rhs, rtol, maxiter: (np.array([1.0, 2.0], dtype=float), 0),
    )

    delta, info = solve_reduced_step(
        reduced_operator={
            "U": np.eye(2, dtype=float),
            "H": np.eye(2, dtype=float),
        },
        rhs=np.array([1.0, 2.0], dtype=float),
        inexact_tol=1e-3,
        maxiter=5,
    )

    np.testing.assert_allclose(delta, np.array([1.0, 2.0], dtype=float))
    assert info["solver"] == "cg"
    assert float(info["linear_residual_ratio"]) == pytest.approx(0.0)


def test_solve_reduced_step_falls_back_to_dense_after_cg_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        reduced_step_module,
        "cg",
        lambda _h, _rhs, rtol, maxiter: (np.zeros(2, dtype=float), 7),
    )
    monkeypatch.setattr(
        reduced_step_module.np.linalg,
        "solve",
        lambda _h, _rhs: np.array([3.0, 4.0], dtype=float),
    )

    delta, info = solve_reduced_step(
        reduced_operator={
            "U": np.eye(2, dtype=float),
            "H": np.eye(2, dtype=float),
        },
        rhs=np.array([1.0, 2.0], dtype=float),
        inexact_tol=1e-3,
        maxiter=3,
    )

    np.testing.assert_allclose(delta, np.array([3.0, 4.0], dtype=float))
    assert info["solver"] == "dense-fallback"
    assert info["cg_info"] == 7
