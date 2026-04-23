"""Additional branch coverage for reduced Gauss-Newton step helpers."""

from __future__ import annotations

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
