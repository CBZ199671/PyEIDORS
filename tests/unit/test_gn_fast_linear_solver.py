"""Fast-mode linear solver unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import diags

import pyeidors.inverse.solvers.gauss_newton_runtime as gn_runtime
from pyeidors.inverse.solvers.gauss_newton_runtime import _solve_linear_system_fast


def _dummy_reconstructor(
    linear_solver: str,
    *,
    preconditioner: str = "diag",
    fast_linear_path: str = "auto",
):
    return SimpleNamespace(
        R_matrix=diags([1.0, 2.0, 3.0], 0, format="csr"),
        R_diag=np.array([1.0, 2.0, 3.0], dtype=float),
        use_prior_term=True,
        performance_mode="aggressive",
        linear_solver=linear_solver,
        preconditioner=preconditioner,
        fast_linear_path=fast_linear_path,
        cholmod_max_n=12000,
        cholmod_max_memory_gib=4.0,
    )


def _expected_solution(J: np.ndarray, residual: np.ndarray, de: np.ndarray, lam: float) -> np.ndarray:
    R = np.diag([1.0, 2.0, 3.0])
    A = J.T @ J + lam * R
    rhs = -(J.T @ residual + lam * (R @ de))
    return np.linalg.solve(A, rhs)


def test_fast_solver_auto_woodbury_matches_dense_reference():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15

    delta, _, _ = _solve_linear_system_fast(
        _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="auto"),
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )
    expected = _expected_solution(J, residual, de, lam)
    assert np.allclose(delta, expected, rtol=1e-6, atol=1e-8)


def test_fast_solver_explicit_pcg_matches_dense_reference():
    J = np.array(
        [
            [0.9, 0.1, -0.3],
            [-0.5, 0.7, 0.4],
            [0.3, -0.2, 0.6],
            [0.2, 0.8, -0.1],
            [0.1, -0.4, 0.7],
        ],
        dtype=float,
    )
    residual = np.array([0.06, -0.03, 0.01, 0.05, -0.02], dtype=float)
    de = np.array([0.05, -0.08, 0.12], dtype=float)
    lam = 0.05

    delta, _, _ = _solve_linear_system_fast(
        _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="pcg"),
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=1,
    )
    expected = _expected_solution(J, residual, de, lam)
    assert np.allclose(delta, expected, rtol=1e-5, atol=1e-7)


def test_fast_solver_cholmod_preconditioner_and_fallback(monkeypatch: pytest.MonkeyPatch):
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15

    class _FakeFactor:
        def __init__(self, mat):
            dense = mat.toarray()
            dense = 0.5 * (dense + dense.T)
            self._dense = dense

        def solve_A(self, rhs):
            return np.linalg.solve(self._dense, rhs)

    monkeypatch.setattr(
        gn_runtime,
        "detect_performance_capabilities",
        lambda: {
            "pyamg": False,
            "cholmod": True,
            "petsc_mat_solve": False,
            "petsc_gamg": False,
        },
    )

    recon = _dummy_reconstructor("auto", preconditioner="cholmod", fast_linear_path="pcg")
    monkeypatch.setattr(gn_runtime, "cholmod_cholesky", lambda mat: _FakeFactor(mat))
    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )
    expected = _expected_solution(J, residual, de, lam)
    assert np.allclose(delta, expected, rtol=1e-5, atol=1e-7)
    meta = getattr(recon, "_last_fast_linear_meta", {})
    assert "cholmod-precond" in str(meta.get("path", ""))

    monkeypatch.setattr(
        gn_runtime,
        "cholmod_cholesky",
        lambda _mat: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    recon2 = _dummy_reconstructor("auto", preconditioner="cholmod", fast_linear_path="pcg")
    delta2, _, _ = _solve_linear_system_fast(
        recon2,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=1,
    )
    assert np.allclose(delta2, expected, rtol=5e-4, atol=1e-5)
    meta2 = getattr(recon2, "_last_fast_linear_meta", {})
    assert "cholmod" in str(meta2.get("fallback_reason", ""))
