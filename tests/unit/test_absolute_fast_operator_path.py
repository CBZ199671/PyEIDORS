"""Coverage for absolute fast operator solve path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from pyeidors.inverse.solvers.gauss_newton_runtime import _solve_linear_system_fast


def _dummy_reconstructor(*, linear_solver: str = "auto", preconditioner: str = "auto"):
    return SimpleNamespace(
        use_prior_term=True,
        R_matrix=sparse.diags(np.array([1.5, 2.0, 2.5], dtype=float), offsets=0, format="csr"),
        R_diag=np.array([1.5, 2.0, 2.5], dtype=float),
        performance_mode="safe",
        linear_solver=linear_solver,
        preconditioner=preconditioner,
    )


def test_absolute_fast_operator_matches_dense_reference():
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
    rec = _dummy_reconstructor(linear_solver="auto", preconditioner="auto")

    delta, _, _ = _solve_linear_system_fast(
        rec,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=1,
    )

    rhs = -(J.T @ residual + lam * (rec.R_matrix @ de))
    dense_h = J.T @ J + lam * rec.R_matrix.toarray()
    expected = np.linalg.solve(dense_h, rhs)
    np.testing.assert_allclose(delta, expected, atol=1e-8, rtol=1e-8)
