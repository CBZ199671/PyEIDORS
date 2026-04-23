"""Reduced Gauss-Newton linear-step helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse.linalg import cg

from ...utils.numeric_ops import safe_dot


def build_reduced_operator(
    *,
    jacobian: np.ndarray,
    basis: np.ndarray,
    regularization_apply: Callable[[np.ndarray], np.ndarray],
    lambda_eff: float,
) -> dict[str, np.ndarray]:
    """Build reduced Hessian operator data for a given basis."""
    j_mat = np.asarray(jacobian, dtype=np.float64)
    u_mat = np.asarray(basis, dtype=np.float64)
    if j_mat.ndim != 2 or u_mat.ndim != 2:
        raise ValueError("jacobian and basis must be 2D")
    if u_mat.shape[0] != j_mat.shape[1]:
        raise ValueError("basis dimension mismatch")

    ju = np.asarray(
        safe_dot(j_mat, u_mat, "gauss_newton.reduced.ju"),
        dtype=np.float64,
    )
    h_meas = np.asarray(
        safe_dot(ju.T, ju, "gauss_newton.reduced.h_meas"),
        dtype=np.float64,
    )

    r_u = np.column_stack(
        [
            np.asarray(regularization_apply(u_mat[:, col]), dtype=np.float64)
            for col in range(u_mat.shape[1])
        ]
    )
    h_reg = np.asarray(
        safe_dot(u_mat.T, r_u, "gauss_newton.reduced.h_reg"),
        dtype=np.float64,
    )
    with np.errstate(all="ignore"):
        h_reduced = np.asarray(h_meas + float(lambda_eff) * h_reg, dtype=np.float64)
    h_reduced = 0.5 * (h_reduced + h_reduced.T)

    return {
        "U": u_mat,
        "JU": ju,
        "H": h_reduced,
    }


def solve_reduced_step(
    *,
    reduced_operator: dict[str, np.ndarray],
    rhs: np.ndarray,
    inexact_tol: float | None = None,
    maxiter: int | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Solve reduced linear system and recover full-space step."""
    u_mat = np.asarray(reduced_operator["U"], dtype=np.float64)
    h_reduced = np.asarray(reduced_operator["H"], dtype=np.float64)
    rhs_vec = np.asarray(rhs, dtype=np.float64).reshape(-1)

    reduced_rhs = np.asarray(
        safe_dot(u_mat.T, rhs_vec, "gauss_newton.reduced.rhs"),
        dtype=np.float64,
    )

    info: dict[str, float | int | str] = {
        "solver": "dense",
        "linear_residual_ratio": 0.0,
    }
    if inexact_tol is not None and float(inexact_tol) > 0:
        tol = float(inexact_tol)
        cg_maxiter = (
            int(maxiter) if maxiter is not None else max(40, h_reduced.shape[0] * 3)
        )
        alpha, cg_info = cg(h_reduced, reduced_rhs, rtol=tol, maxiter=cg_maxiter)
        if cg_info == 0:
            info["solver"] = "cg"
            residual = h_reduced @ alpha - reduced_rhs
            denom = max(np.linalg.norm(reduced_rhs), 1e-12)
            info["linear_residual_ratio"] = float(np.linalg.norm(residual) / denom)
        else:
            alpha = np.linalg.solve(h_reduced, reduced_rhs)
            info["solver"] = "dense-fallback"
            info["cg_info"] = int(cg_info)
    else:
        alpha = np.linalg.solve(h_reduced, reduced_rhs)

    delta = np.asarray(
        safe_dot(u_mat, alpha, "gauss_newton.reduced.delta"),
        dtype=np.float64,
    )
    return delta, info
