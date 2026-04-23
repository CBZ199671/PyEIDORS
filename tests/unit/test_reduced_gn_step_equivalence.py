"""Reduced GN step should match dense solve when basis is full-rank identity."""

from __future__ import annotations

import numpy as np

from pyeidors.inverse.reduced.reduced_gn_step import (
    build_reduced_operator,
    solve_reduced_step,
)


def test_reduced_step_matches_dense_identity_basis():
    rng = np.random.default_rng(4)
    j_mat = rng.standard_normal((24, 18))
    lam = 0.2
    rhs = rng.standard_normal(18)
    r_diag = np.linspace(1.0, 2.0, 18)

    basis = np.eye(18)
    reduced = build_reduced_operator(
        jacobian=j_mat,
        basis=basis,
        regularization_apply=lambda v: r_diag * v,
        lambda_eff=lam,
    )
    delta_reduced, info = solve_reduced_step(
        reduced_operator=reduced,
        rhs=rhs,
        inexact_tol=None,
    )

    with np.errstate(all="ignore"):
        dense_h = j_mat.T @ j_mat + lam * np.diag(r_diag)
    delta_dense = np.linalg.solve(dense_h, rhs)
    assert np.allclose(delta_reduced, delta_dense, atol=1e-8, rtol=1e-6)
    assert info["solver"] in {"dense", "dense-fallback", "cg"}
