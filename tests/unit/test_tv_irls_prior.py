"""Tests for TV-IRLS inverse priors."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pyeidors.inverse import VoxelGrid, build_one_step_rm
from pyeidors.inverse.prior import (
    solve_tv_irls_frame,
    tv_irls_objective,
    tv_irls_prior_from_state,
)


def test_tv_irls_prior_matches_weighted_ltl_formula_and_changes_signature() -> None:
    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))
    state = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
    beta = 1.0e-4

    prior = tv_irls_prior_from_state(mesh, state, beta=beta, iteration=2)
    difference = sparse.csr_matrix(
        np.array(
            [
                [1.0, -1.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, -1.0],
            ],
            dtype=float,
        )
    )
    weights = 1.0 / np.sqrt((difference @ state) ** 2 + beta)
    expected = difference.T @ sparse.diags(weights, 0, format="csr") @ difference

    np.testing.assert_allclose(prior.as_RtR(dense=True), expected.toarray())
    assert prior.metadata["prior_family"] == "tv_irls"
    assert prior.metadata["irls_iteration"] == 2
    assert prior.metadata["effective_beta"] == pytest.approx(beta)

    changed = tv_irls_prior_from_state(mesh, state + 0.25, beta=beta, iteration=3)
    assert prior.signature_hash != changed.signature_hash
    assert prior.metadata["state_signature"] != changed.metadata["state_signature"]


def test_tv_irls_prior_rejects_bad_beta_and_nonfinite_state() -> None:
    mesh = VoxelGrid.from_bounds([0.0], [3.0], shape=(3,))
    with pytest.raises(ValueError, match="beta"):
        tv_irls_prior_from_state(mesh, np.ones(3), beta=0.0)
    with pytest.raises(ValueError, match="beta_floor"):
        tv_irls_prior_from_state(mesh, np.ones(3), beta=1.0e-6, beta_floor=0.0)
    with pytest.raises(FloatingPointError, match="state"):
        tv_irls_prior_from_state(mesh, np.array([1.0, np.nan, 0.0]))


def test_solve_tv_irls_batch_rejects_nonfinite_frames_and_initial() -> None:
    from pyeidors.inverse.prior import solve_tv_irls_batch

    mesh = VoxelGrid.from_bounds([0.0], [3.0], shape=(3,))
    jacobian = np.eye(3, dtype=float)
    with pytest.raises(FloatingPointError, match="frames"):
        solve_tv_irls_batch(jacobian, np.array([1.0, np.nan, 0.0]), mesh)
    with pytest.raises(FloatingPointError, match="initial"):
        solve_tv_irls_batch(
            jacobian,
            np.ones((2, 3), dtype=float),
            mesh,
            initial=np.array([[0.0, 0.0, 0.0], [0.0, np.inf, 0.0]]),
        )


def test_build_one_step_rm_accepts_tv_irls_prior_contract() -> None:
    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))
    jacobian = np.eye(4, dtype=float)
    state = np.array([0.0, 1.0, 0.5, 0.0], dtype=float)
    prior = tv_irls_prior_from_state(mesh, state, beta=1.0e-3)

    result = build_one_step_rm(
        jacobian,
        regularization=prior,
        lambda_=0.1,
        mode="tv_irls",
        return_metadata=True,
    )

    assert result.metadata["mode"] == "tv_irls"
    assert result.metadata["regularization_source"] == "provided_tv_irls"
    assert result.metadata["RtR_signature_hash"] == prior.signature_hash
    np.testing.assert_allclose(
        result.rm, np.linalg.inv(np.eye(4) + 0.01 * prior.as_RtR(dense=True))
    )


def test_solve_tv_irls_frame_has_monotone_objective_and_stale_rm_tokens() -> None:
    mesh = VoxelGrid.from_bounds([0.0], [6.0], shape=(6,))
    jacobian = np.eye(6, dtype=float)
    truth = np.array([0.0, 0.1, 1.0, 0.9, 0.1, 0.0], dtype=float)
    initial = np.zeros(6, dtype=float)

    result = solve_tv_irls_frame(
        jacobian,
        truth,
        mesh,
        lambda_=0.15,
        initial=initial,
        beta=1.0e-4,
        max_outer_iterations=5,
        tolerance=0.0,
    )

    objectives = result.metadata["objective_history"]
    assert result.metadata["method"] == "tv-irls"
    assert result.metadata["objective_monotone"] is True
    assert all(
        right <= left + 1.0e-10 for left, right in zip(objectives, objectives[1:])
    )
    assert result.metadata["tv_pdhg_postprocess_separate"] is True
    assert (
        len(result.metadata["stale_rm_token_history"]) == result.metadata["iterations"]
    )
    assert len(set(result.metadata["RtR_signature_hash_history"])) > 1
    assert len(set(result.metadata["stale_rm_token_history"])) > 1
    assert tv_irls_objective(
        jacobian,
        truth,
        result.values,
        mesh,
        lambda_=0.15,
        beta=1.0e-4,
    ) == pytest.approx(objectives[-1])
