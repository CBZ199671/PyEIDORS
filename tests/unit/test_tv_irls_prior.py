"""Tests for TV-IRLS inverse priors."""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest
from scipy import sparse

import pyeidors.inverse.prior.tv_irls as tv_irls_module
from pyeidors.inverse import VoxelGrid, build_one_step_rm
from pyeidors.inverse.prior import (
    solve_tv_irls_frame,
    tv_irls_objective,
    tv_irls_prior_from_state,
)


def test_tv_irls_state_digest_streams_payload_without_tobytes_copy() -> None:
    state = np.array([0.0, 1.5, -2.0, 3.25], dtype=float)
    contiguous = np.ascontiguousarray(state, dtype=np.float64)
    expected = hashlib.sha256(
        str(contiguous.dtype).encode()
        + b"|"
        + json.dumps([int(v) for v in contiguous.shape]).encode()
        + b"|"
        + contiguous.tobytes()
    ).hexdigest()

    assert tv_irls_module._digest_array(state) == expected
    source = inspect.getsource(tv_irls_module._digest_array)
    assert "update_digest_with_array_payload" in source
    assert ".tobytes(" not in source
    assert "np.ascontiguousarray(np.asarray" not in source


def test_tv_irls_initial_batch_broadcasts_vector_without_broadcast_to_copy() -> None:
    initial = np.array([0.0, 1.0, 2.0], dtype=float)
    batch = tv_irls_module._initial_batch(initial, n_frames=2)

    assert batch is not None
    np.testing.assert_allclose(batch, np.vstack([initial, initial]))
    assert batch.flags.c_contiguous
    assert not np.shares_memory(batch, initial)
    source = inspect.getsource(tv_irls_module._initial_batch)
    assert "broadcast_to" not in source
    assert "np.copyto" in source


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


def test_v294_tv_irls_batch_direct_fills_frame_rows(monkeypatch) -> None:
    def fail_vstack(*_args, **_kwargs):
        raise AssertionError("TV-IRLS batch values must not call np.vstack")

    monkeypatch.setattr(tv_irls_module.np, "vstack", fail_vstack)
    mesh = VoxelGrid.from_bounds([0.0], [3.0], shape=(3,))
    jacobian = np.eye(3, dtype=float)
    frames = np.array(
        [[0.0, 1.0, 0.0], [0.2, 0.8, 0.1]],
        dtype=float,
    )

    result = tv_irls_module.solve_tv_irls_batch(
        jacobian,
        frames,
        mesh,
        lambda_=0.05,
        max_outer_iterations=2,
        tolerance=0.0,
    )

    assert result.values.shape == frames.shape
    assert result.metadata["n_frames"] == frames.shape[0]
    assert "np.vstack" not in inspect.getsource(tv_irls_module.solve_tv_irls_batch)


def test_v488_tv_irls_guards_use_bounded_finite_scans() -> None:
    prior_source = inspect.getsource(tv_irls_module.tv_irls_prior_from_state)
    diff_source = inspect.getsource(tv_irls_module._difference_operator)
    state_source = inspect.getsource(tv_irls_module._state_vector)
    measurement_source = inspect.getsource(tv_irls_module._measurement_vector)
    frame_source = inspect.getsource(tv_irls_module._frame_batch)
    initial_source = inspect.getsource(tv_irls_module._initial_batch)

    assert "all_finite_values(gradient)" in prior_source
    assert "all_finite_values(weights)" in prior_source
    assert "np.isfinite(gradient).all()" not in prior_source
    assert "np.isfinite(weights).all()" not in prior_source
    assert "all_finite_values(difference.data)" in diff_source
    assert "np.isfinite(difference.data).all()" not in diff_source
    assert "all_finite_values(vector)" in state_source
    assert "np.isfinite(vector).all()" not in state_source
    assert "all_finite_values(vector)" in measurement_source
    assert "np.isfinite(vector).all()" not in measurement_source
    assert "all_finite_values(arr)" in frame_source
    assert "np.isfinite(arr).all()" not in frame_source
    assert "all_finite_values(arr)" in initial_source
    assert "np.isfinite(arr).all()" not in initial_source


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


def test_tv_irls_objective_applies_beta_floor() -> None:
    """``tv_irls_objective`` must clamp ``beta`` to ``beta_floor`` (V74)."""

    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))
    jacobian = np.eye(4, dtype=float)
    measurement = np.array([0.1, 0.4, 0.4, 0.1], dtype=float)
    state = np.array([0.0, 1.0, 1.0, 0.0], dtype=float)

    floored_default = tv_irls_objective(
        jacobian, measurement, state, mesh, lambda_=0.1, beta=1.0e-12
    )
    clamped = tv_irls_objective(
        jacobian, measurement, state, mesh, lambda_=0.1, beta=1.0e-20
    )
    assert clamped == floored_default

    explicit_floor = tv_irls_objective(
        jacobian,
        measurement,
        state,
        mesh,
        lambda_=0.1,
        beta=1.0e-4,
        beta_floor=1.0e-12,
    )
    same_call = tv_irls_objective(
        jacobian, measurement, state, mesh, lambda_=0.1, beta=1.0e-4
    )
    assert explicit_floor == same_call

    raised_floor = tv_irls_objective(
        jacobian,
        measurement,
        state,
        mesh,
        lambda_=0.1,
        beta=1.0e-4,
        beta_floor=1.0e-2,
    )
    raised_via_beta = tv_irls_objective(
        jacobian, measurement, state, mesh, lambda_=0.1, beta=1.0e-2
    )
    assert raised_floor == raised_via_beta
    assert raised_floor != pytest.approx(same_call, rel=1.0e-3)
