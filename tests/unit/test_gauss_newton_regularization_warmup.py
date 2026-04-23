"""Tests for Gauss-Newton regularization warmup and call ordering."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.inverse.contracts import SolverOutput


def _build_background_and_target(eit_system):
    baseline = eit_system.create_homogeneous_image(1.0)
    background = eit_system.forward_solve(baseline)
    phantom = eit_system.add_phantom(
        base_conductivity=1.0,
        phantom_conductivity=2.0,
        phantom_center=(0.2, 0.2),
        phantom_radius=0.15,
    )
    target = eit_system.forward_solve(phantom)
    return background, target


def test_ensure_regularization_ready_is_idempotent(eit_system, monkeypatch):
    rec = eit_system.reconstructor
    n_elem = rec.n_elements
    calls = SimpleNamespace(count=0)

    def _fake_regularization_matrix():
        calls.count += 1
        return np.eye(n_elem, dtype=float)

    monkeypatch.setattr(
        rec.regularization, "get_regularization_matrix", _fake_regularization_matrix
    )
    rec.R_torch = None

    rec.ensure_regularization_ready()
    first_tensor = rec.R_torch
    rec.ensure_regularization_ready()

    assert calls.count == 1
    assert rec.R_torch is first_tensor
    assert tuple(rec.R_torch.shape) == (n_elem, n_elem)


def test_ensure_regularization_ready_rejects_non_finite(eit_system, monkeypatch):
    rec = eit_system.reconstructor
    n_elem = rec.n_elements

    def _non_finite_matrix():
        matrix = np.eye(n_elem, dtype=float)
        matrix[0, 0] = np.nan
        return matrix

    monkeypatch.setattr(
        rec.regularization, "get_regularization_matrix", _non_finite_matrix
    )
    rec.R_torch = None

    with pytest.raises(FloatingPointError, match="non-finite"):
        rec.ensure_regularization_ready()


def test_inverse_solve_warms_regularization_before_reconstruct(eit_system, monkeypatch):
    rec = eit_system.reconstructor
    background, target = _build_background_and_target(eit_system)
    calls: list[str] = []

    def _warmup():
        calls.append("warmup")

    def _reconstruct(measured_data, initial_guess=None):
        _ = measured_data
        _ = initial_guess
        calls.append("reconstruct")
        return SolverOutput(
            conductivity=np.ones(rec.n_elements, dtype=float),
            iterations=0,
            converged=False,
            final_residual=0.0,
            final_relative_change=0.0,
        )

    monkeypatch.setattr(rec, "ensure_regularization_ready", _warmup)
    monkeypatch.setattr(rec, "reconstruct", _reconstruct)

    _ = eit_system.inverse_solve(
        data=target, reference_data=background, initial_guess=None
    )

    assert calls == ["warmup", "reconstruct"]
