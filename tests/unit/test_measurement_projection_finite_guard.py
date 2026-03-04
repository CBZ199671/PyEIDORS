"""Tests for finite-value guards in measurement projection."""

from __future__ import annotations

import numpy as np
import pytest


def test_apply_meas_pattern_rejects_non_finite_input(eit_system):
    manager = eit_system.fwd_model.pattern_manager
    voltages = np.zeros((manager.n_stim, manager.tn_elec), dtype=float)
    voltages[0, 0] = np.nan

    with pytest.raises(FloatingPointError, match="non-finite"):
        manager.apply_meas_pattern(voltages)


def test_apply_meas_pattern_rejects_non_finite_projection(eit_system, monkeypatch):
    manager = eit_system.fwd_model.pattern_manager
    projection = np.asarray(manager._meas_projection, dtype=float).copy()
    projection[0, 0] = np.inf
    monkeypatch.setattr(manager, "_meas_projection", projection)

    voltages = np.ones((manager.n_stim, manager.tn_elec), dtype=float)
    with pytest.raises(FloatingPointError, match="non-finite"):
        manager.apply_meas_pattern(voltages)


def test_apply_meas_pattern_matches_projection_for_finite_values(eit_system):
    manager = eit_system.fwd_model.pattern_manager
    voltages = np.linspace(
        0.0,
        1e-4,
        manager.n_stim * manager.tn_elec,
        dtype=float,
    ).reshape(manager.n_stim, manager.tn_elec)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        expected = manager._meas_projection @ voltages.reshape(-1)
    actual = manager.apply_meas_pattern(voltages)

    assert np.allclose(actual, expected)
