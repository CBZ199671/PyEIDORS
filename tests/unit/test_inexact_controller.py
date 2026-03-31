"""Tests for inexact GN forcing controller."""

from __future__ import annotations

from pyeidors.inverse.reduced.inexact_controller import InexactController


def test_fixed_mode_keeps_eta_and_tightens_on_reject():
    ctl = InexactController(mode="fixed", eta0=0.2, eta_min=1e-3, eta_max=0.5)
    eta1 = ctl.update(
        outer_prev=1.0,
        outer_curr=0.8,
        linear_residual_ratio=0.1,
        step_rejected=False,
        stalled=False,
    )
    assert eta1 == 0.2
    eta2 = ctl.update(
        outer_prev=0.8,
        outer_curr=0.9,
        linear_residual_ratio=0.4,
        step_rejected=True,
        stalled=False,
    )
    assert eta2 < eta1


def test_eisenstat_walker_updates_with_bounds():
    ctl = InexactController(mode="eisenstat-walker", eta0=0.2, eta_min=1e-3, eta_max=0.5)
    eta = ctl.update(
        outer_prev=1.0,
        outer_curr=0.4,
        linear_residual_ratio=0.05,
        step_rejected=False,
        stalled=False,
    )
    assert 1e-3 <= eta <= 0.5
    eta_reject = ctl.update(
        outer_prev=0.4,
        outer_curr=0.45,
        linear_residual_ratio=0.2,
        step_rejected=True,
        stalled=True,
    )
    assert 1e-3 <= eta_reject <= 0.5
    assert len(ctl.history) >= 3
