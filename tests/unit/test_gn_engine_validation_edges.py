"""Additional constructor validation tests for GN engine edge parameters."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

import pyeidors.inverse.solvers.gauss_newton_engine as gn_engine


@pytest.mark.parametrize(
    ("kwargs", "expected_substring"),
    [
        ({"inexact_eta_min": 0.0}, "inexact eta bounds"),
        ({"inexact_eta_min": 0.9, "inexact_eta_max": 0.1}, "inexact_eta_min"),
        ({"lowrank_energy": 1.5}, "lowrank_energy"),
    ],
)
def test_invalid_edge_parameters_raise_inprocess(kwargs, expected_substring):
    fm = mock.MagicMock()
    fm.n_elec = 4
    fm.mesh.topology.dim = 2

    with pytest.raises(Exception) as exc_info:
        gn_engine.GaussNewtonReconstructor(fwd_model=fm, **kwargs)

    assert expected_substring in str(exc_info.value)


def test_cholmod_guard_bounds_are_clamped_inprocess(
    eit_system, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        gn_engine,
        "resolve_torch_device",
        lambda *args, **kwargs: SimpleNamespace(
            requested="cpu",
            effective="cpu",
            fallback_reason=None,
            torch_device="cpu",
        ),
    )

    recon = gn_engine.GaussNewtonReconstructor(
        fwd_model=eit_system.fwd_model,
        cholmod_max_n=0,
        cholmod_max_memory_gib=0.0,
        verbose=False,
    )

    assert recon.cholmod_max_n == 1
    assert recon.cholmod_max_memory_gib == 0.25


def test_invalid_inexact_and_lowrank_validation_inprocess(
    eit_system, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        gn_engine,
        "resolve_torch_device",
        lambda *args, **kwargs: SimpleNamespace(
            requested="cpu",
            effective="cpu",
            fallback_reason=None,
            torch_device="cpu",
        ),
    )

    with pytest.raises(ValueError, match="inexact eta bounds"):
        gn_engine.GaussNewtonReconstructor(
            fwd_model=eit_system.fwd_model,
            inexact_eta_min=0.0,
            verbose=False,
        )

    with pytest.raises(ValueError, match="inexact_eta_min"):
        gn_engine.GaussNewtonReconstructor(
            fwd_model=eit_system.fwd_model,
            inexact_eta_min=0.9,
            inexact_eta_max=0.1,
            verbose=False,
        )

    with pytest.raises(ValueError, match="lowrank_energy"):
        gn_engine.GaussNewtonReconstructor(
            fwd_model=eit_system.fwd_model,
            lowrank_energy=1.5,
            verbose=False,
        )
