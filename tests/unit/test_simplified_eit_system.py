"""Workflow-level reconstruction wrappers."""

from __future__ import annotations

import numpy as np
from dolfinx import fem

from pyeidors.inverse import (
    perform_absolute_reconstruction,
    perform_difference_reconstruction,
)
from pyeidors.inverse.contracts import SolverOutput


def _patch_solver(monkeypatch, eit_system):
    def _fake_reconstruct(*args, **kwargs):
        conductivity = fem.Function(eit_system.fwd_model.V_sigma)
        conductivity.x.array[:] = 1.0
        return SolverOutput(
            conductivity=conductivity,
            residual_history=[0.5],
            sigma_change_history=[0.1],
            iterations=1,
            converged=True,
            final_residual=0.5,
            final_relative_change=0.1,
        )

    monkeypatch.setattr(eit_system.reconstructor, "reconstruct", _fake_reconstruct)


def test_absolute_workflow_wrapper(eit_system, monkeypatch):
    _patch_solver(monkeypatch, eit_system)
    baseline = eit_system.create_homogeneous_image(1.0)
    target_img = eit_system.add_phantom(
        base_conductivity=1.0,
        phantom_conductivity=1.8,
        phantom_center=(-0.2, 0.15),
        phantom_radius=0.12,
    )
    target_data = eit_system.forward_solve(target_img)

    result = perform_absolute_reconstruction(
        eit_system=eit_system,
        measurement_data=target_data,
        baseline_image=baseline,
    )

    assert result.mode == "absolute"
    assert result.conductivity.shape[0] == eit_system.get_system_info()["n_elements"]
    assert np.isfinite(result.residual).all()
    assert result.to_dict()["mode"] == "absolute"


def test_difference_workflow_wrapper(eit_system, monkeypatch):
    _patch_solver(monkeypatch, eit_system)
    baseline_img = eit_system.create_homogeneous_image(1.0)
    baseline_data = eit_system.forward_solve(baseline_img)

    target_img = eit_system.add_phantom(
        base_conductivity=1.0,
        phantom_conductivity=2.2,
        phantom_center=(0.1, -0.25),
        phantom_radius=0.1,
    )
    target_data = eit_system.forward_solve(target_img)

    result = perform_difference_reconstruction(
        eit_system=eit_system,
        measurement_data=target_data,
        reference_data=baseline_data,
    )

    assert result.mode == "difference"
    assert result.measured.shape == result.simulated.shape
    assert np.isfinite(result.conductivity).all()
