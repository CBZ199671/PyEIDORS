"""End-to-end system tests on a real gmsh-generated mesh."""

from __future__ import annotations

import numpy as np
from dolfinx import fem

from pyeidors.data.difference import build_difference_vector
from pyeidors.data.structures import EITImage
from pyeidors.inverse.contracts import SolverOutput


def test_system_setup_and_forward(eit_system):
    info = eit_system.get_system_info()
    assert info["initialized"] is True
    assert info["n_elec"] == 16
    assert info["n_elements"] > 0
    assert info["n_measurements"] > 0
    assert "cache_stats" in info

    image = eit_system.create_homogeneous_image(conductivity=1.0)
    data = eit_system.forward_solve(image)

    assert data.n_elec == 16
    assert data.n_meas == info["n_measurements"]
    assert np.isfinite(data.meas).all()
    stats = eit_system.get_cache_stats()
    assert "total_hits" in stats
    eit_system.clear_cache(scope="process")


def test_difference_inverse_smoke(eit_system, monkeypatch):
    baseline = eit_system.create_homogeneous_image(1.0)
    background = eit_system.forward_solve(baseline)

    phantom = eit_system.add_phantom(
        base_conductivity=1.0,
        phantom_conductivity=2.0,
        phantom_center=(0.2, 0.2),
        phantom_radius=0.15,
    )
    target = eit_system.forward_solve(phantom)

    def _fake_reconstruct(*args, **kwargs):
        conductivity = fem.Function(eit_system.fwd_model.V_sigma)
        conductivity.x.array[:] = 1.0
        return SolverOutput(
            conductivity=conductivity,
            residual_history=[1.0, 0.5],
            sigma_change_history=[0.1, 0.05],
            iterations=2,
            converged=True,
            final_residual=0.5,
            final_relative_change=0.05,
        )

    monkeypatch.setattr(eit_system.reconstructor, "reconstruct", _fake_reconstruct)
    result = eit_system.inverse_solve(data=target, reference_data=background)
    conductivity_fn = result.conductivity
    conductivity = conductivity_fn.x.array
    assert conductivity.size == eit_system.get_system_info()["n_elements"]
    assert np.isfinite(conductivity).all()

    recon_image = EITImage(elem_data=conductivity.copy(), fwd_model=eit_system.fwd_model)
    recon_data = eit_system.forward_solve(recon_image)
    assert recon_data.meas.shape == target.meas.shape


def test_inverse_solve_builds_configured_difference_measurement(eit_system, monkeypatch):
    baseline = eit_system.create_homogeneous_image(1.0)
    background = eit_system.forward_solve(baseline)

    phantom = eit_system.add_phantom(
        base_conductivity=1.0,
        phantom_conductivity=2.0,
        phantom_center=(0.2, 0.2),
        phantom_radius=0.15,
    )
    target = eit_system.forward_solve(phantom)
    eit_system.difference_mode = "normalized"
    eit_system.difference_orientation = "reference_minus_target"

    captured = {}

    def _fake_reconstruct(measured_data, initial_conductivity=None, **kwargs):
        _ = initial_conductivity
        _ = kwargs
        captured["measured_data"] = measured_data
        conductivity = fem.Function(eit_system.fwd_model.V_sigma)
        conductivity.x.array[:] = 1.0
        return SolverOutput(conductivity=conductivity)

    monkeypatch.setattr(eit_system.reconstructor, "reconstruct", _fake_reconstruct)
    _ = eit_system.inverse_solve(data=target, reference_data=background)

    measured_data = captured["measured_data"]
    expected = build_difference_vector(
        target.meas,
        background.meas,
        mode="normalized",
        orientation="reference_minus_target",
    )
    assert measured_data.type == "difference"
    assert measured_data.reference_meas is not None
    assert measured_data.target_meas is not None
    assert measured_data.difference_mode == "normalized"
    assert measured_data.difference_orientation == "reference_minus_target"
    assert np.allclose(measured_data.meas, expected)
