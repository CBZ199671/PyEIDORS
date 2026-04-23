"""Sparse workflow wrappers and CUQI adapter guards."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.inverse.contracts import SolverOutput
from pyeidors.inverse.solvers import eit_pde as eit_pde_module
from pyeidors.inverse.workflows.sparse_bayesian import (
    perform_sparse_absolute_reconstruction,
    perform_sparse_difference_reconstruction,
)


class _DummySparseReconstructor:
    def __init__(self, conductivity: np.ndarray, simulated: np.ndarray):
        self._conductivity = conductivity
        self._simulated = simulated

    def reconstruct(self, **kwargs):
        return SolverOutput(
            conductivity=self._conductivity.copy(),
            simulated_measurement=self._simulated.copy(),
            likelihood_noise_std=1e-3,
            prior_scale=5e-2,
            metadata={"dummy": True},
            iterations=1,
            converged=True,
            final_residual=0.0,
            final_relative_change=0.0,
        )


def test_sparse_absolute_wrapper_without_cuqi(eit_system):
    baseline = eit_system.create_homogeneous_image(1.0)
    target = eit_system.forward_solve(baseline)

    dummy = _DummySparseReconstructor(
        conductivity=baseline.elem_data,
        simulated=target.meas,
    )
    result = perform_sparse_absolute_reconstruction(
        eit_system=eit_system,
        measurement_data=target,
        baseline_image=baseline,
        reconstructor=dummy,
    )
    assert result.mode == "absolute"
    assert np.isfinite(result.residual).all()
    assert result.metadata["dummy"] is True


def test_sparse_difference_wrapper_without_cuqi(eit_system):
    baseline = eit_system.create_homogeneous_image(1.0)
    base_data = eit_system.forward_solve(baseline)
    target_data = eit_system.forward_solve(
        eit_system.add_phantom(
            base_conductivity=1.0,
            phantom_conductivity=2.0,
            phantom_center=(0.15, 0.0),
            phantom_radius=0.1,
        )
    )
    dummy = _DummySparseReconstructor(
        conductivity=baseline.elem_data,
        simulated=target_data.meas,
    )
    result = perform_sparse_difference_reconstruction(
        eit_system=eit_system,
        measurement_data=target_data,
        reference_data=base_data,
        baseline_image=baseline,
        reconstructor=dummy,
    )
    assert result.mode == "difference"
    assert result.simulated.shape == result.measured.shape


def test_eit_pde_requires_cuqi_when_missing(monkeypatch, eit_system):
    monkeypatch.setattr(eit_pde_module, "cuqi_pde", None)
    with pytest.raises(ImportError):
        eit_pde_module.EITPDE(eit_system)
