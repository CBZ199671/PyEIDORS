"""Coverage tests for vectorized forward solve and measurement projection."""

from __future__ import annotations

import numpy as np
import pytest
from dolfinx import fem
from scipy.sparse.linalg import splu

from pyeidors.forward import EITForwardModel


def test_measurement_projection_matches_manual_loop(eit_system):
    manager = eit_system.fwd_model.pattern_manager
    rng = np.random.default_rng(20260304)
    voltages = rng.normal(size=(manager.n_stim, manager.tn_elec))

    expected = np.zeros(manager.n_meas_total, dtype=float)
    for i, (start_idx, meas_mat) in enumerate(
        zip(manager.meas_start_indices, manager.meas_matrices, strict=True)
    ):
        n_meas = meas_mat.shape[0]
        expected[start_idx : start_idx + n_meas] = meas_mat @ voltages[i]

    actual = manager.apply_meas_pattern(voltages)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    with pytest.raises(ValueError, match="shape mismatch"):
        manager.apply_meas_pattern(voltages[:, :-1])


def test_forward_solve_vectorized_matches_per_pattern_solve(eit_system):
    model = eit_system.fwd_model
    sigma = fem.Function(model.V_sigma)
    sigma.x.array[:] = 1.0

    _, electrode_voltages = model.forward_solve(sigma)
    pattern_matrix = np.asarray(model.pattern_manager.stim_matrix, dtype=float)

    lu = splu(model.create_full_matrix(sigma).tocsc())
    manual_voltages = np.zeros_like(electrode_voltages)
    for i in range(pattern_matrix.shape[0]):
        rhs = np.zeros(model.dofs + model.n_elec + 1, dtype=float)
        rhs[model.dofs : model.dofs + model.n_elec] = pattern_matrix[i]
        sol = lu.solve(rhs)
        manual_voltages[i, :] = sol[model.dofs : model.dofs + model.n_elec]

    np.testing.assert_allclose(
        electrode_voltages, manual_voltages, atol=1e-10, rtol=1e-10
    )


def test_forward_backends_match(eit_mesh):
    from pyeidors.data.structures import PatternConfig

    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    z = np.full(16, 1e-5, dtype=float)
    model_petsc = EITForwardModel(
        n_elec=16,
        pattern_config=pattern,
        z=z,
        mesh=eit_mesh,
        linear_backend="petsc",
    )
    model_scipy = EITForwardModel(
        n_elec=16,
        pattern_config=pattern,
        z=z,
        mesh=eit_mesh,
        linear_backend="scipy",
    )

    sigma = fem.Function(model_petsc.V_sigma)
    sigma.x.array[:] = 1.0
    _, u_petsc = model_petsc.forward_solve(sigma)
    _, u_scipy = model_scipy.forward_solve(sigma)
    np.testing.assert_allclose(u_petsc, u_scipy, atol=1e-9, rtol=1e-9)
