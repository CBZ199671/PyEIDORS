"""Tests for forward solve output view semantics."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from scipy.sparse.linalg import splu


def test_forward_solve_returns_tuple_of_column_views(eit_system):
    model = eit_system.fwd_model
    sigma = fem.Function(model.V_sigma)
    sigma.x.array[:] = 1.0

    u_all, electrode_voltages = model.forward_solve(sigma)
    assert isinstance(u_all, tuple)
    assert len(u_all) == electrode_voltages.shape[0]
    assert all(col.base is not None for col in u_all)

    solve_dtype = np.asarray(electrode_voltages).dtype
    pattern_matrix = np.asarray(model.pattern_manager.stim_matrix, dtype=solve_dtype)
    rhs_matrix = np.zeros(
        (model.dofs + model.n_elec + 1, pattern_matrix.shape[0]), dtype=solve_dtype
    )
    rhs_matrix[model.dofs : model.dofs + model.n_elec, :] = pattern_matrix.T
    sol_matrix = splu(model.create_full_matrix(sigma).tocsc()).solve(rhs_matrix)
    expected_potential = np.asarray(sol_matrix[: model.dofs, :], dtype=solve_dtype)

    for idx, column in enumerate(u_all):
        np.testing.assert_allclose(
            column, expected_potential[:, idx], atol=5e-7, rtol=1e-5
        )
