"""Regression tests for shared Jacobian gradient extraction."""

from __future__ import annotations

import numpy as np

from pyeidors.inverse.jacobian._core import (
    build_jacobian_geometry,
    compute_field_gradients,
)


def test_p1_simplex_gradient_cache_returns_exact_affine_cell_gradients(eit_system):
    geometry = build_jacobian_geometry(eit_system.fwd_model)

    assert geometry.linear_cell_dofs is not None
    assert geometry.linear_gradient_weights is not None

    coords = np.asarray(eit_system.fwd_model.V.tabulate_dof_coordinates(), dtype=float)[
        :, : geometry.gdim
    ]
    field = 1.25 + 2.0 * coords[:, 0] - 0.75 * coords[:, 1]

    gradients = compute_field_gradients([field], geometry)[0]

    expected = np.tile(
        np.array([2.0, -0.75], dtype=float),
        (geometry.cell_areas.size, 1),
    )
    np.testing.assert_allclose(gradients, expected, atol=1.0e-10)
