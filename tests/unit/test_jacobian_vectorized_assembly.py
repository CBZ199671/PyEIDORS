"""Vectorized Jacobian assembly regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyeidors.inverse.jacobian.direct_jacobian import DirectJacobianCalculator


def _manual_assembly(grad_u_all, adjoint_gradients, n_meas_per_stim, cell_areas):
    n_measurements = len(adjoint_gradients)
    n_elements = len(cell_areas)
    out = np.zeros((n_measurements, n_elements), dtype=float)
    meas_idx = 0
    for stim_idx, grad_u in enumerate(grad_u_all):
        n_meas = n_meas_per_stim[stim_idx]
        for local in range(n_meas):
            g_idx = meas_idx + local
            sens = np.sum(grad_u * adjoint_gradients[g_idx], axis=1) * cell_areas
            out[g_idx, :] = sens
        meas_idx += n_meas
    return out


def test_vectorized_efficient_assembly_matches_manual_loop():
    calc = DirectJacobianCalculator.__new__(DirectJacobianCalculator)
    calc.cell_areas = np.array([1.0, 0.8, 1.2], dtype=float)
    calc.gdim = 2
    calc.block_tune_mode = "off"
    calc.block_size = 2
    calc.block_candidates = (1, 2, 4)
    calc._resolved_block_size = None
    calc._block_tune_source = "unset"
    calc.fwd_model = SimpleNamespace(
        pattern_manager=SimpleNamespace(n_meas_per_stim=[2, 3]),
        cache_manager=SimpleNamespace(enabled=False),
    )

    grad_u_all = [
        np.array([[1.0, 0.2], [0.3, -0.5], [0.6, 0.1]], dtype=float),
        np.array([[0.4, -0.1], [0.7, 0.9], [-0.2, 0.5]], dtype=float),
    ]
    adjoint_gradients = [
        np.array([[0.9, 0.3], [-0.4, 0.2], [0.2, 0.1]], dtype=float),
        np.array([[0.1, -0.6], [0.8, 0.4], [0.5, -0.3]], dtype=float),
        np.array([[0.5, 0.7], [-0.2, -0.1], [0.4, 0.9]], dtype=float),
        np.array([[0.6, -0.2], [0.1, 0.5], [0.7, 0.2]], dtype=float),
        np.array([[-0.3, 0.4], [0.2, 0.6], [0.8, -0.5]], dtype=float),
    ]

    actual = calc._assemble_jacobian_efficient(grad_u_all, adjoint_gradients)
    expected = _manual_assembly(
        grad_u_all,
        adjoint_gradients,
        [2, 3],
        calc.cell_areas,
    )
    assert np.allclose(actual, expected)
    info = calc.block_tuning_info()
    assert info["selected_block_size"] == 2
    assert float(info["assembly_elapsed_only"]) >= 0.0
