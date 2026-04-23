"""Synthetic data and visualization coverage tests."""

from __future__ import annotations

import matplotlib
import numpy as np

from pyeidors.data.synthetic_data import create_custom_phantom, create_synthetic_data
from pyeidors.visualization import create_visualizer


matplotlib.use("Agg")


def test_synthetic_data_generation(eit_system):
    synthetic = create_synthetic_data(
        eit_system.fwd_model,
        inclusion_conductivity=2.0,
        background_conductivity=1.0,
        noise_level=0.01,
        center=(0.1, 0.1),
        radius=0.15,
    )
    assert (
        synthetic["sigma_true"].x.array.size
        == eit_system.get_system_info()["n_elements"]
    )
    assert synthetic["data_clean"].meas.shape == synthetic["data_noisy"].meas.shape
    assert np.isfinite(synthetic["snr_db"])


def test_custom_phantom_and_visualization(eit_system):
    sigma = create_custom_phantom(
        eit_system.fwd_model,
        anomalies=[{"center": (0.2, -0.1), "radius": 0.12, "conductivity": 1.9}],
    )
    viz = create_visualizer(style="default")
    fig_mesh = viz.plot_mesh(eit_system.mesh, show_electrodes=False)
    fig_cond = viz.plot_conductivity(eit_system.mesh, sigma)
    fig_cmp = viz.plot_reconstruction_comparison(
        eit_system.mesh, sigma.x.array, sigma.x.array.copy()
    )

    assert fig_mesh is not None
    assert fig_cond is not None
    assert fig_cmp is not None
