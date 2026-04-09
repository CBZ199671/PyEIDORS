"""Synthetic EIT data generator."""

from __future__ import annotations

from typing import Any

import numpy as np
from dolfinx import fem

from ..femx import cell_midpoints
from .structures import EITData, EITImage


def _paint_circle(values: np.ndarray, centers: np.ndarray, center: tuple[float, float], radius: float, conductivity: float) -> None:
    if centers.size == 0:
        return
    dist2 = (centers[:, 0] - center[0]) ** 2 + (centers[:, 1] - center[1]) ** 2
    values[dist2 < radius**2] = conductivity


def create_synthetic_data(
    fwd_model,
    inclusion_conductivity: float = 2.5,
    background_conductivity: float = 1.0,
    noise_level: float = 0.02,
    center: tuple[float, float] = (0.2, 0.2),
    radius: float = 0.3,
) -> dict[str, Any]:
    """Create synthetic EIT test data."""

    sigma_true = fem.Function(fwd_model.V_sigma)
    sigma_true.x.array[:] = background_conductivity

    centers = cell_midpoints(fwd_model.mesh)
    _paint_circle(sigma_true.x.array, centers, center, radius, inclusion_conductivity)

    img_true = EITImage(elem_data=sigma_true.x.array.copy(), fwd_model=fwd_model)
    data_clean, _ = fwd_model.fwd_solve(img_true)

    np.random.seed(42)
    noise = noise_level * np.std(data_clean.meas) * np.random.randn(len(data_clean.meas))
    data_noisy = EITData(
        meas=data_clean.meas + noise,
        stim_pattern=data_clean.stim_pattern,
        n_elec=data_clean.n_elec,
        n_stim=data_clean.n_stim,
        n_meas=data_clean.n_meas,
        type="simulated_noisy",
    )

    snr_db = 20 * np.log10(np.std(data_clean.meas) / np.std(noise))

    return {
        "sigma_true": sigma_true,
        "data_clean": data_clean,
        "data_noisy": data_noisy,
        "noise": noise,
        "snr_db": snr_db,
    }


def create_custom_phantom(
    fwd_model,
    background_conductivity: float = 1.0,
    anomalies: list[dict[str, Any]] | None = None,
):
    """Create custom phantom conductivity field."""

    if anomalies is None:
        anomalies = []

    sigma = fem.Function(fwd_model.V_sigma)
    sigma.x.array[:] = background_conductivity

    centers = cell_midpoints(fwd_model.mesh)
    for anomaly in anomalies:
        center = anomaly.get("center", (0.0, 0.0))
        radius = anomaly.get("radius", 0.2)
        conductivity = anomaly.get("conductivity", 2.0)
        _paint_circle(sigma.x.array, centers, center, radius, conductivity)

    return sigma
