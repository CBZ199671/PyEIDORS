"""Synthetic EIT data generator."""

import numpy as np
from typing import Tuple
from fenics import Function, cells

from .structures import EITData, EITImage


def create_synthetic_data(fwd_model,
                         inclusion_conductivity: float = 2.5,
                         background_conductivity: float = 1.0,
                         noise_level: float = 0.02,
                         center: Tuple[float, float] = (0.2, 0.2),
                         radius: float = 0.3):
    """Create synthetic EIT test data.

    Args:
        fwd_model: Forward model.
        inclusion_conductivity: Anomaly conductivity.
        background_conductivity: Background conductivity.
        noise_level: Noise level.
        center: Anomaly center position.
        radius: Anomaly radius.

    Returns:
        Dictionary containing ground truth distribution, clean data, noisy data, etc.
    """

    # Create ground truth conductivity distribution
    sigma_true = Function(fwd_model.V_sigma)
    sigma_true.vector()[:] = background_conductivity

    # Add circular anomaly
    for cell in cells(fwd_model.mesh):
        cell_center = cell.midpoint()
        x, y = cell_center.x(), cell_center.y()
        if (x - center[0])**2 + (y - center[1])**2 < radius**2:
            sigma_true.vector()[cell.index()] = inclusion_conductivity

    # Generate clean measurement data
    img_true = EITImage(elem_data=sigma_true.vector()[:], fwd_model=fwd_model)
    data_clean, _ = fwd_model.fwd_solve(img_true)

    # Add Gaussian white noise
    np.random.seed(42)  # Ensure reproducibility
    noise = noise_level * np.std(data_clean.meas) * np.random.randn(len(data_clean.meas))
    data_noisy = EITData(
        meas=data_clean.meas + noise,
        stim_pattern=data_clean.stim_pattern,
        n_elec=data_clean.n_elec,
        n_stim=data_clean.n_stim,
        n_meas=data_clean.n_meas,
        type='simulated_noisy'
    )

    snr_db = 20 * np.log10(np.std(data_clean.meas) / np.std(noise))

    return {
        'sigma_true': sigma_true,
        'data_clean': data_clean,
        'data_noisy': data_noisy,
        'noise': noise,
        'snr_db': snr_db
    }


def create_custom_phantom(fwd_model,
                         background_conductivity: float = 1.0,
                         anomalies: list = None):
    """Create custom phantom.

    Args:
        fwd_model: Forward model.
        background_conductivity: Background conductivity.
        anomalies: List of anomalies, each anomaly is a dict containing center, radius, conductivity.

    Returns:
        Conductivity distribution Function object.
    """

    if anomalies is None:
        anomalies = []

    # Create background conductivity distribution
    sigma = Function(fwd_model.V_sigma)
    sigma.vector()[:] = background_conductivity

    # Add anomalies
    for anomaly in anomalies:
        center = anomaly.get('center', (0.0, 0.0))
        radius = anomaly.get('radius', 0.2)
        conductivity = anomaly.get('conductivity', 2.0)

        for cell in cells(fwd_model.mesh):
            cell_center = cell.midpoint()
            x, y = cell_center.x(), cell_center.y()
            if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                sigma.vector()[cell.index()] = conductivity

    return sigma