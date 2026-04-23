"""Synthetic EIT data generator."""

from __future__ import annotations

from typing import Any

import numpy as np

from .structures import EITData, EITImage


def _center_vector(center: tuple[float, ...] | list[float], *, ndim: int) -> np.ndarray:
    arr = np.asarray(center, dtype=float).reshape(-1)
    if arr.size >= ndim:
        return arr[:ndim]
    padded = np.zeros(ndim, dtype=float)
    padded[: arr.size] = arr
    return padded


def _paint_circle(
    values: np.ndarray,
    centers: np.ndarray,
    center: tuple[float, ...] | list[float],
    radius: float,
    conductivity: float,
) -> None:
    if centers.size == 0:
        return
    ndim = min(centers.shape[1], 3)
    center_vec = _center_vector(center, ndim=ndim)
    deltas = centers[:, :ndim] - center_vec[None, :]
    dist2 = np.sum(deltas**2, axis=1)
    values[dist2 < float(radius) ** 2] = conductivity


def _paint_ellipse(
    values: np.ndarray,
    centers: np.ndarray,
    center: tuple[float, ...] | list[float],
    rx: float,
    ry: float,
    conductivity: float,
    rz: float | None = None,
) -> None:
    """Paint an ellipse in 2D or an ellipsoid in 3D."""
    if centers.size == 0 or rx <= 0 or ry <= 0:
        return
    if centers.shape[1] >= 3:
        z_radius = float(rz if rz is not None and rz > 0 else rx)
        center_vec = _center_vector(center, ndim=3)
        norm = (
            ((centers[:, 0] - center_vec[0]) / rx) ** 2
            + ((centers[:, 1] - center_vec[1]) / ry) ** 2
            + ((centers[:, 2] - center_vec[2]) / z_radius) ** 2
        )
        values[norm < 1.0] = conductivity
        return
    center_vec = _center_vector(center, ndim=2)
    norm = ((centers[:, 0] - center_vec[0]) / rx) ** 2 + (
        (centers[:, 1] - center_vec[1]) / ry
    ) ** 2
    values[norm < 1.0] = conductivity


def _paint_rectangle(
    values: np.ndarray,
    centers: np.ndarray,
    center: tuple[float, ...] | list[float],
    half_w: float,
    half_h: float,
    conductivity: float,
    half_d: float | None = None,
) -> None:
    """Paint a rectangle in 2D or an axis-aligned box in 3D."""
    if centers.size == 0 or half_w <= 0 or half_h <= 0:
        return
    if centers.shape[1] >= 3:
        z_half = float(half_d if half_d is not None and half_d > 0 else half_w)
        center_vec = _center_vector(center, ndim=3)
        mask = (
            (np.abs(centers[:, 0] - center_vec[0]) < half_w)
            & (np.abs(centers[:, 1] - center_vec[1]) < half_h)
            & (np.abs(centers[:, 2] - center_vec[2]) < z_half)
        )
    else:
        center_vec = _center_vector(center, ndim=2)
        mask = (np.abs(centers[:, 0] - center_vec[0]) < half_w) & (
            np.abs(centers[:, 1] - center_vec[1]) < half_h
        )
    values[mask] = conductivity


def create_synthetic_data(
    fwd_model,
    inclusion_conductivity: float = 2.5,
    background_conductivity: float = 1.0,
    noise_level: float = 0.02,
    center: tuple[float, ...] = (0.2, 0.2),
    radius: float = 0.3,
) -> dict[str, Any]:
    """Create synthetic EIT test data."""
    from dolfinx import fem
    from ..femx import cell_midpoints

    sigma_true = fem.Function(fwd_model.V_sigma)
    sigma_true.x.array[:] = background_conductivity

    centers = cell_midpoints(fwd_model.mesh)
    _paint_circle(sigma_true.x.array, centers, center, radius, inclusion_conductivity)

    img_true = EITImage(elem_data=sigma_true.x.array.copy(), fwd_model=fwd_model)
    data_clean, _ = fwd_model.fwd_solve(img_true)

    np.random.seed(42)
    noise = (
        noise_level * np.std(data_clean.meas) * np.random.randn(len(data_clean.meas))
    )
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
    from dolfinx import fem
    from ..femx import cell_midpoints

    if anomalies is None:
        anomalies = []

    sigma = fem.Function(fwd_model.V_sigma)
    sigma.x.array[:] = background_conductivity

    centers = cell_midpoints(fwd_model.mesh)
    for anomaly in anomalies:
        center = anomaly.get("center", (0.0, 0.0))
        conductivity = anomaly.get("conductivity", 2.0)
        shape = anomaly.get("shape", "circle")

        if shape in {"ellipse", "ellipsoid"}:
            rx = anomaly.get("rx", anomaly.get("radius", 0.2))
            ry = anomaly.get("ry", rx)
            rz = anomaly.get("rz", anomaly.get("size_z", rx))
            _paint_ellipse(sigma.x.array, centers, center, rx, ry, conductivity, rz=rz)
        elif shape in {"rectangle", "box"}:
            half_w = anomaly.get("half_w", anomaly.get("radius", 0.2))
            half_h = anomaly.get("half_h", half_w)
            half_d = anomaly.get("half_d", anomaly.get("size_z", half_w))
            _paint_rectangle(
                sigma.x.array,
                centers,
                center,
                half_w,
                half_h,
                conductivity,
                half_d=half_d,
            )
        else:
            radius = anomaly.get("radius", 0.2)
            _paint_circle(sigma.x.array, centers, center, radius, conductivity)

    return sigma
