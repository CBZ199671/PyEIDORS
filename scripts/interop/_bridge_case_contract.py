"""Stable case metadata and metrics for the EIDORS interoperability CLIs."""

from __future__ import annotations

from typing import Any

import numpy as np


PYEIDORS_REFINEMENTS = {
    "coarse": 6,
    "medium": 12,
    "fine": 18,
}

SCENARIO_CONFIG: dict[str, dict[str, Any]] = {
    "low_z": {
        "contact_impedance": 1e-6,
        "background": 1.0,
        "phantom_conductivity": 2.0,
        "phantom_center": (0.30, 0.20),
        "phantom_radius": 0.20,
        "label": "saline-like",
    },
    "high_z": {
        "contact_impedance": 1e-2,
        "background": 1.0,
        "phantom_conductivity": 2.0,
        "phantom_center": (0.25, -0.22),
        "phantom_radius": 0.18,
        "label": "plant-like",
    },
}


def real_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if np.iscomplexobj(array):
        scale = max(1.0, float(np.max(np.abs(array.real), initial=0.0)))
        imag_max = float(np.max(np.abs(array.imag), initial=0.0))
        tolerance = 100.0 * np.finfo(array.real.dtype).eps * scale
        if imag_max > tolerance:
            raise ValueError(
                f"{name} has non-negligible imaginary values: "
                f"max_abs_imag={imag_max:.3e}, tolerance={tolerance:.3e}"
            )
        array = array.real
    return np.asarray(array, dtype=np.float64)


def conductivity_metrics(
    true_values: np.ndarray,
    recon_values: np.ndarray,
) -> dict[str, float]:
    target = real_vector(true_values, name="true conductivity")
    predicted = real_vector(recon_values, name="reconstructed conductivity")
    error = predicted - target
    return {
        "conductivity_mae": float(np.mean(np.abs(error))),
        "conductivity_rmse": float(np.sqrt(np.mean(error**2))),
        "conductivity_relative_error_pct": float(
            np.linalg.norm(error)
            / max(np.linalg.norm(target), np.finfo(float).eps)
            * 100.0
        ),
    }


def voltage_metrics(
    target_values: np.ndarray,
    predicted_values: np.ndarray,
) -> dict[str, float]:
    target = real_vector(target_values, name="target voltage")
    predicted = real_vector(predicted_values, name="predicted voltage")
    error = predicted - target
    return {
        "voltage_rmse": float(np.sqrt(np.mean(error**2))),
        "voltage_mae": float(np.mean(np.abs(error))),
        "voltage_relative_error_pct": float(
            np.linalg.norm(error)
            / max(np.linalg.norm(target), np.finfo(float).eps)
            * 100.0
        ),
    }
