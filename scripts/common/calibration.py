"""Measurement data calibration utility functions."""

from __future__ import annotations

import numpy as np


def compute_scale_bias(measured: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    """Compute linear coefficients to scale/shift model response to measured scale.

    Uses least squares to find scale and bias such that:
        measured ≈ scale * model + bias

    Args:
        measured: Measured data vector.
        model: Model prediction vector.

    Returns:
        (scale, bias) linear transformation coefficients.
    """
    x = np.asarray(model, dtype=float)
    y = np.asarray(measured, dtype=float)

    denom = float(np.dot(x, x))
    if denom < 1e-18:
        return 1.0, float(y.mean())

    scale = float(np.dot(y, x) / denom)
    if abs(scale) < 1e-12:
        scale = 1.0 if scale >= 0 else -1.0

    bias = float(y.mean() - scale * x.mean())
    return scale, bias


def apply_calibration(vector: np.ndarray, scale: float, bias: float) -> np.ndarray:
    """Apply inverse calibration transform.

    Transform data from calibrated space back to original space:
        original = (calibrated - bias) / scale

    Args:
        vector: Data vector to transform.
        scale: Scale coefficient.
        bias: Bias coefficient.

    Returns:
        Transformed vector.
    """
    arr = np.asarray(vector, dtype=float)
    safe_scale = abs(scale)
    if safe_scale < np.finfo(float).eps:
        safe_scale = 1.0
    return (arr - bias) / safe_scale


def normalize_measurements(
    measurements: np.ndarray,
    target_amplitude: float | None = None,
    source_amplitude: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Normalize measurements to target amplitude.

    Args:
        measurements: Measurement data.
        target_amplitude: Target stimulation amplitude, None means no scaling.
        source_amplitude: Original stimulation amplitude.

    Returns:
        (normalized_measurements, scale_factor)
    """
    if target_amplitude is None or target_amplitude <= 0:
        return measurements, 1.0
    if source_amplitude <= 0:
        raise ValueError("source_amplitude must be positive")

    factor = target_amplitude / source_amplitude
    return measurements * factor, factor
