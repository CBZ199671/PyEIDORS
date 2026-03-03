"""Measurement weighting helpers for Gauss-Newton solver."""

from __future__ import annotations

import numpy as np


def scale_baseline_to_measured(
    baseline_vector: np.ndarray,
    measured_vector: np.ndarray | None,
) -> np.ndarray:
    """Linearly scale baseline measurements to measured scale."""
    if measured_vector is None:
        return baseline_vector

    x = np.asarray(baseline_vector, dtype=float)
    y = np.asarray(measured_vector, dtype=float)
    denom = float(np.dot(x, x))
    if denom < 1e-18:
        return x
    scale = float(np.dot(y, x) / denom)
    if abs(scale) < 1e-12:
        scale = 1.0 if scale >= 0 else -1.0
    bias = float(y.mean() - scale * x.mean())
    return scale * x + bias


def difference_with_baseline(
    baseline_vector: np.ndarray,
    measured_vector: np.ndarray | None,
    floor: float,
) -> np.ndarray:
    """Difference-magnitude weighting mode."""
    if measured_vector is None:
        return baseline_vector
    diff = np.abs(np.asarray(measured_vector, dtype=float) - np.asarray(baseline_vector, dtype=float))
    return np.where(diff > floor, diff, floor)


def build_weight_reference(
    strategy: str,
    baseline_vector: np.ndarray,
    measured_vector: np.ndarray | None,
    floor: float,
) -> np.ndarray:
    """Resolve weighting reference according to the configured strategy."""
    if strategy == "scaled_baseline":
        return scale_baseline_to_measured(baseline_vector, measured_vector)
    if strategy == "difference":
        return difference_with_baseline(baseline_vector, measured_vector, floor)
    return baseline_vector
