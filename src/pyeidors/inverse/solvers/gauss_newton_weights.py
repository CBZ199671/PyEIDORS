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

    x = np.asarray(baseline_vector)
    y = np.asarray(
        measured_vector,
        dtype=np.result_type(x.dtype, np.asarray(measured_vector).dtype),
    )
    denom = float(np.vdot(x, x).real)
    if denom < 1e-18:
        return x
    scale = np.vdot(x, y) / denom
    if abs(scale) < 1e-12:
        scale = 1.0
    bias = y.mean() - scale * x.mean()
    return scale * x + bias


def difference_with_baseline(
    baseline_vector: np.ndarray,
    measured_vector: np.ndarray | None,
    floor: float,
) -> np.ndarray:
    """Difference-magnitude weighting mode."""
    if measured_vector is None:
        return baseline_vector
    measured = np.asarray(measured_vector)
    baseline = np.asarray(baseline_vector)
    delta = np.subtract(measured, baseline)
    if np.iscomplexobj(delta):
        diff = np.empty(delta.shape, dtype=np.empty((), dtype=delta.dtype).real.dtype)
        np.abs(delta, out=diff)
    else:
        diff = delta
        np.abs(diff, out=diff)
    np.maximum(diff, float(floor), out=diff)
    return diff


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
