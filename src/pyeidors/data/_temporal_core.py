"""Shared temporal-array validation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_frame_batch(values: Any) -> tuple[np.ndarray, bool]:
    """Return a contiguous ``(n_frames, n_values)`` batch and vector flag."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        batch = array.reshape(1, -1)
        was_vector = True
    elif array.ndim == 2:
        batch = array
        was_vector = False
    else:
        raise ValueError("frames must be a 1D vector or 2D frame batch.")
    if 0 in batch.shape:
        raise ValueError("frames must be non-empty.")
    if not np.isfinite(batch).all():
        raise FloatingPointError("frames contain non-finite values.")
    return np.ascontiguousarray(batch, dtype=np.float64), was_vector


def positive_int(value: int, name: str) -> int:
    """Validate a positive integer parameter."""

    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def unit_interval(value: float, name: str) -> float:
    """Validate a finite scalar in the closed interval ``[0, 1]``."""

    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1].")
    return resolved


__all__ = ["as_frame_batch", "positive_int", "unit_interval"]
