"""Array metric helpers for benchmark and diagnostic scripts."""

from __future__ import annotations

import numpy as np


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Pearson correlation without stacking inputs into a 2-by-N matrix."""

    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.size <= 1 or left_arr.size != right_arr.size:
        return float("nan")

    left_centered = np.array(left_arr, dtype=np.float64, copy=True)
    right_centered = np.array(right_arr, dtype=np.float64, copy=True)
    left_centered -= float(np.mean(left_centered))
    right_centered -= float(np.mean(right_centered))

    numerator = float(np.dot(left_centered, right_centered))
    left_norm2 = float(np.dot(left_centered, left_centered))
    right_norm2 = float(np.dot(right_centered, right_centered))
    denominator = float(np.sqrt(left_norm2 * right_norm2))
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def finite_pearson_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_count: int = 2,
    variance_floor: float | None = None,
) -> float:
    """Compute Pearson correlation over finite pairs without compacting them."""

    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.size != right_arr.size:
        return float("nan")

    finite = np.isfinite(left_arr)
    np.logical_and(finite, np.isfinite(right_arr), out=finite)
    count = int(np.count_nonzero(finite))
    if count < int(min_count):
        return float("nan")

    left_mean = float(np.sum(left_arr, where=finite, initial=0.0) / count)
    right_mean = float(np.sum(right_arr, where=finite, initial=0.0) / count)
    left_centered = np.array(left_arr, dtype=np.float64, copy=True)
    right_centered = np.array(right_arr, dtype=np.float64, copy=True)
    left_centered -= left_mean
    right_centered -= right_mean
    invalid = np.logical_not(finite)
    if np.any(invalid):
        left_centered[invalid] = 0.0
        right_centered[invalid] = 0.0

    numerator = float(np.dot(left_centered, right_centered))
    left_norm2 = float(np.dot(left_centered, left_centered))
    right_norm2 = float(np.dot(right_centered, right_centered))
    floor = (
        np.finfo(np.float64).eps
        if variance_floor is None
        else max(float(variance_floor), 0.0)
    )
    if left_norm2 <= floor or right_norm2 <= floor:
        return float("nan")
    return float(numerator / np.sqrt(left_norm2 * right_norm2))


def safe_finite_pearson_correlation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_count: int = 2,
) -> float:
    """Finite-pair Pearson with legacy constant-vector 1/0 fallback."""

    corr = finite_pearson_correlation(left, right, min_count=min_count)
    if np.isfinite(corr):
        return float(corr)

    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.size != right_arr.size:
        return float("nan")
    finite = np.isfinite(left_arr)
    np.logical_and(finite, np.isfinite(right_arr), out=finite)
    if int(np.count_nonzero(finite)) < int(min_count):
        return float("nan")

    left_span = float(
        np.max(left_arr, where=finite, initial=-np.inf)
        - np.min(left_arr, where=finite, initial=np.inf)
    )
    right_span = float(
        np.max(right_arr, where=finite, initial=-np.inf)
        - np.min(right_arr, where=finite, initial=np.inf)
    )
    left_scale = max(
        float(np.max(np.abs(left_arr), where=finite, initial=0.0)),
        1.0,
    )
    right_scale = max(
        float(np.max(np.abs(right_arr), where=finite, initial=0.0)),
        1.0,
    )
    left_constant = left_span <= 1.0e-8 + 1.0e-5 * left_scale
    right_constant = right_span <= 1.0e-8 + 1.0e-5 * right_scale
    if not (left_constant or right_constant):
        return float("nan")

    diff = np.abs(left_arr - right_arr)
    tolerance = 1.0e-8 + 1.0e-5 * np.abs(right_arr)
    mismatch = np.greater(diff, tolerance)
    return 0.0 if bool(np.any(mismatch & finite)) else 1.0


def mean_where(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    fallback: float = float("nan"),
) -> float:
    """Mean over a boolean mask without compacting selected values."""

    mask_arr = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(mask_arr))
    if count == 0:
        return float(fallback)
    return float(
        np.sum(
            np.asarray(values, dtype=np.float64),
            where=mask_arr,
            initial=0.0,
            dtype=np.float64,
        )
        / count
    )
