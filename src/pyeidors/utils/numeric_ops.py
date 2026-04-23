"""Numerical helpers with finite-value guards."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_summary(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    finite_count = int(finite.size)
    total_count = int(values.size)
    non_finite_count = total_count - finite_count
    if finite_count == 0:
        return f"total={total_count} finite=0 non_finite={non_finite_count}"
    return (
        f"total={total_count} finite={finite_count} non_finite={non_finite_count} "
        f"min={float(finite.min()):.6e} max={float(finite.max()):.6e}"
    )


def _as_finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise FloatingPointError(
            f"{name} contains non-finite values: {_finite_summary(array)}"
        )
    return array


def safe_dot(lhs: Any, rhs: Any, op_name: str) -> np.ndarray | float:
    """Compute dot product with strict finite-value checks.

    Uses ``np.dot`` to avoid platform-specific ``np.matmul`` warning noise
    observed in the current Nix/macOS stack.
    """

    lhs_array = _as_finite_array(lhs, f"{op_name}.lhs")
    rhs_array = _as_finite_array(rhs, f"{op_name}.rhs")
    result = np.dot(lhs_array, rhs_array)
    result_array = np.asarray(result, dtype=float)
    if not np.isfinite(result_array).all():
        raise FloatingPointError(
            f"{op_name} produced non-finite values: {_finite_summary(result_array)}"
        )
    if result_array.ndim == 0:
        return float(result_array)
    return result_array
