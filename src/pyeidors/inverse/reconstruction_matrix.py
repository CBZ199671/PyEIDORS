"""Online reconstruction-matrix helpers for difference EIT."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.data.difference import normalize_time_difference
from pyeidors.utils.numeric_ops import safe_dot


def _as_measurement_vector(values: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim > 2:
        raise ValueError(f"{name} must be a 1D or column-vector measurement array.")
    vector = vector.reshape(-1)
    if vector.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _matvec(rm: Any, vector: np.ndarray) -> np.ndarray:
    if sparse.issparse(rm):
        matrix = rm.tocsr()
        if matrix.ndim != 2:
            raise ValueError("rm must be a 2D reconstruction matrix.")
        if matrix.shape[1] != vector.size:
            raise ValueError(
                f"RM column count {matrix.shape[1]} does not match dv length {vector.size}."
            )
        out = np.asarray(matrix @ vector, dtype=np.float64)
    else:
        matrix = np.asarray(rm, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("rm must be a 2D reconstruction matrix.")
        if matrix.shape[1] != vector.size:
            raise ValueError(
                f"RM column count {matrix.shape[1]} does not match dv length {vector.size}."
            )
        out = np.asarray(
            safe_dot(matrix, vector, "reconstruction_matrix.apply"), dtype=np.float64
        )
    if not np.isfinite(out).all():
        raise FloatingPointError("RM application produced non-finite values.")
    return out.reshape(-1)


def reconstruct_difference(
    rm: Any,
    dv,
    *,
    normalize: bool = True,
    v_ref=None,
    floor: float | None = None,
) -> np.ndarray:
    """Apply a precomputed reconstruction matrix to one difference frame.

    If ``normalize`` is true and ``v_ref`` is provided, ``dv`` is interpreted
    as target voltages ``v_t`` and first converted with
    :func:`normalize_time_difference`. Otherwise ``dv`` is treated as an
    already-projected measurement vector. The hot path is deliberately just
    ``RM @ dv_projected``; RM construction belongs to later T16/T17 tasks.
    """

    if normalize and v_ref is not None:
        measurement = normalize_time_difference(dv, v_ref, floor=floor)
    else:
        measurement = _as_measurement_vector(dv, name="dv")
    return _matvec(rm, measurement)


__all__ = ["reconstruct_difference"]
