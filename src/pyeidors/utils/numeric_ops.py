"""Numerical helpers with finite-value guards."""

from __future__ import annotations

from typing import Any

import numpy as np


_COMPLEX_SCAN_CHUNK_ITEMS = 1_048_576
_SCALED_ADD_CHUNK_ITEMS = 1_048_576


def all_finite_values(values: Any, *, chunk_size: int = 1_048_576) -> bool:
    """Return true when every value is finite using a bounded bool work buffer."""

    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return True
    block_size = max(1, min(int(chunk_size), int(array.size)))
    work = np.empty(block_size, dtype=bool)
    for start in range(0, int(array.size), block_size):
        stop = min(start + block_size, int(array.size))
        chunk = array[start:stop]
        chunk_mask = work[: chunk.size]
        np.isfinite(chunk, out=chunk_mask)
        if not bool(chunk_mask.all()):
            return False
    return True


def has_nonzero_imaginary(
    values: Any,
    *,
    tol: float | None = None,
    chunk_size: int = _COMPLEX_SCAN_CHUNK_ITEMS,
) -> bool:
    """Return true when a complex payload has imaginary magnitude above ``tol``."""

    array = np.asarray(values)
    if array.size == 0 or not np.iscomplexobj(array):
        return False
    threshold = np.finfo(np.float64).eps if tol is None else float(tol)
    imag = np.ravel(np.imag(array), order="K")
    block_size = max(1, min(int(chunk_size), int(imag.size)))
    abs_work = np.empty(block_size, dtype=imag.dtype)
    mask_work = np.empty(block_size, dtype=bool)
    for start in range(0, int(imag.size), block_size):
        stop = min(start + block_size, int(imag.size))
        chunk = imag[start:stop]
        abs_chunk = abs_work[: chunk.size]
        mask_chunk = mask_work[: chunk.size]
        np.abs(chunk, out=abs_chunk)
        np.greater(abs_chunk, threshold, out=mask_chunk)
        if bool(mask_chunk.any()):
            return True
    return False


def any_equal_values(
    values: Any,
    target: Any,
    *,
    chunk_size: int = _SCALED_ADD_CHUNK_ITEMS,
) -> bool:
    """Return true if any element equals ``target`` using bounded scratch."""

    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return False
    block_size = max(1, min(int(chunk_size), int(array.size)))
    work = np.empty(block_size, dtype=bool)
    for start in range(0, int(array.size), block_size):
        stop = min(start + block_size, int(array.size))
        chunk = array[start:stop]
        mask = work[: chunk.size]
        np.equal(chunk, target, out=mask)
        if bool(mask.any()):
            return True
    return False


def any_not_equal_values(
    left: Any,
    right: Any,
    *,
    chunk_size: int = _SCALED_ADD_CHUNK_ITEMS,
) -> bool:
    """Return true if same-shaped arrays differ using bounded scratch."""

    left_array = np.asarray(left).reshape(-1)
    right_array = np.asarray(right).reshape(-1)
    if left_array.shape != right_array.shape:
        raise ValueError("left and right must have identical element counts.")
    if left_array.size == 0:
        return False
    block_size = max(1, min(int(chunk_size), int(left_array.size)))
    work = np.empty(block_size, dtype=bool)
    for start in range(0, int(left_array.size), block_size):
        stop = min(start + block_size, int(left_array.size))
        chunk = work[: stop - start]
        np.not_equal(left_array[start:stop], right_array[start:stop], out=chunk)
        if bool(chunk.any()):
            return True
    return False


def any_abs_less_equal_values(
    values: Any,
    threshold: float,
    *,
    chunk_size: int = _SCALED_ADD_CHUNK_ITEMS,
) -> bool:
    """Return true if any absolute value is ``<= threshold`` with bounded scratch."""

    array = np.asarray(values).reshape(-1)
    if array.size == 0:
        return False
    block_size = max(1, min(int(chunk_size), int(array.size)))
    abs_dtype = np.float64 if np.iscomplexobj(array) else array.dtype
    abs_work = np.empty(block_size, dtype=abs_dtype)
    mask_work = np.empty(block_size, dtype=bool)
    limit = float(threshold)
    for start in range(0, int(array.size), block_size):
        stop = min(start + block_size, int(array.size))
        chunk = array[start:stop]
        abs_chunk = abs_work[: chunk.size]
        mask_chunk = mask_work[: chunk.size]
        np.abs(chunk, out=abs_chunk)
        np.less_equal(abs_chunk, limit, out=mask_chunk)
        if bool(mask_chunk.any()):
            return True
    return False


def min_alpha_for_value_floor(
    values: Any,
    deltas: Any,
    floor: float,
    *,
    chunk_size: int = _SCALED_ADD_CHUNK_ITEMS,
) -> float:
    """Return min ``(value-floor)/(-delta)`` for negative deltas, or ``inf``."""

    value_array = np.asarray(values, dtype=np.float64).reshape(-1)
    delta_array = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if value_array.shape != delta_array.shape:
        raise ValueError("values and deltas must have identical element counts.")
    if value_array.size == 0:
        return np.inf
    block_size = max(1, min(int(chunk_size), int(value_array.size)))
    negative_work = np.empty(block_size, dtype=bool)
    bad_work = np.empty(block_size, dtype=bool)
    margin_work = np.empty(block_size, dtype=np.float64)
    ratio_work = np.empty(block_size, dtype=np.float64)
    floor_value = float(floor)
    best = np.inf
    for start in range(0, int(value_array.size), block_size):
        stop = min(start + block_size, int(value_array.size))
        values_chunk = value_array[start:stop]
        deltas_chunk = delta_array[start:stop]
        negative = negative_work[: stop - start]
        np.less(deltas_chunk, 0.0, out=negative)
        if not bool(negative.any()):
            continue

        margin = margin_work[: stop - start]
        bad = bad_work[: stop - start]
        np.subtract(values_chunk, floor_value, out=margin)
        np.less_equal(margin, 0.0, out=bad)
        np.logical_and(bad, negative, out=bad)
        if bool(bad.any()):
            return 0.0

        ratios = ratio_work[: stop - start]
        np.negative(deltas_chunk, out=ratios)
        np.divide(margin, ratios, out=ratios, where=negative)
        best = float(np.min(ratios, where=negative, initial=best))
    return best


def add_scaled_values_in_place(
    target: Any,
    values: Any,
    scale: float,
    *,
    chunk_size: int = _SCALED_ADD_CHUNK_ITEMS,
) -> None:
    """Compute ``target += scale * values`` with one bounded work buffer."""

    target_array = np.asarray(target)
    values_array = np.asarray(values, dtype=target_array.dtype)
    if target_array.shape != values_array.shape:
        raise ValueError(
            "target and values must have identical shapes for scaled in-place add."
        )
    if target_array.size == 0 or float(scale) == 0.0:
        return
    target_flat = target_array.reshape(-1)
    values_flat = values_array.reshape(-1)
    block_size = max(1, min(int(chunk_size), int(target_flat.size)))
    work = np.empty(block_size, dtype=target_array.dtype)
    for start in range(0, int(target_flat.size), block_size):
        stop = min(start + block_size, int(target_flat.size))
        chunk = work[: stop - start]
        np.multiply(values_flat[start:stop], scale, out=chunk)
        np.add(target_flat[start:stop], chunk, out=target_flat[start:stop])


def add_scaled_diagonal_in_place(
    target: Any,
    diagonal: Any,
    scale: float,
    *,
    chunk_size: int = _SCALED_ADD_CHUNK_ITEMS,
) -> None:
    """Compute ``target[i, i] += scale * diagonal[i]`` with bounded scratch."""

    matrix = np.asarray(target)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("target must be a square 2D matrix.")
    if not matrix.flags.c_contiguous:
        raise ValueError("target must be C-contiguous for diagonal in-place add.")
    diag = np.asarray(diagonal, dtype=matrix.dtype).reshape(-1)
    if diag.size != matrix.shape[0]:
        raise ValueError(
            f"diagonal length {diag.size} does not match matrix size {matrix.shape[0]}."
        )
    if diag.size == 0 or float(scale) == 0.0:
        return
    matrix_diag = matrix.reshape(-1)[:: matrix.shape[0] + 1]
    block_size = max(1, min(int(chunk_size), int(diag.size)))
    work = np.empty(block_size, dtype=matrix.dtype)
    for start in range(0, int(diag.size), block_size):
        stop = min(start + block_size, int(diag.size))
        chunk = work[: stop - start]
        np.multiply(diag[start:stop], scale, out=chunk)
        np.add(matrix_diag[start:stop], chunk, out=matrix_diag[start:stop])


def min_positive_finite_value(
    values: Any,
    *,
    fallback: float,
    chunk_size: int = _SCALED_ADD_CHUNK_ITEMS,
) -> float:
    """Return the minimum finite value greater than zero without subset copies."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return float(fallback)
    block_size = max(1, min(int(chunk_size), int(array.size)))
    finite_work = np.empty(block_size, dtype=bool)
    positive_work = np.empty(block_size, dtype=bool)
    best = np.inf
    for start in range(0, int(array.size), block_size):
        stop = min(start + block_size, int(array.size))
        chunk = array[start:stop]
        finite_chunk = finite_work[: chunk.size]
        positive_chunk = positive_work[: chunk.size]
        np.isfinite(chunk, out=finite_chunk)
        np.greater(chunk, 0.0, out=positive_chunk)
        np.logical_and(positive_chunk, finite_chunk, out=positive_chunk)
        if not bool(positive_chunk.any()):
            continue
        candidate = float(np.min(chunk, where=positive_chunk, initial=np.inf))
        if candidate < best:
            best = candidate
    return float(fallback) if not np.isfinite(best) else float(best)


def _finite_summary(values: np.ndarray) -> str:
    arr = np.asarray(values).reshape(-1)
    total_count = int(arr.size)
    use_abs_range = np.iscomplexobj(arr)
    finite_count = 0
    min_value = np.inf
    max_value = -np.inf
    for raw_value in np.nditer(arr, flags=["refs_ok"], op_flags=["readonly"]):
        value = raw_value.item()
        if not bool(np.isfinite(value)):
            continue
        finite_count += 1
        range_value = float(abs(value)) if use_abs_range else float(value)
        if range_value < min_value:
            min_value = range_value
        if range_value > max_value:
            max_value = range_value
    non_finite_count = total_count - finite_count
    if finite_count == 0:
        return f"total={total_count} finite=0 non_finite={non_finite_count}"
    return (
        f"total={total_count} finite={finite_count} non_finite={non_finite_count} "
        f"min={float(min_value):.6e} "
        f"max={float(max_value):.6e}"
    )


def _as_finite_array(value: Any, name: str) -> np.ndarray:
    raw = np.asarray(value)
    array = (
        np.asarray(raw) if np.iscomplexobj(raw) else np.asarray(raw, dtype=np.float64)
    )
    if not all_finite_values(array):
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
    result_array = np.asarray(result)
    if not all_finite_values(result_array):
        raise FloatingPointError(
            f"{op_name} produced non-finite values: {_finite_summary(result_array)}"
        )
    if result_array.ndim == 0:
        scalar = result_array.reshape(()).item()
        return scalar if isinstance(scalar, complex) else float(scalar)
    return result_array


def squared_distances_to_point(
    points: Any,
    center: Any,
    *,
    ndim: int | None = None,
) -> np.ndarray:
    """Return squared distances using one reusable work vector."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D coordinate array.")
    resolved_ndim = pts.shape[1] if ndim is None else min(int(ndim), pts.shape[1])
    if resolved_ndim <= 0:
        return np.zeros(pts.shape[0], dtype=np.float64)

    center_values = np.asarray(center, dtype=np.float64).reshape(-1)
    center_vec = np.zeros(resolved_ndim, dtype=np.float64)
    copy_count = min(center_values.size, resolved_ndim)
    if copy_count:
        center_vec[:copy_count] = center_values[:copy_count]

    distances = np.zeros(pts.shape[0], dtype=np.float64)
    work = np.empty(pts.shape[0], dtype=np.float64)
    for axis in range(resolved_ndim):
        np.subtract(pts[:, axis], center_vec[axis], out=work)
        np.square(work, out=work)
        distances += work
    return distances
