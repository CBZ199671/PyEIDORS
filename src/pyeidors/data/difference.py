"""Difference-imaging measurement-space helpers."""

from __future__ import annotations

from typing import Final

import numpy as np

DEFAULT_DIFFERENCE_MODE: Final[str] = "raw"
DEFAULT_DIFFERENCE_ORIENTATION: Final[str] = "target_minus_reference"
_VALID_DIFFERENCE_MODES: Final[set[str]] = {"raw", "normalized"}
_VALID_DIFFERENCE_ORIENTATIONS: Final[set[str]] = {
    "target_minus_reference",
    "reference_minus_target",
}
_REFERENCE_FLOOR_CHUNK_ITEMS: Final[int] = 65536


def _complex_or_float_dtype(*values) -> np.dtype:
    dtypes = [np.asarray(value).dtype for value in values]
    complex_dtypes = [
        dtype for dtype in dtypes if np.issubdtype(dtype, np.complexfloating)
    ]
    if complex_dtypes:
        if any(dtype != np.dtype(np.complex64) for dtype in complex_dtypes):
            return np.dtype(np.complex128)
        return np.dtype(np.complex64)
    float_dtypes = [dtype for dtype in dtypes if np.issubdtype(dtype, np.floating)]
    if float_dtypes and all(
        dtype.itemsize <= np.dtype(np.float32).itemsize for dtype in float_dtypes
    ):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def normalize_difference_mode(
    mode: str | None, *, default: str = DEFAULT_DIFFERENCE_MODE
) -> str:
    """Return a validated difference mode."""
    fallback = str(default).strip().lower() or DEFAULT_DIFFERENCE_MODE
    resolved = fallback if mode is None else str(mode).strip().lower()
    if resolved not in _VALID_DIFFERENCE_MODES:
        raise ValueError(
            f"Unsupported difference_mode={mode!r}. "
            "Expected one of: 'raw', 'normalized'."
        )
    return resolved


def normalize_difference_orientation(
    orientation: str | None,
    *,
    default: str = DEFAULT_DIFFERENCE_ORIENTATION,
) -> str:
    """Return a validated difference orientation."""
    fallback = str(default).strip().lower() or DEFAULT_DIFFERENCE_ORIENTATION
    resolved = fallback if orientation is None else str(orientation).strip().lower()
    if resolved not in _VALID_DIFFERENCE_ORIENTATIONS:
        raise ValueError(
            f"Unsupported difference_orientation={orientation!r}. "
            "Expected one of: 'target_minus_reference', 'reference_minus_target'."
        )
    return resolved


def _as_measurement_vector(values, *, name: str) -> np.ndarray:
    dtype = _complex_or_float_dtype(values)
    array = np.asarray(values, dtype=dtype)
    if array.ndim > 2:
        raise ValueError(
            f"{name} must be a 1D or 2D measurement vector, got {array.ndim}D."
        )
    return array.reshape(-1)


def _reference_floor_eps(floor: float | None) -> float:
    return (
        np.finfo(np.float64).eps
        if floor is None
        else float(max(floor, np.finfo(np.float64).eps))
    )


def _abs_work_dtype(values: np.ndarray) -> np.dtype:
    dtype = np.asarray(values).dtype
    if np.issubdtype(dtype, np.floating):
        return (
            dtype
            if dtype.itemsize <= np.dtype(np.float32).itemsize
            else np.dtype(np.float64)
        )
    if np.issubdtype(dtype, np.complexfloating):
        return (
            np.dtype(np.float32)
            if dtype.itemsize <= np.dtype(np.complex64).itemsize
            else np.dtype(np.float64)
        )
    return np.dtype(np.float64)


def _reference_has_near_zero(
    reference_meas: np.ndarray,
    eps: float,
    *,
    chunk_size: int = _REFERENCE_FLOOR_CHUNK_ITEMS,
) -> bool:
    arr = np.asarray(reference_meas).reshape(-1)
    if arr.size == 0:
        return False
    block_size = max(1, min(int(chunk_size), int(arr.size)))
    abs_work = np.empty(block_size, dtype=_abs_work_dtype(arr))
    small_work = np.empty(block_size, dtype=bool)
    for start in range(0, int(arr.size), block_size):
        stop = min(start + block_size, int(arr.size))
        chunk = arr[start:stop]
        abs_chunk = abs_work[: chunk.size]
        small_chunk = small_work[: chunk.size]
        np.abs(chunk, out=abs_chunk)
        np.less(abs_chunk, eps, out=small_chunk)
        if bool(small_chunk.any()):
            return True
    return False


def _clamp_reference_floor_in_place(
    reference_meas: np.ndarray,
    eps: float,
    *,
    chunk_size: int = _REFERENCE_FLOOR_CHUNK_ITEMS,
) -> bool:
    arr = np.asarray(reference_meas).reshape(-1)
    if arr.size == 0:
        return False
    block_size = max(1, min(int(chunk_size), int(arr.size)))
    abs_work = np.empty(block_size, dtype=_abs_work_dtype(arr))
    small_work = np.empty(block_size, dtype=bool)
    aux_work = np.empty(block_size, dtype=bool)
    replacement_work = np.empty(block_size, dtype=arr.dtype)
    changed = False
    for start in range(0, int(arr.size), block_size):
        stop = min(start + block_size, int(arr.size))
        chunk = arr[start:stop]
        abs_chunk = abs_work[: chunk.size]
        small_chunk = small_work[: chunk.size]
        aux_chunk = aux_work[: chunk.size]
        replacement = replacement_work[: chunk.size]
        np.abs(chunk, out=abs_chunk)
        np.less(abs_chunk, eps, out=small_chunk)
        if not bool(small_chunk.any()):
            continue
        changed = True
        if np.iscomplexobj(arr):
            replacement.fill(1.0 + 0.0j)
            np.greater(abs_chunk, 0.0, out=aux_chunk)
            np.divide(chunk, abs_chunk, out=replacement, where=aux_chunk)
        else:
            np.sign(chunk, out=replacement)
            np.equal(replacement, 0.0, out=aux_chunk)
            np.copyto(replacement, 1.0, where=aux_chunk)
        replacement *= eps
        np.copyto(chunk, replacement, where=small_chunk)
    return changed


def _safe_reference(
    reference_meas: np.ndarray, *, floor: float | None = None
) -> np.ndarray:
    """Clamp near-zero reference values to ``+/-eps``, preserving sign."""
    dtype = _complex_or_float_dtype(reference_meas)
    safe = np.asarray(reference_meas, dtype=dtype).copy()
    _clamp_reference_floor_in_place(safe, _reference_floor_eps(floor))
    return safe


def build_difference_vector(
    target_meas,
    reference_meas,
    *,
    mode: str = DEFAULT_DIFFERENCE_MODE,
    orientation: str = DEFAULT_DIFFERENCE_ORIENTATION,
    floor: float | None = None,
) -> np.ndarray:
    """Project absolute target/reference voltages into difference-measurement space."""
    target = _as_measurement_vector(target_meas, name="target_meas")
    reference = _as_measurement_vector(reference_meas, name="reference_meas")
    if target.shape != reference.shape:
        raise ValueError(
            "target_meas and reference_meas must have identical shapes: "
            f"{target.shape!r} vs {reference.shape!r}."
        )

    resolved_mode = normalize_difference_mode(mode)
    resolved_orientation = normalize_difference_orientation(orientation)

    diff = target - reference
    if resolved_mode == "normalized":
        eps = _reference_floor_eps(floor)
        safe = (
            _safe_reference(reference, floor=floor)
            if _reference_has_near_zero(reference, eps)
            else reference
        )
        np.divide(diff, safe, out=diff)
    if resolved_orientation == "reference_minus_target":
        np.negative(diff, out=diff)
    return np.asarray(diff, dtype=np.result_type(target.dtype, reference.dtype))


def build_difference_frames(
    targets: np.ndarray,
    references: np.ndarray,
    *,
    mode: str = DEFAULT_DIFFERENCE_MODE,
    orientation: str = DEFAULT_DIFFERENCE_ORIENTATION,
    floor: float | None = None,
) -> np.ndarray:
    """Vectorized 2D batch of :func:`build_difference_vector`.

    ``targets`` must be a ``(n_frames, n_meas)`` array. ``references`` may be
    either the same shape, a single ``(n_meas,)`` vector, or a ``(1, n_meas)``
    row that broadcasts across frames. Real float32 inputs preserve float32;
    complex inputs preserve complex phase.
    """
    dtype = _complex_or_float_dtype(targets, references)
    target_batch = np.asarray(targets, dtype=dtype)
    reference_raw = np.asarray(references, dtype=dtype)
    if target_batch.ndim != 2:
        raise ValueError("targets must be a 2D frame batch.")
    if reference_raw.ndim == 1:
        if reference_raw.size != target_batch.shape[1]:
            raise ValueError(
                "1D references length must match target measurements: "
                f"{reference_raw.size!r} vs {target_batch.shape[1]!r}."
            )
        reference_batch = reference_raw.reshape(1, -1)
    elif reference_raw.ndim == 2:
        reference_batch = reference_raw
    else:
        raise ValueError("references must be a 1D vector or 2D frame batch.")
    if target_batch.shape != reference_batch.shape and reference_batch.shape != (
        1,
        target_batch.shape[1],
    ):
        raise ValueError(
            "targets and references must have compatible frame shapes: "
            f"{target_batch.shape!r} vs {reference_batch.shape!r}."
        )
    resolved_mode = normalize_difference_mode(mode)
    resolved_orientation = normalize_difference_orientation(orientation)
    diff = np.empty(target_batch.shape, dtype=dtype, order="C")
    np.subtract(target_batch, reference_batch, out=diff)
    if resolved_mode == "normalized":
        eps = _reference_floor_eps(floor)
        safe = reference_batch
        if _reference_has_near_zero(reference_batch, eps):
            safe = reference_batch.copy()
            _clamp_reference_floor_in_place(safe, eps)
        np.divide(diff, safe, out=diff)
    if resolved_orientation == "reference_minus_target":
        np.negative(diff, out=diff)
    return np.ascontiguousarray(diff, dtype=dtype)


def normalize_time_difference(
    v_t,
    v_ref,
    *,
    floor: float | None = None,
    orientation: str = DEFAULT_DIFFERENCE_ORIENTATION,
) -> np.ndarray:
    """Return normalized time-difference data ``(v_t - v_ref) / v_ref``.

    This is the public v1 front-end for offline-RM online reconstruction.
    It intentionally delegates to :func:`build_difference_vector` so the
    zero guard and orientation contract stay identical across legacy GN and
    EIDORS-style one-step/GREIT paths.
    """

    return build_difference_vector(
        v_t,
        v_ref,
        mode="normalized",
        orientation=orientation,
        floor=floor,
    )


def project_measurement_vector(
    simulated_meas,
    *,
    measurement_type: str = "real",
    reference_meas=None,
    difference_mode: str = DEFAULT_DIFFERENCE_MODE,
    difference_orientation: str = DEFAULT_DIFFERENCE_ORIENTATION,
    floor: float | None = None,
) -> np.ndarray:
    """Map absolute simulated measurements into the active inverse-data space."""
    simulated = _as_measurement_vector(simulated_meas, name="simulated_meas")
    if str(measurement_type).strip().lower() != "difference" or reference_meas is None:
        return simulated
    return build_difference_vector(
        simulated,
        reference_meas,
        mode=difference_mode,
        orientation=difference_orientation,
        floor=floor,
    )


def project_measurement_jacobian(
    jacobian,
    *,
    measurement_type: str = "real",
    reference_meas=None,
    difference_mode: str = DEFAULT_DIFFERENCE_MODE,
    difference_orientation: str = DEFAULT_DIFFERENCE_ORIENTATION,
    floor: float | None = None,
) -> np.ndarray:
    """Map an absolute Jacobian into the active inverse-data space."""
    jac_dtype = _complex_or_float_dtype(jacobian)
    jac = np.asarray(jacobian, dtype=jac_dtype)
    if jac.ndim != 2:
        raise ValueError(f"jacobian must be a 2D array, got shape {jac.shape!r}.")
    if str(measurement_type).strip().lower() != "difference" or reference_meas is None:
        return jac

    reference = _as_measurement_vector(reference_meas, name="reference_meas")
    if jac.shape[0] != reference.shape[0]:
        raise ValueError(
            "Jacobian row count must match reference measurement length: "
            f"{jac.shape[0]!r} vs {reference.shape[0]!r}."
        )

    resolved_mode = normalize_difference_mode(difference_mode)
    resolved_orientation = normalize_difference_orientation(difference_orientation)

    result_dtype = np.result_type(jac.dtype, reference.dtype)
    owns_projection = False
    projected = np.asarray(jac, dtype=result_dtype)
    if resolved_mode == "normalized":
        projected = np.empty(jac.shape, dtype=result_dtype)
        eps = _reference_floor_eps(floor)
        safe = (
            _safe_reference(reference, floor=floor)
            if _reference_has_near_zero(reference, eps)
            else reference
        )
        safe_reference = np.asarray(safe, dtype=result_dtype)
        np.divide(jac, safe_reference.reshape(-1, 1), out=projected)
        owns_projection = True
    if resolved_orientation == "reference_minus_target":
        if owns_projection:
            np.negative(projected, out=projected)
        else:
            projected = np.negative(projected, out=np.empty_like(projected))
    return np.asarray(projected, dtype=result_dtype)
