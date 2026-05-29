"""Channel helpers for real and complex EIT result views."""

from __future__ import annotations

from typing import Any

import numpy as np


REAL_CHANNEL = "real"
IMAG_CHANNEL = "imag"
MAGNITUDE_CHANNEL = "magnitude"
PHASE_CHANNEL = "phase"
COMPOSITE_CHANNEL = "composite"

DISPLAY_CHANNELS = (
    REAL_CHANNEL,
    IMAG_CHANNEL,
    MAGNITUDE_CHANNEL,
    PHASE_CHANNEL,
    COMPOSITE_CHANNEL,
)

_COMPLEX_SCAN_CHUNK_ITEMS = 1_048_576


def _floating_dtype_for(values: np.ndarray) -> np.dtype:
    if np.issubdtype(values.dtype, np.floating):
        return np.dtype(values.dtype)
    if values.dtype == np.dtype("complex64"):
        return np.dtype("float32")
    if np.issubdtype(values.dtype, np.complexfloating):
        return np.dtype("float64")
    return np.dtype("float32")


def _display_float_array(values: Any) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.floating):
        return arr
    return np.asarray(arr, dtype=np.float32)


def _has_significant_imaginary(values: np.ndarray, *, tol: float) -> bool:
    imag = np.ravel(np.imag(values), order="K")
    if imag.size == 0:
        return False
    chunk_items = min(imag.size, max(1, int(_COMPLEX_SCAN_CHUNK_ITEMS)))
    finite = np.empty(chunk_items, dtype=bool)
    abs_work = np.empty(chunk_items, dtype=imag.dtype)
    for start in range(0, imag.size, chunk_items):
        stop = min(start + chunk_items, imag.size)
        chunk = imag[start:stop]
        finite_chunk = finite[: stop - start]
        abs_chunk = abs_work[: stop - start]
        np.isfinite(chunk, out=finite_chunk)
        np.abs(chunk, out=abs_chunk)
        max_abs = float(np.max(abs_chunk, where=finite_chunk, initial=0.0))
        if max_abs > float(tol):
            return True
    return False


def _composite_channel_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size == 0:
        return np.asarray(np.real(arr), dtype=_floating_dtype_for(arr))
    flat = np.ravel(arr, order="C")
    out = np.empty(arr.shape, dtype=_floating_dtype_for(arr), order="C")
    out_flat = out.reshape(-1)
    chunk_items = min(flat.size, max(1, int(_COMPLEX_SCAN_CHUNK_ITEMS)))
    work = np.empty(chunk_items, dtype=out.dtype)
    for start in range(0, flat.size, work.size):
        stop = min(start + work.size, flat.size)
        source_chunk = flat[start:stop]
        real_chunk = np.real(source_chunk)
        imag_chunk = np.imag(source_chunk)
        out_chunk = out_flat[start:stop]
        np.arctan2(imag_chunk, real_chunk, out=out_chunk)
        out_chunk /= np.pi
        magnitude = work[: stop - start]
        np.hypot(real_chunk, imag_chunk, out=magnitude)
        out_chunk *= magnitude
    return out


def has_complex_component(*values: Any, tol: float = 1.0e-12) -> bool:
    """Return true when any payload carries a meaningful imaginary part."""

    for value in values:
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.size == 0 or not np.iscomplexobj(arr):
            continue
        if _has_significant_imaginary(arr, tol=float(tol)):
            return True
    return False


def channel_values(values: Any, channel: str) -> np.ndarray:
    """Project a real/complex array to the scalar channel shown in the GUI."""

    arr = np.asarray(values)
    selected = str(channel or REAL_CHANNEL).strip().lower()
    if selected == IMAG_CHANNEL:
        return _display_float_array(np.imag(arr))
    if selected == MAGNITUDE_CHANNEL:
        return _display_float_array(np.abs(arr))
    if selected == PHASE_CHANNEL:
        return _display_float_array(np.angle(arr))
    if selected == COMPOSITE_CHANNEL:
        # Scalar fallback for non-RGB renderers: phase-weighted magnitude.
        # It is not a replacement for separate Re/Im/|.|/phase views, but it
        # gives one compact map where both amplitude and phase can move pixels.
        return _composite_channel_values(arr)
    return _display_float_array(np.real(arr))


def channel_is_complex_only(channel: str) -> bool:
    return str(channel or "").strip().lower() in {
        IMAG_CHANNEL,
        MAGNITUDE_CHANNEL,
        PHASE_CHANNEL,
        COMPOSITE_CHANNEL,
    }
