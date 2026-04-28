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
    array = np.asarray(values, dtype=np.float64)
    if array.ndim > 2:
        raise ValueError(
            f"{name} must be a 1D or 2D measurement vector, got {array.ndim}D."
        )
    return array.reshape(-1)


def _safe_reference(
    reference_meas: np.ndarray, *, floor: float | None = None
) -> np.ndarray:
    """Clamp near-zero reference values to ``+/-eps``, preserving sign."""
    safe = np.asarray(reference_meas, dtype=np.float64).copy()
    eps = (
        np.finfo(np.float64).eps
        if floor is None
        else float(max(floor, np.finfo(np.float64).eps))
    )
    small = np.abs(safe) < eps
    # Preserve sign for tiny nonzero values; default to +eps for exact zeros
    signs = np.sign(safe[small])
    signs[signs == 0] = 1.0
    safe[small] = signs * eps
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
        diff = diff / _safe_reference(reference, floor=floor)
    if resolved_orientation == "reference_minus_target":
        diff = -diff
    return np.asarray(diff, dtype=np.float64)


def build_difference_frames(
    targets: np.ndarray,
    references: np.ndarray,
    *,
    mode: str = DEFAULT_DIFFERENCE_MODE,
    orientation: str = DEFAULT_DIFFERENCE_ORIENTATION,
    floor: float | None = None,
) -> np.ndarray:
    """Vectorized 2D batch of :func:`build_difference_vector`.

    Both ``targets`` and ``references`` must already be ``(n_frames, n_meas)``
    float64 arrays. Returns a contiguous float64 ``(n_frames, n_meas)`` array.
    """
    target_batch = np.asarray(targets, dtype=np.float64)
    reference_batch = np.asarray(references, dtype=np.float64)
    if target_batch.ndim != 2 or reference_batch.ndim != 2:
        raise ValueError("targets and references must be 2D frame batches.")
    if target_batch.shape != reference_batch.shape:
        raise ValueError(
            "targets and references must have identical 2D shapes: "
            f"{target_batch.shape!r} vs {reference_batch.shape!r}."
        )
    resolved_mode = normalize_difference_mode(mode)
    resolved_orientation = normalize_difference_orientation(orientation)
    diff = target_batch - reference_batch
    if resolved_mode == "normalized":
        eps = (
            np.finfo(np.float64).eps
            if floor is None
            else float(max(floor, np.finfo(np.float64).eps))
        )
        safe = reference_batch.copy()
        small = np.abs(safe) < eps
        if np.any(small):
            signs = np.sign(safe[small])
            signs[signs == 0] = 1.0
            safe[small] = signs * eps
        diff = diff / safe
    if resolved_orientation == "reference_minus_target":
        diff = -diff
    return np.ascontiguousarray(diff, dtype=np.float64)


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
    jac = np.asarray(jacobian, dtype=np.float64)
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

    projected = jac
    if resolved_mode == "normalized":
        projected = projected / _safe_reference(reference, floor=floor)[:, None]
    if resolved_orientation == "reference_minus_target":
        projected = -projected
    return np.asarray(projected, dtype=np.float64)
