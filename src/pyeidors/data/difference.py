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


def normalize_difference_mode(mode: str | None, *, default: str = DEFAULT_DIFFERENCE_MODE) -> str:
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
        raise ValueError(f"{name} must be a 1D or 2D measurement vector, got {array.ndim}D.")
    return array.reshape(-1)


def _safe_reference(reference_meas: np.ndarray, *, floor: float | None = None) -> np.ndarray:
    """Clamp near-zero reference values to ``+/-eps``, preserving sign."""
    safe = np.asarray(reference_meas, dtype=np.float64).copy()
    eps = np.finfo(np.float64).eps if floor is None else float(max(floor, np.finfo(np.float64).eps))
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
