"""Measurement-channel masks and weighting contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MeasurementContract:
    """Prepared channel mask and square-root measurement weighting."""

    channel_mask: np.ndarray
    weight_transform: np.ndarray
    weight_matrix: np.ndarray
    weight_kind: str

    @property
    def n_measurements(self) -> int:
        return int(self.channel_mask.size)

    @property
    def bad_channel_count(self) -> int:
        return int(np.count_nonzero(self.channel_mask))


def bad_channel_mask(
    n_measurements: int,
    bad_channels: Any | None = None,
) -> np.ndarray:
    """Return a boolean mask where ``True`` marks a bad measurement channel."""

    n = int(n_measurements)
    if n <= 0:
        raise ValueError("n_measurements must be positive.")
    mask = np.zeros(n, dtype=bool)
    if bad_channels is None:
        return mask
    channels = np.asarray(bad_channels)
    if channels.dtype == bool:
        return normalize_bad_channel_mask(channels, n_measurements=n)
    indices = channels.astype(np.int64, copy=False).reshape(-1)
    if np.any((indices < 0) | (indices >= n)):
        raise ValueError("bad channel indices are out of range.")
    mask[indices] = True
    return mask


def normalize_bad_channel_mask(
    channel_mask: Any | None,
    *,
    n_measurements: int,
) -> np.ndarray:
    """Normalize a bad-channel mask.

    ``True`` means the channel is bad and must be zeroed. Integer inputs are
    interpreted as bad-channel indices for convenience.
    """

    n = int(n_measurements)
    if n <= 0:
        raise ValueError("n_measurements must be positive.")
    if channel_mask is None:
        return np.zeros(n, dtype=bool)
    mask = np.asarray(channel_mask)
    if mask.dtype == bool:
        mask = mask.reshape(-1)
        if mask.size != n:
            raise ValueError(f"channel_mask length {mask.size} does not match {n}.")
        return mask.astype(bool, copy=True)
    return bad_channel_mask(n, mask)


def zero_bad_channel_rows(jacobian: Any, channel_mask: Any | None = None) -> np.ndarray:
    """Return a copy of ``jacobian`` with bad measurement rows zeroed."""

    matrix = _as_2d_array(jacobian, name="jacobian")
    mask = normalize_bad_channel_mask(channel_mask, n_measurements=matrix.shape[0])
    out = matrix.copy()
    if np.any(mask):
        out[mask, :] = 0.0
    return out


def zero_bad_channel_vector(vector: Any, channel_mask: Any | None = None) -> np.ndarray:
    """Return a copy of ``vector`` with bad measurement entries zeroed."""

    values = _as_vector(vector, name="vector")
    mask = normalize_bad_channel_mask(channel_mask, n_measurements=values.size)
    out = values.copy()
    if np.any(mask):
        out[mask] = 0.0
    return out


def zero_bad_channel_weights(
    measurement_weights: Any | None,
    channel_mask: Any | None = None,
    *,
    n_measurements: int,
) -> tuple[np.ndarray, str]:
    """Return masked measurement precision weights ``W``.

    One-dimensional inputs are treated as diagonal precision weights. Two-
    dimensional inputs are treated as full precision matrices and have both
    rows and columns for bad channels zeroed.
    """

    n = int(n_measurements)
    mask = normalize_bad_channel_mask(channel_mask, n_measurements=n)
    if measurement_weights is None:
        weights = np.ones(n, dtype=np.float64)
        kind = "identity"
    elif sparse.issparse(measurement_weights):
        weights = np.asarray(measurement_weights.toarray(), dtype=np.float64)
        kind = "full"
    else:
        weights = np.asarray(measurement_weights, dtype=np.float64)
        kind = "diagonal" if weights.ndim == 1 else "full"

    if weights.ndim == 1:
        if weights.size != n:
            raise ValueError(
                f"measurement_weights length {weights.size} does not match {n}."
            )
        if not np.isfinite(weights).all():
            raise FloatingPointError("measurement_weights contain non-finite values.")
        if np.any(weights < 0.0):
            raise ValueError("measurement_weights entries must be non-negative.")
        masked = weights.copy()
        if np.any(mask):
            masked[mask] = 0.0
        return masked, "identity" if measurement_weights is None else kind

    if weights.ndim != 2 or weights.shape != (n, n):
        raise ValueError(
            "measurement_weights must be a length-n diagonal vector or an n-by-n matrix."
        )
    if not np.isfinite(weights).all():
        raise FloatingPointError("measurement_weights contain non-finite values.")
    if not np.allclose(weights, weights.T, rtol=1e-10, atol=1e-12):
        raise ValueError("measurement_weights matrix must be symmetric.")
    masked = weights.copy()
    if np.any(mask):
        masked[mask, :] = 0.0
        masked[:, mask] = 0.0
    return masked, kind


def prepare_measurement_contract(
    *,
    n_measurements: int,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
) -> MeasurementContract:
    """Prepare a reusable measurement mask/weight transform.

    The returned transform ``S`` satisfies ``S.T @ S == W`` for the masked
    precision matrix. Offline RM build and online RM application should both
    use this same contract.
    """

    mask = normalize_bad_channel_mask(channel_mask, n_measurements=n_measurements)
    weights, kind = zero_bad_channel_weights(
        measurement_weights,
        mask,
        n_measurements=n_measurements,
    )
    transform = _sqrt_weight_transform(weights)
    matrix = np.diag(weights) if weights.ndim == 1 else weights
    return MeasurementContract(
        channel_mask=mask,
        weight_transform=transform,
        weight_matrix=np.asarray(matrix, dtype=np.float64),
        weight_kind=kind,
    )


def apply_measurement_contract_to_jacobian(
    jacobian: Any,
    *,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
) -> tuple[np.ndarray, MeasurementContract]:
    """Zero bad rows and apply ``sqrt(W)`` to a Jacobian."""

    matrix = _as_2d_array(jacobian, name="jacobian")
    contract = prepare_measurement_contract(
        n_measurements=matrix.shape[0],
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    masked = zero_bad_channel_rows(matrix, contract.channel_mask)
    return np.asarray(contract.weight_transform @ masked, dtype=np.float64), contract


def apply_measurement_contract_to_vector(
    vector: Any,
    *,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
) -> tuple[np.ndarray, MeasurementContract]:
    """Zero bad entries and apply ``sqrt(W)`` to a residual vector."""

    values = _as_vector(vector, name="vector")
    contract = prepare_measurement_contract(
        n_measurements=values.size,
        channel_mask=channel_mask,
        measurement_weights=measurement_weights,
    )
    masked = zero_bad_channel_vector(values, contract.channel_mask)
    return np.asarray(contract.weight_transform @ masked, dtype=np.float64), contract


def _as_vector(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim > 2:
        raise ValueError(f"{name} must be a 1D or column-vector array.")
    array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _as_2d_array(values: Any, *, name: str) -> np.ndarray:
    if sparse.issparse(values):
        array = np.asarray(values.toarray(), dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if 0 in array.shape:
        raise ValueError(f"{name} must be non-empty.")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array, dtype=np.float64)


def _sqrt_weight_transform(weights: np.ndarray) -> np.ndarray:
    if weights.ndim == 1:
        return np.diag(np.sqrt(weights))
    eigenvalues, eigenvectors = np.linalg.eigh(weights)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("measurement_weights matrix must be positive semidefinite.")
    clipped = np.maximum(eigenvalues, 0.0)
    return np.diag(np.sqrt(clipped)) @ eigenvectors.T


__all__ = [
    "MeasurementContract",
    "apply_measurement_contract_to_jacobian",
    "apply_measurement_contract_to_vector",
    "bad_channel_mask",
    "normalize_bad_channel_mask",
    "prepare_measurement_contract",
    "zero_bad_channel_rows",
    "zero_bad_channel_vector",
    "zero_bad_channel_weights",
]
