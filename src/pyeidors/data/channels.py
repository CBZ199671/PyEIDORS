"""Measurement-channel masks and weighting contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.data._temporal_core import as_real_float_array
from pyeidors.utils.numeric_ops import all_finite_values


@dataclass(frozen=True)
class MeasurementContract:
    """Prepared channel mask and square-root measurement weighting.

    Diagonal weights use an internal lightweight array-like matrix. Calling
    ``np.asarray`` on it still returns a dense matrix for compatibility.
    """

    channel_mask: np.ndarray
    weight_transform: Any
    weight_matrix: Any
    weight_kind: str

    @property
    def n_measurements(self) -> int:
        return int(self.channel_mask.size)

    @property
    def bad_channel_count(self) -> int:
        return int(np.count_nonzero(self.channel_mask))


@dataclass(frozen=True)
class _DiagonalMatrix:
    diagonal: np.ndarray

    __array_priority__ = 1000

    def __post_init__(self) -> None:
        values = as_real_float_array(self.diagonal).reshape(-1)
        object.__setattr__(
            self,
            "diagonal",
            np.ascontiguousarray(values),
        )

    @property
    def shape(self) -> tuple[int, int]:
        n = int(self.diagonal.size)
        return (n, n)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> np.dtype:
        return self.diagonal.dtype

    @property
    def T(self) -> "_DiagonalMatrix":
        return self

    def __array__(self, dtype: Any | None = None) -> np.ndarray:
        resolved_dtype = self.diagonal.dtype if dtype is None else np.dtype(dtype)
        size = int(self.diagonal.size)
        dense = np.zeros((size, size), dtype=resolved_dtype)
        if size > 0:
            dense.reshape(-1)[:: size + 1] = np.asarray(
                self.diagonal, dtype=resolved_dtype
            )
        return dense

    def __matmul__(self, other: Any) -> np.ndarray:
        values = np.asarray(as_real_float_array(other), dtype=self.diagonal.dtype)
        if values.ndim == 1:
            if values.shape[0] != self.diagonal.size:
                raise ValueError("matrix dimension mismatch")
            return self.diagonal * values
        if values.ndim == 2:
            if values.shape[0] != self.diagonal.size:
                raise ValueError("matrix dimension mismatch")
            return self.diagonal.reshape(-1, 1) * values
        return np.asarray(self) @ values

    def __rmatmul__(self, other: Any) -> np.ndarray:
        values = np.asarray(as_real_float_array(other), dtype=self.diagonal.dtype)
        if values.ndim == 1:
            if values.shape[0] != self.diagonal.size:
                raise ValueError("matrix dimension mismatch")
            return values * self.diagonal
        if values.ndim == 2:
            if values.shape[1] != self.diagonal.size:
                raise ValueError("matrix dimension mismatch")
            return values * self.diagonal.reshape(1, -1)
        return values @ np.asarray(self)


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
    if indices.size and (
        int(np.min(indices, initial=0)) < 0 or int(np.max(indices, initial=-1)) >= n
    ):
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


def _zero_masked_rows_in_place(values: np.ndarray, mask: np.ndarray) -> None:
    for row_idx, is_bad in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
        if bool(is_bad):
            values[row_idx, ...] = 0.0


def _zero_masked_entries_in_place(values: np.ndarray, mask: np.ndarray) -> None:
    for idx, is_bad in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
        if bool(is_bad):
            values[idx] = 0.0


def _zero_masked_square_rows_cols_in_place(
    values: np.ndarray,
    mask: np.ndarray,
) -> None:
    for idx, is_bad in enumerate(np.asarray(mask, dtype=bool).reshape(-1)):
        if not bool(is_bad):
            continue
        values[idx, :] = 0.0
        values[:, idx] = 0.0


def zero_bad_channel_rows(jacobian: Any, channel_mask: Any | None = None) -> np.ndarray:
    """Return a copy of ``jacobian`` with bad measurement rows zeroed."""

    matrix = _as_2d_array(jacobian, name="jacobian")
    mask = normalize_bad_channel_mask(channel_mask, n_measurements=matrix.shape[0])
    out = matrix.copy()
    if np.any(mask):
        _zero_masked_rows_in_place(out, mask)
    return out


def zero_bad_channel_vector(vector: Any, channel_mask: Any | None = None) -> np.ndarray:
    """Return a copy of ``vector`` with bad measurement entries zeroed."""

    values = _as_vector(vector, name="vector")
    mask = normalize_bad_channel_mask(channel_mask, n_measurements=values.size)
    out = values.copy()
    if np.any(mask):
        _zero_masked_entries_in_place(out, mask)
    return out


def zero_bad_channel_weights(
    measurement_weights: Any | None,
    channel_mask: Any | None = None,
    *,
    n_measurements: int,
    dtype: Any | None = None,
) -> tuple[np.ndarray, str]:
    """Return masked measurement precision weights ``W``.

    One-dimensional inputs are treated as diagonal precision weights. Two-
    dimensional inputs are treated as full precision matrices and have both
    rows and columns for bad channels zeroed.
    """

    n = int(n_measurements)
    resolved_dtype = np.dtype(np.float64 if dtype is None else dtype)
    mask = normalize_bad_channel_mask(channel_mask, n_measurements=n)
    if measurement_weights is None:
        weights = np.ones(n, dtype=resolved_dtype)
        kind = "identity"
    elif sparse.issparse(measurement_weights):
        weights = np.asarray(measurement_weights.toarray(), dtype=resolved_dtype)
        kind = "full"
    else:
        weights = np.asarray(measurement_weights, dtype=resolved_dtype)
        kind = "diagonal" if weights.ndim == 1 else "full"

    if weights.ndim == 1:
        if weights.size != n:
            raise ValueError(
                f"measurement_weights length {weights.size} does not match {n}."
            )
        if not all_finite_values(weights):
            raise FloatingPointError("measurement_weights contain non-finite values.")
        if float(np.min(weights, initial=np.inf)) < 0.0:
            raise ValueError("measurement_weights entries must be non-negative.")
        masked = weights.copy()
        if np.any(mask):
            _zero_masked_entries_in_place(masked, mask)
        return masked, "identity" if measurement_weights is None else kind

    if weights.ndim != 2 or weights.shape != (n, n):
        raise ValueError(
            "measurement_weights must be a length-n diagonal vector or an n-by-n matrix."
        )
    if not all_finite_values(weights):
        raise FloatingPointError("measurement_weights contain non-finite values.")
    if not np.allclose(weights, weights.T, rtol=1e-10, atol=1e-12):
        raise ValueError("measurement_weights matrix must be symmetric.")
    masked = weights.copy()
    if np.any(mask):
        _zero_masked_square_rows_cols_in_place(masked, mask)
    return masked, kind


def prepare_measurement_contract(
    *,
    n_measurements: int,
    channel_mask: Any | None = None,
    measurement_weights: Any | None = None,
    dtype: Any | None = None,
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
        dtype=dtype,
    )
    transform = _sqrt_weight_transform(weights)
    matrix = _DiagonalMatrix(weights) if weights.ndim == 1 else weights
    return MeasurementContract(
        channel_mask=mask,
        weight_transform=transform,
        weight_matrix=matrix,
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
        dtype=matrix.dtype,
    )
    masked = zero_bad_channel_rows(matrix, contract.channel_mask)
    return np.asarray(contract.weight_transform @ masked, dtype=matrix.dtype), contract


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
        dtype=values.dtype,
    )
    masked = zero_bad_channel_vector(values, contract.channel_mask)
    return np.asarray(contract.weight_transform @ masked, dtype=values.dtype), contract


def _as_vector(values: Any, *, name: str) -> np.ndarray:
    array = as_real_float_array(values)
    if array.ndim > 2:
        raise ValueError(f"{name} must be a 1D or column-vector array.")
    array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not all_finite_values(array):
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array)


def _as_2d_array(values: Any, *, name: str) -> np.ndarray:
    if sparse.issparse(values):
        array = as_real_float_array(values.toarray())
    else:
        array = as_real_float_array(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    if 0 in array.shape:
        raise ValueError(f"{name} must be non-empty.")
    if not all_finite_values(array):
        raise FloatingPointError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(array)


def _sqrt_weight_transform(weights: np.ndarray) -> Any:
    if weights.ndim == 1:
        return _DiagonalMatrix(np.sqrt(weights))
    eigenvalues, eigenvectors = np.linalg.eigh(weights)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("measurement_weights matrix must be positive semidefinite.")
    clipped = np.maximum(eigenvalues, 0.0)
    sqrt_values = np.sqrt(clipped)
    return sqrt_values.reshape(-1, 1) * eigenvectors.T


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
