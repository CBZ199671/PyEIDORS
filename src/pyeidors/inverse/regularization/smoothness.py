"""Regularization operators for inverse EIT solvers."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from scipy.sparse import csr_matrix, diags

from ..jacobian.direct_jacobian import DirectJacobianCalculator
from ..prior._graph_core import dolfinx_cell_difference_operator
from ...utils.numeric_ops import all_finite_values
from .base_regularization import BaseRegularization


def _finite_median_or_default(
    values: np.ndarray,
    default: float = 1.0,
    *,
    chunk_size: int = 65536,
) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float(default)
    block_size = max(1, min(int(chunk_size), int(arr.size)))
    mask_work = np.empty(block_size, dtype=bool)
    finite_count = 0
    for start in range(0, int(arr.size), block_size):
        stop = min(start + block_size, int(arr.size))
        chunk = arr[start:stop]
        mask_chunk = mask_work[: chunk.size]
        np.isfinite(chunk, out=mask_chunk)
        finite_count += int(np.count_nonzero(mask_chunk))
    if finite_count == 0:
        return float(default)
    finite_values = np.empty(finite_count, dtype=np.float64)
    offset = 0
    for start in range(0, int(arr.size), block_size):
        stop = min(start + block_size, int(arr.size))
        chunk = arr[start:stop]
        mask_chunk = mask_work[: chunk.size]
        np.isfinite(chunk, out=mask_chunk)
        n_finite = int(np.count_nonzero(mask_chunk))
        if n_finite:
            np.compress(
                mask_chunk,
                chunk,
                out=finite_values[offset : offset + n_finite],
            )
            offset += n_finite
    return float(np.median(finite_values))


def _scaled_identity_csr(n_elements: int, alpha: float) -> csr_matrix:
    diagonal = np.full(int(n_elements), float(alpha), dtype=np.float64)
    return diags(diagonal, offsets=0, format="csr")


def _dense_scaled_diagonal(values: np.ndarray, scale: float) -> np.ndarray:
    diagonal = np.asarray(values, dtype=np.float64).reshape(-1)
    size = int(diagonal.size)
    matrix = np.zeros((size, size), dtype=np.float64)
    if size > 0 and float(scale) != 0.0:
        matrix_diagonal = matrix.reshape(-1)[:: size + 1]
        np.multiply(diagonal, float(scale), out=matrix_diagonal)
    return matrix


class SmoothnessRegularization(BaseRegularization):
    """Smoothness regularization based on facet-adjacent cell differences."""

    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha

    def create_matrix(self):
        L = _cell_difference_operator(self.mesh, self.n_elements)
        if L.shape[0] == 0:
            return _scaled_identity_csr(self.n_elements, self.alpha)
        return (self.alpha * (L.T @ L)).tocsr()


class CurvatureRegularization(SmoothnessRegularization):
    """Named ``L.T @ L`` curvature prior on the current parameter mesh."""


class TikhonovRegularization(BaseRegularization):
    """Tikhonov regularization."""

    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha

    def create_matrix(self) -> csr_matrix:
        return _scaled_identity_csr(self.n_elements, self.alpha)


class TotalVariationRegularization(BaseRegularization):
    """Frozen linearized Total Variation prior around a background conductivity."""

    def __init__(
        self,
        fwd_model,
        alpha: float = 1.0,
        epsilon: float = 1e-6,
        reference_conductivity: float | np.ndarray = 1.0,
    ):
        super().__init__(fwd_model)
        self.alpha = alpha
        self.epsilon = epsilon
        self.reference_conductivity = reference_conductivity

    def _reference_vector(self) -> np.ndarray:
        if np.isscalar(self.reference_conductivity):
            return np.full(
                self.n_elements, float(self.reference_conductivity), dtype=np.float64
            )
        reference = np.asarray(self.reference_conductivity, dtype=np.float64).reshape(
            -1
        )
        if reference.shape[0] != self.n_elements:
            raise ValueError(
                "reference_conductivity must match the number of elements: "
                f"{reference.shape[0]} vs {self.n_elements}."
            )
        return reference

    def create_matrix(self):
        L = _cell_difference_operator(self.mesh, self.n_elements)
        if L.shape[0] == 0:
            return _scaled_identity_csr(self.n_elements, self.alpha)
        reference = self._reference_vector()
        weights = np.asarray(L @ reference, dtype=np.float64).reshape(-1)
        np.square(weights, out=weights)
        weights += float(self.epsilon) ** 2
        np.sqrt(weights, out=weights)
        np.reciprocal(weights, out=weights)
        if all_finite_values(weights):
            median_weight = float(np.median(weights)) if weights.size else 1.0
        else:
            median_weight = _finite_median_or_default(weights)
        if median_weight > 0.0:
            weights /= median_weight
        W = diags(weights, offsets=0, format="csr")
        return (self.alpha * (L.T @ W @ L)).tocsr()

    def create_nonlinear_term(self, sigma_current: np.ndarray) -> np.ndarray:
        grad_magnitude = np.abs(np.gradient(sigma_current))
        weights = 1.0 / (grad_magnitude + self.epsilon)
        return _dense_scaled_diagonal(weights, self.alpha)


class NOSERRegularization(BaseRegularization):
    """NOSER regularization matrix from baseline Jacobian diagonal."""

    def __init__(
        self,
        fwd_model,
        jacobian_calculator: DirectJacobianCalculator,
        base_conductivity: float = 1.0,
        alpha: float = 1.0,
        exponent: float = 0.5,
        floor: float = 1e-12,
        adaptive_floor: bool = True,
        floor_fraction: float = 1e-6,
    ):
        super().__init__(fwd_model)
        self.alpha = alpha
        self.base_conductivity = base_conductivity
        self.exponent = exponent
        self.floor = floor
        self.adaptive_floor = adaptive_floor
        self.floor_fraction = floor_fraction
        self._jacobian_calculator = jacobian_calculator
        self._baseline_diag: np.ndarray | None = None

    def _compute_baseline_diag(self) -> np.ndarray:
        sigma_fn = fem.Function(self.fwd_model.V_sigma)
        sigma_fn.x.array[:] = self.base_conductivity

        jac = self._jacobian_calculator.calculate(sigma_fn)
        diag_entries = np.sum(jac * jac, axis=0)

        if self.adaptive_floor:
            adaptive_floor_value = np.max(diag_entries) * self.floor_fraction
            effective_floor = max(adaptive_floor_value, 1e-100)
        else:
            effective_floor = self.floor

        return np.maximum(diag_entries, effective_floor)

    def create_matrix(self):
        if self._baseline_diag is None:
            self._baseline_diag = self._compute_baseline_diag()
        scaled_diag = self._baseline_diag**self.exponent
        return diags(self.alpha * scaled_diag, offsets=0, format="csr")


def _cell_difference_operator(mesh, n_elements: int) -> csr_matrix:
    """Build the cell-adjacency difference operator used by smoothness/TV priors."""
    return dolfinx_cell_difference_operator(mesh, n_elements)
