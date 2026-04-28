"""Regularization operators for inverse EIT solvers."""

from __future__ import annotations

import numpy as np
from dolfinx import fem
from scipy.sparse import csr_matrix, diags

from ..jacobian.direct_jacobian import DirectJacobianCalculator
from ..prior._graph_core import dolfinx_cell_difference_operator
from .base_regularization import BaseRegularization


class SmoothnessRegularization(BaseRegularization):
    """Smoothness regularization based on facet-adjacent cell differences."""

    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha

    def create_matrix(self):
        L = _cell_difference_operator(self.mesh, self.n_elements)
        if L.shape[0] == 0:
            return csr_matrix(self.alpha * np.eye(self.n_elements))
        return (self.alpha * (L.T @ L)).tocsr()


class CurvatureRegularization(SmoothnessRegularization):
    """Named ``L.T @ L`` curvature prior on the current parameter mesh."""


class TikhonovRegularization(BaseRegularization):
    """Tikhonov regularization."""

    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha

    def create_matrix(self) -> np.ndarray:
        return self.alpha * np.eye(self.n_elements)


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
            return csr_matrix(self.alpha * np.eye(self.n_elements))
        reference = self._reference_vector()
        grad_ref = np.asarray(L @ reference, dtype=np.float64).reshape(-1)
        weights = 1.0 / np.sqrt(np.square(grad_ref) + float(self.epsilon) ** 2)
        finite_weights = weights[np.isfinite(weights)]
        median_weight = float(np.median(finite_weights)) if finite_weights.size else 1.0
        if median_weight > 0.0:
            weights = weights / median_weight
        W = diags(weights, offsets=0, format="csr")
        return (self.alpha * (L.T @ W @ L)).tocsr()

    def create_nonlinear_term(self, sigma_current: np.ndarray) -> np.ndarray:
        grad_magnitude = np.abs(np.gradient(sigma_current))
        weights = 1.0 / (grad_magnitude + self.epsilon)
        return self.alpha * np.diag(weights)


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
