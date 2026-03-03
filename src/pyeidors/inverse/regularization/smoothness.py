"""Regularization operators for inverse EIT solvers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from dolfinx import fem

from .base_regularization import BaseRegularization
from ..jacobian.direct_jacobian import DirectJacobianCalculator


class SmoothnessRegularization(BaseRegularization):
    """Smoothness regularization based on facet-adjacent cell differences."""

    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha

    def create_matrix(self) -> np.ndarray:
        mesh = self.mesh
        tdim = mesh.topology.dim
        fdim = tdim - 1

        mesh.topology.create_connectivity(fdim, tdim)
        facet_to_cell = mesh.topology.connectivity(fdim, tdim)
        facet_map = mesh.topology.index_map(fdim)
        if facet_to_cell is None or facet_map is None:
            return np.eye(self.n_elements)

        n_cells = int(mesh.topology.index_map(tdim).size_local)
        rows = []
        cols = []
        data = []
        row_idx = 0

        for facet in range(int(facet_map.size_local)):
            adjacent_cells = facet_to_cell.links(facet)
            if len(adjacent_cells) == 2:
                cell1, cell2 = int(adjacent_cells[0]), int(adjacent_cells[1])
                rows.extend([row_idx, row_idx])
                cols.extend([cell1, cell2])
                data.extend([1.0, -1.0])
                row_idx += 1

        if row_idx == 0:
            return self.alpha * np.eye(n_cells)

        L = csr_matrix((data, (rows, cols)), shape=(row_idx, n_cells))
        regularization_matrix = self.alpha * (L.T @ L).toarray()
        return regularization_matrix


class TikhonovRegularization(BaseRegularization):
    """Tikhonov regularization."""

    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha

    def create_matrix(self) -> np.ndarray:
        return self.alpha * np.eye(self.n_elements)


class TotalVariationRegularization(BaseRegularization):
    """Approximate Total Variation regularization."""

    def __init__(self, fwd_model, alpha: float = 1.0, epsilon: float = 1e-6):
        super().__init__(fwd_model)
        self.alpha = alpha
        self.epsilon = epsilon

    def create_matrix(self) -> np.ndarray:
        return self.alpha * np.eye(self.n_elements)

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
        self._baseline_diag: Optional[np.ndarray] = None

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

    def create_matrix(self) -> np.ndarray:
        if self._baseline_diag is None:
            self._baseline_diag = self._compute_baseline_diag()
        scaled_diag = self._baseline_diag ** self.exponent
        return self.alpha * np.diag(scaled_diag)
