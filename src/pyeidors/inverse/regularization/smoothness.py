"""Smoothness Regularization."""

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from fenics import cells, edges, Function

from .base_regularization import BaseRegularization
from ..jacobian.direct_jacobian import DirectJacobianCalculator


class SmoothnessRegularization(BaseRegularization):
    """Smoothness Regularization - Based on Laplacian operator."""

    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha

    def create_matrix(self) -> np.ndarray:
        """Create smoothness regularization matrix."""
        n_cells = self.mesh.num_cells()

        # Build Laplacian operator
        rows, cols, data = [], [], []
        row_idx = 0

        # Build Laplacian based on mesh topology
        for edge in edges(self.mesh):
            adjacent_cells = []
            for cell in cells(edge):
                adjacent_cells.append(cell.index())

            if len(adjacent_cells) == 2:
                cell1, cell2 = adjacent_cells
                rows.extend([row_idx, row_idx])
                cols.extend([cell1, cell2])
                data.extend([1.0, -1.0])
                row_idx += 1

        # Build difference matrix
        L = csr_matrix((data, (rows, cols)), shape=(row_idx, n_cells))

        # Return L^T * L * alpha
        regularization_matrix = self.alpha * (L.T @ L).toarray()

        return regularization_matrix


class TikhonovRegularization(BaseRegularization):
    """Tikhonov Regularization."""

    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha

    def create_matrix(self) -> np.ndarray:
        """Create Tikhonov regularization matrix (identity matrix)."""
        n_elements = self.n_elements
        return self.alpha * np.eye(n_elements)


class TotalVariationRegularization(BaseRegularization):
    """Total Variation Regularization."""

    def __init__(self, fwd_model, alpha: float = 1.0, epsilon: float = 1e-6):
        super().__init__(fwd_model)
        self.alpha = alpha
        self.epsilon = epsilon

    def create_matrix(self) -> np.ndarray:
        """Create total variation regularization matrix (approximation).

        Total variation regularization is typically nonlinear; this provides
        a linear approximation. May need to be updated during the solve.
        """
        return self.alpha * np.eye(self.n_elements)

    def create_nonlinear_term(self, sigma_current: np.ndarray) -> np.ndarray:
        """Create nonlinear TV term (can be called from solver).

        Implements TV regularization term based on current solution.
        This is a simplified version; actual implementation would be more complex.
        """
        grad_magnitude = np.abs(np.gradient(sigma_current))
        weights = 1.0 / (grad_magnitude + self.epsilon)

        # Build weighted Laplacian matrix
        # Simplified implementation...
        return self.alpha * np.diag(weights)


class NOSERRegularization(BaseRegularization):
    """NOSER Regularization - Diagonal matrix based on J^T J diagonal.

    EIDORS-style implementation: Reg = diag(sum(J.^2, 1)).^exponent

    Args:
        fwd_model: Forward model.
        jacobian_calculator: Jacobian calculator.
        base_conductivity: Conductivity for baseline Jacobian computation.
        alpha: Regularization coefficient.
        exponent: NOSER exponent (EIDORS default: 0.5).
        floor: Minimum value for diagonal elements to avoid numerical issues.
    """

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
        """Initialize NOSER regularization.

        Args:
            floor: Absolute floor value (used when adaptive_floor=False).
            adaptive_floor: If True, floor adapts to J'J magnitude.
            floor_fraction: Adaptive floor = max(diag(J'J)) × floor_fraction.
        """
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
        V_sigma = self.fwd_model.V_sigma
        sigma_fn = Function(V_sigma)
        sigma_fn.vector()[:] = self.base_conductivity

        # DirectJacobianCalculator expects a Function
        jac = self._jacobian_calculator.calculate(sigma_fn)
        # EIDORS: diag_col = sum(J.^2, 1)'  (column vector)
        diag_entries = np.sum(jac * jac, axis=0)

        # Apply floor to avoid numerical issues
        if self.adaptive_floor:
            # Adaptive floor: small fraction of J'J maximum
            adaptive_floor_value = np.max(diag_entries) * self.floor_fraction
            effective_floor = max(adaptive_floor_value, 1e-100)  # Absolute minimum to prevent all-zeros
        else:
            effective_floor = self.floor

        diag_entries = np.maximum(diag_entries, effective_floor)
        return diag_entries

    def create_matrix(self) -> np.ndarray:
        """Create NOSER regularization matrix.

        EIDORS: Reg = spdiags(diag_col.^exponent, 0, n, n)
        """
        if self._baseline_diag is None:
            self._baseline_diag = self._compute_baseline_diag()
        # Apply EIDORS-style exponent
        scaled_diag = self._baseline_diag ** self.exponent
        return self.alpha * np.diag(scaled_diag)
