"""Direct Method Jacobian Calculator - Using existing stimulation patterns."""

import numpy as np
from fenics import *

from .base_jacobian import BaseJacobianCalculator


class DirectJacobianCalculator(BaseJacobianCalculator):
    """Direct Method Jacobian Calculator - Optimized version."""

    def __init__(self, fwd_model):
        super().__init__(fwd_model)
        self._setup_computation()

    def _setup_computation(self):
        """Set up function spaces and test functions for computation."""
        self.mesh = self.fwd_model.mesh
        self.V = self.fwd_model.V
        self.V_sigma = self.fwd_model.V_sigma

        # Create gradient function space
        self.Q_DG = VectorFunctionSpace(self.mesh, "DG", 0)
        self.DG0 = FunctionSpace(self.mesh, "DG", 0)

        # Compute cell areas
        v = TestFunction(self.DG0)
        self.cell_areas = assemble(v * dx).get_local()

    def calculate(self, sigma: Function, method: str = 'efficient', **kwargs) -> np.ndarray:
        """Calculate Jacobian matrix.

        Args:
            sigma: Conductivity distribution.
            method: Computation method ('efficient' or 'traditional').
        """
        if method == 'efficient':
            return self._calculate_efficient(sigma)
        elif method == 'traditional':
            return self._calculate_traditional(sigma)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _calculate_efficient(self, sigma: Function) -> np.ndarray:
        """Efficient Jacobian computation - directly using stimulation patterns.

        Avoids solving separately for each electrode.
        """
        # 1. Forward solve - using existing stimulation patterns
        u_all, U_all = self.fwd_model.forward_solve(sigma)

        # 2. Compute forward field gradients
        grad_u_all = self._compute_field_gradients(u_all)

        # 3. Compute adjoint fields for measurement patterns
        # Use transpose of measurement patterns as stimulation for adjoint fields
        adjoint_fields = self._compute_adjoint_fields_efficient(sigma)

        # 4. Compute Jacobian matrix
        jacobian = self._assemble_jacobian_efficient(grad_u_all, adjoint_fields)

        scale = float(getattr(self.fwd_model.pattern_manager.config, "amplitude", 1.0))
        return jacobian * scale

    def _calculate_traditional(self, sigma: Function) -> np.ndarray:
        """Traditional Jacobian computation - compatible with original code."""
        # Forward solve
        u_all, _ = self.fwd_model.forward_solve(sigma)

        # Construct unit current patterns (for adjoint field computation)
        I2_all = np.eye(self.fwd_model.n_elec)
        bu_all, _ = self.fwd_model.forward_solve(sigma, I2_all)

        # Compute gradients
        grad_u_all = self._compute_field_gradients(u_all)
        grad_bu_all = self._compute_field_gradients(bu_all)

        # Assemble Jacobian matrix
        jacobian = self._assemble_jacobian_traditional(grad_u_all, grad_bu_all)

        # Convert to measurement Jacobian
        measurement_jacobian = self._convert_to_measurement_jacobian(jacobian)

        return measurement_jacobian

    def _compute_field_gradients(self, field_solutions):
        """Compute field gradients."""
        gradients = []
        for field in field_solutions:
            u_fun = Function(self.V)
            u_fun.vector()[:] = field

            grad_u = project(grad(u_fun), self.Q_DG)
            grad_u_vec = grad_u.vector().get_local().reshape(-1, 2)
            gradients.append(grad_u_vec)

        return gradients

    def _compute_adjoint_fields_efficient(self, sigma: Function):
        """Efficiently compute adjoint fields - using measurement patterns."""
        # Convert measurement patterns to current stimulation patterns
        adjoint_patterns = self._measurement_to_current_patterns()

        # Solve adjoint fields
        adjoint_fields, _ = self.fwd_model.forward_solve(sigma, adjoint_patterns)

        # Compute gradients
        adjoint_gradients = self._compute_field_gradients(adjoint_fields)

        return adjoint_gradients

    def _measurement_to_current_patterns(self):
        """Convert measurement patterns to current stimulation patterns."""
        # Construct current patterns from measurement patterns
        # Simplified implementation - can be further optimized
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elec = self.fwd_model.n_elec

        current_patterns = np.zeros((n_elec, n_meas))

        meas_idx = 0
        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            n_meas_this_stim = meas_matrix.shape[0]

            # Transpose measurement matrix as current pattern
            current_patterns[:, meas_idx:meas_idx + n_meas_this_stim] = meas_matrix.T
            meas_idx += n_meas_this_stim

        return current_patterns

    def _assemble_jacobian_efficient(self, grad_u_all, adjoint_gradients):
        """Efficiently assemble Jacobian matrix."""
        n_measurements = len(adjoint_gradients)
        n_elements = len(self.cell_areas)

        jacobian = np.zeros((n_measurements, n_elements))

        # Compute Jacobian based on stimulation-measurement correspondence
        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this_stim = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]

            for local_meas_idx in range(n_meas_this_stim):
                global_meas_idx = meas_idx + local_meas_idx
                adjoint_grad = adjoint_gradients[global_meas_idx]

                # Compute sensitivity
                sensitivity = np.sum(grad_u * adjoint_grad, axis=1) * self.cell_areas
                jacobian[global_meas_idx, :] = sensitivity

            meas_idx += n_meas_this_stim

        return jacobian

    def _assemble_jacobian_traditional(self, grad_u_all, grad_bu_all):
        """Traditional way to assemble Jacobian matrix."""
        jacobian_blocks = []

        for h, grad_u in enumerate(grad_u_all):
            derivatives = []
            for j, grad_bu in enumerate(grad_bu_all):
                sensitivity = np.sum(grad_bu * grad_u, axis=1) * self.cell_areas
                derivatives.append(sensitivity)

            jacobian_block = np.array(derivatives)
            jacobian_blocks.append(jacobian_block)

        return np.vstack(jacobian_blocks)

    def _convert_to_measurement_jacobian(self, electrode_jacobian):
        """Convert electrode Jacobian to measurement Jacobian."""
        measurement_jacobian_blocks = []

        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            elec_start = stim_idx * self.fwd_model.n_elec
            elec_end = (stim_idx + 1) * self.fwd_model.n_elec
            electrode_jac_for_stim = electrode_jacobian[elec_start:elec_end, :]

            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            meas_jacobian_for_stim = meas_matrix @ electrode_jac_for_stim

            measurement_jacobian_blocks.append(meas_jacobian_for_stim)

        return np.vstack(measurement_jacobian_blocks)
