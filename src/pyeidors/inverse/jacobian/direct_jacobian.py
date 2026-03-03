"""Direct Jacobian calculator using DOLFINx function spaces."""

from __future__ import annotations

import numpy as np
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc

from .base_jacobian import BaseJacobianCalculator


class DirectJacobianCalculator(BaseJacobianCalculator):
    """Direct method Jacobian calculator."""

    def __init__(self, fwd_model):
        super().__init__(fwd_model)
        self._setup_computation()

    def _setup_computation(self):
        self.mesh = self.fwd_model.mesh
        self.V = self.fwd_model.V
        self.V_sigma = self.fwd_model.V_sigma
        self.gdim = self.mesh.geometry.dim

        self.Q_DG = fem.functionspace(self.mesh, ("DG", 0, (self.gdim,)))
        self.DG0 = fem.functionspace(self.mesh, ("DG", 0))

        v = ufl.TestFunction(self.DG0)
        areas_vec = fem_petsc.assemble_vector(fem.form(v * ufl.dx))
        areas_vec.assemble()
        self.cell_areas = np.asarray(areas_vec.array, dtype=float)

    def calculate(self, sigma: fem.Function, method: str = "efficient", **kwargs) -> np.ndarray:
        if method == "efficient":
            return self._calculate_efficient(sigma)
        if method == "traditional":
            return self._calculate_traditional(sigma)
        raise ValueError(f"Unknown method: {method}")

    def _calculate_efficient(self, sigma: fem.Function) -> np.ndarray:
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)

        adjoint_fields = self._compute_adjoint_fields_efficient(sigma)
        jacobian = self._assemble_jacobian_efficient(grad_u_all, adjoint_fields)

        scale = float(getattr(self.fwd_model.pattern_manager.config, "amplitude", 1.0))
        return jacobian * scale

    def _calculate_traditional(self, sigma: fem.Function) -> np.ndarray:
        u_all, _ = self.fwd_model.forward_solve(sigma)

        I2_all = np.eye(self.fwd_model.n_elec)
        bu_all, _ = self.fwd_model.forward_solve(sigma, I2_all)

        grad_u_all = self._compute_field_gradients(u_all)
        grad_bu_all = self._compute_field_gradients(bu_all)

        jacobian = self._assemble_jacobian_traditional(grad_u_all, grad_bu_all)
        return self._convert_to_measurement_jacobian(jacobian)

    def _compute_field_gradients(self, field_solutions):
        gradients = []
        interpolation_points = self.Q_DG.element.interpolation_points
        if callable(interpolation_points):
            interpolation_points = interpolation_points()
        for field in field_solutions:
            u_fun = fem.Function(self.V)
            u_fun.x.array[:] = field

            grad_expr = fem.Expression(ufl.grad(u_fun), interpolation_points)
            grad_u = fem.Function(self.Q_DG)
            grad_u.interpolate(grad_expr)
            grad_u_vec = grad_u.x.array.reshape(-1, self.gdim)
            gradients.append(grad_u_vec)

        return gradients

    def _compute_adjoint_fields_efficient(self, sigma: fem.Function):
        adjoint_patterns = self._measurement_to_current_patterns()
        adjoint_fields, _ = self.fwd_model.forward_solve(sigma, adjoint_patterns)
        return self._compute_field_gradients(adjoint_fields)

    def _measurement_to_current_patterns(self):
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elec = self.fwd_model.n_elec

        current_patterns = np.zeros((n_elec, n_meas), dtype=float)

        meas_idx = 0
        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            n_meas_this_stim = meas_matrix.shape[0]

            current_patterns[:, meas_idx : meas_idx + n_meas_this_stim] = meas_matrix.T
            meas_idx += n_meas_this_stim

        return current_patterns

    def _assemble_jacobian_efficient(self, grad_u_all, adjoint_gradients):
        n_measurements = len(adjoint_gradients)
        n_elements = len(self.cell_areas)

        jacobian = np.zeros((n_measurements, n_elements), dtype=float)

        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this_stim = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]

            for local_meas_idx in range(n_meas_this_stim):
                global_meas_idx = meas_idx + local_meas_idx
                adjoint_grad = adjoint_gradients[global_meas_idx]

                sensitivity = np.sum(grad_u * adjoint_grad, axis=1) * self.cell_areas
                jacobian[global_meas_idx, :] = sensitivity

            meas_idx += n_meas_this_stim

        return jacobian

    def _assemble_jacobian_traditional(self, grad_u_all, grad_bu_all):
        jacobian_blocks = []

        for grad_u in grad_u_all:
            derivatives = []
            for grad_bu in grad_bu_all:
                sensitivity = np.sum(grad_bu * grad_u, axis=1) * self.cell_areas
                derivatives.append(sensitivity)

            jacobian_blocks.append(np.array(derivatives))

        return np.vstack(jacobian_blocks)

    def _convert_to_measurement_jacobian(self, electrode_jacobian):
        measurement_jacobian_blocks = []

        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            elec_start = stim_idx * self.fwd_model.n_elec
            elec_end = (stim_idx + 1) * self.fwd_model.n_elec
            electrode_jac_for_stim = electrode_jacobian[elec_start:elec_end, :]

            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            meas_jacobian_for_stim = meas_matrix @ electrode_jac_for_stim

            measurement_jacobian_blocks.append(meas_jacobian_for_stim)

        return np.vstack(measurement_jacobian_blocks)
