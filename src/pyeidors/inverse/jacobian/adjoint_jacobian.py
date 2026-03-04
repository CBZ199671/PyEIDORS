"""EIDORS-style adjoint Jacobian calculator with optional Torch accumulation."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
import ufl
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc

from .base_jacobian import BaseJacobianCalculator


class EidorsStyleAdjointJacobian(BaseJacobianCalculator):
    """Adjoint Jacobian calculator with EIDORS sign convention."""

    def __init__(
        self,
        fwd_model,
        use_torch: bool = False,
        device: Optional[str] = None,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        torch_batch_all: bool = False,
    ):
        super().__init__(fwd_model)
        self.use_torch = use_torch
        self.torch_batch_all = torch_batch_all
        self.torch_dtype = self._resolve_torch_dtype(torch_dtype)
        if device is None:
            if torch.cuda.is_available():
                self.torch_device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.torch_device = torch.device("mps")
            else:
                self.torch_device = torch.device("cpu")
        else:
            self.torch_device = torch.device(device)
        self._setup()

    @staticmethod
    def _resolve_torch_dtype(dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
        if dtype is None:
            return torch.float64
        if isinstance(dtype, torch.dtype):
            return dtype
        dtype_str = str(dtype).lower()
        if dtype_str in {"float32", "fp32", "f32", "torch.float32"}:
            return torch.float32
        if dtype_str in {"float64", "fp64", "f64", "double", "torch.float64"}:
            return torch.float64
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    def _setup(self):
        self.mesh = self.fwd_model.mesh
        self.V = self.fwd_model.V
        self.V_sigma = self.fwd_model.V_sigma
        self.gdim = self.mesh.geometry.dim

        self.Q_DG = fem.functionspace(self.mesh, ("DG", 0, (self.gdim,)))
        self.DG0 = fem.functionspace(self.mesh, ("DG", 0))

        v = ufl.TestFunction(self.DG0)
        cell_vec = fem_petsc.assemble_vector(fem.form(v * ufl.dx))
        cell_vec.assemble()
        self.cell_areas = np.asarray(cell_vec.array, dtype=float)

        if self.use_torch:
            self.cell_areas_t = torch.from_numpy(self.cell_areas).to(self.torch_device, dtype=self.torch_dtype)

    def calculate(self, sigma: fem.Function, **kwargs) -> np.ndarray:
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)

        meas_curr = self._measurement_to_current_patterns()
        adj_fields, _ = self.fwd_model.forward_solve(sigma, meas_curr)
        grad_adj_all = self._compute_field_gradients(adj_fields)

        if self.use_torch:
            J = self._assemble_torch(grad_u_all, grad_adj_all)
        else:
            J = self._assemble_numpy(grad_u_all, grad_adj_all)
        return J

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

    def _measurement_to_current_patterns(self) -> np.ndarray:
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elec = self.fwd_model.n_elec
        current_patterns = np.zeros((n_elec, n_meas), dtype=float)

        meas_idx = 0
        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            n_meas_this = meas_matrix.shape[0]
            current_patterns[:, meas_idx : meas_idx + n_meas_this] = meas_matrix.T
            meas_idx += n_meas_this
        return current_patterns

    def _assemble_numpy(self, grad_u_all: List[np.ndarray], grad_adj_all: List[np.ndarray]) -> np.ndarray:
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elem = len(self.cell_areas)
        J = np.zeros((n_meas, n_elem), dtype=float)

        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]
            for k in range(n_meas_this):
                adj_grad = grad_adj_all[meas_idx + k]
                sensitivity = -np.sum(grad_u * adj_grad, axis=1) * self.cell_areas
                J[meas_idx + k, :] = sensitivity
            meas_idx += n_meas_this
        return J

    def _assemble_torch(self, grad_u_all: List[np.ndarray], grad_adj_all: List[np.ndarray]) -> np.ndarray:
        if self.torch_batch_all:
            return self._assemble_torch_all(grad_u_all, grad_adj_all)

        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elem = len(self.cell_areas)
        J_t = torch.zeros((n_meas, n_elem), device=self.torch_device, dtype=self.torch_dtype)

        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]
            grad_u_t = torch.from_numpy(grad_u).to(self.torch_device, dtype=self.torch_dtype)
            adj_block = np.stack(grad_adj_all[meas_idx : meas_idx + n_meas_this], axis=0)
            adj_block_t = torch.from_numpy(adj_block).to(self.torch_device, dtype=self.torch_dtype)
            sensitivity = -(adj_block_t * grad_u_t.unsqueeze(0)).sum(dim=2) * self.cell_areas_t
            J_t[meas_idx : meas_idx + n_meas_this, :] = sensitivity
            meas_idx += n_meas_this
        return J_t.cpu().numpy()

    def _assemble_torch_all(self, grad_u_all: List[np.ndarray], grad_adj_all: List[np.ndarray]) -> np.ndarray:
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elem = len(self.cell_areas)
        np_dtype = np.float32 if self.torch_dtype == torch.float32 else np.float64

        adj_block = np.stack(grad_adj_all, axis=0).astype(np_dtype, copy=False)
        grad_u_block = np.zeros((n_meas, n_elem, adj_block.shape[2]), dtype=np_dtype)

        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]
            grad_u_block[meas_idx : meas_idx + n_meas_this] = grad_u.astype(np_dtype, copy=False)
            meas_idx += n_meas_this

        grad_u_t = torch.from_numpy(grad_u_block).to(self.torch_device, dtype=self.torch_dtype)
        adj_block_t = torch.from_numpy(adj_block).to(self.torch_device, dtype=self.torch_dtype)
        sensitivity = -(adj_block_t * grad_u_t).sum(dim=2) * self.cell_areas_t
        return sensitivity.cpu().numpy()
