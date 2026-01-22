"""EIDORS-style adjoint method Jacobian calculator with sign convention, optional Torch acceleration."""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, List, Union
from fenics import Function

from .base_jacobian import BaseJacobianCalculator


class EidorsStyleAdjointJacobian(BaseJacobianCalculator):
    """EIDORS-style adjoint method Jacobian calculator.

    Features:
      - Uses measurement pattern transpose as adjoint current, single factorization, batch solve;
      - Sign convention consistent with EIDORS (dV/dσ is negative, so Jacobian has built-in negative sign);
      - Optional torch acceleration for sensitivity accumulation (vectorized accumulation on CPU/GPU).
    """

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
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # Cell area/volume
        v = self.fwd_model.DG0_Test if hasattr(self.fwd_model, "DG0_Test") else None
        if v is None:
            from fenics import TestFunction, FunctionSpace, assemble, dx
            DG0 = FunctionSpace(self.mesh, "DG", 0)
            v = TestFunction(DG0)
            self.cell_areas = assemble(v * dx).get_local()
        else:
            self.cell_areas = self.fwd_model.cell_areas
        # For Torch convenience
        if self.use_torch:
            self.cell_areas_t = torch.from_numpy(self.cell_areas).to(self.torch_device, dtype=self.torch_dtype)

    def calculate(self, sigma: Function, **kwargs) -> np.ndarray:
        """Calculate measurement Jacobian matrix (shape: n_meas x n_elem)."""
        # 1) Forward solve
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)

        # 2) Adjoint field (measurement pattern transpose as current)
        meas_curr = self._measurement_to_current_patterns()
        adj_fields, _ = self.fwd_model.forward_solve(sigma, meas_curr)
        grad_adj_all = self._compute_field_gradients(adj_fields)

        # 3) Assemble J, sign is negative (dV/dσ < 0)
        if self.use_torch:
            J = self._assemble_torch(grad_u_all, grad_adj_all)
        else:
            J = self._assemble_numpy(grad_u_all, grad_adj_all)

        # Stimulation amplitude scaling
        try:
            amp = float(getattr(self.fwd_model.pattern_manager.config, "amplitude", 1.0))
        except Exception:
            amp = 1.0
        return J * amp

    def _compute_field_gradients(self, field_solutions):
        """Convert nodal solutions to cell gradients."""
        gradients = []
        for field in field_solutions:
            u_fun = Function(self.V)
            u_fun.vector()[:] = field
            from fenics import project, grad, VectorFunctionSpace
            Q_DG = VectorFunctionSpace(self.mesh, "DG", 0)
            grad_u = project(grad(u_fun), Q_DG)
            grad_u_vec = grad_u.vector().get_local().reshape(-1, self.mesh.geometry().dim())
            gradients.append(grad_u_vec)
        return gradients

    def _measurement_to_current_patterns(self) -> np.ndarray:
        """Transpose measurement matrix to adjoint current patterns (EIDORS adjoint method)."""
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elec = self.fwd_model.n_elec
        current_patterns = np.zeros((n_elec, n_meas))

        meas_idx = 0
        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            n_meas_this = meas_matrix.shape[0]
            current_patterns[:, meas_idx: meas_idx + n_meas_this] = meas_matrix.T
            meas_idx += n_meas_this
        return current_patterns

    def _assemble_numpy(self, grad_u_all: List[np.ndarray], grad_adj_all: List[np.ndarray]) -> np.ndarray:
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elem = len(self.cell_areas)
        J = np.zeros((n_meas, n_elem))

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
            adj_block = np.stack(grad_adj_all[meas_idx: meas_idx + n_meas_this], axis=0)
            adj_block_t = torch.from_numpy(adj_block).to(self.torch_device, dtype=self.torch_dtype)
            # Batch assemble per stimulation to reduce host/device transfers.
            sensitivity = -(adj_block_t * grad_u_t.unsqueeze(0)).sum(dim=2) * self.cell_areas_t
            J_t[meas_idx: meas_idx + n_meas_this, :] = sensitivity
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
            grad_u_block[meas_idx: meas_idx + n_meas_this] = grad_u.astype(np_dtype, copy=False)
            meas_idx += n_meas_this

        grad_u_t = torch.from_numpy(grad_u_block).to(self.torch_device, dtype=self.torch_dtype)
        adj_block_t = torch.from_numpy(adj_block).to(self.torch_device, dtype=self.torch_dtype)
        sensitivity = -(adj_block_t * grad_u_t).sum(dim=2) * self.cell_areas_t
        return sensitivity.cpu().numpy()
