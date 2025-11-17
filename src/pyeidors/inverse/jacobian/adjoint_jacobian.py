"""EIDORS风格的伴随法雅可比计算器（含符号约定），可选Torch加速。"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional, List
from fenics import Function

from .base_jacobian import BaseJacobianCalculator


class EidorsStyleAdjointJacobian(BaseJacobianCalculator):
    """按 EIDORS 伴随法实现的雅可比计算器。

    特点：
      - 使用测量模式转置作为伴随-current，单次因子分解，批量求解；
      - 符号约定与 EIDORS 一致（dV/dσ 为负，因此最终雅可比内置负号）；
      - 可选 torch 加速敏感度积累（仅在 CPU/GPU 上做向量化累加）。
    """

    def __init__(self, fwd_model, use_torch: bool = False, device: Optional[str] = None):
        super().__init__(fwd_model)
        self.use_torch = use_torch
        if device is None:
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.torch_device = torch.device(device)
        self._setup()

    def _setup(self):
        self.mesh = self.fwd_model.mesh
        self.V = self.fwd_model.V
        self.V_sigma = self.fwd_model.V_sigma
        # 单元面积/体积
        v = self.fwd_model.DG0_Test if hasattr(self.fwd_model, "DG0_Test") else None
        if v is None:
            from fenics import TestFunction, FunctionSpace, assemble, dx
            DG0 = FunctionSpace(self.mesh, "DG", 0)
            v = TestFunction(DG0)
            self.cell_areas = assemble(v * dx).get_local()
        else:
            self.cell_areas = self.fwd_model.cell_areas
        # 方便 Torch
        if self.use_torch:
            self.cell_areas_t = torch.from_numpy(self.cell_areas).to(self.torch_device, dtype=torch.float64)

    def calculate(self, sigma: Function, **kwargs) -> np.ndarray:
        """计算测量雅可比矩阵（形状：n_meas x n_elem）。"""
        # 1) 正解
        u_all, _ = self.fwd_model.forward_solve(sigma)
        grad_u_all = self._compute_field_gradients(u_all)

        # 2) 伴随场（测量模式转置为电流）
        meas_curr = self._measurement_to_current_patterns()
        adj_fields, _ = self.fwd_model.forward_solve(sigma, meas_curr)
        grad_adj_all = self._compute_field_gradients(adj_fields)

        # 3) 组装 J，符号为负（dV/dσ<0）
        if self.use_torch:
            J = self._assemble_torch(grad_u_all, grad_adj_all)
        else:
            J = self._assemble_numpy(grad_u_all, grad_adj_all)

        # 刺激幅值缩放
        try:
            amp = float(getattr(self.fwd_model.pattern_manager.config, "amplitude", 1.0))
        except Exception:
            amp = 1.0
        return J * amp

    def _compute_field_gradients(self, field_solutions):
        """把节点解转换为单元梯度。"""
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
        """将测量矩阵转置为伴随电流模式（与 EIDORS 伴随法对应）。"""
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
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elem = len(self.cell_areas)
        J_t = torch.zeros((n_meas, n_elem), device=self.torch_device, dtype=torch.float64)

        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]
            grad_u_t = torch.from_numpy(grad_u).to(self.torch_device, dtype=torch.float64)
            for k in range(n_meas_this):
                adj_grad_t = torch.from_numpy(grad_adj_all[meas_idx + k]).to(self.torch_device, dtype=torch.float64)
                sensitivity = -(grad_u_t * adj_grad_t).sum(dim=1) * self.cell_areas_t
                J_t[meas_idx + k, :] = sensitivity
            meas_idx += n_meas_this
        return J_t.cpu().numpy()
