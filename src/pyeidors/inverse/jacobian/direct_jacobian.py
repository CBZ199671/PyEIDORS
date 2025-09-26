"""直接方法雅可比计算器 - 使用已有激励模式"""

import numpy as np
from fenics import *

from .base_jacobian import BaseJacobianCalculator


class DirectJacobianCalculator(BaseJacobianCalculator):
    """直接方法雅可比计算器 - 优化版本"""
    
    def __init__(self, fwd_model):
        super().__init__(fwd_model)
        self._setup_computation()
    
    def _setup_computation(self):
        """设置计算所需的函数空间和测试函数"""
        self.mesh = self.fwd_model.mesh
        self.V = self.fwd_model.V
        self.V_sigma = self.fwd_model.V_sigma
        
        # 创建梯度函数空间
        self.Q_DG = VectorFunctionSpace(self.mesh, "DG", 0)
        self.DG0 = FunctionSpace(self.mesh, "DG", 0)
        
        # 计算单元面积
        v = TestFunction(self.DG0)
        self.cell_areas = assemble(v * dx).get_local()
    
    def calculate(self, sigma: Function, method: str = 'efficient', **kwargs) -> np.ndarray:
        """
        计算雅可比矩阵
        
        参数:
            sigma: 导电率分布
            method: 计算方法 ('efficient' 或 'traditional')
        """
        if method == 'efficient':
            return self._calculate_efficient(sigma)
        elif method == 'traditional':
            return self._calculate_traditional(sigma)
        else:
            raise ValueError(f"未知方法: {method}")
    
    def _calculate_efficient(self, sigma: Function) -> np.ndarray:
        """
        高效雅可比计算 - 直接使用激励模式
        避免为每个电极单独求解
        """
        # 1. 正向求解 - 使用现有激励模式
        u_all, U_all = self.fwd_model.forward_solve(sigma)
        
        # 2. 计算正向场梯度
        grad_u_all = self._compute_field_gradients(u_all)
        
        # 3. 计算测量模式对应的伴随场
        # 这里使用测量模式的转置作为激励来计算伴随场
        adjoint_fields = self._compute_adjoint_fields_efficient(sigma)
        
        # 4. 计算雅可比矩阵
        jacobian = self._assemble_jacobian_efficient(grad_u_all, adjoint_fields)
        
        return jacobian
    
    def _calculate_traditional(self, sigma: Function) -> np.ndarray:
        """传统雅可比计算方法 - 与原代码兼容"""
        # 正向求解
        u_all, _ = self.fwd_model.forward_solve(sigma)
        
        # 构造单位电流模式（伴随场计算）
        I2_all = np.eye(self.fwd_model.n_elec)
        bu_all, _ = self.fwd_model.forward_solve(sigma, I2_all)
        
        # 计算梯度
        grad_u_all = self._compute_field_gradients(u_all)
        grad_bu_all = self._compute_field_gradients(bu_all)
        
        # 组装雅可比矩阵
        jacobian = self._assemble_jacobian_traditional(grad_u_all, grad_bu_all)
        
        # 转换为测量雅可比
        measurement_jacobian = self._convert_to_measurement_jacobian(jacobian)
        
        return measurement_jacobian
    
    def _compute_field_gradients(self, field_solutions):
        """计算场梯度"""
        gradients = []
        for field in field_solutions:
            u_fun = Function(self.V)
            u_fun.vector()[:] = field
            
            grad_u = project(grad(u_fun), self.Q_DG)
            grad_u_vec = grad_u.vector().get_local().reshape(-1, 2)
            gradients.append(grad_u_vec)
        
        return gradients
    
    def _compute_adjoint_fields_efficient(self, sigma: Function):
        """高效计算伴随场 - 使用测量模式"""
        # 将测量模式转换为电流激励模式
        adjoint_patterns = self._measurement_to_current_patterns()
        
        # 求解伴随场
        adjoint_fields, _ = self.fwd_model.forward_solve(sigma, adjoint_patterns)
        
        # 计算梯度
        adjoint_gradients = self._compute_field_gradients(adjoint_fields)
        
        return adjoint_gradients
    
    def _measurement_to_current_patterns(self):
        """将测量模式转换为电流激励模式"""
        # 这里需要根据测量模式构造相应的电流模式
        # 简化实现 - 可以进一步优化
        n_meas = self.fwd_model.pattern_manager.n_meas_total
        n_elec = self.fwd_model.n_elec
        
        current_patterns = np.zeros((n_elec, n_meas))
        
        meas_idx = 0
        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            n_meas_this_stim = meas_matrix.shape[0]
            
            # 将测量矩阵转置作为电流模式
            current_patterns[:, meas_idx:meas_idx + n_meas_this_stim] = meas_matrix.T
            meas_idx += n_meas_this_stim
        
        return current_patterns
    
    def _assemble_jacobian_efficient(self, grad_u_all, adjoint_gradients):
        """高效组装雅可比矩阵"""
        n_measurements = len(adjoint_gradients)
        n_elements = len(self.cell_areas)
        
        jacobian = np.zeros((n_measurements, n_elements))
        
        # 根据激励-测量对应关系计算雅可比
        meas_idx = 0
        for stim_idx, grad_u in enumerate(grad_u_all):
            n_meas_this_stim = self.fwd_model.pattern_manager.n_meas_per_stim[stim_idx]
            
            for local_meas_idx in range(n_meas_this_stim):
                global_meas_idx = meas_idx + local_meas_idx
                adjoint_grad = adjoint_gradients[global_meas_idx]
                
                # 计算敏感度
                sensitivity = np.sum(grad_u * adjoint_grad, axis=1) * self.cell_areas
                jacobian[global_meas_idx, :] = sensitivity
            
            meas_idx += n_meas_this_stim
        
        return jacobian
    
    def _assemble_jacobian_traditional(self, grad_u_all, grad_bu_all):
        """传统方式组装雅可比矩阵"""
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
        """将电极雅可比转换为测量雅可比"""
        measurement_jacobian_blocks = []
        
        for stim_idx in range(self.fwd_model.pattern_manager.n_stim):
            elec_start = stim_idx * self.fwd_model.n_elec
            elec_end = (stim_idx + 1) * self.fwd_model.n_elec
            electrode_jac_for_stim = electrode_jacobian[elec_start:elec_end, :]
            
            meas_matrix = self.fwd_model.pattern_manager.meas_matrices[stim_idx]
            meas_jacobian_for_stim = meas_matrix @ electrode_jac_for_stim
            
            measurement_jacobian_blocks.append(meas_jacobian_for_stim)
        
        return np.vstack(measurement_jacobian_blocks)