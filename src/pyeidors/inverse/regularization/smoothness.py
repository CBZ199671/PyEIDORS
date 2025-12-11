"""平滑性正则化"""

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from fenics import cells, edges, Function

from .base_regularization import BaseRegularization
from ..jacobian.direct_jacobian import DirectJacobianCalculator


class SmoothnessRegularization(BaseRegularization):
    """平滑性正则化 - 基于拉普拉斯算子"""
    
    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha
    
    def create_matrix(self) -> np.ndarray:
        """创建平滑性正则化矩阵"""
        n_cells = self.mesh.num_cells()
        
        # 构建拉普拉斯算子
        rows, cols, data = [], [], []
        row_idx = 0
        
        # 基于网格拓扑构建拉普拉斯算子
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
        
        # 构建差分矩阵
        L = csr_matrix((data, (rows, cols)), shape=(row_idx, n_cells))
        
        # 返回 L^T * L * alpha
        regularization_matrix = self.alpha * (L.T @ L).toarray()
        
        return regularization_matrix


class TikhonovRegularization(BaseRegularization):
    """Tikhonov正则化"""
    
    def __init__(self, fwd_model, alpha: float = 1.0):
        super().__init__(fwd_model)
        self.alpha = alpha
    
    def create_matrix(self) -> np.ndarray:
        """创建Tikhonov正则化矩阵（单位矩阵）"""
        n_elements = self.n_elements
        return self.alpha * np.eye(n_elements)


class TotalVariationRegularization(BaseRegularization):
    """全变分正则化"""
    
    def __init__(self, fwd_model, alpha: float = 1.0, epsilon: float = 1e-6):
        super().__init__(fwd_model)
        self.alpha = alpha
        self.epsilon = epsilon
    
    def create_matrix(self) -> np.ndarray:
        """创建全变分正则化矩阵（近似）"""
        # 全变分正则化通常是非线性的，这里提供线性近似
        # 在实际使用中可能需要在求解过程中更新
        return self.alpha * np.eye(self.n_elements)
    
    def create_nonlinear_term(self, sigma_current: np.ndarray) -> np.ndarray:
        """创建非线性TV项（可在求解器中调用）"""
        # 实现基于当前解的TV正则化项
        # 这是一个简化版本，实际实现会更复杂
        grad_magnitude = np.abs(np.gradient(sigma_current))
        weights = 1.0 / (grad_magnitude + self.epsilon)
        
        # 构建加权拉普拉斯矩阵
        # 简化实现...
        return self.alpha * np.diag(weights)


class NOSERRegularization(BaseRegularization):
    """NOSER正则化 - 对角矩阵基于 J^T J 的对角线
    
    EIDORS 风格实现: Reg = diag(sum(J.^2, 1)).^exponent
    
    参数:
        fwd_model: 前向模型
        jacobian_calculator: 雅可比计算器
        base_conductivity: 用于计算基线雅可比的导电率
        alpha: 正则化系数
        exponent: NOSER 指数 (EIDORS 默认为 0.5)
        floor: 对角线元素的最小值，避免数值问题
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
        """
        参数:
            floor: 绝对 floor 值（当 adaptive_floor=False 时使用）
            adaptive_floor: 如果为 True，floor 会自适应到 J'J 的量级
            floor_fraction: 自适应 floor = max(diag(J'J)) × floor_fraction
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
        # EIDORS: diag_col = sum(J.^2, 1)'  (列向量)
        diag_entries = np.sum(jac * jac, axis=0)
        
        # 应用 floor 避免数值问题
        if self.adaptive_floor:
            # 自适应 floor：基于 J'J 最大值的一个小比例
            adaptive_floor_value = np.max(diag_entries) * self.floor_fraction
            effective_floor = max(adaptive_floor_value, 1e-100)  # 绝对最小值防止全零
        else:
            effective_floor = self.floor
        
        diag_entries = np.maximum(diag_entries, effective_floor)
        return diag_entries

    def create_matrix(self) -> np.ndarray:
        """创建 NOSER 正则化矩阵
        
        EIDORS: Reg = spdiags(diag_col.^exponent, 0, n, n)
        """
        if self._baseline_diag is None:
            self._baseline_diag = self._compute_baseline_diag()
        # 应用 EIDORS 风格的 exponent
        scaled_diag = self._baseline_diag ** self.exponent
        return self.alpha * np.diag(scaled_diag)
