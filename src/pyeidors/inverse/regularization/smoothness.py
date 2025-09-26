"""平滑性正则化"""

import numpy as np
from scipy.sparse import csr_matrix
from fenics import cells, edges

from .base_regularization import BaseRegularization


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