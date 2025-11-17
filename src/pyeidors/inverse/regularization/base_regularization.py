"""正则化基类"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix


class BaseRegularization(ABC):
    """正则化基类"""
    
    def __init__(self, fwd_model):
        self.fwd_model = fwd_model
        self.mesh = fwd_model.mesh
        V_sigma = fwd_model.fwd_model.V_sigma if hasattr(fwd_model, 'fwd_model') else fwd_model.V_sigma
        try:
            self.n_elements = V_sigma.dim()
        except AttributeError:
            self.n_elements = len(V_sigma)
    
    @abstractmethod
    def create_matrix(self) -> np.ndarray:
        """创建正则化矩阵"""
        pass
    
    def get_regularization_matrix(self, cache: bool = True) -> np.ndarray:
        """获取正则化矩阵（支持缓存）"""
        if not hasattr(self, '_cached_matrix') or not cache:
            self._cached_matrix = self.create_matrix()
        return self._cached_matrix
