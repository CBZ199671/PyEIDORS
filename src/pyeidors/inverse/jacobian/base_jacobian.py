"""EIT雅可比矩阵计算基类"""

import numpy as np
from abc import ABC, abstractmethod
from fenics import Function

from ...data.structures import EITImage


class BaseJacobianCalculator(ABC):
    """雅可比计算器基类"""
    
    def __init__(self, fwd_model):
        self.fwd_model = fwd_model
        self.n_elements = len(Function(fwd_model.V_sigma).vector()[:])
        self.n_measurements = fwd_model.pattern_manager.n_meas_total
    
    @abstractmethod
    def calculate(self, sigma: Function, **kwargs) -> np.ndarray:
        """计算雅可比矩阵
        
        参数:
            sigma: 当前导电率分布
            
        返回:
            雅可比矩阵 (n_measurements × n_elements)
        """
        pass
    
    def calculate_from_image(self, img: EITImage, **kwargs) -> np.ndarray:
        """从EIT图像计算雅可比矩阵"""
        sigma = Function(self.fwd_model.V_sigma)
        sigma.vector()[:] = img.get_conductivity()
        return self.calculate(sigma, **kwargs)