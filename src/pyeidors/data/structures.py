"""PyEidors数据结构定义"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Any


@dataclass
class PatternConfig:
    """激励和测量模式配置"""
    n_elec: int
    n_rings: int = 1
    stim_pattern: Union[str, List[int]] = '{ad}'
    meas_pattern: Union[str, List[int]] = '{ad}'
    amplitude: float = 1.0
    use_meas_current: bool = False
    use_meas_current_next: int = 0
    rotate_meas: bool = True


@dataclass
class EITData:
    """EIT数据容器"""
    meas: np.ndarray
    stim_pattern: np.ndarray
    n_elec: int
    n_stim: int
    n_meas: int
    type: str = 'real'


@dataclass
class EITImage:
    """EIT图像容器"""
    elem_data: np.ndarray
    fwd_model: Any
    type: str = 'conductivity'
    name: str = ''
    
    def get_conductivity(self) -> np.ndarray:
        if self.type == 'resistivity':
            return 1.0 / self.elem_data
        return self.elem_data


@dataclass
class MeshConfig:
    """网格配置参数"""
    radius: float = 1.0
    refinement: int = 8
    electrode_vertices: int = 8
    gap_vertices: int = 4
    mesh_size: float = 0.1


@dataclass
class ElectrodePosition:
    """电极位置信息"""
    L: int  # 电极数量
    positions: List[Tuple[float, float]]  # 电极位置角度对 (start, end)
    
    @classmethod
    def create_circular(cls, n_elec: int = 16, radius: float = 1.0) -> 'ElectrodePosition':
        """创建圆形电极位置"""
        import math
        
        # 计算电极覆盖角度
        electrode_width = 2 * math.pi / n_elec / 4  # 每个电极覆盖1/4圆周
        gap_width = 2 * math.pi / n_elec * 3 / 4    # 间隙覆盖3/4圆周
        
        positions = []
        for i in range(n_elec):
            center_angle = 2 * math.pi * i / n_elec
            start_angle = center_angle - electrode_width / 2
            end_angle = center_angle + electrode_width / 2
            positions.append((start_angle, end_angle))
        
        return cls(L=n_elec, positions=positions)