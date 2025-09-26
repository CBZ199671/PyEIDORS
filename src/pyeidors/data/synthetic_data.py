"""合成EIT数据生成器"""

import numpy as np
from typing import Tuple
from fenics import Function, cells

from .structures import EITData, EITImage


def create_synthetic_data(fwd_model,
                         inclusion_conductivity: float = 2.5,
                         background_conductivity: float = 1.0,
                         noise_level: float = 0.02,
                         center: Tuple[float, float] = (0.2, 0.2),
                         radius: float = 0.3):
    """创建合成EIT测试数据
    
    参数:
        fwd_model: 前向模型
        inclusion_conductivity: 异常导电率
        background_conductivity: 背景导电率
        noise_level: 噪声水平
        center: 异常中心位置
        radius: 异常半径
        
    返回:
        包含真实分布、清洁数据、噪声数据等的字典
    """
    
    # 创建真实导电率分布
    sigma_true = Function(fwd_model.V_sigma)
    sigma_true.vector()[:] = background_conductivity
    
    # 添加圆形异常
    for cell in cells(fwd_model.mesh):
        cell_center = cell.midpoint()
        x, y = cell_center.x(), cell_center.y()
        if (x - center[0])**2 + (y - center[1])**2 < radius**2:
            sigma_true.vector()[cell.index()] = inclusion_conductivity
    
    # 生成清洁测量数据
    img_true = EITImage(elem_data=sigma_true.vector()[:], fwd_model=fwd_model)
    data_clean, _ = fwd_model.fwd_solve(img_true)
    
    # 添加高斯白噪声
    np.random.seed(42)  # 确保可重复性
    noise = noise_level * np.std(data_clean.meas) * np.random.randn(len(data_clean.meas))
    data_noisy = EITData(
        meas=data_clean.meas + noise,
        stim_pattern=data_clean.stim_pattern,
        n_elec=data_clean.n_elec,
        n_stim=data_clean.n_stim,
        n_meas=data_clean.n_meas,
        type='simulated_noisy'
    )
    
    snr_db = 20 * np.log10(np.std(data_clean.meas) / np.std(noise))
    
    return {
        'sigma_true': sigma_true,
        'data_clean': data_clean,
        'data_noisy': data_noisy,
        'noise': noise,
        'snr_db': snr_db
    }


def create_custom_phantom(fwd_model,
                         background_conductivity: float = 1.0,
                         anomalies: list = None):
    """创建自定义幻象
    
    参数:
        fwd_model: 前向模型
        background_conductivity: 背景导电率
        anomalies: 异常列表，每个异常为字典，包含center, radius, conductivity
        
    返回:
        导电率分布Function对象
    """
    
    if anomalies is None:
        anomalies = []
    
    # 创建背景导电率分布
    sigma = Function(fwd_model.V_sigma)
    sigma.vector()[:] = background_conductivity
    
    # 添加异常
    for anomaly in anomalies:
        center = anomaly.get('center', (0.0, 0.0))
        radius = anomaly.get('radius', 0.2)
        conductivity = anomaly.get('conductivity', 2.0)
        
        for cell in cells(fwd_model.mesh):
            cell_center = cell.midpoint()
            x, y = cell_center.x(), cell_center.y()
            if (x - center[0])**2 + (y - center[1])**2 < radius**2:
                sigma.vector()[cell.index()] = conductivity
    
    return sigma