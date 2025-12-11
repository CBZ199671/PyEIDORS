"""测量数据校准工具函数。"""

from __future__ import annotations

import numpy as np


def compute_scale_bias(measured: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    """计算将模型响应缩放/平移到实测尺度的线性系数。
    
    使用最小二乘法找到 scale 和 bias，使得:
        measured ≈ scale * model + bias
    
    Args:
        measured: 实测数据向量
        model: 模型预测向量
        
    Returns:
        (scale, bias) 线性变换系数
    """
    x = np.asarray(model, dtype=float)
    y = np.asarray(measured, dtype=float)
    
    denom = float(np.dot(x, x))
    if denom < 1e-18:
        return 1.0, float(y.mean())
    
    scale = float(np.dot(y, x) / denom)
    if abs(scale) < 1e-12:
        scale = 1.0 if scale >= 0 else -1.0
    
    bias = float(y.mean() - scale * x.mean())
    return scale, bias


def apply_calibration(vector: np.ndarray, scale: float, bias: float) -> np.ndarray:
    """应用校准变换的逆变换。
    
    将数据从校准后的空间变换回原始空间:
        original = (calibrated - bias) / scale
    
    Args:
        vector: 待变换的数据向量
        scale: 缩放系数
        bias: 偏移系数
        
    Returns:
        变换后的向量
    """
    arr = np.asarray(vector, dtype=float)
    safe_scale = abs(scale)
    if safe_scale < np.finfo(float).eps:
        safe_scale = 1.0
    return (arr - bias) / safe_scale


def normalize_measurements(
    measurements: np.ndarray,
    target_amplitude: float | None = None,
    source_amplitude: float = 1.0,
) -> tuple[np.ndarray, float]:
    """归一化测量数据到目标幅值。
    
    Args:
        measurements: 测量数据
        target_amplitude: 目标激励幅值，None 表示不缩放
        source_amplitude: 原始激励幅值
        
    Returns:
        (normalized_measurements, scale_factor)
    """
    if target_amplitude is None or target_amplitude <= 0:
        return measurements, 1.0
    if source_amplitude <= 0:
        raise ValueError("source_amplitude must be positive")
    
    factor = target_amplitude / source_amplitude
    return measurements * factor, factor
