"""PyEidors - Python版本的EIDORS电阻抗成像系统

基于FEniCS、PyTorch和CUQIpy的模块化EIT系统
在Docker容器环境中运行，提供GPU加速和贝叶斯推断功能
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# 检查关键依赖
try:
    import fenics
    _FENICS_AVAILABLE = True
except ImportError:
    _FENICS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE = False

try:
    import cuqi
    _CUQI_AVAILABLE = True
except ImportError:
    _CUQI_AVAILABLE = False

# 主要接口
from .core_system import EITSystem

# 环境信息
def check_environment():
    """检查运行环境"""
    info = {
        'fenics_available': _FENICS_AVAILABLE,
        'torch_available': _TORCH_AVAILABLE,
        'cuda_available': _CUDA_AVAILABLE,
        'cuqi_available': _CUQI_AVAILABLE,
    }
    if _TORCH_AVAILABLE:
        info['torch_version'] = torch.__version__
        info['cuda_device_count'] = torch.cuda.device_count() if _CUDA_AVAILABLE else 0
    return info

__all__ = ["EITSystem", "check_environment", "__version__"]