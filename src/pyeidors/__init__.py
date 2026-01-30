"""PyEIDORS - Python implementation of EIDORS for Electrical Impedance Tomography.

A modular EIT system based on FEniCS, PyTorch, and CUQIpy.
Runs in Docker container environment with GPU acceleration and Bayesian inference support.
"""

__version__ = "1.0.0"
__author__ = "BingZhou Chen"

# Check critical dependencies
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

# Main interface
from .core_system import EITSystem

# Environment info
def check_environment():
    """Check runtime environment and available dependencies."""
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