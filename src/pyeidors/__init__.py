"""PyEIDORS - Python implementation of EIDORS for Electrical Impedance Tomography.

A modular EIT system based on DOLFINx, PyTorch, and CUQIpy.
"""

from __future__ import annotations

from .utils.cuqi_imports import suppress_known_cuqi_import_warnings

__version__ = "1.0.0"
__author__ = "BingZhou Chen"

# Check critical dependencies
try:
    import dolfinx  # noqa: F401

    _DOLFINX_AVAILABLE = True
except ImportError:
    _DOLFINX_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _MPS_AVAILABLE = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
except ImportError:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE = False
    _MPS_AVAILABLE = False

try:
    with suppress_known_cuqi_import_warnings():
        import cuqi  # noqa: F401

    _CUQI_AVAILABLE = True
except ImportError:
    _CUQI_AVAILABLE = False

# Main interface
from .core_system import EITSystem


# Environment info
def check_environment():
    """Check runtime environment and available dependencies."""

    info = {
        "dolfinx_available": _DOLFINX_AVAILABLE,
        "torch_available": _TORCH_AVAILABLE,
        "cuda_available": _CUDA_AVAILABLE,
        "mps_available": _MPS_AVAILABLE,
        "cuqi_available": _CUQI_AVAILABLE,
    }
    if _TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__
        info["cuda_device_count"] = torch.cuda.device_count() if _CUDA_AVAILABLE else 0

    return info


__all__ = ["EITSystem", "check_environment", "__version__"]
