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

# Environment info
def check_environment() -> dict[str, object]:
    """Check runtime environment and available dependencies."""
    info: dict[str, object] = {
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


def _runtime_import_error(exc: ImportError) -> ImportError:
    msg = (
        "EITSystem requires the supported FEniCSx/DOLFINx runtime. "
        "Run `nix develop` to enter the supported environment, then retry. "
        "For setup details see docs/NIX_FENICSX.md."
    )
    detail = str(exc).strip()
    if detail:
        msg = f"{msg} Original import error: {detail}"
    return ImportError(msg)


def __getattr__(name: str) -> object:
    if name != "EITSystem":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from .core_system import EITSystem as _EITSystem
    except ImportError as exc:
        raise _runtime_import_error(exc) from exc
    return _EITSystem


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"EITSystem"})


__all__ = ["EITSystem", "check_environment", "__version__"]
