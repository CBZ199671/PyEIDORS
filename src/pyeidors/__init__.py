"""PyEIDORS - Python implementation of EIDORS for Electrical Impedance Tomography.

A modular EIT system based on DOLFINx, PyTorch, and CUQIpy.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"
__author__ = "BingZhou Chen"

_ENVIRONMENT_CACHE: dict[str, object] | None = None
_PRIVATE_ENV_FLAGS = {
    "_DOLFINX_AVAILABLE": "dolfinx_available",
    "_TORCH_AVAILABLE": "torch_available",
    "_CUDA_AVAILABLE": "cuda_available",
    "_MPS_AVAILABLE": "mps_available",
    "_CUQI_AVAILABLE": "cuqi_available",
}


def _probe_dolfinx_available() -> bool:
    try:
        import dolfinx  # noqa: F401
    except Exception:
        return False
    return True


def _probe_torch() -> dict[str, object]:
    try:
        import torch
    except Exception:
        return {
            "torch_available": False,
            "cuda_available": False,
            "mps_available": False,
        }

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "torch_available": True,
        "cuda_available": cuda_available,
        "mps_available": bool(mps_backend and mps_backend.is_available()),
        "torch_version": torch.__version__,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }


def _probe_cuqi_available() -> bool:
    try:
        from .utils.cuqi_imports import suppress_known_cuqi_import_warnings

        with suppress_known_cuqi_import_warnings():
            import cuqi  # noqa: F401
    except Exception:
        return False
    return True


def _compute_environment() -> dict[str, object]:
    global _ENVIRONMENT_CACHE
    if _ENVIRONMENT_CACHE is not None:
        return dict(_ENVIRONMENT_CACHE)

    info: dict[str, object] = {
        "dolfinx_available": _probe_dolfinx_available(),
        **_probe_torch(),
        "cuqi_available": _probe_cuqi_available(),
    }
    info.setdefault("torch_version", None)
    info.setdefault("cuda_device_count", 0)
    _ENVIRONMENT_CACHE = dict(info)
    return info


def check_environment() -> dict[str, object]:
    """Check runtime environment and available dependencies."""
    return _compute_environment()


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


def __getattr__(name: str) -> Any:
    if name in _PRIVATE_ENV_FLAGS:
        return bool(_compute_environment()[_PRIVATE_ENV_FLAGS[name]])
    if name != "EITSystem":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from .core_system import EITSystem as _EITSystem
    except ImportError as exc:
        raise _runtime_import_error(exc) from exc
    return _EITSystem


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"EITSystem"} | set(_PRIVATE_ENV_FLAGS))


__all__ = ["EITSystem", "check_environment", "__version__"]
