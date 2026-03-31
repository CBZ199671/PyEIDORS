"""PyEIDORS forward problem helpers with lazy heavy imports."""

from __future__ import annotations

from typing import Any

__all__ = [
    "CudaStructuredForwardBackend",
    "EITForwardModel",
    "LinearBackendConfig",
]


def __getattr__(name: str) -> Any:
    if name not in {
        "CudaStructuredForwardBackend",
        "EITForwardModel",
        "LinearBackendConfig",
    }:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .cuda_structured_backend import CudaStructuredForwardBackend
    from .eit_forward_model import EITForwardModel, LinearBackendConfig

    exports = {
        "CudaStructuredForwardBackend": CudaStructuredForwardBackend,
        "EITForwardModel": EITForwardModel,
        "LinearBackendConfig": LinearBackendConfig,
    }
    return exports[name]
