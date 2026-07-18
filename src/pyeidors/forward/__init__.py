"""PyEIDORS forward problem helpers with lazy heavy imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".cuda_structured_backend": ("CudaStructuredForwardBackend",),
    ".eit_forward_model": ("EITForwardModel", "LinearBackendConfig"),
    ".robin_transconductance": (
        "RobinTransconductanceForwardModel",
        "normalize_cem_formulation",
        "zero_sum_helmert_basis",
    ),
    ".complex_support": (
        "petsc_scalar_dtype",
        "petsc_scalar_dtype_name",
        "petsc_scalar_is_complex",
        "require_complex_scalar_support",
        "require_runtime_scalar_dtype",
        "runtime_scalar_summary",
    ),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
