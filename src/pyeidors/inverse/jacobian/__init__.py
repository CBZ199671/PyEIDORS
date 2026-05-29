"""PyEIDORS Jacobian calculation module.

Importing this package should be cheap: calculator implementations depend on
DOLFINx runtime objects, while cache and linearization helpers are often useful
in lightweight tests and diagnostics.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".base_jacobian": ("BaseJacobianCalculator",),
    ".direct_jacobian": ("DirectJacobianCalculator",),
    ".adjoint_jacobian": ("EidorsJacobianAdapter",),
    ".linearized": (
        "JacobianLinearization",
        "compute_sigma_fingerprint",
    ),
    ".process_jacobian_cache": (
        "build_process_jacobian_key",
        "clear_process_jacobian_cache",
        "get_process_cached_jacobian",
        "process_jacobian_cache_stats",
        "put_process_cached_jacobian",
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
