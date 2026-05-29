"""EIT imaging workflow wrapper.

This subpackage provides high-level helper functions for difference and absolute
imaging.  Workflow implementations pull solver stacks, so resolve them lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".absolute": ("perform_absolute_reconstruction",),
    ".difference": ("perform_difference_reconstruction",),
    ".sparse_bayesian": (
        "perform_sparse_absolute_reconstruction",
        "perform_sparse_difference_reconstruction",
    ),
    ".base": ("ReconstructionResult",),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = [
    "perform_absolute_reconstruction",
    "perform_difference_reconstruction",
    "perform_sparse_absolute_reconstruction",
    "perform_sparse_difference_reconstruction",
    "ReconstructionResult",
]

_SUBMODULE_NAMES = frozenset({"absolute", "base", "difference", "sparse_bayesian"})


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is not None:
        module = import_module(module_name, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SUBMODULE_NAMES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULE_NAMES))
