"""PyEIDORS geometry modeling module."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".derived_cache": (
        "MESH_DERIVED_SCHEMA",
        "MeshDerivedArrays",
        "build_mesh_derived_arrays",
        "load_mesh_derived_artifact",
        "load_or_build_mesh_derived_artifact",
        "mesh_derived_cache_path",
        "mesh_derived_signature",
        "mesh_derived_signature_payload",
        "write_mesh_derived_artifact",
    ),
    ".mesh_converter": ("MeshConverter",),
    ".mesh_generator": ("MeshGenerator",),
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
