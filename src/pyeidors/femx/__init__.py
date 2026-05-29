"""FEM helpers for the DOLFINx-only runtime.

The helper implementations import DOLFINx/UFL, so keep package import light and
resolve helper functions only when callers actually need FEM runtime objects.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "build_eit_mesh": ".helpers",
    "cell_midpoints": ".helpers",
    "create_ds_measure": ".helpers",
    "estimate_radius": ".helpers",
    "function_get_array": ".helpers",
    "function_set_array": ".helpers",
    "function_size": ".helpers",
    "mesh_cell_vertices": ".helpers",
    "mesh_coordinates": ".helpers",
    "mesh_facet_vertices": ".helpers",
    "mesh_num_cells": ".helpers",
    "mesh_num_edges": ".helpers",
    "mesh_num_vertices": ".helpers",
}

__all__ = [
    "build_eit_mesh",
    "cell_midpoints",
    "create_ds_measure",
    "estimate_radius",
    "function_get_array",
    "function_set_array",
    "function_size",
    "mesh_cell_vertices",
    "mesh_coordinates",
    "mesh_facet_vertices",
    "mesh_num_cells",
    "mesh_num_edges",
    "mesh_num_vertices",
]

_SUBMODULE_NAMES = frozenset({"helpers"})


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
