"""PyEIDORS visualization module.

Plotting helpers depend on Matplotlib/DOLFINx runtime modules, so keep package
import lightweight and resolve visualizers only when plotting is requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["EITVisualizer", "create_visualizer"]

_EXPORT_MODULES = {
    "EITVisualizer": ".eit_plots",
    "create_visualizer": ".eit_plots",
}


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
