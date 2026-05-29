"""PyEIDORS electrode system module.

Keep package import lightweight so helpers such as
``pyeidors.electrodes.layout`` do not eagerly import pattern generation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {"StimMeasPatternManager": ".patterns"}

__all__ = ["StimMeasPatternManager"]

_SUBMODULE_NAMES = frozenset({"layout", "patterns"})


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
