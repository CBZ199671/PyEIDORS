"""Physics helpers for DOLFINx-based EIT modeling.

The public helpers are small, but their implementations import NumPy and data
model code.  Resolve them lazily so package import stays cheap for cache/status
tools and GUI startup probes.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "build_stim_currents": ".current_drive",
    "normalize_drive_mode": ".current_drive",
    "normalize_pattern_config_for_mesh": ".current_drive",
    "resolve_electrode_lengths_m": ".current_drive",
    "validate_drive_config": ".current_drive",
    "UnitCheckItem": ".unit_consistency",
    "UnitCheckLevel": ".unit_consistency",
    "UnitCheckReport": ".unit_consistency",
    "run_unit_consistency_checks": ".unit_consistency",
}

__all__ = [
    "build_stim_currents",
    "normalize_drive_mode",
    "normalize_pattern_config_for_mesh",
    "resolve_electrode_lengths_m",
    "validate_drive_config",
    "UnitCheckItem",
    "UnitCheckLevel",
    "UnitCheckReport",
    "run_unit_consistency_checks",
]

_SUBMODULE_NAMES = frozenset({"current_drive", "unit_consistency"})


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
