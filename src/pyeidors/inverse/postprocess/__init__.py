"""Post-processing helpers for inverse EIT reconstructions.

Temporal and TV post-processing pull NumPy/SciPy helpers.  Keep package import
light and resolve post-processing routines only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".temporal": (
        "TemporalTVPipelineResult",
        "exponential_smooth_frames",
        "moving_average_frames",
        "postprocess_rm_frames",
    ),
    ".tv": ("TVRefinementResult", "refine_tv_pdhg", "total_variation_norm"),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = [
    "TemporalTVPipelineResult",
    "TVRefinementResult",
    "exponential_smooth_frames",
    "moving_average_frames",
    "postprocess_rm_frames",
    "refine_tv_pdhg",
    "total_variation_norm",
]

_SUBMODULE_NAMES = frozenset({"temporal", "tv"})


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
