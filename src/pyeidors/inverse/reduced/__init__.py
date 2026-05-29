"""Reduced-order helpers for fused 3D GN acceleration.

Reduced operators and low-rank helpers import NumPy/SciPy kernels.  Keep the
package import light and resolve concrete helpers on demand.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".inexact_controller": ("InexactController",),
    ".lowrank_subspace": ("build_lowrank_subspace",),
    ".pod_basis": ("compute_pod_basis", "merge_orthonormal_bases"),
    ".reduced_gn_step": ("build_reduced_operator", "solve_reduced_step"),
    ".snapshot_bank": ("SnapshotBank", "select_snapshot_matrix"),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = [
    "InexactController",
    "SnapshotBank",
    "build_lowrank_subspace",
    "compute_pod_basis",
    "merge_orthonormal_bases",
    "build_reduced_operator",
    "select_snapshot_matrix",
    "solve_reduced_step",
]

_SUBMODULE_NAMES = frozenset(
    {
        "inexact_controller",
        "lowrank_subspace",
        "pod_basis",
        "reduced_gn_step",
        "snapshot_bank",
    }
)


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
