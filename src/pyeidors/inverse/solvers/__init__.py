"""PyEIDORS inverse problem solver module.

Solver implementations pull numerical backends and optional sparse/Bayesian
stacks.  Keep the package import cheap and resolve public solver symbols or
submodules on demand.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".gauss_newton": ("GaussNewtonReconstructor",),
    ".matrix_free_gn": ("MatrixFreeGNStepResult", "solve_matrix_free_gn_step"),
    ".sparse_bayesian_engine": (
        "SparseBayesianConfig",
        "SparseBayesianReconstructor",
    ),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = [
    "GaussNewtonReconstructor",
    "MatrixFreeGNStepResult",
    "solve_matrix_free_gn_step",
    "SparseBayesianReconstructor",
    "SparseBayesianConfig",
]

_SUBMODULE_NAMES = frozenset(
    {
        "eit_pde",
        "gauss_newton",
        "gauss_newton_device",
        "gauss_newton_engine",
        "gauss_newton_line_search",
        "gauss_newton_linear_system",
        "gauss_newton_measurement_space",
        "gauss_newton_runtime",
        "gauss_newton_startup_cache",
        "gauss_newton_step_size",
        "gauss_newton_weights",
        "matrix_free_gn",
        "sparse_bayesian_engine",
        "sparse_map_solver",
        "sparse_optimizers",
        "sparse_projection",
        "sparse_runtime",
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
