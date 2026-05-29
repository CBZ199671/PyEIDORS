"""Prior builders for reconstruction-matrix workflows.

Most prior builders depend on SciPy sparse matrices.  Keep package import light
and resolve individual prior helpers only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".laplace": (
        "graph_curvature_prior",
        "graph_difference_operator",
        "graph_laplacian",
        "graph_ltl",
        "graph_ltl_prior",
    ),
    ".rtr": (
        "RtRPrior",
        "as_rtr_prior",
        "load_rtr_prior_artifact",
        "write_rtr_prior_artifact",
    ),
    ".tv_irls": (
        "TVIRLSResult",
        "solve_tv_irls_batch",
        "solve_tv_irls_frame",
        "tv_irls_objective",
        "tv_irls_prior_from_state",
    ),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = [
    "RtRPrior",
    "TVIRLSResult",
    "as_rtr_prior",
    "graph_curvature_prior",
    "graph_difference_operator",
    "graph_laplacian",
    "graph_ltl",
    "graph_ltl_prior",
    "load_rtr_prior_artifact",
    "solve_tv_irls_batch",
    "solve_tv_irls_frame",
    "tv_irls_objective",
    "tv_irls_prior_from_state",
    "write_rtr_prior_artifact",
]

_SUBMODULE_NAMES = frozenset({"_graph_core", "laplace", "rtr", "tv_irls"})


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
