"""PyEIDORS shared script helpers.

Keep package import side effects minimal so lightweight CLI operations such as
`--help` and `--dry-run` do not require the full scientific runtime.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "load_csv_measurements": ("io_utils", "load_csv_measurements"),
    "load_metadata": ("io_utils", "load_metadata"),
    "save_reconstruction_results": ("io_utils", "save_reconstruction_results"),
    "compute_scale_bias": ("calibration", "compute_scale_bias"),
    "apply_calibration": ("calibration", "apply_calibration"),
    "cell_to_node": ("mesh_utils", "cell_to_node"),
}

_LAZY_MODULES = {
    "gn_absolute_runner",
    "gn_difference_runner",
    "sparse_bayes_runner",
    "case_discovery",
    "case_loader",
    "method_runners",
    "output_writer",
    "recon_cli_models",
    "io_utils",
    "calibration",
    "mesh_utils",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        return getattr(module, attr_name)
    if name in _LAZY_MODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | _LAZY_MODULES)


__all__ = sorted(set(_LAZY_EXPORTS) | _LAZY_MODULES)
