"""In-process coverage for pyeidors import guard branches."""

from __future__ import annotations

import importlib
import sys

import pyeidors


def _reload_with_missing(module_name: str):
    saved = sys.modules.get(module_name)
    sys.modules[module_name] = None
    try:
        mod = importlib.reload(pyeidors)
        return {
            "dolfinx": mod._DOLFINX_AVAILABLE,
            "torch": mod._TORCH_AVAILABLE,
            "cuda": mod._CUDA_AVAILABLE,
            "mps": mod._MPS_AVAILABLE,
            "cuqi": mod._CUQI_AVAILABLE,
        }
    finally:
        if saved is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved
        importlib.reload(pyeidors)


def test_import_guard_sets_dolfinx_flag_false():
    flags = _reload_with_missing("dolfinx")
    assert flags["dolfinx"] is False


def test_import_guard_sets_torch_flags_false():
    flags = _reload_with_missing("torch")
    assert flags["torch"] is False
    assert flags["cuda"] is False
    assert flags["mps"] is False


def test_import_guard_sets_cuqi_flag_false():
    flags = _reload_with_missing("cuqi")
    assert flags["cuqi"] is False
