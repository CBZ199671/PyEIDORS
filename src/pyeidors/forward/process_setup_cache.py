"""Process-local cache for forward-model static setup bundles."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import threading
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ..cache.keys import _normalize, hash_array
from ..data.structures import PatternConfig


@dataclass(frozen=True)
class ForwardStaticSetupBundle:
    """Read-only static objects shared across compatible forward models."""

    ds_electrodes: Any
    electrode_tags: tuple[int, ...]
    electrode_boundary_measures: dict[int, float]
    geometry_scale_to_m: float
    mesh_tdim: int
    boundary_scale_to_m: float
    electrode_lengths_m: np.ndarray
    pattern_manager: Any
    V: Any
    V_sigma: Any
    dofs: int
    electrode_matrix: csr_matrix


_PROCESS_FORWARD_SETUP_CACHE_MAX_ITEMS = 8
_PROCESS_FORWARD_SETUP_CACHE: OrderedDict[str, ForwardStaticSetupBundle] = OrderedDict()
_PROCESS_FORWARD_SETUP_CACHE_LOCK = threading.Lock()


def _pattern_signature(config: PatternConfig) -> dict[str, Any]:
    return _normalize(asdict(config))


def build_process_forward_setup_key(
    *,
    mesh_runtime_id: int,
    mesh_file: str | None,
    n_elec: int,
    z: np.ndarray,
    pattern_config: PatternConfig,
) -> str:
    payload = {
        "mesh_runtime_id": int(mesh_runtime_id),
        "mesh_file": str(mesh_file or ""),
        "n_elec": int(n_elec),
        "z_hash": hash_array(np.asarray(z, dtype=np.float64).reshape(-1)),
        "pattern_config": _pattern_signature(pattern_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_process_forward_setup_bundle(key: str) -> ForwardStaticSetupBundle | None:
    with _PROCESS_FORWARD_SETUP_CACHE_LOCK:
        bundle = _PROCESS_FORWARD_SETUP_CACHE.get(key)
        if bundle is None:
            return None
        _PROCESS_FORWARD_SETUP_CACHE.move_to_end(key)
        return bundle


def put_process_forward_setup_bundle(key: str, bundle: ForwardStaticSetupBundle) -> None:
    with _PROCESS_FORWARD_SETUP_CACHE_LOCK:
        _PROCESS_FORWARD_SETUP_CACHE.pop(key, None)
        _PROCESS_FORWARD_SETUP_CACHE[key] = bundle
        while len(_PROCESS_FORWARD_SETUP_CACHE) > _PROCESS_FORWARD_SETUP_CACHE_MAX_ITEMS:
            _PROCESS_FORWARD_SETUP_CACHE.popitem(last=False)


def clear_process_forward_setup_cache() -> None:
    with _PROCESS_FORWARD_SETUP_CACHE_LOCK:
        _PROCESS_FORWARD_SETUP_CACHE.clear()


def process_forward_setup_cache_stats() -> dict[str, int]:
    with _PROCESS_FORWARD_SETUP_CACHE_LOCK:
        return {"items": len(_PROCESS_FORWARD_SETUP_CACHE), "max_items": _PROCESS_FORWARD_SETUP_CACHE_MAX_ITEMS}
