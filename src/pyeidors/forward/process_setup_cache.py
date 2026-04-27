"""Process-local cache for forward-model static setup bundles.

T79 Path C: the OrderedDict LRU + threading.Lock + JSON-hash key
machinery moved to :mod:`pyeidors.cache.process_lru`; this module
pins the value type to :class:`ForwardStaticSetupBundle` and keeps the
historical public function names so ``EITForwardModel`` callers stay
intact. The cache-key payload formula (V16 / V17) is preserved
bytewise — same field set, same sort order, same JSON separators
through :func:`pyeidors.cache.process_lru.hash_json_payload`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ..cache.keys import _normalize, hash_array
from ..cache.process_lru import ProcessLRUCache, hash_json_payload
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
_PROCESS_FORWARD_SETUP_CACHE: ProcessLRUCache[ForwardStaticSetupBundle] = (
    ProcessLRUCache(max_items=_PROCESS_FORWARD_SETUP_CACHE_MAX_ITEMS)
)


def _pattern_signature(config: PatternConfig) -> dict[str, Any]:
    return _normalize(asdict(config))


def build_process_forward_setup_key(
    *,
    mesh_file: str | None,
    n_elec: int,
    z: np.ndarray,
    pattern_config: PatternConfig,
    mesh_content_hash: str | None = None,
) -> str:
    """Build a content-addressed cache key for forward static setup.

    A stable cache key requires at least one of ``mesh_file`` (file-backed
    mesh) or ``mesh_content_hash`` (in-memory mesh). Earlier revisions mixed
    ``id(self.mesh)`` into the key, which was unsafe: Python is free to reuse
    addresses after garbage collection, so an in-memory mesh could share an
    id with a freshly allocated replacement.
    """
    file_token = str(mesh_file or "")
    content_token = str(mesh_content_hash or "")
    if not file_token and not content_token:
        raise ValueError(
            "build_process_forward_setup_key requires either mesh_file "
            "or mesh_content_hash to form a stable cache key."
        )
    payload = {
        "mesh_file": file_token,
        "mesh_content_hash": content_token,
        "n_elec": int(n_elec),
        "z_hash": hash_array(np.asarray(z, dtype=np.float64).reshape(-1)),
        "pattern_config": _pattern_signature(pattern_config),
    }
    return hash_json_payload(payload)


def get_process_forward_setup_bundle(key: str) -> ForwardStaticSetupBundle | None:
    return _PROCESS_FORWARD_SETUP_CACHE.get(key)


def put_process_forward_setup_bundle(
    key: str, bundle: ForwardStaticSetupBundle
) -> None:
    _PROCESS_FORWARD_SETUP_CACHE.put(key, bundle)


def clear_process_forward_setup_cache() -> None:
    _PROCESS_FORWARD_SETUP_CACHE.clear()


def process_forward_setup_cache_stats() -> dict[str, int]:
    return _PROCESS_FORWARD_SETUP_CACHE.stats()
