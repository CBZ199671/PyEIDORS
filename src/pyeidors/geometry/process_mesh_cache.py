"""Process-local cache for loaded EIT mesh objects.

T79 Path C: storage + key-hashing primitives now live in
:mod:`pyeidors.cache.process_lru`. This module is a thin wrapper that
pins the value type to :class:`EITMesh` and preserves the historical
public function names (``build_process_mesh_cache_key`` /
``get_process_cached_mesh`` / ``put_process_cached_mesh`` /
``clear_process_mesh_cache``) so existing callers (``MeshLoader``,
``optimized_mesh_generator.load_or_create_mesh`` etc.) do not need to
update.

The cache-key payload formula is bytewise identical to the
pre-consolidation implementation: same field set, same sort order,
same JSON separators (see :func:`pyeidors.cache.process_lru.hash_json_payload`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..cache.process_lru import (
    ProcessLRUCache,
    env_bytes_limit,
    hash_json_payload,
    path_signature,
)
from ..data.structures import EITMesh

_PROCESS_MESH_CACHE_MAX_ITEMS = 8
_PROCESS_MESH_CACHE_MAX_BYTES = 512 * 1024 * 1024


def _array_nbytes(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(np.asarray(value).nbytes)
    except Exception:
        return 0


def _mesh_cache_size_bytes(mesh: EITMesh) -> int:
    total = _array_nbytes(getattr(getattr(mesh, "geometry", None), "x", None))
    for tag_name in ("facet_tags", "cell_tags"):
        tags = getattr(mesh, tag_name, None)
        total += _array_nbytes(getattr(tags, "indices", None))
        total += _array_nbytes(getattr(tags, "values", None))
    for vertices in getattr(mesh, "electrode_vertices", None) or ():
        total += _array_nbytes(vertices)
    derived = getattr(mesh, "_derived_arrays", None)
    for name in ("node_coords", "cell_connectivity", "cell_centers", "cell_measures"):
        total += _array_nbytes(getattr(derived, name, None))
    return max(int(total), 1)


_RESOLVED_PROCESS_MESH_CACHE_MAX_BYTES = env_bytes_limit(
    "PYEIDORS_PROCESS_MESH_CACHE_MAX_BYTES",
    "EIT_APP_PROCESS_MESH_CACHE_MAX_BYTES",
    default=_PROCESS_MESH_CACHE_MAX_BYTES,
)
_PROCESS_MESH_CACHE: ProcessLRUCache[EITMesh] = ProcessLRUCache(
    max_items=_PROCESS_MESH_CACHE_MAX_ITEMS,
    max_bytes=_RESOLVED_PROCESS_MESH_CACHE_MAX_BYTES,
    sizeof=_mesh_cache_size_bytes,
)


def build_process_mesh_cache_key(
    *,
    mesh_file: str | Path,
    gdim: int,
    n_elec: int | None = None,
    association_file: str | Path | None = None,
    sidecar_file: str | Path | None = None,
    extra_files: Sequence[str | Path] | None = None,
    mesh_name: str | None = None,
    geometry_dtype: Any | None = None,
) -> str:
    mesh_path = Path(mesh_file)
    payload: dict[str, object] = {
        "mesh_file": str(mesh_path.resolve()),
        "mesh_sig": path_signature(mesh_path),
        "gdim": int(gdim),
        "mesh_name": str(mesh_name or mesh_path.stem),
    }
    if geometry_dtype is not None:
        payload["geometry_dtype"] = str(np.dtype(geometry_dtype))
    if n_elec is not None:
        payload["n_elec"] = int(n_elec)
    if association_file is not None:
        payload["association_file"] = str(Path(association_file).resolve())
        payload["association_sig"] = path_signature(association_file)
    if sidecar_file is not None:
        payload["sidecar_file"] = str(Path(sidecar_file).resolve())
        payload["sidecar_sig"] = path_signature(sidecar_file)
    if extra_files:
        payload["extra_files"] = [
            {
                "file": str(Path(extra_file).resolve()),
                "sig": path_signature(extra_file),
            }
            for extra_file in sorted(extra_files, key=lambda item: str(item))
        ]
    return hash_json_payload(payload)


def get_process_cached_mesh(key: str) -> EITMesh | None:
    return _PROCESS_MESH_CACHE.get(key)


def put_process_cached_mesh(key: str, mesh: EITMesh) -> None:
    max_bytes = int(_RESOLVED_PROCESS_MESH_CACHE_MAX_BYTES)
    if max_bytes <= 0 or _mesh_cache_size_bytes(mesh) > max_bytes:
        _PROCESS_MESH_CACHE.discard(key)
        return
    _PROCESS_MESH_CACHE.put(key, mesh)


def clear_process_mesh_cache() -> None:
    _PROCESS_MESH_CACHE.clear()


def process_mesh_cache_stats() -> dict[str, int]:
    return _PROCESS_MESH_CACHE.stats()
