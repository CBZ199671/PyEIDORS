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
from typing import Sequence

from ..cache.process_lru import ProcessLRUCache, hash_json_payload, path_signature
from ..data.structures import EITMesh

_PROCESS_MESH_CACHE_MAX_ITEMS = 8
_PROCESS_MESH_CACHE: ProcessLRUCache[EITMesh] = ProcessLRUCache(
    max_items=_PROCESS_MESH_CACHE_MAX_ITEMS
)


def _path_signature(path: str | Path) -> str:
    """Backward-compat alias for :func:`pyeidors.cache.process_lru.path_signature`."""
    return path_signature(path)


def build_process_mesh_cache_key(
    *,
    mesh_file: str | Path,
    gdim: int,
    n_elec: int | None = None,
    association_file: str | Path | None = None,
    sidecar_file: str | Path | None = None,
    extra_files: Sequence[str | Path] | None = None,
    mesh_name: str | None = None,
) -> str:
    mesh_path = Path(mesh_file)
    payload: dict[str, object] = {
        "mesh_file": str(mesh_path.resolve()),
        "mesh_sig": path_signature(mesh_path),
        "gdim": int(gdim),
        "mesh_name": str(mesh_name or mesh_path.stem),
    }
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
    _PROCESS_MESH_CACHE.put(key, mesh)


def clear_process_mesh_cache() -> None:
    _PROCESS_MESH_CACHE.clear()
