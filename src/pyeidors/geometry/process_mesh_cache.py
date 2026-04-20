"""Process-local cache for loaded EIT mesh objects."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

from ..data.structures import EITMesh

_PROCESS_MESH_CACHE_MAX_ITEMS = 8
_PROCESS_MESH_CACHE: OrderedDict[str, EITMesh] = OrderedDict()
_PROCESS_MESH_CACHE_LOCK = threading.Lock()


def _path_signature(path: str | Path) -> str:
    """Return a cheap process-cache signature for large mesh artifacts."""
    mesh_path = Path(path)
    try:
        stat = mesh_path.stat()
        return (
            f"{mesh_path.resolve()}::{stat.st_size}::"
            f"{stat.st_mtime_ns}::{int(mesh_path.is_dir())}"
        )
    except OSError:
        return str(mesh_path)


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
        "mesh_sig": _path_signature(mesh_path),
        "gdim": int(gdim),
        "mesh_name": str(mesh_name or mesh_path.stem),
    }
    if n_elec is not None:
        payload["n_elec"] = int(n_elec)
    if association_file is not None:
        payload["association_file"] = str(Path(association_file).resolve())
        payload["association_sig"] = _path_signature(association_file)
    if sidecar_file is not None:
        payload["sidecar_file"] = str(Path(sidecar_file).resolve())
        payload["sidecar_sig"] = _path_signature(sidecar_file)
    if extra_files:
        payload["extra_files"] = [
            {
                "file": str(Path(extra_file).resolve()),
                "sig": _path_signature(extra_file),
            }
            for extra_file in sorted(extra_files, key=lambda item: str(item))
        ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def get_process_cached_mesh(key: str) -> EITMesh | None:
    with _PROCESS_MESH_CACHE_LOCK:
        mesh = _PROCESS_MESH_CACHE.get(key)
        if mesh is None:
            return None
        _PROCESS_MESH_CACHE.move_to_end(key)
        return mesh


def put_process_cached_mesh(key: str, mesh: EITMesh) -> None:
    with _PROCESS_MESH_CACHE_LOCK:
        _PROCESS_MESH_CACHE.pop(key, None)
        _PROCESS_MESH_CACHE[key] = mesh
        while len(_PROCESS_MESH_CACHE) > _PROCESS_MESH_CACHE_MAX_ITEMS:
            _PROCESS_MESH_CACHE.popitem(last=False)


def clear_process_mesh_cache() -> None:
    with _PROCESS_MESH_CACHE_LOCK:
        _PROCESS_MESH_CACHE.clear()
