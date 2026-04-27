"""Shared process-local LRU cache primitives.

T79 Path C consolidation: the historical
:mod:`pyeidors.geometry.process_mesh_cache` and
:mod:`pyeidors.forward.process_setup_cache` both implemented the same
``OrderedDict`` + ``threading.Lock`` + ``max_items`` LRU pattern with
SHA256-of-JSON cache key construction. This module factors that pattern
out so the two callers shrink to thin wrappers that pin a value type
and a default item budget. The persistent file-backed cache layers
(:mod:`pyeidors.geometry.dolfinx_mesh_cache` for XDMF/HDF5,
:mod:`pyeidors.geometry.adios4dolfinx_checkpoint` for ADIOS2) are
deliberately *not* unified here — their bodies legitimately diverge
along disk-format lines and forcing a single ``MeshCacheLayer``
interface across in-memory + disk would be premature abstraction.

The cache-key payload contract (V36 / V62 / V65 / V66 / V67) and the
forward setup cache-key contract (V16 / V17) are preserved bytewise:
:func:`hash_json_payload` always uses ``sort_keys=True``,
``separators=(",", ":")``, ``ensure_ascii=True`` so a payload that
hashed to ``H`` before the consolidation still hashes to ``H``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Generic, Mapping, TypeVar

V = TypeVar("V")


def hash_json_payload(payload: Mapping[str, object]) -> str:
    """SHA256-hex of a deterministic JSON encoding of ``payload``.

    The encoding fixes ``sort_keys=True``, ``separators=(",", ":")`` and
    ``ensure_ascii=True`` so the resulting digest is bytewise identical
    to the historical inline implementations in
    ``process_mesh_cache.build_process_mesh_cache_key`` and
    ``process_setup_cache.build_process_forward_setup_key``.
    """
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def path_signature(path: str | Path) -> str:
    """Cheap process-cache signature for a filesystem path.

    Returns ``"<resolved>::<size>::<mtime_ns>::<is_dir>"`` when the
    path exists, otherwise the raw string form. Used by mesh cache
    keys to invalidate when an artifact on disk changes without
    requiring a full content hash.
    """
    mesh_path = Path(path)
    try:
        stat = mesh_path.stat()
        return (
            f"{mesh_path.resolve()}::{stat.st_size}::"
            f"{stat.st_mtime_ns}::{int(mesh_path.is_dir())}"
        )
    except OSError:
        return str(mesh_path)


class ProcessLRUCache(Generic[V]):
    """Thread-safe in-process LRU cache with a fixed item budget.

    Mirrors the historical pattern (``OrderedDict`` LRU +
    ``threading.Lock`` + ``max_items`` eviction) that
    :mod:`pyeidors.geometry.process_mesh_cache` and
    :mod:`pyeidors.forward.process_setup_cache` each implemented
    independently. Concrete callers wrap an instance of this class
    behind module-level ``get_*`` / ``put_*`` / ``clear_*`` functions
    so existing public callsites stay unchanged.
    """

    __slots__ = ("_cache", "_lock", "_max_items")

    def __init__(self, *, max_items: int = 8) -> None:
        if int(max_items) <= 0:
            raise ValueError(f"max_items must be positive, got {max_items!r}")
        self._cache: "OrderedDict[str, V]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_items = int(max_items)

    @property
    def max_items(self) -> int:
        return self._max_items

    def get(self, key: str) -> V | None:
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                return None
            self._cache.move_to_end(key)
            return value

    def put(self, key: str, value: V) -> None:
        with self._lock:
            self._cache.pop(key, None)
            self._cache[key] = value
            while len(self._cache) > self._max_items:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"items": len(self._cache), "max_items": self._max_items}
