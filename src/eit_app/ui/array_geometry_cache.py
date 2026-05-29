"""Process-local derived geometry cache for GUI NumPy arrays."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
import threading

import numpy as np

from pyeidors.cache.keys import update_digest_with_array_payload


ARRAY_GEOMETRY_CACHE_SCHEMA = "eit-app-array-geometry-v1"
_MAX_ITEMS_ENV = "EIT_APP_ARRAY_GEOMETRY_CACHE_ITEMS"
_MAX_BYTES_ENV = "EIT_APP_ARRAY_GEOMETRY_CACHE_MAX_BYTES"
_DEFAULT_MAX_ITEMS = 8
_DEFAULT_MAX_BYTES = 64 * 1024 * 1024
_CACHE_LOCK = threading.RLock()
_CACHE: OrderedDict[str, "ArrayGeometryDerived"] = OrderedDict()
_STATS = {
    "hits": 0,
    "misses": 0,
    "invalid": 0,
    "evictions": 0,
    "oversize": 0,
}


@dataclass(frozen=True)
class ArrayGeometryDerived:
    """Derived geometry arrays keyed by node/cell array content."""

    signature: str
    coordinate_dims: int
    node_shape: tuple[int, ...]
    node_dtype: str
    cell_shape: tuple[int, ...]
    cell_dtype: str
    cell_centers: np.ndarray


def _max_items() -> int:
    raw = os.environ.get(_MAX_ITEMS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_ITEMS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_ITEMS
    return min(max(value, 1), 128)


def _max_bytes() -> int:
    raw = os.environ.get(_MAX_BYTES_ENV, "").strip().lower()
    if raw in {"0", "false", "no", "off", "none", "disabled"}:
        return 0
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        value = int(float(raw))
    except ValueError:
        return _DEFAULT_MAX_BYTES
    return min(max(value, 1), 16 * 1024 * 1024 * 1024)


def _cache_bytes_locked() -> int:
    return int(sum(item.cell_centers.nbytes for item in _CACHE.values()))


def _trim_cache_locked() -> None:
    max_items = _max_items()
    max_bytes = _max_bytes()
    while len(_CACHE) > max_items:
        _CACHE.popitem(last=False)
        _STATS["evictions"] += 1
    if max_bytes <= 0:
        return
    while len(_CACHE) > 1 and _cache_bytes_locked() > max_bytes:
        _CACHE.popitem(last=False)
        _STATS["evictions"] += 1


def _as_hashable_arrays(
    node_coords: np.ndarray | None,
    cell_connectivity: np.ndarray | None,
    *,
    coordinate_dims: int | None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if node_coords is None or cell_connectivity is None:
        return None
    try:
        coords_raw = np.asarray(node_coords)
        cells_raw = np.asarray(cell_connectivity)
    except Exception:
        return None
    if coords_raw.ndim != 2 or cells_raw.ndim != 2:
        return None
    if coords_raw.size == 0 or cells_raw.size == 0:
        return None
    if coords_raw.shape[1] <= 0:
        return None

    if coordinate_dims is None:
        dims = int(coords_raw.shape[1])
    else:
        try:
            dims = int(coordinate_dims)
        except (TypeError, ValueError):
            return None
        dims = min(dims, int(coords_raw.shape[1]))
    if dims <= 0:
        return None

    if np.issubdtype(coords_raw.dtype, np.floating):
        coords = np.asarray(coords_raw[:, :dims])
    else:
        coords = np.asarray(coords_raw[:, :dims], dtype=np.float64)
    if np.issubdtype(cells_raw.dtype, np.integer):
        cells = np.asarray(cells_raw)
    else:
        cells = np.asarray(cells_raw, dtype=np.intp)

    if int(cells.min()) < 0 or int(cells.max()) >= coords.shape[0]:
        return None
    return coords, cells, dims


def _hash_array_into(hasher: "hashlib._Hash", array: np.ndarray) -> None:
    hasher.update(str(array.dtype).encode("ascii", "replace"))
    update_digest_with_array_payload(hasher, np.asarray(array.shape, dtype=np.int64))
    update_digest_with_array_payload(hasher, array)


def _signature_for_arrays(coords: np.ndarray, cells: np.ndarray, dims: int) -> str:
    hasher = hashlib.sha256()
    hasher.update(ARRAY_GEOMETRY_CACHE_SCHEMA.encode("ascii"))
    update_digest_with_array_payload(hasher, np.asarray([dims], dtype=np.int64))
    _hash_array_into(hasher, coords)
    _hash_array_into(hasher, cells)
    return hasher.hexdigest()


def _compute_cell_centers(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Compute centers without materializing ``coords[cells]``."""

    n_cells, vertices_per_cell = cells.shape
    dims = coords.shape[1]
    dtype = np.result_type(coords.dtype, np.float32)
    source = np.asarray(coords, dtype=dtype)
    centers = np.zeros((n_cells, dims), dtype=dtype)
    work = np.empty((n_cells, dims), dtype=dtype)
    for local_idx in range(vertices_per_cell):
        np.take(source, cells[:, local_idx], axis=0, out=work)
        centers += work
    centers /= float(vertices_per_cell)
    centers = np.ascontiguousarray(centers)
    centers.setflags(write=False)
    return centers


def array_geometry_signature(
    node_coords: np.ndarray | None,
    cell_connectivity: np.ndarray | None,
    *,
    coordinate_dims: int | None = None,
) -> str | None:
    arrays = _as_hashable_arrays(
        node_coords,
        cell_connectivity,
        coordinate_dims=coordinate_dims,
    )
    if arrays is None:
        return None
    coords, cells, dims = arrays
    return _signature_for_arrays(coords, cells, dims)


def cached_array_geometry(
    node_coords: np.ndarray | None,
    cell_connectivity: np.ndarray | None,
    *,
    coordinate_dims: int | None = None,
) -> ArrayGeometryDerived | None:
    arrays = _as_hashable_arrays(
        node_coords,
        cell_connectivity,
        coordinate_dims=coordinate_dims,
    )
    if arrays is None:
        with _CACHE_LOCK:
            _STATS["invalid"] += 1
        return None
    coords, cells, dims = arrays

    signature = _signature_for_arrays(coords, cells, dims)

    with _CACHE_LOCK:
        cached = _CACHE.get(signature)
        if cached is not None:
            _CACHE.move_to_end(signature)
            _STATS["hits"] += 1
            return cached
        _STATS["misses"] += 1

    cell_centers = _compute_cell_centers(coords, cells)
    derived = ArrayGeometryDerived(
        signature=signature,
        coordinate_dims=dims,
        node_shape=tuple(int(v) for v in coords.shape),
        node_dtype=str(coords.dtype),
        cell_shape=tuple(int(v) for v in cells.shape),
        cell_dtype=str(cells.dtype),
        cell_centers=cell_centers,
    )

    with _CACHE_LOCK:
        existing = _CACHE.get(signature)
        if existing is not None:
            _CACHE.move_to_end(signature)
            _STATS["hits"] += 1
            return existing
        max_bytes = _max_bytes()
        if max_bytes > 0 and int(derived.cell_centers.nbytes) > max_bytes:
            _STATS["oversize"] += 1
            return derived
        _CACHE[signature] = derived
        _trim_cache_locked()
    return derived


def cached_cell_centers(
    node_coords: np.ndarray | None,
    cell_connectivity: np.ndarray | None,
    *,
    coordinate_dims: int | None = None,
) -> np.ndarray | None:
    derived = cached_array_geometry(
        node_coords,
        cell_connectivity,
        coordinate_dims=coordinate_dims,
    )
    if derived is None:
        return None
    return derived.cell_centers


def clear_array_geometry_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
        for key in _STATS:
            _STATS[key] = 0


def array_geometry_cache_stats() -> dict[str, int]:
    with _CACHE_LOCK:
        return {
            "items": len(_CACHE),
            "max_items": _max_items(),
            "max_bytes": _max_bytes(),
            "bytes": _cache_bytes_locked(),
            **{key: int(value) for key, value in _STATS.items()},
        }


def array_geometry_cache_entries(limit: int = 16) -> list[dict[str, object]]:
    """Return lightweight metadata for cached entries, newest first."""

    try:
        max_entries = max(0, int(limit))
    except (TypeError, ValueError):
        max_entries = 16
    with _CACHE_LOCK:
        rows = list(reversed(_CACHE.values()))[:max_entries]
        return [
            {
                "signature": item.signature,
                "signature_prefix": item.signature[:16],
                "coordinate_dims": item.coordinate_dims,
                "node_shape": item.node_shape,
                "node_dtype": item.node_dtype,
                "cell_shape": item.cell_shape,
                "cell_dtype": item.cell_dtype,
                "cell_centers_bytes": int(item.cell_centers.nbytes),
            }
            for item in rows
        ]


def array_geometry_cache_snapshot(limit: int = 16) -> dict[str, object]:
    """Return JSON-safe stats and entry metadata for this Python process."""

    return {
        "schema": ARRAY_GEOMETRY_CACHE_SCHEMA,
        "process_local": True,
        "stats": array_geometry_cache_stats(),
        "entries": array_geometry_cache_entries(limit=limit),
    }
