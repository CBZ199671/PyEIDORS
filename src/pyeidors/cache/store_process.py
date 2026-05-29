"""In-process cache store with EIDORS-style score-aware eviction."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import sys
import threading
import time
from typing import Any, Mapping

import numpy as np

from .index_fields import CACHE_INDEX_FIELD_NAMES, normalize_cache_index_fields
from .types import compute_score_eff, compute_score_size


def estimate_object_size_bytes(value: Any) -> int:
    """Estimate object size in bytes for eviction decisions."""

    return _estimate_object_size_bytes(value, seen=set())


def _estimate_object_size_bytes(value: Any, *, seen: set[int]) -> int:
    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return int(len(value))
    if isinstance(value, (str, int, float, bool, type(None))):
        return 64
    if isinstance(value, dict):
        return 96 + sum(
            _estimate_object_size_bytes(k, seen=seen)
            + _estimate_object_size_bytes(v, seen=seen)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return 96 + sum(_estimate_object_size_bytes(v, seen=seen) for v in value)

    explicit_nbytes = getattr(value, "nbytes", None)
    if isinstance(explicit_nbytes, (int, np.integer)) and int(explicit_nbytes) >= 0:
        return int(explicit_nbytes)

    sparse_payload_size = _estimate_sparse_like_size_bytes(value)
    if sparse_payload_size > 0:
        return sparse_payload_size

    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, dict) and attrs:
        return 96 + sum(
            _estimate_object_size_bytes(v, seen=seen) for v in attrs.values()
        )

    return max(1024, int(sys.getsizeof(value, 1024)))


def _estimate_sparse_like_size_bytes(value: Any) -> int:
    total = 0
    found = False
    for attr in ("data", "indices", "indptr", "row", "col"):
        arr = getattr(value, attr, None)
        if isinstance(arr, np.ndarray):
            total += int(arr.nbytes)
            found = True
    if not found:
        return 0
    shape = getattr(value, "shape", None)
    ndim_cost = 16 * len(shape) if isinstance(shape, tuple) else 0
    return int(128 + ndim_cost + total)


@dataclass
class _Entry:
    value: Any
    size: int
    artifact: str
    name: str
    namespace: str
    cost: float
    effort: float
    priority: float
    use_count: int
    score_eff: float
    score_size: float
    score: float
    created_at: float
    last_access: float
    dtype: str | None = None
    backend: str | None = None
    device: str | None = None
    dim: int | None = None
    n_elec: int | None = None
    mesh_hash: str | None = None


def _eviction_key(entry: _Entry) -> tuple[float, float, float]:
    """Evict lower-priority entries first.

    Retention rank follows EIDORS-style:
    (-score_eff, score_size, -last_access) => higher priority first.
    Therefore eviction key is the inverse:
    (score_eff, -score_size, last_access) => lower priority first.
    """
    return (entry.score_eff, -entry.score_size, entry.last_access)


class ProcessCacheStore:
    """Thread-safe process cache with score-aware eviction."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = int(max(0, max_bytes))
        self._items: OrderedDict[str, _Entry] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.admission_rejections = 0
        self.admission_rejected_bytes = 0
        self.admission_rejection_reasons: dict[str, int] = {}

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            entry.last_access = now
            entry.use_count += 1
            entry.score_eff = compute_score_eff(
                effort=entry.effort,
                use_count=entry.use_count,
                priority=entry.priority,
            )
            entry.score = entry.score_eff
            self.hits += 1
            return entry.value

    def put(
        self,
        key: str,
        value: Any,
        *,
        artifact: str,
        cost: float,
        name: str = "",
        namespace: str = "default",
        effort: float | None = None,
        priority: float = 0.0,
        index_fields: Mapping[str, Any] | None = None,
    ) -> bool:
        size = estimate_object_size_bytes(value)
        now = time.time()
        use_effort = float(cost if effort is None else effort)
        index = normalize_cache_index_fields(index_fields)
        score_eff = compute_score_eff(
            effort=use_effort,
            use_count=1,
            priority=float(priority),
        )
        score_size = compute_score_size(size)
        candidate = _Entry(
            value=value,
            size=size,
            artifact=artifact,
            name=str(name),
            namespace=str(namespace),
            cost=float(cost),
            effort=use_effort,
            priority=float(priority),
            use_count=1,
            score_eff=score_eff,
            score_size=score_size,
            score=score_eff,
            created_at=now,
            last_access=now,
            dtype=index["dtype"] if isinstance(index["dtype"], str) else None,
            backend=index["backend"] if isinstance(index["backend"], str) else None,
            device=index["device"] if isinstance(index["device"], str) else None,
            dim=index["dim"] if isinstance(index["dim"], int) else None,
            n_elec=index["n_elec"] if isinstance(index["n_elec"], int) else None,
            mesh_hash=(
                index["mesh_hash"] if isinstance(index["mesh_hash"], str) else None
            ),
        )
        with self._lock:
            if self.max_bytes <= 0:
                self._items.clear()
                self._total_bytes = 0
                self._record_admission_rejection(
                    reason="process_cache_disabled",
                    size=size,
                )
                return False
            if size > self.max_bytes:
                self._record_admission_rejection(
                    reason="entry_too_large",
                    size=size,
                )
                return False
            if not self._candidate_survives_admission(key, candidate):
                self._record_admission_rejection(
                    reason="would_evict_immediately",
                    size=size,
                )
                return False
            existing = self._items.pop(key, None)
            if existing is not None:
                self._total_bytes -= existing.size
            self._items[key] = candidate
            self._total_bytes += size
            self._evict_if_needed()
            return True

    def _record_admission_rejection(self, *, reason: str, size: int) -> None:
        self.admission_rejections += 1
        self.admission_rejected_bytes += int(max(0, size))
        self.admission_rejection_reasons[reason] = (
            self.admission_rejection_reasons.get(reason, 0) + 1
        )

    def _candidate_survives_admission(self, key: str, candidate: _Entry) -> bool:
        existing = self._items.get(key)
        total = self._total_bytes + candidate.size
        if existing is not None:
            total -= existing.size
        if total <= self.max_bytes:
            return True
        pending: list[tuple[str, _Entry]] = [
            (item_key, entry)
            for item_key, entry in self._items.items()
            if item_key != key
        ]
        pending.append((key, candidate))
        pending_total = sum(entry.size for _item_key, entry in pending)
        for evict_key, evicted in sorted(
            pending,
            key=lambda item: _eviction_key(item[1]),
        ):
            if pending_total <= self.max_bytes:
                break
            if evict_key == key:
                return False
            pending_total -= evicted.size
        return True

    def _evict_if_needed(self) -> None:
        if self.max_bytes <= 0:
            self._items.clear()
            self._total_bytes = 0
            return
        while self._total_bytes > self.max_bytes and self._items:
            evict_key, evicted = min(
                self._items.items(),
                key=lambda item: _eviction_key(item[1]),
            )
            self._items.pop(evict_key, None)
            self._total_bytes -= evicted.size

    def invalidate(self, prefix: str = "") -> int:
        removed = 0
        with self._lock:
            keys = list(self._items.keys())
            for key in keys:
                if not prefix or key.startswith(prefix):
                    entry = self._items.pop(key)
                    self._total_bytes -= entry.size
                    removed += 1
        return removed

    def clear_name(self, name: str, namespace: str | None = None) -> int:
        removed = 0
        with self._lock:
            keys = list(self._items.keys())
            for key in keys:
                entry = self._items[key]
                if entry.name != name:
                    continue
                if namespace is not None and entry.namespace != namespace:
                    continue
                self._items.pop(key, None)
                self._total_bytes -= entry.size
                removed += 1
        return removed

    def clear_max(self, max_bytes: int) -> int:
        target = int(max(0, max_bytes))
        removed = 0
        with self._lock:
            while self._total_bytes > target and self._items:
                evict_key, evicted = min(
                    self._items.items(),
                    key=lambda item: _eviction_key(item[1]),
                )
                self._items.pop(evict_key, None)
                self._total_bytes -= evicted.size
                removed += 1
        return removed

    def clear_old(self, timestamp: float) -> int:
        removed = 0
        ts = float(timestamp)
        with self._lock:
            for key in list(self._items.keys()):
                if self._items[key].last_access < ts:
                    entry = self._items.pop(key)
                    self._total_bytes -= entry.size
                    removed += 1
        return removed

    def clear_new(self, timestamp: float) -> int:
        removed = 0
        ts = float(timestamp)
        with self._lock:
            for key in list(self._items.keys()):
                if self._items[key].last_access > ts:
                    entry = self._items.pop(key)
                    self._total_bytes -= entry.size
                    removed += 1
        return removed

    def get_value(self, key: str) -> Any | None:
        with self._lock:
            entry = self._items.get(key)
            if entry is None:
                return None
            return entry.value

    def list_entries(
        self,
        *,
        name: str | None = None,
        namespace: str | None = None,
        limit: int | None = None,
        dtype: str | None = None,
        backend: str | None = None,
        device: str | None = None,
        dim: int | None = None,
        n_elec: int | None = None,
        mesh_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        index_filter = normalize_cache_index_fields(
            {
                "dtype": dtype,
                "backend": backend,
                "device": device,
                "dim": dim,
                "n_elec": n_elec,
                "mesh_hash": mesh_hash,
            }
        )
        with self._lock:
            entries = []
            for key, entry in self._items.items():
                if name is not None and entry.name != name:
                    continue
                if namespace is not None and entry.namespace != namespace:
                    continue
                if not _entry_matches_index_filter(entry, index_filter):
                    continue
                entries.append(
                    {
                        "key": key,
                        "artifact": entry.artifact,
                        "name": entry.name,
                        "namespace": entry.namespace,
                        "size_bytes": entry.size,
                        "cost": entry.cost,
                        "effort": entry.effort,
                        "priority": entry.priority,
                        "use_count": entry.use_count,
                        "score_eff": entry.score_eff,
                        "score_size": entry.score_size,
                        "score": entry.score,
                        "created_at": entry.created_at,
                        "last_access": entry.last_access,
                        "layer": "process",
                        "dtype": entry.dtype,
                        "backend": entry.backend,
                        "device": entry.device,
                        "dim": entry.dim,
                        "n_elec": entry.n_elec,
                        "mesh_hash": entry.mesh_hash,
                    }
                )
            entries.sort(key=lambda item: item["last_access"], reverse=True)
            if limit is not None and limit > 0:
                entries = entries[:limit]
            return entries

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._total_bytes = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            artifacts: dict[str, int] = {}
            namespaces: dict[str, int] = {}
            for entry in self._items.values():
                artifacts[entry.artifact] = artifacts.get(entry.artifact, 0) + 1
                namespaces[entry.namespace] = namespaces.get(entry.namespace, 0) + 1
            return {
                "hits": self.hits,
                "misses": self.misses,
                "items": len(self._items),
                "bytes": int(self._total_bytes),
                "max_bytes": int(self.max_bytes),
                "admission_rejections": int(self.admission_rejections),
                "admission_rejected_bytes": int(self.admission_rejected_bytes),
                "admission_rejection_reasons": dict(self.admission_rejection_reasons),
                "artifacts": artifacts,
                "namespaces": namespaces,
            }


def _entry_matches_index_filter(
    entry: _Entry, index_filter: Mapping[str, str | int | None]
) -> bool:
    for field in CACHE_INDEX_FIELD_NAMES:
        expected = index_filter.get(field)
        if expected is None:
            continue
        if getattr(entry, field) != expected:
            return False
    return True
