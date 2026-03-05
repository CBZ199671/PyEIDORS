"""In-process cache store with EIDORS-style score-aware eviction."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import pickle
import threading
import time
from typing import Any

import numpy as np


def estimate_object_size_bytes(value: Any) -> int:
    """Estimate object size in bytes for eviction decisions."""

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return int(len(value))
    if isinstance(value, (str, int, float, bool, type(None))):
        return 64
    if isinstance(value, dict):
        return 96 + sum(estimate_object_size_bytes(k) + estimate_object_size_bytes(v) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return 96 + sum(estimate_object_size_bytes(v) for v in value)
    try:
        return int(len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)))
    except Exception:
        return 1024


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


def _compute_score_eff(*, effort: float, use_count: int, priority: float) -> float:
    """Mirrors EIDORS score_eff = round(10*log10(effort*count)) + priority."""
    scaled_effort = max(float(effort), 1e-9)
    return float(round(10.0 * np.log10(scaled_effort * max(int(use_count), 1))) + float(priority))


def _compute_score_size(size_bytes: int) -> float:
    """Mirrors EIDORS score_sz = round(10*log10(size_bytes/1024))."""
    scaled_size = max(int(size_bytes), 1)
    return float(round(10.0 * np.log10(float(scaled_size) / 1024.0)))


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
            entry.score_eff = _compute_score_eff(
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
    ) -> None:
        size = estimate_object_size_bytes(value)
        now = time.time()
        use_effort = float(cost if effort is None else effort)
        score_eff = _compute_score_eff(
            effort=use_effort,
            use_count=1,
            priority=float(priority),
        )
        score_size = _compute_score_size(size)
        with self._lock:
            existing = self._items.pop(key, None)
            if existing is not None:
                self._total_bytes -= existing.size
            self._items[key] = _Entry(
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
            )
            self._total_bytes += size
            self._evict_if_needed()

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
    ) -> list[dict[str, Any]]:
        with self._lock:
            entries = []
            for key, entry in self._items.items():
                if name is not None and entry.name != name:
                    continue
                if namespace is not None and entry.namespace != namespace:
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
                "artifacts": artifacts,
                "namespaces": namespaces,
            }
