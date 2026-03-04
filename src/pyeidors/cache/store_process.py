"""In-process LRU cache store."""

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
    cost: float
    created_at: float
    last_access: float


class ProcessCacheStore:
    """Simple thread-safe LRU cache with max-byte eviction."""

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
            self.hits += 1
            return entry.value

    def put(self, key: str, value: Any, *, artifact: str, cost: float) -> None:
        size = estimate_object_size_bytes(value)
        now = time.time()
        with self._lock:
            existing = self._items.pop(key, None)
            if existing is not None:
                self._total_bytes -= existing.size
            self._items[key] = _Entry(
                value=value,
                size=size,
                artifact=artifact,
                cost=float(cost),
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
            _, evicted = self._items.popitem(last=False)
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

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._total_bytes = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "items": len(self._items),
                "bytes": int(self._total_bytes),
                "max_bytes": int(self.max_bytes),
            }

