"""Typed definitions shared by cache stores and cache manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

DEFAULT_CACHE_LIFECYCLE = "session"


def compute_score_eff(*, effort: float, use_count: int, priority: float) -> float:
    """EIDORS-style efficiency score: round(10*log10(effort*count)) + priority."""
    scaled_effort = max(effort, 1e-9)
    return float(round(10.0 * np.log10(scaled_effort * max(use_count, 1))) + priority)


def compute_score_size(size_bytes: int) -> float:
    """EIDORS-style size score: round(10*log10(size_bytes/1024))."""
    scaled_size = max(size_bytes, 1)
    return float(round(10.0 * np.log10(float(scaled_size) / 1024.0)))

CacheDiskLifecycle = Literal["session", "persistent"]


def normalize_cache_lifecycle(
    value: object,
    *,
    default: CacheDiskLifecycle = DEFAULT_CACHE_LIFECYCLE,
) -> CacheDiskLifecycle:
    normalized = str(value).strip().lower() if value is not None else str(default)
    if normalized in {"session", "persistent"}:
        return normalized
    return default

CacheScope = Literal["process", "disk", "both", "off"]
CacheArtifactKind = Literal[
    "mesh_bundle",
    "pattern_bundle",
    "forward_factor",
    "jacobian",
    "single_step_operator",
    "sparse_basis",
    "measurement_projection",
]


@dataclass(frozen=True)
class CachePolicy:
    """Global policy controls for process/disk caches."""

    process_max_bytes: int = 3 * 1024**3
    disk_max_bytes: int = 20 * 1024**3
    ttl_seconds: float | None = None
    compress_disk_payloads: bool = True
    read_only: bool = False
    disk_lifecycle: CacheDiskLifecycle = DEFAULT_CACHE_LIFECYCLE
    cleanup_on_exit: bool = True
    cleanup_stale_sessions_on_startup: bool = True
    stale_session_max_age_seconds: float = 7 * 24 * 60 * 60
    artifact_cost: dict[str, float] = field(
        default_factory=lambda: {
            "forward_factor": 16.0,
            "jacobian": 12.0,
            "single_step_operator": 10.0,
            "sparse_basis": 8.0,
            "measurement_projection": 4.0,
            "pattern_bundle": 3.0,
            "mesh_bundle": 2.0,
        }
    )


@dataclass(frozen=True)
class CacheLookup:
    """Metadata returned by ``CacheManager.get_or_compute``."""

    key: str
    hit: bool
    layer: Literal["process", "disk", "compute", "disabled"]
    artifact: str


@dataclass
class CacheStats:
    """Runtime counters for cache lookups and storage usage."""

    process_hits: int = 0
    process_misses: int = 0
    disk_hits: int = 0
    disk_misses: int = 0
    process_items: int = 0
    process_bytes: int = 0
    process_max_bytes: int = 0
    disk_items: int = 0
    disk_bytes: int = 0
    disk_max_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        total_hits = self.process_hits + self.disk_hits
        total_misses = self.process_misses + self.disk_misses
        return {
            "process_hits": self.process_hits,
            "process_misses": self.process_misses,
            "disk_hits": self.disk_hits,
            "disk_misses": self.disk_misses,
            "process_items": self.process_items,
            "process_bytes": self.process_bytes,
            "process_max_bytes": self.process_max_bytes,
            "disk_items": self.disk_items,
            "disk_bytes": self.disk_bytes,
            "disk_max_bytes": self.disk_max_bytes,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": total_hits / max(1, total_hits + total_misses),
        }
