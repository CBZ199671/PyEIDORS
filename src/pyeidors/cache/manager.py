"""Unified multi-layer cache manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .keys import CacheKeyParts, build_cache_key
from .store_disk import DiskCacheStore
from .store_process import ProcessCacheStore, estimate_object_size_bytes
from .types import CacheLookup, CachePolicy, CacheScope, CacheStats


class CacheManager:
    """Coordinate process/disk caches with stable key generation."""

    def __init__(
        self,
        *,
        scope: CacheScope = "both",
        cache_dir: str | Path = ".pyeidors_cache/v2",
        policy: CachePolicy | None = None,
        code_fingerprint: str = "unknown",
    ) -> None:
        self.scope: CacheScope = scope
        self.policy = policy or CachePolicy()
        self.code_fingerprint = code_fingerprint
        self.cache_dir = Path(cache_dir)

        self._process = (
            ProcessCacheStore(self.policy.process_max_bytes)
            if scope in {"process", "both"}
            else None
        )
        self._disk = (
            DiskCacheStore(
                self.cache_dir,
                max_bytes=self.policy.disk_max_bytes,
                compress_payloads=self.policy.compress_disk_payloads,
                read_only=self.policy.read_only,
                default_ttl_seconds=self.policy.ttl_seconds,
            )
            if scope in {"disk", "both"}
            else None
        )

    @property
    def enabled(self) -> bool:
        return self.scope != "off"

    def build_key(self, artifact: str, payload: dict[str, Any], namespace: str = "default") -> str:
        return build_cache_key(
            CacheKeyParts(
                artifact=artifact,
                payload=payload,
                schema_version=2,
                code_fingerprint=self.code_fingerprint,
                namespace=namespace,
            )
        )

    def get_or_compute(
        self,
        *,
        artifact: str,
        payload: dict[str, Any],
        compute_fn: Callable[[], Any],
        namespace: str = "default",
        cost: float | None = None,
        ttl_seconds: float | None = None,
        persist: bool = True,
    ) -> tuple[Any, CacheLookup]:
        """Retrieve from cache or compute and populate layers."""

        cache_key = self.build_key(artifact, payload, namespace=namespace)
        if not self.enabled:
            value = compute_fn()
            return value, CacheLookup(key=cache_key, hit=False, layer="disabled", artifact=artifact)

        if self._process is not None:
            value = self._process.get(cache_key)
            if value is not None:
                return value, CacheLookup(key=cache_key, hit=True, layer="process", artifact=artifact)

        if persist and self._disk is not None:
            value = self._disk.get(cache_key)
            if value is not None:
                if self._process is not None:
                    use_cost = self._resolve_cost(artifact, value, cost)
                    self._process.put(cache_key, value, artifact=artifact, cost=use_cost)
                return value, CacheLookup(key=cache_key, hit=True, layer="disk", artifact=artifact)

        value = compute_fn()
        use_cost = self._resolve_cost(artifact, value, cost)
        if self._process is not None:
            self._process.put(cache_key, value, artifact=artifact, cost=use_cost)
        if persist and self._disk is not None:
            self._disk.put(
                cache_key,
                value,
                artifact=artifact,
                cost=use_cost,
                ttl_seconds=ttl_seconds,
            )
        return value, CacheLookup(key=cache_key, hit=False, layer="compute", artifact=artifact)

    def _resolve_cost(self, artifact: str, value: Any, explicit_cost: float | None) -> float:
        if explicit_cost is not None:
            return float(explicit_cost)
        base = float(self.policy.artifact_cost.get(artifact, 1.0))
        size_mb = estimate_object_size_bytes(value) / (1024.0 * 1024.0)
        return max(1.0, base * max(1.0, size_mb))

    def invalidate(self, prefix: str = "", reason: str = "") -> int:
        del reason
        removed = 0
        if self._process is not None:
            removed += self._process.invalidate(prefix=prefix)
        if self._disk is not None:
            removed += self._disk.invalidate(prefix=prefix)
        return removed

    def clear(self, scope: CacheScope = "both") -> None:
        if scope in {"process", "both"} and self._process is not None:
            self._process.clear()
        if scope in {"disk", "both"} and self._disk is not None:
            self._disk.clear()

    def stats(self) -> dict[str, Any]:
        stats = CacheStats()
        if self._process is not None:
            p = self._process.stats()
            stats.process_hits = int(p["hits"])
            stats.process_misses = int(p["misses"])
            stats.process_items = int(p["items"])
            stats.process_bytes = int(p["bytes"])
            stats.process_max_bytes = int(p["max_bytes"])
        if self._disk is not None:
            d = self._disk.stats()
            stats.disk_hits = int(d["hits"])
            stats.disk_misses = int(d["misses"])
            stats.disk_items = int(d["items"])
            stats.disk_bytes = int(d["bytes"])
            stats.disk_max_bytes = int(d["max_bytes"])
        return stats.to_dict()

