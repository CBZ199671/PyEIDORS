"""Unified multi-layer cache manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .keys import CacheKeyParts, build_cache_key
from .object_signature import signature_of_cache_obj
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
        name: str = "",
        priority_boost: float = 0.0,
        effort_seconds: float | None = None,
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
                    self._process.put(
                        cache_key,
                        value,
                        artifact=artifact,
                        cost=use_cost,
                        name=name,
                        namespace=namespace,
                        effort=effort_seconds,
                        priority=priority_boost,
                    )
                return value, CacheLookup(key=cache_key, hit=True, layer="disk", artifact=artifact)

        value = compute_fn()
        use_cost = self._resolve_cost(artifact, value, cost)
        if self._process is not None:
            self._process.put(
                cache_key,
                value,
                artifact=artifact,
                cost=use_cost,
                name=name,
                namespace=namespace,
                effort=effort_seconds,
                priority=priority_boost,
            )
        if persist and self._disk is not None:
            self._disk.put(
                cache_key,
                value,
                artifact=artifact,
                cost=use_cost,
                ttl_seconds=ttl_seconds,
                name=name,
                namespace=namespace,
                effort=effort_seconds,
                priority=priority_boost,
            )
        return value, CacheLookup(key=cache_key, hit=False, layer="compute", artifact=artifact)

    def get_or_compute_semantic(
        self,
        *,
        artifact: str,
        name: str,
        cache_obj: Any,
        compute_fn: Callable[[], Any],
        namespace: str = "default",
        payload: dict[str, Any] | None = None,
        priority_boost: float = 0.0,
        effort_seconds: float | None = None,
        cost: float | None = None,
        ttl_seconds: float | None = None,
        persist: bool = True,
    ) -> tuple[Any, CacheLookup]:
        """Cache helper that keys on semantic dependency objects."""

        semantic_payload: dict[str, Any] = {
            "cache_obj_signature": signature_of_cache_obj(cache_obj),
        }
        if payload:
            semantic_payload["payload"] = payload
        return self.get_or_compute(
            artifact=artifact,
            payload=semantic_payload,
            compute_fn=compute_fn,
            namespace=namespace,
            name=name,
            priority_boost=priority_boost,
            effort_seconds=effort_seconds,
            cost=cost,
            ttl_seconds=ttl_seconds,
            persist=persist,
        )

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

    def clear_name(self, name: str, namespace: str | None = None) -> int:
        removed = 0
        if self._process is not None:
            removed += self._process.clear_name(name=name, namespace=namespace)
        if self._disk is not None:
            removed += self._disk.clear_name(name=name, namespace=namespace)
        return removed

    def clear_max(self, max_bytes: int) -> int:
        removed = 0
        if self._process is not None:
            removed += self._process.clear_max(max_bytes=max_bytes)
        if self._disk is not None:
            removed += self._disk.clear_max(max_bytes=max_bytes)
        return removed

    def collect_recent(
        self,
        *,
        names: list[str],
        limit_per_name: int = 1,
        namespace: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        collected: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        if self._process is not None:
            for name in names:
                entries = self._process.list_entries(
                    name=name,
                    namespace=namespace,
                    limit=max(1, int(limit_per_name)),
                )
                collected[name].extend(entries)
        if self._disk is not None:
            disk_entries = self._disk.collect_recent(
                names=names,
                limit_per_name=max(1, int(limit_per_name)),
                namespace=namespace,
            )
            for name in names:
                collected[name].extend(disk_entries.get(name, []))
                collected[name].sort(key=lambda item: item["last_access"], reverse=True)
                collected[name] = collected[name][: max(1, int(limit_per_name))]
        return collected

    def list_entries(
        self,
        *,
        name: str | None = None,
        namespace: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        if self._process is not None:
            entries.extend(
                self._process.list_entries(name=name, namespace=namespace, limit=limit)
            )
        if self._disk is not None:
            entries.extend(self._disk.list_entries(name=name, namespace=namespace, limit=limit))
        entries.sort(key=lambda item: item["last_access"], reverse=True)
        if limit is not None and limit > 0:
            entries = entries[:limit]
        return entries

    def clear(self, scope: CacheScope = "both") -> None:
        if scope in {"process", "both"} and self._process is not None:
            self._process.clear()
        if scope in {"disk", "both"} and self._disk is not None:
            self._disk.clear()

    def stats(self) -> dict[str, Any]:
        stats = CacheStats()
        process_artifacts: dict[str, int] = {}
        process_namespaces: dict[str, int] = {}
        disk_artifacts: dict[str, int] = {}
        disk_namespaces: dict[str, int] = {}
        if self._process is not None:
            p = self._process.stats()
            stats.process_hits = int(p["hits"])
            stats.process_misses = int(p["misses"])
            stats.process_items = int(p["items"])
            stats.process_bytes = int(p["bytes"])
            stats.process_max_bytes = int(p["max_bytes"])
            process_artifacts = dict(p.get("artifacts", {}))
            process_namespaces = dict(p.get("namespaces", {}))
        if self._disk is not None:
            d = self._disk.stats()
            stats.disk_hits = int(d["hits"])
            stats.disk_misses = int(d["misses"])
            stats.disk_items = int(d["items"])
            stats.disk_bytes = int(d["bytes"])
            stats.disk_max_bytes = int(d["max_bytes"])
            disk_artifacts = dict(d.get("artifacts", {}))
            disk_namespaces = dict(d.get("namespaces", {}))

        payload = stats.to_dict()
        payload["process_artifacts"] = process_artifacts
        payload["process_namespaces"] = process_namespaces
        payload["disk_artifacts"] = disk_artifacts
        payload["disk_namespaces"] = disk_namespaces
        return payload

