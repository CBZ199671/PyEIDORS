"""Unified multi-layer cache manager."""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Callable

from .keys import CacheKeyParts, build_cache_key
from .lifecycle import resolve_cache_directory
from .object_signature import signature_of_cache_obj
from .store_disk import DiskCacheStore
from .store_process import ProcessCacheStore, estimate_object_size_bytes
from .types import (
    CacheLookup,
    CachePolicy,
    CacheScope,
    CacheStats,
    normalize_cache_lifecycle,
)


_SHARED_PROCESS_STORES: dict[tuple[str, str], ProcessCacheStore] = {}
_SHARED_PROCESS_STORES_LOCK = threading.Lock()


def _shared_process_store_key(
    *, cache_dir: Path, code_fingerprint: str
) -> tuple[str, str]:
    return (str(cache_dir.resolve()), str(code_fingerprint or "unknown"))


def _get_shared_process_store(
    *,
    cache_dir: Path,
    max_bytes: int,
    code_fingerprint: str,
) -> ProcessCacheStore:
    key = _shared_process_store_key(
        cache_dir=cache_dir,
        code_fingerprint=code_fingerprint,
    )
    with _SHARED_PROCESS_STORES_LOCK:
        store = _SHARED_PROCESS_STORES.get(key)
        if store is None:
            store = ProcessCacheStore(int(max(0, max_bytes)))
            _SHARED_PROCESS_STORES[key] = store
            return store
        if int(max_bytes) > int(store.max_bytes):
            store.max_bytes = int(max_bytes)
        return store


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
        self.requested_cache_dir = Path(cache_dir)
        self.disk_lifecycle = normalize_cache_lifecycle(
            getattr(self.policy, "disk_lifecycle", "session")
        )
        if scope in {"disk", "both"}:
            self._cache_dir_spec = resolve_cache_directory(
                self.requested_cache_dir,
                lifecycle=self.disk_lifecycle,
                cleanup_on_exit=bool(getattr(self.policy, "cleanup_on_exit", True)),
                cleanup_stale_sessions_on_startup=bool(
                    getattr(self.policy, "cleanup_stale_sessions_on_startup", True)
                ),
                stale_session_max_age_seconds=float(
                    getattr(
                        self.policy, "stale_session_max_age_seconds", 7 * 24 * 60 * 60
                    )
                ),
            )
        else:
            self._cache_dir_spec = None
        self.cache_dir = Path(
            self._cache_dir_spec.effective_dir
            if self._cache_dir_spec is not None
            else self.requested_cache_dir
        )

        self._cache_enable: float = 0.0 if scope == "off" else 1.0
        self._cache_disabled_on: set[str] = set()
        self._debug_enable: float = 0.0
        self._debug_enabled_on: set[str] = set()
        self._priority_boost: float = 0.0

        self._process = (
            _get_shared_process_store(
                cache_dir=self.requested_cache_dir,
                max_bytes=self.policy.process_max_bytes,
                code_fingerprint=self.code_fingerprint,
            )
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
        return self.status() > 0.0

    @property
    def session_cache_enabled(self) -> bool:
        return bool(self.disk_lifecycle == "session" and self.scope in {"disk", "both"})

    def build_key(
        self, artifact: str, payload: dict[str, Any], namespace: str = "default"
    ) -> str:
        return build_cache_key(
            CacheKeyParts(
                artifact=artifact,
                payload=payload,
                schema_version=2,
                code_fingerprint=self.code_fingerprint,
                namespace=namespace,
            )
        )

    def status(self, name: str | None = None) -> float:
        """Return cache enable status in EIDORS style: 0, 0.5, 1."""
        if name is None:
            return float(self._cache_enable)
        if self._cache_enable == 0.0:
            return 0.0
        if self._cache_enable == 0.5 and name in self._cache_disabled_on:
            return 0.0
        return 1.0

    def set_enabled(self, on: bool, name: str | None = None) -> float:
        """Enable/disable cache globally or for a specific name."""
        if name is None:
            self._cache_enable = 1.0 if on else 0.0
            self._cache_disabled_on.clear()
            return float(self._cache_enable)

        if on:
            self._cache_disabled_on.discard(name)
            if not self._cache_disabled_on:
                self._cache_enable = 1.0
        else:
            self._cache_enable = 0.5
            self._cache_disabled_on.add(name)
        return self.status(name)

    def debug_status(self, name: str | None = None) -> float:
        """Return debug status in EIDORS style: 0, 0.5, 1."""
        if name is None:
            return float(self._debug_enable)
        if self._debug_enable == 1.0:
            return 1.0
        if self._debug_enable == 0.5 and name in self._debug_enabled_on:
            return 1.0
        return 0.0

    def set_debug(self, on: bool, name: str | None = None) -> float:
        """Enable/disable debug globally or for a specific name."""
        if name is None:
            self._debug_enable = 1.0 if on else 0.0
            self._debug_enabled_on.clear()
            return float(self._debug_enable)

        if on:
            self._debug_enable = 0.5
            self._debug_enabled_on.add(name)
        else:
            self._debug_enabled_on.discard(name)
            if self._debug_enable == 0.5 and not self._debug_enabled_on:
                self._debug_enable = 0.0
        return self.debug_status(name)

    def boost_priority(self, delta: float = 0.0) -> float:
        """Adjust global priority boost used by subsequent cache writes."""
        self._priority_boost += float(delta)
        return float(self._priority_boost)

    def _is_enabled_for(self, name: str) -> bool:
        if self.scope == "off":
            return False
        if self.status(name) <= 0.0:
            return False
        return self._process is not None or self._disk is not None

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
        effective_name = name or artifact
        if not self._is_enabled_for(effective_name):
            value = compute_fn()
            return value, CacheLookup(
                key=cache_key, hit=False, layer="disabled", artifact=artifact
            )

        effective_priority = float(priority_boost) + float(self._priority_boost)

        if self._process is not None:
            value = self._process.get(cache_key)
            if value is not None:
                return value, CacheLookup(
                    key=cache_key, hit=True, layer="process", artifact=artifact
                )

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
                        name=effective_name,
                        namespace=namespace,
                        effort=effort_seconds,
                        priority=effective_priority,
                    )
                return value, CacheLookup(
                    key=cache_key, hit=True, layer="disk", artifact=artifact
                )

        value = compute_fn()
        use_cost = self._resolve_cost(artifact, value, cost)
        if self._process is not None:
            self._process.put(
                cache_key,
                value,
                artifact=artifact,
                cost=use_cost,
                name=effective_name,
                namespace=namespace,
                effort=effort_seconds,
                priority=effective_priority,
            )
        if persist and self._disk is not None:
            self._disk.put(
                cache_key,
                value,
                artifact=artifact,
                cost=use_cost,
                ttl_seconds=ttl_seconds,
                name=effective_name,
                namespace=namespace,
                effort=effort_seconds,
                priority=effective_priority,
            )
        return value, CacheLookup(
            key=cache_key, hit=False, layer="compute", artifact=artifact
        )

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

    def _resolve_cost(
        self, artifact: str, value: Any, explicit_cost: float | None
    ) -> float:
        if explicit_cost is not None:
            return float(explicit_cost)
        base = float(self.policy.artifact_cost.get(artifact, 1.0))
        size_mb = estimate_object_size_bytes(value) / (1024.0 * 1024.0)
        return max(1.0, base * max(1.0, size_mb))

    def _apply_to_stores(self, method: str, *args: Any, **kwargs: Any) -> int:
        """Apply a named method to both cache stores and sum the results."""
        removed = 0
        for store in (self._process, self._disk):
            if store is not None:
                removed += getattr(store, method)(*args, **kwargs)
        return removed

    def invalidate(self, prefix: str = "", reason: str = "") -> int:  # noqa: ARG002
        return self._apply_to_stores("invalidate", prefix=prefix)

    def clear_name(self, name: str, namespace: str | None = None) -> int:
        return self._apply_to_stores("clear_name", name=name, namespace=namespace)

    def clear_max(self, max_bytes: int) -> int:
        return self._apply_to_stores("clear_max", max_bytes=max_bytes)

    def clear_old(self, timestamp: float) -> int:
        return self._apply_to_stores("clear_old", timestamp)

    def clear_new(self, timestamp: float) -> int:
        return self._apply_to_stores("clear_new", timestamp)

    def _entry_value_for_layer(self, key: str, layer: str) -> Any | None:
        if layer == "process" and self._process is not None:
            return self._process.get_value(key)
        if layer == "disk" and self._disk is not None:
            return self._disk.get_value(key)
        return None

    @staticmethod
    def _snapshot_record(entry: dict[str, Any], value: Any | None) -> dict[str, Any]:
        record = {
            "id": entry.get("key"),
            "key": entry.get("key"),
            "name": entry.get("name"),
            "layer": entry.get("layer"),
            "meta": {
                "artifact": entry.get("artifact"),
                "namespace": entry.get("namespace"),
                "size_bytes": entry.get("size_bytes"),
                "cost": entry.get("cost"),
                "effort": entry.get("effort"),
                "priority": entry.get("priority"),
                "use_count": entry.get("use_count"),
                "score_eff": entry.get("score_eff", entry.get("score")),
                "score_size": entry.get("score_size"),
                "score": entry.get("score"),
                "created_at": entry.get("created_at"),
                "last_access": entry.get("last_access"),
                "layer": entry.get("layer"),
            },
        }
        if value is not None:
            record["val"] = value
        return record

    def collect_recent(
        self,
        *,
        names: list[str],
        limit_per_name: int = 1,
        namespace: str | None = None,
        include_value: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        name_counts: dict[str, int] = {}
        for name in names:
            name_counts[name] = name_counts.get(name, 0) + 1

        collected: dict[str, list[dict[str, Any]]] = {}
        for name, repeats in name_counts.items():
            required = max(int(limit_per_name), int(repeats), 1)
            entries: list[dict[str, Any]] = []
            if self._process is not None:
                entries.extend(
                    self._process.list_entries(
                        name=name,
                        namespace=namespace,
                        limit=None,
                    )
                )
            if self._disk is not None:
                entries.extend(
                    self._disk.list_entries(
                        name=name,
                        namespace=namespace,
                        limit=None,
                    )
                )

            entries.sort(
                key=lambda item: float(item.get("last_access", 0.0)), reverse=True
            )
            selected: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            for entry in entries:
                key = str(entry.get("key", ""))
                if not key or key in seen_keys:
                    continue
                selected.append(entry)
                seen_keys.add(key)
                if len(selected) >= required:
                    break
            records: list[dict[str, Any]] = []
            for entry in selected:
                value = (
                    self._entry_value_for_layer(
                        key=str(entry.get("key")),
                        layer=str(entry.get("layer")),
                    )
                    if include_value
                    else None
                )
                records.append(self._snapshot_record(entry, value))
            collected[name] = records
        return collected

    def install_to_cache(
        self, snapshot: Any, target_layers: CacheScope = "both"
    ) -> int:
        """Install exported cache snapshot entries back to cache stores."""

        if target_layers not in {"process", "disk", "both"}:
            raise ValueError("target_layers must be one of: process, disk, both")

        if isinstance(snapshot, dict):
            candidates = []
            for value in snapshot.values():
                if isinstance(value, list):
                    candidates.extend(value)
        elif isinstance(snapshot, list):
            candidates = snapshot
        else:
            raise TypeError("snapshot must be dict[str, list] or list")

        installed = 0
        for item in candidates:
            if not isinstance(item, dict):
                continue
            value = item.get("val")
            if value is None:
                continue
            key = item.get("key") or item.get("id")
            if not isinstance(key, str) or not key:
                continue
            meta = item.get("meta")
            if not isinstance(meta, dict):
                meta = {}

            artifact = str(meta.get("artifact") or "")
            if not artifact:
                continue
            name = str(item.get("name") or meta.get("name") or artifact)
            namespace = str(meta.get("namespace") or "default")
            cost = float(meta.get("cost", self._resolve_cost(artifact, value, None)))
            effort = meta.get("effort")
            effort_value = float(effort) if effort is not None else None
            priority = float(meta.get("priority", 0.0))

            if target_layers in {"process", "both"} and self._process is not None:
                self._process.put(
                    key,
                    value,
                    artifact=artifact,
                    cost=cost,
                    name=name,
                    namespace=namespace,
                    effort=effort_value,
                    priority=priority,
                )
                installed += 1
            if target_layers in {"disk", "both"} and self._disk is not None:
                ttl_seconds = meta.get("ttl_seconds")
                self._disk.put(
                    key,
                    value,
                    artifact=artifact,
                    cost=cost,
                    ttl_seconds=float(ttl_seconds) if ttl_seconds is not None else None,
                    name=name,
                    namespace=namespace,
                    effort=effort_value,
                    priority=priority,
                )
                installed += 1
        return installed

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
                self._process.list_entries(name=name, namespace=namespace, limit=None)
            )
        if self._disk is not None:
            entries.extend(
                self._disk.list_entries(name=name, namespace=namespace, limit=None)
            )
        entries.sort(key=lambda item: float(item.get("last_access", 0.0)), reverse=True)
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
        payload["cache_status"] = self.status()
        payload["debug_status"] = self.debug_status()
        payload["priority_boost"] = float(self._priority_boost)
        payload["disabled_names"] = sorted(self._cache_disabled_on)
        payload["debug_names"] = sorted(self._debug_enabled_on)
        payload["process_artifacts"] = process_artifacts
        payload["process_namespaces"] = process_namespaces
        payload["disk_artifacts"] = disk_artifacts
        payload["disk_namespaces"] = disk_namespaces
        payload["disk_cache_lifecycle"] = str(self.disk_lifecycle)
        payload["disk_cache_requested_dir"] = str(self.requested_cache_dir)
        payload["disk_cache_effective_dir"] = str(self.cache_dir)
        payload["disk_cache_cleanup_on_exit"] = bool(
            self.session_cache_enabled and getattr(self.policy, "cleanup_on_exit", True)
        )
        return payload
