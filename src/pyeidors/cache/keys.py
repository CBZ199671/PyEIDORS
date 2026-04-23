"""Stable cache key builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _normalize(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return {"__bytes__": hashlib.sha256(obj).hexdigest()}
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "sha256": hashlib.sha256(obj.tobytes()).hexdigest(),
        }
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {
            str(k): _normalize(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, set):
        return sorted(
            (_normalize(v) for v in obj), key=lambda x: json.dumps(x, sort_keys=True)
        )
    if hasattr(obj, "__dataclass_fields__"):
        return _normalize(asdict(obj))
    return obj


@dataclass(frozen=True)
class CacheKeyParts:
    """Semantic parts that determine cache identity."""

    artifact: str
    payload: dict[str, Any]
    schema_version: int = 2
    code_fingerprint: str = ""
    namespace: str = "default"

    def normalized(self) -> dict[str, Any]:
        return _normalize(
            {
                "artifact": self.artifact,
                "payload": self.payload,
                "schema_version": self.schema_version,
                "code_fingerprint": self.code_fingerprint,
                "namespace": self.namespace,
            }
        )


def build_cache_key(parts: CacheKeyParts) -> str:
    """Build SHA-256 key from normalized parts."""

    normalized = parts.normalized()
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_path(path: str | Path) -> str:
    """Hash a file path plus stable file metadata and content for invalidation."""

    p = Path(path)
    try:
        stat = p.stat()
        content_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        payload = (
            f"{p.resolve()}::{stat.st_size}::{stat.st_mtime_ns}::{content_hash}"
        ).encode("utf-8")
    except OSError:
        payload = str(p).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_array(arr: np.ndarray) -> str:
    """Hash an ndarray with dtype/shape metadata."""

    a = np.ascontiguousarray(np.asarray(arr))
    payload = f"{a.dtype}:{a.shape}:".encode("utf-8") + a.tobytes()
    return hashlib.sha256(payload).hexdigest()
