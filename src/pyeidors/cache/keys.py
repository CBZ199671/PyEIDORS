"""Stable cache key builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np


_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _sha256_file_content(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_file_content(path: str | Path) -> str:
    """Hash file bytes with bounded reads, independent of path metadata."""

    return _sha256_file_content(Path(path))


def _update_digest_with_array_bytes(
    digest: Any,
    array: np.ndarray,
) -> None:
    arr = np.asarray(array)
    if arr.dtype.hasobject or arr.dtype.kind not in {"b", "i", "u", "f", "c", "m", "M"}:
        _update_digest_with_memoryview(digest, np.ascontiguousarray(arr))
        return
    if arr.flags.c_contiguous:
        _update_digest_with_memoryview(digest, arr)
        return
    if arr.ndim == 0:
        _update_digest_with_memoryview(digest, np.ascontiguousarray(arr))
        return

    trailing = int(np.prod(arr.shape[1:], dtype=np.int64)) if arr.ndim > 1 else 1
    row_bytes = max(int(arr.dtype.itemsize) * max(trailing, 1), 1)
    rows_per_chunk = max(1, int(_HASH_CHUNK_BYTES) // row_bytes)
    for start in range(0, arr.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, arr.shape[0])
        _update_digest_with_memoryview(digest, np.ascontiguousarray(arr[start:stop]))


def _update_digest_with_memoryview(digest: Any, arr: np.ndarray) -> None:
    try:
        view = memoryview(arr).cast("B")
    except TypeError:
        _update_digest_with_numpy_save_payload(digest, np.ascontiguousarray(arr))
        return
    for offset in range(0, view.nbytes, _HASH_CHUNK_BYTES):
        digest.update(view[offset : offset + _HASH_CHUNK_BYTES])


def _update_digest_with_numpy_save_payload(digest: Any, arr: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=True)
    view = buffer.getbuffer()
    for offset in range(0, view.nbytes, _HASH_CHUNK_BYTES):
        digest.update(view[offset : offset + _HASH_CHUNK_BYTES])


def update_digest_with_array_payload(digest: Any, arr: np.ndarray) -> None:
    """Update an existing digest with ndarray payload bytes in bounded chunks."""

    _update_digest_with_array_bytes(digest, arr)


def hash_array_payload(arr: np.ndarray, *, prefix: bytes = b"") -> str:
    """Hash only ndarray payload bytes, preserving legacy callers' prefixes."""

    digest = hashlib.sha256()
    if prefix:
        digest.update(prefix)
    update_digest_with_array_payload(digest, arr)
    return digest.hexdigest()


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
            "sha256": hash_array_payload(obj),
        }
    if isinstance(obj, (complex, np.complexfloating)):
        value = complex(obj)
        return {
            "__complex__": {
                "real": float(value.real),
                "imag": float(value.imag),
            }
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
        content_hash = _sha256_file_content(p)
        payload = (
            f"{p.resolve()}::{stat.st_size}::{stat.st_mtime_ns}::{content_hash}"
        ).encode("utf-8")
    except OSError:
        payload = str(p).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def hash_array(arr: np.ndarray) -> str:
    """Hash an ndarray with dtype/shape metadata."""

    a = np.asarray(arr)
    digest = hashlib.sha256()
    digest.update(f"{a.dtype}:{a.shape}:".encode("utf-8"))
    _update_digest_with_array_bytes(digest, a)
    return digest.hexdigest()
