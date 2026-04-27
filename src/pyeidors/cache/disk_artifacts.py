"""Shared disk-artifact key and manifest helpers.

Persistent caches may use different physical formats (XDMF/HDF5, ADIOS2,
large numeric HDF5 packages), but their semantic artifact identity should be
formed in one place. This module deliberately does not implement a storage
backend. It only builds deterministic keys and small metadata manifests that
format-specific writers can embed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .keys import _normalize

DISK_ARTIFACT_MANIFEST_VERSION = 1


@dataclass(frozen=True)
class DiskArtifactManifest:
    """Serializable manifest embedded into persistent cache metadata."""

    artifact_kind: str
    artifact_key: str
    namespace: str
    schema_version: int
    key_payload: Mapping[str, Any]
    files: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_key": self.artifact_key,
            "namespace": self.namespace,
            "schema_version": int(self.schema_version),
            "manifest_version": DISK_ARTIFACT_MANIFEST_VERSION,
            "key_payload": _normalize(self.key_payload),
            "files": _normalize(self.files),
            "metadata": _normalize(self.metadata),
        }


def stable_json_digest(payload: Mapping[str, Any]) -> str:
    """Return SHA256 hex of project-canonical JSON for a metadata payload."""

    encoded = json.dumps(
        _normalize(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str | None:
    """Return file SHA256 hex, or None when the path is absent/unreadable."""

    source = Path(path)
    try:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def file_fingerprint(
    path: str | Path | None,
    *,
    include_sha256: bool = False,
) -> dict[str, Any] | None:
    """Return path/status metadata for artifact files without affecting keys."""

    if path is None:
        return None
    target = Path(path)
    payload: dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
    }
    try:
        stat = target.stat()
    except OSError:
        return payload
    payload.update(
        {
            "resolved_path": str(target.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "is_dir": target.is_dir(),
        }
    )
    if include_sha256 and target.is_file():
        payload["sha256"] = file_sha256(target)
    return payload


def build_disk_artifact_key(
    artifact_kind: str,
    key_payload: Mapping[str, Any],
    *,
    namespace: str = "pyeidors",
    schema_version: int = DISK_ARTIFACT_MANIFEST_VERSION,
) -> str:
    """Build a deterministic semantic key for a persistent artifact.

    The key intentionally excludes output file paths and storage backend
    details. Callers should place those details in the manifest ``files`` map
    so moving an artifact between cache directories does not change its
    mathematical identity.
    """

    return stable_json_digest(
        {
            "artifact_kind": str(artifact_kind),
            "namespace": str(namespace),
            "schema_version": int(schema_version),
            "payload": dict(key_payload),
        }
    )


def build_disk_artifact_manifest(
    artifact_kind: str,
    key_payload: Mapping[str, Any],
    *,
    files: Mapping[str, str | Path | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
    namespace: str = "pyeidors",
    schema_version: int = DISK_ARTIFACT_MANIFEST_VERSION,
    include_file_sha256: bool = False,
) -> DiskArtifactManifest:
    """Build a manifest whose key is independent of physical file locations."""

    key = build_disk_artifact_key(
        artifact_kind,
        key_payload,
        namespace=namespace,
        schema_version=schema_version,
    )
    file_payload = {
        str(name): file_fingerprint(path, include_sha256=include_file_sha256)
        for name, path in (files or {}).items()
    }
    return DiskArtifactManifest(
        artifact_kind=str(artifact_kind),
        artifact_key=key,
        namespace=str(namespace),
        schema_version=int(schema_version),
        key_payload=_normalize(dict(key_payload)),
        files=file_payload,
        metadata=_normalize(dict(metadata or {})),
    )


def ensure_disk_artifact_metadata(
    metadata: Mapping[str, Any],
    artifact_kind: str,
    key_payload: Mapping[str, Any],
    *,
    files: Mapping[str, str | Path | None] | None = None,
    manifest_metadata: Mapping[str, Any] | None = None,
    namespace: str = "pyeidors",
    schema_version: int = DISK_ARTIFACT_MANIFEST_VERSION,
    include_file_sha256: bool = False,
) -> dict[str, Any]:
    """Return metadata with ``artifact_key`` / ``artifact_manifest`` populated.

    Existing values are preserved so newer artifacts round-trip byte-for-byte in
    memory. Legacy artifacts that predate T82 phase 1 get an in-memory manifest
    from the same semantic payload without mutating the file on disk.
    """

    out = dict(metadata)
    if out.get("artifact_key") and isinstance(out.get("artifact_manifest"), Mapping):
        return out
    manifest = build_disk_artifact_manifest(
        artifact_kind,
        key_payload,
        files=files,
        metadata=manifest_metadata,
        namespace=namespace,
        schema_version=schema_version,
        include_file_sha256=include_file_sha256,
    )
    out.setdefault("artifact_key", manifest.artifact_key)
    out.setdefault("artifact_manifest", manifest.to_metadata())
    return out


__all__ = [
    "DISK_ARTIFACT_MANIFEST_VERSION",
    "DiskArtifactManifest",
    "build_disk_artifact_key",
    "build_disk_artifact_manifest",
    "ensure_disk_artifact_metadata",
    "file_fingerprint",
    "file_sha256",
    "stable_json_digest",
]
