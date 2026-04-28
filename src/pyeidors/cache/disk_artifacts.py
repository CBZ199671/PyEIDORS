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
from typing import Any, Literal, Mapping

from .keys import _normalize

DISK_ARTIFACT_MANIFEST_VERSION = 1
DiskArtifactKindStatus = Literal["integrated", "future-scope", "read-only"]


@dataclass(frozen=True)
class DiskArtifactKindPolicy:
    """Governance status for one persistent disk-artifact kind."""

    status: DiskArtifactKindStatus
    gate: str
    note: str


DISK_ARTIFACT_KIND_POLICIES: Mapping[str, DiskArtifactKindPolicy] = {
    "hdf5-artifact": DiskArtifactKindPolicy(
        status="integrated",
        gate="T82 phases 1..4",
        note="Shared numeric HDF5 packages via pyeidors.io.hdf5_artifacts.",
    ),
    "dolfinx-mesh-cache": DiskArtifactKindPolicy(
        status="integrated",
        gate="T82 phases 1..4",
        note="DOLFINx XDMF/HDF5 mesh cache with optional ADIOS side files.",
    ),
    "adios4dolfinx-checkpoint": DiskArtifactKindPolicy(
        status="future-scope",
        gate="Add only if ADIOS4DOLFINx becomes an independent reload source.",
        note="Today it remains a side file recorded inside dolfinx-mesh-cache.",
    ),
    "adios2-vtx-side-artifact": DiskArtifactKindPolicy(
        status="future-scope",
        gate="Add only if VTX/BP gains a supported reader.",
        note="Today it remains optional write-side output.",
    ),
    "cache-manager-disk-object": DiskArtifactKindPolicy(
        status="future-scope",
        gate="Add only for durable cache-manager export/import workflows.",
        note=".pyeidors_cache/v2 keeps its own runtime index.",
    ),
    "legacy-npz-artifact": DiskArtifactKindPolicy(
        status="read-only",
        gate="V65/V67: legacy compatibility loaders only; new writers forbidden.",
        note="Migrate large numeric artifacts to HDF5 instead of registering writers.",
    ),
    "mesh-cache-layer": DiskArtifactKindPolicy(
        status="future-scope",
        gate="Add only if multiple formats share a real storage backend.",
        note="Do not introduce a leaky protocol over format-divergent code.",
    ),
}

INTEGRATED_DISK_ARTIFACT_KINDS = tuple(
    sorted(
        kind
        for kind, policy in DISK_ARTIFACT_KIND_POLICIES.items()
        if policy.status == "integrated"
    )
)
FUTURE_DISK_ARTIFACT_KINDS = tuple(
    sorted(
        kind
        for kind, policy in DISK_ARTIFACT_KIND_POLICIES.items()
        if policy.status == "future-scope"
    )
)
READ_ONLY_DISK_ARTIFACT_KINDS = tuple(
    sorted(
        kind
        for kind, policy in DISK_ARTIFACT_KIND_POLICIES.items()
        if policy.status == "read-only"
    )
)


@dataclass(frozen=True)
class DiskArtifactManifest:
    """Serializable manifest embedded into persistent cache metadata."""

    artifact_kind: str
    artifact_key: str
    subkeys: Mapping[str, str]
    namespace: str
    schema_version: int
    key_payload: Mapping[str, Any]
    files: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_key": self.artifact_key,
            "subkeys": _normalize(self.subkeys),
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


def disk_artifact_kind_policy(
    artifact_kind: str,
) -> DiskArtifactKindPolicy | None:
    """Return registered governance policy for one artifact kind."""

    return DISK_ARTIFACT_KIND_POLICIES.get(str(artifact_kind))


def assert_integrated_disk_artifact_kind(artifact_kind: str) -> None:
    """Fail unless an artifact kind is explicitly integrated for writing."""

    kind = str(artifact_kind)
    policy = disk_artifact_kind_policy(kind)
    if policy is None:
        raise ValueError(
            f"Unregistered disk artifact kind {kind!r}. Add a T82 governance "
            "policy before writing manifests for a new persistent format."
        )
    if policy.status != "integrated":
        raise ValueError(
            f"Disk artifact kind {kind!r} is {policy.status}, not integrated. "
            f"Gate before integration: {policy.gate}"
        )


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

    assert_integrated_disk_artifact_kind(artifact_kind)
    return stable_json_digest(
        {
            "artifact_kind": str(artifact_kind),
            "namespace": str(namespace),
            "schema_version": int(schema_version),
            "payload": dict(key_payload),
        }
    )


def build_disk_artifact_subkey(
    subkey_name: str,
    payload: Mapping[str, Any],
    *,
    namespace: str = "pyeidors",
    schema_version: int = DISK_ARTIFACT_MANIFEST_VERSION,
) -> str:
    """Build a stable digest for provenance shared across artifact kinds."""

    return stable_json_digest(
        {
            "namespace": str(namespace),
            "schema_version": int(schema_version),
            "subkey_name": str(subkey_name),
            "payload": dict(payload),
        }
    )


def build_disk_artifact_subkeys(
    subkey_payloads: Mapping[str, Mapping[str, Any]] | None,
    *,
    namespace: str = "pyeidors",
    schema_version: int = DISK_ARTIFACT_MANIFEST_VERSION,
) -> dict[str, str]:
    """Build named subkeys from shared semantic payloads."""

    return {
        str(name): build_disk_artifact_subkey(
            str(name),
            payload,
            namespace=namespace,
            schema_version=schema_version,
        )
        for name, payload in sorted((subkey_payloads or {}).items())
    }


def build_disk_artifact_manifest(
    artifact_kind: str,
    key_payload: Mapping[str, Any],
    *,
    files: Mapping[str, str | Path | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
    subkey_payloads: Mapping[str, Mapping[str, Any]] | None = None,
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
        subkeys=build_disk_artifact_subkeys(
            subkey_payloads,
            namespace=namespace,
            schema_version=schema_version,
        ),
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
    subkey_payloads: Mapping[str, Mapping[str, Any]] | None = None,
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
        subkey_payloads=subkey_payloads,
        namespace=namespace,
        schema_version=schema_version,
        include_file_sha256=include_file_sha256,
    )
    out.setdefault("artifact_key", manifest.artifact_key)
    out.setdefault("artifact_manifest", manifest.to_metadata())
    return out


__all__ = [
    "DISK_ARTIFACT_MANIFEST_VERSION",
    "DISK_ARTIFACT_KIND_POLICIES",
    "DiskArtifactManifest",
    "DiskArtifactKindPolicy",
    "FUTURE_DISK_ARTIFACT_KINDS",
    "INTEGRATED_DISK_ARTIFACT_KINDS",
    "READ_ONLY_DISK_ARTIFACT_KINDS",
    "assert_integrated_disk_artifact_kind",
    "build_disk_artifact_key",
    "build_disk_artifact_manifest",
    "build_disk_artifact_subkey",
    "build_disk_artifact_subkeys",
    "disk_artifact_kind_policy",
    "ensure_disk_artifact_metadata",
    "file_fingerprint",
    "file_sha256",
    "stable_json_digest",
]
