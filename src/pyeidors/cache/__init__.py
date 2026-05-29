"""Cache subsystem for PyEIDORS.

Cache helpers cover everything from tiny key utilities to persistent store
managers.  Keep package import light so tools can inspect cache metadata without
eagerly importing NumPy-backed signatures or disk/session store backends.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    ".object_signature": (
        "backend_signature_from_forward_model",
        "model_signature_from_forward_model",
        "pattern_signature_from_forward_model",
        "signature_of_cache_obj",
        "stable_signature_hash",
    ),
    ".lifecycle": (
        "cleanup_registered_session_caches",
        "cleanup_stale_session_caches",
    ),
    ".types": (
        "CacheArtifactKind",
        "DEFAULT_CACHE_LIFECYCLE",
        "CacheLookup",
        "CachePolicy",
        "CacheScope",
        "normalize_cache_lifecycle",
    ),
    ".keys": (
        "CacheKeyParts",
        "build_cache_key",
        "hash_array",
        "hash_array_payload",
        "hash_path",
        "update_digest_with_array_payload",
    ),
    ".manager": ("CacheManager",),
    ".disk_artifacts": (
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
    ),
}

_EXPORT_MODULES = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = [
    "backend_signature_from_forward_model",
    "cleanup_registered_session_caches",
    "cleanup_stale_session_caches",
    "CacheArtifactKind",
    "DEFAULT_CACHE_LIFECYCLE",
    "CacheKeyParts",
    "CacheLookup",
    "CacheManager",
    "CachePolicy",
    "CacheScope",
    "DISK_ARTIFACT_MANIFEST_VERSION",
    "DISK_ARTIFACT_KIND_POLICIES",
    "DiskArtifactManifest",
    "DiskArtifactKindPolicy",
    "FUTURE_DISK_ARTIFACT_KINDS",
    "INTEGRATED_DISK_ARTIFACT_KINDS",
    "READ_ONLY_DISK_ARTIFACT_KINDS",
    "assert_integrated_disk_artifact_kind",
    "build_cache_key",
    "build_disk_artifact_key",
    "build_disk_artifact_manifest",
    "build_disk_artifact_subkey",
    "build_disk_artifact_subkeys",
    "disk_artifact_kind_policy",
    "ensure_disk_artifact_metadata",
    "file_fingerprint",
    "file_sha256",
    "hash_array",
    "hash_array_payload",
    "hash_path",
    "model_signature_from_forward_model",
    "normalize_cache_lifecycle",
    "pattern_signature_from_forward_model",
    "signature_of_cache_obj",
    "stable_signature_hash",
    "stable_json_digest",
    "update_digest_with_array_payload",
]

_SUBMODULE_NAMES = frozenset(
    {
        "cli",
        "disk_artifacts",
        "keys",
        "lifecycle",
        "manager",
        "object_signature",
        "ops",
        "process_lru",
        "store_disk",
        "store_process",
        "types",
    }
)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is not None:
        module = import_module(module_name, __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SUBMODULE_NAMES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULE_NAMES))
