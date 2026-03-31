"""Cache subsystem for PyEIDORS."""

from .keys import CacheKeyParts, build_cache_key, hash_array, hash_path
from .lifecycle import cleanup_registered_session_caches, cleanup_stale_session_caches
from .manager import CacheManager
from .object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
    signature_of_cache_obj,
    stable_signature_hash,
)
from .types import (
    DEFAULT_CACHE_LIFECYCLE,
    CacheArtifactKind,
    CacheLookup,
    CachePolicy,
    CacheScope,
    normalize_cache_lifecycle,
)

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
    "build_cache_key",
    "hash_array",
    "hash_path",
    "model_signature_from_forward_model",
    "normalize_cache_lifecycle",
    "pattern_signature_from_forward_model",
    "signature_of_cache_obj",
    "stable_signature_hash",
]
