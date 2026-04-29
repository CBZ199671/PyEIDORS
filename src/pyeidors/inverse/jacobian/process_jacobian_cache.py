"""Process-local Jacobian cache for persistent across-iteration reuse (T6).

The Gauss-Newton runtime ``prev_jacobian`` only lives inside one
``run_reconstruction`` call. T5 wired ``JacobianLinearization.assert_compatible``
on the within-loop reuse path, but if the same conductivity (same
``sigma_fingerprint``) is re-evaluated in a later run on the same mesh
the Jacobian had to be rebuilt from scratch.

This module pins a thread-safe :class:`pyeidors.cache.process_lru.ProcessLRUCache`
to the dense measurement Jacobian (``np.ndarray``) keyed by
``sigma_fingerprint`` + mesh identifier (file path or content hash) +
forward-model signatures + Jacobian method + calculator signature. The cache is purely
in-memory (process-local LRU). Disk persistence is out of scope here —
that would be a T82-style HDF5 artifact registered through
``pyeidors.cache.disk_artifacts``.

Cache-key contract (V9, V17, V16-style):

- ``sigma_fingerprint`` MUST be non-empty (V9 fingerprints are the
  primary axis distinguishing two Jacobians on the same mesh).
- At least one of ``mesh_file`` / ``mesh_content_hash`` MUST be
  non-empty (V17 — both empty is unsafe, mirroring V16's guard for
  forward setup keys). The mesh identifier feeds straight into the
  hash payload, so a fresh in-memory mesh with empty ``mesh_file`` and
  empty ``mesh_content_hash`` is rejected up front.
- Backend / pattern / model signatures are part of the payload so
  swapping solver backends or stim/meas patterns invalidates the
  cache entry.

Encoded with the same JSON conventions as
:func:`pyeidors.cache.process_lru.hash_json_payload`
(``sort_keys=True``, ``separators=(",", ":")``, ``ensure_ascii=True``)
so the digest is byte-stable across Python releases.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...cache.process_lru import ProcessLRUCache, hash_json_payload


_PROCESS_JACOBIAN_CACHE_MAX_ITEMS = 4
_PROCESS_JACOBIAN_CACHE: ProcessLRUCache[np.ndarray] = ProcessLRUCache(
    max_items=_PROCESS_JACOBIAN_CACHE_MAX_ITEMS
)


def build_process_jacobian_key(
    *,
    sigma_fingerprint: str,
    mesh_file: str | None = None,
    mesh_content_hash: str | None = None,
    jacobian_method: str = "default",
    calculator_signature: Mapping[str, Any] | str | None = None,
    model_signature: Mapping[str, Any] | str | None = None,
    pattern_signature: Mapping[str, Any] | str | None = None,
    backend_signature: Mapping[str, Any] | str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Build a content-addressed cache key for a measurement-space Jacobian.

    Raises ``ValueError`` when ``sigma_fingerprint`` is empty (V9 cannot
    distinguish two stored Jacobians without a fingerprint) or when both
    ``mesh_file`` and ``mesh_content_hash`` are empty (V17 — same guard as
    :func:`pyeidors.forward.process_setup_cache.build_process_forward_setup_key`).

    The forward-model signature helpers return SHA256 hex strings; the
    builder accepts either a string or a mapping for each of
    ``calculator_signature`` / ``model_signature`` / ``pattern_signature`` /
    ``backend_signature`` so callers can pass the existing hex digests directly.
    """
    sigma_token = str(sigma_fingerprint or "")
    if not sigma_token:
        raise ValueError(
            "build_process_jacobian_key requires a non-empty sigma_fingerprint "
            "to form a stable cache key (V9)."
        )
    file_token = str(mesh_file or "")
    content_token = str(mesh_content_hash or "")
    if not file_token and not content_token:
        raise ValueError(
            "build_process_jacobian_key requires either mesh_file or "
            "mesh_content_hash to form a stable cache key (V17)."
        )

    def _signature_token(value: Mapping[str, Any] | str | None) -> Any:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return dict(value)

    payload: dict[str, Any] = {
        "sigma_fingerprint": sigma_token,
        "mesh_file": file_token,
        "mesh_content_hash": content_token,
        "jacobian_method": str(jacobian_method or "default"),
        "calculator_signature": _signature_token(calculator_signature),
        "model_signature": _signature_token(model_signature),
        "pattern_signature": _signature_token(pattern_signature),
        "backend_signature": _signature_token(backend_signature),
    }
    if extra:
        payload["extra"] = dict(extra)
    return hash_json_payload(payload)


def get_process_cached_jacobian(key: str) -> np.ndarray | None:
    """Return a copy-on-write view of the cached Jacobian, or ``None``."""
    cached = _PROCESS_JACOBIAN_CACHE.get(key)
    if cached is None:
        return None
    return cached


def put_process_cached_jacobian(key: str, jacobian: np.ndarray) -> None:
    """Store a Jacobian under ``key`` (LRU evicts the oldest if full)."""
    arr = np.ascontiguousarray(np.asarray(jacobian))
    _PROCESS_JACOBIAN_CACHE.put(key, arr)


def clear_process_jacobian_cache() -> None:
    """Drop every cached Jacobian (test isolation + manual invalidation)."""
    _PROCESS_JACOBIAN_CACHE.clear()


def process_jacobian_cache_stats() -> dict[str, int]:
    """Return ``{"items": int, "max_items": int}``."""
    return _PROCESS_JACOBIAN_CACHE.stats()
