"""Process-local cache for forward-model static setup bundles.

T79 Path C: the OrderedDict LRU + threading.Lock + JSON-hash key
machinery moved to :mod:`pyeidors.cache.process_lru`; this module
pins the value type to :class:`ForwardStaticSetupBundle` and keeps the
historical public function names so ``EITForwardModel`` callers stay
intact. The cache-key payload is still JSON-normalized through
:func:`pyeidors.cache.process_lru.hash_json_payload`; current schema also
records the active PETSc scalar dtype so real and complex CEM setup objects
cannot collide in process memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from ..cache.keys import _normalize, hash_array
from ..cache.process_lru import ProcessLRUCache, env_bytes_limit, hash_json_payload
from ..data.structures import PatternConfig
from ..utils.numeric_ops import has_nonzero_imaginary


@dataclass(frozen=True)
class ForwardStaticSetupBundle:
    """Read-only static objects shared across compatible forward models."""

    ds_electrodes: Any
    electrode_tags: tuple[int, ...]
    electrode_boundary_measures: dict[int, float]
    geometry_scale_to_m: float
    mesh_tdim: int
    boundary_scale_to_m: float
    electrode_lengths_m: np.ndarray
    pattern_manager: Any
    V: Any
    V_sigma: Any
    dofs: int
    electrode_matrix: csr_matrix


_PROCESS_FORWARD_SETUP_CACHE_MAX_ITEMS = 8
_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES = 512 * 1024 * 1024


def _array_nbytes(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(np.asarray(value).nbytes)
    except Exception:
        return 0


def _csr_nbytes(matrix: csr_matrix) -> int:
    return (
        _array_nbytes(getattr(matrix, "data", None))
        + _array_nbytes(getattr(matrix, "indices", None))
        + _array_nbytes(getattr(matrix, "indptr", None))
    )


def _forward_setup_bundle_size_bytes(bundle: ForwardStaticSetupBundle) -> int:
    total = _array_nbytes(bundle.electrode_lengths_m)
    total += _csr_nbytes(bundle.electrode_matrix)
    return max(int(total), 1)


_RESOLVED_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES = env_bytes_limit(
    "PYEIDORS_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES",
    "EIT_APP_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES",
    default=_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES,
)
_PROCESS_FORWARD_SETUP_CACHE: ProcessLRUCache[ForwardStaticSetupBundle] = (
    ProcessLRUCache(
        max_items=_PROCESS_FORWARD_SETUP_CACHE_MAX_ITEMS,
        max_bytes=_RESOLVED_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES,
        sizeof=_forward_setup_bundle_size_bytes,
    )
)


def _pattern_signature(config: PatternConfig) -> dict[str, Any]:
    return _normalize(asdict(config))


def build_process_forward_setup_key(
    *,
    mesh_file: str | None,
    n_elec: int,
    z: np.ndarray,
    pattern_config: PatternConfig,
    mesh_content_hash: str | None = None,
    potential_order: int = 1,
    scalar_dtype: str | np.dtype | None = None,
) -> str:
    """Build a content-addressed cache key for forward static setup.

    A stable cache key requires at least one of ``mesh_file`` (file-backed
    mesh) or ``mesh_content_hash`` (in-memory mesh). Earlier revisions mixed
    ``id(self.mesh)`` into the key, which was unsafe: Python is free to reuse
    addresses after garbage collection, so an in-memory mesh could share an
    id with a freshly allocated replacement.
    """
    file_token = str(mesh_file or "")
    content_token = str(mesh_content_hash or "")
    if not file_token and not content_token:
        raise ValueError(
            "build_process_forward_setup_key requires either mesh_file "
            "or mesh_content_hash to form a stable cache key."
        )
    z_dtype = (
        np.dtype(scalar_dtype)
        if scalar_dtype is not None
        else np.dtype(np.complex128 if np.iscomplexobj(z) else np.float64)
    )
    z_array = np.asarray(z)
    if not np.issubdtype(z_dtype, np.complexfloating) and np.iscomplexobj(z_array):
        if has_nonzero_imaginary(z_array):
            raise RuntimeError(
                "complex contact impedance requires a complex PETSc/DOLFINx "
                "runtime; use nix develop .#complex or .#complex64"
            )
        z_array = np.real(z_array)
    payload = {
        "mesh_file": file_token,
        "mesh_content_hash": content_token,
        "n_elec": int(n_elec),
        "potential_order": int(potential_order),
        "scalar_dtype": str(z_dtype),
        "z_hash": hash_array(np.asarray(z_array, dtype=z_dtype).reshape(-1)),
        "pattern_config": _pattern_signature(pattern_config),
    }
    return hash_json_payload(payload)


def get_process_forward_setup_bundle(key: str) -> ForwardStaticSetupBundle | None:
    return _PROCESS_FORWARD_SETUP_CACHE.get(key)


def put_process_forward_setup_bundle(
    key: str, bundle: ForwardStaticSetupBundle
) -> None:
    max_bytes = int(_RESOLVED_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES)
    if max_bytes <= 0 or _forward_setup_bundle_size_bytes(bundle) > max_bytes:
        _PROCESS_FORWARD_SETUP_CACHE.discard(key)
        return
    _PROCESS_FORWARD_SETUP_CACHE.put(key, bundle)


def clear_process_forward_setup_cache() -> None:
    _PROCESS_FORWARD_SETUP_CACHE.clear()


def process_forward_setup_cache_stats() -> dict[str, int]:
    return _PROCESS_FORWARD_SETUP_CACHE.stats()
