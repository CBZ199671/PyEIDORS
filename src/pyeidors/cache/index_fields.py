"""Queryable cache index metadata extracted from cache payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

CACHE_INDEX_FIELD_NAMES = (
    "dtype",
    "backend",
    "device",
    "dim",
    "n_elec",
    "mesh_hash",
)

_TEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "dtype": (
        "dtype",
        "scalar_dtype",
        "sigma_dtype",
        "rm_dtype",
        "matrix_dtype",
        "compute_dtype",
        "petsc_scalar_dtype",
        "petsc_scalar_type",
    ),
    "backend": (
        "backend",
        "linear_backend",
        "forward_backend_effective",
        "forward_backend",
        "solver_backend",
        "runtime_backend",
    ),
    "device": (
        "device",
        "petsc_device_effective",
        "petsc_device",
        "effective_device",
        "rm_device",
        "cuda_device",
    ),
    "mesh_hash": (
        "mesh_hash",
        "mesh_file_hash",
        "mesh_content_hash",
        "mesh_sha256",
    ),
}

_INT_ALIASES: dict[str, tuple[str, ...]] = {
    "dim": ("dim", "tdim", "gdim", "mesh_dim", "mesh_dimension", "dimension"),
    "n_elec": ("n_elec", "n_electrodes", "electrode_count"),
}


def normalize_cache_index_fields(
    fields: Mapping[str, Any] | None,
) -> dict[str, str | int | None]:
    """Normalize optional cache index fields for stores and queries."""

    source = fields or {}
    normalized: dict[str, str | int | None] = {
        "dtype": _normalize_text(source.get("dtype")),
        "backend": _normalize_text(source.get("backend")),
        "device": _normalize_text(source.get("device")),
        "dim": _normalize_int(source.get("dim")),
        "n_elec": _normalize_int(source.get("n_elec")),
        "mesh_hash": _normalize_text(source.get("mesh_hash")),
    }
    return normalized


def extract_cache_index_fields(
    payload: Mapping[str, Any],
) -> dict[str, str | int | None]:
    """Best-effort extraction of queryable cache dimensions from key payloads.

    The extracted values are metadata only: they never participate in the cache
    key beyond whatever the original payload already encoded.
    """

    explicit = _explicit_index_mapping(payload)
    result = normalize_cache_index_fields(explicit)

    petsc_backend = payload.get("petsc_backend")
    if result["device"] is None and isinstance(petsc_backend, Mapping):
        result["device"] = _normalize_text(
            petsc_backend.get("petsc_device_effective")
            or petsc_backend.get("effective")
            or petsc_backend.get("requested")
        )

    backend_config = payload.get("backend_config")
    if result["device"] is None and isinstance(backend_config, Mapping):
        result["device"] = _normalize_text(backend_config.get("petsc_device"))

    for field, aliases in _TEXT_ALIASES.items():
        if result[field] is None:
            result[field] = _normalize_text(_first_alias_value(payload, aliases))
    for field, aliases in _INT_ALIASES.items():
        if result[field] is None:
            result[field] = _normalize_int(_first_alias_value(payload, aliases))
    return result


def _explicit_index_mapping(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("cache_index", "index_fields"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _first_alias_value(payload: Any, aliases: tuple[str, ...]) -> Any:
    alias_set = {str(alias) for alias in aliases}
    for mapping in _iter_mappings(payload):
        for key in aliases:
            if key in mapping:
                return mapping[key]
        for key, value in mapping.items():
            if str(key) in alias_set:
                return value
    return None


def _iter_mappings(value: Any, *, max_depth: int = 6) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    queue: list[tuple[Any, int]] = [(value, 0)]
    while queue:
        current, depth = queue.pop(0)
        if depth > max_depth:
            continue
        if isinstance(current, Mapping):
            found.append(current)
            for child in current.values():
                if isinstance(child, Mapping):
                    queue.append((child, depth + 1))
                elif _is_small_sequence(child):
                    queue.extend((item, depth + 1) for item in child)
        elif _is_small_sequence(current):
            queue.extend((item, depth + 1) for item in current)
    return found


def _is_small_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return False
    return isinstance(value, Sequence) and len(value) <= 32


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _normalize_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric
