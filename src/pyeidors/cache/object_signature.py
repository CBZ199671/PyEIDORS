"""Semantic cache object signatures inspired by EIDORS var-id behavior."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .keys import hash_array, hash_path


def _normalize_callable(func: Callable[..., Any]) -> dict[str, Any]:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", "callable"))
    payload: dict[str, Any] = {"module": module, "qualname": qualname}
    try:
        source_file = inspect.getsourcefile(func) or inspect.getfile(func)
    except Exception:
        source_file = None
    if source_file:
        payload["source_hash"] = hash_path(source_file)
        payload["source_path"] = str(Path(source_file).resolve())
    return payload


def _normalize_for_signature(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        array = np.ascontiguousarray(obj)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "hash": hash_array(array),
        }
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, Path):
        return {"__path__": str(obj.resolve()), "path_hash": hash_path(obj)}
    if isinstance(obj, bytes):
        return {"__bytes__": hashlib.sha256(obj).hexdigest()}
    if isinstance(obj, dict):
        return {
            str(k): _normalize_for_signature(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_signature(v) for v in obj]
    if isinstance(obj, set):
        normalized = [_normalize_for_signature(v) for v in obj]
        return sorted(normalized, key=lambda v: json.dumps(v, sort_keys=True))
    if hasattr(obj, "__dataclass_fields__"):
        return _normalize_for_signature(asdict(obj))
    if callable(obj):
        return {"__callable__": _normalize_callable(obj)}
    return obj


def signature_of_cache_obj(cache_obj: Any) -> dict[str, Any]:
    """Return a normalized signature payload for cache object dependencies."""
    return {"cache_obj": _normalize_for_signature(cache_obj)}


def stable_signature_hash(cache_obj: Any) -> str:
    """Return deterministic SHA-256 signature for arbitrary dependency objects."""
    payload = signature_of_cache_obj(cache_obj)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pattern_signature_from_forward_model(fwd_model: Any) -> str:
    manager = fwd_model.pattern_manager
    payload = {
        "stim_matrix": np.asarray(manager.stim_matrix, dtype=np.float64),
        "meas_matrices": [
            np.asarray(matrix, dtype=np.float64) for matrix in manager.meas_matrices
        ],
        "n_stim": int(manager.n_stim),
        "n_meas_total": int(manager.n_meas_total),
        "n_meas_per_stim": list(int(v) for v in manager.n_meas_per_stim),
    }
    return stable_signature_hash(payload)


def backend_signature_from_forward_model(fwd_model: Any) -> str:
    config = fwd_model.backend_config
    payload = {
        "linear_backend": str(fwd_model.linear_backend),
        "performance_mode": str(getattr(fwd_model, "performance_mode", "aggressive")),
        "ksp_type": str(config.ksp_type),
        "pc_type": str(config.pc_type),
        "rtol": float(config.rtol),
        "atol": float(config.atol),
        "max_it": int(config.max_it),
        "reuse_preconditioner": bool(config.reuse_preconditioner),
    }
    return stable_signature_hash(payload)


def model_signature_from_forward_model(fwd_model: Any) -> str:
    cached = getattr(fwd_model, "_semantic_model_signature", None)
    if cached:
        return str(cached)

    mesh = getattr(fwd_model, "eit_mesh", None)
    if mesh is not None:
        mesh_payload: dict[str, Any] = {
            "mesh_file": mesh.mesh_file,
            "radius": float(mesh.radius),
            "n_vertices": int(mesh.num_vertices()),
            "n_cells": int(mesh.num_cells()),
            "association_table": dict(mesh.association_table),
        }
        try:
            mesh_payload["coordinates"] = np.asarray(mesh.coordinates(), dtype=np.float64)
            mesh_payload["cells"] = np.asarray(mesh.cells(), dtype=np.int32)
        except Exception:
            pass
    else:
        mesh_payload = {"mesh": "missing"}

    payload = {
        "n_elec": int(fwd_model.n_elec),
        "z": np.asarray(fwd_model.z, dtype=np.float64),
        "electrode_tags": list(getattr(fwd_model, "electrode_tags", [])),
        "electrode_lengths_m": np.asarray(
            getattr(fwd_model, "electrode_lengths_m", []), dtype=np.float64
        ),
        "geometry_scale_to_m": float(getattr(fwd_model, "geometry_scale_to_m", 1.0)),
        "mesh": mesh_payload,
    }
    signature = stable_signature_hash(payload)
    setattr(fwd_model, "_semantic_model_signature", signature)
    return signature

