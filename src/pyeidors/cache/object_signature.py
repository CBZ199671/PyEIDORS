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


def rom_signature(
    *,
    rank_global: int,
    rank_adaptive: int,
    lowrank_rank: int,
    lowrank_energy: float,
    lowrank_method: str,
    snapshot_source: str,
    snapshot_hash: str,
    refresh_every: int,
) -> str:
    """Return a stable signature hash for reduced-order model configuration."""
    payload = {
        "rank_global": int(rank_global),
        "rank_adaptive": int(rank_adaptive),
        "lowrank_rank": int(lowrank_rank),
        "lowrank_energy": float(lowrank_energy),
        "lowrank_method": str(lowrank_method),
        "snapshot_source": str(snapshot_source),
        "snapshot_hash": str(snapshot_hash),
        "refresh_every": int(refresh_every),
    }
    return stable_signature_hash(payload)


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


def _forward_model_comm_size(fwd_model: Any) -> int:
    mesh = getattr(fwd_model, "mesh", None)
    comm = getattr(mesh, "comm", None)
    try:
        if comm is not None and hasattr(comm, "Get_size"):
            return int(comm.Get_size())
        if comm is not None and hasattr(comm, "size"):
            return int(comm.size)
    except Exception:
        pass
    return 1


def _canonicalize_cuda_mat_type(mat_type: Any, *, comm_size: int) -> Any:
    if mat_type is None:
        return None
    text = str(mat_type).strip().lower()
    if text in {"aijcusparse", "seqaijcusparse", "mpiaijcusparse"}:
        return "mpiaijcusparse" if int(comm_size) > 1 else "seqaijcusparse"
    if text in {"densecuda", "seqdensecuda", "mpidensecuda"}:
        return "mpidensecuda" if int(comm_size) > 1 else "seqdensecuda"
    return text


def backend_signature_from_forward_model(fwd_model: Any) -> str:
    config = fwd_model.backend_config
    petsc_backend = getattr(fwd_model, "_petsc_backend_info", {}) or {}
    petsc_device_effective = str(petsc_backend.get("petsc_device_effective", "cpu"))
    petsc_mat_type = petsc_backend.get("petsc_mat_type")
    petsc_vec_type = petsc_backend.get("petsc_vec_type")
    petsc_dense_mat_type = petsc_backend.get("petsc_dense_mat_type")
    gpu_constraint_strategy = petsc_backend.get("gpu_constraint_strategy")
    if petsc_device_effective == "cpu" and (
        petsc_mat_type is None or petsc_vec_type is None
    ):
        stable_types_fn = getattr(fwd_model, "_stable_cpu_petsc_types", None)
        stable_mat_type = None
        stable_vec_type = None
        if callable(stable_types_fn):
            try:
                stable_mat_type, stable_vec_type = stable_types_fn()
            except Exception:
                stable_mat_type, stable_vec_type = None, None
        petsc_mat_type = petsc_mat_type or stable_mat_type or "cpu-default"
        petsc_vec_type = petsc_vec_type or stable_vec_type or "cpu-default"
    elif petsc_device_effective == "cuda":
        comm_size = _forward_model_comm_size(fwd_model)
        petsc_mat_type = _canonicalize_cuda_mat_type(
            petsc_mat_type, comm_size=comm_size
        )
        petsc_dense_mat_type = _canonicalize_cuda_mat_type(
            petsc_dense_mat_type, comm_size=comm_size
        )
        if gpu_constraint_strategy is None:
            gpu_constraint_strategy = "electrode-zero"
    payload = {
        "linear_backend": str(fwd_model.linear_backend),
        "forward_backend": str(getattr(fwd_model, "forward_backend", "dolfinx")),
        "mesh_family": str(getattr(fwd_model, "mesh_family", "tetra")),
        "geometry_version": str(getattr(fwd_model, "geometry_version", "legacy")),
        "generator_revision": str(getattr(fwd_model, "generator_revision", "g3d0")),
        "structured_sidecar_version": str(
            getattr(
                getattr(fwd_model, "eit_mesh", None), "structured_sidecar_version", None
            )
        ),
        "performance_mode": str(getattr(fwd_model, "performance_mode", "aggressive")),
        "ksp_type": str(config.ksp_type),
        "pc_type": str(config.pc_type),
        "rtol": float(config.rtol),
        "atol": float(config.atol),
        "max_it": int(config.max_it),
        "reuse_preconditioner": bool(config.reuse_preconditioner),
        "mat_solve_mode": str(getattr(config, "mat_solve_mode", "off")),
        "petsc_device": str(getattr(config, "petsc_device", "auto")),
        "petsc_device_effective": petsc_device_effective,
        "petsc_mat_type": petsc_mat_type,
        "petsc_vec_type": petsc_vec_type,
        "petsc_dense_mat_type": petsc_dense_mat_type,
        "gpu_constraint_strategy": gpu_constraint_strategy,
        "forward_backend_effective": petsc_backend.get(
            "forward_backend_effective",
            getattr(fwd_model, "forward_backend", "dolfinx"),
        ),
        "structured_backend_version": petsc_backend.get("structured_backend_version"),
        "structured_sidecar_loaded": petsc_backend.get("structured_sidecar_loaded"),
        "operator_backend": petsc_backend.get("operator_backend"),
    }
    payload.update(
        {
            "solver_preset": str(getattr(config, "solver_preset", "legacy")),
            "pc_factor_mat_solver_type": str(
                getattr(config, "pc_factor_mat_solver_type", None)
            ),
            "pc_hypre_type": str(getattr(config, "pc_hypre_type", None)),
            "pc_gamg_type": str(getattr(config, "pc_gamg_type", None)),
            "petsc_options": dict(getattr(config, "petsc_options", {}) or {}),
            "forward_pc_refresh_policy": str(
                getattr(config, "forward_pc_refresh_policy", "auto")
            ),
            "forward_pc_refresh_iter_threshold": int(
                getattr(config, "forward_pc_refresh_iter_threshold", 0) or 0
            ),
            "forward_pc_refresh_lag": int(
                getattr(config, "forward_pc_refresh_lag", 0) or 0
            ),
        }
    )
    return stable_signature_hash(payload)


def model_signature_from_forward_model(fwd_model: Any) -> str:
    cached = getattr(fwd_model, "_semantic_model_signature", None)
    if cached:
        return str(cached)

    mesh = getattr(fwd_model, "eit_mesh", None)
    if mesh is not None:
        mesh_file = getattr(mesh, "mesh_file", None)
        mesh_payload: dict[str, Any] = {
            "association_table": dict(mesh.association_table),
            "tdim": int(getattr(mesh.mesh.topology, "dim", 0)),
            "mesh_family": getattr(mesh, "mesh_family", None),
            "geometry_version": getattr(mesh, "geometry_version", None),
            "generator_revision": getattr(mesh, "generator_revision", None),
            "structured_sidecar_file": getattr(mesh, "structured_sidecar_file", None),
            "structured_sidecar_version": getattr(
                mesh, "structured_sidecar_version", None
            ),
        }
        if mesh_file:
            try:
                mesh_payload["mesh_file_hash"] = hash_path(mesh_file)
            except Exception:
                mesh_payload["mesh_file"] = str(mesh_file)
        else:
            try:
                mesh_payload["coordinates"] = np.asarray(
                    mesh.coordinates(), dtype=np.float64
                )
                mesh_payload["cells"] = np.asarray(mesh.cells(), dtype=np.int32)
            except Exception:
                pass
    else:
        mesh_payload = {"mesh": "missing"}

    payload = {
        "n_elec": int(fwd_model.n_elec),
        "z": np.asarray(fwd_model.z, dtype=np.float64),
        "geometry_scale_to_m": float(getattr(fwd_model, "geometry_scale_to_m", 1.0)),
        "mesh": mesh_payload,
    }
    signature = stable_signature_hash(payload)
    setattr(fwd_model, "_semantic_model_signature", signature)
    return signature
