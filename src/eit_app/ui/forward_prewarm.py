"""Lightweight helpers for simulation forward backend prewarm."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from eit_app.models.forward_model_config import mapping_complex_value


def simulation_signature_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return simulation_signature_value(value.item())
    if isinstance(value, np.ndarray):
        return simulation_signature_value(value.tolist())
    if isinstance(value, complex):
        return mapping_complex_value(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): simulation_signature_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [simulation_signature_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def backend_forward_setup_warm_key(
    *,
    profile: str,
    request: Any,
    setup_prime: bool,
) -> str:
    profile_name = str(profile or "default").strip() or "default"
    if not setup_prime:
        return profile_name
    config = dict(request.forward_model_config or {})
    volatile_or_non_setup_keys = {
        "background_conductivity",
        "noise_level",
        "request_source",
        "simulation_input_signature",
        "simulation_input_signature_payload",
    }

    def _setup_config(payload: dict[str, object]) -> dict[str, object]:
        return {
            str(key): value
            for key, value in payload.items()
            if str(key) not in volatile_or_non_setup_keys
            and str(key) != "inhomogeneities"
            and not str(key).startswith("inhomogeneities_")
        }

    setup_payload = {
        "schema": "simulation_forward_setup_prime_v1",
        "profile": profile_name,
        "mesh_dimension": int(request.mesh_dimension),
        "mesh_refinement": float(request.mesh_refinement),
        "n_electrodes": int(request.n_electrodes),
        "background_is_complex": bool(
            np.iscomplexobj(np.asarray(request.background_conductivity))
        ),
        "forward_model_config": _setup_config(config),
    }
    signature_payload = config.get("simulation_input_signature_payload")
    if isinstance(signature_payload, dict):
        signature_config = signature_payload.get("forward_model_config")
        if isinstance(signature_config, dict):
            setup_payload["forward_model_config"] = _setup_config(
                dict(signature_config)
            )
    canonical = simulation_signature_value(setup_payload)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{profile_name}:setup:{hashlib.sha256(encoded).hexdigest()[:16]}"


def sim_forward_prewarm_mode(*, mesh_dimension: int) -> str:
    if int(mesh_dimension) != 3:
        return "solve"
    raw = os.getenv("EIT_APP_FORWARD_PREWARM_3D_MODE", "setup").strip().lower()
    if raw in {"0", "false", "no", "off", "none", "disabled"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "solve", "full"}:
        return "solve"
    if raw in {"setup", "forward_setup", "jit", "prime", "setup_prime"}:
        return "setup"
    return "worker"


def _backend_worker_probe_summary(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict):
        return {
            "petsc_cuda": None,
            "cache_hit": None,
            "cache_layer": "",
            "status_text": "--",
        }
    nested_probe = metadata.get("petsc_cuda_probe")
    if isinstance(nested_probe, dict):
        probe_cache = nested_probe.get("probe_cache")
        petsc_cuda = nested_probe.get("petsc_cuda")
    else:
        probe_cache = metadata.get("petsc_cuda_probe_cache")
        petsc_cuda = metadata.get("petsc_cuda")
    if not isinstance(probe_cache, dict):
        probe_cache = {}
    cache_hit_raw = probe_cache.get("hit")
    cache_hit = bool(cache_hit_raw) if cache_hit_raw is not None else None
    cache_layer = str(probe_cache.get("layer", "") or "")
    if cache_hit is None:
        status_text = "--"
    else:
        hit_label = "hit" if cache_hit else "miss"
        status_text = f"{hit_label}/{cache_layer}" if cache_layer else hit_label
    return {
        "petsc_cuda": petsc_cuda if isinstance(petsc_cuda, bool) else None,
        "cache_hit": cache_hit,
        "cache_layer": cache_layer,
        "status_text": status_text,
    }


def simulation_backend_warm_report(
    meta: object,
    *,
    profile: str,
    warm_key: str,
    setup_prime: bool,
) -> dict[str, object]:
    prime_metadata = dict(getattr(meta, "prime_metadata", {}) or {})
    probe_summary = _backend_worker_probe_summary(prime_metadata)
    return {
        "profile": str(getattr(meta, "profile", profile)),
        "pid": int(getattr(meta, "pid", 0) or 0),
        "rss_bytes": int(getattr(meta, "rss_bytes", 0) or 0),
        "rss_limit_bytes": int(getattr(meta, "rss_limit_bytes", 0) or 0),
        "primed_runtime": bool(getattr(meta, "primed_runtime", False)),
        "prime_command": str(getattr(meta, "prime_command", "") or ""),
        "prime_duration_ms": float(getattr(meta, "prime_duration_ms", 0.0) or 0.0),
        "prime_metadata": prime_metadata,
        "petsc_cuda_probe_cache_hit": probe_summary["cache_hit"],
        "petsc_cuda_probe_cache_layer": probe_summary["cache_layer"],
        "petsc_cuda_probe_status": probe_summary["status_text"],
        "petsc_cuda_available": probe_summary["petsc_cuda"],
        "request_duration_ms": float(getattr(meta, "request_duration_ms", 0.0) or 0.0),
        "setup_prime": bool(setup_prime),
        "warm_key": warm_key,
        "recycled_after_request": bool(getattr(meta, "recycled_after_request", False)),
        "recycle_reason": str(getattr(meta, "recycle_reason", "") or ""),
    }
