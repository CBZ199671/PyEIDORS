"""Absolute Gauss-Newton startup-Jacobian cache helpers.

T77 phase 2 commit #2 — second sub-module split lifted out of the
3604-line ``gauss_newton_runtime.py``. This module owns the
``_startup_cache_payload`` builder and the ``_startup_cache_lookup``
helper that load (or compute on miss) the absolute-mode start-of-run
Jacobian via the project cache manager.

The cache-key payload formula is V36/V62-style — model + pattern +
backend signatures plus a sigma SHA256 plus the solver-config block
that pins ``jacobian_update_every`` / ``jacobian_reuse_tol`` and the
ROM / inexact / lowrank tunables. The fields and their string keys
are part of the V73-style contract: existing on-disk artifacts are
keyed by this exact payload, so the field set must stay byte-stable.
``gauss_newton_runtime`` re-exports both symbols so existing call
sites (``gn_runtime._startup_cache_payload`` / ``_startup_cache_lookup``)
keep working untouched.
"""

from __future__ import annotations

import hashlib

import numpy as np
from dolfinx import fem

from ...cache.object_signature import (
    backend_signature_from_forward_model,
    model_signature_from_forward_model,
    pattern_signature_from_forward_model,
)
from ...femx import function_get_array


def _startup_cache_payload(
    reconstructor, sigma_array: np.ndarray, jacobian_method: str
) -> dict[str, object]:
    sigma_hash = hashlib.sha256(
        np.ascontiguousarray(sigma_array, dtype=np.float64).tobytes()
    ).hexdigest()
    return {
        "solver": "gn_absolute",
        "mode": str(getattr(reconstructor, "solver_mode", "strict")),
        "jacobian_method": str(jacobian_method),
        "sigma_hash": sigma_hash,
        "model_signature": model_signature_from_forward_model(reconstructor.fwd_model),
        "pattern_signature": pattern_signature_from_forward_model(
            reconstructor.fwd_model
        ),
        "backend_signature": backend_signature_from_forward_model(
            reconstructor.fwd_model
        ),
        "solver_config": {
            "linear_solver": str(getattr(reconstructor, "linear_solver", "auto")),
            "preconditioner": str(getattr(reconstructor, "preconditioner", "auto")),
            "line_search_mode": str(getattr(reconstructor, "line_search_mode", "full")),
            "jacobian_update_every": int(
                getattr(reconstructor, "jacobian_update_every", 1)
            ),
            "jacobian_reuse_tol": float(
                getattr(reconstructor, "jacobian_reuse_tol", 0.0)
            ),
            "rom_mode": str(getattr(reconstructor, "rom_mode", "off")),
            "rom_rank_global": int(getattr(reconstructor, "rom_rank_global", 32)),
            "rom_rank_adaptive": int(getattr(reconstructor, "rom_rank_adaptive", 16)),
            "rom_refresh_every": int(getattr(reconstructor, "rom_refresh_every", 2)),
            "rom_snapshot_source": str(
                getattr(reconstructor, "rom_snapshot_source", "hybrid")
            ),
            "inexact_mode": str(getattr(reconstructor, "inexact_mode", "off")),
            "inexact_forcing": str(
                getattr(reconstructor, "inexact_forcing", "eisenstat-walker")
            ),
            "inexact_eta0": float(getattr(reconstructor, "inexact_eta0", 0.2)),
            "inexact_eta_min": float(getattr(reconstructor, "inexact_eta_min", 1e-3)),
            "inexact_eta_max": float(getattr(reconstructor, "inexact_eta_max", 0.5)),
            "lowrank_mode": str(getattr(reconstructor, "lowrank_mode", "off")),
            "lowrank_rank": int(getattr(reconstructor, "lowrank_rank", 16)),
            "lowrank_method": str(getattr(reconstructor, "lowrank_method", "tsvd")),
            "lowrank_energy": float(getattr(reconstructor, "lowrank_energy", 0.995)),
        },
    }


def _startup_cache_lookup(
    reconstructor,
    sigma_current: fem.Function,
    jacobian_method: str,
) -> tuple[np.ndarray | None, dict[str, object]]:
    if (
        reconstructor.solver_mode != "fast"
        or not bool(getattr(reconstructor, "absolute_startup_cache", True))
        or getattr(reconstructor, "cache_manager", None) is None
    ):
        return None, {
            "hit": False,
            "layer": "disabled",
            "artifact": "absolute_startup_jacobian",
        }

    sigma_array = function_get_array(sigma_current)
    payload = _startup_cache_payload(reconstructor, sigma_array, jacobian_method)
    jacobian, lookup = reconstructor.cache_manager.get_or_compute_semantic(
        artifact="absolute_startup_jacobian",
        name="gn_absolute_startup_jacobian",
        namespace="absolute",
        cache_obj=payload,
        payload=payload,
        compute_fn=lambda: reconstructor.jacobian_calculator.calculate(
            sigma_current,
            method=jacobian_method,
        ),
        persist=True,
        cost=10.0,
        effort_seconds=6.0,
    )
    cached = np.asarray(jacobian, dtype=np.float64)
    if reconstructor.negate_jacobian:
        cached = -cached
    return cached, {
        "hit": bool(lookup.hit),
        "layer": str(lookup.layer),
        "artifact": str(lookup.artifact),
        "key": str(lookup.key),
    }
