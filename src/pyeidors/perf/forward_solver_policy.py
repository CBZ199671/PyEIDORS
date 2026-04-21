"""Forward PETSc solver policy helpers for high-level GPU runtimes."""

from __future__ import annotations

from typing import Any

CUDA_HYPRE_BLACKLISTED_PRESETS = frozenset(
    {"3d_hypre", "hypre_boomeramg", "spd_hypre", "cg_hypre"}
)
CUDA_HYPRE_BLACKLIST_REASON = "hypre_cuda_blacklisted_sigsegv_b4"
AMGX_DOWNGRADE_REASON = "amgx_unavailable_downgraded_to_spd_gamg"
CUDA_GAMG_DEFAULT_REASON = "3d_cuda_spd_gamg_default"


def _token(value: Any, default: str = "auto") -> str:
    raw = str(value if value is not None else default).strip().lower()
    return raw or default


def is_hypre_cuda_blacklisted_solver(
    *,
    solver_preset: Any = "auto",
    pc_type: Any = "",
) -> bool:
    """Return True for PETSc Hypre CUDA routes known unsafe in B4."""
    preset = _token(solver_preset)
    pc = _token(pc_type, "")
    return preset in CUDA_HYPRE_BLACKLISTED_PRESETS or pc == "hypre"


def resolve_3d_cuda_forward_solver_policy(
    *,
    requested_solver_preset: Any = "auto",
    mesh_dim: int,
    petsc_device: Any,
    forward_backend: Any,
    capability: dict[str, Any] | None = None,
    prefer_amgx: bool = True,
) -> dict[str, Any]:
    """Resolve high-level 3D CUDA forward solver preset.

    This helper is intentionally policy-level: it may downgrade risky or
    unavailable high-level choices before the low-level PETSc setup runs. Direct
    low-level users still get fail-fast errors in :mod:`eit_forward_model`.
    """
    requested = _token(requested_solver_preset)
    device = _token(petsc_device)
    backend = _token(forward_backend, "dolfinx")
    cap = dict(capability or {})
    amgx_available = bool(cap.get("petsc_amgx", False))
    hypre_available = bool(cap.get("petsc_hypre", False))

    effective = requested
    reason = ""
    warning = ""
    blacklisted = False
    active = int(mesh_dim) == 3 and device == "cuda" and backend == "dolfinx"

    if active:
        if requested in {"", "auto"}:
            if prefer_amgx and amgx_available:
                effective = "cuda_amgx"
                reason = "amgx_available_selected"
            elif prefer_amgx:
                effective = "spd_gamg"
                reason = AMGX_DOWNGRADE_REASON
                warning = "PETSc AmgX unavailable; using spd_gamg CUDA instead."
            else:
                effective = "spd_gamg"
                reason = CUDA_GAMG_DEFAULT_REASON
        elif requested in {"amgx", "cuda_amgx"} and not amgx_available:
            effective = "spd_gamg"
            reason = AMGX_DOWNGRADE_REASON
            warning = "PETSc AmgX unavailable; using spd_gamg CUDA instead."
        elif requested in CUDA_HYPRE_BLACKLISTED_PRESETS:
            effective = "spd_gamg"
            reason = CUDA_HYPRE_BLACKLIST_REASON
            warning = (
                "PETSc Hypre CUDA route is blacklisted after B4 SIGSEGV; "
                "using spd_gamg CUDA instead."
            )
            blacklisted = True

    return {
        "forward_solver_preset_requested": requested,
        "forward_solver_preset_effective": effective,
        "forward_solver_policy_reason": reason,
        "forward_solver_policy_warning": warning,
        "petsc_amgx_available": amgx_available,
        "petsc_hypre_available": hypre_available,
        "petsc_hypre_cuda_blacklisted": blacklisted,
    }


__all__ = [
    "AMGX_DOWNGRADE_REASON",
    "CUDA_GAMG_DEFAULT_REASON",
    "CUDA_HYPRE_BLACKLIST_REASON",
    "CUDA_HYPRE_BLACKLISTED_PRESETS",
    "is_hypre_cuda_blacklisted_solver",
    "resolve_3d_cuda_forward_solver_policy",
]
