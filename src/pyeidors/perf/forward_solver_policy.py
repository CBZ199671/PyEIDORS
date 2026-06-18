"""Forward PETSc solver policy helpers for high-level GPU runtimes."""

from __future__ import annotations

from typing import Any

CUDA_HYPRE_BLACKLISTED_PRESETS = frozenset(
    {"3d_hypre", "hypre_boomeramg", "spd_hypre", "cg_hypre"}
)
CUDA_HYPRE_BLACKLIST_REASON = "hypre_cuda_blacklisted_sigsegv_b4"
AMGX_DOWNGRADE_REASON = "amgx_unavailable_downgraded_to_spd_gamg"
CUDA_GAMG_DEFAULT_REASON = "3d_cuda_spd_gamg_default"
TETRA_CUDA_GAMG_DEFAULT_REASON = "tetra_cuda_3d_gamg_default"
TETRA_REAL_AMGX_DEFAULT_REASON = "tetra_real_cuda_amgx_default"
COMPLEX_CUDA_NATIVE_GAMG_DEFAULT_REASON = "complex_cuda_native_gamg_default"
COMPLEX_CUDA_BLOCK_REAL_AMGX_DEFAULT_REASON = "complex_cuda_block_real_amgx_default"
TETRA_COMPLEX_BLOCK_REAL_AMGX_DEFAULT_REASON = (
    COMPLEX_CUDA_BLOCK_REAL_AMGX_DEFAULT_REASON
)
TETRA_AMGX_DOWNGRADE_REASON = "tetra_amgx_unavailable_downgraded_to_3d_gamg"
TETRA_HYPRE_BLACKLIST_REASON = "tetra_hypre_cuda_blacklisted_to_3d_gamg"
CUDA_SPD_GAMG_MATSOLVE_DISABLED_REASON = "cuda_spd_gamg_matsolve_disabled_b6"
CUDA_GAMG_MATSOLVE_DISABLED_REASON = "cuda_gamg_matsolve_disabled_b658"
CUDA_AMGX_MATSOLVE_DISABLED_REASON = "cuda_amgx_matsolve_disabled_mainline"


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
    mesh_family: Any = "auto",
    capability: dict[str, Any] | None = None,
    prefer_amgx: bool = True,
    complex_admittivity_requested: bool = False,
    complex_high_accuracy: bool = False,
) -> dict[str, Any]:
    """Resolve high-level 3D CUDA forward solver preset.

    This helper is intentionally policy-level: it may downgrade risky or
    unavailable high-level choices before the low-level PETSc setup runs. Direct
    low-level users still get fail-fast errors in :mod:`eit_forward_model`.
    """
    requested = _token(requested_solver_preset)
    device = _token(petsc_device)
    backend = _token(forward_backend, "dolfinx")
    family = _token(mesh_family, "auto")
    cap = dict(capability or {})
    amgx_available = bool(cap.get("petsc_amgx", False))
    hypre_available = bool(cap.get("petsc_hypre", False))
    complex_requested = bool(complex_admittivity_requested)
    strict_complex = bool(complex_high_accuracy)

    effective = requested
    reason = ""
    warning = ""
    blacklisted = False
    active = int(mesh_dim) == 3 and device == "cuda" and backend == "dolfinx"
    tetra_cuda = active and family == "tetra"

    if active:
        if requested in {"", "auto"} and complex_requested:
            if strict_complex:
                effective = "complex_block_real_amgx"
                reason = COMPLEX_CUDA_BLOCK_REAL_AMGX_DEFAULT_REASON
            else:
                effective = "3d_gamg"
                reason = COMPLEX_CUDA_NATIVE_GAMG_DEFAULT_REASON
        elif tetra_cuda and requested in {"", "auto"}:
            if prefer_amgx and amgx_available:
                effective = "cuda_amgx"
                reason = TETRA_REAL_AMGX_DEFAULT_REASON
            else:
                effective = "3d_gamg"
                reason = TETRA_AMGX_DOWNGRADE_REASON
                warning = "PETSc AmgX unavailable; using 3d_gamg CUDA instead."
        elif requested in {"", "auto"}:
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
            effective = "3d_gamg" if tetra_cuda else "spd_gamg"
            reason = (
                TETRA_AMGX_DOWNGRADE_REASON if tetra_cuda else AMGX_DOWNGRADE_REASON
            )
            warning = (
                "PETSc AmgX unavailable; using 3d_gamg CUDA instead."
                if tetra_cuda
                else "PETSc AmgX unavailable; using spd_gamg CUDA instead."
            )
        elif requested in CUDA_HYPRE_BLACKLISTED_PRESETS:
            effective = "3d_gamg" if tetra_cuda else "spd_gamg"
            reason = (
                TETRA_HYPRE_BLACKLIST_REASON
                if tetra_cuda
                else CUDA_HYPRE_BLACKLIST_REASON
            )
            warning = (
                "PETSc Hypre CUDA route is blacklisted after B4 SIGSEGV; "
                f"using {effective} CUDA instead."
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


def resolve_3d_cuda_mat_solve_policy(
    *,
    requested_mat_solve: Any = "auto",
    mesh_dim: int,
    petsc_device: Any,
    forward_backend: Any,
    solver_preset: Any,
) -> dict[str, Any]:
    """Resolve high-level 3D CUDA multi-RHS policy.

    PETSc ``KSPMatSolve`` is not a stable production default for the current
    ``spd_gamg + CUDA`` route. Keep explicit ``on`` available for experiments,
    but make GUI/runtime ``auto`` use the proven vector RHS loop.
    """
    requested = _token(requested_mat_solve)
    device = _token(petsc_device)
    backend = _token(forward_backend, "dolfinx")
    solver = _token(solver_preset)

    effective = requested
    reason = ""
    warning = ""
    active = int(mesh_dim) == 3 and device == "cuda" and backend == "dolfinx"
    if (
        active
        and requested == "auto"
        and solver
        in {
            "spd_gamg",
            "3d_gamg",
            "3d_amg",
            "cuda_amgx",
            "complex_block_real_amgx",
        }
    ):
        effective = "off"
        if solver == "spd_gamg":
            reason = CUDA_SPD_GAMG_MATSOLVE_DISABLED_REASON
        elif solver in {"cuda_amgx", "complex_block_real_amgx"}:
            reason = CUDA_AMGX_MATSOLVE_DISABLED_REASON
        else:
            reason = CUDA_GAMG_MATSOLVE_DISABLED_REASON
        warning = (
            f"PETSc KSPMatSolve is disabled for {solver} CUDA; using vector RHS loop."
        )

    return {
        "forward_mat_solve_requested": requested,
        "forward_mat_solve_effective_policy": effective,
        "forward_mat_solve_policy_reason": reason,
        "forward_mat_solve_policy_warning": warning,
    }


__all__ = [
    "AMGX_DOWNGRADE_REASON",
    "CUDA_GAMG_DEFAULT_REASON",
    "CUDA_GAMG_MATSOLVE_DISABLED_REASON",
    "CUDA_AMGX_MATSOLVE_DISABLED_REASON",
    "COMPLEX_CUDA_NATIVE_GAMG_DEFAULT_REASON",
    "COMPLEX_CUDA_BLOCK_REAL_AMGX_DEFAULT_REASON",
    "CUDA_HYPRE_BLACKLIST_REASON",
    "CUDA_HYPRE_BLACKLISTED_PRESETS",
    "CUDA_SPD_GAMG_MATSOLVE_DISABLED_REASON",
    "TETRA_AMGX_DOWNGRADE_REASON",
    "TETRA_CUDA_GAMG_DEFAULT_REASON",
    "TETRA_COMPLEX_BLOCK_REAL_AMGX_DEFAULT_REASON",
    "TETRA_REAL_AMGX_DEFAULT_REASON",
    "TETRA_HYPRE_BLACKLIST_REASON",
    "is_hypre_cuda_blacklisted_solver",
    "resolve_3d_cuda_forward_solver_policy",
    "resolve_3d_cuda_mat_solve_policy",
]
