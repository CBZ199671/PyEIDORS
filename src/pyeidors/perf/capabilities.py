"""Runtime capability detection and backend selection for performance paths."""

from __future__ import annotations

from functools import lru_cache

from .policy import FEATURE_MODE_AUTO, normalize_feature_mode


def _load_petsc_runtime():
    try:
        from ..forward.eit_forward_model import PETSc
    except Exception:
        return None
    return PETSc


def _has_cuda_structured() -> bool:
    try:
        from ..forward.cuda_structured_backend import _torch_cuda_available
    except Exception:
        return False
    return bool(_torch_cuda_available())


def _has_pyamg() -> bool:
    try:
        import pyamg  # noqa: F401
    except Exception:
        return False
    return True


def _has_cholmod() -> bool:
    try:
        from sksparse.cholmod import cholesky  # noqa: F401
    except Exception:
        return False
    return True


def _has_petsc_mat_solve() -> bool:
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return False
    return hasattr(PETSc.KSP, "matSolve")


def _has_petsc_gamg() -> bool:
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return False
    pc_type = getattr(getattr(PETSc, "PC", None), "Type", None)
    return pc_type is not None and hasattr(pc_type, "GAMG")


def _enum_name(namespace, name: str) -> str | None:
    if namespace is None or not hasattr(namespace, name):
        return None
    try:
        return str(getattr(namespace, name))
    except Exception:
        return None


def _create_mat(PETSc):
    mat = PETSc.Mat()
    creator = getattr(mat, "create", None)
    if callable(creator):
        comm = getattr(PETSc, "COMM_SELF", None)
        try:
            creator(comm=comm)
        except TypeError:
            creator()
    return mat


def _create_vec(PETSc):
    vec = PETSc.Vec()
    creator = getattr(vec, "create", None)
    if callable(creator):
        comm = getattr(PETSc, "COMM_SELF", None)
        try:
            creator(comm=comm)
        except TypeError:
            creator()
    return vec


def _probe_petsc_mat_type(type_name: str | None) -> tuple[bool, str | None]:
    if not type_name:
        return False, "mat_type_symbol_missing"
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return False, "petsc_unavailable"
    mat = None
    try:
        mat = _create_mat(PETSc)
        mat.setSizes((1, 1))
        mat.setType(type_name)
        if hasattr(mat, "setPreallocationNNZ"):
            mat.setPreallocationNNZ(1)
        if hasattr(mat, "setUp"):
            mat.setUp()
        if hasattr(mat, "setValue"):
            mat.setValue(0, 0, 1.0)
        if hasattr(mat, "assemblyBegin"):
            mat.assemblyBegin()
        if hasattr(mat, "assemblyEnd"):
            mat.assemblyEnd()
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if mat is not None and hasattr(mat, "destroy"):
            try:
                mat.destroy()
            except Exception:
                pass


def _probe_petsc_vec_type(type_name: str | None) -> tuple[bool, str | None]:
    if not type_name:
        return False, "vec_type_symbol_missing"
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return False, "petsc_unavailable"
    vec = None
    try:
        vec = _create_vec(PETSc)
        vec.setSizes(1)
        vec.setType(type_name)
        if hasattr(vec, "setUp"):
            vec.setUp()
        if hasattr(vec, "setValue"):
            vec.setValue(0, 1.0)
        if hasattr(vec, "assemblyBegin"):
            vec.assemblyBegin()
        if hasattr(vec, "assemblyEnd"):
            vec.assemblyEnd()
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if vec is not None and hasattr(vec, "destroy"):
            try:
                vec.destroy()
            except Exception:
                pass


def _petsc_runtime_cache_key() -> tuple[object, ...]:
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return ("petsc:none",)
    return (
        id(PETSc),
        _enum_name(getattr(getattr(PETSc, "Mat", None), "Type", None), "AIJCUSPARSE"),
        _enum_name(getattr(getattr(PETSc, "Vec", None), "Type", None), "CUDA"),
        _enum_name(getattr(getattr(PETSc, "Mat", None), "Type", None), "DENSECUDA"),
        bool(hasattr(getattr(PETSc, "KSP", None), "matSolve")),
        bool(hasattr(getattr(getattr(PETSc, "PC", None), "Type", None), "GAMG")),
    )


@lru_cache(maxsize=8)
def _probe_petsc_cuda_runtime_cached(runtime_key: tuple[object, ...]) -> dict[str, object]:
    del runtime_key
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return {
            "petsc_available": False,
            "petsc_cuda": False,
            "petsc_cuda_mat": False,
            "petsc_cuda_vec": False,
            "petsc_cuda_dense": False,
            "mat_type_name": None,
            "vec_type_name": None,
            "dense_mat_type_name": None,
            "errors": {"petsc": "petsc_unavailable"},
        }

    mat_type = _enum_name(getattr(getattr(PETSc, "Mat", None), "Type", None), "AIJCUSPARSE")
    vec_type = _enum_name(getattr(getattr(PETSc, "Vec", None), "Type", None), "CUDA")
    dense_type = _enum_name(getattr(getattr(PETSc, "Mat", None), "Type", None), "DENSECUDA")

    mat_ok, mat_error = _probe_petsc_mat_type(mat_type)
    vec_ok, vec_error = _probe_petsc_vec_type(vec_type)
    dense_ok, dense_error = _probe_petsc_mat_type(dense_type)

    errors: dict[str, str] = {}
    if mat_error:
        errors["mat"] = mat_error
    if vec_error:
        errors["vec"] = vec_error
    if dense_error:
        errors["dense"] = dense_error

    return {
        "petsc_available": True,
        "petsc_cuda": bool(mat_ok and vec_ok),
        "petsc_cuda_mat": bool(mat_ok),
        "petsc_cuda_vec": bool(vec_ok),
        "petsc_cuda_dense": bool(dense_ok),
        "mat_type_name": mat_type,
        "vec_type_name": vec_type,
        "dense_mat_type_name": dense_type,
        "errors": errors,
    }


def probe_petsc_cuda_runtime() -> dict[str, object]:
    """Probe whether PETSc CUDA matrix/vector backends are truly usable.

    We intentionally verify actual object creation because some PETSc builds expose
    enum names like ``AIJCUSPARSE``/``CUDA`` while failing at runtime with
    ``Unknown type`` when CUDA support was not compiled in.
    """
    return _probe_petsc_cuda_runtime_cached(_petsc_runtime_cache_key())


probe_petsc_cuda_runtime.cache_clear = _probe_petsc_cuda_runtime_cached.cache_clear
probe_petsc_cuda_runtime.cache_info = _probe_petsc_cuda_runtime_cached.cache_info


@lru_cache(maxsize=8)
def _detect_performance_capabilities_cached(cache_key: tuple[object, ...]) -> dict[str, bool]:
    del cache_key
    cuda_probe = probe_petsc_cuda_runtime()
    return {
        "pyamg": _has_pyamg(),
        "cholmod": _has_cholmod(),
        "cuda_structured": _has_cuda_structured(),
        "petsc_mat_solve": _has_petsc_mat_solve(),
        "petsc_gamg": _has_petsc_gamg(),
        "petsc_cuda_mat": bool(cuda_probe.get("petsc_cuda_mat", False)),
        "petsc_cuda_vec": bool(cuda_probe.get("petsc_cuda_vec", False)),
        "petsc_cuda_dense": bool(cuda_probe.get("petsc_cuda_dense", False)),
        "petsc_cuda": bool(cuda_probe.get("petsc_cuda", False)),
    }


def detect_performance_capabilities() -> dict[str, bool]:
    """Detect optional acceleration features available in the runtime."""
    return _detect_performance_capabilities_cached(
        (
            _petsc_runtime_cache_key(),
            bool(_has_pyamg()),
            bool(_has_cholmod()),
            bool(_has_cuda_structured()),
        )
    )


detect_performance_capabilities.cache_clear = _detect_performance_capabilities_cached.cache_clear
detect_performance_capabilities.cache_info = _detect_performance_capabilities_cached.cache_info


def select_preconditioner(
    mode: str,
    capabilities: dict[str, bool] | None = None,
) -> str:
    """Resolve preconditioner mode to a concrete backend."""
    resolved_mode = str(mode).strip().lower()
    if capabilities is None:
        capabilities = detect_performance_capabilities()
    if resolved_mode in {"diag", "pyamg", "cholmod", "petsc-gamg"}:
        if resolved_mode == "pyamg" and not capabilities.get("pyamg", False):
            return "diag"
        if resolved_mode == "cholmod" and not capabilities.get("cholmod", False):
            return "diag"
        if resolved_mode == "petsc-gamg" and not capabilities.get("petsc_gamg", False):
            return "diag"
        return resolved_mode
    if resolved_mode != "auto":
        return "diag"

    if capabilities.get("cholmod", False):
        return "cholmod"
    if capabilities.get("pyamg", False):
        return "pyamg"
    if capabilities.get("petsc_gamg", False):
        return "petsc-gamg"
    return "diag"


def select_fast_linear_path(
    mode: str,
    *,
    regularization_is_diagonal: bool,
    regularization_is_sparse_spd: bool,
    capabilities: dict[str, bool] | None = None,
) -> str:
    """Resolve fast linear solve strategy.

    Returns one of: ``woodbury``, ``pcg``, ``cholmod-direct``, ``strict``.
    """
    resolved_mode = str(mode).strip().lower()
    if capabilities is None:
        capabilities = detect_performance_capabilities()

    if resolved_mode in {"woodbury", "pcg", "cholmod-direct", "strict"}:
        return resolved_mode
    if resolved_mode != "auto":
        return "pcg"

    if regularization_is_diagonal:
        return "woodbury"
    if regularization_is_sparse_spd and capabilities.get("cholmod", False):
        return "pcg"
    return "pcg"


def select_fused_strategy(
    *,
    solver_mode: str,
    mesh_dim: int,
    n_param: int,
    n_meas: int,
    rom_mode: str,
    inexact_mode: str,
    lowrank_mode: str,
    regularization_is_diagonal: bool,
    capabilities: dict[str, bool] | None = None,
) -> dict[str, object]:
    """Resolve fused reduced-order acceleration switches.

    Returns a decision payload with:
    ``enabled``, ``rom``, ``inexact``, ``lowrank`` and ``reason``.
    """
    if capabilities is None:
        capabilities = detect_performance_capabilities()

    solver_mode_norm = str(solver_mode).strip().lower()
    if solver_mode_norm != "fast":
        return {
            "enabled": False,
            "rom": False,
            "inexact": False,
            "lowrank": False,
            "reason": "solver_mode_not_fast",
        }

    if int(mesh_dim) < 3:
        return {
            "enabled": False,
            "rom": False,
            "inexact": False,
            "lowrank": False,
            "reason": "mesh_dim_not_3d",
        }

    ratio = float(n_param) / max(float(n_meas), 1.0)
    rom_mode_norm = normalize_feature_mode(rom_mode, default=FEATURE_MODE_AUTO)
    inexact_mode_norm = normalize_feature_mode(inexact_mode, default=FEATURE_MODE_AUTO)
    lowrank_mode_norm = normalize_feature_mode(lowrank_mode, default=FEATURE_MODE_AUTO)

    n_param_mid = int(n_param) >= 8000
    n_param_xlarge = int(n_param) >= 12000

    rom_enabled = rom_mode_norm == "on"
    inexact_enabled = inexact_mode_norm == "on" or (
        rom_enabled and inexact_mode_norm == "auto" and ratio >= 4.0 and n_param_mid
    )
    lowrank_enabled = lowrank_mode_norm == "on" or (
        rom_enabled
        and lowrank_mode_norm == "auto"
        and ratio >= 5.0
        and n_param_xlarge
        and (capabilities.get("cholmod", False) or regularization_is_diagonal)
    )

    enabled = bool(rom_enabled)
    reason = "enabled" if enabled else "rom_disabled_by_policy"
    if enabled and not lowrank_enabled:
        reason = "enabled_without_lowrank"
    if enabled and not inexact_enabled:
        reason = "enabled_without_inexact"

    return {
        "enabled": enabled,
        "rom": bool(rom_enabled),
        "inexact": bool(inexact_enabled),
        "lowrank": bool(lowrank_enabled),
        "reason": reason,
        "ratio_n_over_m": float(ratio),
    }
