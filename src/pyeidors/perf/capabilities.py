"""Runtime capability detection and backend selection for performance paths."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

from ..runtime_paths import pyeidors_cache_path
from .policy import FEATURE_MODE_AUTO, normalize_feature_mode

MPI_SINGLE_RANK_FALLBACK_REASON = "mpi_size_gt_1_not_supported_phase2_single_rank_only"
MPI_SINGLE_RANK_GUIDANCE = (
    "PyEIDORS phase-2 CEM forward currently supports MPI size=1 only; "
    "use single-rank execution until distributed PETSc/DOLFINx production "
    "paths have mpiexec smoke coverage."
)
PETSC_CUDA_RUNTIME_PROBE_CACHE_SCHEMA = "petsc_cuda_runtime_probe_cache_v2"


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


def _has_petsc_pc_type(type_attr: str) -> bool:
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return False
    pc_type = getattr(getattr(PETSc, "PC", None), "Type", None)
    return pc_type is not None and hasattr(pc_type, type_attr)


def _has_petsc_gamg() -> bool:
    return _has_petsc_pc_type("GAMG")


def _has_petsc_hypre() -> bool:
    return _has_petsc_pc_type("HYPRE")


def _has_petsc_amgx() -> bool:
    return _has_petsc_pc_type("AMGX")


def _probe_petsc_amgx_setup_solve(PETSc) -> tuple[bool, str | None]:
    """Verify PCAMGX can set up and solve a tiny CUDA system.

    PETSc 3.24's AMGX defaults can be rejected by AMGX 2.4.0.  The smoke
    uses scalar-type-specific options validated for this runtime instead of
    treating the mere presence of ``PC.Type.AMGX`` as a usable solver path.
    """

    objects = []
    scalar_name = str(getattr(PETSc, "ScalarType", "")).lower()
    complex_scalar = "complex" in scalar_name
    prefix = "pyeidors_amgx_probe_"
    probe_options = (
        {
            "pc_amgx_amg_method": "AGGREGATION",
            "pc_amgx_selector": "SIZE_8",
            "pc_amgx_smoother": "BLOCK_JACOBI",
            "pc_amgx_exact_coarse_solve": "0",
            "pc_amgx_presweeps": "2",
            "pc_amgx_postsweeps": "2",
            "pc_amgx_coarse_solver": "NOSOLVER",
        }
        if complex_scalar
        else {
            "pc_amgx_smoother": "JACOBI_L1",
            "pc_amgx_exact_coarse_solve": "0",
        }
    )
    option_keys = [f"{prefix}{key}" for key in probe_options]
    options = None
    try:
        options = PETSc.Options()
        for key, value in probe_options.items():
            options[f"{prefix}{key}"] = value

        size = 16 if complex_scalar else 2
        A = PETSc.Mat().createAIJ([size, size], nnz=3)
        objects.append(A)
        try:
            A.setType("aijcusparse")
        except Exception:
            pass
        A.setUp()
        for row in range(size):
            A[row, row] = 4.0 + (0.2j if complex_scalar else 0.0)
            if row > 0:
                A[row, row - 1] = -1.0 + (0.05j if complex_scalar else 0.0)
            if row < size - 1:
                A[row, row + 1] = -1.0 + (-0.05j if complex_scalar else 0.0)
        A.assemblyBegin()
        A.assemblyEnd()

        b = PETSc.Vec().createSeq(size)
        objects.append(b)
        try:
            b.setType("cuda")
        except Exception:
            pass
        b.set(1.0 + (0.1j if complex_scalar else 0.0))
        b.assemblyBegin()
        b.assemblyEnd()

        x = b.duplicate()
        objects.append(x)

        ksp = PETSc.KSP().create()
        objects.append(ksp)
        ksp.setOptionsPrefix(prefix)
        ksp.setOperators(A)
        ksp.setType("fgmres" if complex_scalar else "cg")
        ksp.getPC().setType("amgx")
        ksp.setTolerances(rtol=1e-10, max_it=20)
        ksp.setFromOptions()
        ksp.solve(b, x)
        residual = b.duplicate()
        objects.append(residual)
        A.mult(x, residual)
        residual.axpy(-1.0, b)
        b_norm = b.norm()
        relres = residual.norm() / b_norm if b_norm else residual.norm()
        if relres != relres or relres > 1e-6:
            return False, f"AMGX smoke residual too high: {relres:.3g}"
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if options is not None:
            for key in option_keys:
                try:
                    del options[key]
                except Exception:
                    pass
        for obj in reversed(objects):
            destroy = getattr(obj, "destroy", None)
            if callable(destroy):
                try:
                    destroy()
                except Exception:
                    pass


def _load_mpi_comm_world():
    try:
        from mpi4py import MPI
    except Exception:
        return None
    return getattr(MPI, "COMM_WORLD", None)


def _comm_int(comm, *, method_name: str, attr_name: str, default: int) -> int:
    if comm is None:
        return int(default)
    try:
        method = getattr(comm, method_name, None)
        if callable(method):
            return int(method())
    except Exception:
        pass
    try:
        if hasattr(comm, attr_name):
            return int(getattr(comm, attr_name))
    except Exception:
        pass
    return int(default)


def probe_mpi_runtime(
    comm=None,
    *,
    supports_parallel: bool = False,
) -> dict[str, object]:
    """Report MPI rank/size and the current PyEIDORS production support boundary."""
    source = "provided"
    if comm is None:
        comm = _load_mpi_comm_world()
        source = "mpi4py.COMM_WORLD" if comm is not None else "unavailable"

    size = max(1, _comm_int(comm, method_name="Get_size", attr_name="size", default=1))
    rank = max(0, _comm_int(comm, method_name="Get_rank", attr_name="rank", default=0))
    parallel = size > 1
    size_supported = bool((not parallel) or supports_parallel)
    fallback_reason = None if size_supported else MPI_SINGLE_RANK_FALLBACK_REASON
    return {
        "mpi_available": comm is not None,
        "mpi_source": source,
        "mpi_size": int(size),
        "mpi_rank": int(rank),
        "mpi_parallel": bool(parallel),
        "mpi_parallel_supported": bool(supports_parallel),
        "mpi_size_supported": bool(size_supported),
        "mpi_fallback_reason": fallback_reason,
        "mpi_guidance": MPI_SINGLE_RANK_GUIDANCE if fallback_reason else None,
    }


def _enum_name(namespace, name: str) -> str | None:
    if namespace is None or not hasattr(namespace, name):
        return None
    try:
        return str(getattr(namespace, name))
    except Exception:
        return None


def _create_petsc_object(PETSc, cls_name: str):
    """Create and initialize a PETSc Mat or Vec object."""
    obj = getattr(PETSc, cls_name)()
    creator = getattr(obj, "create", None)
    if callable(creator):
        comm = getattr(PETSc, "COMM_SELF", None)
        try:
            creator(comm=comm)
        except TypeError:
            creator()
    return obj


def _probe_petsc_type(
    type_name: str | None,
    *,
    cls_name: str,
    missing_label: str,
    setup_fn,
) -> tuple[bool, str | None]:
    """Probe whether a PETSc type is truly usable at runtime."""
    if not type_name:
        return False, missing_label
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return False, "petsc_unavailable"
    obj = None
    try:
        obj = _create_petsc_object(PETSc, cls_name)
        setup_fn(obj, type_name)
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if obj is not None and hasattr(obj, "destroy"):
            try:
                obj.destroy()
            except Exception:
                pass


def _setup_mat_probe(mat, type_name: str) -> None:
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


def _setup_vec_probe(vec, type_name: str) -> None:
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


def _probe_petsc_mat_type(type_name: str | None) -> tuple[bool, str | None]:
    return _probe_petsc_type(
        type_name,
        cls_name="Mat",
        missing_label="mat_type_symbol_missing",
        setup_fn=_setup_mat_probe,
    )


def _probe_petsc_vec_type(type_name: str | None) -> tuple[bool, str | None]:
    return _probe_petsc_type(
        type_name,
        cls_name="Vec",
        missing_label="vec_type_symbol_missing",
        setup_fn=_setup_vec_probe,
    )


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
        bool(_has_petsc_gamg()),
        bool(_has_petsc_hypre()),
        bool(_has_petsc_amgx()),
    )


def _petsc_cuda_probe_disk_cache_enabled() -> bool:
    raw = os.getenv("PYEIDORS_PETSC_CUDA_PROBE_CACHE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _petsc_cuda_probe_disk_cache_dir() -> Path:
    override = os.getenv("PYEIDORS_PETSC_CUDA_PROBE_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return pyeidors_cache_path("capabilities")


def _petsc_runtime_disk_cache_payload() -> dict[str, object]:
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return {"petsc": None}
    runtime_type = type(PETSc)
    runtime_name = str(
        getattr(
            PETSc,
            "__name__",
            getattr(runtime_type, "__qualname__", runtime_type.__name__),
        )
    )
    runtime_module = str(
        getattr(PETSc, "__module__", getattr(runtime_type, "__module__", ""))
    )
    version = None
    try:
        version_fn = getattr(getattr(PETSc, "Sys", None), "getVersion", None)
        if callable(version_fn):
            version = version_fn()
    except Exception:
        version = None
    return {
        "schema": PETSC_CUDA_RUNTIME_PROBE_CACHE_SCHEMA,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "petsc_module": getattr(sys.modules.get("petsc4py"), "__file__", ""),
        "petsc_runtime_module": runtime_module,
        "petsc_runtime_name": runtime_name,
        "petsc_version": repr(version),
        "petsc_scalar_type": str(getattr(PETSc, "ScalarType", "")),
        "mat_aijcusparse": _enum_name(
            getattr(getattr(PETSc, "Mat", None), "Type", None), "AIJCUSPARSE"
        ),
        "vec_cuda": _enum_name(
            getattr(getattr(PETSc, "Vec", None), "Type", None), "CUDA"
        ),
        "mat_densecuda": _enum_name(
            getattr(getattr(PETSc, "Mat", None), "Type", None), "DENSECUDA"
        ),
        "has_mat_solve": bool(hasattr(getattr(PETSc, "KSP", None), "matSolve")),
        "has_gamg": bool(_has_petsc_gamg()),
        "has_hypre": bool(_has_petsc_hypre()),
        "has_amgx": bool(_has_petsc_amgx()),
    }


def _petsc_runtime_disk_cache_key() -> str:
    encoded = json.dumps(
        _petsc_runtime_disk_cache_payload(),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_petsc_cuda_probe_disk_cache(cache_key: str) -> dict[str, object] | None:
    if not _petsc_cuda_probe_disk_cache_enabled():
        return None
    path = _petsc_cuda_probe_disk_cache_dir() / f"petsc_cuda_{cache_key}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema") != PETSC_CUDA_RUNTIME_PROBE_CACHE_SCHEMA:
        return None
    if payload.get("key") != cache_key:
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    cached = dict(result)
    cached["probe_cache"] = {
        "enabled": True,
        "hit": True,
        "layer": "disk",
        "key": cache_key,
        "path": str(path),
    }
    return cached


def _write_petsc_cuda_probe_disk_cache(
    cache_key: str,
    result: dict[str, object],
) -> dict[str, object]:
    if not _petsc_cuda_probe_disk_cache_enabled():
        result["probe_cache"] = {"enabled": False, "hit": False}
        return result
    cache_dir = _petsc_cuda_probe_disk_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"petsc_cuda_{cache_key}.json"
    stored = {key: value for key, value in result.items() if key != "probe_cache"}
    payload = {
        "schema": PETSC_CUDA_RUNTIME_PROBE_CACHE_SCHEMA,
        "key": cache_key,
        "runtime": _petsc_runtime_disk_cache_payload(),
        "result": stored,
    }
    tmp_path = path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, sort_keys=True, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError:
        result["probe_cache"] = {
            "enabled": True,
            "hit": False,
            "stored": False,
            "layer": "disk",
            "key": cache_key,
            "path": str(path),
        }
        return result
    result["probe_cache"] = {
        "enabled": True,
        "hit": False,
        "stored": True,
        "layer": "disk",
        "key": cache_key,
        "path": str(path),
    }
    return result


@lru_cache(maxsize=8)
def _probe_petsc_cuda_runtime_cached(
    runtime_key: tuple[object, ...],
) -> dict[str, object]:
    del runtime_key
    disk_key = _petsc_runtime_disk_cache_key()
    cached = _read_petsc_cuda_probe_disk_cache(disk_key)
    if cached is not None:
        return cached
    PETSc = _load_petsc_runtime()
    if PETSc is None:
        return _write_petsc_cuda_probe_disk_cache(
            disk_key,
            {
                "petsc_available": False,
                "petsc_cuda": False,
                "petsc_cuda_mat": False,
                "petsc_cuda_vec": False,
                "petsc_cuda_dense": False,
                "petsc_hypre": False,
                "petsc_amgx": False,
                "petsc_amgx_cuda_candidate": False,
                "petsc_amgx_smoke": False,
                "mat_type_name": None,
                "vec_type_name": None,
                "dense_mat_type_name": None,
                "errors": {"petsc": "petsc_unavailable"},
            },
        )

    mat_type = _enum_name(
        getattr(getattr(PETSc, "Mat", None), "Type", None), "AIJCUSPARSE"
    )
    vec_type = _enum_name(getattr(getattr(PETSc, "Vec", None), "Type", None), "CUDA")
    dense_type = _enum_name(
        getattr(getattr(PETSc, "Mat", None), "Type", None), "DENSECUDA"
    )

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

    amgx_symbol_available = bool(_has_petsc_amgx())
    amgx_available = amgx_symbol_available
    amgx_smoke_ok = False
    amgx_error = None
    if mat_ok and vec_ok:
        amgx_smoke_ok, amgx_error = _probe_petsc_amgx_setup_solve(PETSc)
        amgx_available = bool(amgx_available or amgx_smoke_ok)
    if amgx_error:
        errors["amgx"] = amgx_error

    return _write_petsc_cuda_probe_disk_cache(
        disk_key,
        {
            "petsc_available": True,
            "petsc_cuda": bool(mat_ok and vec_ok),
            "petsc_cuda_mat": bool(mat_ok),
            "petsc_cuda_vec": bool(vec_ok),
            "petsc_cuda_dense": bool(dense_ok),
            "petsc_hypre": bool(_has_petsc_hypre()),
            "petsc_amgx": amgx_available,
            "petsc_amgx_smoke": bool(amgx_smoke_ok),
            "petsc_amgx_cuda_candidate": bool(
                amgx_available and mat_ok and vec_ok and amgx_smoke_ok
            ),
            "mat_type_name": mat_type,
            "vec_type_name": vec_type,
            "dense_mat_type_name": dense_type,
            "errors": errors,
        },
    )


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
def _detect_performance_capabilities_cached(
    cache_key: tuple[object, ...],
) -> dict[str, bool]:
    del cache_key
    cuda_probe = probe_petsc_cuda_runtime()
    mpi_probe = probe_mpi_runtime()
    return {
        "pyamg": _has_pyamg(),
        "cholmod": _has_cholmod(),
        "cuda_structured": _has_cuda_structured(),
        "petsc_mat_solve": _has_petsc_mat_solve(),
        "petsc_gamg": _has_petsc_gamg(),
        "petsc_hypre": _has_petsc_hypre(),
        "petsc_amgx": bool(cuda_probe.get("petsc_amgx", False)),
        "petsc_amgx_cuda_candidate": bool(
            cuda_probe.get("petsc_amgx_cuda_candidate", False)
        ),
        "petsc_cuda_mat": bool(cuda_probe.get("petsc_cuda_mat", False)),
        "petsc_cuda_vec": bool(cuda_probe.get("petsc_cuda_vec", False)),
        "petsc_cuda_dense": bool(cuda_probe.get("petsc_cuda_dense", False)),
        "petsc_cuda": bool(cuda_probe.get("petsc_cuda", False)),
        "mpi": bool(mpi_probe.get("mpi_available", False)),
        "mpi_parallel": bool(mpi_probe.get("mpi_parallel", False)),
        "mpi_size_supported": bool(mpi_probe.get("mpi_size_supported", False)),
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


detect_performance_capabilities.cache_clear = (
    _detect_performance_capabilities_cached.cache_clear
)
detect_performance_capabilities.cache_info = (
    _detect_performance_capabilities_cached.cache_info
)


def select_preconditioner(
    mode: str,
    capabilities: dict[str, bool] | None = None,
) -> str:
    """Resolve preconditioner mode to a concrete backend."""
    resolved_mode = str(mode).strip().lower()
    if capabilities is None:
        capabilities = detect_performance_capabilities()
    mode_capability = {
        "pyamg": "pyamg",
        "cholmod": "cholmod",
        "petsc-gamg": "petsc_gamg",
    }
    if resolved_mode in {"diag", "noser", "prior", "pmat", "coarse", "custom"}:
        return resolved_mode
    if resolved_mode in mode_capability:
        required = mode_capability[resolved_mode]
        return resolved_mode if capabilities.get(required, False) else "diag"
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
