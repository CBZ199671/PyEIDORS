"""Runs EIT reconstruction in a background QThread.

Accepts reference/target frame pairs, builds MeasurementDataset,
and calls pyeidors EITSystem for difference reconstruction.
"""

from __future__ import annotations

import contextlib
import glob
import importlib
import io
import json
import logging
from functools import lru_cache
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from eit_app.models.forward_model_config import drive_mode_for_mesh_dimension
from eit_app.models.frame_model import FrameData
from pyeidors.data.difference import build_difference_vector
from pyeidors.electrodes.layout import effective_pattern_layout_for_3d_mesh
from pyeidors.perf.capabilities import probe_petsc_cuda_runtime
from pyeidors.perf.forward_solver_policy import (
    resolve_3d_cuda_forward_solver_policy,
    resolve_3d_cuda_mat_solve_policy,
)

log = logging.getLogger(__name__)

_SYSTEM_CACHE_LOCK = threading.Lock()
_SYSTEM_CACHE_MAX_ITEMS = 4
_SYSTEM_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

_FAST_CONTEXT_CACHE_LOCK = threading.Lock()
_FAST_CONTEXT_CACHE_MAX_ITEMS = 4
_FAST_CONTEXT_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
LINEARIZED_SINGLE_STEP_AUTO_MAX_MEASUREMENTS = 512
_RM_ARTIFACT_CACHE_LOCK = threading.Lock()
_RM_ARTIFACT_CACHE_MAX_ITEMS = 4
_RM_ARTIFACT_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_RM_ARTIFACT_META_KEYS = (
    "rm_artifact_path",
    "dual_model_rm_path",
    "greit_rm_path",
    "reconstruction_matrix_path",
)


def _total_electrodes_from_meta(meta: dict[str, Any]) -> int:
    return max(int(meta.get("n_elec", 16)), 1) * max(int(meta.get("n_rings", 1)), 1)


def _request_measurement_count(req: ReconstructionRequest) -> int:
    try:
        return int(req.reference_frame.to_measurement_vector(req.use_part).size)
    except Exception:
        return 0


def _contact_impedance_scalar(value: Any, default: float = 0.01) -> float:
    if value is None or value == "":
        return float(default)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return float(default)
    if arr.size == 0:
        return float(default)
    return float(arr[0])


def _contact_impedance_vector_from_meta(meta: dict[str, Any], *, total_electrodes: int) -> np.ndarray:
    raw = meta.get("contact_impedance", 0.01)
    total = max(int(total_electrodes), 1)
    if raw is None or raw == "":
        return np.full(total, 0.01, dtype=float)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.full(total, float(arr[0]), dtype=float)
    if arr.size > 0 and total % arr.size == 0:
        return np.tile(arr, total // arr.size).astype(float, copy=False)
    if arr.size != total:
        raise ValueError(
            "contact_impedance length mismatch: "
            f"expected {total} or a divisor of it, got {arr.size}."
        )
    return arr.astype(float, copy=False)


def _resolve_reconstruction_runtime(meta: dict[str, Any], *, mesh_dim: int) -> dict[str, Any]:
    gui_profile = os.getenv("EIT_APP_GUI_PROFILE", "").strip().lower()

    def _auto(key: str, default: str) -> str:
        raw = str(meta.get(key, "") or "").strip().lower()
        return default if raw in {"", "auto"} else raw

    requested_profile = _auto("acceleration_profile", "default")
    mesh_family = _auto("mesh_family", "tetra")
    forward_backend = _auto("forward_backend", "dolfinx")
    wants_gpu_request = gui_profile == "gpu" or requested_profile in {
        "gpu3d",
        "gpu3d_fused",
    }
    wants_structured_gpu = (
        int(mesh_dim) == 3
        and mesh_family == "hex"
        and (wants_gpu_request or forward_backend == "cuda_structured")
    )
    wants_3d_cuda = int(mesh_dim) == 3 and (
        wants_gpu_request or forward_backend == "cuda_structured"
    )

    acceleration_profile = requested_profile
    if wants_3d_cuda and acceleration_profile == "default":
        acceleration_profile = "gpu3d"
    if int(mesh_dim) != 3 and acceleration_profile in {"gpu3d", "gpu3d_fused"}:
        acceleration_profile = "default"

    if wants_structured_gpu and forward_backend == "dolfinx":
        forward_backend = "cuda_structured"
    elif not wants_structured_gpu and forward_backend == "cuda_structured":
        forward_backend = "dolfinx"

    petsc_device = _auto("petsc_device", "cuda" if wants_3d_cuda else "auto")
    capability: dict[str, Any] = {}
    if int(mesh_dim) == 3 and petsc_device == "cuda":
        try:
            capability = dict(probe_petsc_cuda_runtime())
        except Exception as exc:
            capability = {"errors": {"forward_solver_policy": str(exc)}}
    solver_policy = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset=_auto("forward_solver_preset", "auto"),
        mesh_dim=int(mesh_dim),
        petsc_device=petsc_device,
        forward_backend=forward_backend,
        capability=capability,
        prefer_amgx=True,
    )
    mat_solve_policy = resolve_3d_cuda_mat_solve_policy(
        requested_mat_solve=_auto(
            "forward_mat_solve", "auto" if int(mesh_dim) == 3 else "off"
        ),
        mesh_dim=int(mesh_dim),
        petsc_device=petsc_device,
        forward_backend=forward_backend,
        solver_preset=solver_policy["forward_solver_preset_effective"],
    )

    return {
        "solver_mode": _auto("solver_mode", "fast" if int(mesh_dim) == 3 else "strict"),
        "line_search_mode": _auto("line_search_mode", "fast" if int(mesh_dim) == 3 else "full"),
        "linear_solver": _auto("linear_solver", "auto"),
        "preconditioner": _auto("preconditioner", "auto"),
        "fast_linear_path": _auto("fast_linear_path", "auto"),
        "forward_solver_preset": str(solver_policy["forward_solver_preset_effective"]),
        "forward_solver_preset_requested": str(
            solver_policy["forward_solver_preset_requested"]
        ),
        "forward_solver_policy_reason": str(solver_policy["forward_solver_policy_reason"]),
        "forward_solver_policy_warning": str(
            solver_policy["forward_solver_policy_warning"]
        ),
        "petsc_amgx_available": bool(solver_policy["petsc_amgx_available"]),
        "petsc_hypre_available": bool(solver_policy["petsc_hypre_available"]),
        "petsc_hypre_cuda_blacklisted": bool(
            solver_policy["petsc_hypre_cuda_blacklisted"]
        ),
        "forward_mat_solve": str(
            mat_solve_policy["forward_mat_solve_effective_policy"]
        ),
        "forward_mat_solve_requested": str(
            mat_solve_policy["forward_mat_solve_requested"]
        ),
        "forward_mat_solve_policy_reason": str(
            mat_solve_policy["forward_mat_solve_policy_reason"]
        ),
        "forward_mat_solve_policy_warning": str(
            mat_solve_policy["forward_mat_solve_policy_warning"]
        ),
        "petsc_device": petsc_device,
        "device": _auto("device", "cuda" if wants_3d_cuda else "auto"),
        "forward_backend": forward_backend,
        "mesh_family": mesh_family,
        "geometry_version": _auto("geometry_version", "geomv2"),
        "acceleration_profile": acceleration_profile,
    }


def clear_reconstruction_system_cache() -> None:
    """Clear the in-process EITSystem cache used by realtime reconstruction."""
    with _SYSTEM_CACHE_LOCK:
        _SYSTEM_CACHE.clear()
    with _FAST_CONTEXT_CACHE_LOCK:
        _FAST_CONTEXT_CACHE.clear()


@dataclass
class ReconstructionRequest:
    """Input for a reconstruction job."""

    reference_frame: FrameData
    target_frame: FrameData
    use_part: str = "real"
    method: str = "gn-difference"
    regularization_alpha: float = 1.0
    max_iterations: int = 10
    mesh_dimension: int = 2
    mesh_refinement: float = 4.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    """Output from a reconstruction job."""

    conductivity: np.ndarray  # element-wise conductivity
    node_coords: np.ndarray  # (n_nodes, 2 or 3)
    cell_connectivity: np.ndarray  # (n_cells, verts_per_cell)
    measured: np.ndarray | None = None
    simulated: np.ndarray | None = None
    error_msg: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SingleStepCachedRuntimeConfig:
    meta: dict[str, Any]
    mesh_dim: int
    refinement: int
    lam: float
    background_sigma: float
    contact_impedance: float
    mesh_height: float
    electrode_height_ratio: float
    z_center: float
    cache_key: tuple[Any, ...]


class _ReconstructionWorker(QObject):
    """Runs reconstruction in a background thread."""

    finished = Signal(object)  # ReconstructionResult
    progress = Signal(str)  # status messages
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._request: ReconstructionRequest | None = None
        self._eit_system = None  # lazy import pyeidors

    @Slot()
    def run(self) -> None:
        req = self._request
        if req is None:
            self.error.emit("No reconstruction request set")
            return

        result = run_reconstruction_request(req, progress_cb=self.progress.emit)
        if result.error_msg:
            self.error.emit(result.error_msg)
        self.finished.emit(result)


def _get_cached_system(cache_key: tuple[Any, ...]):
    with _SYSTEM_CACHE_LOCK:
        system = _SYSTEM_CACHE.get(cache_key)
        if system is None:
            return None
        _SYSTEM_CACHE.move_to_end(cache_key)
        return system


def _put_cached_system(cache_key: tuple[Any, ...], system: Any) -> None:
    with _SYSTEM_CACHE_LOCK:
        _SYSTEM_CACHE.pop(cache_key, None)
        _SYSTEM_CACHE[cache_key] = system
        while len(_SYSTEM_CACHE) > _SYSTEM_CACHE_MAX_ITEMS:
            _SYSTEM_CACHE.popitem(last=False)


def _get_cached_fast_context(cache_key: tuple[Any, ...]):
    with _FAST_CONTEXT_CACHE_LOCK:
        ctx = _FAST_CONTEXT_CACHE.get(cache_key)
        if ctx is None:
            return None
        _FAST_CONTEXT_CACHE.move_to_end(cache_key)
        return ctx


def _put_cached_fast_context(cache_key: tuple[Any, ...], ctx: Any) -> None:
    with _FAST_CONTEXT_CACHE_LOCK:
        _FAST_CONTEXT_CACHE.pop(cache_key, None)
        _FAST_CONTEXT_CACHE[cache_key] = ctx
        while len(_FAST_CONTEXT_CACHE) > _FAST_CONTEXT_CACHE_MAX_ITEMS:
            _FAST_CONTEXT_CACHE.popitem(last=False)


def _quiet_call(fn: Callable[[], Any]) -> Any:
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = fn()
    captured = sink.getvalue().strip()
    if captured:
        log.debug("Suppressed realtime reconstruction output:\n%s", captured)
    return result


def _recover_nix_runtime_site_packages(missing_name: str) -> tuple[str, ...]:
    """Best-effort recovery for nix-provided Python runtime packages.

    This keeps the GUI realtime reconstruction path working even when the app
    was launched with `PYTHONPATH=src`, which accidentally drops the nix
    shell's FEniCSx-related site-packages.
    """
    missing = str(missing_name or "").strip().lower()
    if missing not in {"ufl", "dolfinx", "mpi4py", "petsc4py", "ffcx", "basix"}:
        return ()
    nix_store = Path("/nix/store")
    if not nix_store.exists():
        return ()

    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    patterns = (
        f"/nix/store/*-{pyver}-fenics-dolfinx-*/lib/{pyver}/site-packages",
        f"/nix/store/*-{pyver}-fenics-ufl-*/lib/{pyver}/site-packages",
        f"/nix/store/*-{pyver}-fenics-basix-*/lib/{pyver}/site-packages",
        f"/nix/store/*-{pyver}-fenics-ffcx-*/lib/{pyver}/site-packages",
        f"/nix/store/*-{pyver}-mpi4py-*/lib/{pyver}/site-packages",
        f"/nix/store/*-petsc-*/lib/{pyver}/site-packages",
        f"/nix/store/*-slepc-*/lib/{pyver}/site-packages",
    )
    discovered: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for candidate in sorted(glob.glob(pattern)):
            if candidate in seen or not os.path.isdir(candidate):
                continue
            seen.add(candidate)
            discovered.append(candidate)
    added: list[str] = []
    for candidate in reversed(discovered):
        if candidate in sys.path:
            continue
        sys.path.insert(0, candidate)
        added.append(candidate)
    if added:
        current = os.environ.get("PYTHONPATH", "")
        prefix = os.pathsep.join(added)
        os.environ["PYTHONPATH"] = prefix if not current else f"{prefix}{os.pathsep}{current}"
        log.info(
            "Recovered nix runtime site-packages for realtime reconstruction (%s): %s",
            missing,
            added,
        )
    return tuple(added)


@lru_cache(maxsize=1)
def _load_gn_difference_runner_module():
    """Load the realtime GN helper module even when only `src/` is on sys.path."""
    module_name = "scripts.common.gn_difference_runner"
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "scripts" / "common" / "gn_difference_runner.py"
    repo_root_str = str(repo_root)

    for _attempt in range(4):
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_name = str(getattr(exc, "name", "") or "")
            if missing_name in {"scripts", "scripts.common", module_name}:
                if not module_path.exists():
                    raise
                if repo_root_str not in sys.path:
                    sys.path.insert(0, repo_root_str)
                    log.info(
                        "Added repository root to sys.path for realtime reconstruction imports: %s",
                        repo_root_str,
                    )
                importlib.invalidate_caches()
                sys.modules.pop(module_name, None)
                continue
            recovered = _recover_nix_runtime_site_packages(missing_name)
            if recovered:
                importlib.invalidate_caches()
                sys.modules.pop(module_name, None)
                continue
            raise
    raise ModuleNotFoundError(f"Unable to import {module_name}")


def _compute_effective_refinement(
    radius: float,
    mesh_refinement: float,
    *,
    mesh_size: float | None = None,
) -> int:
    """Resolve the optimized-mesh refinement used by reconstruction.

    Hardware reconstruction passes the historical integer refinement control
    (4, 8, ...).  The Simulation tab passes a physical mesh_size such as 0.1.
    Treating that mesh_size as ``1 / mesh_size`` and then applying the legacy
    conversion inflates 0.1 to ref20, which makes simulation inverse appear to
    hang while loading/building an unnecessarily dense cache mesh.
    """

    radius_f = max(float(radius), 1e-9)
    size_f = None
    if mesh_size is not None:
        try:
            size_f = float(mesh_size)
        except (TypeError, ValueError):
            size_f = None
    if size_f is not None and np.isfinite(size_f) and size_f > 0.0:
        return max(2, int(round(radius_f / max(size_f, 1e-6) / 2.0)))

    try:
        refinement_f = float(mesh_refinement)
    except (TypeError, ValueError):
        refinement_f = 4.0
    if np.isfinite(refinement_f) and 0.0 < refinement_f < 1.0:
        return max(2, int(round(radius_f / max(refinement_f, 1e-6) / 2.0)))

    mesh_size_f = max(0.02, 0.25 / max(1, int(refinement_f)))
    return max(2, int(round(radius_f / max(mesh_size_f, 1e-6) / 2.0)))


def _resolve_drive_mode(
    meta: dict[str, Any],
    *,
    mesh_dim: int,
    default: str = "total_current",
) -> str:
    raw_mode = meta.get("drive_mode", default)
    mode = drive_mode_for_mesh_dimension(raw_mode, mesh_dim)
    return mode or default


def _resolve_drive_value(
    meta: dict[str, Any],
    *,
    default: float = 1.0e-5,
) -> float:
    raw_value = meta.get("drive_value")
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float("nan")
    if np.isfinite(value) and value > 0.0:
        return value

    raw_stim_uA = meta.get("stim_amp_uA")
    try:
        stim_uA = float(raw_stim_uA)
    except (TypeError, ValueError):
        stim_uA = float("nan")
    if np.isfinite(stim_uA) and stim_uA > 0.0:
        return stim_uA * 1.0e-6

    return float(default)


def _resolve_rm_artifact_path(meta: dict[str, Any]) -> Path | None:
    for key in _RM_ARTIFACT_META_KEYS:
        raw = meta.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            return path
        if not path.is_absolute():
            repo_relative = Path(__file__).resolve().parents[3] / path
            if repo_relative.exists():
                return repo_relative
        raise FileNotFoundError(f"RM artifact path does not exist: {text}")
    return None


def _parse_int_shape(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        for sep in ("x", "X", ",", " "):
            if sep in raw:
                parts = [part for part in raw.replace("x", sep).replace("X", sep).split(sep) if part]
                break
        else:
            parts = [raw]
        try:
            return tuple(int(part) for part in parts if str(part).strip())
        except ValueError:
            return ()
    try:
        arr = np.asarray(value, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError):
        return ()
    return tuple(int(v) for v in arr if int(v) > 0)


def _rm_shape_from_meta(meta: dict[str, Any]) -> tuple[int, ...]:
    for key in ("rm_voxel_shape", "inverse_voxel_shape", "coarse_shape", "voxel_shape"):
        shape = _parse_int_shape(meta.get(key))
        if shape:
            return shape
    return ()


def _optional_npz_array(payload: Any, key: str, *, dtype: Any) -> np.ndarray | None:
    if key not in payload:
        return None
    arr = np.asarray(payload[key], dtype=dtype)
    if arr.size == 0:
        return None
    return arr


def _load_rm_artifact(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        rm = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        artifact_meta: dict[str, Any] = {}
        voxel_shape = _rm_shape_from_meta(meta)
        node_coords = None
        cell_connectivity = None
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            if "rm" not in payload:
                raise ValueError(f"RM artifact is missing 'rm': {path}")
            rm = np.asarray(payload["rm"], dtype=np.float64)
            artifact_meta = {}
            if "metadata_json" in payload:
                raw_meta = str(payload["metadata_json"].item())
                try:
                    artifact_meta = json.loads(raw_meta)
                except json.JSONDecodeError:
                    artifact_meta = {"metadata_json": raw_meta}
            voxel_shape = _parse_int_shape(payload["voxel_shape"]) if "voxel_shape" in payload else ()
            if not voxel_shape:
                voxel_shape = _rm_shape_from_meta(meta)
            node_coords = _optional_npz_array(payload, "node_coords", dtype=np.float64)
            if node_coords is None:
                node_coords = _optional_npz_array(payload, "display_node_coords", dtype=np.float64)
            cell_connectivity = _optional_npz_array(payload, "cell_connectivity", dtype=np.int32)
            if cell_connectivity is None:
                cell_connectivity = _optional_npz_array(payload, "display_cell_connectivity", dtype=np.int32)
    else:
        raise ValueError(f"Unsupported RM artifact suffix {suffix!r}; expected .npz or .npy.")
    if rm.ndim != 2 or 0 in rm.shape:
        raise ValueError(f"RM artifact matrix must be non-empty 2D, got {rm.shape}.")
    return {
        "path": str(path),
        "rm": np.ascontiguousarray(rm, dtype=np.float64),
        "metadata": artifact_meta,
        "voxel_shape": tuple(int(v) for v in voxel_shape),
        "node_coords": node_coords,
        "cell_connectivity": cell_connectivity,
    }


def _rm_artifact_cache_key(path: Path, *, device: str, dtype: str) -> tuple[Any, ...]:
    stat = path.stat()
    return (
        str(path.resolve()),
        int(stat.st_mtime_ns),
        int(stat.st_size),
        str(device).strip().lower(),
        str(dtype).strip().lower(),
    )


def _load_cached_rm_artifact(
    path: Path,
    meta: dict[str, Any],
    *,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    from pyeidors.perf.gpu_kernels import prepare_rm_matmul

    key = _rm_artifact_cache_key(path, device=device, dtype=dtype)
    with _RM_ARTIFACT_CACHE_LOCK:
        cached = _RM_ARTIFACT_CACHE.get(key)
        if cached is not None:
            _RM_ARTIFACT_CACHE.move_to_end(key)
            result = dict(cached)
            result["rm_artifact_cache_hit"] = True
            result["rm_artifact_cache_key"] = key
            return result

    artifact = _load_rm_artifact(path, meta)
    artifact["rm_handle"] = prepare_rm_matmul(
        artifact["rm"],
        device=device,
        dtype=dtype,
        cache_key=str(path),
    )
    with _RM_ARTIFACT_CACHE_LOCK:
        _RM_ARTIFACT_CACHE[key] = dict(artifact)
        _RM_ARTIFACT_CACHE.move_to_end(key)
        while len(_RM_ARTIFACT_CACHE) > _RM_ARTIFACT_CACHE_MAX_ITEMS:
            _RM_ARTIFACT_CACHE.popitem(last=False)
    result = dict(artifact)
    result["rm_artifact_cache_hit"] = False
    result["rm_artifact_cache_key"] = key
    return result


def _voxel_bounds_from_meta(meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_bounds = meta.get("rm_voxel_bounds", meta.get("inverse_bounds"))
    try:
        bounds = np.asarray(raw_bounds, dtype=np.float64)
        if bounds.shape == (2, 3) and np.all(np.isfinite(bounds)) and np.all(bounds[1] > bounds[0]):
            return bounds[0], bounds[1]
    except (TypeError, ValueError):
        pass
    radius = float(meta.get("radius", 1.0) or 1.0)
    height = float(meta.get("mesh_height", meta.get("height", 2.0 * radius)) or (2.0 * radius))
    lower = np.asarray([-radius, -radius, -0.5 * height], dtype=np.float64)
    upper = np.asarray([radius, radius, 0.5 * height], dtype=np.float64)
    return lower, upper


def _voxel_grid_geometry(
    shape: tuple[int, ...],
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(shape) != 3 or any(int(v) <= 0 for v in shape):
        return None
    nx, ny, nz = (int(v) for v in shape)
    lower, upper = _voxel_bounds_from_meta(meta)
    axes = [np.linspace(lower[axis], upper[axis], shape[axis] + 1) for axis in range(3)]
    coords = np.asarray(
        [[x, y, z] for z in axes[2] for y in axes[1] for x in axes[0]],
        dtype=np.float64,
    )

    def node(ix: int, iy: int, iz: int) -> int:
        return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

    cells: list[list[int]] = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                cells.append(
                    [
                        node(ix, iy, iz),
                        node(ix + 1, iy, iz),
                        node(ix + 1, iy + 1, iz),
                        node(ix, iy + 1, iz),
                        node(ix, iy, iz + 1),
                        node(ix + 1, iy, iz + 1),
                        node(ix + 1, iy + 1, iz + 1),
                        node(ix, iy + 1, iz + 1),
                    ]
                )
    return coords, np.asarray(cells, dtype=np.int32)


def _rm_artifact_geometry(
    artifact: dict[str, Any],
    meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    coords = artifact.get("node_coords")
    cells = artifact.get("cell_connectivity")
    if coords is not None and cells is not None:
        return np.asarray(coords, dtype=np.float64), np.asarray(cells, dtype=np.int32)
    generated = _voxel_grid_geometry(tuple(artifact.get("voxel_shape", ())), meta)
    if generated is not None:
        return generated
    raise ValueError(
        "RM artifact hot path requires node/cell geometry or a 3D voxel_shape."
    )


def _try_run_cached_rm_request(
    req: ReconstructionRequest,
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult | None:
    path = _resolve_rm_artifact_path(runtime.meta)
    if path is None:
        return None

    if progress_cb is not None:
        progress_cb("Loading cached reconstruction matrix...")

    from pyeidors.inverse.reconstruction_matrix import reconstruct_difference_batch

    device = str(runtime.meta.get("rm_device", runtime.meta.get("device", "auto")))
    dtype = str(
        runtime.meta.get(
            "rm_dtype",
            runtime.meta.get("rm_matmul_dtype", "float64"),
        )
    )
    artifact = _load_cached_rm_artifact(
        path,
        runtime.meta,
        device=device,
        dtype=dtype,
    )
    node_coords, cell_connectivity = _rm_artifact_geometry(artifact, runtime.meta)
    ref_vec = np.asarray(req.reference_frame.to_measurement_vector(req.use_part), dtype=np.float64)
    tgt_vec = np.asarray(req.target_frame.to_measurement_vector(req.use_part), dtype=np.float64)
    difference_mode = str(runtime.meta.get("difference_mode", "raw"))
    difference_orientation = str(
        runtime.meta.get("difference_orientation", "target_minus_reference")
    )
    dv = build_difference_vector(
        tgt_vec,
        ref_vec,
        mode=difference_mode,
        orientation=difference_orientation,
    )
    rm_result = reconstruct_difference_batch(
        artifact.get("rm_handle", artifact["rm"]),
        dv,
        normalize=False,
        device=device,
        dtype=dtype,
        return_metadata=True,
    )
    conductivity = np.asarray(rm_result.values, dtype=np.float64).reshape(-1)
    result_meta = dict(runtime.meta)
    result_meta.update(
        {
            "n_elec": int(runtime.meta["n_elec"]),
            "reconstruction_runtime": "single_step_cached",
            "single_step_operator_space": "rm",
            "online_hot_path": "rm_matmul",
            "rm_artifact_path": str(path),
            "rm_shape": tuple(int(v) for v in artifact["rm"].shape),
            "rm_voxel_shape": tuple(int(v) for v in artifact.get("voxel_shape", ())),
            "rm_dtype": str(rm_result.metadata.get("rm_dtype", dtype)),
            "rm_artifact_cache_hit": bool(artifact.get("rm_artifact_cache_hit", False)),
            "difference_lambda": runtime.lam,
            "effective_refinement": runtime.refinement,
            "solver_diagnostics": {
                "path": "single_step_cached_rm",
                "strict_solver_backend_effective": "rm",
                "runtime": {
                    "online_hot_path": "rm_matmul",
                    "single_step_operator_space": "rm",
                    "forward_solve_count": 0,
                    "adjoint_solve_count": 0,
                    "jacobian_rebuild_count": 0,
                    "ksp_solve_count": 0,
                    "device_requested": str(rm_result.metadata.get("device_requested", device)),
                    "device_effective": str(rm_result.metadata.get("device_effective", "")),
                    "rm_dtype": str(rm_result.metadata.get("rm_dtype", dtype)),
                    "rm_persistent": bool(rm_result.metadata.get("rm_persistent", False)),
                    "rm_tensor_reused": bool(rm_result.metadata.get("rm_tensor_reused", False)),
                    "rm_prepare_mode": str(rm_result.metadata.get("rm_prepare_mode", "")),
                    "host_device_transfer": str(rm_result.metadata.get("host_device_transfer", "")),
                    "rm_artifact_cache_hit": bool(artifact.get("rm_artifact_cache_hit", False)),
                    "rm_shape": tuple(int(v) for v in artifact["rm"].shape),
                    "rm_artifact_path": str(path),
                },
                "cache_lookups": {
                    "rm_artifact": {
                        "hit": True,
                        "layer": "process"
                        if bool(artifact.get("rm_artifact_cache_hit", False))
                        else "artifact",
                        "process_cache_hit": bool(artifact.get("rm_artifact_cache_hit", False)),
                        "artifact": "reconstruction_matrix",
                        "key": str(path),
                    }
                },
                "rm_metadata": dict(artifact.get("metadata", {}) or {}),
                "rm_matmul": dict(rm_result.metadata),
            },
        }
    )
    if progress_cb is not None:
        progress_cb("Reconstruction complete")
    return ReconstructionResult(
        conductivity=conductivity,
        node_coords=node_coords,
        cell_connectivity=cell_connectivity,
        measured=dv,
        simulated=None,
        metadata=result_meta,
    )


def _prepare_single_step_cached_runtime(
    req: ReconstructionRequest,
) -> _SingleStepCachedRuntimeConfig:
    meta = dict(req.metadata)
    meta.setdefault("n_elec", 16)
    meta.setdefault("n_rings", 1)
    meta.setdefault("electrode_layout", "ring_major")
    meta.setdefault("measurement_protocol", "eidors_full_3d")
    meta.setdefault("custom_stim_matrix", None)
    meta.setdefault("custom_meas_matrices", None)
    meta.setdefault("stim_pattern", "{ad}")
    meta.setdefault("meas_pattern", "{ad}")
    meta.setdefault("rotate_meas", True)
    meta.setdefault("use_meas_current", False)
    meta.setdefault("use_meas_current_next", 0)
    meta.setdefault("stim_direction", "ccw")
    meta.setdefault("meas_direction", "ccw")
    meta.setdefault("stim_first_positive", False)
    meta.setdefault("radius", 1.0)
    meta.setdefault("geometry_scale_to_m", 1.0)
    meta.setdefault("electrode_length_m_override", None)
    meta.setdefault("electrode_coverage", 0.5)
    meta.setdefault("mesh_dir", "eit_meshes")
    meta.setdefault("difference_mode", "raw")
    meta.setdefault("difference_orientation", "target_minus_reference")
    meta.setdefault("step_size_calib", True)
    meta.setdefault("step_size_min", 1.0e-6)
    meta.setdefault("step_size_max", 1.0)
    meta.setdefault("step_size_maxiter", 64)
    meta.setdefault("solver_mode", "auto")
    meta.setdefault("linear_solver", "auto")
    meta.setdefault("preconditioner", "auto")
    meta.setdefault("fast_linear_path", "auto")
    meta.setdefault("forward_solver_preset", "auto")
    meta.setdefault("forward_mat_solve", "auto")
    meta.setdefault("petsc_device", "auto")
    meta.setdefault("device", "auto")
    meta.setdefault("jacobian_representation", "auto")
    meta.setdefault("linearized_solver_strategy", "auto")
    meta.setdefault("linearized_maxiter", 0)
    meta.setdefault("lazy_preconditioner_mode", "auto")
    meta.setdefault("lazy_diag_batch_max_measurements", 512)
    meta.setdefault("forward_backend", "dolfinx")
    meta.setdefault("mesh_family", "tetra")
    meta.setdefault("geometry_version", "geomv2")
    meta.setdefault("acceleration_profile", "default")
    mesh_dim = int(meta.get("mesh_dimension", req.mesh_dimension))
    meta["mesh_dimension"] = mesh_dim
    meta["drive_mode"] = _resolve_drive_mode(meta, mesh_dim=mesh_dim)
    meta["drive_value"] = _resolve_drive_value(meta)
    runtime_options = _resolve_reconstruction_runtime(meta, mesh_dim=mesh_dim)
    meta.update(runtime_options)
    jac_repr = str(meta.get("jacobian_representation", "auto") or "auto").strip().lower()
    jac_repr = jac_repr.replace("_", "-")
    if jac_repr in {"", "auto"}:
        measurement_count = _request_measurement_count(req)
        use_linearized_auto = (
            mesh_dim == 3
            and meta["solver_mode"] == "fast"
            and 0 < measurement_count <= LINEARIZED_SINGLE_STEP_AUTO_MAX_MEASUREMENTS
        )
        jac_repr = "linearized" if use_linearized_auto else "dense"
        meta["jacobian_representation_reason"] = (
            "auto_small_3d_fast"
            if use_linearized_auto
            else "auto_dense_large_or_non3d"
        )
    elif jac_repr in {"jacobian-linearization", "operator"}:
        jac_repr = "linearized"
        meta["jacobian_representation_reason"] = "explicit_linearized"
    elif jac_repr in {"lazy", "lazy-adjoint", "matrix-free", "matrixfree"}:
        jac_repr = "lazy"
        meta["jacobian_representation_reason"] = "explicit_lazy"
    elif jac_repr not in {"dense", "linearized", "lazy"}:
        raise ValueError(
            "jacobian_representation must be auto|dense|linearized|lazy, "
            f"got {meta.get('jacobian_representation')!r}."
        )
    else:
        meta["jacobian_representation_reason"] = f"explicit_{jac_repr}"
    meta["jacobian_representation"] = jac_repr
    radius = float(meta.get("radius", 1.0))
    refinement = _compute_effective_refinement(
        radius,
        req.mesh_refinement,
        mesh_size=meta.get("mesh_size"),
    )
    raw_lam = meta.get("difference_lambda")
    try:
        lam = float(raw_lam)
    except (TypeError, ValueError):
        lam = float("nan")
    if not np.isfinite(lam) or lam <= 0.0:
        try:
            lam = float(req.regularization_alpha)
        except (TypeError, ValueError):
            lam = float("nan")
    if not np.isfinite(lam) or lam <= 0.0:
        lam = 1.0e-2
    meta["difference_lambda"] = lam
    background_sigma = float(meta.get("background_sigma", 1.0))
    contact_impedance = _contact_impedance_scalar(meta.get("contact_impedance", 0.01))
    mesh_height = float(meta.get("mesh_height", meta.get("height", 1.0)))
    electrode_height_ratio = float(meta.get("electrode_height_ratio", 0.2))
    z_center = float(meta.get("z_center", 0.0))
    cache_key = (
        int(meta["n_elec"]),
        int(meta.get("n_rings", 1)),
        mesh_dim,
        refinement,
        radius,
        mesh_height,
        electrode_height_ratio,
        repr(meta.get("electrode_level_fractions", (0.25, 0.75))),
        z_center,
        lam,
        background_sigma,
        contact_impedance,
        float(meta.get("geometry_scale_to_m", 1.0)),
        float(meta.get("electrode_coverage", 0.5)),
        repr(meta.get("electrode_length_m_override")),
        str(meta.get("difference_mode", "raw")),
        str(meta.get("difference_orientation", "target_minus_reference")),
        str(meta.get("stim_pattern", "{ad}")),
        str(meta.get("meas_pattern", "{ad}")),
        str(meta.get("electrode_layout", "ring_major")),
        str(meta.get("measurement_protocol", "eidors_full_3d")),
        repr(meta.get("custom_stim_matrix")),
        repr(meta.get("custom_meas_matrices")),
        bool(meta.get("rotate_meas", True)),
        bool(meta.get("use_meas_current", False)),
        int(meta.get("use_meas_current_next", 0)),
        str(meta.get("stim_direction", "ccw")),
        str(meta.get("meas_direction", "ccw")),
        bool(meta.get("stim_first_positive", False)),
        str(meta.get("drive_mode", "total_current")),
        float(meta.get("drive_value", 1.0e-5)),
        str(meta.get("mesh_dir", "eit_meshes")),
        str(req.use_part),
        str(meta.get("solver_mode", "auto")),
        str(meta.get("linear_solver", "auto")),
        str(meta.get("preconditioner", "auto")),
        str(meta.get("fast_linear_path", "auto")),
        str(meta.get("jacobian_representation", "dense")),
        str(meta.get("linearized_solver_strategy", "auto")),
        int(meta.get("linearized_maxiter", 0)),
        str(meta.get("lazy_preconditioner_mode", "auto")),
        int(meta.get("lazy_diag_batch_max_measurements", 512)),
        str(meta.get("forward_solver_preset", "auto")),
        str(meta.get("forward_mat_solve", "auto")),
        str(meta.get("petsc_device", "auto")),
        str(meta.get("device", "auto")),
        str(meta.get("forward_backend", "dolfinx")),
        str(meta.get("mesh_family", "tetra")),
        str(meta.get("geometry_version", "geomv2")),
        str(meta.get("acceleration_profile", "default")),
    )
    return _SingleStepCachedRuntimeConfig(
        meta=meta,
        mesh_dim=mesh_dim,
        refinement=refinement,
        lam=lam,
        background_sigma=background_sigma,
        contact_impedance=contact_impedance,
        mesh_height=mesh_height,
        electrode_height_ratio=electrode_height_ratio,
        z_center=z_center,
        cache_key=cache_key,
    )


def get_single_step_cached_cache_key(req: ReconstructionRequest) -> tuple[Any, ...]:
    """Return the effective cache key for a single-step cached request."""
    return _prepare_single_step_cached_runtime(req).cache_key


def _cache_hit_summary(cache_lookups: dict[str, Any]) -> tuple[bool | None, dict[str, bool]]:
    hits: dict[str, bool] = {}
    for key, value in cache_lookups.items():
        if not isinstance(value, dict):
            continue
        layer = str(value.get("layer", "")).strip().lower()
        if layer == "disabled":
            continue
        if "hit" in value:
            hits[key] = bool(value.get("hit"))
    if not hits:
        return None, hits
    return all(hits.values()), hits


def _single_step_runtime_diagnostics(ctx: dict[str, Any]) -> dict[str, Any]:
    cache_lookups = dict(ctx.get("cache_lookups", {}))
    cache_hit, cache_hits = _cache_hit_summary(cache_lookups)
    petsc_info = dict(ctx.get("petsc_backend_info", {}))
    return {
        "mesh_family": str(ctx.get("mesh_family", "")),
        "forward_backend": str(ctx.get("forward_backend", "")),
        "forward_backend_effective": str(
            petsc_info.get("forward_backend_effective", ctx.get("forward_backend", ""))
        ),
        "solver_preset": str(petsc_info.get("solver_preset", "")),
        "forward_solver_preset": str(
            petsc_info.get("solver_preset", ctx.get("forward_solver_preset", ""))
        ),
        "forward_solver_policy_reason": str(
            petsc_info.get(
                "forward_solver_policy_reason",
                ctx.get("forward_solver_policy_reason", ""),
            )
        ),
        "forward_solver_policy_warning": str(
            petsc_info.get(
                "forward_solver_policy_warning",
                ctx.get("forward_solver_policy_warning", ""),
            )
        ),
        "petsc_device_requested": str(
            petsc_info.get("petsc_device_requested", ctx.get("petsc_device", ""))
        ),
        "petsc_device_effective": str(petsc_info.get("petsc_device_effective", "")),
        "petsc_amgx_available": bool(
            petsc_info.get("petsc_amgx_available", ctx.get("petsc_amgx_available", False))
        ),
        "petsc_hypre_available": bool(
            petsc_info.get(
                "petsc_hypre_available",
                ctx.get("petsc_hypre_available", False),
            )
        ),
        "petsc_hypre_cuda_blacklisted": bool(
            petsc_info.get(
                "petsc_hypre_cuda_blacklisted",
                ctx.get("petsc_hypre_cuda_blacklisted", False),
            )
        ),
        "forward_mat_solve_effective": str(
            petsc_info.get("forward_mat_solve_effective", "")
        ),
        "forward_mat_solve_policy_reason": str(
            petsc_info.get(
                "forward_mat_solve_policy_reason",
                ctx.get("forward_mat_solve_policy_reason", ""),
            )
        ),
        "torch_device": str(ctx.get("torch_device", "")),
        "device_requested": str(ctx.get("device_requested", "")),
        "device_effective": str(ctx.get("device_effective", "")),
        "jacobian_representation": str(ctx.get("jacobian_representation", "")),
        "jacobian_representation_reason": str(
            ctx.get("jacobian_representation_reason", "")
        ),
        "linearized_solver_strategy": str(
            ctx.get("linearized_solver_strategy", "")
        ),
        "linearized_maxiter": ctx.get("linearized_maxiter"),
        "lazy_preconditioner_mode": str(ctx.get("lazy_preconditioner_mode", "")),
        "mesh_cache_hit": ctx.get("mesh_cache_hit"),
        "mesh_cache_layer": ctx.get("mesh_cache_layer"),
        "mesh_cache_name": ctx.get("mesh_cache_name"),
        "cache_hit": cache_hit,
        "cache_hits": cache_hits,
    }


def _single_step_cached_solver_diagnostics(
    ctx: dict[str, Any],
    *,
    strict_backend: str,
) -> dict[str, Any]:
    return {
        "path": "single_step_cached",
        "strict_solver_backend_effective": strict_backend,
        "runtime": _single_step_runtime_diagnostics(ctx),
        "cache_lookups": dict(ctx.get("cache_lookups", {})),
        "cache_build_seconds": dict(ctx.get("cache_build_seconds", {})),
        "context_build_seconds": ctx.get("context_build_seconds"),
        "cache_miss_reasons": dict(ctx.get("cache_miss_reasons", {})),
        "cache_stats": (
            ctx["cache_manager"].stats()
            if ctx.get("cache_manager") is not None
            else {}
        ),
    }


def _single_step_operator_space(
    operator_bundle: dict[str, Any],
    dv: np.ndarray,
    *,
    measurement_backend: str,
) -> str:
    """Return whether a cached single-step operator solves measurement or parameter space."""
    dv_len = int(np.asarray(dv).reshape(-1).shape[0])
    a_shape = tuple(int(dim) for dim in np.shape(operator_bundle.get("A")))
    if len(a_shape) >= 2 and a_shape[-2] == dv_len and a_shape[-1] == dv_len:
        return "measurement"
    if str(operator_bundle.get("strict_solver_backend_effective", "")) == measurement_backend:
        return "measurement"
    if str(operator_bundle.get("mode", "")).strip().lower() == "fast":
        return "measurement"
    return "parameter"


def _ensure_single_step_cached_context(
    runtime: _SingleStepCachedRuntimeConfig,
    *,
    emit: Callable[[str], None],
    build_shared_context: Callable[..., Any],
) -> dict[str, Any]:
    meta = runtime.meta
    ctx = _get_cached_fast_context(runtime.cache_key)
    if ctx is None:
        emit("Building cached single-step context...")
        ctx = _quiet_call(
            lambda: build_shared_context(
                mesh_dir=str(meta.get("mesh_dir", "eit_meshes")),
                mesh_name=None,
                mesh_dim=runtime.mesh_dim,
                mesh_height=runtime.mesh_height,
                electrode_height_ratio=runtime.electrode_height_ratio,
                z_center=runtime.z_center,
                electrode_level_fractions=meta.get("electrode_level_fractions", (0.25, 0.75)),
                refinement=runtime.refinement,
                n_elec=int(meta["n_elec"]),
                radius=float(meta.get("radius", 1.0)),
                drive_mode=str(meta["drive_mode"]),
                drive_value=float(meta["drive_value"]),
                contact_impedance=runtime.contact_impedance,
                electrode_length_m_override=meta.get("electrode_length_m_override"),
                electrode_coverage=float(meta.get("electrode_coverage", 0.5)),
                geometry_scale_to_m=float(meta.get("geometry_scale_to_m", 1.0)),
                n_rings=int(meta.get("n_rings", 1)),
                electrode_layout=str(meta.get("electrode_layout", "ring_major")),
                measurement_protocol=str(meta.get("measurement_protocol", "eidors_full_3d")),
                custom_stim_matrix=meta.get("custom_stim_matrix"),
                custom_meas_matrices=meta.get("custom_meas_matrices"),
                stim_pattern=str(meta.get("stim_pattern", "{ad}")),
                meas_pattern=str(meta.get("meas_pattern", "{ad}")),
                rotate_meas=bool(meta.get("rotate_meas", True)),
                use_meas_current=bool(meta.get("use_meas_current", False)),
                use_meas_current_next=int(meta.get("use_meas_current_next", 0)),
                stim_direction=str(meta.get("stim_direction", "ccw")),
                meas_direction=str(meta.get("meas_direction", "ccw")),
                stim_first_positive=bool(meta.get("stim_first_positive", False)),
                difference_mode=str(meta.get("difference_mode", "raw")),
                difference_orientation=str(
                    meta.get("difference_orientation", "target_minus_reference")
                ),
                background_sigma=runtime.background_sigma,
                lam=runtime.lam,
                cache_scope="both",
                solver_mode=str(meta.get("solver_mode", "strict")),
                linear_solver=str(meta.get("linear_solver", "auto")),
                preconditioner=str(meta.get("preconditioner", "auto")),
                rom_mode="off",
                lowrank_mode="off",
                forward_solver_preset=str(meta.get("forward_solver_preset", "auto")),
                forward_mat_solve=str(meta.get("forward_mat_solve", "off")),
                petsc_device=str(meta.get("petsc_device", "auto")),
                device=str(meta.get("device", "auto")),
                jacobian_representation=str(
                    meta.get("jacobian_representation", "dense")
                ),
                linearized_solver_strategy=str(
                    meta.get("linearized_solver_strategy", "auto")
                ),
                linearized_maxiter=int(meta.get("linearized_maxiter", 0)),
                lazy_preconditioner_mode=str(
                    meta.get("lazy_preconditioner_mode", "auto")
                ),
                lazy_diag_batch_max_measurements=int(
                    meta.get("lazy_diag_batch_max_measurements", 512)
                ),
                forward_backend=str(meta.get("forward_backend", "dolfinx")),
                mesh_family=str(meta.get("mesh_family", "tetra")),
                geometry_version=str(meta.get("geometry_version", "geomv2")),
            )
        )
        _put_cached_fast_context(runtime.cache_key, ctx)
    else:
        emit("Reusing cached single-step context...")

    mesh = ctx["mesh"]
    if "display_node_coords" not in ctx:
        ctx["display_node_coords"] = np.asarray(mesh.coordinates(), dtype=np.float64)
    if "display_cell_connectivity" not in ctx:
        ctx["display_cell_connectivity"] = np.asarray(mesh.cells(), dtype=np.int32)
    ctx.setdefault(
        "jacobian_representation",
        str(meta.get("jacobian_representation", "dense")),
    )
    ctx["jacobian_representation_reason"] = str(
        meta.get("jacobian_representation_reason", "")
    )
    return ctx


def _run_full_gn_request(
    req: ReconstructionRequest,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    """Execute a reconstruction request via the legacy full GN runtime."""

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    emit("Loading PyEIDORS...")
    from pyeidors import EITSystem
    from pyeidors.data import MeasurementDataset, PatternConfig
    from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh

    ref_vec = req.reference_frame.to_measurement_vector(req.use_part)
    tgt_vec = req.target_frame.to_measurement_vector(req.use_part)

    meta = dict(req.metadata)
    meta.setdefault("n_elec", 16)
    meta.setdefault("n_rings", 1)
    meta.setdefault("electrode_layout", "ring_major")
    meta.setdefault("measurement_protocol", "eidors_full_3d")
    meta.setdefault("custom_stim_matrix", None)
    meta.setdefault("custom_meas_matrices", None)
    meta.setdefault("stim_pattern", "{ad}")
    meta.setdefault("meas_pattern", "{ad}")
    meta.setdefault("rotate_meas", True)
    meta.setdefault("use_meas_current", False)
    meta.setdefault("use_meas_current_next", 0)
    meta.setdefault("stim_direction", "ccw")
    meta.setdefault("meas_direction", "ccw")
    meta.setdefault("stim_first_positive", False)
    meta.setdefault("geometry_scale_to_m", 1.0)
    meta.setdefault("radius", 1.0)
    meta.setdefault("electrode_coverage", 0.5)
    meta.setdefault("electrode_length_m_override", None)
    meta.setdefault("contact_impedance", 0.01)
    meta.setdefault("difference_mode", "raw")
    meta.setdefault("difference_orientation", "target_minus_reference")
    meta.setdefault("difference_preset", "eidors_one_step_noser")
    meta.setdefault("absolute_preset", "eidors_abs_gn")
    meta.setdefault("hyperparameter", None)
    meta.setdefault("solver_mode", "auto")
    meta.setdefault("line_search_mode", "auto")
    meta.setdefault("linear_solver", "auto")
    meta.setdefault("preconditioner", "auto")
    meta.setdefault("fast_linear_path", "auto")
    meta.setdefault("forward_solver_preset", "auto")
    meta.setdefault("forward_mat_solve", "auto")
    meta.setdefault("petsc_device", "auto")
    meta.setdefault("device", "auto")
    meta.setdefault("linearized_solver_strategy", "auto")
    meta.setdefault("linearized_maxiter", 0)
    meta.setdefault("lazy_preconditioner_mode", "auto")
    meta.setdefault("lazy_diag_batch_max_measurements", 512)
    meta.setdefault("forward_backend", "dolfinx")
    meta.setdefault("mesh_family", "tetra")
    meta.setdefault("geometry_version", "geomv2")
    meta.setdefault("acceleration_profile", "default")
    meta["drive_mode"] = _resolve_drive_mode(meta, mesh_dim=int(req.mesh_dimension))
    meta["drive_value"] = _resolve_drive_value(meta)
    runtime_options = _resolve_reconstruction_runtime(meta, mesh_dim=int(req.mesh_dimension))
    meta.update(runtime_options)

    emit("Building measurement datasets...")
    data_type = req.use_part if req.use_part in {"real", "imag", "mag"} else "real"
    ref_ds = MeasurementDataset.from_metadata(
        measurements=ref_vec.reshape(1, -1),
        metadata=meta,
        data_type=data_type,
    )
    tgt_ds = MeasurementDataset.from_metadata(
        measurements=tgt_vec.reshape(1, -1),
        metadata=meta,
        data_type=data_type,
    )
    ref_eit = ref_ds.to_eit_data(frame_index=0)
    tgt_eit = tgt_ds.to_eit_data(frame_index=0)

    emit("Setting up EIT system...")
    radius = float(meta.get("radius", 1.0))
    refinement = _compute_effective_refinement(
        radius,
        req.mesh_refinement,
        mesh_size=meta.get("mesh_size"),
    )
    cache_key = (
        int(meta["n_elec"]),
        int(meta.get("n_rings", 1)),
        str(meta.get("electrode_layout", "ring_major")),
        str(meta.get("measurement_protocol", "eidors_full_3d")),
        repr(meta.get("custom_stim_matrix")),
        repr(meta.get("custom_meas_matrices")),
        str(meta["stim_pattern"]),
        str(meta["meas_pattern"]),
        bool(meta.get("rotate_meas", True)),
        bool(meta.get("use_meas_current", False)),
        int(meta.get("use_meas_current_next", 0)),
        str(meta.get("stim_direction", "ccw")),
        str(meta.get("meas_direction", "ccw")),
        bool(meta.get("stim_first_positive", False)),
        str(meta["drive_mode"]),
        float(meta["drive_value"]),
        float(meta["geometry_scale_to_m"]),
        int(req.mesh_dimension),
        int(refinement),
        float(meta.get("radius", 1.0)),
        float(meta.get("mesh_height", meta.get("height", 1.0))),
        float(meta.get("electrode_height_ratio", 0.2)),
        float(meta.get("z_center", 0.0)),
        repr(meta.get("electrode_level_fractions", (0.25, 0.75))),
        float(meta.get("electrode_coverage", 0.5)),
        repr(meta.get("electrode_length_m_override")),
        _contact_impedance_scalar(meta.get("contact_impedance", 0.01)),
        float(req.regularization_alpha),
        repr(meta.get("hyperparameter")),
        str(meta.get("difference_preset", "eidors_one_step_noser")),
        str(meta.get("absolute_preset", "eidors_abs_gn")),
        str(meta["difference_mode"]),
        str(meta["difference_orientation"]),
        str(meta.get("solver_mode", "auto")),
        str(meta.get("line_search_mode", "auto")),
        str(meta.get("linear_solver", "auto")),
        str(meta.get("preconditioner", "auto")),
        str(meta.get("fast_linear_path", "auto")),
        str(meta.get("linearized_solver_strategy", "auto")),
        int(meta.get("linearized_maxiter", 0)),
        str(meta.get("lazy_preconditioner_mode", "auto")),
        int(meta.get("lazy_diag_batch_max_measurements", 512)),
        str(meta.get("forward_solver_preset", "auto")),
        str(meta.get("forward_mat_solve", "auto")),
        str(meta.get("petsc_device", "auto")),
        str(meta.get("device", "auto")),
        str(meta.get("forward_backend", "dolfinx")),
        str(meta.get("mesh_family", "tetra")),
        str(meta.get("geometry_version", "geomv2")),
        str(meta.get("acceleration_profile", "default")),
    )
    total_electrodes = _total_electrodes_from_meta(meta)
    system = _get_cached_system(cache_key)
    if system is None:
        pattern_n_elec, pattern_n_rings = effective_pattern_layout_for_3d_mesh(
            mesh_tdim=req.mesh_dimension,
            n_elec=int(meta["n_elec"]),
            n_rings=int(meta.get("n_rings", 1)),
            electrode_layout=str(meta.get("electrode_layout", "ring_major")),
        )
        pattern_config = PatternConfig(
            n_elec=pattern_n_elec,
            n_rings=pattern_n_rings,
            stim_pattern=meta["stim_pattern"],
            meas_pattern=meta["meas_pattern"],
            electrode_layout=str(meta.get("electrode_layout", "ring_major")),
            measurement_protocol=str(meta.get("measurement_protocol", "eidors_full_3d")),
            custom_stim_matrix=meta.get("custom_stim_matrix"),
            custom_meas_matrices=meta.get("custom_meas_matrices"),
            drive_mode=meta["drive_mode"],
            drive_value=meta["drive_value"],
            geometry_scale_to_m=meta["geometry_scale_to_m"],
            electrode_length_m_override=meta.get("electrode_length_m_override"),
            use_meas_current=bool(meta.get("use_meas_current", False)),
            use_meas_current_next=int(meta.get("use_meas_current_next", 0)),
            rotate_meas=bool(meta.get("rotate_meas", True)),
            stim_direction=str(meta.get("stim_direction", "ccw")),
            meas_direction=str(meta.get("meas_direction", "ccw")),
            stim_first_positive=bool(meta.get("stim_first_positive", False)),
        )
        hyperparameter = meta.get("hyperparameter")
        if hyperparameter in (None, ""):
            hyperparameter = None
        else:
            hyperparameter = float(hyperparameter)
        system = EITSystem(
            n_elec=total_electrodes,
            pattern_config=pattern_config,
            regularization_alpha=req.regularization_alpha,
            hyperparameter=hyperparameter,
            difference_mode=meta["difference_mode"],
            difference_orientation=meta["difference_orientation"],
            difference_preset=str(meta.get("difference_preset", "eidors_one_step_noser")),
            absolute_preset=str(meta.get("absolute_preset", "eidors_abs_gn")),
            contact_impedance=_contact_impedance_vector_from_meta(
                meta,
                total_electrodes=total_electrodes,
            ),
            solver_mode=str(meta.get("solver_mode", "strict")),
            line_search_mode=str(meta.get("line_search_mode", "full")),
            linear_solver=str(meta.get("linear_solver", "auto")),
            preconditioner=str(meta.get("preconditioner", "auto")),
            fast_linear_path=str(meta.get("fast_linear_path", "auto")),
            linear_backend_config={
                "solver_preset": str(meta.get("forward_solver_preset", "auto")),
                "mat_solve_mode": str(meta.get("forward_mat_solve", "off")),
                "petsc_device": str(meta.get("petsc_device", "auto")),
            },
            petsc_device=str(meta.get("petsc_device", "auto")),
            device=str(meta.get("device", "auto")),
            forward_backend=str(meta.get("forward_backend", "dolfinx")),
            mesh_family=str(meta.get("mesh_family", "tetra")),
            acceleration_profile=str(meta.get("acceleration_profile", "default")),
        )
        mesh = load_or_create_mesh(
            mesh_dir=str(meta.get("mesh_dir", "eit_meshes")),
            n_elec=total_electrodes,
            dimension=int(req.mesh_dimension),
            radius=radius,
            refinement=refinement,
            electrode_coverage=float(meta.get("electrode_coverage", 0.5)),
            height=float(meta.get("mesh_height", meta.get("height", 1.0))),
            electrode_height_ratio=float(meta.get("electrode_height_ratio", 0.2)),
            electrode_level_fractions=meta.get("electrode_level_fractions", (0.25, 0.75)),
            z_center=float(meta.get("z_center", 0.0)),
            mesh_family=str(meta.get("mesh_family", "tetra")),
            geometry_version=str(meta.get("geometry_version", "geomv2")),
            electrode_layout=str(meta.get("electrode_layout", "ring_major")),
        )
        system.setup(mesh=mesh)
        _put_cached_system(cache_key, system)
    else:
        emit("Reusing cached reconstruction system...")

    emit("Running reconstruction...")
    method = req.method.strip().lower()
    if method == "gn-absolute":
        recon = system.absolute_reconstruct(measurement_data=tgt_eit)
    elif method == "sparse-bayes-absolute":
        from pyeidors.inverse.workflows.sparse_bayesian import (
            perform_sparse_absolute_reconstruction,
        )
        recon = perform_sparse_absolute_reconstruction(
            eit_system=system,
            measurement_data=tgt_eit,
        )
    elif method == "sparse-bayes-difference" or method == "sparse-bayes":
        from pyeidors.inverse.workflows.sparse_bayesian import (
            perform_sparse_difference_reconstruction,
        )
        recon = perform_sparse_difference_reconstruction(
            eit_system=system,
            measurement_data=tgt_eit,
            reference_data=ref_eit,
        )
    else:
        # default: gn-difference (single-step Gauss-Newton)
        recon = system.difference_reconstruct(
            measurement_data=tgt_eit,
            reference_data=ref_eit,
        )

    mesh = system.mesh
    coords = mesh.coordinates()
    cells = mesh.cells()

    emit("Reconstruction complete")
    result_meta = dict(meta)
    result_meta["reconstruction_runtime"] = "full_gn"
    diagnostics = getattr(recon, "metadata", {}).get("solver_diagnostics")
    if diagnostics is not None:
        result_meta["solver_diagnostics"] = diagnostics
    return ReconstructionResult(
        conductivity=recon.conductivity
        if hasattr(recon, "conductivity")
        else np.asarray([]),
        node_coords=coords,
        cell_connectivity=cells,
        measured=getattr(recon, "measured", None),
        simulated=getattr(recon, "simulated", None),
        metadata=result_meta,
    )


def _run_single_step_cached_request(
    req: ReconstructionRequest,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    """Execute a reconstruction request via the cached single-step realtime path."""

    def emit(message: str) -> None:
        if progress_cb is not None:
            progress_cb(message)

    runtime = _prepare_single_step_cached_runtime(req)
    rm_result = _try_run_cached_rm_request(req, runtime, progress_cb=progress_cb)
    if rm_result is not None:
        return rm_result

    diff_runner = _load_gn_difference_runner_module()
    STRICT_SOLVER_BACKEND_MEASUREMENT = diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT
    _calibrate_step_size = diff_runner._calibrate_step_size
    _measurement_space_delta = diff_runner._measurement_space_delta
    _solve_linear_from_bundle = diff_runner._solve_linear_from_bundle
    _solve_linearized_delta = getattr(diff_runner, "_solve_linearized_delta", None)
    meta = runtime.meta
    ctx = _ensure_single_step_cached_context(
        runtime,
        emit=emit,
        build_shared_context=diff_runner.build_shared_context,
    )
    result_meta = dict(meta)
    result_meta.update(
        {
            "n_elec": int(meta["n_elec"]),
            "reconstruction_runtime": "single_step_cached",
            "difference_lambda": runtime.lam,
            "effective_refinement": runtime.refinement,
        }
    )

    if bool(meta.get("warmup_only", False)):
        result_meta["cache_warmup_only"] = True
        result_meta["solver_diagnostics"] = _single_step_cached_solver_diagnostics(
            ctx,
            strict_backend="warmup_only",
        )
        emit("Realtime reconstruction context ready")
        return ReconstructionResult(
            conductivity=np.asarray([], dtype=np.float64),
            node_coords=ctx["display_node_coords"],
            cell_connectivity=ctx["display_cell_connectivity"],
            metadata=result_meta,
        )

    from pyeidors.data.structures import EITImage
    from pyeidors.utils.numeric_ops import safe_dot

    ref_vec = np.asarray(req.reference_frame.to_measurement_vector(req.use_part), dtype=np.float64)
    tgt_vec = np.asarray(req.target_frame.to_measurement_vector(req.use_part), dtype=np.float64)

    emit("Running cached single-step reconstruction...")
    difference_mode = str(meta.get("difference_mode", "raw"))
    difference_orientation = str(meta.get("difference_orientation", "target_minus_reference"))
    dv = build_difference_vector(
        tgt_vec,
        ref_vec,
        mode=difference_mode,
        orientation=difference_orientation,
    )
    operator_bundle = ctx["operator_bundle"]
    strict_backend = str(
        operator_bundle.get(
            "strict_solver_backend_effective",
            "dense-param",
        )
    )
    if str(operator_bundle.get("jacobian_representation", "dense")) in {
        "linearized",
        "lazy",
    }:
        if _solve_linearized_delta is None:
            raise RuntimeError("linearized single-step runtime is unavailable.")
        operator_space = "linearized"
        delta_sigma = _solve_linearized_delta(operator_bundle=operator_bundle, rhs=dv)
    else:
        operator_space = _single_step_operator_space(
            operator_bundle,
            dv,
            measurement_backend=STRICT_SOLVER_BACKEND_MEASUREMENT,
        )
    if operator_space == "measurement":
        delta_sigma = _measurement_space_delta(operator_bundle=operator_bundle, rhs=dv)
    elif operator_space != "linearized":
        rhs = np.asarray(
            safe_dot(operator_bundle["Jt"], dv, "eit_app.fast_recon.Jt_dv"),
            dtype=np.float64,
        )
        delta_sigma = _solve_linear_from_bundle(operator_bundle, rhs)

    alpha = 1.0
    if bool(meta.get("step_size_calib", True)):
        try:
            alpha = float(
                _calibrate_step_size(
                    fwd_model=ctx["fwd_model"],
                    sigma_bg=ctx["sigma_bg"],
                    delta_sigma=delta_sigma,
                    dv=dv,
                    base_meas=ctx["base_meas"],
                    step_size_min=float(meta.get("step_size_min", 1.0e-6)),
                    step_size_max=float(meta.get("step_size_max", 1.0)),
                    step_size_maxiter=int(meta.get("step_size_maxiter", 64)),
                    difference_mode=difference_mode,
                    difference_orientation=difference_orientation,
                )
            )
        except Exception as exc:
            log.debug("Realtime step-size calibration failed: %s", exc)
            alpha = 1.0
        if not np.isfinite(alpha) or alpha <= 0.0:
            alpha = 1.0

    display_delta = np.asarray(alpha * delta_sigma, dtype=np.float64)
    sigma_est = np.asarray(ctx["sigma_bg"] + display_delta, dtype=np.float64)
    img_est = EITImage(elem_data=sigma_est, fwd_model=ctx["fwd_model"])
    pred_vi, _ = ctx["fwd_model"].fwd_solve(img_est)
    pred_diff = build_difference_vector(
        pred_vi.meas,
        ctx["base_meas"],
        mode=difference_mode,
        orientation=difference_orientation,
    )

    result_meta = dict(meta)
    result_meta.update(
        {
            "n_elec": int(meta["n_elec"]),
            "reconstruction_runtime": "single_step_cached",
            "difference_lambda": runtime.lam,
            "effective_refinement": runtime.refinement,
            "step_size_alpha": alpha,
            "single_step_operator_space": operator_space,
            "solver_diagnostics": _single_step_cached_solver_diagnostics(
                ctx,
                strict_backend=strict_backend,
            ),
        }
    )

    emit("Reconstruction complete")
    return ReconstructionResult(
        conductivity=display_delta,
        node_coords=ctx["display_node_coords"],
        cell_connectivity=ctx["display_cell_connectivity"],
        measured=dv,
        simulated=pred_diff,
        metadata=result_meta,
    )


def run_reconstruction_request(
    req: ReconstructionRequest,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> ReconstructionResult:
    """Execute a reconstruction request synchronously using the realtime app pipeline."""
    try:
        runtime_path = str((req.metadata or {}).get("reconstruction_runtime", "")).strip().lower()
        method_lc = req.method.strip().lower()
        log.info(
            "[recon-dispatch] method=%r use_part=%r runtime_path=%r source=%r",
            method_lc,
            req.use_part,
            runtime_path,
            (req.metadata or {}).get("request_source"),
        )
        if (
            method_lc == "gn-difference"
            and req.use_part == "real"
            and runtime_path == "single_step_cached"
        ):
            log.info("[recon-dispatch] -> single_step_cached (fast path)")
            return _run_single_step_cached_request(req, progress_cb=progress_cb)
        log.info("[recon-dispatch] -> full_gn (iterative path)")
        return _run_full_gn_request(req, progress_cb=progress_cb)

    except Exception as exc:
        log.exception("Reconstruction failed")
        return ReconstructionResult(
            conductivity=np.array([]),
            node_coords=np.array([]),
            cell_connectivity=np.array([]),
            error_msg=str(exc),
            metadata=dict(getattr(req, "metadata", {}) or {}),
        )


class ReconstructionController(QObject):
    """GUI-facing controller for EIT reconstruction.

    Signals:
        reconstruction_done: Emitted with ReconstructionResult.
        progress: Emitted with status strings during reconstruction.
        error: Emitted on errors.
    """

    reconstruction_done = Signal(object)  # ReconstructionResult
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _ReconstructionWorker | None = None
        self._busy = False

    @property
    def is_busy(self) -> bool:
        return self._busy

    def reconstruct(self, request: ReconstructionRequest) -> bool:
        """Submit a reconstruction request. Runs in a background thread."""
        if self._busy:
            self.error.emit("Reconstruction already in progress")
            return False

        self._busy = True
        self._thread = QThread()
        self._worker = _ReconstructionWorker()
        self._worker._request = request
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self.progress)
        self._worker.error.connect(self.error)

        self._thread.start()
        return True

    def _on_finished(self, result: ReconstructionResult) -> None:
        self._busy = False
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        self.reconstruction_done.emit(result)

    def shutdown(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
