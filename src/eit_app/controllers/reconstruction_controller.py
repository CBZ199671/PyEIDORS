"""Runs EIT reconstruction in a background QThread.

Accepts reference/target frame pairs, builds MeasurementDataset,
and calls pyeidors EITSystem for difference reconstruction.
"""

from __future__ import annotations

import contextlib
import glob
import importlib
import io
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

from eit_app.models.frame_model import FrameData

log = logging.getLogger(__name__)

_SYSTEM_CACHE_LOCK = threading.Lock()
_SYSTEM_CACHE_MAX_ITEMS = 4
_SYSTEM_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()

_FAST_CONTEXT_CACHE_LOCK = threading.Lock()
_FAST_CONTEXT_CACHE_MAX_ITEMS = 4
_FAST_CONTEXT_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()


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


def _resolve_drive_mode(meta: dict[str, Any], *, default: str = "total_current") -> str:
    raw_mode = meta.get("drive_mode", default)
    mode = str(raw_mode).strip().lower()
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


def _prepare_single_step_cached_runtime(
    req: ReconstructionRequest,
) -> _SingleStepCachedRuntimeConfig:
    meta = dict(req.metadata)
    meta.setdefault("n_elec", 16)
    meta.setdefault("n_rings", 1)
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
    meta["drive_mode"] = _resolve_drive_mode(meta)
    meta["drive_value"] = _resolve_drive_value(meta)
    mesh_dim = int(meta.get("mesh_dimension", req.mesh_dimension))
    radius = float(meta.get("radius", 1.0))
    refinement = _compute_effective_refinement(
        radius,
        req.mesh_refinement,
        mesh_size=meta.get("mesh_size"),
    )
    lam = float(meta.get("difference_lambda", 1.0e-2))
    background_sigma = float(meta.get("background_sigma", 1.0))
    contact_impedance = float(meta.get("contact_impedance", 0.01))
    mesh_height = float(meta.get("mesh_height", 1.0))
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
        z_center,
        lam,
        background_sigma,
        contact_impedance,
        float(meta.get("geometry_scale_to_m", 1.0)),
        float(meta.get("electrode_coverage", 0.5)),
        repr(meta.get("electrode_length_m_override")),
        str(meta.get("stim_pattern", "{ad}")),
        str(meta.get("meas_pattern", "{ad}")),
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


def _single_step_cached_solver_diagnostics(
    ctx: dict[str, Any],
    *,
    strict_backend: str,
) -> dict[str, Any]:
    return {
        "path": "single_step_cached",
        "strict_solver_backend_effective": strict_backend,
        "cache_build_seconds": dict(ctx.get("cache_build_seconds", {})),
        "cache_miss_reasons": dict(ctx.get("cache_miss_reasons", {})),
        "cache_stats": (
            ctx["cache_manager"].stats()
            if ctx.get("cache_manager") is not None
            else {}
        ),
    }


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
                stim_pattern=str(meta.get("stim_pattern", "{ad}")),
                meas_pattern=str(meta.get("meas_pattern", "{ad}")),
                rotate_meas=bool(meta.get("rotate_meas", True)),
                use_meas_current=bool(meta.get("use_meas_current", False)),
                use_meas_current_next=int(meta.get("use_meas_current_next", 0)),
                stim_direction=str(meta.get("stim_direction", "ccw")),
                meas_direction=str(meta.get("meas_direction", "ccw")),
                stim_first_positive=bool(meta.get("stim_first_positive", False)),
                background_sigma=runtime.background_sigma,
                lam=runtime.lam,
                cache_scope="both",
                solver_mode="strict",
                linear_solver="auto",
                preconditioner="auto",
                rom_mode="off",
                lowrank_mode="off",
                forward_mat_solve="off",
                petsc_device="auto",
                device="auto",
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
    meta["drive_mode"] = _resolve_drive_mode(meta)
    meta["drive_value"] = _resolve_drive_value(meta)

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
        float(meta.get("electrode_coverage", 0.5)),
        repr(meta.get("electrode_length_m_override")),
        float(meta.get("contact_impedance", 0.01)),
        float(req.regularization_alpha),
        str(meta["difference_mode"]),
        str(meta["difference_orientation"]),
    )
    system = _get_cached_system(cache_key)
    if system is None:
        pattern_config = PatternConfig(
            n_elec=meta["n_elec"],
            n_rings=int(meta.get("n_rings", 1)),
            stim_pattern=meta["stim_pattern"],
            meas_pattern=meta["meas_pattern"],
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
        system = EITSystem(
            n_elec=meta["n_elec"],
            pattern_config=pattern_config,
            regularization_alpha=req.regularization_alpha,
            difference_mode=meta["difference_mode"],
            difference_orientation=meta["difference_orientation"],
            contact_impedance=np.full(
                int(meta["n_elec"]) * int(meta.get("n_rings", 1)),
                float(meta.get("contact_impedance", 0.01)),
                dtype=float,
            ),
        )
        mesh = load_or_create_mesh(
            mesh_dir=str(meta.get("mesh_dir", "eit_meshes")),
            n_elec=int(meta["n_elec"]),
            dimension=int(req.mesh_dimension),
            radius=radius,
            refinement=refinement,
            electrode_coverage=float(meta.get("electrode_coverage", 0.5)),
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

    diff_runner = _load_gn_difference_runner_module()
    STRICT_SOLVER_BACKEND_MEASUREMENT = diff_runner.STRICT_SOLVER_BACKEND_MEASUREMENT
    _calibrate_step_size = diff_runner._calibrate_step_size
    _measurement_space_delta = diff_runner._measurement_space_delta
    _solve_linear_from_bundle = diff_runner._solve_linear_from_bundle
    runtime = _prepare_single_step_cached_runtime(req)
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
    dv = np.asarray(tgt_vec - ref_vec, dtype=np.float64)
    operator_bundle = ctx["operator_bundle"]
    strict_backend = str(
        operator_bundle.get(
            "strict_solver_backend_effective",
            "dense-param",
        )
    )
    if strict_backend == STRICT_SOLVER_BACKEND_MEASUREMENT:
        delta_sigma = _measurement_space_delta(operator_bundle=operator_bundle, rhs=dv)
    else:
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
    pred_diff = np.asarray(pred_vi.meas - ctx["base_meas"], dtype=np.float64)

    result_meta = dict(meta)
    result_meta.update(
        {
            "n_elec": int(meta["n_elec"]),
            "reconstruction_runtime": "single_step_cached",
            "difference_lambda": runtime.lam,
            "effective_refinement": runtime.refinement,
            "step_size_alpha": alpha,
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
