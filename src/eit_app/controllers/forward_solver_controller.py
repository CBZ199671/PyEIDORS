"""Forward problem solver running in a background QThread."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from eit_app.models.precision import compute_dtype
from eit_app.models.simulation_state import InhomogeneitySpec
from eit_app.models.forward_model_config import ForwardModelConfig

log = logging.getLogger(__name__)


def probe_petsc_cuda_runtime() -> dict[str, Any]:
    from pyeidors.perf.capabilities import probe_petsc_cuda_runtime as _probe

    return _probe()


def resolve_3d_cuda_forward_solver_policy(*args, **kwargs) -> dict[str, Any]:
    from pyeidors.perf.forward_solver_policy import (
        resolve_3d_cuda_forward_solver_policy as _resolve,
    )

    return _resolve(*args, **kwargs)


def resolve_3d_cuda_mat_solve_policy(*args, **kwargs) -> dict[str, Any]:
    from pyeidors.perf.forward_solver_policy import (
        resolve_3d_cuda_mat_solve_policy as _resolve,
    )

    return _resolve(*args, **kwargs)


@dataclass
class ForwardSolverRequest:
    """Input parameters for a single forward solve."""

    mesh_dimension: int = 2
    mesh_refinement: float = 0.1
    n_electrodes: int = 16
    background_conductivity: float = 1.0
    inhomogeneities: list[InhomogeneitySpec] = field(default_factory=list)
    noise_level: float = 0.0
    forward_model_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForwardSolverResult:
    """Output of a forward solve."""

    boundary_voltages: np.ndarray
    ground_truth_conductivity: np.ndarray
    node_coords: np.ndarray
    cell_connectivity: np.ndarray
    n_elements: int
    n_measurements: int
    homogeneous_voltages: np.ndarray | None = None
    forward_model_config: dict[str, Any] = field(default_factory=dict)
    error_msg: str | None = None


def _paint_shape(
    values: np.ndarray,
    centers: np.ndarray,
    spec: InhomogeneitySpec,
    *,
    mesh_dimension: int = 2,
) -> None:
    """Paint a single inhomogeneity shape onto element-centered values."""
    if centers.size == 0:
        return

    cx, cy = spec.center_x, spec.center_y
    rx = abs(float(spec.size_x))
    ry = abs(float(spec.size_y))
    rz = abs(float(getattr(spec, "size_z", spec.size_x)))
    if rx <= 0:
        return
    if ry <= 0:
        ry = rx
    if rz <= 0:
        rz = rx

    is_3d = int(mesh_dimension) == 3 and centers.shape[1] >= 3

    if spec.shape == "circle":
        if is_3d:
            cz = float(getattr(spec, "center_z", 0.0))
            dist2 = (
                (centers[:, 0] - cx) ** 2
                + (centers[:, 1] - cy) ** 2
                + (centers[:, 2] - cz) ** 2
            )
            values[dist2 < rx**2] = spec.conductivity
            return
        dist2 = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
        values[dist2 < rx**2] = spec.conductivity

    elif spec.shape == "ellipse":
        if is_3d:
            cz = float(getattr(spec, "center_z", 0.0))
            norm = (
                ((centers[:, 0] - cx) / rx) ** 2
                + ((centers[:, 1] - cy) / ry) ** 2
                + ((centers[:, 2] - cz) / rz) ** 2
            )
            values[norm < 1.0] = spec.conductivity
            return
        norm = ((centers[:, 0] - cx) / rx) ** 2 + ((centers[:, 1] - cy) / ry) ** 2
        values[norm < 1.0] = spec.conductivity

    elif spec.shape == "rectangle":
        if is_3d:
            cz = float(getattr(spec, "center_z", 0.0))
            mask = (
                (np.abs(centers[:, 0] - cx) < rx)
                & (np.abs(centers[:, 1] - cy) < ry)
                & (np.abs(centers[:, 2] - cz) < rz)
            )
        else:
            mask = (np.abs(centers[:, 0] - cx) < rx) & (np.abs(centers[:, 1] - cy) < ry)
        values[mask] = spec.conductivity

    else:
        log.warning("Unknown shape %r, falling back to circle", spec.shape)
        if is_3d:
            cz = float(getattr(spec, "center_z", 0.0))
            dist2 = (
                (centers[:, 0] - cx) ** 2
                + (centers[:, 1] - cy) ** 2
                + (centers[:, 2] - cz) ** 2
            )
        else:
            dist2 = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
        values[dist2 < rx**2] = spec.conductivity


def _total_electrode_count(forward_cfg: ForwardModelConfig) -> int:
    return max(int(forward_cfg.n_elec), 1) * max(int(forward_cfg.n_rings), 1)


def _contact_impedance_vector(value: Any, *, total_electrodes: int) -> np.ndarray:
    total = max(int(total_electrodes), 1)
    if value is None or value == "":
        return np.full(total, 0.01, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
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


def _resolve_forward_runtime(forward_cfg: ForwardModelConfig) -> dict[str, Any]:
    mesh_dim = int(forward_cfg.mesh_dimension)
    gui_profile = os.getenv("EIT_APP_GUI_PROFILE", "").strip().lower()

    def _auto(value: str, default: str) -> str:
        raw = str(value or "").strip().lower()
        return default if raw in {"", "auto"} else raw

    requested_profile = _auto(forward_cfg.acceleration_profile, "default")
    mesh_family = _auto(forward_cfg.mesh_family, "tetra")
    forward_backend = _auto(forward_cfg.forward_backend, "dolfinx")
    wants_gpu_request = gui_profile == "gpu" or requested_profile in {
        "gpu3d",
        "gpu3d_fused",
    }
    wants_structured_gpu = (
        mesh_dim == 3
        and mesh_family == "hex"
        and (wants_gpu_request or forward_backend == "cuda_structured")
    )
    wants_3d_cuda = mesh_dim == 3 and (
        wants_gpu_request or forward_backend == "cuda_structured"
    )

    acceleration_profile = requested_profile
    if wants_3d_cuda and acceleration_profile == "default":
        acceleration_profile = "gpu3d"
    if mesh_dim != 3 and acceleration_profile in {"gpu3d", "gpu3d_fused"}:
        acceleration_profile = "default"

    if wants_structured_gpu and forward_backend == "dolfinx":
        forward_backend = "cuda_structured"
    elif not wants_structured_gpu and forward_backend == "cuda_structured":
        # The structured CUDA backend is deliberately hex-only.  Keep tetra
        # on the stable generic DOLFINx path so forward and inverse use the
        # same CEM/Jacobian convention.
        forward_backend = "dolfinx"

    petsc_device = _auto(forward_cfg.petsc_device, "cuda" if wants_3d_cuda else "cpu")
    capability: dict[str, Any] = {}
    if mesh_dim == 3 and petsc_device == "cuda":
        try:
            capability = dict(probe_petsc_cuda_runtime())
        except Exception as exc:
            capability = {"errors": {"forward_solver_policy": str(exc)}}
    solver_policy = resolve_3d_cuda_forward_solver_policy(
        requested_solver_preset=_auto(forward_cfg.forward_solver_preset, "auto"),
        mesh_dim=mesh_dim,
        petsc_device=petsc_device,
        forward_backend=forward_backend,
        capability=capability,
        prefer_amgx=True,
    )
    mat_solve_policy = resolve_3d_cuda_mat_solve_policy(
        requested_mat_solve=_auto(
            forward_cfg.forward_mat_solve, "auto" if mesh_dim == 3 else "off"
        ),
        mesh_dim=mesh_dim,
        petsc_device=petsc_device,
        forward_backend=forward_backend,
        solver_preset=solver_policy["forward_solver_preset_effective"],
    )

    return {
        "solver_mode": _auto(
            forward_cfg.solver_mode, "fast" if mesh_dim == 3 else "strict"
        ),
        "line_search_mode": _auto(
            forward_cfg.line_search_mode, "fast" if mesh_dim == 3 else "full"
        ),
        "linear_solver": _auto(forward_cfg.linear_solver, "auto"),
        "preconditioner": _auto(forward_cfg.preconditioner, "auto"),
        "fast_linear_path": _auto(forward_cfg.fast_linear_path, "auto"),
        "forward_solver_preset": str(solver_policy["forward_solver_preset_effective"]),
        "forward_solver_preset_requested": str(
            solver_policy["forward_solver_preset_requested"]
        ),
        "forward_solver_policy_reason": str(
            solver_policy["forward_solver_policy_reason"]
        ),
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
        "device": _auto(forward_cfg.device, "cuda" if wants_3d_cuda else "auto"),
        "forward_backend": forward_backend,
        "mesh_family": mesh_family,
        "acceleration_profile": acceleration_profile,
    }


def _forward_runtime_diagnostics(system: Any) -> dict[str, Any]:
    """Return the small runtime summary the GUI should surface to users."""
    fwd_model = getattr(system, "fwd_model", None)
    backend_diag = (
        fwd_model.get_backend_diagnostics()
        if fwd_model is not None and hasattr(fwd_model, "get_backend_diagnostics")
        else {}
    )
    mesh = getattr(system, "mesh", None)
    return {
        "mesh_family": str(getattr(mesh, "mesh_family", "") or ""),
        "forward_backend": str(getattr(system, "forward_backend", "") or ""),
        "forward_backend_effective": str(
            backend_diag.get(
                "forward_backend_effective",
                getattr(fwd_model, "forward_backend", "")
                if fwd_model is not None
                else "",
            )
        ),
        "solver_preset": str(backend_diag.get("solver_preset", "")),
        "forward_solver_preset": str(backend_diag.get("solver_preset", "")),
        "forward_solver_policy_reason": str(
            backend_diag.get("forward_solver_policy_reason", "")
        ),
        "forward_solver_policy_warning": str(
            backend_diag.get("forward_solver_policy_warning", "")
        ),
        "petsc_device_requested": str(backend_diag.get("petsc_device_requested", "")),
        "petsc_device_effective": str(backend_diag.get("petsc_device_effective", "")),
        "petsc_amgx_available": bool(backend_diag.get("petsc_amgx_available", False)),
        "petsc_hypre_available": bool(backend_diag.get("petsc_hypre_available", False)),
        "petsc_hypre_cuda_blacklisted": bool(
            backend_diag.get("petsc_hypre_cuda_blacklisted", False)
        ),
        "forward_mat_solve_effective": str(
            backend_diag.get("forward_mat_solve_effective", "")
        ),
        "forward_mat_solve_policy_reason": str(
            backend_diag.get("forward_mat_solve_policy_reason", "")
        ),
        "torch_device": str(getattr(system, "device", "") or ""),
        "mesh_cache_hit": getattr(mesh, "_pyeidors_mesh_cache_hit", None),
        "mesh_cache_layer": getattr(mesh, "_pyeidors_mesh_cache_layer", None),
        "mesh_cache_name": getattr(mesh, "_pyeidors_mesh_cache_name", None),
    }


class _ForwardSolverWorker(QObject):
    finished = Signal(object)  # ForwardSolverResult
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, request: ForwardSolverRequest) -> None:
        super().__init__()
        self._request = request

    def run(self) -> None:
        req = self._request
        try:
            self.progress.emit("Initializing EIT system...")
            from pyeidors import EITSystem
            from pyeidors.data.structures import PatternConfig
            from pyeidors.electrodes.layout import effective_pattern_layout_for_3d_mesh
            from pyeidors.femx import cell_midpoints

            forward_cfg = ForwardModelConfig.from_mapping(
                req.forward_model_config
                or {
                    "mesh_dimension": req.mesh_dimension,
                    "mesh_refinement": req.mesh_refinement,
                    "n_elec": req.n_electrodes,
                    "background_conductivity": req.background_conductivity,
                    "noise_level": req.noise_level,
                }
            )
            total_electrodes = _total_electrode_count(forward_cfg)
            pattern_n_elec, pattern_n_rings = effective_pattern_layout_for_3d_mesh(
                mesh_tdim=forward_cfg.mesh_dimension,
                n_elec=forward_cfg.n_elec,
                n_rings=forward_cfg.n_rings,
                electrode_layout=forward_cfg.electrode_layout,
            )
            pattern = PatternConfig(
                n_elec=pattern_n_elec,
                n_rings=pattern_n_rings,
                stim_pattern=forward_cfg.stim_pattern,
                meas_pattern=forward_cfg.meas_pattern,
                electrode_layout=forward_cfg.electrode_layout,
                measurement_protocol=forward_cfg.measurement_protocol,
                custom_stim_matrix=forward_cfg.custom_stim_matrix,
                custom_meas_matrices=forward_cfg.custom_meas_matrices,
                drive_mode=forward_cfg.drive_mode,
                drive_value=forward_cfg.drive_value,
                geometry_scale_to_m=forward_cfg.geometry_scale_to_m,
                electrode_length_m_override=forward_cfg.electrode_length_m_override,
                use_meas_current=forward_cfg.use_meas_current,
                use_meas_current_next=forward_cfg.use_meas_current_next,
                rotate_meas=forward_cfg.rotate_meas,
                stim_direction=forward_cfg.stim_direction,
                meas_direction=forward_cfg.meas_direction,
                stim_first_positive=forward_cfg.stim_first_positive,
            )
            runtime = _resolve_forward_runtime(forward_cfg)
            system = EITSystem(
                n_elec=total_electrodes,
                pattern_config=pattern,
                contact_impedance=_contact_impedance_vector(
                    forward_cfg.contact_impedance,
                    total_electrodes=total_electrodes,
                ),
                base_conductivity=forward_cfg.background_conductivity,
                solver_mode=runtime["solver_mode"],
                line_search_mode=runtime["line_search_mode"],
                linear_solver=runtime["linear_solver"],
                preconditioner=runtime["preconditioner"],
                fast_linear_path=runtime["fast_linear_path"],
                linear_backend_config={
                    "solver_preset": runtime["forward_solver_preset"],
                    "mat_solve_mode": runtime["forward_mat_solve"],
                    "petsc_device": runtime["petsc_device"],
                },
                petsc_device=runtime["petsc_device"],
                device=runtime["device"],
                forward_backend=runtime["forward_backend"],
                mesh_family=runtime["mesh_family"],
                acceleration_profile=runtime["acceleration_profile"],
            )

            self.progress.emit("Generating mesh...")
            system.setup(
                mesh_source="generated",
                dimension=forward_cfg.mesh_dimension,
                mesh_size=forward_cfg.mesh_refinement,
                radius=forward_cfg.radius,
                height=forward_cfg.height,
                electrode_coverage=forward_cfg.electrode_coverage,
                electrode_height_ratio=forward_cfg.electrode_height_ratio,
                electrode_level_fractions=forward_cfg.electrode_level_fractions,
                z_center=forward_cfg.z_center,
                mesh_family=runtime["mesh_family"],
                geometry_version=forward_cfg.geometry_version,
                electrode_layout=forward_cfg.electrode_layout,
            )

            self.progress.emit("Building conductivity distribution...")
            fwd = system.fwd_model
            centers = cell_midpoints(fwd.mesh)
            sigma = np.full(
                len(centers), forward_cfg.background_conductivity, dtype=np.float64
            )
            for spec in req.inhomogeneities:
                _paint_shape(
                    sigma, centers, spec, mesh_dimension=forward_cfg.mesh_dimension
                )

            self.progress.emit("Running forward solve...")
            data = system.forward_solve(sigma)

            # Also solve homogeneous for difference reference
            self.progress.emit("Computing homogeneous reference...")
            sigma_homog = np.full_like(sigma, forward_cfg.background_conductivity)
            data_homog = system.forward_solve(sigma_homog)

            # Add noise if requested
            voltages = data.meas.copy()
            homog_voltages = data_homog.meas.copy()
            if forward_cfg.noise_level > 0:
                rng = np.random.default_rng()
                noise_std = forward_cfg.noise_level * np.std(voltages)
                voltages += noise_std * rng.standard_normal(voltages.shape)

            # Extract mesh geometry
            mesh = system.mesh
            node_coords = mesh.geometry.x[:, : forward_cfg.mesh_dimension].copy()
            cells = mesh.topology.connectivity(mesh.topology.dim, 0)
            n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            cell_connectivity = np.array(
                [cells.links(i) for i in range(n_cells)], dtype=np.int32
            )

            out_dtype = compute_dtype()
            result = ForwardSolverResult(
                boundary_voltages=np.asarray(voltages, dtype=out_dtype),
                ground_truth_conductivity=np.asarray(sigma, dtype=out_dtype),
                node_coords=node_coords,
                cell_connectivity=cell_connectivity,
                n_elements=n_cells,
                n_measurements=len(voltages),
                homogeneous_voltages=np.asarray(homog_voltages, dtype=out_dtype),
                forward_model_config={
                    **forward_cfg.to_mapping(),
                    **runtime,
                    "runtime_diagnostics": _forward_runtime_diagnostics(system),
                },
            )
            self.progress.emit("Forward solve complete.")
            self.finished.emit(result)

        except Exception as exc:
            log.exception("Forward solver failed")
            self.error.emit(str(exc))
            self.finished.emit(
                ForwardSolverResult(
                    boundary_voltages=np.array([]),
                    ground_truth_conductivity=np.array([]),
                    node_coords=np.array([]),
                    cell_connectivity=np.array([]),
                    n_elements=0,
                    n_measurements=0,
                    error_msg=str(exc),
                )
            )


class ForwardSolverController(QObject):
    """Manages forward problem solving in a background thread."""

    forward_done = Signal(object)  # ForwardSolverResult
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _ForwardSolverWorker | None = None

    def solve(self, request: ForwardSolverRequest) -> None:
        """Start a forward solve in a background thread."""
        if self._thread is not None and self._thread.isRunning():
            self.error.emit("A forward solve is already running.")
            return

        self._thread = QThread()
        self._worker = _ForwardSolverWorker(request)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self.progress)
        self._worker.error.connect(self.error)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()

    def _on_finished(self, result: ForwardSolverResult) -> None:
        self.forward_done.emit(result)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def shutdown(self) -> None:
        """Stop any running worker and clean up."""
        self._cleanup()
