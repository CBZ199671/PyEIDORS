"""Forward problem solver running in a background QThread."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from eit_app.models.simulation_state import InhomogeneitySpec

log = logging.getLogger(__name__)


@dataclass
class ForwardSolverRequest:
    """Input parameters for a single forward solve."""

    mesh_dimension: int = 2
    mesh_refinement: float = 0.1
    n_electrodes: int = 16
    background_conductivity: float = 1.0
    inhomogeneities: list[InhomogeneitySpec] = field(default_factory=list)
    noise_level: float = 0.0


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
    error_msg: str | None = None


def _paint_shape(
    values: np.ndarray,
    centers: np.ndarray,
    spec: InhomogeneitySpec,
) -> None:
    """Paint a single inhomogeneity shape onto element-centered values."""
    cx, cy = spec.center_x, spec.center_y
    rx, ry = spec.size_x, spec.size_y

    if spec.shape == "circle":
        dist2 = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
        values[dist2 < rx**2] = spec.conductivity

    elif spec.shape == "ellipse":
        if ry <= 0:
            ry = rx
        norm = ((centers[:, 0] - cx) / rx) ** 2 + ((centers[:, 1] - cy) / ry) ** 2
        values[norm < 1.0] = spec.conductivity

    elif spec.shape == "rectangle":
        mask = (
            (np.abs(centers[:, 0] - cx) < rx)
            & (np.abs(centers[:, 1] - cy) < ry)
        )
        values[mask] = spec.conductivity

    else:
        log.warning("Unknown shape %r, falling back to circle", spec.shape)
        dist2 = (centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2
        values[dist2 < rx**2] = spec.conductivity


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
            from pyeidors.femx import cell_midpoints

            pattern = PatternConfig(
                n_elec=req.n_electrodes,
                stim_pattern="{ad}",
                meas_pattern="{ad}",
                drive_mode="line_current_density",
                drive_value=1.0,
                geometry_scale_to_m=1.0,
            )
            system = EITSystem(n_elec=req.n_electrodes, pattern_config=pattern)

            self.progress.emit("Generating mesh...")
            system.setup(
                mesh_source="generated",
                dimension=req.mesh_dimension,
                mesh_size=req.mesh_refinement,
            )

            self.progress.emit("Building conductivity distribution...")
            fwd = system.fwd_model
            centers = cell_midpoints(fwd.mesh)
            sigma = np.full(len(centers), req.background_conductivity, dtype=np.float64)
            for spec in req.inhomogeneities:
                _paint_shape(sigma, centers, spec)

            self.progress.emit("Running forward solve...")
            data = system.forward_solve(sigma)

            # Also solve homogeneous for difference reference
            self.progress.emit("Computing homogeneous reference...")
            sigma_homog = np.full_like(sigma, req.background_conductivity)
            data_homog = system.forward_solve(sigma_homog)

            # Add noise if requested
            voltages = data.meas.copy()
            homog_voltages = data_homog.meas.copy()
            if req.noise_level > 0:
                rng = np.random.default_rng()
                noise_std = req.noise_level * np.std(voltages)
                voltages += noise_std * rng.standard_normal(voltages.shape)

            # Extract mesh geometry
            mesh = system.mesh
            node_coords = mesh.geometry.x[:, :req.mesh_dimension].copy()
            cells = mesh.topology.connectivity(
                mesh.topology.dim, 0
            )
            n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            cell_connectivity = np.array(
                [cells.links(i) for i in range(n_cells)], dtype=np.int32
            )

            result = ForwardSolverResult(
                boundary_voltages=voltages,
                ground_truth_conductivity=sigma,
                node_coords=node_coords,
                cell_connectivity=cell_connectivity,
                n_elements=n_cells,
                n_measurements=len(voltages),
                homogeneous_voltages=homog_voltages,
            )
            self.progress.emit("Forward solve complete.")
            self.finished.emit(result)

        except Exception as exc:
            log.exception("Forward solver failed")
            self.error.emit(str(exc))
            self.finished.emit(ForwardSolverResult(
                boundary_voltages=np.array([]),
                ground_truth_conductivity=np.array([]),
                node_coords=np.array([]),
                cell_connectivity=np.array([]),
                n_elements=0,
                n_measurements=0,
                error_msg=str(exc),
            ))


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
