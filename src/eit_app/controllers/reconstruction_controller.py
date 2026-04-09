"""Runs EIT reconstruction in a background QThread.

Accepts reference/target frame pairs, builds MeasurementDataset,
and calls pyeidors EITSystem for difference reconstruction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from eit_app.models.frame_model import FrameData

log = logging.getLogger(__name__)


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
    mesh_refinement: int = 4
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

        try:
            self.progress.emit("Loading PyEIDORS...")
            from pyeidors import EITSystem
            from pyeidors.data import PatternConfig, MeasurementDataset

            # Build measurement vectors
            ref_vec = req.reference_frame.to_measurement_vector(req.use_part)
            tgt_vec = req.target_frame.to_measurement_vector(req.use_part)

            meta = dict(req.metadata)
            meta.setdefault("n_elec", 16)
            meta.setdefault("stim_pattern", "{ad}")
            meta.setdefault("meas_pattern", "{ad}")
            meta.setdefault("drive_mode", "total_current")
            meta.setdefault("drive_value", 1.0e-5)
            meta.setdefault("geometry_scale_to_m", 1.0)
            meta.setdefault("difference_mode", "raw")
            meta.setdefault("difference_orientation", "target_minus_reference")

            # Build reference and target datasets
            self.progress.emit("Building measurement datasets...")
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

            # Set up EIT system
            self.progress.emit("Setting up EIT system...")
            pattern_config = PatternConfig(
                n_elec=meta["n_elec"],
                stim_pattern=meta["stim_pattern"],
                meas_pattern=meta["meas_pattern"],
                drive_mode=meta["drive_mode"],
                drive_value=meta["drive_value"],
                geometry_scale_to_m=meta["geometry_scale_to_m"],
            )
            system = EITSystem(
                n_elec=meta["n_elec"],
                pattern_config=pattern_config,
                regularization_alpha=req.regularization_alpha,
                difference_mode=meta["difference_mode"],
                difference_orientation=meta["difference_orientation"],
            )
            try:
                system.setup(mesh_source="cache", gdim=req.mesh_dimension)
            except Exception:
                mesh_size = max(0.02, 0.25 / max(1, req.mesh_refinement))
                system.setup(
                    mesh_source="generated",
                    dimension=req.mesh_dimension,
                    mesh_size=mesh_size,
                )

            # Run difference reconstruction
            self.progress.emit("Running reconstruction...")
            method = req.method.strip().lower()
            if method == "gn-absolute":
                recon = system.absolute_reconstruct(measurement_data=tgt_eit)
            else:
                if method == "sparse-bayes":
                    self.progress.emit("sparse-bayes 暂按差分工作流执行")
                recon = system.difference_reconstruct(
                    measurement_data=tgt_eit,
                    reference_data=ref_eit,
                )

            # Extract mesh geometry for visualization
            mesh = system.mesh
            coords = mesh.coordinates()
            cells = mesh.cells()

            result = ReconstructionResult(
                conductivity=recon.conductivity
                if hasattr(recon, "conductivity")
                else np.asarray([]),
                node_coords=coords,
                cell_connectivity=cells,
                measured=getattr(recon, "measured", None),
                simulated=getattr(recon, "simulated", None),
            )
            self.progress.emit("Reconstruction complete")
            self.finished.emit(result)

        except Exception as exc:
            log.exception("Reconstruction failed")
            self.error.emit(str(exc))
            self.finished.emit(
                ReconstructionResult(
                    conductivity=np.array([]),
                    node_coords=np.array([]),
                    cell_connectivity=np.array([]),
                    error_msg=str(exc),
                )
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

    def reconstruct(self, request: ReconstructionRequest) -> None:
        """Submit a reconstruction request. Runs in a background thread."""
        if self._busy:
            self.error.emit("Reconstruction already in progress")
            return

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
