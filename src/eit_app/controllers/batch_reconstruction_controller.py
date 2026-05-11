"""Batch reconstruction controller.

Iterates through all frame CSVs in an input folder and runs a chosen
reconstruction algorithm on each, saving conductivity images and
optional voltage-fit plots to an output folder.

For difference methods a single reference frame is provided; if it
lives inside the same folder it is skipped as a target.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from eit_app.controllers.reconstruction_controller import (
    ReconstructionRequest,
    ReconstructionResult,
    run_reconstruction_request,
)
from eit_app.models.frame_model import FrameData
from eit_app.models.reconstruction_methods import (
    database_method_requires_reference,
    prepare_database_reconstruction_method,
)

log = logging.getLogger(__name__)


_FRAME_CSV_RE = re.compile(r"_frame_(\d+)\.csv$", re.IGNORECASE)


def _discover_frame_csvs(folder: Path) -> list[Path]:
    """Return sorted list of frame CSV files in a folder."""
    if not folder.exists():
        return []
    results: list[Path] = []
    for csv_file in folder.iterdir():
        if not csv_file.is_file() or csv_file.suffix.lower() != ".csv":
            continue
        if csv_file.name.endswith("_AD.csv"):
            continue
        results.append(csv_file)

    # Sort by frame index if names follow the per-frame pattern, else name
    def key(p: Path) -> tuple:
        m = _FRAME_CSV_RE.search(p.name)
        return (0, int(m.group(1))) if m else (1, p.name)

    return sorted(results, key=key)


class BatchReconstructionRequest:
    """Configuration for a batch reconstruction run."""

    def __init__(
        self,
        *,
        input_folder: Path,
        output_folder: Path,
        method: str,
        method_label: str,
        reference_csv: Path | None,
        use_part: str,
        regularization_alpha: float,
        max_iterations: int,
        save_recon_image: bool,
        save_voltage_fit: bool,
        lambda_eff_custom_enabled: bool = False,
        custom_lambda_eff: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.method = method
        self.method_label = method_label
        self.reference_csv = Path(reference_csv) if reference_csv else None
        self.use_part = use_part
        self.regularization_alpha = regularization_alpha
        self.max_iterations = max_iterations
        self.lambda_eff_custom_enabled = bool(lambda_eff_custom_enabled)
        self.custom_lambda_eff = (
            float(custom_lambda_eff) if custom_lambda_eff is not None else None
        )
        self.save_recon_image = save_recon_image
        self.save_voltage_fit = save_voltage_fit
        self.metadata = dict(metadata or {})


class _BatchWorker(QObject):
    """Runs the batch job in a background thread."""

    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(int, int)  # succeeded, failed
    error = Signal(str)

    def __init__(self, request: BatchReconstructionRequest) -> None:
        super().__init__()
        self._request = request
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        req = self._request
        try:
            targets = _discover_frame_csvs(req.input_folder)
            if not targets:
                self.error.emit(f"No frame CSV files found in {req.input_folder}")
                self.finished.emit(0, 0)
                return

            needs_reference = database_method_requires_reference(req.method)
            ref_frame: FrameData | None = None
            ref_path_resolved: Path | None = None

            if needs_reference:
                if req.reference_csv is None:
                    self.error.emit("Difference methods require a reference frame.")
                    self.finished.emit(0, 0)
                    return
                ref_path_resolved = req.reference_csv.resolve()
                try:
                    ref_frame = _load_frame(req.reference_csv)
                except Exception as exc:
                    self.error.emit(f"Failed to load reference frame: {exc}")
                    self.finished.emit(0, 0)
                    return

            # Filter targets: skip the reference if it's in the input folder
            effective_targets: list[Path] = []
            for p in targets:
                if ref_path_resolved is not None and p.resolve() == ref_path_resolved:
                    continue
                effective_targets.append(p)

            if not effective_targets:
                self.error.emit(
                    "No target frames remain after excluding the reference."
                )
                self.finished.emit(0, 0)
                return

            req.output_folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            total = len(effective_targets)
            succeeded = 0
            failed = 0

            for idx, tgt_path in enumerate(effective_targets):
                if self._cancelled:
                    log.info("Batch cancelled at %d/%d", idx, total)
                    break

                self.progress.emit(idx + 1, total, f"Processing {tgt_path.name}")

                try:
                    tgt_frame = _load_frame(tgt_path)
                    request_obj = _build_request(
                        req,
                        reference=ref_frame if ref_frame is not None else tgt_frame,
                        target=tgt_frame,
                    )

                    def _silent_progress(msg: str) -> None:
                        # keep worker logs quiet during batch; per-job messages
                        # are too noisy for the GUI
                        log.debug("batch recon [%d/%d]: %s", idx + 1, total, msg)

                    result = run_reconstruction_request(
                        request_obj, progress_cb=_silent_progress
                    )
                    if result.error_msg:
                        failed += 1
                        log.warning(
                            "Batch item %s failed: %s", tgt_path.name, result.error_msg
                        )
                        continue

                    _save_outputs(result, tgt_path, req, stamp)
                    succeeded += 1
                except Exception:
                    failed += 1
                    log.exception("Batch item %s crashed", tgt_path.name)

            self.finished.emit(succeeded, failed)
        except Exception as exc:
            log.exception("Batch worker crashed")
            self.error.emit(str(exc))
            self.finished.emit(0, 0)


def _load_frame(path: Path) -> FrameData:
    from pyeidors.data.frame_io import read_frame_csv

    real, imag = read_frame_csv(path)
    match = _FRAME_CSV_RE.search(path.name)
    frame_index = int(match.group(1)) if match else 0
    timestamp = path.stat().st_mtime
    return FrameData(
        real=real,
        imag=imag,
        timestamp=timestamp,
        frame_index=frame_index,
    )


def _build_request(
    batch: BatchReconstructionRequest,
    *,
    reference: FrameData,
    target: FrameData,
) -> ReconstructionRequest:
    meta = {
        "n_elec": 16,
        "n_rings": 1,
        "stim_pattern": "{ad}",
        "meas_pattern": "{ad}",
        "rotate_meas": True,
        "use_meas_current": False,
        "use_meas_current_next": 0,
        "stim_direction": "ccw",
        "meas_direction": "ccw",
        "stim_first_positive": False,
        "drive_mode": "total_current",
        "drive_value": 1.0e-5,
        "geometry_scale_to_m": 1.0,
        "radius": 1.0,
        "electrode_coverage": 0.5,
        "electrode_length_m_override": None,
        "contact_impedance": 0.01,
        "difference_mode": "raw",
        "difference_orientation": "target_minus_reference",
    }
    meta.update(batch.metadata)
    alpha = (
        batch.custom_lambda_eff
        if batch.lambda_eff_custom_enabled and batch.custom_lambda_eff is not None
        else batch.regularization_alpha
    )
    prepared = prepare_database_reconstruction_method(
        batch.method,
        regularization_alpha=float(alpha),
        max_iterations=batch.max_iterations,
        custom_lambda_eff_enabled=batch.lambda_eff_custom_enabled,
        metadata=meta,
    )
    return ReconstructionRequest(
        reference_frame=reference,
        target_frame=target,
        use_part=batch.use_part,
        method=prepared.method,
        regularization_alpha=prepared.regularization_alpha,
        max_iterations=prepared.max_iterations,
        mesh_dimension=2,
        mesh_refinement=4,
        metadata=prepared.metadata,
    )


def _save_outputs(
    result: ReconstructionResult,
    target_path: Path,
    batch: BatchReconstructionRequest,
    stamp: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib import pyplot as plt
    from matplotlib.tri import Triangulation

    method_slug = batch.method.replace("-", "_")
    base = f"{stamp}_{method_slug}_{target_path.stem}"
    out = batch.output_folder

    if batch.save_recon_image:
        sigma = np.asarray(result.conductivity, dtype=float).reshape(-1)
        coords = np.asarray(result.node_coords, dtype=float)
        cells = np.asarray(result.cell_connectivity, dtype=int)
        if sigma.size > 0 and coords.size > 0 and cells.size > 0:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
            fig.patch.set_facecolor("#f4f7fb")
            ax.set_facecolor("#fbfdff")
            tri = Triangulation(coords[:, 0], coords[:, 1], cells)
            if sigma.size == len(cells):
                tpc = ax.tripcolor(tri, sigma, shading="flat", cmap="viridis")
            else:
                tpc = ax.tripcolor(tri, sigma, shading="gouraud", cmap="viridis")
            ax.set_aspect("equal")
            ax.set_title(f"{batch.method_label}  ·  {target_path.stem}")
            fig.colorbar(tpc, ax=ax, label="S/m")
            fig.tight_layout()
            fig.savefig(out / f"{base}_conductivity.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    if batch.save_voltage_fit and result.measured is not None:
        measured = np.asarray(result.measured, dtype=float).reshape(-1)
        if measured.size > 0:
            x = np.arange(1, measured.size + 1)
            fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
            fig.patch.set_facecolor("#f4f7fb")
            ax.set_facecolor("#fbfdff")
            ax.plot(x, measured, color="#4ecdc4", label="Measured")
            if result.simulated is not None:
                sim = np.asarray(result.simulated, dtype=float).reshape(-1)
                ax.plot(
                    x, sim, color="#ff6b6b", linestyle="--", label="Reconstructed fit"
                )
            ax.set_xlabel("Measurement index")
            ax.set_ylabel("Voltage (V)")
            ax.set_title(f"Voltage fit · {target_path.stem}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"{base}_voltage_fit.png", dpi=120, bbox_inches="tight")
            plt.close(fig)


class BatchReconstructionController(QObject):
    """Qt-facing controller for batch reconstruction jobs."""

    progress = Signal(int, int, str)  # current, total, message
    finished = Signal(int, int)  # succeeded, failed
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _BatchWorker | None = None

    def start(self, request: BatchReconstructionRequest) -> bool:
        if self._thread is not None and self._thread.isRunning():
            self.error.emit("A batch job is already running.")
            return False

        self._thread = QThread(self)
        self._worker = _BatchWorker(request)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress)
        self._worker.error.connect(self.error)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()
        return True

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def shutdown(self) -> None:
        self.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)

    def _on_finished(self, succeeded: int, failed: int) -> None:
        self.finished.emit(succeeded, failed)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
