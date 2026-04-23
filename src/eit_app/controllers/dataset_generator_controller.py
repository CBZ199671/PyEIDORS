"""Batch dataset generator for deep learning training data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from eit_app.controllers.forward_solver_controller import (
    _contact_impedance_vector,
    _paint_shape,
    _resolve_forward_runtime,
    _total_electrode_count,
)
from eit_app.io.hdf5_packages import (
    write_dataset_mesh_info_package,
    write_dataset_sample_package,
)
from eit_app.models.forward_model_config import ForwardModelConfig
from eit_app.models.simulation_state import DatasetGeneratorConfig, InhomogeneitySpec

log = logging.getLogger(__name__)


@dataclass
class DatasetGeneratorRequest:
    """Input for batch dataset generation."""

    config: DatasetGeneratorConfig
    forward_model_config: dict[str, Any] = field(default_factory=dict)


class _DatasetGeneratorWorker(QObject):
    finished = Signal(int)  # total samples generated
    progress = Signal(int, int)  # current, total
    error = Signal(str)

    def __init__(self, request: DatasetGeneratorRequest) -> None:
        super().__init__()
        self._request = request
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        cfg = self._request.config
        total = cfg.n_samples

        try:
            from pyeidors import EITSystem
            from pyeidors.data.structures import PatternConfig
            from pyeidors.electrodes.layout import effective_pattern_layout_for_3d_mesh
            from pyeidors.femx import cell_midpoints

            forward_cfg = ForwardModelConfig.from_mapping(
                self._request.forward_model_config
                or {
                    "mesh_dimension": cfg.mesh_dimension,
                    "mesh_refinement": cfg.mesh_refinement,
                    "n_elec": cfg.n_electrodes,
                    "noise_level": cfg.noise_level,
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
            system.setup(
                mesh_source="generated",
                dimension=forward_cfg.mesh_dimension,
                mesh_size=forward_cfg.mesh_refinement,
                radius=forward_cfg.radius,
                height=forward_cfg.height,
                electrode_height_ratio=forward_cfg.electrode_height_ratio,
                electrode_level_fractions=forward_cfg.electrode_level_fractions,
                z_center=forward_cfg.z_center,
                mesh_family=runtime["mesh_family"],
                geometry_version=forward_cfg.geometry_version,
                electrode_layout=forward_cfg.electrode_layout,
            )

            fwd = system.fwd_model
            centers = cell_midpoints(fwd.mesh)
            n_elements = len(centers)

            # Extract mesh geometry once
            mesh = system.mesh
            node_coords = mesh.geometry.x[:, : cfg.mesh_dimension].copy()
            cells_conn = mesh.topology.connectivity(mesh.topology.dim, 0)
            n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
            cell_connectivity = np.array(
                [cells_conn.links(i) for i in range(n_cells)], dtype=np.int32
            )

            # Compute homogeneous reference once
            sigma_homog = np.ones(n_elements, dtype=np.float64)
            data_homog = system.forward_solve(sigma_homog)
            homog_voltages = data_homog.meas.copy()

            # Prepare output directory
            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save mesh info once
            write_dataset_mesh_info_package(
                out_dir,
                node_coords=node_coords,
                cell_connectivity=cell_connectivity,
                n_electrodes=forward_cfg.n_elec,
                homogeneous_voltages=homog_voltages,
                forward_model_config=forward_cfg.to_mapping(),
                total_electrodes=total_electrodes,
            )

            rng = np.random.default_rng()
            generated = 0

            for i in range(total):
                if self._cancel:
                    log.info("Dataset generation cancelled at sample %d/%d", i, total)
                    break

                # Random background conductivity
                bg = rng.uniform(
                    cfg.background_conductivity_min,
                    cfg.background_conductivity_max,
                )
                sigma = np.full(n_elements, bg, dtype=np.float64)

                # Random number of inhomogeneities
                n_inhom = rng.integers(
                    cfg.n_inhomogeneities_min,
                    cfg.n_inhomogeneities_max + 1,
                )

                specs = []
                for _ in range(n_inhom):
                    shape = rng.choice(cfg.shapes)
                    if int(forward_cfg.mesh_dimension) == 3:
                        radial_scale = max(float(forward_cfg.radius), 1.0e-12)
                        half_height = max(float(forward_cfg.height) * 0.5, 1.0e-12)
                        cx = (
                            rng.uniform(cfg.position_min, cfg.position_max)
                            * radial_scale
                        )
                        cy = (
                            rng.uniform(cfg.position_min, cfg.position_max)
                            * radial_scale
                        )
                        cz = (
                            float(forward_cfg.z_center)
                            + rng.uniform(cfg.position_min, cfg.position_max)
                            * half_height
                        )
                        sx = rng.uniform(cfg.size_min, cfg.size_max) * radial_scale
                        sy = (
                            sx
                            if shape == "circle"
                            else rng.uniform(cfg.size_min, cfg.size_max) * radial_scale
                        )
                        sz = (
                            sx
                            if shape == "circle"
                            else rng.uniform(cfg.size_min, cfg.size_max) * half_height
                        )
                    else:
                        cx = rng.uniform(cfg.position_min, cfg.position_max)
                        cy = rng.uniform(cfg.position_min, cfg.position_max)
                        cz = 0.0
                        sx = rng.uniform(cfg.size_min, cfg.size_max)
                        sy = (
                            sx
                            if shape == "circle"
                            else rng.uniform(cfg.size_min, cfg.size_max)
                        )
                        sz = sx
                    cond = rng.uniform(cfg.conductivity_min, cfg.conductivity_max)
                    spec = InhomogeneitySpec(
                        shape=shape,
                        center_x=float(cx),
                        center_y=float(cy),
                        center_z=float(cz),
                        size_x=float(sx),
                        size_y=float(sy),
                        size_z=float(sz),
                        conductivity=float(cond),
                    )
                    specs.append(spec)
                    _paint_shape(
                        sigma, centers, spec, mesh_dimension=forward_cfg.mesh_dimension
                    )

                # Forward solve
                data = system.forward_solve(sigma)
                voltages = data.meas.copy()

                # Add noise if requested
                if cfg.noise_level > 0:
                    noise_std = cfg.noise_level * np.std(voltages)
                    voltages += noise_std * rng.standard_normal(voltages.shape)

                # Save sample
                write_dataset_sample_package(
                    out_dir,
                    i,
                    ground_truth=sigma,
                    boundary_voltages=voltages,
                    background_conductivity=bg,
                    n_inhomogeneities=n_inhom,
                )

                generated += 1
                self.progress.emit(generated, total)

            self.finished.emit(generated)

        except Exception as exc:
            log.exception("Dataset generation failed")
            self.error.emit(str(exc))
            self.finished.emit(0)


class DatasetGeneratorController(QObject):
    """Manages batch dataset generation in a background thread."""

    generation_done = Signal(int)  # total samples
    progress = Signal(int, int)  # current, total
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._worker: _DatasetGeneratorWorker | None = None

    def generate(self, request: DatasetGeneratorRequest) -> None:
        """Start dataset generation in a background thread."""
        if self._thread is not None and self._thread.isRunning():
            self.error.emit("Dataset generation is already running.")
            return

        self._thread = QThread()
        self._worker = _DatasetGeneratorWorker(request)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self.progress)
        self._worker.error.connect(self.error)
        self._worker.finished.connect(self._thread.quit)

        self._thread.start()

    def cancel(self) -> None:
        """Request cancellation of the running generation."""
        if self._worker is not None:
            self._worker.cancel()

    def _on_finished(self, total: int) -> None:
        self.generation_done.emit(total)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    def shutdown(self) -> None:
        """Cancel and clean up."""
        self.cancel()
        self._cleanup()
