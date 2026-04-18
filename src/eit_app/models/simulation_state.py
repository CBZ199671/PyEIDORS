"""Observable state and configuration models for the simulation module."""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal


@dataclass
class InhomogeneitySpec:
    """Specification for a single inhomogeneity (anomaly) in the domain."""

    # Shape names are dimension-neutral:
    # - 2D: circle / ellipse / rectangle paint areas.
    # - 3D: circle / ellipse / rectangle paint sphere / ellipsoid / box volumes.
    shape: str = "circle"
    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    size_x: float = 0.2  # radius for circle; half-width for rect/ellipse
    size_y: float = 0.2  # same as size_x for circle; half-height for rect/ellipse
    size_z: float = 0.2  # 3D radius/depth; ignored by 2D paints
    conductivity: float = 2.0


@dataclass
class SimulationConfig:
    """Mesh and forward-problem configuration for a single simulation run."""

    mesh_dimension: int = 2
    mesh_refinement: float = 0.1  # mesh_size parameter for EITSystem
    n_electrodes: int = 16
    background_conductivity: float = 1.0
    noise_level: float = 0.0
    inhomogeneities: list[InhomogeneitySpec] = field(default_factory=list)

    # Inverse problem settings
    recon_method: str = "one_step_noser"
    regularization_alpha: float = 1.0
    max_iterations: int = 10


@dataclass
class DatasetGeneratorConfig:
    """Configuration for batch deep-learning dataset generation."""

    n_samples: int = 1000
    output_dir: str = ""
    n_inhomogeneities_min: int = 1
    n_inhomogeneities_max: int = 3
    shapes: list[str] = field(default_factory=lambda: ["circle"])
    position_min: float = -0.7
    position_max: float = 0.7
    size_min: float = 0.05
    size_max: float = 0.3
    conductivity_min: float = 0.5
    conductivity_max: float = 3.0
    background_conductivity_min: float = 0.8
    background_conductivity_max: float = 1.2
    noise_level: float = 0.0

    # Mesh settings (shared with SimulationConfig)
    mesh_dimension: int = 2
    mesh_refinement: float = 0.1
    n_electrodes: int = 16


class SimulationState(QObject):
    """Observable state for the simulation module."""

    forward_running_changed = Signal(bool)
    inverse_running_changed = Signal(bool)
    dataset_running_changed = Signal(bool)
    dataset_progress_changed = Signal(int, int)  # current, total

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.simulation_config = SimulationConfig()
        self.dataset_config = DatasetGeneratorConfig()
        self._forward_running = False
        self._inverse_running = False
        self._dataset_running = False

    @property
    def forward_running(self) -> bool:
        return self._forward_running

    @forward_running.setter
    def forward_running(self, value: bool) -> None:
        if self._forward_running != value:
            self._forward_running = value
            self.forward_running_changed.emit(value)

    @property
    def inverse_running(self) -> bool:
        return self._inverse_running

    @inverse_running.setter
    def inverse_running(self, value: bool) -> None:
        if self._inverse_running != value:
            self._inverse_running = value
            self.inverse_running_changed.emit(value)

    @property
    def dataset_running(self) -> bool:
        return self._dataset_running

    @dataset_running.setter
    def dataset_running(self, value: bool) -> None:
        if self._dataset_running != value:
            self._dataset_running = value
            self.dataset_running_changed.emit(value)
