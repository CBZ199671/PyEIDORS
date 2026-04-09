"""Container widget for the Simulation tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QSplitter, QVBoxLayout, QWidget

from eit_app.ui.simulation.dataset_generator_panel import DatasetGeneratorPanel
from eit_app.ui.simulation.forward_problem_panel import ForwardProblemPanel
from eit_app.ui.simulation.inhomogeneity_editor import InhomogeneityEditor
from eit_app.ui.simulation.inverse_problem_panel import InverseProblemPanel
from eit_app.ui.simulation.mesh_setup_panel import MeshSetupPanel
from eit_app.ui.simulation.simulation_results_widget import SimulationResultsWidget


class SimulationTab(QWidget):
    """Top-level container for the simulation workflow.

    Assembles left control panels and central visualization into a
    single QWidget for embedding in a QTabWidget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left panel (scrollable) ---
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setMinimumWidth(380)
        left_scroll.setMaximumWidth(480)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(6)

        self._mesh_panel = MeshSetupPanel()
        self._inhom_editor = InhomogeneityEditor()
        self._forward_panel = ForwardProblemPanel()
        self._inverse_panel = InverseProblemPanel()
        self._dataset_panel = DatasetGeneratorPanel()

        left_layout.addWidget(self._mesh_panel)
        left_layout.addWidget(self._inhom_editor)
        left_layout.addWidget(self._forward_panel)
        left_layout.addWidget(self._inverse_panel)
        left_layout.addWidget(self._dataset_panel)
        left_layout.addStretch()

        left_scroll.setWidget(left_container)

        # --- Central visualization ---
        self._results_widget = SimulationResultsWidget()

        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(self._results_widget)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)

        root.addWidget(main_splitter)

    # --- Property accessors for signal wiring ---

    @property
    def mesh_setup_panel(self) -> MeshSetupPanel:
        return self._mesh_panel

    @property
    def inhomogeneity_editor(self) -> InhomogeneityEditor:
        return self._inhom_editor

    @property
    def forward_problem_panel(self) -> ForwardProblemPanel:
        return self._forward_panel

    @property
    def inverse_problem_panel(self) -> InverseProblemPanel:
        return self._inverse_panel

    @property
    def dataset_generator_panel(self) -> DatasetGeneratorPanel:
        return self._dataset_panel

    @property
    def results_widget(self) -> SimulationResultsWidget:
        return self._results_widget
