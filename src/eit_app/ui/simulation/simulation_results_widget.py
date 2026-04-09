"""Composite visualization widget for simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSplitter, QVBoxLayout, QWidget

from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.conductivity_image_widget import ConductivityImageWidget
from eit_app.ui.simulation.metrics_panel import MetricsPanel

if TYPE_CHECKING:
    from eit_app.controllers.forward_solver_controller import ForwardSolverResult


class SimulationResultsWidget(QWidget):
    """Displays ground truth vs reconstruction + voltage fitting + metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._last_forward_result: ForwardSolverResult | None = None

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Top: side-by-side conductivity images
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._ground_truth_widget = ConductivityImageWidget("Ground Truth")
        self._reconstruction_widget = ConductivityImageWidget("Reconstruction")
        top_splitter.addWidget(self._ground_truth_widget)
        top_splitter.addWidget(self._reconstruction_widget)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)

        # Bottom: voltage plot + metrics
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self._voltage_plot = BoundaryVoltagePlotWidget()
        self._metrics_panel = MetricsPanel()
        self._metrics_panel.setMaximumWidth(280)

        bottom_layout.addWidget(self._voltage_plot, 1)
        bottom_layout.addWidget(self._metrics_panel)

        # Main vertical splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setChildrenCollapsible(False)

        layout.addWidget(main_splitter)

    def update_forward_result(self, result: ForwardSolverResult) -> None:
        """Show ground truth and boundary voltages from forward solve."""
        self._last_forward_result = result

        if result.error_msg:
            return

        self._ground_truth_widget.update_image(
            result.ground_truth_conductivity,
            result.node_coords,
            result.cell_connectivity,
            title="Ground Truth",
        )
        self._reconstruction_widget.clear()
        self._metrics_panel.clear()

        self._voltage_plot.update_voltages(
            simulated=result.boundary_voltages,
            homogeneous=result.homogeneous_voltages,
        )

    def update_inverse_result(
        self,
        reconstructed_conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        reconstructed_voltages: np.ndarray | None = None,
    ) -> None:
        """Show reconstruction alongside ground truth."""
        self._reconstruction_widget.update_image(
            reconstructed_conductivity,
            node_coords,
            cell_connectivity,
            title="Reconstruction",
        )

        if self._last_forward_result is not None:
            self._metrics_panel.update_metrics(
                self._last_forward_result.ground_truth_conductivity,
                reconstructed_conductivity,
            )

            # Update voltage plot with reconstructed overlay
            self._voltage_plot.update_voltages(
                simulated=self._last_forward_result.boundary_voltages,
                reconstructed=reconstructed_voltages,
                homogeneous=self._last_forward_result.homogeneous_voltages,
            )

    def clear(self) -> None:
        self._ground_truth_widget.clear()
        self._reconstruction_widget.clear()
        self._voltage_plot.clear()
        self._metrics_panel.clear()
        self._last_forward_result = None
