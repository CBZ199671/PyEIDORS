"""Composite visualization widget for simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.conductivity_image_widget import ConductivityImageWidget

if TYPE_CHECKING:
    from eit_app.controllers.forward_solver_controller import ForwardSolverResult


class SimulationResultsWidget(QWidget):
    """Displays ground truth vs reconstruction + voltage fitting + metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._last_forward_result: ForwardSolverResult | None = None
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Top: side-by-side conductivity images — initial titles come from
        # the i18n dict; _retranslate() re-applies them on language change.
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._ground_truth_widget = ConductivityImageWidget(
            t("sim.results.ground_truth_title")
        )
        self._reconstruction_widget = ConductivityImageWidget(
            t("sim.results.reconstruction_title")
        )
        top_splitter.addWidget(self._ground_truth_widget)
        top_splitter.addWidget(self._reconstruction_widget)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setSizes([520, 520])

        self._voltage_plot = BoundaryVoltagePlotWidget(mode="simulation")

        # Main vertical splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self._voltage_plot)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setSizes([520, 280])

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
            title=t("sim.results.ground_truth_title"),
        )
        self._reconstruction_widget.clear()

        self._voltage_plot.update_simulation_voltages(
            ground_truth=result.boundary_voltages,
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
            title=t("sim.results.reconstruction_title"),
        )

        if self._last_forward_result is not None:
            self._voltage_plot.update_simulation_voltages(
                ground_truth=self._last_forward_result.boundary_voltages,
                reconstructed=reconstructed_voltages,
            )

    def clear(self) -> None:
        self._ground_truth_widget.clear()
        self._reconstruction_widget.clear()
        self._voltage_plot.clear()
        self._last_forward_result = None

    def set_loading_forward(self, on: bool) -> None:
        """Mark the ground-truth image + voltage plot as busy during
        a forward solve.  Called by main_window's forward lifecycle
        slots so the plots advertise that work is in flight.
        """
        if on:
            self._ground_truth_widget.set_loading(
                t("sim.results.ground_truth_loading")
            )
            # When a fresh forward run starts, yesterday's reconstruction
            # is no longer meaningful.
            self._reconstruction_widget.clear()
            self._voltage_plot.set_loading(True)
        else:
            # update_forward_result() repaints on success; if the solver
            # errored with no result, drop back to a clean state instead
            # of leaving the "Solving…" caption stuck on screen.
            if self._last_forward_result is None:
                self._ground_truth_widget.clear()
                self._voltage_plot.set_loading(False)

    def set_loading_inverse(self, on: bool) -> None:
        """Mark the reconstruction image + voltage plot as busy during
        an inverse solve."""
        if on:
            self._reconstruction_widget.set_loading(
                t("sim.results.reconstruction_loading")
            )
            self._voltage_plot.set_loading(True)
        else:
            # The next update_inverse_result() call will repaint on
            # success; if nothing arrived, fall back to a clean slate.
            self._voltage_plot.set_loading(False)

    def set_expected_point_count(self, point_count: int) -> None:
        self._voltage_plot.set_expected_point_count(point_count)

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh the two conductivity-image titles on language change."""
        self._ground_truth_widget.setTitle(t("sim.results.ground_truth_title"))
        self._reconstruction_widget.setTitle(t("sim.results.reconstruction_title"))

    @property
    def voltage_plot(self) -> BoundaryVoltagePlotWidget:
        return self._voltage_plot
