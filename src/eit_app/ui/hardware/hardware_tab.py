"""Container widget for the Hardware Measurement tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QScrollArea,
    QSplitter,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.hardware.acquisition_panel import AcquisitionPanel
from eit_app.ui.hardware.connection_panel import ConnectionPanel
from eit_app.ui.hardware.control_panel import ControlPanel
from eit_app.ui.hardware.frame_browser_widget import FrameBrowserWidget
from eit_app.ui.hardware.live_plot_widget import LivePlotWidget
from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget
from eit_app.ui.hardware.session_summary_panel import SessionSummaryPanel
from eit_app.ui.theme import set_panel_role


class HardwareTab(QWidget):
    """Top-level container for the hardware measurement workflow.

    Layout:
        Left   – scrollable QToolBox with Step 1-3 panels
        Center – LivePlot (top) | Reconstruction + Summary + VoltageFit (bottom)
        Right  – FrameBrowser
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: only Step 1-3 (scrollable) ──
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setMinimumWidth(420)
        left_scroll.setMaximumWidth(520)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

        self._conn_panel = ConnectionPanel()
        self._control_panel = ControlPanel()
        self._acq_panel = AcquisitionPanel()

        self._workflow_toolbox = QToolBox()
        self._workflow_toolbox.setObjectName("workflowToolbox")
        set_panel_role(self._conn_panel, "workflow")
        set_panel_role(self._control_panel, "workflow")
        set_panel_role(self._acq_panel, "workflow")
        self._workflow_toolbox.addItem(self._conn_panel, "Step 1 \u00b7 Link & Verify")
        self._workflow_toolbox.addItem(self._control_panel, "Step 2 \u00b7 Setup & Diagnostics")
        self._workflow_toolbox.addItem(self._acq_panel, "Step 3 \u00b7 Acquire & Record")

        left_layout.addWidget(self._workflow_toolbox, 1)
        left_scroll.setWidget(left_container)

        # ── Central visualization ──
        center_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: live measurement plot
        self._live_plot = LivePlotWidget()

        # Bottom: horizontal split  →  Reconstruction | (Summary + VoltageFit)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._recon_widget = ReconstructionWidget()

        right_info = QWidget()
        right_info_layout = QVBoxLayout(right_info)
        right_info_layout.setContentsMargins(0, 0, 0, 0)
        right_info_layout.setSpacing(4)

        self._summary_panel = SessionSummaryPanel()
        self._voltage_plot = BoundaryVoltagePlotWidget()

        right_info_layout.addWidget(self._summary_panel, 1)
        right_info_layout.addWidget(self._voltage_plot, 1)

        bottom_splitter.addWidget(self._recon_widget)
        bottom_splitter.addWidget(right_info)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)
        bottom_splitter.setChildrenCollapsible(False)

        center_splitter.addWidget(self._live_plot)
        center_splitter.addWidget(bottom_splitter)
        center_splitter.setStretchFactor(0, 2)
        center_splitter.setStretchFactor(1, 1)
        center_splitter.setChildrenCollapsible(False)

        # ── Right panel: frame browser ──
        self._frame_browser = FrameBrowserWidget()
        self._frame_browser.setMinimumWidth(320)
        self._frame_browser.setMaximumWidth(420)

        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(center_splitter)
        main_splitter.addWidget(self._frame_browser)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)

        root.addWidget(main_splitter)

    # ── Property accessors for signal wiring in main_window ──

    @property
    def connection_panel(self) -> ConnectionPanel:
        return self._conn_panel

    @property
    def control_panel(self) -> ControlPanel:
        return self._control_panel

    @property
    def acquisition_panel(self) -> AcquisitionPanel:
        return self._acq_panel

    @property
    def summary_panel(self) -> SessionSummaryPanel:
        return self._summary_panel

    @property
    def workflow_toolbox(self) -> QToolBox:
        return self._workflow_toolbox

    @property
    def live_plot(self) -> LivePlotWidget:
        return self._live_plot

    @property
    def reconstruction_widget(self) -> ReconstructionWidget:
        return self._recon_widget

    @property
    def frame_browser(self) -> FrameBrowserWidget:
        return self._frame_browser

    @property
    def voltage_plot(self) -> BoundaryVoltagePlotWidget:
        return self._voltage_plot
