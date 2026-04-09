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

    Assembles the left control panel, central visualization area,
    and right frame browser into a single QWidget suitable for
    embedding in a QTabWidget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left panel (scrollable to prevent cramping) ---
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setMinimumWidth(380)
        left_scroll.setMaximumWidth(480)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 4, 0)
        left_layout.setSpacing(4)

        self._summary_panel = SessionSummaryPanel()
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

        left_layout.addWidget(self._summary_panel)
        left_layout.addWidget(self._workflow_toolbox, 1)

        left_scroll.setWidget(left_container)

        # --- Central visualization ---
        center_splitter = QSplitter(Qt.Orientation.Vertical)
        self._live_plot = LivePlotWidget()
        self._recon_widget = ReconstructionWidget()
        center_splitter.addWidget(self._live_plot)
        center_splitter.addWidget(self._recon_widget)
        center_splitter.setStretchFactor(0, 2)
        center_splitter.setStretchFactor(1, 1)
        center_splitter.setChildrenCollapsible(False)

        # --- Right panel ---
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

    # --- Property accessors for signal wiring in main_window ---

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
