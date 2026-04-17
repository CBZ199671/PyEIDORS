"""Container widget for the Hardware Measurement tab."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QSplitter,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.hardware.acquisition_panel import AcquisitionPanel
from eit_app.ui.hardware.connection_panel import ConnectionPanel
from eit_app.ui.hardware.control_panel import ControlPanel
from eit_app.ui.hardware.frame_browser_widget import FrameBrowserWidget
from eit_app.ui.hardware.live_plot_widget import LivePlotWidget
from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget
from eit_app.ui.hardware.session_summary_panel import SessionSummaryPanel
from eit_app.ui.theme import set_panel_role
from eit_app.ui.workflow_shell import WorkflowShell


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
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        self._conn_panel = ConnectionPanel()
        self._control_panel = ControlPanel()
        self._acq_panel = AcquisitionPanel()
        set_panel_role(self._conn_panel, "workflow")
        set_panel_role(self._control_panel, "workflow")
        set_panel_role(self._acq_panel, "workflow")

        center_splitter = QSplitter(Qt.Orientation.Vertical)
        center_splitter.setChildrenCollapsible(False)

        self._live_plot = LivePlotWidget()

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.setChildrenCollapsible(False)

        self._recon_widget = ReconstructionWidget()
        self._voltage_plot = BoundaryVoltagePlotWidget(mode="hardware")

        bottom_splitter.addWidget(self._recon_widget)
        bottom_splitter.addWidget(self._voltage_plot)
        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 1)
        # Compact default — 460 px total split evenly between
        # reconstruction and voltage plot.  User can always drag the
        # handle to give one of them more room.
        bottom_splitter.setSizes([240, 220])

        center_splitter.addWidget(self._live_plot)
        center_splitter.addWidget(bottom_splitter)
        center_splitter.setStretchFactor(0, 2)
        center_splitter.setStretchFactor(1, 1)
        # Live plot gets ~340 px tall by default so the X-axis ticks
        # are visible without scrolling on an 800-tall window.
        center_splitter.setSizes([340, 240])

        self._summary_panel = SessionSummaryPanel()
        self._frame_browser = FrameBrowserWidget()

        # Step titles are assigned by _retranslate(); pass empty strings now.
        # Right-context (frame browser) width unified to 300px across all
        # WorkflowShell-based tabs — see Phase 7 in TASKS.md.  Keeping
        # the total splitter width stable (was 1500) by trimming 40px
        # from the center panel.
        self._shell = WorkflowShell(
            steps=[
                ("", self._conn_panel),
                ("", self._control_panel),
                ("", self._acq_panel),
            ],
            center_widget=center_splitter,
            context_widget=self._frame_browser,
            left_footer=self._summary_panel,
            compact_toolbox=True,
            # Comfortable defaults via splitter_sizes; floors are kept
            # low so the user can shrink the window down to ~720 px on
            # a small laptop.  The horizontal scroll on the left
            # toolbox absorbs anything that would otherwise clip.
            step_min_width=260,
            context_min_width=220,
            # Splitter total opens at 1080 px so the centre + frame
            # browser fit comfortably on a 1280-px laptop.  Larger
            # screens are free to drag the splitter handles out;
            # smaller windows shrink everything proportionally.
            splitter_sizes=(360, 480, 240),
            parent=self,
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._shell)
        self._workflow_toolbox = self._shell.toolbox

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh Step titles on the left-panel QToolBox."""
        toolbox = self._workflow_toolbox
        toolbox.setItemText(0, t("hw.step.link"))
        toolbox.setItemText(1, t("hw.step.setup"))
        toolbox.setItemText(2, t("hw.step.acquire"))

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
    def main_splitter(self) -> QSplitter:
        return self._shell.main_splitter

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
