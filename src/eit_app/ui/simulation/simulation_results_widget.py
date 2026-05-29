"""Composite visualization widget for simulation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.complex_channels import (
    DISPLAY_CHANNELS,
    REAL_CHANNEL,
    channel_values,
    has_complex_component,
)
from eit_app.ui.conductivity_3d_widget import (
    Conductivity3DWidget,
    SUPPORTED_3D_CELL_VERTEX_COUNTS,
)
from eit_app.ui.conductivity_image_widget import ConductivityImageWidget
from eit_app.ui.electrode_overlay import (
    ElectrodeGeometry,
    electrode_geometry_from_config,
)

if TYPE_CHECKING:
    from eit_app.controllers.forward_solver_controller import ForwardSolverResult


def _is_3d_payload(node_coords: np.ndarray, cell_connectivity: np.ndarray) -> bool:
    """Detect a 3D volume mesh from the shape of incoming payload."""
    coords = np.asarray(node_coords)
    cells = np.asarray(cell_connectivity)
    if coords.ndim != 2 or coords.shape[1] < 3:
        return False
    if cells.ndim != 2 or cells.shape[1] not in SUPPORTED_3D_CELL_VERTEX_COUNTS:
        return False
    return bool(np.ptp(coords[:, 2]) > 1.0e-9)


class _ConductivityViewSlot(QWidget):
    """Holds a 2D matplotlib widget and a 3D PyVista widget side-by-side
    in a QStackedLayout, then dispatches calls to whichever matches the
    most recent payload's dimension.

    The 3D widget object is cheap to construct.  On runtimes where
    embedded Qt/VTK is unsafe (WSLg/offscreen/headless), that widget
    chooses a safe in-process 3D renderer internally; this dispatcher
    never downgrades 3D payloads to the 2D projection view.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._mpl = ConductivityImageWidget(title)
        self._three_d = Conductivity3DWidget(title)
        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._mpl)
        self._stack.addWidget(self._three_d)
        self._stack.setCurrentWidget(self._mpl)

    # ------------------------------------------------------------------
    # Size hints
    #
    # QStackedLayout's default size hints take the *union* of all
    # child hints.  That means the 2D matplotlib page — which shrinks
    # down to 10 × 10 — inherited the 3D page's ~628 × 232 minimum,
    # which cascaded up into the simulation tab's splitter and locked
    # the main window at a ~1260 px minimum width.  Responsive shrink
    # behaviour was gone.
    #
    # Overriding sizeHint() / minimumSizeHint() to return the active
    # child's hints only restores the old behaviour for the 2D case
    # and keeps the 3D case honest about what it really needs.
    # ------------------------------------------------------------------

    def sizeHint(self):  # noqa: N802 (Qt API)
        active = self._stack.currentWidget()
        return active.sizeHint() if active is not None else super().sizeHint()

    def minimumSizeHint(self):  # noqa: N802 (Qt API)
        # IMPORTANT: clamp to a tiny minimum (80 × 80) regardless of
        # which inner widget is active.  When the active child is the
        # PyVista 3D widget its own minimumSizeHint balloons to ~640
        # px wide because of the controls bar + plot canvas; if we
        # propagate that into the parent QSplitter, the splitter
        # respects the 640 px floor and cannot honour our 50 / 50
        # setSizes call after a forward solve — the 3D ground-truth
        # pane then visibly squeezes the still-empty reconstruction
        # pane to a few-pixel sliver.  Reporting a small minimum lets
        # the splitter distribute space evenly while the inner
        # widget happily renders into whatever space it gets.
        from PySide6.QtCore import QSize

        return QSize(80, 80)

    # Public API mirrors ConductivityImageWidget so the parent stays simple.

    def update_image(
        self,
        conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        title: str | None = None,
    ) -> None:
        if _is_3d_payload(node_coords, cell_connectivity):
            self._three_d.update_image(
                conductivity, node_coords, cell_connectivity, title=title
            )
            self._stack.setCurrentWidget(self._three_d)
        else:
            self._mpl.update_image(
                conductivity, node_coords, cell_connectivity, title=title
            )
            self._stack.setCurrentWidget(self._mpl)
        # Our sizeHint / minimumSizeHint overrides track the active
        # child, but parent layouts only re-query them on an explicit
        # geometry invalidation.
        self.updateGeometry()

    def set_electrode_geometry(self, geometry: ElectrodeGeometry | None) -> None:
        """Update electrode overlays on whichever child widgets can use it.

        The 2D widget consumes only the arc list; the 3D widget consumes
        only the patch list.  Sending ``None`` clears any cached overlay
        on both.
        """
        self._mpl.set_electrode_geometry(geometry)
        self._three_d.set_electrode_geometry(geometry)

    def clear(self) -> None:
        self._mpl.clear()
        self._three_d.clear()
        # Default to 2D view between runs so the placeholder caption
        # reads naturally.
        self._stack.setCurrentWidget(self._mpl)
        self.updateGeometry()

    def set_loading(self, message: str | None = None) -> None:
        # Drive the loading caption on whichever widget is visible —
        # users won't see the hidden one's spinner anyway.
        active = self._stack.currentWidget()
        if active is self._three_d:
            self._three_d.set_loading(message)
        else:
            self._mpl.set_loading(message)

    def setTitle(self, title: str) -> None:
        self._title = title
        self._mpl.setTitle(title)
        self._three_d.setTitle(title)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        # Forward shutdown to the 3D widget so it can tear the VTK
        # plotter down cleanly — otherwise pyvistaqt's render thread
        # outlives the QApplication and dumps QThreadStorage warnings
        # at process exit.
        try:
            self._three_d.close()
        except Exception:  # pragma: no cover — best-effort shutdown
            pass
        super().closeEvent(event)


class SimulationResultsWidget(QWidget):
    """Displays ground truth vs reconstruction + voltage fitting + metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._complex_mode = False
        self._current_channel = REAL_CHANNEL
        self._updating_channel_combo = False
        self._last_forward_result: ForwardSolverResult | None = None
        self._last_reconstruction_payload: dict[str, np.ndarray | None] | None = None
        self._last_electrode_geometry = None
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._mode_toolbar = QWidget()
        toolbar_layout = QHBoxLayout(self._mode_toolbar)
        toolbar_layout.setContentsMargins(6, 0, 6, 0)
        toolbar_layout.setSpacing(8)
        self._mode_label = QLabel("")
        self._channel_label = QLabel("")
        self._channel_combo = AutoCloseComboBox()
        for channel in DISPLAY_CHANNELS:
            self._channel_combo.addItem("", channel)
        self._channel_combo.currentIndexChanged.connect(
            lambda _idx: self._on_channel_changed()
        )
        toolbar_layout.addWidget(self._mode_label)
        toolbar_layout.addStretch(1)
        toolbar_layout.addWidget(self._channel_label)
        toolbar_layout.addWidget(self._channel_combo)
        layout.addWidget(self._mode_toolbar)

        # Top: side-by-side conductivity images — initial titles come from
        # the i18n dict; _retranslate() re-applies them on language change.
        self._top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._ground_truth_widget = _ConductivityViewSlot(
            t("sim.results.ground_truth_title")
        )
        self._reconstruction_widget = _ConductivityViewSlot(
            t("sim.results.reconstruction_title")
        )
        self._top_splitter.addWidget(self._ground_truth_widget)
        self._top_splitter.addWidget(self._reconstruction_widget)
        self._top_splitter.setStretchFactor(0, 1)
        self._top_splitter.setStretchFactor(1, 1)
        self._top_splitter.setChildrenCollapsible(False)
        self._top_splitter.setSizes([520, 520])

        self._voltage_plot = BoundaryVoltagePlotWidget(mode="simulation")

        # Main vertical splitter
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(self._top_splitter)
        main_splitter.addWidget(self._voltage_plot)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setSizes([520, 280])

        layout.addWidget(main_splitter)

    def _balance_top_splitter(self) -> None:
        """Force a 50 / 50 split between the GT and recon widgets.

        The dispatcher slots' sizeHint changes once one of them
        switches its active child away from the empty placeholder
        (matplotlib widgets gain a colorbar, PyVista widgets pick up
        a wider control bar, etc.).  QSplitter then re-runs its
        layout and biases extra space toward the side with the
        bigger hint, which made the GT pane visibly grow and squeeze
        the still-empty reconstruction pane after a forward solve.
        Rewriting setSizes after every update keeps both panes the
        same width.
        """
        total = max(self._top_splitter.width(), 1)
        half = total // 2
        target = [half, total - half]
        # Don't trigger a layout invalidation chain when the splitter
        # is already at the target sizes (typical case after the
        # first balance — every subsequent update_*/clear/set_loading
        # call would otherwise re-run the same setSizes).
        if list(self._top_splitter.sizes()) == target:
            return
        self._top_splitter.setSizes(target)

    def update_forward_result(self, result: ForwardSolverResult) -> None:
        """Show ground truth and boundary voltages from forward solve."""
        self._last_forward_result = result
        self._last_reconstruction_payload = None

        if result.error_msg:
            return

        # Resolve electrode overlay geometry from the forward config so
        # both the GT and (later) reconstruction widgets render the same
        # electrode footprints.  Computed once and forwarded; the
        # widgets cache & toggle without recomputing.
        geometry = electrode_geometry_from_config(
            getattr(result, "forward_model_config", None) or {}
        )
        self._last_electrode_geometry = geometry
        self._set_complex_mode(
            has_complex_component(
                result.ground_truth_conductivity,
                result.boundary_voltages,
                result.homogeneous_voltages,
            )
        )

        self._render_result_views()

    def update_inverse_result(
        self,
        reconstructed_conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        reconstructed_voltages: np.ndarray | None = None,
    ) -> None:
        """Show reconstruction alongside ground truth."""
        self._last_reconstruction_payload = {
            "conductivity": np.asarray(reconstructed_conductivity),
            "node_coords": np.asarray(node_coords),
            "cell_connectivity": np.asarray(cell_connectivity),
            "voltages": None
            if reconstructed_voltages is None
            else np.asarray(reconstructed_voltages),
        }
        self._set_complex_mode(
            self._complex_mode
            or has_complex_component(reconstructed_conductivity, reconstructed_voltages)
        )
        self._render_result_views()

    def clear(self) -> None:
        self._ground_truth_widget.clear()
        self._reconstruction_widget.clear()
        self._voltage_plot.clear()
        self._last_forward_result = None
        self._last_reconstruction_payload = None
        self._last_electrode_geometry = None
        self._set_complex_mode(False)
        self._balance_top_splitter()

    def set_loading_forward(self, on: bool) -> None:
        """Mark the ground-truth image + voltage plot as busy during
        a forward solve.  Called by main_window's forward lifecycle
        slots so the plots advertise that work is in flight.
        """
        if on:
            self._ground_truth_widget.set_loading(t("sim.results.ground_truth_loading"))
            # When a fresh forward run starts, yesterday's reconstruction
            # is no longer meaningful.
            self._reconstruction_widget.clear()
            self._voltage_plot.set_loading(True)
            self._balance_top_splitter()
        else:
            # update_forward_result() repaints on success; if the solver
            # errored with no result, drop back to a clean state instead
            # of leaving the "Solving…" caption stuck on screen.
            if self._last_forward_result is None:
                self._ground_truth_widget.clear()
                self._voltage_plot.set_loading(False)
                self._balance_top_splitter()

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

    def current_result_mode(self) -> str:
        return "complex" if self._complex_mode else "real"

    def current_channel(self) -> str:
        return self._current_channel

    def _on_channel_changed(self) -> None:
        if self._updating_channel_combo:
            return
        channel = str(self._channel_combo.currentData() or REAL_CHANNEL)
        if channel == self._current_channel:
            return
        self._current_channel = channel
        self._render_result_views()

    def _set_complex_mode(self, enabled: bool) -> None:
        self._complex_mode = bool(enabled)
        if not self._complex_mode:
            self._current_channel = REAL_CHANNEL
        self._refresh_channel_controls()

    def _refresh_channel_controls(self) -> None:
        self._updating_channel_combo = True
        try:
            self._mode_label.setText(
                t(
                    "sim.results.mode_complex"
                    if self._complex_mode
                    else "sim.results.mode_real"
                )
            )
            self._channel_label.setText(t("sim.results.channel_label"))
            for idx in range(self._channel_combo.count()):
                channel = str(self._channel_combo.itemData(idx) or REAL_CHANNEL)
                self._channel_combo.setItemText(
                    idx,
                    t(f"sim.results.channel.{channel}"),
                )
                if channel == self._current_channel:
                    self._channel_combo.setCurrentIndex(idx)
            if not self._complex_mode:
                self._channel_combo.setCurrentIndex(0)
            self._channel_combo.setEnabled(self._complex_mode)
            self._channel_label.setEnabled(self._complex_mode)
        finally:
            self._updating_channel_combo = False

    def _display_channel(self) -> str:
        return self._current_channel if self._complex_mode else REAL_CHANNEL

    def _channel_title(self, base_key: str) -> str:
        base = t(base_key)
        if not self._complex_mode:
            return base
        return f"{base} · {t(f'sim.results.channel.{self._display_channel()}')}"

    def _render_result_views(self) -> None:
        result = self._last_forward_result
        if result is None or result.error_msg:
            return
        channel = self._display_channel()
        geometry = self._last_electrode_geometry

        self._ground_truth_widget.update_image(
            channel_values(result.ground_truth_conductivity, channel),
            result.node_coords,
            result.cell_connectivity,
            title=self._channel_title("sim.results.ground_truth_title"),
        )
        self._ground_truth_widget.set_electrode_geometry(geometry)

        reconstructed_voltages = None
        payload = self._last_reconstruction_payload
        if payload is None:
            self._reconstruction_widget.clear()
            self._reconstruction_widget.set_electrode_geometry(geometry)
        else:
            self._reconstruction_widget.update_image(
                channel_values(payload["conductivity"], channel),
                np.asarray(payload["node_coords"]),
                np.asarray(payload["cell_connectivity"]),
                title=self._channel_title("sim.results.reconstruction_title"),
            )
            self._reconstruction_widget.set_electrode_geometry(geometry)
            reconstructed_voltages = payload.get("voltages")
        self._balance_top_splitter()

        self._voltage_plot.update_simulation_voltages(
            ground_truth=result.boundary_voltages,
            reconstructed=reconstructed_voltages,
            component=channel,
        )

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh the two conductivity-image titles on language change."""
        self._refresh_channel_controls()
        if self._last_forward_result is None:
            self._ground_truth_widget.setTitle(t("sim.results.ground_truth_title"))
            self._reconstruction_widget.setTitle(t("sim.results.reconstruction_title"))
        else:
            self._render_result_views()

    @property
    def voltage_plot(self) -> BoundaryVoltagePlotWidget:
        return self._voltage_plot
