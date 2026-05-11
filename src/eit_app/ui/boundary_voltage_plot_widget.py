"""pyqtgraph-based boundary voltage comparison plot."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

import pyqtgraph as pg

from eit_app.i18n import t, translator
from eit_app.ui.fonts import serif_font_family
from eit_app.ui.plot_legend_overlay import LegendEntry, PlotLegendOverlay
from eit_app.ui.theme import (
    empty_placeholder_stylesheet,
    loading_scrim_stylesheet,
    plot_palette,
    subscribe_theme_mode,
)


class _PlotHost(QWidget):
    """Container that pins a child legend frame to its top-right corner.

    The boundary-voltage plot floats a translucent legend over the
    plot canvas.  Anchoring it to a fixed (18, 18) top-left position
    looked off-balance once we centred the title, and the user
    explicitly asked for a top-right initial placement.  Since the
    legend's width changes with the active language (the Chinese
    label is wider than the English one) we re-align on every
    resize and on every label refresh.
    """

    LEGEND_MARGIN = 12

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._legend_frame: QWidget | None = None

    def attach_legend(self, frame: QWidget) -> None:
        self._legend_frame = frame
        frame.installEventFilter(self)
        self._reposition_legend()

    def eventFilter(self, watched, event):  # noqa: N802 (Qt API)
        if watched is self._legend_frame and event.type() in (
            event.Type.Resize,
            event.Type.Show,
        ):
            self._reposition_legend()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._reposition_legend()

    def _reposition_legend(self) -> None:
        legend = self._legend_frame
        if legend is None:
            return
        host_w = self.width()
        legend_w = legend.width()
        if host_w <= 0 or legend_w <= 0:
            return
        x = max(self.LEGEND_MARGIN, host_w - legend_w - self.LEGEND_MARGIN)
        legend.move(x, self.LEGEND_MARGIN)


class BoundaryVoltagePlotWidget(QWidget):
    """Displays boundary voltages with optional overlay for comparison."""

    def __init__(self, mode: str = "simulation", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = str(mode).strip().lower()
        self._serif = serif_font_family()
        # Pull plot colors from the theme palette so dark mode applies
        # automatically.  subscribe_theme_mode at the end of __init__
        # rewires the plot on later mode flips.
        palette = plot_palette()
        self._plot_bg = palette["bg"]
        self._plot_text = palette["text"]
        self._plot_grid = palette["grid"]
        self._plot_border = palette["border"]
        self._point_count = 208
        self._expected_point_count = 208
        self._has_data = False
        # Phase 4 tri-state overlay: "empty" | "loading" | "data".
        self._overlay_state = "empty"
        self._label_style = {
            "color": self._plot_text,
            "font-family": self._serif,
            "font-size": "11pt",
        }

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(self._plot_bg)
        # Axis / title text is applied by _retranslate() below.
        self._plot_widget.showGrid(x=True, y=True, alpha=0.55)

        tick_font = QFont(self._serif, 9)
        for axis_name in ("left", "bottom"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setTextPen(pg.mkPen(self._plot_text))
            axis.setPen(pg.mkPen(self._plot_border))
            axis.setTickPen(pg.mkPen(self._plot_border))
            axis.setStyle(tickFont=tick_font)

        # Data curves
        self._curve_primary = self._plot_widget.plot(
            pen=pg.mkPen("#4ecdc4", width=2.0), name=self._primary_label()
        )
        self._curve_reconstructed_outline = self._plot_widget.plot(
            pen=pg.mkPen((255, 255, 255, 215), width=4.2),
        )
        self._curve_reconstructed = self._plot_widget.plot(
            pen=pg.mkPen("#ff6b6b", width=2.2, style=Qt.PenStyle.DashLine),
            name=self._secondary_label(),
        )
        self._curve_reconstructed_markers = self._plot_widget.plot(
            pen=None,
            symbol="o",
            symbolSize=6,
            symbolBrush=pg.mkBrush("#ff6b6b"),
            symbolPen=pg.mkPen("#ffffff", width=1.1),
        )
        self._curve_reconstructed_outline.setZValue(2)
        self._curve_primary.setZValue(3)
        self._curve_reconstructed.setZValue(4)
        self._curve_reconstructed_markers.setZValue(5)
        self._curve_reconstructed_outline.setVisible(False)
        self._curve_reconstructed_markers.setVisible(False)

        plot_host = _PlotHost()
        plot_stack = QStackedLayout(plot_host)
        plot_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        plot_stack.setContentsMargins(0, 0, 0, 0)
        plot_stack.addWidget(self._plot_widget)
        self._legend_entries = self._build_legend_entries()
        self._legend_frame = PlotLegendOverlay(
            self._legend_entries,
            interactive=False,
            compact=True,
            parent=plot_host,
        )
        # Pin the legend to the top-right corner; _PlotHost re-aligns
        # it on every resize and on every legend size change so it
        # stays glued to the corner regardless of language / font.
        plot_host.attach_legend(self._legend_frame)
        self._legend_frame.raise_()
        # Empty-overlay text is filled in by _retranslate() below.
        self._empty_overlay = QLabel("")
        self._empty_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_overlay.setWordWrap(True)
        self._empty_overlay.setStyleSheet(
            "color: #5b6573; "
            "font-size: 12px; "
            "font-weight: 600; "
            "background: transparent;"
        )
        plot_stack.addWidget(self._empty_overlay)
        layout.addWidget(plot_host)
        self._configure_index_axis(self._point_count)

        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        # Re-paint canvas + axis pens when the user toggles dark mode.
        subscribe_theme_mode(self._on_theme_mode_changed)

    def _on_theme_mode_changed(self, _mode: str) -> None:
        """Re-pull the plot palette and re-paint pyqtgraph canvas + axes.

        pyqtgraph caches the canvas brush + axis pens; QSS doesn't reach
        them, so push the new colors explicitly.
        """
        palette = plot_palette()
        self._plot_bg = palette["bg"]
        self._plot_text = palette["text"]
        self._plot_grid = palette["grid"]
        self._plot_border = palette["border"]
        self._label_style["color"] = self._plot_text
        self._plot_widget.setBackground(self._plot_bg)
        for axis_name in ("left", "bottom"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setTextPen(pg.mkPen(self._plot_text))
            axis.setPen(pg.mkPen(self._plot_border))
            axis.setTickPen(pg.mkPen(self._plot_border))
        # _retranslate rebuilds the title HTML (uses _plot_text) and the
        # bottom axis label via _configure_index_axis.
        self._retranslate()
        # Refresh the loading/empty overlay so it picks up the new
        # palette colors too.
        self._apply_overlay_text()

    def update_simulation_voltages(
        self,
        ground_truth: np.ndarray,
        reconstructed: np.ndarray | None = None,
    ) -> None:
        """Update the simulation-oriented voltage comparison plot."""
        ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1)
        reconstructed_arr = self._coerce_reconstructed_overlay(
            reconstructed,
            expected_size=len(ground_truth),
        )
        x = np.arange(1, len(ground_truth) + 1, dtype=np.float64)
        self._configure_index_axis(len(ground_truth))
        self._has_data = True
        self._overlay_state = "data"
        self._empty_overlay.hide()
        self._curve_primary.setData(x, ground_truth)
        self._curve_primary.setVisible(True)
        self._set_reconstructed_overlay(
            x, reconstructed_arr, expected_size=len(ground_truth)
        )
        self._apply_y_range(ground_truth, reconstructed_arr)

    def update_voltages(
        self,
        simulated: np.ndarray,
        reconstructed: np.ndarray | None = None,
        homogeneous: np.ndarray | None = None,
    ) -> None:
        """Backward-compatible wrapper for legacy simulation call sites."""
        _ = homogeneous
        self.update_simulation_voltages(simulated, reconstructed)

    def update_hardware_voltages(
        self,
        measured: np.ndarray,
        reconstructed: np.ndarray | None = None,
    ) -> None:
        """Update the hardware-oriented voltage fit plot."""
        measured = np.asarray(measured, dtype=np.float64).reshape(-1)
        reconstructed_arr = self._coerce_reconstructed_overlay(
            reconstructed,
            expected_size=len(measured),
        )
        x = np.arange(1, len(measured) + 1, dtype=np.float64)
        self._configure_index_axis(len(measured))
        self._has_data = True
        self._overlay_state = "data"
        self._empty_overlay.hide()
        self._curve_primary.setData(x, measured)
        self._curve_primary.setVisible(True)
        self._set_reconstructed_overlay(
            x,
            reconstructed_arr,
            expected_size=len(measured),
        )
        self._apply_y_range(measured, reconstructed_arr)

    def clear(self) -> None:
        """Clear all curves."""
        empty = np.array([])
        self._curve_primary.setData(empty, empty)
        self._curve_reconstructed_outline.setData(empty, empty)
        self._curve_reconstructed.setData(empty, empty)
        self._curve_reconstructed_markers.setData(empty, empty)
        self._curve_reconstructed_outline.setVisible(False)
        self._curve_reconstructed.setVisible(False)
        self._curve_reconstructed_markers.setVisible(False)
        self._has_data = False
        self._overlay_state = "empty"
        self._apply_overlay_text()
        self._empty_overlay.show()
        self._configure_index_axis(self._expected_point_count)

    def set_loading(self, on: bool) -> None:
        """Toggle the 'computing voltages' overlay.

        Used during forward solve (simulation) or reconstruction
        (hardware) so the plot communicates that fresh data is coming.
        If on=False and no data arrived, fall back to the empty hint.
        """
        if on:
            self._overlay_state = "loading"
            self._apply_overlay_text()
            self._empty_overlay.show()
        elif not self._has_data:
            self._overlay_state = "empty"
            self._apply_overlay_text()
            self._empty_overlay.show()
        else:
            self._overlay_state = "data"
            self._empty_overlay.hide()

    def _apply_overlay_text(self) -> None:
        """Render the overlay label for the current state + language.

        Loading state uses the theme-aware *scrim* stylesheet — a full
        panel-bg fill that covers any stale curves underneath so the
        "正问题求解中" caption reads as a clean modal state rather than
        floating on top of old data.  Empty state keeps the transparent
        background since there's nothing behind to mask.
        """
        if self._overlay_state == "loading":
            self._empty_overlay.setText(t("voltage_plot.loading_overlay"))
            self._empty_overlay.setStyleSheet(loading_scrim_stylesheet())
        else:
            self._empty_overlay.setText(self._empty_hint())
            self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())

    def _primary_label(self) -> str:
        if self._mode == "hardware":
            return t("hw.boundary.primary.measured")
        return t("hw.boundary.primary.ground_truth")

    def _secondary_label(self) -> str:
        return t("hw.boundary.secondary")

    def _plot_title(self) -> str:
        return t("hw.boundary.title")

    def _empty_hint(self) -> str:
        if self._mode == "hardware":
            return t("hw.boundary.empty.hardware")
        return t("hw.boundary.empty.simulation")

    def legend_labels(self) -> list[str]:
        return [entry.label for entry in self._legend_entries]

    def current_point_count(self) -> int:
        return self._point_count

    def set_expected_point_count(self, point_count: int) -> None:
        self._expected_point_count = max(int(point_count), 1)
        if not self._has_data:
            self._configure_index_axis(self._expected_point_count)

    def _build_legend_entries(self) -> list[LegendEntry]:
        return [
            LegendEntry("primary", self._primary_label(), "#4ecdc4", 2.0),
            LegendEntry(
                "fit",
                self._secondary_label(),
                "#ff6b6b",
                1.8,
                style=Qt.PenStyle.DashLine,
            ),
        ]

    def _set_reconstructed_overlay(
        self,
        x: np.ndarray,
        reconstructed: np.ndarray | None,
        *,
        expected_size: int,
    ) -> None:
        if reconstructed is None:
            self._hide_reconstructed_overlay()
            return
        reconstructed_arr = self._coerce_reconstructed_overlay(
            reconstructed,
            expected_size=expected_size,
        )
        if reconstructed_arr is None:
            self._hide_reconstructed_overlay()
            return

        self._curve_reconstructed_outline.setData(x, reconstructed_arr)
        self._curve_reconstructed_outline.setVisible(True)
        self._curve_reconstructed.setData(x, reconstructed_arr)
        self._curve_reconstructed.setVisible(True)

        marker_step = max(1, int(np.ceil(reconstructed_arr.size / 18.0)))
        marker_idx = np.arange(0, reconstructed_arr.size, marker_step, dtype=np.int32)
        if marker_idx.size == 0 or marker_idx[-1] != reconstructed_arr.size - 1:
            marker_idx = np.append(marker_idx, reconstructed_arr.size - 1)
        self._curve_reconstructed_markers.setData(
            x[marker_idx], reconstructed_arr[marker_idx]
        )
        self._curve_reconstructed_markers.setVisible(True)

    def _hide_reconstructed_overlay(self) -> None:
        self._curve_reconstructed_outline.setVisible(False)
        self._curve_reconstructed.setVisible(False)
        self._curve_reconstructed_markers.setVisible(False)

    @staticmethod
    def _coerce_reconstructed_overlay(
        reconstructed: np.ndarray | None,
        *,
        expected_size: int,
    ) -> np.ndarray | None:
        if reconstructed is None:
            return None
        try:
            reconstructed_arr = np.asarray(reconstructed, dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if reconstructed_arr.size != int(expected_size):
            return None
        return reconstructed_arr

    def _apply_y_range(self, *series: np.ndarray | None) -> None:
        finite_parts: list[np.ndarray] = []
        for values in series:
            if values is None:
                continue
            arr = np.asarray(values, dtype=np.float64).reshape(-1)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                finite_parts.append(finite)
        if not finite_parts:
            return
        merged = np.concatenate(finite_parts)
        y_min = float(np.min(merged))
        y_max = float(np.max(merged))
        span = y_max - y_min
        if not np.isfinite(span):
            return
        if span <= 0.0:
            padding = max(abs(y_min) * 0.08, 1.0e-12)
        else:
            padding = max(span * 0.08, 1.0e-12)
        self._plot_widget.setYRange(y_min - padding, y_max + padding, padding=0.0)

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh title, axis labels, empty-overlay, and legend labels."""
        self._plot_widget.setLabel(
            "left", t("hw.boundary.y_label"), **self._label_style
        )
        self._plot_widget.setTitle(
            f'<span style="color:{self._plot_text};'
            f"font-family:'{self._serif}';font-size:13pt;\">"
            f"{self._plot_title()}"
            "</span>"
        )
        # Overlay text follows the current state (empty or loading).
        self._apply_overlay_text()
        # Push new legend labels without rebuilding the widget.
        self._legend_frame.update_labels(
            {"primary": self._primary_label(), "fit": self._secondary_label()}
        )
        # Bottom axis label is dynamic (depends on point count) — reapply.
        self._configure_index_axis(self._point_count)

    def _configure_index_axis(self, point_count: int) -> None:
        count = max(int(point_count), 1)
        self._point_count = count
        self._plot_widget.setLabel(
            "bottom",
            t("hw.boundary.x_label_dynamic", count=count),
            **self._label_style,
        )
        padding = 0.02 if count > 1 else 0.4
        view_box = self._plot_widget.getPlotItem().getViewBox()
        try:
            view_box.enableAutoRange(x=False)
        except Exception:
            pass
        try:
            view_box.setLimits(xMin=1, xMax=count)
            view_box.setRange(xRange=(1, count), padding=padding, disableAutoRange=True)
        except Exception:
            self._plot_widget.setXRange(1, count, padding=padding)
        axis = self._plot_widget.getPlotItem().getAxis("bottom")
        axis.setTicks([self._major_ticks(count), []])

    @staticmethod
    def _major_ticks(count: int) -> list[tuple[int, str]]:
        if count <= 1:
            return [(1, "1")]
        candidates = [
            1,
            round(count * 0.25),
            round(count * 0.5),
            round(count * 0.75),
            count,
        ]
        ticks: list[tuple[int, str]] = []
        seen: set[int] = set()
        for value in candidates:
            clamped = min(max(int(value), 1), count)
            if clamped in seen:
                continue
            seen.add(clamped)
            ticks.append((clamped, str(clamped)))
        return ticks
