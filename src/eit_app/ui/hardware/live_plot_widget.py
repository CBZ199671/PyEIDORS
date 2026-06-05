"""Real-time measurement line plot using pyqtgraph."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
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


class LivePlotWidget(QWidget):
    """Displays measurement points as a live-updating line chart.

    Supports overlaying real and imaginary channels.
    Uses pyqtgraph for GPU-accelerated rendering at 30fps.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_frame = None
        self._serif_family = serif_font_family()
        # Pull plot-canvas colors from the theme palette so the widget
        # picks up dark mode automatically.  subscribe_theme_mode at
        # the end of __init__ wires re-paint on later mode flips.
        palette = plot_palette()
        self._plot_bg = palette["bg"]
        self._plot_text = palette["text"]
        self._plot_grid = palette["grid"]
        self._plot_border = palette["border"]
        self._point_count = 208
        self._expected_point_count = 208
        self._has_data = False
        # Tri-state for the placeholder overlay: "empty" | "loading" |
        # "data".  _retranslate() re-applies the correct language for
        # the current state instead of always showing the empty text.
        self._overlay_state = "empty"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(self._plot_bg)
        label_style = {
            "color": self._plot_text,
            "font-family": self._serif_family,
            "font-size": "12pt",
        }
        # Axis label + title are assigned by _retranslate() so they follow
        # the active UI language.  Keep handles around for that call.
        self._axis_label_style = label_style
        self._plot_widget.showGrid(x=True, y=True, alpha=0.55)

        plot_host = QWidget()
        plot_stack = QStackedLayout(plot_host)
        plot_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        plot_stack.setContentsMargins(0, 0, 0, 0)
        plot_stack.addWidget(self._plot_widget)
        # Overlay text is assigned by _retranslate() below.
        self._empty_overlay = QLabel("")
        self._empty_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_overlay.setStyleSheet(
            "color: #5b6573; "
            "font-size: 12px; "
            "font-weight: 600; "
            "background: transparent;"
        )
        plot_stack.addWidget(self._empty_overlay)
        layout.addWidget(plot_host)

        tick_font = QFont(self._serif_family, 10)
        for axis_name in ("left", "bottom"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setTextPen(pg.mkPen(self._plot_text))
            axis.setPen(pg.mkPen(self._plot_border))
            axis.setTickPen(pg.mkPen(self._plot_border))
            axis.setStyle(tickFont=tick_font)

        # Data curves — localized display names applied in _retranslate()
        self._curve_real = self._plot_widget.plot(pen=pg.mkPen("#f4d35e", width=2.2))
        self._curve_real_markers = self._plot_widget.plot(
            pen=None,
            symbol="o",
            symbolSize=4,
            symbolBrush=pg.mkBrush("#f4d35e"),
            symbolPen=pg.mkPen("#ffffff", width=0.8),
        )
        self._curve_imag = self._plot_widget.plot(pen=pg.mkPen("#4ecdc4", width=1.9))
        self._curve_imag_markers = self._plot_widget.plot(
            pen=None,
            symbol="o",
            symbolSize=4,
            symbolBrush=pg.mkBrush("#4ecdc4"),
            symbolPen=pg.mkPen("#ffffff", width=0.8),
        )
        self._curve_real.setZValue(3)
        self._curve_real_markers.setZValue(4)
        self._curve_imag.setZValue(5)
        self._curve_imag_markers.setZValue(6)
        self._legend_frame = PlotLegendOverlay(
            [
                LegendEntry(
                    "real", t("hw.live_plot.curve.real"), "#f4d35e", 2.2, checked=True
                ),
                LegendEntry(
                    "imag", t("hw.live_plot.curve.imag"), "#4ecdc4", 1.9, checked=False
                ),
            ],
            interactive=True,
            compact=False,
            parent=plot_host,
        )
        self._legend_frame.move(66, 56)
        self._legend_frame.raise_()
        self._show_real = self._legend_frame.button("real")
        self._show_imag = self._legend_frame.button("imag")

        # X-axis (measurement indices)
        self._x = np.arange(1, self._point_count + 1, dtype=np.float64)
        self._configure_index_axis(self._point_count)

        # Connect checkbox toggles
        self._show_real.toggled.connect(self._on_visibility_changed)
        self._show_imag.toggled.connect(self._on_visibility_changed)

        # Initial visibility
        self._curve_imag.setVisible(False)
        self._curve_imag_markers.setVisible(False)

        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        # Re-paint canvas + axis colors when the user toggles dark mode.
        subscribe_theme_mode(self._on_theme_mode_changed)

    def _on_theme_mode_changed(self, _mode: str) -> None:
        """Re-pull the plot palette and re-paint canvas + axis pens.

        pyqtgraph caches the canvas brush + axis pens internally — they
        don't honour QSS — so we have to push the new colors explicitly.
        """
        palette = plot_palette()
        self._plot_bg = palette["bg"]
        self._plot_text = palette["text"]
        self._plot_grid = palette["grid"]
        self._plot_border = palette["border"]
        self._plot_widget.setBackground(self._plot_bg)
        self._axis_label_style["color"] = self._plot_text
        for axis_name in ("left", "bottom"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setTextPen(pg.mkPen(self._plot_text))
            axis.setPen(pg.mkPen(self._plot_border))
            axis.setTickPen(pg.mkPen(self._plot_border))
        # Re-render the title HTML (uses _plot_text) and the bottom-axis
        # dynamic label (uses _plot_text via _configure_index_axis).
        self._retranslate()

    @Slot(object)
    def update_frame(self, frame) -> None:
        """Update plot with a new FrameData.

        Args:
            frame: FrameData instance with .real and .imag arrays.
        """
        self._last_frame = frame
        self._has_data = True
        self._overlay_state = "data"
        self._empty_overlay.hide()
        self._refresh_curves(frame)

    def set_loading(self, on: bool) -> None:
        """Toggle the 'waiting for frames' overlay.

        Called when hardware connects / acquisition starts but no frame
        has arrived yet — conveys "device is active, data is coming".
        If on=False and no data yet, fall back to the empty state.
        """
        if on:
            self._overlay_state = "loading"
            self._apply_overlay_text()
            self._empty_overlay.show()
        else:
            if self._has_data:
                self._empty_overlay.hide()
                self._overlay_state = "data"
            else:
                self._overlay_state = "empty"
                self._apply_overlay_text()
                self._empty_overlay.show()

    def _apply_overlay_text(self) -> None:
        """Re-apply the overlay text for the current state + language.

        Loading state uses a full-panel scrim (theme.loading_scrim_*)
        so any previous frames underneath the overlay are hidden
        cleanly instead of bleeding through under the caption text.
        Empty state stays transparent — there's no prior content to
        mask.
        """
        if self._overlay_state == "loading":
            self._empty_overlay.setText(t("hw.live_plot.loading_overlay"))
            self._empty_overlay.setStyleSheet(loading_scrim_stylesheet())
        else:
            self._empty_overlay.setText(t("hw.live_plot.empty_overlay"))
            self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())

    def _on_visibility_changed(self, _checked: bool) -> None:
        show_real = self._show_real.isChecked()
        show_imag = self._show_imag.isChecked()
        self._curve_real.setVisible(show_real)
        self._curve_real_markers.setVisible(show_real)
        self._curve_imag.setVisible(show_imag)
        self._curve_imag_markers.setVisible(show_imag)
        if self._last_frame is not None:
            self._refresh_curves(self._last_frame)

    def _refresh_curves(self, frame) -> None:
        n = len(frame.real)
        if n != len(self._x):
            self._x = np.arange(1, n + 1, dtype=np.float64)
        self._configure_index_axis(n)

        self._curve_real.setData(self._x, frame.real)
        self._curve_real_markers.setData(self._x, frame.real)
        self._curve_imag.setData(self._x, frame.imag)
        self._curve_imag_markers.setData(self._x, frame.imag)

        show_real = self._show_real.isChecked()
        show_imag = self._show_imag.isChecked()
        self._curve_real.setVisible(show_real)
        self._curve_real_markers.setVisible(show_real)
        self._curve_imag.setVisible(show_imag)
        self._curve_imag_markers.setVisible(show_imag)

    def clear(self) -> None:
        """Clear all curves."""
        self._last_frame = None
        self._has_data = False
        self._overlay_state = "empty"
        empty = np.array([])
        self._curve_real.setData(empty, empty)
        self._curve_real_markers.setData(empty, empty)
        self._curve_imag.setData(empty, empty)
        self._curve_imag_markers.setData(empty, empty)
        self._curve_real_markers.setVisible(False)
        self._curve_imag_markers.setVisible(False)
        self._configure_index_axis(self._expected_point_count)
        self._apply_overlay_text()
        self._empty_overlay.show()

    def set_expected_point_count(self, point_count: int) -> None:
        self._expected_point_count = max(int(point_count), 1)
        if not self._has_data:
            self._configure_index_axis(self._expected_point_count)

    def current_point_count(self) -> int:
        return self._point_count

    # ── i18n ──

    def _retranslate(self) -> None:
        """Refresh title, axis labels, empty-overlay and curve display names."""
        self._plot_widget.setLabel(
            "left", t("hw.live_plot.y_label"), **self._axis_label_style
        )
        self._plot_widget.setTitle(
            f'<span style="color:{self._plot_text};'
            f"font-family:'{self._serif_family}';font-size:14pt;\">"
            f"{t('hw.live_plot.title')}"
            "</span>"
        )
        # Refresh overlay text for the current state (empty or loading).
        self._apply_overlay_text()
        self._show_real.setText(t("hw.live_plot.curve.real"))
        self._show_imag.setText(t("hw.live_plot.curve.imag"))
        # Re-render the bottom-axis label with the localized prefix.
        self._configure_index_axis(self._point_count)

    def _configure_index_axis(self, point_count: int) -> None:
        count = max(int(point_count), 1)
        self._point_count = count
        self._plot_widget.setLabel(
            "bottom",
            t("hw.live_plot.x_label_dynamic", count=count),
            color=self._plot_text,
            **{"font-family": self._serif_family, "font-size": "12pt"},
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
