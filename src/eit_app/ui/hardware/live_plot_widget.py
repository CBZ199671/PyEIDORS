"""Real-time measurement line plot using pyqtgraph."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

import pyqtgraph as pg

from eit_app.ui.fonts import serif_font_family
from eit_app.ui.plot_legend_overlay import LegendEntry, PlotLegendOverlay


class LivePlotWidget(QWidget):
    """Displays measurement points as a live-updating line chart.

    Supports overlaying real and imaginary channels.
    Uses pyqtgraph for GPU-accelerated rendering at 30fps.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_frame = None
        self._serif_family = serif_font_family()
        self._plot_bg = "#f8fbfe"
        self._plot_text = "#243447"
        self._plot_grid = "#d6e1ec"
        self._plot_border = "#c7d4e2"
        self._point_count = 208
        self._expected_point_count = 208
        self._has_data = False

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
        self._plot_widget.setLabel("left", "Voltage (V)", **label_style)
        self._plot_widget.setTitle(
            f"<span style=\"color:{self._plot_text};font-family:'{self._serif_family}';font-size:14pt;\">"
            "Live Measurement Channels"
            "</span>"
        )
        self._plot_widget.showGrid(x=True, y=True, alpha=0.55)

        plot_host = QWidget()
        plot_stack = QStackedLayout(plot_host)
        plot_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        plot_stack.setContentsMargins(0, 0, 0, 0)
        plot_stack.addWidget(self._plot_widget)
        self._empty_overlay = QLabel(
            "No live frames yet.\nStart acquisition to display Real and Imag."
        )
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

        # Data curves
        self._curve_real = self._plot_widget.plot(
            pen=pg.mkPen("#f4d35e", width=2.2), name="Real"
        )
        self._curve_imag = self._plot_widget.plot(
            pen=pg.mkPen("#4ecdc4", width=1.9), name="Imag"
        )
        self._legend_frame = PlotLegendOverlay(
            [
                LegendEntry("real", "Real", "#f4d35e", 2.2, checked=True),
                LegendEntry("imag", "Imag", "#4ecdc4", 1.9, checked=False),
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
        self._configure_index_axis(self._point_count, label_prefix="Measurement Index")

        # Connect checkbox toggles
        self._show_real.toggled.connect(self._on_visibility_changed)
        self._show_imag.toggled.connect(self._on_visibility_changed)

        # Initial visibility
        self._curve_imag.setVisible(False)

    @Slot(object)
    def update_frame(self, frame) -> None:
        """Update plot with a new FrameData.

        Args:
            frame: FrameData instance with .real and .imag arrays.
        """
        self._last_frame = frame
        self._has_data = True
        self._empty_overlay.hide()
        self._refresh_curves(frame)

    def _on_visibility_changed(self, _checked: bool) -> None:
        self._curve_real.setVisible(self._show_real.isChecked())
        self._curve_imag.setVisible(self._show_imag.isChecked())
        if self._last_frame is not None:
            self._refresh_curves(self._last_frame)

    def _refresh_curves(self, frame) -> None:
        n = len(frame.real)
        if n != len(self._x):
            self._x = np.arange(1, n + 1, dtype=np.float64)
        self._configure_index_axis(n, label_prefix="Measurement Index")

        self._curve_real.setData(self._x, frame.real)
        self._curve_imag.setData(self._x, frame.imag)

        self._curve_real.setVisible(self._show_real.isChecked())
        self._curve_imag.setVisible(self._show_imag.isChecked())

    def clear(self) -> None:
        """Clear all curves."""
        self._last_frame = None
        self._has_data = False
        empty = np.array([])
        self._curve_real.setData(empty, empty)
        self._curve_imag.setData(empty, empty)
        self._configure_index_axis(self._expected_point_count, label_prefix="Measurement Index")
        self._empty_overlay.show()

    def set_expected_point_count(self, point_count: int) -> None:
        self._expected_point_count = max(int(point_count), 1)
        if not self._has_data:
            self._configure_index_axis(self._expected_point_count, label_prefix="Measurement Index")

    def current_point_count(self) -> int:
        return self._point_count

    def _configure_index_axis(self, point_count: int, *, label_prefix: str) -> None:
        count = max(int(point_count), 1)
        self._point_count = count
        self._plot_widget.setLabel(
            "bottom",
            f"{label_prefix} (1-{count})",
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
        candidates = [1, round(count * 0.25), round(count * 0.5), round(count * 0.75), count]
        ticks: list[tuple[int, str]] = []
        seen: set[int] = set()
        for value in candidates:
            clamped = min(max(int(value), 1), count)
            if clamped in seen:
                continue
            seen.add(clamped)
            ticks.append((clamped, str(clamped)))
        return ticks
