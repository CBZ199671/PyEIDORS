"""Real-time measurement line plot using pyqtgraph."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

import pyqtgraph as pg

from eit_app.ui.fonts import serif_font_family


class LivePlotWidget(QWidget):
    """Displays 208 measurement points as a live-updating line chart.

    Supports overlaying real, imaginary, and magnitude channels.
    Uses pyqtgraph for GPU-accelerated rendering at 30fps.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_frame = None
        self._serif_family = serif_font_family()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        self._show_real = QCheckBox("Real")
        self._show_real.setChecked(True)
        self._show_imag = QCheckBox("Imag")
        self._show_mag = QCheckBox("Magnitude")
        self._show_mag.setChecked(True)
        toolbar.addWidget(self._show_real)
        toolbar.addWidget(self._show_imag)
        toolbar.addWidget(self._show_mag)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#0d1321")
        label_style = {"color": "#dbe4f0", "font-family": self._serif_family, "font-size": "12pt"}
        self._plot_widget.setLabel("bottom", "Measurement Index", **label_style)
        self._plot_widget.setLabel("left", "Voltage (V)", **label_style)
        self._plot_widget.setTitle(
            f"<span style=\"color:#f5f7fa;font-family:'{self._serif_family}';font-size:14pt;\">"
            "Live Measurement Channels"
            "</span>"
        )
        self._legend = self._plot_widget.addLegend(offset=(12, 12), labelTextColor="#dbe4f0")
        self._legend.setBrush(pg.mkBrush(10, 18, 32, 180))
        self._legend.setPen(pg.mkPen("#2f415e"))
        self._plot_widget.showGrid(x=True, y=True, alpha=0.18)
        layout.addWidget(self._plot_widget)

        tick_font = QFont(self._serif_family, 10)
        for axis_name in ("left", "bottom"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setTextPen(pg.mkPen("#dbe4f0"))
            axis.setPen(pg.mkPen("#52607a"))
            axis.setTickPen(pg.mkPen("#52607a"))
            axis.setStyle(tickFont=tick_font)

        # Data curves
        self._curve_real = self._plot_widget.plot(
            pen=pg.mkPen("#f4d35e", width=2.2), name="Real"
        )
        self._curve_imag = self._plot_widget.plot(
            pen=pg.mkPen("#4ecdc4", width=1.9), name="Imag"
        )
        self._curve_mag = self._plot_widget.plot(
            pen=pg.mkPen("#ff6b6b", width=2.0, style=Qt.PenStyle.DashLine), name="Magnitude"
        )
        self._apply_legend_font()

        # X-axis (measurement indices)
        self._x = np.arange(208, dtype=np.float64)

        # Connect checkbox toggles
        self._show_real.toggled.connect(self._on_visibility_changed)
        self._show_imag.toggled.connect(self._on_visibility_changed)
        self._show_mag.toggled.connect(self._on_visibility_changed)

        # Initial visibility
        self._curve_imag.setVisible(False)
        self._curve_mag.setVisible(True)

    @Slot(object)
    def update_frame(self, frame) -> None:
        """Update plot with a new FrameData.

        Args:
            frame: FrameData instance with .real and .imag arrays.
        """
        self._last_frame = frame
        self._refresh_curves(frame)

    def _on_visibility_changed(self, _checked: bool) -> None:
        self._curve_real.setVisible(self._show_real.isChecked())
        self._curve_imag.setVisible(self._show_imag.isChecked())
        self._curve_mag.setVisible(self._show_mag.isChecked())
        if self._last_frame is not None:
            self._refresh_curves(self._last_frame)

    def _refresh_curves(self, frame) -> None:
        n = len(frame.real)
        if n != len(self._x):
            self._x = np.arange(n, dtype=np.float64)

        self._curve_real.setData(self._x, frame.real)
        self._curve_imag.setData(self._x, frame.imag)
        mag = np.abs(frame.real + 1j * frame.imag)
        self._curve_mag.setData(self._x, mag)

        self._curve_real.setVisible(self._show_real.isChecked())
        self._curve_imag.setVisible(self._show_imag.isChecked())
        self._curve_mag.setVisible(self._show_mag.isChecked())

    def clear(self) -> None:
        """Clear all curves."""
        self._last_frame = None
        empty = np.array([])
        self._curve_real.setData(empty, empty)
        self._curve_imag.setData(empty, empty)
        self._curve_mag.setData(empty, empty)

    def _apply_legend_font(self) -> None:
        legend_font = QFont(self._serif_family, 10)
        for _sample, label in getattr(self._legend, "items", []):
            try:
                label.item.setFont(legend_font)
            except Exception:
                continue
