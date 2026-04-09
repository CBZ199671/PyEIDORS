"""pyqtgraph-based boundary voltage comparison plot."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QVBoxLayout, QWidget

import pyqtgraph as pg

from eit_app.ui.fonts import serif_font_family


class BoundaryVoltagePlotWidget(QWidget):
    """Displays boundary voltages with optional overlay for comparison."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._serif = serif_font_family()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("#0d1321")

        label_style = {
            "color": "#dbe4f0",
            "font-family": self._serif,
            "font-size": "11pt",
        }
        self._plot_widget.setLabel("bottom", "Measurement Index", **label_style)
        self._plot_widget.setLabel("left", "Voltage (V)", **label_style)
        self._plot_widget.setTitle(
            f'<span style="color:#f5f7fa;font-family:\'{self._serif}\';font-size:13pt;">'
            "Boundary Voltages"
            "</span>"
        )

        self._legend = self._plot_widget.addLegend(offset=(12, 12), labelTextColor="#dbe4f0")
        self._legend.setBrush(pg.mkBrush(10, 18, 32, 180))
        self._legend.setPen(pg.mkPen("#2f415e"))
        self._plot_widget.showGrid(x=True, y=True, alpha=0.18)

        tick_font = QFont(self._serif, 9)
        for axis_name in ("left", "bottom"):
            axis = self._plot_widget.getPlotItem().getAxis(axis_name)
            axis.setTextPen(pg.mkPen("#dbe4f0"))
            axis.setPen(pg.mkPen("#52607a"))
            axis.setTickPen(pg.mkPen("#52607a"))
            axis.setStyle(tickFont=tick_font)

        # Data curves
        self._curve_simulated = self._plot_widget.plot(
            pen=pg.mkPen("#4ecdc4", width=2.0), name="Simulated"
        )
        self._curve_reconstructed = self._plot_widget.plot(
            pen=pg.mkPen("#ff6b6b", width=1.8, style=Qt.PenStyle.DashLine),
            name="Reconstructed",
        )
        self._curve_homogeneous = self._plot_widget.plot(
            pen=pg.mkPen("#f4d35e", width=1.5, style=Qt.PenStyle.DotLine),
            name="Homogeneous",
        )

        layout.addWidget(self._plot_widget)

    def update_voltages(
        self,
        simulated: np.ndarray,
        reconstructed: np.ndarray | None = None,
        homogeneous: np.ndarray | None = None,
    ) -> None:
        """Update the voltage comparison plot."""
        x = np.arange(len(simulated), dtype=np.float64)
        self._curve_simulated.setData(x, simulated)
        self._curve_simulated.setVisible(True)

        if reconstructed is not None and len(reconstructed) == len(simulated):
            self._curve_reconstructed.setData(x, reconstructed)
            self._curve_reconstructed.setVisible(True)
        else:
            self._curve_reconstructed.setVisible(False)

        if homogeneous is not None and len(homogeneous) == len(simulated):
            self._curve_homogeneous.setData(x, homogeneous)
            self._curve_homogeneous.setVisible(True)
        else:
            self._curve_homogeneous.setVisible(False)

    def clear(self) -> None:
        """Clear all curves."""
        empty = np.array([])
        self._curve_simulated.setData(empty, empty)
        self._curve_reconstructed.setData(empty, empty)
        self._curve_homogeneous.setData(empty, empty)
