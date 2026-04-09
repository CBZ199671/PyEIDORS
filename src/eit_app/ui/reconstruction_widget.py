"""Matplotlib-based conductivity reconstruction display widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from PySide6.QtCore import Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget

from eit_app.ui.fonts import serif_font_family

matplotlib.use("QtAgg")

if TYPE_CHECKING:
    from eit_app.controllers.reconstruction_controller import ReconstructionResult


class ReconstructionWidget(QWidget):
    """Displays EIT conductivity reconstruction using matplotlib tripcolor.

    Embeds a matplotlib FigureCanvas in a Qt widget. Receives
    ReconstructionResult objects and renders the conductivity map.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._serif_family = serif_font_family()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._figure.patch.set_facecolor("#f4f7fb")
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)

        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("#fbfdff")
        self._ax.set_title("Reconstruction", fontname=self._serif_family, fontsize=15)
        self._ax.set_aspect("equal")
        self._colorbar = None

        # Show placeholder
        self._ax.text(
            0.5,
            0.5,
            "No reconstruction yet",
            transform=self._ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#5b6573",
            fontname=self._serif_family,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw()

    @Slot(object)
    def update_reconstruction(self, result: ReconstructionResult) -> None:
        """Render a new reconstruction result."""
        if result.error_msg or result.conductivity.size == 0:
            self._show_error(result.error_msg or "Empty result")
            return

        self._ax.clear()

        coords = result.node_coords
        cells = result.cell_connectivity
        sigma = result.conductivity

        if coords.ndim != 2 or coords.shape[1] < 2:
            self._show_error("Invalid mesh coordinates")
            return

        x = coords[:, 0]
        y = coords[:, 1]

        try:
            tri = Triangulation(x, y, cells)
        except Exception as exc:
            self._show_error(f"Triangulation failed: {exc}")
            return

        # Cell-centered data: one value per triangle
        if len(sigma) == len(cells):
            tpc = self._ax.tripcolor(tri, sigma, shading="flat", cmap="viridis")
        # Node-centered data: one value per vertex
        elif len(sigma) == len(x):
            tpc = self._ax.tripcolor(tri, sigma, shading="gouraud", cmap="viridis")
        else:
            self._show_error(
                f"Conductivity size ({len(sigma)}) doesn't match "
                f"cells ({len(cells)}) or nodes ({len(x)})"
            )
            return

        self._ax.set_title("Conductivity Reconstruction", fontname=self._serif_family, fontsize=15)
        self._ax.set_aspect("equal")
        self._ax.tick_params(labelsize=10)
        for label in self._ax.get_xticklabels() + self._ax.get_yticklabels():
            label.set_fontname(self._serif_family)

        if self._colorbar is not None:
            self._colorbar.remove()
        self._colorbar = self._figure.colorbar(tpc, ax=self._ax, label="S/m")
        self._colorbar.ax.yaxis.label.set_fontname(self._serif_family)
        self._colorbar.ax.yaxis.label.set_size(11)
        for label in self._colorbar.ax.get_yticklabels():
            label.set_fontname(self._serif_family)

        self._canvas.draw()

    def _show_error(self, msg: str) -> None:
        self._ax.clear()
        self._ax.text(
            0.5, 0.5, msg,
            transform=self._ax.transAxes,
            ha="center", va="center",
            fontsize=10, color="#8b2f2f", fontname=self._serif_family,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw()

    def clear(self) -> None:
        self._ax.clear()
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        self._ax.set_facecolor("#fbfdff")
        self._ax.set_title("Reconstruction", fontname=self._serif_family, fontsize=15)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw()
