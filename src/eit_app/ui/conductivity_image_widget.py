"""Reusable matplotlib tripcolor conductivity display."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from PySide6.QtWidgets import QVBoxLayout, QWidget

from eit_app.ui.fonts import serif_font_family


class ConductivityImageWidget(QWidget):
    """Displays a conductivity distribution as a matplotlib tripcolor plot."""

    def __init__(self, title: str = "Conductivity", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._serif = serif_font_family()
        self._default_title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._figure.patch.set_facecolor("#f4f7fb")
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)

        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("#fbfdff")
        self._ax.set_title(title, fontname=self._serif, fontsize=14)
        self._ax.set_aspect("equal")
        self._colorbar = None
        self._show_placeholder()

    def update_image(
        self,
        conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Render a conductivity distribution on the mesh."""
        self._ax.clear()

        if node_coords.ndim != 2 or node_coords.shape[1] < 2:
            self._show_error("Invalid mesh coordinates")
            return

        x = node_coords[:, 0]
        y = node_coords[:, 1]

        try:
            tri = Triangulation(x, y, cell_connectivity)
        except Exception as exc:
            self._show_error(f"Triangulation failed: {exc}")
            return

        if len(conductivity) == len(cell_connectivity):
            tpc = self._ax.tripcolor(tri, conductivity, shading="flat", cmap="viridis")
        elif len(conductivity) == len(x):
            tpc = self._ax.tripcolor(tri, conductivity, shading="gouraud", cmap="viridis")
        else:
            self._show_error(
                f"Size mismatch: sigma={len(conductivity)}, "
                f"cells={len(cell_connectivity)}, nodes={len(x)}"
            )
            return

        display_title = title or self._default_title
        self._ax.set_title(display_title, fontname=self._serif, fontsize=14)
        self._ax.set_aspect("equal")
        self._ax.tick_params(labelsize=9)
        for label in self._ax.get_xticklabels() + self._ax.get_yticklabels():
            label.set_fontname(self._serif)

        if self._colorbar is not None:
            self._colorbar.remove()
        self._colorbar = self._figure.colorbar(tpc, ax=self._ax, label="S/m")
        self._colorbar.ax.yaxis.label.set_fontname(self._serif)
        self._colorbar.ax.yaxis.label.set_size(10)

        self._canvas.draw()

    def clear(self) -> None:
        """Reset to placeholder state."""
        self._ax.clear()
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        self._ax.set_facecolor("#fbfdff")
        self._ax.set_title(self._default_title, fontname=self._serif, fontsize=14)
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self._ax.text(
            0.5, 0.5, "No data",
            transform=self._ax.transAxes,
            ha="center", va="center",
            fontsize=11, color="#5b6573", fontname=self._serif,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw()

    def _show_error(self, msg: str) -> None:
        self._ax.clear()
        self._ax.text(
            0.5, 0.5, msg,
            transform=self._ax.transAxes,
            ha="center", va="center",
            fontsize=10, color="#8b2f2f", fontname=self._serif,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw()
