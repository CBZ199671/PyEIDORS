"""Reusable matplotlib tripcolor conductivity display."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.tri import Triangulation
from PySide6.QtWidgets import QVBoxLayout, QWidget

from eit_app.ui.fonts import plot_font_families, serif_font_family


class ConductivityImageWidget(QWidget):
    """Displays a conductivity distribution as a matplotlib tripcolor plot."""

    def __init__(self, title: str = "Conductivity", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._serif = serif_font_family()
        # Title uses a FontProperties with a Latin-serif-first family list
        # so matplotlib's per-glyph fallback can reach CJK faces when the
        # title is translated to Chinese.  Without this fallback Times New
        # Roman emits "Glyph X missing" warnings and renders tofu boxes.
        self._title_font = FontProperties(family=plot_font_families(), size=14)
        self._default_title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._figure.patch.set_facecolor("#f4f7fb")
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)

        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("#fbfdff")
        self._ax.set_title(title, fontproperties=self._title_font)
        self._ax.set_aspect("equal")
        self._colorbar = None
        self._show_placeholder()

    def setTitle(self, title: str) -> None:
        """Update the plot title (used by i18n retranslate pipelines)."""
        self._default_title = title
        self._ax.set_title(title, fontproperties=self._title_font)
        self._canvas.draw_idle()

    def update_image(
        self,
        conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Render a conductivity distribution on the mesh."""
        self._remove_colorbar()
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
        self._ax.set_title(display_title, fontproperties=self._title_font)
        self._ax.set_aspect("equal")
        self._ax.tick_params(labelsize=9)
        # Tick labels stay Latin (numbers only) — safe to keep Times New Roman.
        for label in self._ax.get_xticklabels() + self._ax.get_yticklabels():
            label.set_fontname(self._serif)

        # shrink + aspect + pad keep the colorbar from dominating the
        # plot height.  shrink=0.72 trims ~30% off its length, aspect=16
        # keeps it slim, pad=0.04 pulls it closer to the image so the
        # matplotlib auto-layout does not leave a huge right-hand gap.
        self._colorbar = self._figure.colorbar(
            tpc, ax=self._ax, label="S/m",
            shrink=0.72, aspect=16, pad=0.04,
        )
        self._colorbar.ax.yaxis.label.set_fontname(self._serif)
        self._colorbar.ax.yaxis.label.set_size(10)
        # Slightly smaller tick labels so the numbers don't compete with
        # the main title for visual weight.
        self._colorbar.ax.tick_params(labelsize=8)

        self._canvas.draw()

    def clear(self) -> None:
        """Reset to placeholder state."""
        self._remove_colorbar()
        self._ax.clear()
        self._ax.set_facecolor("#fbfdff")
        self._ax.set_title(self._default_title, fontproperties=self._title_font)
        self._show_placeholder()

    def _remove_colorbar(self) -> None:
        """Remove the existing colorbar even if matplotlib has orphaned it.

        Matplotlib can leave a colorbar with a partially detached axes after
        repeated Qt redraws / layout changes.  Calling ``Colorbar.remove()``
        in that state raises internally, so make removal idempotent and fall
        back to removing the colorbar axes directly.
        """
        colorbar = self._colorbar
        self._colorbar = None
        if colorbar is None:
            return

        cax = getattr(colorbar, "ax", None)
        try:
            colorbar.remove()
            return
        except (AttributeError, KeyError, RuntimeError, ValueError):
            pass

        if cax is None:
            return
        try:
            if cax in self._figure.axes:
                self._figure.delaxes(cax)
            else:
                cax.remove()
        except (AttributeError, KeyError, RuntimeError, ValueError):
            # Best effort only: a failed cleanup should never prevent the new
            # reconstruction image from being displayed.
            pass

    def _show_placeholder(self) -> None:
        self._ax.text(
            0.5, 0.5, "No data",
            transform=self._ax.transAxes,
            ha="center", va="center",
            fontsize=11, color="#5b6573", fontproperties=self._title_font,
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
            fontsize=10, color="#8b2f2f", fontproperties=self._title_font,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw()
