"""Equipotential / iso-conductivity contour view of a reconstruction.

Pairs with :class:`ReconstructionWidget` on the Hardware tab — that
widget shows the filled conductivity image, this one overlays the
filled image with iso-σ contour lines so the operator can read off
the spatial gradient and the boundaries between regions of similar
conductivity at a glance.

Public API mirrors the reconstruction widget so :mod:`main_window`
can wire it into the same controllers (``update_reconstruction`` /
``set_loading`` / ``clear``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.tri import Triangulation
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.fonts import plot_font_families, serif_font_family
from eit_app.ui.theme import (
    empty_placeholder_stylesheet,
    error_scrim_stylesheet,
    loading_scrim_stylesheet,
    plot_palette,
    subscribe_theme_mode,
)

if TYPE_CHECKING:
    from eit_app.controllers.reconstruction_controller import ReconstructionResult


log = logging.getLogger(__name__)


def _project_tetra_to_triangles(cells: np.ndarray) -> np.ndarray:
    """Reduce a 4-vertex tetra mesh to its 2D boundary triangles.

    The contour drawer only needs a 2D triangulation; a 3D mesh's
    boundary triangulation is the union of every face that appears
    exactly once across all tetrahedra (i.e. the surface).
    """
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 4:
        return np.empty((0, 3), dtype=np.int32)
    faces: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
    offsets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    for cell in cells:
        for idx in offsets:
            face = tuple(int(cell[i]) for i in idx)
            key = tuple(sorted(face))
            faces[key] = None if key in faces else face
    boundary = [face for face in faces.values() if face is not None]
    if not boundary:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(boundary, dtype=np.int32)


class EquipotentialPlotWidget(QWidget):
    """Filled iso-σ contour view for a reconstructed conductivity field."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._serif = serif_font_family()
        self._title_font = FontProperties(family=plot_font_families(), size=13)
        self._last_result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._overlay_mode = "empty"  # "empty" | "loading" | "error"
        self._overlay_message: str | None = None
        self._colorbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        plot_host = QWidget(self)
        plot_stack = QStackedLayout(plot_host)
        plot_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        plot_stack.setContentsMargins(0, 0, 0, 0)

        palette = plot_palette()
        self._figure = Figure(figsize=(4, 4), tight_layout=True)
        self._figure.patch.set_facecolor(palette["panel_bg"])
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor(palette["axes_bg"])
        self._ax.set_aspect("equal")
        plot_stack.addWidget(self._canvas)

        # Phase-4 overlay system: show a centred caption when the widget
        # has no data, is busy, or hit an error.  Identical pattern to
        # the live plot / reconstruction widget so the operator sees a
        # consistent loading / error idiom across the whole tab.
        self._empty_overlay = QLabel("", parent=plot_host)
        self._empty_overlay.setWordWrap(True)
        self._empty_overlay.setMinimumWidth(0)
        self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())
        plot_stack.addWidget(self._empty_overlay)

        layout.addWidget(plot_host)

        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        subscribe_theme_mode(self._on_theme_mode_changed)
        # Render a blank frame so the widget has chrome before the
        # first reconstruction arrives.
        self._show_empty()

    # ------------------------------------------------------------------
    # Public API (mirrors ReconstructionWidget)
    # ------------------------------------------------------------------

    def update_reconstruction(self, result: ReconstructionResult) -> None:
        """Render iso-σ contour lines for a fresh reconstruction."""
        if result.error_msg or result.conductivity.size == 0:
            self._show_status(result.error_msg or "Empty result", error=True)
            return

        coords = np.asarray(result.node_coords, dtype=np.float64)
        cells = np.asarray(result.cell_connectivity, dtype=np.int32)
        sigma = np.asarray(result.conductivity, dtype=np.float64).reshape(-1)

        if coords.ndim != 2 or coords.shape[1] < 2:
            self._show_status("Invalid mesh coordinates", error=True)
            return
        if cells.ndim != 2 or cells.shape[1] < 3:
            self._show_status(
                t("hw.reconstruction.error.expect_2d_triangles"), error=True
            )
            return

        # Tetra meshes need their boundary projected to 2D triangles
        # before tricontour can take them.
        if cells.shape[1] == 4:
            cells = _project_tetra_to_triangles(cells)
            if cells.shape[0] == 0:
                self._show_status(
                    t("hw.equipotential.no_surface"), error=True
                )
                return

        try:
            self._render_contour(sigma, coords, cells)
        except Exception as exc:  # pragma: no cover — runtime safety net
            log.exception("Equipotential render failed")
            self._show_status(str(exc), error=True)
            return

        self._last_result = (sigma, coords, cells)

    def set_loading(self, on: bool) -> None:
        """Drive the busy overlay to match the reconstruction-running state."""
        if on:
            self._show_status(
                t("hw.reconstruction.loading_overlay"), loading=True
            )
        else:
            # If we already have data, the next update_reconstruction()
            # repaints; otherwise drop back to the empty hint.
            if self._last_result is None:
                self._show_empty()

    def clear(self) -> None:
        self._last_result = None
        self._remove_colorbar()
        self._ax.clear()
        self._apply_axes_chrome()
        self._show_empty()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_contour(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> None:
        x = coords[:, 0]
        y = coords[:, 1]
        try:
            tri = Triangulation(x, y, cells)
        except Exception as exc:
            self._show_status(f"Triangulation failed: {exc}", error=True)
            return

        # Convert cell-centred σ to node values via the standard
        # area-weighted average so tricontour (which expects
        # per-node scalars) gets a clean field.
        if sigma.size == cells.shape[0]:
            node_values = self._cell_to_node(sigma, cells, len(x))
        elif sigma.size == len(x):
            node_values = sigma
        else:
            self._show_status(
                f"Size mismatch: sigma={sigma.size}, cells={cells.shape[0]}, "
                f"nodes={len(x)}",
                error=True,
            )
            return

        finite = node_values[np.isfinite(node_values)]
        if finite.size == 0:
            sigma_min, sigma_max = 0.0, 1.0
        else:
            sigma_min = float(np.nanmin(finite))
            sigma_max = float(np.nanmax(finite))
        if sigma_max - sigma_min < 1.0e-12:
            sigma_max = sigma_min + 1.0e-12

        self._remove_colorbar()
        self._ax.clear()

        levels = np.linspace(sigma_min, sigma_max, 12)
        # Filled background so the contour lines have something to
        # sit on top of — keeps the widget visually paired with the
        # adjacent ReconstructionWidget.
        filled = self._ax.tricontourf(
            tri, node_values, levels=levels, cmap="viridis", alpha=0.85
        )
        line_levels = np.linspace(sigma_min, sigma_max, 8)
        # Use a darker colour for the lines so they read on top of
        # the filled cmap regardless of light / dark mode.
        contour_colour = plot_palette().get("border", "#222")
        cs = self._ax.tricontour(
            tri, node_values, levels=line_levels,
            colors=contour_colour, linewidths=0.7,
        )
        try:
            self._ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        except Exception:  # pragma: no cover — matplotlib quirk
            pass

        palette = plot_palette()
        self._ax.set_aspect("equal")
        self._ax.set_title(
            t("hw.equipotential.title"),
            fontproperties=self._title_font,
            color=palette["text"],
        )
        self._apply_axes_chrome()

        self._colorbar = self._figure.colorbar(
            filled, ax=self._ax, label="S/m",
            shrink=0.72, aspect=16, pad=0.04,
        )
        self._colorbar.ax.yaxis.label.set_color(palette["text"])
        self._colorbar.ax.yaxis.label.set_size(9)
        self._colorbar.ax.tick_params(labelsize=8, colors=palette["text"])
        for spine in self._colorbar.ax.spines.values():
            spine.set_color(palette["border"])

        self._empty_overlay.hide()
        self._overlay_mode = "data"
        self._canvas.draw()

    @staticmethod
    def _cell_to_node(sigma: np.ndarray, cells: np.ndarray, n_nodes: int) -> np.ndarray:
        """Average per-cell scalars onto nodes using a uniform weight."""
        node_sum = np.zeros(n_nodes, dtype=np.float64)
        node_count = np.zeros(n_nodes, dtype=np.float64)
        for cell_idx, cell in enumerate(cells):
            value = float(sigma[cell_idx])
            for vertex in cell:
                vidx = int(vertex)
                if 0 <= vidx < n_nodes:
                    node_sum[vidx] += value
                    node_count[vidx] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            node_values = np.where(node_count > 0, node_sum / node_count, np.nan)
        # Replace residual NaNs with the global mean so triangulation
        # doesn't choke on the rare orphan node.
        if np.any(np.isnan(node_values)):
            mean = float(np.nanmean(node_values)) if np.any(np.isfinite(node_values)) else 0.0
            node_values = np.where(np.isnan(node_values), mean, node_values)
        return node_values

    def _apply_axes_chrome(self) -> None:
        palette = plot_palette()
        text = palette["text"]
        self._ax.set_facecolor(palette["axes_bg"])
        self._figure.patch.set_facecolor(palette["panel_bg"])
        for spine in self._ax.spines.values():
            spine.set_color(palette["border"])
        self._ax.tick_params(colors=text, labelsize=8)
        for label in self._ax.get_xticklabels() + self._ax.get_yticklabels():
            label.set_color(text)
            label.set_fontname(self._serif)

    # ------------------------------------------------------------------
    # Overlay state
    # ------------------------------------------------------------------

    def _show_empty(self) -> None:
        self._overlay_mode = "empty"
        self._overlay_message = None
        self._empty_overlay.setText(t("hw.equipotential.empty_overlay"))
        self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())
        self._empty_overlay.show()
        self._apply_axes_chrome()
        self._canvas.draw_idle()

    def _show_status(
        self, message: str, *, loading: bool = False, error: bool = False
    ) -> None:
        if loading:
            self._overlay_mode = "loading"
            self._empty_overlay.setStyleSheet(loading_scrim_stylesheet())
        elif error:
            self._overlay_mode = "error"
            self._empty_overlay.setStyleSheet(error_scrim_stylesheet())
        else:
            self._overlay_mode = "empty"
            self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())
        self._overlay_message = message
        self._empty_overlay.setText(message)
        self._empty_overlay.show()

    # ------------------------------------------------------------------
    # Theme / i18n
    # ------------------------------------------------------------------

    def _on_theme_mode_changed(self, _mode: str) -> None:
        if self._last_result is not None:
            sigma, coords, cells = self._last_result
            self._render_contour(sigma, coords, cells)
        else:
            self._apply_axes_chrome()
            self._canvas.draw_idle()
        # Refresh the overlay's stylesheet so the colours follow the
        # active palette.
        if self._overlay_mode == "loading":
            self._empty_overlay.setStyleSheet(loading_scrim_stylesheet())
        elif self._overlay_mode == "error":
            self._empty_overlay.setStyleSheet(error_scrim_stylesheet())
        elif self._overlay_mode == "empty":
            self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())

    def _retranslate(self) -> None:
        if self._overlay_mode == "empty" and self._overlay_message is None:
            self._empty_overlay.setText(t("hw.equipotential.empty_overlay"))
        if self._last_result is not None:
            palette = plot_palette()
            self._ax.set_title(
                t("hw.equipotential.title"),
                fontproperties=self._title_font,
                color=palette["text"],
            )
            self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remove_colorbar(self) -> None:
        if self._colorbar is None:
            return
        try:
            self._colorbar.remove()
        except (AttributeError, KeyError, RuntimeError, ValueError):
            cax = getattr(self._colorbar, "ax", None)
            if cax is not None and cax in self._figure.axes:
                try:
                    self._figure.delaxes(cax)
                except (AttributeError, KeyError, RuntimeError, ValueError):
                    pass
        self._colorbar = None
