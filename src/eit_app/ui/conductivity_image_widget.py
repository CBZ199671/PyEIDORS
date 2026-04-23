"""Reusable matplotlib tripcolor conductivity display."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.tri import Triangulation
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from eit_app.i18n import t, translator
from eit_app.ui.electrode_overlay import (
    ElectrodeGeometry,
    default_arc_segments,
)
from eit_app.ui.fonts import plot_font_families, serif_font_family
from eit_app.ui.theme import plot_palette, set_hint_text, subscribe_theme_mode


def _triangle_area_xy(
    triangles: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    x0 = x[triangles[:, 0]]
    y0 = y[triangles[:, 0]]
    x1 = x[triangles[:, 1]]
    y1 = y[triangles[:, 1]]
    x2 = x[triangles[:, 2]]
    y2 = y[triangles[:, 2]]
    return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def _project_cells_to_triangles(
    cell_connectivity: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D triangles plus the source cell index for each triangle."""
    cells = np.asarray(cell_connectivity, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] < 3:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32)

    if cells.shape[1] == 3:
        triangles = cells.astype(np.int32, copy=False)
        sources = np.arange(len(cells), dtype=np.int32)
    elif cells.shape[1] == 4:
        # Tetrahedra: draw boundary faces only, otherwise internal faces
        # overpaint the projection and make 3D phantoms unreadable.
        faces: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {}
        face_offsets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        for cell_idx, cell in enumerate(cells):
            for offsets in face_offsets:
                face = tuple(int(cell[offset]) for offset in offsets)
                key = tuple(sorted(face))
                faces[key] = None if key in faces else (face, cell_idx)
        kept = [payload for payload in faces.values() if payload is not None]
        if not kept:
            return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32)
        triangles = np.asarray([face for face, _ in kept], dtype=np.int32)
        sources = np.asarray([idx for _, idx in kept], dtype=np.int32)
    else:
        tris: list[tuple[int, int, int]] = []
        sources_list: list[int] = []
        for cell_idx, cell in enumerate(cells):
            unique = [int(value) for value in dict.fromkeys(cell.tolist())]
            if len(unique) < 3:
                continue
            for idx in range(1, len(unique) - 1):
                tris.append((unique[0], unique[idx], unique[idx + 1]))
                sources_list.append(cell_idx)
        if not tris:
            return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32)
        triangles = np.asarray(tris, dtype=np.int32)
        sources = np.asarray(sources_list, dtype=np.int32)

    valid_index = (triangles >= 0).all(axis=1) & (triangles < len(x)).all(axis=1)
    if np.any(valid_index):
        valid_triangles = triangles[valid_index]
        valid_sources = sources[valid_index]
        area_index = _triangle_area_xy(valid_triangles, x, y) > 1.0e-14
        return valid_triangles[area_index], valid_sources[area_index]
    return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32)


class ConductivityImageWidget(QWidget):
    """Displays a conductivity distribution as a matplotlib tripcolor plot."""

    def __init__(
        self, title: str = "Conductivity", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._serif = serif_font_family()
        # Title uses a FontProperties with a Latin-serif-first family list
        # so matplotlib's per-glyph fallback can reach CJK faces when the
        # title is translated to Chinese.  Without this fallback Times New
        # Roman emits "Glyph X missing" warnings and renders tofu boxes.
        self._title_font = FontProperties(family=plot_font_families(), size=14)
        self._default_title = title
        # Cache the most recent rendered state so dark-mode toggles can
        # repaint without losing the conductivity field.  None means
        # "currently showing the placeholder".
        self._last_image: (
            tuple[np.ndarray, np.ndarray, np.ndarray, str | None] | None
        ) = None
        self._last_caption: tuple[str, str] | None = (
            None  # (text, kind: 'placeholder'|'loading'|'error')
        )

        # Electrode overlay state.  Cached LineCollection lives on the
        # axes; toggling visibility never rebuilds geometry.
        self._electrode_geometry: ElectrodeGeometry | None = None
        self._electrode_collection: LineCollection | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        palette = plot_palette()
        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._figure.patch.set_facecolor(palette["panel_bg"])
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas, 1)

        # Bottom controls strip — only one toggle today, but kept in its
        # own bar so future overlays (cell IDs, etc.) can sit alongside.
        self._controls = QWidget()
        controls_layout = QHBoxLayout(self._controls)
        controls_layout.setContentsMargins(6, 0, 6, 2)
        controls_layout.setSpacing(8)
        self._electrode_check = QCheckBox("")
        self._electrode_check.setChecked(False)
        self._electrode_check.toggled.connect(self._on_electrode_toggled)
        set_hint_text(self._electrode_check)
        controls_layout.addWidget(self._electrode_check)
        controls_layout.addStretch(1)
        layout.addWidget(self._controls)
        # Hidden until a forward result actually provides electrode geometry.
        self._controls.hide()

        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor(palette["axes_bg"])
        self._ax.set_title(
            title, fontproperties=self._title_font, color=palette["text"]
        )
        self._ax.set_aspect("equal")
        self._colorbar = None
        self._show_placeholder()

        # Re-paint canvas / axes / caption colors on dark-mode toggles.
        subscribe_theme_mode(self._on_theme_mode_changed)
        translator().language_changed.connect(self._retranslate)
        self._retranslate()

    # ------------------------------------------------------------------
    # Theme integration
    # ------------------------------------------------------------------

    def _on_theme_mode_changed(self, _mode: str) -> None:
        """Re-pull colors from the active palette and re-render."""
        palette = plot_palette()
        self._figure.patch.set_facecolor(palette["panel_bg"])
        self._ax.set_facecolor(palette["axes_bg"])
        # Re-render whichever state the widget was in last.  This keeps
        # the user's data on screen across the mode flip instead of
        # collapsing back to the placeholder.
        if self._last_image is not None:
            conductivity, node_coords, cell_connectivity, title = self._last_image
            # update_image() resets _last_image, so don't pass it back
            # in to avoid a recursion.
            self.update_image(conductivity, node_coords, cell_connectivity, title=title)
        elif self._last_caption is not None:
            text, kind = self._last_caption
            self._draw_caption(text, kind)
        else:
            # Re-apply the title color even when nothing is rendered.
            self._ax.set_title(
                self._default_title,
                fontproperties=self._title_font,
                color=palette["text"],
            )
            self._canvas.draw_idle()

    def _apply_axes_chrome(self) -> None:
        """Push the active palette onto axis spines, ticks, and title."""
        palette = plot_palette()
        text = palette["text"]
        for spine in self._ax.spines.values():
            spine.set_color(palette["border"])
        self._ax.tick_params(colors=text, labelsize=9)
        for label in self._ax.get_xticklabels() + self._ax.get_yticklabels():
            label.set_color(text)
            label.set_fontname(self._serif)
        self._ax.set_facecolor(palette["axes_bg"])
        self._figure.patch.set_facecolor(palette["panel_bg"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setTitle(self, title: str) -> None:
        """Update the plot title (used by i18n retranslate pipelines)."""
        self._default_title = title
        palette = plot_palette()
        self._ax.set_title(
            title, fontproperties=self._title_font, color=palette["text"]
        )
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
        # ax.clear() drops every artist including the cached electrode
        # collection.  Drop our reference here so _redraw_electrodes
        # rebuilds from scratch with the surviving geometry — without
        # this, a stale Python handle to a removed Artist raises on the
        # next redraw.
        self._electrode_collection = None
        self._ax.clear()

        if node_coords.ndim != 2 or node_coords.shape[1] < 2:
            self._show_error("Invalid mesh coordinates")
            return

        x = node_coords[:, 0]
        y = node_coords[:, 1]

        triangles, source_cells = _project_cells_to_triangles(cell_connectivity, x, y)
        if len(triangles) == 0:
            self._show_error("Triangulation failed: no drawable 2D projection")
            return

        try:
            tri = Triangulation(x, y, triangles)
        except Exception as exc:
            self._show_error(f"Triangulation failed: {exc}")
            return

        if len(conductivity) == len(cell_connectivity):
            face_values = np.asarray(conductivity, dtype=float)[source_cells]
            tpc = self._ax.tripcolor(tri, face_values, shading="flat", cmap="viridis")
        elif len(conductivity) == len(x):
            tpc = self._ax.tripcolor(
                tri, conductivity, shading="gouraud", cmap="viridis"
            )
        else:
            self._show_error(
                f"Size mismatch: sigma={len(conductivity)}, "
                f"cells={len(cell_connectivity)}, nodes={len(x)}"
            )
            return

        palette = plot_palette()
        display_title = title or self._default_title
        self._ax.set_title(
            display_title, fontproperties=self._title_font, color=palette["text"]
        )
        self._ax.set_aspect("equal")
        self._apply_axes_chrome()

        # shrink + aspect + pad keep the colorbar from dominating the
        # plot height.  shrink=0.72 trims ~30% off its length, aspect=16
        # keeps it slim, pad=0.04 pulls it closer to the image so the
        # matplotlib auto-layout does not leave a huge right-hand gap.
        self._colorbar = self._figure.colorbar(
            tpc,
            ax=self._ax,
            label="S/m",
            shrink=0.72,
            aspect=16,
            pad=0.04,
        )
        self._colorbar.ax.yaxis.label.set_fontname(self._serif)
        self._colorbar.ax.yaxis.label.set_size(10)
        self._colorbar.ax.yaxis.label.set_color(palette["text"])
        # Slightly smaller tick labels so the numbers don't compete with
        # the main title for visual weight.
        self._colorbar.ax.tick_params(labelsize=8, colors=palette["text"])
        for spine in self._colorbar.ax.spines.values():
            spine.set_color(palette["border"])

        # Cache so dark-mode toggles can re-render without losing data.
        self._last_image = (conductivity, node_coords, cell_connectivity, title)
        self._last_caption = None

        # Re-attach electrode overlay if the user has it enabled.  The
        # ax.clear() above wiped the previous LineCollection.
        self._redraw_electrodes()

        self._canvas.draw()

    def set_electrode_geometry(self, geometry: ElectrodeGeometry | None) -> None:
        """Cache new electrode geometry and reveal / hide the toggle.

        Geometry is rebuilt only here; toggling visibility of the cached
        LineCollection is what runs on every checkbox click.
        """
        self._electrode_geometry = geometry
        if geometry is None or not geometry.arcs:
            self._controls.hide()
            if self._electrode_collection is not None:
                try:
                    self._electrode_collection.remove()
                except (AttributeError, ValueError):
                    pass
                self._electrode_collection = None
                self._canvas.draw_idle()
            return
        self._controls.show()
        self._redraw_electrodes()
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Reset to placeholder state."""
        palette = plot_palette()
        self._remove_colorbar()
        # Wipe stale electrode handle before clearing axes — see notes
        # in update_image() for why this matters.
        self._electrode_collection = None
        self._ax.clear()
        self._ax.set_facecolor(palette["axes_bg"])
        self._ax.set_title(
            self._default_title, fontproperties=self._title_font, color=palette["text"]
        )
        self._show_placeholder()

    def set_loading(self, message: str | None = None) -> None:
        """Show a centered loading chip and hide any previous image.

        Used by SimulationResultsWidget.set_loading() while a forward
        or inverse solve is in flight.  Clears the existing tripcolor
        and colorbar first so the "正问题求解中" / "逆问题求解中"
        caption doesn't render on top of stale data — a cleaner modal
        loading state than the earlier transparent-overlay-on-data
        behaviour.
        """
        self._remove_colorbar()
        self._ax.clear()
        self._draw_caption(message or "Loading\u2026", kind="loading")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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

    def _redraw_electrodes(self) -> None:
        """Attach the cached electrode arcs to the current axes if visible.

        Idempotent: removes any prior collection first so repeated calls
        (after ax.clear, theme switch, language change) don't stack
        artists.
        """
        if self._electrode_collection is not None:
            try:
                self._electrode_collection.remove()
            except (AttributeError, ValueError):
                pass
            self._electrode_collection = None
        geometry = self._electrode_geometry
        if (
            geometry is None
            or not geometry.arcs
            or not self._electrode_check.isChecked()
        ):
            return
        segments = default_arc_segments(geometry.arcs, geometry.radius)
        if not segments:
            return
        palette = plot_palette()
        # Use the highlight palette colour so the arcs read clearly
        # against either the viridis bulk or an empty placeholder.
        color = palette.get("highlight", "#f39c12")
        collection = LineCollection(
            segments,
            colors=[color] * len(segments),
            linewidths=3.0,
            capstyle="round",
            zorder=5,
        )
        self._ax.add_collection(collection)
        self._electrode_collection = collection

    def _on_electrode_toggled(self, _checked: bool) -> None:
        # Cheap path: rebuild only if visibility flipped to on; remove
        # without rebuild if flipped to off.
        self._redraw_electrodes()
        self._canvas.draw_idle()

    def _retranslate(self) -> None:
        self._electrode_check.setText(t("sim.results.electrodes_toggle"))

    def _show_placeholder(self) -> None:
        # Reuse the unified caption painter so placeholder/loading/error
        # all look consistent and repaint on theme changes.
        self._draw_caption("No data", kind="placeholder")

    def _show_error(self, msg: str) -> None:
        self._ax.clear()
        self._draw_caption(msg, kind="error")

    def _draw_caption(self, text: str, kind: str) -> None:
        """Centered caption painter; kind ∈ {'placeholder','loading','error'}.

        Loading / error states render the caption inside a rounded chip
        bbox so they read as active state badges rather than floating
        text.  Placeholder stays as bare text (passive "No data" hint
        that should not compete for attention).
        """
        palette = plot_palette()
        color = {
            "placeholder": palette["caption"],
            "loading": palette["caption_loading"],
            "error": palette["caption_error"],
        }.get(kind, palette["caption"])
        # Make sure background colors track the theme even when the
        # caption pre-empts a render that would otherwise call
        # _apply_axes_chrome().
        self._ax.set_facecolor(palette["axes_bg"])
        self._figure.patch.set_facecolor(palette["panel_bg"])
        self._ax.set_title(
            self._default_title, fontproperties=self._title_font, color=palette["text"]
        )
        # Loading / error captions get a rounded chip bbox so they're
        # visually distinct from the idle placeholder and so the caption
        # stands out cleanly even if any stale alpha-blended data
        # happened to survive underneath.
        bbox = None
        fontsize = 11
        fontweight = "normal"
        if kind in ("loading", "error"):
            bbox = dict(
                boxstyle="round,pad=0.7",
                facecolor=palette["panel_bg"],
                edgecolor=color,
                linewidth=1.5,
            )
            fontsize = 12
            fontweight = "bold"
        self._ax.text(
            0.5,
            0.5,
            text,
            transform=self._ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
            fontproperties=self._title_font,
            bbox=bbox,
        )
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        # Cache so the theme-mode listener can repaint.
        self._last_image = None
        self._last_caption = (text, kind)
        self._canvas.draw()
