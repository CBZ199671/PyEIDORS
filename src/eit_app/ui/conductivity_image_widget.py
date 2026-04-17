"""Reusable matplotlib conductivity display (2D tripcolor / 3D Poly3D)."""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySide6.QtWidgets import QVBoxLayout, QWidget

from eit_app.ui.fonts import plot_font_families, serif_font_family
from eit_app.ui.theme import plot_palette, subscribe_theme_mode


def _triangle_area_xy(triangles: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = x[triangles[:, 0]]
    y0 = y[triangles[:, 0]]
    x1 = x[triangles[:, 1]]
    y1 = y[triangles[:, 1]]
    x2 = x[triangles[:, 2]]
    y2 = y[triangles[:, 2]]
    return 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))


def _is_3d_mesh(node_coords: np.ndarray, cell_connectivity: np.ndarray) -> bool:
    """True iff node coords carry a Z column and cells are tetrahedra."""
    coords = np.asarray(node_coords)
    cells = np.asarray(cell_connectivity)
    if coords.ndim != 2 or coords.shape[1] < 3:
        return False
    if cells.ndim != 2 or cells.shape[1] != 4:
        return False
    return bool(np.ptp(coords[:, 2]) > 1.0e-9)


def _extract_boundary_triangles_3d(
    cell_connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tetrahedral boundary faces as (triangles[N,3], source_cell[N])."""
    cells = np.asarray(cell_connectivity, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] != 4:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32)

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
    return triangles, sources


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

    def __init__(self, title: str = "Conductivity", parent: QWidget | None = None) -> None:
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
        self._last_image: tuple[np.ndarray, np.ndarray, np.ndarray, str | None] | None = None
        self._last_caption: tuple[str, str] | None = None  # (text, kind: 'placeholder'|'loading'|'error')

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        palette = plot_palette()
        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._figure.patch.set_facecolor(palette["panel_bg"])
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas)

        self._ax_is_3d = False
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor(palette["axes_bg"])
        self._ax.set_title(title, fontproperties=self._title_font, color=palette["text"])
        self._ax.set_aspect("equal")
        self._colorbar = None
        self._show_placeholder()

        # Re-paint canvas / axes / caption colors on dark-mode toggles.
        subscribe_theme_mode(self._on_theme_mode_changed)

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
            self._ax.set_title(self._default_title, fontproperties=self._title_font, color=palette["text"])
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
        self._ax.set_title(title, fontproperties=self._title_font, color=palette["text"])
        self._canvas.draw_idle()

    def update_image(
        self,
        conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Render a conductivity distribution.

        Picks 2D tripcolor for 2D meshes (triangles or projected tetra)
        and a true 3D Poly3DCollection for 3D tetrahedral meshes whose
        node coords carry a non-degenerate Z column.
        """
        if node_coords.ndim != 2 or node_coords.shape[1] < 2:
            self._ensure_axes(is_3d=False)
            self._remove_colorbar()
            self._ax.clear()
            self._show_error("Invalid mesh coordinates")
            return

        if _is_3d_mesh(node_coords, cell_connectivity):
            self._render_3d(conductivity, node_coords, cell_connectivity, title)
        else:
            self._render_2d(conductivity, node_coords, cell_connectivity, title)

    def _render_2d(
        self,
        conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        title: str | None,
    ) -> None:
        self._ensure_axes(is_3d=False)
        self._remove_colorbar()
        self._ax.clear()

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
            tpc = self._ax.tripcolor(tri, conductivity, shading="gouraud", cmap="viridis")
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

        self._attach_colorbar(tpc)

        self._last_image = (conductivity, node_coords, cell_connectivity, title)
        self._last_caption = None

        self._canvas.draw()

    def _render_3d(
        self,
        conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        title: str | None,
    ) -> None:
        """Draw the tetrahedral mesh as a 3D Poly3DCollection of boundary faces."""
        self._ensure_axes(is_3d=True)
        self._remove_colorbar()
        self._ax.clear()

        triangles, source_cells = _extract_boundary_triangles_3d(cell_connectivity)
        if len(triangles) == 0:
            self._show_error("3D render failed: no boundary faces")
            return

        coords3 = np.asarray(node_coords, dtype=float)[:, :3]
        sigma = np.asarray(conductivity, dtype=float)

        if len(sigma) == len(cell_connectivity):
            face_values = sigma[source_cells]
        elif len(sigma) == len(coords3):
            face_values = sigma[triangles].mean(axis=1)
        else:
            self._show_error(
                f"Size mismatch: sigma={len(sigma)}, "
                f"cells={len(cell_connectivity)}, nodes={len(coords3)}"
            )
            return

        verts = coords3[triangles]  # shape (F, 3, 3)

        sigma_min = float(np.nanmin(face_values)) if face_values.size else 0.0
        sigma_max = float(np.nanmax(face_values)) if face_values.size else 1.0
        if not np.isfinite(sigma_min) or not np.isfinite(sigma_max):
            sigma_min, sigma_max = 0.0, 1.0
        if sigma_max - sigma_min < 1.0e-12:
            sigma_max = sigma_min + 1.0e-12
        norm = Normalize(vmin=sigma_min, vmax=sigma_max)
        cmap = mpl.colormaps["viridis"]
        face_colors = cmap(norm(face_values))

        palette = plot_palette()
        collection = Poly3DCollection(
            verts,
            facecolors=face_colors,
            edgecolors=palette["border"],
            linewidths=0.2,
            antialiased=True,
        )
        self._ax.add_collection3d(collection)

        self._set_3d_bounds(coords3)

        display_title = title or self._default_title
        self._ax.set_title(
            display_title, fontproperties=self._title_font, color=palette["text"]
        )
        self._apply_axes_chrome_3d()

        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(face_values)
        self._attach_colorbar(mappable)

        self._last_image = (conductivity, node_coords, cell_connectivity, title)
        self._last_caption = None

        self._canvas.draw()

    def _set_3d_bounds(self, coords3: np.ndarray) -> None:
        """Equalise the 3D bounding cube so the mesh doesn't squish."""
        if coords3.size == 0:
            return
        mins = coords3.min(axis=0)
        maxs = coords3.max(axis=0)
        spans = maxs - mins
        span = float(np.nanmax(spans))
        if not np.isfinite(span) or span <= 0.0:
            span = 1.0
        centers = (mins + maxs) / 2.0
        half = span / 2.0
        self._ax.set_xlim(centers[0] - half, centers[0] + half)
        self._ax.set_ylim(centers[1] - half, centers[1] + half)
        self._ax.set_zlim(centers[2] - half, centers[2] + half)
        try:
            self._ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

    def _attach_colorbar(self, mappable) -> None:
        """Bolt a slim S/m colorbar onto the right side of the active axes."""
        palette = plot_palette()
        # shrink + aspect + pad keep the colorbar from dominating the
        # plot height.  shrink=0.72 trims ~30% off its length, aspect=16
        # keeps it slim, pad=0.04 pulls it closer to the image so the
        # matplotlib auto-layout does not leave a huge right-hand gap.
        self._colorbar = self._figure.colorbar(
            mappable, ax=self._ax, label="S/m",
            shrink=0.72, aspect=16, pad=0.04,
        )
        self._colorbar.ax.yaxis.label.set_fontname(self._serif)
        self._colorbar.ax.yaxis.label.set_size(10)
        self._colorbar.ax.yaxis.label.set_color(palette["text"])
        self._colorbar.ax.tick_params(labelsize=8, colors=palette["text"])
        for spine in self._colorbar.ax.spines.values():
            spine.set_color(palette["border"])

    def _ensure_axes(self, *, is_3d: bool) -> None:
        """Recreate the axes if the requested projection differs from current."""
        if is_3d == self._ax_is_3d:
            return
        self._remove_colorbar()
        try:
            self._figure.delaxes(self._ax)
        except (KeyError, ValueError):
            pass
        if is_3d:
            self._ax = self._figure.add_subplot(111, projection="3d")
        else:
            self._ax = self._figure.add_subplot(111)
            self._ax.set_aspect("equal")
        self._ax_is_3d = is_3d

    def _apply_axes_chrome_3d(self) -> None:
        """Push the active palette onto a 3D axes' panes, ticks, and title."""
        palette = plot_palette()
        text = palette["text"]
        # Hide the boxy 3D pane fills so the figure background bleeds
        # through; matches the dark-mode-friendly look of the 2D path.
        for axis in (self._ax.xaxis, self._ax.yaxis, self._ax.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["color"] = palette["border"]
            axis.label.set_color(text)
            axis.label.set_fontname(self._serif)
            axis.set_tick_params(colors=text, labelsize=8)
        for spine in self._ax.spines.values():
            spine.set_color(palette["border"])
        self._figure.patch.set_facecolor(palette["panel_bg"])

    def clear(self) -> None:
        """Reset to placeholder state."""
        palette = plot_palette()
        self._ensure_axes(is_3d=False)
        self._remove_colorbar()
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
        self._ensure_axes(is_3d=False)
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

    def _show_placeholder(self) -> None:
        # Reuse the unified caption painter so placeholder/loading/error
        # all look consistent and repaint on theme changes.
        self._ensure_axes(is_3d=False)
        self._draw_caption("No data", kind="placeholder")

    def _show_error(self, msg: str) -> None:
        self._ensure_axes(is_3d=False)
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
            "loading":     palette["caption_loading"],
            "error":       palette["caption_error"],
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
            0.5, 0.5, text,
            transform=self._ax.transAxes,
            ha="center", va="center",
            fontsize=fontsize, color=color,
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
