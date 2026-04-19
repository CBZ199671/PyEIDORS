"""3D equipotential view of a reconstructed conductivity field.

Pairs with :class:`ReconstructionWidget` on the Hardware tab.  The
reconstruction widget shows the σ field as a flat 2D image; this
widget warps the same field into a 3D height surface (Z = σ × scale)
so peaks of high conductivity rise above valleys of low conductivity
and the spatial gradient becomes immediately legible.

Implementation notes:

* Render via PyVista *offscreen* — VTK draws to an off-screen
  framebuffer, we screenshot the result into a QPixmap and paint it
  into a QLabel.  This is the same pattern Conductivity3DWidget uses
  for its WSLg-friendly path (XCB / native VTK windows are flaky on
  WSLg, but offscreen rendering bypasses that entirely).
* Drag-to-rotate and wheel-to-zoom go through the same
  ``_OffscreenRenderLabel`` helper Conductivity3DWidget uses.
* Falls back to a 2D matplotlib contour view when PyVista isn't
  available, so the widget never crashes the GUI on minimal builds.

Public API mirrors ReconstructionWidget so the existing wiring in
main_window stays unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.tri import Triangulation
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.conductivity_3d_widget import _OffscreenRenderLabel, _hex_to_rgb
from eit_app.ui.fonts import plot_font_families, serif_font_family
from eit_app.ui.theme import (
    empty_placeholder_stylesheet,
    error_scrim_stylesheet,
    loading_scrim_stylesheet,
    plot_palette,
    set_button_role,
    set_hint_text,
    subscribe_theme_mode,
)


if TYPE_CHECKING:
    from eit_app.controllers.reconstruction_controller import ReconstructionResult


log = logging.getLogger(__name__)


def _project_tetra_to_triangles(cells: np.ndarray) -> np.ndarray:
    """Reduce a 4-vertex tetra mesh to its 2D boundary triangles.

    The 3D height-surface only needs a planar triangulation (Z = σ
    becomes the new height); a 3D mesh's boundary triangulation is
    the union of every face that appears exactly once across all
    tetrahedra.
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


def _cell_to_node(sigma: np.ndarray, cells: np.ndarray, n_nodes: int) -> np.ndarray:
    """Average per-cell scalars onto nodes via uniform weighting."""
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
    if np.any(np.isnan(node_values)):
        mean = (
            float(np.nanmean(node_values))
            if np.any(np.isfinite(node_values))
            else 0.0
        )
        node_values = np.where(np.isnan(node_values), mean, node_values)
    return node_values


class EquipotentialPlotWidget(QWidget):
    """3D height-surface view of a conductivity reconstruction."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._serif = serif_font_family()
        self._title_font = FontProperties(family=plot_font_families(), size=13)

        self._last_payload: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = None
        self._overlay_mode = "empty"  # "empty" | "loading" | "error" | "data"
        self._overlay_message: str | None = None

        # PyVista 3D state
        self._render_backend = "caption"  # "caption" | "pyvista" | "mpl3d"
        self._plotter = None  # pv.Plotter
        self._mesh_actor = None
        self._scalar_bar_args: dict | None = None

        # Matplotlib fallback state
        self._mpl_figure: Figure | None = None
        self._mpl_canvas: FigureCanvasQTAgg | None = None
        self._mpl_ax = None

        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        subscribe_theme_mode(self._on_theme_mode_changed)
        self._show_empty()

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._stack_host = QFrame(self)
        self._stack = QStackedLayout(self._stack_host)
        self._stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self._stack.setContentsMargins(0, 0, 0, 0)

        # Caption layer (placeholder / loading / error).
        self._caption_label = QLabel("")
        self._caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption_label.setWordWrap(True)
        self._caption_label.setStyleSheet(empty_placeholder_stylesheet())
        self._stack.addWidget(self._caption_label)

        # PyVista offscreen pixmap layer with drag/zoom.
        self._offscreen_label = _OffscreenRenderLabel(self._stack_host)
        self._offscreen_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._offscreen_label.dragged.connect(self._on_drag)
        self._offscreen_label.zoomed.connect(self._on_zoom)
        self._offscreen_label.hide()
        self._stack.addWidget(self._offscreen_label)

        outer.addWidget(self._stack_host, 1)

        # Compact controls bar — height-scale slider + reset.  Only
        # visible while a 3D scene is on screen.
        self._controls = QFrame()
        self._controls.setObjectName("equipotentialControls")
        bar = QHBoxLayout(self._controls)
        bar.setContentsMargins(6, 2, 6, 2)
        bar.setSpacing(6)

        self._height_label = QLabel("")
        set_hint_text(self._height_label)
        bar.addWidget(self._height_label)

        self._height_slider = QSlider(Qt.Orientation.Horizontal)
        self._height_slider.setRange(0, 100)
        self._height_slider.setValue(35)  # ~moderate warp
        self._height_slider.setMinimumWidth(60)
        self._height_slider.valueChanged.connect(self._on_height_changed)
        bar.addWidget(self._height_slider, 1)

        self._height_value = QLabel("0.35")
        set_hint_text(self._height_value)
        bar.addWidget(self._height_value)

        self._reset_btn = QPushButton("")
        set_button_role(self._reset_btn, "tertiary")
        self._reset_btn.clicked.connect(self._reset_camera)
        bar.addWidget(self._reset_btn)

        outer.addWidget(self._controls)
        self._controls.hide()

    # ------------------------------------------------------------------
    # Public API (mirrors ReconstructionWidget)
    # ------------------------------------------------------------------

    def update_reconstruction(self, result: ReconstructionResult) -> None:
        """Render the 3D height surface for a fresh reconstruction."""
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

        if cells.shape[1] == 4:
            cells = _project_tetra_to_triangles(cells)
            if cells.shape[0] == 0:
                self._show_status(
                    t("hw.equipotential.no_surface"), error=True
                )
                return

        if sigma.size == cells.shape[0]:
            node_values = _cell_to_node(sigma, cells, coords.shape[0])
        elif sigma.size == coords.shape[0]:
            node_values = sigma
        else:
            self._show_status(
                f"Size mismatch: sigma={sigma.size}, cells={cells.shape[0]}, "
                f"nodes={coords.shape[0]}",
                error=True,
            )
            return

        self._last_payload = (node_values, coords, cells)

        # Try the PyVista path first; on any import / runtime failure
        # fall back to the matplotlib 3D surface so the widget still
        # produces something useful.
        if self._render_pyvista(node_values, coords, cells):
            self._render_backend = "pyvista"
            self._stack.setCurrentWidget(self._offscreen_label)
            self._offscreen_label.show()
            self._caption_label.hide()
            self._controls.show()
            self._overlay_mode = "data"
            return

        if self._render_mpl3d(node_values, coords, cells):
            self._render_backend = "mpl3d"
            self._stack.setCurrentWidget(self._mpl_canvas)
            self._caption_label.hide()
            self._controls.show()
            self._overlay_mode = "data"

    def set_loading(self, on: bool) -> None:
        if on:
            self._show_status(t("hw.reconstruction.loading_overlay"), loading=True)
        elif self._last_payload is None:
            self._show_empty()

    def clear(self) -> None:
        self._last_payload = None
        self._discard_plotter()
        self._show_empty()

    # ------------------------------------------------------------------
    # PyVista offscreen rendering
    # ------------------------------------------------------------------

    def _render_pyvista(
        self,
        node_values: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> bool:
        try:
            import pyvista as pv
        except Exception as exc:  # pragma: no cover — env without VTK
            log.info("PyVista unavailable for equipotential 3D: %s", exc)
            return False

        try:
            self._discard_plotter()

            # Build planar PolyData (z=0) with sigma stored as a point
            # scalar.  warp_by_scalar later promotes Z = factor * sigma
            # so the field becomes a 3D surface.
            n_pts = coords.shape[0]
            points = np.zeros((n_pts, 3), dtype=np.float64)
            points[:, :2] = coords[:, :2]
            faces = np.empty((cells.shape[0], 4), dtype=np.int64)
            faces[:, 0] = 3
            faces[:, 1:] = cells
            mesh = pv.PolyData(points, faces.flatten())
            mesh.point_data["sigma"] = node_values

            warp_factor = self._compute_warp_factor(node_values, coords)
            warped = mesh.warp_by_scalar("sigma", factor=warp_factor)

            width, height = self._render_size()
            plotter = pv.Plotter(off_screen=True, window_size=(width, height))
            palette = plot_palette()
            plotter.set_background(_hex_to_rgb(palette.get("axes_bg", "#ffffff")))
            text_color = _hex_to_rgb(palette.get("text", "#222"))
            scalar_bar_args = {
                "title": "S/m",
                "color": text_color,
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.05,
                "width": 0.07,
                "height": 0.6,
                "title_font_size": 14,
                "label_font_size": 11,
            }
            self._mesh_actor = plotter.add_mesh(
                warped,
                scalars="sigma",
                cmap="viridis",
                show_edges=False,
                show_scalar_bar=True,
                scalar_bar_args=scalar_bar_args,
                smooth_shading=True,
            )

            # Ground-plane outline at z = 0 — same XY footprint the
            # ReconstructionWidget on the left uses.  Gives the
            # operator a visual anchor between "where in the recon
            # image am I looking" and "where on the warped surface".
            outline = mesh.extract_feature_edges(
                boundary_edges=True, feature_edges=True,
                non_manifold_edges=False, manifold_edges=False,
                feature_angle=15.0,
            )
            if outline.n_points > 0:
                plotter.add_mesh(
                    outline, color=palette.get("border", "#888"),
                    line_width=1.4, opacity=0.55,
                )

            # Make the camera + axis labels match the 2D recon's
            # convention: X right, Y up.  reconstruction_widget.py
            # stores y-up (math convention) via a negated QRectF
            # height — without an explicit camera the default
            # view_isometric() rotates the scene unpredictably and
            # peaks no longer line up with regions of the 2D image.
            self._apply_recon_aligned_camera(plotter, coords, warped)
            plotter.add_axes(
                xlabel="X", ylabel="Y", zlabel="\u03c3", line_width=2,
            )

            self._plotter = plotter
            self._scalar_bar_args = scalar_bar_args
            self._refresh_offscreen_pixmap()
            return True
        except Exception as exc:
            log.warning("PyVista 3D equipotential render failed: %s", exc)
            self._discard_plotter()
            return False

    @staticmethod
    def _apply_recon_aligned_camera(plotter, coords: np.ndarray, warped) -> None:
        """Park the camera so the warped surface lines up with the 2D recon.

        ``ReconstructionWidget`` paints the conductivity image with X
        going right and Y going up (math convention).  PyVista's
        default ``view_isometric()`` rotates the scene 30 / 30 around
        the cube diagonal which does NOT preserve that orientation —
        peaks of the warped surface end up at unpredictable screen
        positions, so the user cannot mentally map a peak back to a
        region in the recon image.

        This helper points the camera from a +X / -Y / +Z position
        toward the bounding-box centre with up = +Z.  Result:

          * X axis still points right on screen
          * Y axis goes back-and-up (positive Y is "deeper into"
            the scene, matching the recon's "up" direction)
          * Z (sigma height) points straight up
        """
        x = np.asarray(coords[:, 0], dtype=float)
        y = np.asarray(coords[:, 1], dtype=float)
        if x.size == 0 or y.size == 0:
            return
        cx = float((np.nanmin(x) + np.nanmax(x)) / 2.0)
        cy = float((np.nanmin(y) + np.nanmax(y)) / 2.0)
        diameter = float(
            max(
                np.nanmax(x) - np.nanmin(x),
                np.nanmax(y) - np.nanmin(y),
                1.0e-6,
            )
        )
        # Pull Z range from the warped mesh so the focal point sits at
        # the geometric centre of the visible scene, not the un-warped
        # ground plane.
        try:
            zmin, zmax = float(warped.bounds[4]), float(warped.bounds[5])
        except Exception:
            zmin, zmax = 0.0, diameter * 0.3
        cz = (zmin + zmax) / 2.0

        camera = plotter.camera
        camera.position = (
            cx + 1.1 * diameter,
            cy - 1.6 * diameter,
            zmax + 0.9 * diameter,
        )
        camera.focal_point = (cx, cy, cz)
        camera.up = (0.0, 0.0, 1.0)
        plotter.reset_camera_clipping_range()

    def _compute_warp_factor(
        self, node_values: np.ndarray, coords: np.ndarray
    ) -> float:
        """Pick a warp scale that makes the surface visually balanced.

        Targets a vertical span ≈ ``slider_value`` × the mesh's planar
        diameter, so the height variation reads naturally regardless
        of σ's absolute magnitude.
        """
        finite = node_values[np.isfinite(node_values)]
        if finite.size == 0:
            return 1.0
        sigma_span = float(np.nanmax(finite) - np.nanmin(finite))
        if sigma_span < 1.0e-12:
            return 0.0
        x = coords[:, 0]
        y = coords[:, 1]
        diameter = float(
            np.hypot(np.nanmax(x) - np.nanmin(x), np.nanmax(y) - np.nanmin(y))
        )
        if diameter <= 0.0 or not np.isfinite(diameter):
            diameter = 1.0
        scale_norm = self._height_slider.value() / 100.0
        return (scale_norm * diameter) / sigma_span

    def _render_size(self) -> tuple[int, int]:
        dpr = max(self.devicePixelRatioF(), 1.0)
        width = max(280, int(self._offscreen_label.width() * dpr))
        height = max(220, int(self._offscreen_label.height() * dpr))
        return min(width, 2400), min(height, 1800)

    def _refresh_offscreen_pixmap(self) -> None:
        plotter = self._plotter
        if plotter is None:
            return
        width, height = self._render_size()
        try:
            plotter.window_size = (width, height)
            plotter.render()
            image = np.ascontiguousarray(plotter.screenshot(return_img=True))
        except Exception as exc:  # pragma: no cover — VTK runtime quirk
            log.warning("Equipotential offscreen render failed: %s", exc)
            return
        if image.ndim != 3 or image.shape[2] < 3:
            return
        image = image[:, :, :3]
        qimage = QImage(
            image.data,
            int(image.shape[1]),
            int(image.shape[0]),
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimage)
        pixmap.setDevicePixelRatio(max(self.devicePixelRatioF(), 1.0))
        self._offscreen_label.setPixmap(pixmap)

    def _discard_plotter(self) -> None:
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:  # pragma: no cover — best effort
                pass
        self._plotter = None
        self._mesh_actor = None
        self._scalar_bar_args = None

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        # Re-rasterise on resize so the surface fills the new size
        # without obvious blur from up-scaling the cached pixmap.
        if self._render_backend == "pyvista" and self._plotter is not None:
            QTimer.singleShot(0, self._refresh_offscreen_pixmap)

    # ------------------------------------------------------------------
    # Matplotlib 3D fallback
    # ------------------------------------------------------------------

    def _render_mpl3d(
        self,
        node_values: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> bool:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')
        except Exception:  # pragma: no cover — bundled with matplotlib
            return False

        if self._mpl_figure is None:
            palette = plot_palette()
            self._mpl_figure = Figure(figsize=(4, 4), tight_layout=True)
            self._mpl_figure.patch.set_facecolor(palette["panel_bg"])
            self._mpl_canvas = FigureCanvasQTAgg(self._mpl_figure)
            self._mpl_canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._mpl_ax = self._mpl_figure.add_subplot(111, projection="3d")
            self._stack.addWidget(self._mpl_canvas)

        ax = self._mpl_ax
        ax.clear()
        palette = plot_palette()
        ax.set_facecolor(palette["axes_bg"])
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            axis.label.set_color(palette["text"])
            axis.set_tick_params(colors=palette["text"], labelsize=8)

        try:
            tri = Triangulation(coords[:, 0], coords[:, 1], cells)
        except Exception:
            return False

        ax.plot_trisurf(
            tri,
            node_values,
            cmap="viridis",
            edgecolor="none",
            antialiased=True,
        )
        ax.set_title(
            t("hw.equipotential.title"),
            fontproperties=self._title_font,
            color=palette["text"],
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("σ")
        # Match the PyVista path's recon-aligned camera: looking from
        # +X / -Y / +Z so X reads right and Y reads back-and-up,
        # matching the 2D ReconstructionWidget on the left.
        ax.view_init(elev=25.0, azim=-55.0)
        self._mpl_canvas.draw()
        return True

    # ------------------------------------------------------------------
    # Camera controls
    # ------------------------------------------------------------------

    def _on_drag(self, dx: float, dy: float) -> None:
        if self._render_backend != "pyvista" or self._plotter is None:
            return
        camera = self._plotter.camera
        camera.Azimuth(-dx * 0.45)
        camera.Elevation(dy * 0.45)
        camera.OrthogonalizeViewUp()
        self._refresh_offscreen_pixmap()

    def _on_zoom(self, delta_y: float) -> None:
        if self._render_backend != "pyvista" or self._plotter is None:
            return
        self._plotter.camera.Zoom(1.12 if delta_y > 0 else 0.89)
        self._refresh_offscreen_pixmap()

    def _on_height_changed(self, value: int) -> None:
        scale = value / 100.0
        self._height_value.setText(f"{scale:.2f}")
        if self._last_payload is None:
            return
        if self._render_backend == "pyvista":
            # Cheapest re-warp: rebuild the warped mesh and re-render.
            node_values, coords, cells = self._last_payload
            self._render_pyvista(node_values, coords, cells)
            self._refresh_offscreen_pixmap()
        elif self._render_backend == "mpl3d":
            # mpl3d's surface height is rendered straight from σ; no
            # warp factor needed (the user can spin the camera instead).
            pass

    def _reset_camera(self) -> None:
        if self._render_backend == "pyvista" and self._plotter is not None:
            # Re-park the camera at the same recon-aligned position
            # we picked at first render — rather than view_isometric()
            # which would rotate to PyVista's generic 30/30 view and
            # break the X / Y correspondence with the 2D recon.
            if self._last_payload is not None:
                _, coords, _ = self._last_payload
                # The warped mesh is hidden inside the actor; reuse its
                # bounding box so the camera focal point stays centred.
                actor = self._mesh_actor
                warped_bounds = (
                    actor.GetBounds() if actor is not None else None
                )
                self._apply_recon_aligned_camera_from_bounds(
                    self._plotter, coords, warped_bounds
                )
            self._refresh_offscreen_pixmap()
        elif self._render_backend == "mpl3d" and self._mpl_ax is not None:
            # Match the matplotlib 3D fallback to the same X-right /
            # Y-back-up / Z-up convention as the PyVista path.
            self._mpl_ax.view_init(elev=25.0, azim=-55.0)
            if self._mpl_canvas is not None:
                self._mpl_canvas.draw_idle()

    def _apply_recon_aligned_camera_from_bounds(
        self, plotter, coords: np.ndarray, bounds
    ) -> None:
        """Reset-view variant: positions the camera from a flat (xmin,
        xmax, ymin, ymax, zmin, zmax) tuple instead of needing the
        warped PyVista mesh.  Used by _reset_camera so the button
        works without re-running the full warp pipeline.
        """
        x = np.asarray(coords[:, 0], dtype=float)
        y = np.asarray(coords[:, 1], dtype=float)
        if x.size == 0 or y.size == 0:
            return
        cx = float((np.nanmin(x) + np.nanmax(x)) / 2.0)
        cy = float((np.nanmin(y) + np.nanmax(y)) / 2.0)
        diameter = float(
            max(
                np.nanmax(x) - np.nanmin(x),
                np.nanmax(y) - np.nanmin(y),
                1.0e-6,
            )
        )
        if bounds is not None and len(bounds) >= 6:
            zmin, zmax = float(bounds[4]), float(bounds[5])
        else:
            zmin, zmax = 0.0, diameter * 0.3
        cz = (zmin + zmax) / 2.0
        camera = plotter.camera
        camera.position = (
            cx + 1.1 * diameter,
            cy - 1.6 * diameter,
            zmax + 0.9 * diameter,
        )
        camera.focal_point = (cx, cy, cz)
        camera.up = (0.0, 0.0, 1.0)
        plotter.reset_camera_clipping_range()

    # ------------------------------------------------------------------
    # Caption / theme
    # ------------------------------------------------------------------

    def _show_empty(self) -> None:
        self._render_backend = "caption"
        self._overlay_mode = "empty"
        self._overlay_message = None
        self._caption_label.setText(t("hw.equipotential.empty_overlay"))
        self._caption_label.setStyleSheet(empty_placeholder_stylesheet())
        self._caption_label.show()
        self._offscreen_label.hide()
        if self._mpl_canvas is not None:
            self._mpl_canvas.hide()
        self._stack.setCurrentWidget(self._caption_label)
        self._controls.hide()

    def _show_status(
        self, message: str, *, loading: bool = False, error: bool = False
    ) -> None:
        if loading:
            self._overlay_mode = "loading"
            self._caption_label.setStyleSheet(loading_scrim_stylesheet())
        elif error:
            self._overlay_mode = "error"
            self._caption_label.setStyleSheet(error_scrim_stylesheet())
        else:
            self._overlay_mode = "empty"
            self._caption_label.setStyleSheet(empty_placeholder_stylesheet())
        self._overlay_message = message
        self._caption_label.setText(message)
        self._caption_label.show()
        self._stack.setCurrentWidget(self._caption_label)
        self._controls.hide()

    def _on_theme_mode_changed(self, _mode: str) -> None:
        if self._last_payload is not None and self._render_backend == "pyvista":
            node_values, coords, cells = self._last_payload
            # Re-render with the new background / text colours.
            self._render_pyvista(node_values, coords, cells)
        elif self._last_payload is not None and self._render_backend == "mpl3d":
            node_values, coords, cells = self._last_payload
            self._render_mpl3d(node_values, coords, cells)
        if self._overlay_mode == "loading":
            self._caption_label.setStyleSheet(loading_scrim_stylesheet())
        elif self._overlay_mode == "error":
            self._caption_label.setStyleSheet(error_scrim_stylesheet())
        elif self._overlay_mode == "empty":
            self._caption_label.setStyleSheet(empty_placeholder_stylesheet())

    def _retranslate(self) -> None:
        self._height_label.setText(t("hw.equipotential.height_label"))
        self._reset_btn.setText(t("hw.equipotential.reset_button"))
        if self._overlay_mode == "empty" and self._overlay_message is None:
            self._caption_label.setText(t("hw.equipotential.empty_overlay"))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._discard_plotter()
        super().closeEvent(event)
