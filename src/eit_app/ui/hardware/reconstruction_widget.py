"""Fast pyqtgraph-based conductivity reconstruction display widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QLabel, QStackedLayout, QVBoxLayout, QWidget
from scipy.spatial import Delaunay

from eit_app.i18n import t, translator
from eit_app.ui.theme import (
    empty_placeholder_stylesheet,
    error_scrim_stylesheet,
    loading_scrim_stylesheet,
    plot_palette,
    subscribe_theme_mode,
)

if TYPE_CHECKING:
    from eit_app.controllers.reconstruction_controller import ReconstructionResult


class ReconstructionWidget(QWidget):
    """Displays conductivity maps with a cached static scene and fast image refresh."""

    _GRID_SIZE = 256

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._font_family = "Times New Roman"
        # Pull plot colors from the theme palette so the widget honours
        # dark mode automatically.  subscribe_theme_mode in __init__
        # rewires re-paint on later mode flips.
        palette = plot_palette()
        self._plot_bg = palette["panel_bg"]
        self._domain_border = palette["domain"]
        self._electrode_fill = "#ffffff"
        self._electrode_border = palette["electrode"]
        self._label_color = palette["label"]
        self._mesh_cache_key: tuple[Any, ...] | None = None
        self._grid_vertices: np.ndarray | None = None
        self._grid_weights: np.ndarray | None = None
        self._grid_valid_mask: np.ndarray | None = None
        self._grid_shape = (self._GRID_SIZE, self._GRID_SIZE)
        self._grid_rect = QRectF(-1.0, 1.0, 2.0, -2.0)
        self._electrode_label_items: list[pg.TextItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        plot_host = QWidget(self)
        plot_stack = QStackedLayout(plot_host)
        plot_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        plot_stack.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(self._plot_bg)
        self._plot_widget.setMenuEnabled(False)
        self._plot_widget.hideButtons()
        self._plot_widget.setMouseEnabled(x=False, y=False)
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.setSizePolicy(self.sizePolicy())
        plot_item = self._plot_widget.getPlotItem()
        plot_item.hideAxis("left")
        plot_item.hideAxis("bottom")
        plot_item.getViewBox().setDefaultPadding(0.0)
        # Title HTML is rebuilt in _retranslate() so it follows the UI language.
        self._plot_item_ref = plot_item  # kept for _retranslate

        self._image_item = pg.ImageItem(axisOrder="row-major")
        plot_item.addItem(self._image_item)

        self._boundary_item = pg.PlotDataItem(
            pen=pg.mkPen(self._domain_border, width=2.0)
        )
        plot_item.addItem(self._boundary_item)

        self._electrode_arc_item = pg.PlotDataItem(
            pen=pg.mkPen(self._electrode_border, width=4.0),
            antialias=True,
        )
        plot_item.addItem(self._electrode_arc_item)

        # Empty-overlay text is populated by _retranslate() below so it
        # follows the active language.  The `_overlay_mode` flag lets
        # _retranslate distinguish the idle placeholder from transient
        # error messages surfaced via _show_status() and from the
        # Phase 4 "loading" state set via set_loading().
        self._empty_overlay = QLabel("")
        self._empty_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_overlay.setStyleSheet(
            "color: #5b6573; font-size: 12px; font-weight: 600; background: transparent;"
        )
        self._overlay_mode = "empty"  # "empty" | "loading" | "error"

        plot_stack.addWidget(self._plot_widget)
        plot_stack.addWidget(self._empty_overlay)
        layout.addWidget(plot_host)

        self._colormap = pg.ColorMap(
            pos=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            color=np.array(
                [
                    [44, 123, 182, 255],
                    [248, 248, 248, 255],
                    [215, 48, 39, 255],
                ],
                dtype=np.ubyte,
            ),
        )
        self._reset_plot()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        # Re-paint canvas + electrode + boundary colors when the user
        # toggles dark mode.
        subscribe_theme_mode(self._on_theme_mode_changed)

    def _on_theme_mode_changed(self, _mode: str) -> None:
        """Re-pull the plot palette and re-paint pyqtgraph chrome.

        The conductivity image itself is colormapped from the data
        and doesn't follow the theme; only the canvas background,
        domain outline, electrode arc, and labels are re-skinned.
        """
        palette = plot_palette()
        self._plot_bg = palette["panel_bg"]
        self._domain_border = palette["domain"]
        self._electrode_border = palette["electrode"]
        self._label_color = palette["label"]
        self._plot_widget.setBackground(self._plot_bg)
        # Re-pen the static scene items so they match the new palette.
        self._boundary_item.setPen(pg.mkPen(self._domain_border, width=2.0))
        self._electrode_arc_item.setPen(pg.mkPen(self._electrode_border, width=4.0))
        # Refresh title HTML (uses _label_color) and electrode label
        # text colors via the standard retranslate / scene rebuild.
        self._retranslate()
        # Force the electrode-number labels to pick up the new color.
        for item in self._electrode_label_items:
            try:
                item.setColor(self._label_color)
            except Exception:
                pass

    def set_loading(self, on: bool) -> None:
        """Toggle the 'reconstructing' placeholder overlay.

        Called before a reconstruction job is dispatched; falls back to
        "empty" if the job failed and produced no data.  No-op when the
        widget is already showing a real image.

        The loading overlay uses the theme-aware scrim stylesheet
        (opaque panel-bg) so the previous reconstruction image is
        cleanly covered while a new solve is in flight — avoids the
        visual mess of "Reconstructing…" text rendered on top of stale
        conductivity data.
        """
        if on:
            self._overlay_mode = "loading"
            self._empty_overlay.setText(t("hw.reconstruction.loading_overlay"))
            self._empty_overlay.setStyleSheet(loading_scrim_stylesheet())
            self._empty_overlay.show()
        else:
            # If the next update_reconstruction() already painted a
            # result, the overlay is already hidden; leave it alone.
            # Otherwise revert to the empty placeholder so the widget
            # doesn't stay stuck on "Reconstructing…" forever.
            if self._overlay_mode == "loading":
                self._overlay_mode = "empty"
                self._empty_overlay.setText(t("hw.reconstruction.empty_overlay"))
                self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())

    @Slot(object)
    def update_reconstruction(self, result: ReconstructionResult) -> None:
        """Render a new reconstruction result using cached interpolation weights."""
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

        metadata = getattr(result, "metadata", {}) or {}
        n_elec = max(int(metadata.get("n_elec", 16)), 1)
        electrode_coverage = float(metadata.get("electrode_coverage", 0.5))

        mesh_key = self._build_mesh_cache_key(coords, cells, n_elec, electrode_coverage)
        if mesh_key != self._mesh_cache_key:
            self._prepare_static_scene(coords, cells, n_elec, electrode_coverage)
            self._prepare_grid_cache(coords)
            self._mesh_cache_key = mesh_key

        try:
            node_values = self._to_node_values(sigma, cells, coords.shape[0])
        except ValueError as exc:
            self._show_status(str(exc), error=True)
            return
        rgba = self._interpolate_to_rgba(node_values)
        if rgba is None:
            self._show_status("Interpolation cache is not ready", error=True)
            return

        self._image_item.setImage(rgba, autoLevels=False)
        self._image_item.setRect(self._grid_rect)
        self._empty_overlay.hide()

    def configure_layout(
        self, *, n_elec: int, radius: float = 1.0, electrode_coverage: float = 0.5
    ) -> None:
        """Pre-render the static domain, electrodes, and labels before the first frame."""
        self._prepare_domain_outline(
            center=(0.0, 0.0),
            radius=max(float(radius), 1e-6),
            n_elec=max(int(n_elec), 1),
            electrode_coverage=float(electrode_coverage),
        )
        label_radius = max(float(radius), 1e-6) * 1.2
        self._plot_widget.getPlotItem().getViewBox().setRange(
            xRange=(-label_radius, label_radius),
            yRange=(-label_radius, label_radius),
            padding=0.0,
            disableAutoRange=True,
        )

    def clear(self) -> None:
        self._image_item.clear()
        self._overlay_mode = "empty"
        self._empty_overlay.setText(t("hw.reconstruction.empty_overlay"))
        self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())
        self._empty_overlay.show()

    # ── i18n ──

    def _retranslate(self) -> None:
        """Rebuild the plot title HTML and reset the empty-state overlay."""
        self._plot_item_ref.setTitle(
            f'<span style="color:{self._label_color};'
            f"font-family:'{self._font_family}';font-size:14pt;\">"
            f"{t('hw.reconstruction.title')}"
            "</span>"
        )
        # Only rewrite the overlay when showing the idle placeholder or
        # the Phase 4 "Reconstructing…" loading state.  A transient
        # error (from _show_status) keeps its message until the next
        # update_reconstruction() call.
        if self._overlay_mode == "empty":
            self._empty_overlay.setText(t("hw.reconstruction.empty_overlay"))
        elif self._overlay_mode == "loading":
            self._empty_overlay.setText(t("hw.reconstruction.loading_overlay"))

    def _reset_plot(self) -> None:
        self.configure_layout(n_elec=16, radius=1.0)

    def _show_status(self, text: str, *, error: bool) -> None:
        """Replace the current image with a status overlay.

        Error states use an opaque scrim so broken reconstructions
        don't visually mix with any stale data the user ran earlier;
        empty states use a transparent placeholder because the
        underlying image is already cleared.
        """
        self._overlay_mode = "error" if error else "empty"
        self._empty_overlay.setText(text)
        if error:
            self._empty_overlay.setStyleSheet(error_scrim_stylesheet())
        else:
            self._empty_overlay.setStyleSheet(empty_placeholder_stylesheet())
        self._image_item.clear()
        self._empty_overlay.show()

    def _prepare_static_scene(
        self,
        coords: np.ndarray,
        cells: np.ndarray,
        n_elec: int,
        electrode_coverage: float,
    ) -> None:
        xy = coords[:, :2]
        center = xy.mean(axis=0)
        radius = float(np.linalg.norm(xy - center, axis=1).max())
        radius = max(radius, 1e-6)
        self._prepare_domain_outline(tuple(center), radius, n_elec, electrode_coverage)
        label_radius = radius * 1.2
        self._plot_widget.getPlotItem().getViewBox().setRange(
            xRange=(center[0] - label_radius, center[0] + label_radius),
            yRange=(center[1] - label_radius, center[1] + label_radius),
            padding=0.0,
            disableAutoRange=True,
        )

    def _prepare_domain_outline(
        self,
        center: tuple[float, float],
        radius: float,
        n_elec: int,
        electrode_coverage: float,
    ) -> None:
        plot_item = self._plot_widget.getPlotItem()
        theta = np.linspace(0.0, 2.0 * np.pi, 361, dtype=np.float64)
        circle_x = center[0] + radius * np.cos(theta)
        circle_y = center[1] + radius * np.sin(theta)
        self._boundary_item.setData(circle_x, circle_y)

        electrode_theta = np.linspace(
            np.pi / 2.0,
            np.pi / 2.0 + 2.0 * np.pi,
            n_elec,
            endpoint=False,
            dtype=np.float64,
        )
        electrode_pitch = 2.0 * np.pi / max(int(n_elec), 1)
        arc_span = electrode_pitch * min(max(float(electrode_coverage), 1e-6), 1.0)
        arc_radius = radius * 1.02
        label_radius = radius * 1.12
        arc_x: list[float] = []
        arc_y: list[float] = []
        for angle in electrode_theta:
            arc_theta = np.linspace(
                angle - arc_span / 2.0,
                angle + arc_span / 2.0,
                18,
                dtype=np.float64,
            )
            arc_x.extend((center[0] + arc_radius * np.cos(arc_theta)).tolist())
            arc_y.extend((center[1] + arc_radius * np.sin(arc_theta)).tolist())
            arc_x.append(np.nan)
            arc_y.append(np.nan)
        self._electrode_arc_item.setData(np.asarray(arc_x), np.asarray(arc_y))

        for item in self._electrode_label_items:
            plot_item.removeItem(item)
        self._electrode_label_items.clear()

        label_font = QFont(self._font_family, 10)
        for index, angle in enumerate(electrode_theta, start=1):
            label = pg.TextItem(
                text=str(index),
                color=self._label_color,
                anchor=(0.5, 0.5),
            )
            label.setFont(label_font)
            label.setPos(
                center[0] + label_radius * np.cos(angle),
                center[1] + label_radius * np.sin(angle),
            )
            plot_item.addItem(label)
            self._electrode_label_items.append(label)

    def _prepare_grid_cache(self, coords: np.ndarray) -> None:
        xy = coords[:, :2]
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        x_coords = np.linspace(xmin, xmax, self._GRID_SIZE, dtype=np.float64)
        y_coords = np.linspace(ymax, ymin, self._GRID_SIZE, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        sample_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

        delaunay = Delaunay(xy)
        simplex = delaunay.find_simplex(sample_points)
        valid_mask = simplex >= 0
        vertices = np.zeros((sample_points.shape[0], 3), dtype=np.int32)
        weights = np.zeros((sample_points.shape[0], 3), dtype=np.float64)

        if np.any(valid_mask):
            valid_simplex = simplex[valid_mask]
            transforms = delaunay.transform[valid_simplex, :2]
            offsets = sample_points[valid_mask] - delaunay.transform[valid_simplex, 2]
            bary = np.einsum("nij,nj->ni", transforms, offsets)
            weights_valid = np.column_stack((bary, 1.0 - bary.sum(axis=1)))
            vertices[valid_mask] = delaunay.simplices[valid_simplex]
            weights[valid_mask] = weights_valid

        self._grid_vertices = vertices
        self._grid_weights = weights
        self._grid_valid_mask = valid_mask
        self._grid_shape = grid_x.shape
        self._grid_rect = QRectF(
            float(xmin), float(ymax), float(xmax - xmin), float(ymin - ymax)
        )

    def _to_node_values(
        self, sigma: np.ndarray, cells: np.ndarray, n_nodes: int
    ) -> np.ndarray:
        if sigma.size == n_nodes:
            return sigma
        if sigma.size != len(cells):
            raise ValueError(
                f"Conductivity size ({sigma.size}) doesn't match cells ({len(cells)}) or nodes ({n_nodes})"
            )

        node_sum = np.zeros(n_nodes, dtype=np.float64)
        node_hits = np.zeros(n_nodes, dtype=np.float64)
        tri_cells = cells[:, :3]
        for local_index in range(tri_cells.shape[1]):
            node_ids = tri_cells[:, local_index]
            np.add.at(node_sum, node_ids, sigma)
            np.add.at(node_hits, node_ids, 1.0)
        return node_sum / np.maximum(node_hits, 1.0)

    def _interpolate_to_rgba(self, node_values: np.ndarray) -> np.ndarray | None:
        if (
            self._grid_vertices is None
            or self._grid_weights is None
            or self._grid_valid_mask is None
        ):
            return None

        interpolated = np.zeros(self._grid_vertices.shape[0], dtype=np.float64)
        if np.any(self._grid_valid_mask):
            valid_vertices = self._grid_vertices[self._grid_valid_mask]
            valid_weights = self._grid_weights[self._grid_valid_mask]
            interpolated[self._grid_valid_mask] = np.einsum(
                "ij,ij->i",
                node_values[valid_vertices],
                valid_weights,
            )

        valid_values = interpolated[self._grid_valid_mask]
        if valid_values.size == 0:
            return None

        vmax = float(np.nanpercentile(np.abs(valid_values), 98))
        vmax = max(vmax, 1e-9)
        normalized = np.clip((interpolated / vmax + 1.0) * 0.5, 0.0, 1.0)
        rgba = self._colormap.map(normalized, mode="byte").reshape(*self._grid_shape, 4)
        rgba[~self._grid_valid_mask.reshape(self._grid_shape), 3] = 0
        return rgba

    @staticmethod
    def _build_mesh_cache_key(
        coords: np.ndarray,
        cells: np.ndarray,
        n_elec: int,
        electrode_coverage: float,
    ) -> tuple[Any, ...]:
        xy = coords[:, :2]
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        return (
            tuple(coords.shape),
            tuple(cells.shape),
            round(float(xmin), 6),
            round(float(xmax), 6),
            round(float(ymin), 6),
            round(float(ymax), 6),
            int(n_elec),
            round(float(electrode_coverage), 6),
        )
