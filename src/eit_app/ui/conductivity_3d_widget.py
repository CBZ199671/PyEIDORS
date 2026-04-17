"""PyVista (VTK) 3D conductivity display with opacity / clipping controls.

Used by ``SimulationResultsWidget`` for 3D tetrahedral phantoms — far
smoother than matplotlib's mpl_toolkits.mplot3d under interactive
rotation, and exposes interior cells via a transparency slider so
the user can see internal inclusions through the bulk.

PyVista (built on VTK) is the visualisation library that the FEniCSx
project itself ships with — see
https://docs.fenicsproject.org/dolfinx/latest/python/demos/demo_pyvista.html
— so we get a render path that's both first-class for our forward
solver's mesh format and hardware-accelerated.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from eit_app.i18n import t, translator
from eit_app.ui.theme import (
    plot_palette,
    set_hint_text,
    set_section_header,
    subscribe_theme_mode,
)


log = logging.getLogger(__name__)


def _hex_to_rgb(value: str) -> tuple[float, float, float]:
    """Parse a CSS-style ``#rrggbb`` colour into 0–1 floats for VTK."""
    text = value.strip().lstrip("#")
    if len(text) != 6:
        return (1.0, 1.0, 1.0)
    return (
        int(text[0:2], 16) / 255.0,
        int(text[2:4], 16) / 255.0,
        int(text[4:6], 16) / 255.0,
    )


class Conductivity3DWidget(QWidget):
    """Hardware-accelerated 3D conductivity viewer with transparency / clipping.

    The widget mirrors ``ConductivityImageWidget``'s public surface
    (``update_image`` / ``clear`` / ``set_loading`` / ``setTitle``) so it
    can be swapped in by the dispatcher in ``SimulationResultsWidget``.
    """

    def __init__(self, title: str = "Conductivity", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._default_title = title
        self._last_image: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray, str | None]
        ] = None
        self._mesh_actor = None
        self._wire_actor = None
        # Defer the heavy QtInteractor + VTK import until we actually
        # need it.  On import-only test environments the class is
        # instantiated without ever rendering, and pulling VTK at module
        # import time would slow every Python import in eit_app.
        self._plotter = None
        self._build_ui()
        translator().language_changed.connect(self._retranslate)
        self._retranslate()
        subscribe_theme_mode(self._on_theme_mode_changed)

    # ------------------------------------------------------------------
    # UI assembly
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._title_label = QLabel(self._default_title)
        set_section_header(self._title_label)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("padding: 4px 0;")
        outer.addWidget(self._title_label)

        # Stacked area: pyvista interactor on top, caption (placeholder /
        # loading / error) when no data is available.
        self._stack_host = QFrame()
        self._stack = QStackedLayout(self._stack_host)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._caption_label = QLabel("")
        self._caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption_label.setWordWrap(True)
        self._caption_label.setMinimumHeight(180)
        self._stack.addWidget(self._caption_label)

        # Lazily created interactor — only spun up when 3D data first
        # arrives so widget construction doesn't pull VTK / start an
        # OpenGL context up front.
        self._interactor_host = QFrame()
        self._interactor_layout = QVBoxLayout(self._interactor_host)
        self._interactor_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._interactor_host)
        self._stack.setCurrentWidget(self._caption_label)

        outer.addWidget(self._stack_host, 1)

        # Control row: opacity, edge toggle, reset view.
        controls = QFrame()
        controls.setObjectName("conductivity3DControls")
        bar = QHBoxLayout(controls)
        bar.setContentsMargins(8, 4, 8, 4)
        bar.setSpacing(8)

        self._opacity_label = QLabel("")
        set_hint_text(self._opacity_label)
        bar.addWidget(self._opacity_label)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(5, 100)
        self._opacity_slider.setValue(45)
        self._opacity_slider.setFixedWidth(120)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        bar.addWidget(self._opacity_slider)

        self._opacity_value = QLabel("0.45")
        set_hint_text(self._opacity_value)
        self._opacity_value.setMinimumWidth(36)
        bar.addWidget(self._opacity_value)

        bar.addSpacing(12)

        self._highlight_check = QCheckBox("")
        self._highlight_check.setChecked(True)
        self._highlight_check.toggled.connect(self._refresh_render)
        bar.addWidget(self._highlight_check)

        self._wire_check = QCheckBox("")
        self._wire_check.setChecked(True)
        self._wire_check.toggled.connect(self._refresh_render)
        bar.addWidget(self._wire_check)

        bar.addStretch()

        self._reset_btn = QPushButton("")
        self._reset_btn.clicked.connect(self._reset_camera)
        bar.addWidget(self._reset_btn)

        outer.addWidget(controls)

    # ------------------------------------------------------------------
    # Lazy plotter setup
    # ------------------------------------------------------------------

    def _ensure_plotter(self) -> bool:
        """Create the pyvistaqt interactor on first use; return False on import error."""
        if self._plotter is not None:
            return True
        try:
            import pyvista  # noqa: F401  (side-effect: VTK init + warmup)
            from pyvistaqt import QtInteractor
        except Exception as exc:  # pragma: no cover — env without VTK
            log.warning("pyvistaqt unavailable, 3D widget falls back to caption: %s", exc)
            self._show_caption(
                t("sim.results.viewer3d_unavailable"), kind="error"
            )
            return False

        palette = plot_palette()
        self._plotter = QtInteractor(self._interactor_host)
        bg = _hex_to_rgb(palette.get("axes_bg", "#ffffff"))
        self._plotter.set_background(bg)
        self._plotter.add_axes()
        self._interactor_layout.addWidget(self._plotter.interactor)
        return True

    # ------------------------------------------------------------------
    # Public API (mirrors ConductivityImageWidget)
    # ------------------------------------------------------------------

    def setTitle(self, title: str) -> None:
        self._default_title = title
        self._title_label.setText(title)

    def update_image(
        self,
        conductivity: np.ndarray,
        node_coords: np.ndarray,
        cell_connectivity: np.ndarray,
        title: str | None = None,
    ) -> None:
        """Render a 3D tetrahedral conductivity field."""
        if not self._ensure_plotter():
            return

        cells = np.asarray(cell_connectivity, dtype=np.int64)
        coords = np.asarray(node_coords, dtype=float)
        if coords.shape[1] < 3 or cells.shape[1] != 4:
            self._show_caption(
                t("sim.results.viewer3d_bad_mesh"), kind="error"
            )
            return

        sigma = np.asarray(conductivity, dtype=float)
        if sigma.shape[0] not in (cells.shape[0], coords.shape[0]):
            self._show_caption(
                t("sim.results.viewer3d_size_mismatch"), kind="error"
            )
            return

        if title is not None:
            self.setTitle(title)

        self._last_image = (sigma, coords, cells, title)
        self._render_pyvista_mesh(sigma, coords, cells)
        self._stack.setCurrentWidget(self._interactor_host)

    def clear(self) -> None:
        """Drop any rendered data and show the placeholder caption."""
        if self._plotter is not None:
            self._plotter.clear()
        self._mesh_actor = None
        self._wire_actor = None
        self._last_image = None
        self._show_caption(t("sim.results.viewer3d_no_data"), kind="placeholder")

    def set_loading(self, message: str | None = None) -> None:
        """Show a centered loading caption while a forward / inverse solve runs."""
        text = message or t("sim.results.viewer3d_loading")
        self._show_caption(text, kind="loading")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_pyvista_mesh(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> None:
        import pyvista as pv

        plotter = self._plotter
        plotter.clear()

        n_cells = cells.shape[0]
        if sigma.shape[0] == n_cells:
            cell_sigma = sigma
            scalar_mode = "cell"
        else:
            # Node-centered field — VTK takes that natively.
            cell_sigma = sigma[cells].mean(axis=1)
            scalar_mode = "point"

        # Build the unstructured tet grid.  VTK expects [n_pts, p0, p1, …]
        # rows so prepend a column of 4s.
        cell_array = np.empty((n_cells, 5), dtype=np.int64)
        cell_array[:, 0] = 4
        cell_array[:, 1:] = cells
        cell_types = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cell_array.flatten(), cell_types, coords)
        if scalar_mode == "cell":
            grid.cell_data["sigma"] = cell_sigma
            scalars_kw = "sigma"
        else:
            grid.point_data["sigma"] = sigma
            scalars_kw = "sigma"

        opacity = self._opacity_slider.value() / 100.0
        palette = plot_palette()
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

        # Bulk volume — translucent so you can see through to interior
        # cells whose conductivity differs from background.
        self._mesh_actor = plotter.add_mesh(
            grid,
            scalars=scalars_kw,
            cmap="viridis",
            opacity=opacity,
            show_edges=False,
            preference="cell" if scalar_mode == "cell" else "point",
            scalar_bar_args=scalar_bar_args,
        )

        # Highlight overlay: pull out cells whose conductivity is far
        # from the median (= the "inhomogeneity") and re-render them
        # opaque on top.  This is what makes inner inclusions readable
        # even when the bulk opacity is high — without it, a small
        # inclusion at the centre is washed out by alpha-blending.
        if self._highlight_check.isChecked() and scalar_mode == "cell":
            median = float(np.nanmedian(cell_sigma))
            spread = float(np.nanstd(cell_sigma))
            if spread > 1.0e-6:
                threshold = max(spread * 0.5, 1.0e-6)
                inhom_mask = np.abs(cell_sigma - median) > threshold
                if np.any(inhom_mask):
                    inhom_grid = grid.extract_cells(np.where(inhom_mask)[0])
                    if inhom_grid.n_cells > 0:
                        plotter.add_mesh(
                            inhom_grid,
                            scalars=scalars_kw,
                            cmap="viridis",
                            opacity=1.0,
                            show_edges=False,
                            show_scalar_bar=False,
                            preference="cell",
                        )

        # Optional wireframe outline so the bulk shape stays legible
        # when opacity is low.  Slim, neutral colour, no fill.
        if self._wire_check.isChecked():
            outline = grid.extract_surface().extract_feature_edges(
                boundary_edges=True,
                feature_edges=True,
                feature_angle=30.0,
                non_manifold_edges=False,
                manifold_edges=False,
            )
            if outline.n_points > 0:
                self._wire_actor = plotter.add_mesh(
                    outline,
                    color=palette.get("border", "#888"),
                    line_width=1.0,
                    opacity=0.4,
                )

        plotter.reset_camera()
        plotter.render()

    def _refresh_render(self) -> None:
        if self._last_image is None:
            return
        sigma, coords, cells, _ = self._last_image
        self._render_pyvista_mesh(sigma, coords, cells)

    def _on_opacity_changed(self, value: int) -> None:
        opacity = value / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        if self._mesh_actor is None:
            return
        try:
            self._mesh_actor.GetProperty().SetOpacity(opacity)
            if self._plotter is not None:
                self._plotter.render()
        except Exception:  # pragma: no cover — VTK quirk
            self._refresh_render()

    def _reset_camera(self) -> None:
        if self._plotter is not None:
            self._plotter.reset_camera()
            self._plotter.render()

    # ------------------------------------------------------------------
    # Caption / theme handling
    # ------------------------------------------------------------------

    def _show_caption(self, text: str, *, kind: str) -> None:
        palette = plot_palette()
        color = {
            "placeholder": palette.get("caption", "#888"),
            "loading": palette.get("caption_loading", "#1f5d8b"),
            "error": palette.get("caption_error", "#c0392b"),
        }.get(kind, palette.get("caption", "#888"))
        self._caption_label.setText(text)
        self._caption_label.setStyleSheet(
            f"color: {color}; font-size: 13px; padding: 36px;"
        )
        self._stack.setCurrentWidget(self._caption_label)

    def _on_theme_mode_changed(self, _mode: str) -> None:
        if self._plotter is not None:
            palette = plot_palette()
            self._plotter.set_background(_hex_to_rgb(palette.get("axes_bg", "#fff")))
            if self._last_image is not None:
                self._refresh_render()
        if self._last_image is None and self._caption_label.text():
            # Re-apply caption color through the active palette.
            current_text = self._caption_label.text()
            self._show_caption(current_text, kind="placeholder")

    # ------------------------------------------------------------------
    # i18n
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        self._opacity_label.setText(t("sim.results.viewer3d_opacity"))
        self._highlight_check.setText(t("sim.results.viewer3d_highlight"))
        self._wire_check.setText(t("sim.results.viewer3d_wireframe"))
        self._reset_btn.setText(t("sim.results.viewer3d_reset"))
        if self._last_image is None:
            current = self._caption_label.text()
            if not current:
                self._show_caption(
                    t("sim.results.viewer3d_no_data"), kind="placeholder"
                )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:  # pragma: no cover — best-effort shutdown
                pass
        super().closeEvent(event)
