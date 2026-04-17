"""PyVista (VTK) 3D conductivity display with opacity / clipping controls.

Used by ``SimulationResultsWidget`` for 3D tetrahedral phantoms — far
smoother than matplotlib's mpl_toolkits.mplot3d under interactive
rotation, and exposes interior cells via a transparency slider so
the user can see internal inclusions through the bulk.

PyVista (built on VTK) is the visualisation library that the FEniCSx
project itself ships with — see
https://docs.fenicsproject.org/dolfinx/latest/python/demos/demo_pyvista.html
— so the render path is both first-class for our forward solver's
mesh format and hardware-accelerated.

Two non-obvious design points worth keeping in mind for future
maintenance:

1.  ``QtInteractor`` is constructed **eagerly** (not lazily on first
    ``update_image``).  Lazy creation deferred VTK setup until a
    ``QStackedLayout`` switch made the host widget visible, but at
    that moment the host had no native X11 window assigned — VTK then
    asked X to configure a window that didn't exist and the process
    died with ``BadWindow / X_ConfigureWindow``.  Eager construction
    plus ``WA_NativeWindow`` on the host guarantees a real X window
    before VTK touches it.

2.  ``auto_update=False``.  The pyvistaqt default is a 5 Hz background
    render timer, which means VTK redraws the scene 5×/second forever
    even if the user is just hovering over the form.  For our
    use-case nothing changes between user gestures so we drive renders
    explicitly from ``update_image`` / slider callbacks instead.
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
    """Hardware-accelerated 3D conductivity viewer with transparency controls.

    The widget mirrors ``ConductivityImageWidget``'s public surface
    (``update_image`` / ``clear`` / ``set_loading`` / ``setTitle``) so
    it can be swapped in by the dispatcher in
    ``SimulationResultsWidget``.
    """

    def __init__(self, title: str = "Conductivity", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._default_title = title

        # Last data + cached actors.  Actors are created once per
        # update_image and toggled via SetVisibility for fast
        # checkbox / slider response without re-extracting cells or
        # recomputing edges.
        self._last_image: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray, str | None]
        ] = None
        self._mesh_actor = None
        self._highlight_actor = None
        self._wire_actor = None

        self._plotter = None
        self._plotter_ready = False

        self._build_ui()
        self._init_plotter()
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

        self._stack_host = QFrame()
        self._stack = QStackedLayout(self._stack_host)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._caption_label = QLabel("")
        self._caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption_label.setWordWrap(True)
        self._caption_label.setMinimumHeight(180)
        self._stack.addWidget(self._caption_label)

        self._interactor_host = QFrame()
        # Force allocation of a real X11 / native window for the host
        # *before* anyone (especially VTK) tries to query it.  Without
        # this, QStackedLayout keeps the host unrealised while the
        # caption is showing — VTK then tries to attach to a window
        # that the X server doesn't know about and the process dies
        # with ``BadWindow / X_ConfigureWindow``.
        self._interactor_host.setAttribute(
            Qt.WidgetAttribute.WA_NativeWindow, True
        )
        self._interactor_host.setAttribute(
            Qt.WidgetAttribute.WA_DontCreateNativeAncestors, False
        )
        self._interactor_layout = QVBoxLayout(self._interactor_host)
        self._interactor_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._interactor_host)
        self._stack.setCurrentWidget(self._caption_label)

        outer.addWidget(self._stack_host, 1)

        # Control row: opacity slider + visibility toggles + reset.
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
        self._highlight_check.toggled.connect(self._on_highlight_toggled)
        bar.addWidget(self._highlight_check)

        self._wire_check = QCheckBox("")
        self._wire_check.setChecked(True)
        self._wire_check.toggled.connect(self._on_wire_toggled)
        bar.addWidget(self._wire_check)

        bar.addStretch()

        self._reset_btn = QPushButton("")
        self._reset_btn.clicked.connect(self._reset_camera)
        bar.addWidget(self._reset_btn)

        outer.addWidget(controls)

    # ------------------------------------------------------------------
    # Eager plotter init
    # ------------------------------------------------------------------

    def _init_plotter(self) -> None:
        """Construct the QtInteractor up front, with auto_update disabled.

        Eager construction (rather than lazy on first 3D payload) fixes
        the X11 BadWindow crash that happened when VTK touched a not-
        yet-realised QStackedLayout child.  Disabling ``auto_update``
        gets rid of the 5 Hz background render timer that pyvistaqt
        otherwise installs, which was burning CPU between user gestures.
        """
        try:
            import pyvista  # noqa: F401  (side-effect: VTK init)
            from pyvistaqt import QtInteractor
        except Exception as exc:  # pragma: no cover — env without VTK
            log.warning("pyvistaqt unavailable, 3D widget falls back to caption: %s", exc)
            self._show_caption(t("sim.results.viewer3d_unavailable"), kind="error")
            return

        # Make absolutely sure the host has a native window before we
        # hand it to VTK.  winId() forces the platform plug-in to
        # allocate the underlying handle now rather than at first show.
        _ = self._interactor_host.winId()

        palette = plot_palette()
        self._plotter = QtInteractor(
            self._interactor_host,
            auto_update=False,  # no 5 Hz background timer
            multi_samples=4,
        )
        self._plotter.set_background(_hex_to_rgb(palette.get("axes_bg", "#ffffff")))
        self._plotter.add_axes()
        self._interactor_layout.addWidget(self._plotter)
        self._plotter_ready = True

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
        if not self._plotter_ready:
            return

        cells = np.asarray(cell_connectivity, dtype=np.int64)
        coords = np.asarray(node_coords, dtype=float)
        if coords.shape[1] < 3 or cells.shape[1] != 4:
            self._show_caption(t("sim.results.viewer3d_bad_mesh"), kind="error")
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
        self._build_scene(sigma, coords, cells)
        self._stack.setCurrentWidget(self._interactor_host)

    def clear(self) -> None:
        """Drop any rendered data and show the placeholder caption."""
        self._discard_actors()
        self._last_image = None
        self._show_caption(t("sim.results.viewer3d_no_data"), kind="placeholder")

    def set_loading(self, message: str | None = None) -> None:
        """Show a centered loading caption while a forward / inverse solve runs."""
        text = message or t("sim.results.viewer3d_loading")
        self._show_caption(text, kind="loading")

    # ------------------------------------------------------------------
    # Scene construction (heavy work — runs once per update_image only)
    # ------------------------------------------------------------------

    def _discard_actors(self) -> None:
        if self._plotter is None:
            return
        for actor_attr in ("_mesh_actor", "_highlight_actor", "_wire_actor"):
            actor = getattr(self, actor_attr, None)
            if actor is not None:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:  # pragma: no cover — VTK quirk
                    pass
                setattr(self, actor_attr, None)

    def _build_scene(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> None:
        """Build the bulk + highlight + wireframe actors *once* per
        update_image and cache them.  Slider / checkbox interactions
        then mutate actor properties (opacity, visibility) without
        having to rebuild the grid or recompute feature edges — that
        is what keeps interactive response below 16 ms per gesture.
        """
        import pyvista as pv

        plotter = self._plotter
        self._discard_actors()

        n_cells = cells.shape[0]
        if sigma.shape[0] == n_cells:
            cell_sigma = sigma
            scalar_kw = {"scalars": "sigma", "preference": "cell"}
            scalar_mode = "cell"
        else:
            # Node-centered sigma — VTK takes that natively.
            cell_sigma = sigma[cells].mean(axis=1)
            scalar_kw = {"scalars": "sigma", "preference": "point"}
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
        else:
            grid.point_data["sigma"] = sigma

        sigma_min = float(np.nanmin(cell_sigma))
        sigma_max = float(np.nanmax(cell_sigma))
        if not np.isfinite(sigma_min) or not np.isfinite(sigma_max):
            sigma_min, sigma_max = 0.0, 1.0
        if sigma_max - sigma_min < 1.0e-12:
            sigma_max = sigma_min + 1.0e-12

        opacity = self._opacity_slider.value() / 100.0
        palette = plot_palette()
        text_color = _hex_to_rgb(palette.get("text", "#222"))

        # Bulk volume: alpha-blended so we can see through to interior
        # cells whose conductivity differs from background.
        self._mesh_actor = plotter.add_mesh(
            grid,
            cmap="viridis",
            opacity=opacity,
            clim=[sigma_min, sigma_max],
            show_edges=False,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "S/m",
                "color": text_color,
                "vertical": True,
                "position_x": 0.88,
                "position_y": 0.05,
                "width": 0.07,
                "height": 0.6,
                "title_font_size": 14,
                "label_font_size": 11,
            },
            **scalar_kw,
        )

        # Highlight overlay: cells whose conductivity is far from the
        # median (i.e. the "inhomogeneity") rendered opaque so a small
        # central inclusion still reads even when the bulk opacity is
        # high.  Built always; visibility toggles with the checkbox.
        if scalar_mode == "cell":
            median = float(np.nanmedian(cell_sigma))
            spread = float(np.nanstd(cell_sigma))
            threshold = max(spread * 0.5, 1.0e-6)
            inhom_mask = np.abs(cell_sigma - median) > threshold
            if spread > 1.0e-6 and np.any(inhom_mask):
                inhom_grid = grid.extract_cells(np.where(inhom_mask)[0])
                if inhom_grid.n_cells > 0:
                    self._highlight_actor = plotter.add_mesh(
                        inhom_grid,
                        scalars="sigma",
                        preference="cell",
                        cmap="viridis",
                        clim=[sigma_min, sigma_max],
                        opacity=1.0,
                        show_edges=False,
                        show_scalar_bar=False,
                    )
                    self._highlight_actor.SetVisibility(
                        bool(self._highlight_check.isChecked())
                    )

        # Wireframe overlay: feature edges of the boundary surface,
        # gives the bulk shape a clean silhouette at low opacity.
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
                show_scalar_bar=False,
            )
            self._wire_actor.SetVisibility(
                bool(self._wire_check.isChecked())
            )

        plotter.reset_camera()
        plotter.render()

    # ------------------------------------------------------------------
    # Interactive controls — actor mutation only, never a full rebuild
    # ------------------------------------------------------------------

    def _on_opacity_changed(self, value: int) -> None:
        opacity = value / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        if self._mesh_actor is None or self._plotter is None:
            return
        self._mesh_actor.GetProperty().SetOpacity(opacity)
        self._plotter.render()

    def _on_highlight_toggled(self, checked: bool) -> None:
        if self._highlight_actor is None or self._plotter is None:
            return
        self._highlight_actor.SetVisibility(bool(checked))
        self._plotter.render()

    def _on_wire_toggled(self, checked: bool) -> None:
        if self._wire_actor is None or self._plotter is None:
            return
        self._wire_actor.SetVisibility(bool(checked))
        self._plotter.render()

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
                # Cheap rebuild: same data, picks up new border / text
                # colours on the wire actor + scalar bar.
                self._build_scene(*self._last_image[:3])
        if self._last_image is None and self._caption_label.text():
            self._show_caption(self._caption_label.text(), kind="placeholder")

    # ------------------------------------------------------------------
    # i18n
    # ------------------------------------------------------------------

    def _retranslate(self) -> None:
        self._opacity_label.setText(t("sim.results.viewer3d_opacity"))
        self._highlight_check.setText(t("sim.results.viewer3d_highlight"))
        self._wire_check.setText(t("sim.results.viewer3d_wireframe"))
        self._reset_btn.setText(t("sim.results.viewer3d_reset"))
        if self._last_image is None and not self._caption_label.text():
            self._show_caption(
                t("sim.results.viewer3d_no_data"), kind="placeholder"
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Tear down the VTK render thread cleanly.

        Without this, the Python interpreter exits while VTK's render
        timer / signal queue is still alive and Qt prints
        ``QThreadStorage: entry destroyed before end of thread``.
        """
        self._discard_actors()
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:  # pragma: no cover — best-effort shutdown
                pass
            self._plotter = None
            self._plotter_ready = False
        super().closeEvent(event)
