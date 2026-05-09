"""3D conductivity display with opacity / clipping controls.

Used by ``SimulationResultsWidget`` for 3D tetra / hex volume phantoms — far
smoother through PyVista/VTK when the runtime can safely embed a native
OpenGL window, and still a real PyVista/VTK offscreen 3D view when WSLg
must keep the main Qt window on Wayland for crisp HiDPI text.

PyVista (built on VTK) is the visualisation library that the FEniCSx
project itself ships with — see
https://docs.fenicsproject.org/dolfinx/latest/python/demos/demo_pyvista.html
— so the render path is both first-class for our forward solver's
mesh format and hardware-accelerated.

Two non-obvious design points worth keeping in mind for future
maintenance:

1.  ``QtInteractor`` remains the preferred display path for real 3D
    simulation output on native desktop runtimes.  WSLg defaults to
    PyVista offscreen rendering so the main UI can stay on Wayland and
    avoid XWayland blur; Matplotlib 3D is only the last-resort fallback.

2.  When enabled, ``QtInteractor`` is constructed only after the host
    widget is shown and owns a native child window.  Initialising VTK
    while the host is still parentless / hidden can materialise orphan
    top-level windows or stale X handles.

3.  ``auto_update=False``.  The pyvistaqt default is a 5 Hz background
    render timer, which means VTK redraws the scene 5×/second forever
    even if the user is just hovering over the form.  For our
    use-case nothing changes between user gestures so we drive renders
    explicitly from ``update_image`` / slider callbacks instead.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
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
from eit_app.ui.electrode_overlay import (
    ElectrodeGeometry,
    default_patch_quads,
)
from eit_app.ui.fonts import plot_font_families
from eit_app.ui.theme import (
    plot_palette,
    set_button_role,
    set_hint_text,
    set_section_header,
    subscribe_theme_mode,
)


log = logging.getLogger(__name__)


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_OFFSCREEN_DRAG_FPS_ENV = "EIT_APP_3D_DRAG_FPS"
_OFFSCREEN_DRAG_RENDER_SCALE_ENV = "EIT_APP_3D_DRAG_RENDER_SCALE"
_OFFSCREEN_DRAG_IDLE_MS_ENV = "EIT_APP_3D_DRAG_IDLE_MS"
SUPPORTED_3D_CELL_VERTEX_COUNTS = frozenset({4, 8})
_MPL_FONT_FALLBACKS = ("DejaVu Serif", "DejaVu Sans")
_MPL3D_AX_POSITION = (0.04, 0.08, 0.78, 0.84)
_MPL3D_COLORBAR_POSITION = (0.86, 0.18, 0.035, 0.62)
_INHOMOGENEITY_RELATIVE_FLOOR = 0.02
_CELL_FACE_OFFSETS = {
    4: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
    8: (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ),
}


def _cell_inhomogeneity_mask(cell_sigma: np.ndarray) -> np.ndarray:
    values = np.asarray(cell_sigma, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).any():
        return np.zeros(values.shape, dtype=bool)
    median = float(np.nanmedian(values))
    spread = float(np.nanstd(values))
    floor = max(abs(median) * _INHOMOGENEITY_RELATIVE_FLOOR, 1.0e-6)
    if not np.isfinite(spread) or spread <= floor:
        return np.zeros(values.shape, dtype=bool)
    threshold = max(spread * 0.5, floor)
    return np.asarray(np.abs(values - median) > threshold, dtype=bool)


def _conductivity_color_limits(cell_sigma: np.ndarray) -> tuple[float, float]:
    values = np.asarray(cell_sigma, dtype=np.float64).reshape(-1)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    sigma_min = float(np.nanmin(finite_values))
    sigma_max = float(np.nanmax(finite_values))
    median = float(np.nanmedian(finite_values))
    if not all(np.isfinite(value) for value in (sigma_min, sigma_max, median)):
        return 0.0, 1.0

    floor = max(abs(median) * _INHOMOGENEITY_RELATIVE_FLOOR, 1.0e-6)
    if sigma_max - sigma_min < 2.0 * floor:
        return median - floor, median + floor
    return sigma_min, sigma_max


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_ENV_VALUES


def _env_float(name: str, default: float, *, lower: float, upper: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return min(max(value, lower), upper)


def _env_int(name: str, default: int, *, lower: int, upper: int) -> int:
    value = _env_float(name, float(default), lower=float(lower), upper=float(upper))
    return int(round(value))


def _plot_font_families() -> list[str]:
    families = plot_font_families()
    for family in _MPL_FONT_FALLBACKS:
        if family not in families:
            families.append(family)
    try:
        from matplotlib import font_manager

        known = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        known = set()
    available = [family for family in families if not known or family in known]
    return available or list(_MPL_FONT_FALLBACKS)


def _running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(errors="ignore").lower()
    except OSError:
        return False


def embedded_vtk_status() -> tuple[bool, str]:
    """Decide whether it is safe to embed pyvistaqt's VTK widget.

    The FEniCSx-recommended PyVista path is available on native desktop
    runtimes, but WSLg/X11 has repeatedly crashed the whole process from
    inside VTK's native window setup (``BadWindow / X_ConfigureWindow``).
    That class of failure cannot be caught by Python, so WSL defaults to
    the safe Matplotlib 3D backend unless explicitly forced.
    """
    if _env_flag("EIT_APP_DISABLE_EMBEDDED_VTK"):
        return False, "disabled by EIT_APP_DISABLE_EMBEDDED_VTK"
    if _env_flag("EIT_APP_ENABLE_EMBEDDED_VTK"):
        return True, "forced by EIT_APP_ENABLE_EMBEDDED_VTK"

    qpa = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if qpa in {"offscreen", "minimal"}:
        return False, f"Qt platform is {qpa!r}"

    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return False, "no DISPLAY or WAYLAND_DISPLAY is available"

    if _running_under_wsl():
        if qpa == "xcb":
            return True, "WSLg is using Qt XCB, compatible with vtkXOpenGLRenderWindow"
        return False, "WSLg embedded VTK requires QT_QPA_PLATFORM=xcb"

    return True, "runtime looks compatible"


def embedded_vtk_enabled() -> bool:
    enabled, _reason = embedded_vtk_status()
    return enabled


class _InteractorHost(QFrame):
    """QFrame whose ``realized`` signal fires once after the first
    real ``showEvent`` — i.e. when Qt has actually placed the widget
    in the visible hierarchy.

    Used to defer VTK / pyvistaqt construction until Qt has placed the
    host inside the visible hierarchy.  We deliberately do *not* force
    a native window on this frame: ``QVTKRenderWindowInteractor``
    creates and owns the native child window it passes to VTK, and an
    extra native host layer has proven fragile on WSLg/X11.
    """

    realized = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._fired = False

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        if self._fired:
            return
        self._fired = True
        # Defer to the next event-loop tick so pending layout and Qt
        # surface setup finishes before constructing QVTK.
        QTimer.singleShot(0, self.realized.emit)


class _OffscreenRenderLabel(QLabel):
    """Pixmap canvas for PyVista offscreen frames with basic camera controls."""

    dragged = Signal(float, float)
    zoomed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._last_pos: QPoint | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._last_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            delta = pos - self._last_pos
            self._last_pos = pos
            self.dragged.emit(float(delta.x()), float(delta.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._last_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self.zoomed.emit(float(event.angleDelta().y()))
        event.accept()


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


def _boundary_faces(cells: np.ndarray) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Return boundary faces plus the source volume-cell index for each face."""
    faces: dict[tuple[int, ...], tuple[tuple[int, ...], int] | None] = {}
    offsets_for_cell = _CELL_FACE_OFFSETS.get(int(cells.shape[1]))
    if offsets_for_cell is None:
        return [], np.empty((0,), dtype=np.int64)

    for cell_idx, cell in enumerate(cells):
        for offsets in offsets_for_cell:
            face = tuple(int(cell[offset]) for offset in offsets)
            key = tuple(sorted(face))
            faces[key] = None if key in faces else (face, cell_idx)

    kept = [payload for payload in faces.values() if payload is not None]
    return [face for face, _idx in kept], np.asarray(
        [idx for _face, idx in kept], dtype=np.int64
    )


def _configure_vtk_logging() -> None:
    """Keep harmless VTK warnings (e.g. missing Xcursor) out of GUI logs."""
    try:
        from vtkmodules.vtkCommonCore import vtkLogger

        vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_ERROR)
    except Exception:  # pragma: no cover — best-effort log hygiene
        pass


class Conductivity3DWidget(QWidget):
    """Hardware-accelerated 3D conductivity viewer with transparency controls.

    The widget mirrors ``ConductivityImageWidget``'s public surface
    (``update_image`` / ``clear`` / ``set_loading`` / ``setTitle``) so
    it can be swapped in by the dispatcher in
    ``SimulationResultsWidget``.
    """

    def __init__(
        self, title: str = "Conductivity", parent: QWidget | None = None
    ) -> None:
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
        self._electrode_actor = None
        self._offscreen_electrode_actor = None
        self._mpl3d_electrode_collection = None
        self._electrode_geometry: ElectrodeGeometry | None = None

        self._render_backend = "caption"
        self._last_vtk_disabled_reason: str | None = None
        self._title_font = FontProperties(family=_plot_font_families(), size=14)
        self._mpl3d_colorbar = None
        self._mpl3d_mesh_collection = None
        self._mpl3d_highlight_collection = None
        self._mpl3d_mesh_facecolors: np.ndarray | None = None
        self._mpl3d_highlight_facecolors: np.ndarray | None = None
        # Cached "fresh-render" view bounds for the matplotlib 3D
        # backend so the reset-view button can restore the data extent
        # the user saw on first render, not just the camera angle.
        self._mpl3d_initial_bounds: (
            tuple[
                tuple[float, float],
                tuple[float, float],
                tuple[float, float],
            ]
            | None
        ) = None
        self._offscreen_plotter = None
        self._offscreen_mesh_actor = None
        self._offscreen_highlight_actor = None
        self._offscreen_wire_actor = None
        self._offscreen_window_size: tuple[int, int] | None = None

        self._plotter = None
        self._plotter_ready = False
        # Holds the most recent payload while the plotter is still
        # being built (host hasn't fired its first showEvent yet).  As
        # soon as _init_plotter completes, _drain_pending_render kicks
        # in and renders this.
        self._pending_render: Optional[
            tuple[np.ndarray, np.ndarray, np.ndarray, str | None]
        ] = None

        # Drag throttle: coalesce rapid mouseMove events for the
        # offscreen backend.  Default to a high-performance 60 fps,
        # full-resolution interaction path; users on constrained
        # machines can lower EIT_APP_3D_DRAG_FPS or
        # EIT_APP_3D_DRAG_RENDER_SCALE.
        self._offscreen_drag_fps = _env_float(
            _OFFSCREEN_DRAG_FPS_ENV,
            60.0,
            lower=1.0,
            upper=120.0,
        )
        self._offscreen_drag_interval_ms = max(
            1, int(round(1000.0 / self._offscreen_drag_fps))
        )
        self._offscreen_drag_render_scale = _env_float(
            _OFFSCREEN_DRAG_RENDER_SCALE_ENV,
            1.0,
            lower=0.25,
            upper=1.0,
        )
        self._offscreen_render_timer = QTimer(self)
        self._offscreen_render_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._offscreen_render_timer.setSingleShot(True)
        self._offscreen_render_timer.setInterval(self._offscreen_drag_interval_ms)
        self._offscreen_render_timer.timeout.connect(self._refresh_offscreen_pixmap)
        # Track drag state so optional render-scale downsampling applies
        # only while the user is actively rotating / zooming.
        self._is_dragging_offscreen = False
        self._drag_release_timer = QTimer(self)
        self._drag_release_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._drag_release_timer.setSingleShot(True)
        self._drag_release_timer.setInterval(
            _env_int(_OFFSCREEN_DRAG_IDLE_MS_ENV, 80, lower=16, upper=1000)
        )
        self._drag_release_timer.timeout.connect(self._on_drag_idle)

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
        self._title_label.setMinimumSize(0, 0)
        outer.addWidget(self._title_label)

        self._stack_host = QFrame()
        self._stack = QStackedLayout(self._stack_host)
        self._stack.setContentsMargins(0, 0, 0, 0)

        self._caption_label = QLabel("")
        self._caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption_label.setWordWrap(True)
        # No forced min-height — the slot's outer layout gives it all
        # remaining space via the stretch factor on _stack_host; a
        # fixed 180 px floor dragged the whole main window up when the
        # 3D slot was even present.
        self._caption_label.setMinimumSize(0, 0)
        self._stack.addWidget(self._caption_label)

        self._offscreen_host = QWidget()
        self._offscreen_layout = QVBoxLayout(self._offscreen_host)
        self._offscreen_layout.setContentsMargins(0, 0, 0, 0)
        self._offscreen_label = _OffscreenRenderLabel()
        self._offscreen_label.dragged.connect(self._on_offscreen_dragged)
        self._offscreen_label.zoomed.connect(self._on_offscreen_zoomed)
        self._offscreen_layout.addWidget(self._offscreen_label)
        self._stack.addWidget(self._offscreen_host)

        self._mpl3d_host = QWidget()
        self._mpl3d_layout = QVBoxLayout(self._mpl3d_host)
        self._mpl3d_layout.setContentsMargins(0, 0, 0, 0)
        palette = plot_palette()
        self._mpl3d_figure = Figure(figsize=(5, 4))
        self._mpl3d_figure.patch.set_facecolor(palette["panel_bg"])
        self._mpl3d_canvas = FigureCanvasQTAgg(self._mpl3d_figure)
        self._mpl3d_layout.addWidget(self._mpl3d_canvas)
        self._mpl3d_ax = self._mpl3d_figure.add_axes(
            _MPL3D_AX_POSITION, projection="3d"
        )
        self._mpl3d_colorbar_ax = self._mpl3d_figure.add_axes(_MPL3D_COLORBAR_POSITION)
        self._mpl3d_colorbar_ax.set_visible(False)
        self._stack.addWidget(self._mpl3d_host)

        # _InteractorHost defers its ``realized`` signal until the
        # widget is actually shown for the first time AND Qt has had
        # one event-loop tick to finish realisation — that's the
        # earliest moment we can safely hand the underlying native
        # window to VTK.
        self._interactor_host = _InteractorHost()
        self._interactor_host.realized.connect(self._init_plotter)
        self._interactor_layout = QVBoxLayout(self._interactor_host)
        self._interactor_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._interactor_host)
        self._stack.setCurrentWidget(self._caption_label)

        outer.addWidget(self._stack_host, 1)

        # Control row: opacity slider + visibility toggles + reset.
        #
        # Sizing here is deliberately *soft* — every child uses a
        # shrinkable size policy and no fixed widths.  The old
        # implementation pinned the slider to 120 px and the opacity
        # readout to 36 px, which together pushed the whole simulation
        # tab's minimum width past 1200 px and killed the main
        # window's responsive shrink behaviour.
        self._controls = QFrame()
        self._controls.setObjectName("conductivity3DControls")
        self._controls.setSizePolicy(
            self._controls.sizePolicy().horizontalPolicy(),
            self._controls.sizePolicy().verticalPolicy(),
        )
        bar = QHBoxLayout(self._controls)
        bar.setContentsMargins(6, 2, 6, 2)
        bar.setSpacing(6)

        self._opacity_label = QLabel("")
        set_hint_text(self._opacity_label)
        bar.addWidget(self._opacity_label)

        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(5, 100)
        self._opacity_slider.setValue(45)
        self._opacity_slider.setMinimumWidth(60)
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        bar.addWidget(self._opacity_slider, 1)

        self._opacity_value = QLabel("0.45")
        set_hint_text(self._opacity_value)
        bar.addWidget(self._opacity_value)

        self._highlight_check = QCheckBox("")
        self._highlight_check.setChecked(True)
        self._highlight_check.toggled.connect(self._on_highlight_toggled)
        bar.addWidget(self._highlight_check)

        self._wire_check = QCheckBox("")
        self._wire_check.setChecked(True)
        self._wire_check.toggled.connect(self._on_wire_toggled)
        bar.addWidget(self._wire_check)

        # Electrode overlay toggle.  Disabled until a forward result
        # provides actual electrode geometry — checking it without a
        # cached patch list would render nothing and confuse the user.
        self._electrode_check = QCheckBox("")
        self._electrode_check.setChecked(False)
        self._electrode_check.setEnabled(False)
        self._electrode_check.toggled.connect(self._on_electrode_toggled)
        bar.addWidget(self._electrode_check)

        self._reset_btn = QPushButton("")
        set_button_role(self._reset_btn, "tertiary")
        self._reset_btn.clicked.connect(self._reset_camera)
        bar.addWidget(self._reset_btn)

        outer.addWidget(self._controls)
        # Hidden by default — only shown while the VTK interactor is
        # the active page.  This keeps the bar from contributing to
        # the 2D page's footprint inside the stacked dispatcher.
        self._controls.hide()

    # ------------------------------------------------------------------
    # Eager plotter init
    # ------------------------------------------------------------------

    def _init_plotter(self) -> None:
        """Construct the QtInteractor with auto_update disabled.

        Called exactly once, via ``_InteractorHost.realized`` — i.e.
        after Qt has finished placing the host inside the visible
        widget tree and given it a real native window.  Initialising
        VTK any earlier (in ``__init__`` or before the first show)
        crashes the renderer with ``BadWindow / X_ConfigureWindow``.

        ``auto_update=False`` disables pyvistaqt's 5 Hz background
        render timer; we drive renders explicitly from update_image,
        the slider, and the toggle checkboxes.
        """
        if self._plotter_ready:
            return

        vtk_enabled, reason = embedded_vtk_status()
        if not vtk_enabled:
            log.info("embedded PyVista/VTK viewer disabled: %s", reason)
            if self._pending_render is not None:
                sigma, coords, cells, title = self._pending_render
                self._pending_render = None
                if title is not None:
                    self.setTitle(title)
                self._last_image = (sigma, coords, cells, title)
                if not self._render_pyvista_offscreen_scene(sigma, coords, cells):
                    self._render_matplotlib_scene(sigma, coords, cells)
            else:
                self._show_caption(
                    t("sim.results.viewer3d_embedded_disabled"), kind="placeholder"
                )
            return

        try:
            import pyvista  # noqa: F401  (side-effect: VTK init)
            from pyvistaqt import QtInteractor

            _configure_vtk_logging()
        except Exception as exc:  # pragma: no cover — env without VTK
            log.warning("pyvistaqt unavailable; using safe 3D renderer: %s", exc)
            if self._pending_render is not None:
                sigma, coords, cells, title = self._pending_render
                self._pending_render = None
                if title is not None:
                    self.setTitle(title)
                self._last_image = (sigma, coords, cells, title)
                if not self._render_pyvista_offscreen_scene(sigma, coords, cells):
                    self._render_matplotlib_scene(sigma, coords, cells)
            else:
                self._show_caption(t("sim.results.viewer3d_unavailable"), kind="error")
            return

        palette = plot_palette()
        self._plotter = QtInteractor(
            self._interactor_host,
            off_screen=False,
            auto_update=False,
            multi_samples=4,
        )
        self._plotter.set_background(_hex_to_rgb(palette.get("axes_bg", "#ffffff")))
        self._plotter.add_axes()
        self._interactor_layout.addWidget(self._plotter)
        self._plotter_ready = True

        if self._pending_render is not None:
            sigma, coords, cells, title = self._pending_render
            self._pending_render = None
            if title is not None:
                self.setTitle(title)
            self._last_image = (sigma, coords, cells, title)
            self._build_scene(sigma, coords, cells)
            self._render_backend = "vtk"

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
        """Render a 3D volume conductivity field."""
        cells = np.asarray(cell_connectivity, dtype=np.int64)
        coords = np.asarray(node_coords, dtype=float)
        if (
            coords.ndim != 2
            or cells.ndim != 2
            or coords.shape[1] < 3
            or cells.shape[1] not in SUPPORTED_3D_CELL_VERTEX_COUNTS
        ):
            self._show_caption(t("sim.results.viewer3d_bad_mesh"), kind="error")
            return

        sigma = np.asarray(conductivity, dtype=float)
        if sigma.shape[0] not in (cells.shape[0], coords.shape[0]):
            self._show_caption(t("sim.results.viewer3d_size_mismatch"), kind="error")
            return

        if title is not None:
            self.setTitle(title)
        self._last_image = (sigma, coords, cells, title)

        vtk_enabled, reason = embedded_vtk_status()
        if not vtk_enabled:
            if reason != self._last_vtk_disabled_reason:
                log.info(
                    "embedded PyVistaQt viewer unavailable: %s; trying PyVista offscreen",
                    reason,
                )
                self._last_vtk_disabled_reason = reason
            self._pending_render = None
            self._discard_actors()
            if not self._render_pyvista_offscreen_scene(sigma, coords, cells):
                self._render_matplotlib_scene(sigma, coords, cells)
            return

        # Switching the stacked layout to the interactor host first
        # gets the host into the visible tree.  On the very first
        # switch this triggers _InteractorHost.showEvent, which
        # eventually fires .realized → _init_plotter → builds the
        # scene from _pending_render.  On subsequent calls the
        # plotter is already ready and we render straight away.
        self._render_backend = "vtk"
        self._stack.setCurrentWidget(self._interactor_host)
        self._controls.show()

        if not self._plotter_ready:
            self._pending_render = (sigma, coords, cells, title)
            return

        self._build_scene(sigma, coords, cells)

    def clear(self) -> None:
        """Drop any rendered data and show the placeholder caption."""
        self._discard_actors()
        self._last_image = None
        self._pending_render = None
        self._show_caption(t("sim.results.viewer3d_no_data"), kind="placeholder")

    def set_loading(self, message: str | None = None) -> None:
        """Show a centered loading caption while a forward / inverse solve runs."""
        text = message or t("sim.results.viewer3d_loading")
        self._show_caption(text, kind="loading")

    def set_electrode_geometry(self, geometry: ElectrodeGeometry | None) -> None:
        """Cache electrode geometry and (re)build the cached actor if shown.

        Building is gated on ``geometry.is_3d()`` because the 3D widget
        is only meaningful for cylinder data — a 2D-only ElectrodeGeometry
        passed in (e.g. when the underlying forward solve was 2D and the
        3D widget happens to be alive but not visible) yields no patches
        and leaves the toggle disabled.
        """
        self._electrode_geometry = geometry
        has_patches = bool(geometry and geometry.is_3d() and geometry.patches)
        self._electrode_check.setEnabled(has_patches)
        if not has_patches:
            self._electrode_check.setChecked(False)
            self._remove_all_electrode_actors()
            self._render_active_backend_if_visible()
            return
        # Build (or rebuild) the cached actor on whichever backend is
        # currently active, then honour the user's checkbox state.
        self._refresh_electrode_actors()
        self._render_active_backend_if_visible()

    def _remove_all_electrode_actors(self) -> None:
        if self._plotter is not None and self._electrode_actor is not None:
            try:
                self._plotter.remove_actor(self._electrode_actor, render=False)
            except Exception:  # pragma: no cover — VTK quirk
                pass
        self._electrode_actor = None
        if (
            self._offscreen_plotter is not None
            and self._offscreen_electrode_actor is not None
        ):
            try:
                self._offscreen_plotter.remove_actor(
                    self._offscreen_electrode_actor, render=False
                )
            except Exception:  # pragma: no cover — VTK quirk
                pass
        self._offscreen_electrode_actor = None
        if self._mpl3d_electrode_collection is not None:
            try:
                self._mpl3d_electrode_collection.remove()
            except (AttributeError, ValueError):
                pass
            self._mpl3d_electrode_collection = None

    def _refresh_electrode_actors(self) -> None:
        """Build cached electrode geometry on the currently active backend.

        Other backends defer construction to the next time their scene
        rebuilds via update_image() — there is no point materialising a
        VTK actor on a backend the user is not looking at.
        """
        geometry = self._electrode_geometry
        if geometry is None or not geometry.is_3d() or not geometry.patches:
            return
        if self._render_backend == "vtk":
            self._build_vtk_electrode_actor(geometry)
        elif self._render_backend == "pyvista_offscreen":
            self._build_offscreen_electrode_actor(geometry)
        elif self._render_backend == "mpl3d":
            self._build_mpl3d_electrode_collection(geometry)

    def _render_active_backend_if_visible(self) -> None:
        if self._render_backend == "vtk" and self._plotter is not None:
            self._plotter.render()
        elif self._render_backend == "pyvista_offscreen":
            self._refresh_offscreen_pixmap()
        elif self._render_backend == "mpl3d":
            self._mpl3d_canvas.draw_idle()

    def _build_vtk_electrode_actor(self, geometry: ElectrodeGeometry) -> None:
        if self._plotter is None:
            return
        try:
            import pyvista  # noqa: F401  (side-effect: VTK init)
        except Exception:  # pragma: no cover — env without VTK
            return
        if self._electrode_actor is not None:
            try:
                self._plotter.remove_actor(self._electrode_actor, render=False)
            except Exception:  # pragma: no cover — VTK quirk
                pass
            self._electrode_actor = None
        polydata = self._build_electrode_polydata(geometry)
        if polydata is None:
            return
        palette = plot_palette()
        color = palette.get("highlight", "#f39c12")
        self._electrode_actor = self._plotter.add_mesh(
            polydata,
            color=color,
            opacity=0.9,
            show_edges=False,
            show_scalar_bar=False,
        )
        self._electrode_actor.SetVisibility(bool(self._electrode_check.isChecked()))

    def _build_offscreen_electrode_actor(self, geometry: ElectrodeGeometry) -> None:
        if self._offscreen_plotter is None:
            return
        try:
            import pyvista  # noqa: F401  (side-effect: VTK init)
        except Exception:  # pragma: no cover — env without VTK
            return
        if self._offscreen_electrode_actor is not None:
            try:
                self._offscreen_plotter.remove_actor(
                    self._offscreen_electrode_actor, render=False
                )
            except Exception:  # pragma: no cover — VTK quirk
                pass
            self._offscreen_electrode_actor = None
        polydata = self._build_electrode_polydata(geometry)
        if polydata is None:
            return
        palette = plot_palette()
        color = palette.get("highlight", "#f39c12")
        self._offscreen_electrode_actor = self._offscreen_plotter.add_mesh(
            polydata,
            color=color,
            opacity=0.9,
            show_edges=False,
            show_scalar_bar=False,
        )
        self._offscreen_electrode_actor.SetVisibility(
            bool(self._electrode_check.isChecked())
        )

    def _build_electrode_polydata(self, geometry: ElectrodeGeometry):
        try:
            import pyvista as pv
        except Exception:  # pragma: no cover — env without VTK
            return None
        points, triangles = default_patch_quads(geometry.patches, geometry.radius)
        if points.shape[0] == 0 or triangles.shape[0] == 0:
            return None
        # PyVista expects [3, i0, i1, i2, 3, i0, i1, i2, ...] face buffer.
        face_buffer = np.empty((triangles.shape[0], 4), dtype=np.int64)
        face_buffer[:, 0] = 3
        face_buffer[:, 1:] = triangles
        return pv.PolyData(points, face_buffer.flatten())

    def _build_mpl3d_electrode_collection(self, geometry: ElectrodeGeometry) -> None:
        if self._mpl3d_electrode_collection is not None:
            try:
                self._mpl3d_electrode_collection.remove()
            except (AttributeError, ValueError):
                pass
            self._mpl3d_electrode_collection = None
        if not geometry.is_3d() or not geometry.patches:
            return
        palette = plot_palette()
        color = palette.get("highlight", "#f39c12")
        polys: list[np.ndarray] = []
        for patch in geometry.patches:
            n_theta = 6
            thetas = np.linspace(patch.theta_start, patch.theta_end, n_theta)
            cos_t = np.cos(thetas)
            sin_t = np.sin(thetas)
            lower = np.column_stack(
                (
                    geometry.radius * cos_t,
                    geometry.radius * sin_t,
                    np.full_like(cos_t, patch.z_lower),
                )
            )
            upper = np.column_stack(
                (
                    geometry.radius * cos_t,
                    geometry.radius * sin_t,
                    np.full_like(cos_t, patch.z_upper),
                )
            )
            for i in range(n_theta - 1):
                polys.append(
                    np.array(
                        [lower[i], lower[i + 1], upper[i + 1], upper[i]],
                        dtype=float,
                    )
                )
        if not polys:
            return
        collection = Poly3DCollection(
            polys,
            facecolors=color,
            edgecolors=color,
            linewidths=0.6,
            alpha=0.9,
        )
        self._mpl3d_ax.add_collection3d(collection)
        collection.set_visible(bool(self._electrode_check.isChecked()))
        self._mpl3d_electrode_collection = collection

    def _on_electrode_toggled(self, checked: bool) -> None:
        # Cache hit on every toggle — actors already built by
        # set_electrode_geometry / build_scene; just flip visibility.
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_electrode_actor is None and checked:
                self._build_offscreen_electrode_actor(self._electrode_geometry)
            if self._offscreen_electrode_actor is not None:
                self._offscreen_electrode_actor.SetVisibility(bool(checked))
            self._refresh_offscreen_pixmap()
            return
        if self._render_backend == "mpl3d":
            if (
                self._mpl3d_electrode_collection is None
                and checked
                and self._electrode_geometry is not None
            ):
                self._build_mpl3d_electrode_collection(self._electrode_geometry)
            if self._mpl3d_electrode_collection is not None:
                self._mpl3d_electrode_collection.set_visible(bool(checked))
            self._mpl3d_canvas.draw_idle()
            return
        if self._render_backend == "vtk":
            if (
                self._electrode_actor is None
                and checked
                and self._electrode_geometry is not None
            ):
                self._build_vtk_electrode_actor(self._electrode_geometry)
            if self._electrode_actor is not None:
                self._electrode_actor.SetVisibility(bool(checked))
            if self._plotter is not None:
                self._plotter.render()

    # ------------------------------------------------------------------
    # Scene construction (heavy work — runs once per update_image only)
    # ------------------------------------------------------------------

    def _discard_actors(self) -> None:
        if self._plotter is None:
            return
        for actor_attr in (
            "_mesh_actor",
            "_highlight_actor",
            "_wire_actor",
            "_electrode_actor",
        ):
            actor = getattr(self, actor_attr, None)
            if actor is not None:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:  # pragma: no cover — VTK quirk
                    pass
                setattr(self, actor_attr, None)

    def _discard_offscreen_plotter(self) -> None:
        if self._offscreen_plotter is not None:
            try:
                self._offscreen_plotter.close()
            except Exception:  # pragma: no cover — best-effort VTK cleanup
                pass
        self._offscreen_plotter = None
        self._offscreen_mesh_actor = None
        self._offscreen_highlight_actor = None
        self._offscreen_wire_actor = None
        self._offscreen_electrode_actor = None
        self._offscreen_window_size = None

    def _offscreen_render_size(self) -> tuple[int, int]:
        dpr = max(float(self.devicePixelRatioF()), 1.0)
        # Full-resolution drag is the default for high-performance
        # workstations.  Optional downsampling is still available via
        # EIT_APP_3D_DRAG_RENDER_SCALE for constrained machines.
        scale = (
            self._offscreen_drag_render_scale if self._is_dragging_offscreen else 1.0
        )
        width = max(
            320,
            int(round(max(self._offscreen_label.width(), 1) * dpr * scale)),
        )
        height = max(
            240,
            int(round(max(self._offscreen_label.height(), 1) * dpr * scale)),
        )
        return min(width, 2400), min(height, 1800)

    def _offscreen_pixmap_target(self) -> tuple[int, int, float]:
        """Return physical pixmap size + DPR for stable QLabel display.

        If drag downsampling is enabled, Qt would otherwise treat the
        smaller framebuffer as a smaller logical image and the scene would
        visibly shrink until the final full-resolution frame arrives.
        """
        dpr = max(float(self.devicePixelRatioF()), 1.0)
        label_width = max(float(self._offscreen_label.width()), 1.0)
        label_height = max(float(self._offscreen_label.height()), 1.0)
        width = min(max(1, int(round(label_width * dpr))), 2400)
        height = min(max(1, int(round(label_height * dpr))), 1800)
        return width, height, dpr

    def _refresh_offscreen_pixmap(self) -> None:
        plotter = self._offscreen_plotter
        if plotter is None:
            return
        width, height = self._offscreen_render_size()
        try:
            window_size = (width, height)
            if self._offscreen_window_size != window_size:
                plotter.window_size = window_size
                self._offscreen_window_size = window_size
            plotter.render()
            image = np.ascontiguousarray(plotter.screenshot(return_img=True))
        except Exception as exc:  # pragma: no cover — VTK runtime edge case
            log.warning("PyVista offscreen render failed: %s", exc)
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
        target_width, target_height, target_dpr = self._offscreen_pixmap_target()
        if pixmap.width() != target_width or pixmap.height() != target_height:
            pixmap = pixmap.scaled(
                target_width,
                target_height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )
        pixmap.setDevicePixelRatio(target_dpr)
        self._offscreen_label.setPixmap(pixmap)

    def _render_pyvista_offscreen_scene(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> bool:
        try:
            import pyvista as pv

            _configure_vtk_logging()
        except Exception as exc:
            log.info("PyVista offscreen renderer unavailable: %s", exc)
            return False

        self._discard_offscreen_plotter()

        n_cells = cells.shape[0]
        if sigma.shape[0] == n_cells:
            cell_sigma = sigma
            scalar_kw = {"scalars": "sigma", "preference": "cell"}
            scalar_mode = "cell"
        else:
            cell_sigma = sigma[cells].mean(axis=1)
            scalar_kw = {"scalars": "sigma", "preference": "point"}
            scalar_mode = "point"

        verts_per_cell = cells.shape[1]
        if verts_per_cell == 4:
            cell_type = pv.CellType.TETRA
        elif verts_per_cell == 8:
            cell_type = pv.CellType.HEXAHEDRON
        else:
            return False

        cell_array = np.empty((n_cells, verts_per_cell + 1), dtype=np.int64)
        cell_array[:, 0] = verts_per_cell
        cell_array[:, 1:] = cells
        cell_types = np.full(n_cells, cell_type, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cell_array.flatten(), cell_types, coords)
        if scalar_mode == "cell":
            grid.cell_data["sigma"] = cell_sigma
        else:
            grid.point_data["sigma"] = sigma

        sigma_min, sigma_max = _conductivity_color_limits(cell_sigma)

        palette = plot_palette()
        width, height = self._offscreen_render_size()
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.set_background(_hex_to_rgb(palette.get("axes_bg", "#ffffff")))
        plotter.add_axes()
        self._offscreen_plotter = plotter
        self._offscreen_window_size = (width, height)

        opacity = self._opacity_slider.value() / 100.0
        text_color = _hex_to_rgb(palette.get("text", "#222"))
        self._offscreen_mesh_actor = plotter.add_mesh(
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
                "position_y": 0.12,
                "width": 0.06,
                "height": 0.55,
                "title_font_size": 14,
                "label_font_size": 10,
            },
            **scalar_kw,
        )

        if scalar_mode == "cell":
            inhom_mask = _cell_inhomogeneity_mask(cell_sigma)
            if np.any(inhom_mask):
                inhom_grid = grid.extract_cells(np.where(inhom_mask)[0])
                if inhom_grid.n_cells > 0:
                    self._offscreen_highlight_actor = plotter.add_mesh(
                        inhom_grid,
                        scalars="sigma",
                        preference="cell",
                        cmap="viridis",
                        clim=[sigma_min, sigma_max],
                        opacity=1.0,
                        show_edges=False,
                        show_scalar_bar=False,
                    )
                    self._offscreen_highlight_actor.SetVisibility(
                        bool(self._highlight_check.isChecked())
                    )

        outline = grid.extract_surface(
            algorithm="dataset_surface"
        ).extract_feature_edges(
            boundary_edges=True,
            feature_edges=True,
            feature_angle=30.0,
            non_manifold_edges=False,
            manifold_edges=False,
        )
        if outline.n_points > 0:
            self._offscreen_wire_actor = plotter.add_mesh(
                outline,
                color=palette.get("border", "#888"),
                line_width=1.0,
                opacity=0.45,
                show_scalar_bar=False,
            )
            self._offscreen_wire_actor.SetVisibility(bool(self._wire_check.isChecked()))

        plotter.reset_camera()
        self._stack.setCurrentWidget(self._offscreen_host)
        self._controls.show()
        self._render_backend = "pyvista_offscreen"
        # Re-attach cached electrode patches if a forward result already
        # supplied geometry — discard_offscreen_plotter above wiped the
        # actor pointer.
        if self._electrode_geometry is not None:
            self._build_offscreen_electrode_actor(self._electrode_geometry)
        self._refresh_offscreen_pixmap()
        return True

    def _remove_mpl3d_colorbar(self) -> None:
        self._mpl3d_colorbar_ax.clear()
        self._mpl3d_colorbar_ax.set_visible(False)
        self._mpl3d_colorbar = None

    def _apply_mpl3d_chrome(self) -> None:
        palette = plot_palette()
        text = palette.get("text", "#222")
        border = palette.get("border", "#888")
        axes_bg = palette.get("axes_bg", "#ffffff")
        self._mpl3d_figure.patch.set_facecolor(palette.get("panel_bg", "#ffffff"))
        self._mpl3d_ax.set_position(_MPL3D_AX_POSITION)
        self._mpl3d_colorbar_ax.set_position(_MPL3D_COLORBAR_POSITION)
        self._mpl3d_ax.set_facecolor(axes_bg)
        self._mpl3d_ax.tick_params(colors=text, labelsize=8)
        for axis in (self._mpl3d_ax.xaxis, self._mpl3d_ax.yaxis, self._mpl3d_ax.zaxis):
            axis.label.set_color(text)
            axis.label.set_fontfamily(_plot_font_families())
            axis.pane.set_facecolor((*_hex_to_rgb(axes_bg), 0.18))
            axis.pane.set_edgecolor(border)
        for label in (
            self._mpl3d_ax.get_xticklabels()
            + self._mpl3d_ax.get_yticklabels()
            + self._mpl3d_ax.get_zticklabels()
        ):
            label.set_color(text)
            label.set_fontfamily(_plot_font_families())

    def _render_matplotlib_scene(
        self,
        sigma: np.ndarray,
        coords: np.ndarray,
        cells: np.ndarray,
    ) -> None:
        """Render the volume as a real 3D Matplotlib scene.

        This is intentionally *not* the old XY projection.  It draws
        boundary faces in a 3D Axes and, for cell-centered inhomogeneous
        fields, also draws the internal anomalous cells through the
        transparent shell so the result remains spatially inspectable
        without touching VTK's native window layer.
        """
        self._remove_mpl3d_colorbar()
        elev = getattr(self._mpl3d_ax, "elev", 22.0)
        azim = getattr(self._mpl3d_ax, "azim", -45.0)
        self._mpl3d_ax.clear()
        self._apply_mpl3d_chrome()

        faces, source_cells = _boundary_faces(cells)
        valid_face_payload: list[tuple[tuple[int, ...], int]] = []
        for face, source_cell in zip(faces, source_cells, strict=False):
            if all(0 <= idx < len(coords) for idx in face):
                valid_face_payload.append((face, int(source_cell)))
        if not valid_face_payload:
            self._show_caption(t("sim.results.viewer3d_bad_mesh"), kind="error")
            return

        n_cells = cells.shape[0]
        if sigma.shape[0] == n_cells:
            scalar_mode = "cell"
            cell_sigma = sigma.astype(float, copy=False)
            face_values = np.asarray(
                [cell_sigma[source_cell] for _face, source_cell in valid_face_payload],
                dtype=float,
            )
        else:
            scalar_mode = "point"
            cell_sigma = sigma[cells].mean(axis=1)
            face_values = np.asarray(
                [
                    float(np.nanmean(sigma[np.asarray(face, dtype=np.int64)]))
                    for face, _source_cell in valid_face_payload
                ],
                dtype=float,
            )

        sigma_min, sigma_max = _conductivity_color_limits(face_values)

        palette = plot_palette()
        cmap = colormaps["viridis"]
        norm = Normalize(vmin=sigma_min, vmax=sigma_max)
        opacity = self._opacity_slider.value() / 100.0
        face_vertices = [
            coords[np.asarray(face, dtype=np.int64), :3]
            for face, _ in valid_face_payload
        ]
        colors = cmap(norm(face_values))
        colors[:, 3] = opacity
        edge_color = (
            palette.get("border", "#888") if self._wire_check.isChecked() else "none"
        )

        mesh = Poly3DCollection(
            face_vertices,
            facecolors=colors,
            edgecolors=edge_color,
            linewidths=0.35,
            alpha=None,
        )
        self._mpl3d_ax.add_collection3d(mesh)
        self._mpl3d_mesh_collection = mesh
        self._mpl3d_mesh_facecolors = colors.copy()
        self._mpl3d_highlight_collection = None
        self._mpl3d_highlight_facecolors = None

        if scalar_mode == "cell":
            inhom_indices = np.flatnonzero(_cell_inhomogeneity_mask(cell_sigma))
            if inhom_indices.size:
                highlight_vertices: list[np.ndarray] = []
                highlight_values: list[float] = []
                offsets_for_cell = _CELL_FACE_OFFSETS.get(int(cells.shape[1]), ())
                for cell_idx in inhom_indices:
                    cell = cells[int(cell_idx)]
                    for offsets in offsets_for_cell:
                        face = tuple(int(cell[offset]) for offset in offsets)
                        if all(0 <= idx < len(coords) for idx in face):
                            highlight_vertices.append(coords[np.asarray(face), :3])
                            highlight_values.append(float(cell_sigma[int(cell_idx)]))
                if highlight_vertices:
                    highlight_colors = cmap(norm(np.asarray(highlight_values)))
                    highlight_colors[:, 3] = max(0.82, opacity)
                    highlight = Poly3DCollection(
                        highlight_vertices,
                        facecolors=highlight_colors,
                        edgecolors=palette["highlight"],
                        linewidths=0.45,
                        alpha=None,
                    )
                    self._mpl3d_ax.add_collection3d(highlight)
                    highlight.set_visible(bool(self._highlight_check.isChecked()))
                    self._mpl3d_highlight_collection = highlight
                    self._mpl3d_highlight_facecolors = highlight_colors.copy()

        points = coords[:, :3]
        mins = np.nanmin(points, axis=0)
        maxs = np.nanmax(points, axis=0)
        spans = np.maximum(maxs - mins, 1.0e-9)
        center = (mins + maxs) * 0.5
        radius = float(np.nanmax(spans) * 0.56)
        if radius <= 0.0 or not np.isfinite(radius):
            radius = 1.0
        xlim = (center[0] - radius, center[0] + radius)
        ylim = (center[1] - radius, center[1] + radius)
        zlim = (center[2] - radius, center[2] + radius)
        self._mpl3d_ax.set_xlim(*xlim)
        self._mpl3d_ax.set_ylim(*ylim)
        self._mpl3d_ax.set_zlim(*zlim)
        # Cache so the reset-view button can restore the original
        # extent — view_init alone only resets the camera angle,
        # not the user's pan / zoom.
        self._mpl3d_initial_bounds = (xlim, ylim, zlim)
        try:
            self._mpl3d_ax.set_box_aspect(tuple(spans))
        except Exception:  # pragma: no cover — older matplotlib fallback
            pass
        self._mpl3d_ax.view_init(elev=elev, azim=azim)

        title = self._default_title
        self._mpl3d_ax.set_title(
            title,
            fontproperties=self._title_font,
            color=palette.get("text", "#222"),
            pad=8,
        )
        self._mpl3d_ax.set_xlabel("X")
        self._mpl3d_ax.set_ylabel("Y")
        self._mpl3d_ax.set_zlabel("Z")

        scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array(face_values)
        self._mpl3d_colorbar_ax.clear()
        self._mpl3d_colorbar_ax.set_visible(True)
        self._mpl3d_colorbar = self._mpl3d_figure.colorbar(
            scalar_mappable,
            cax=self._mpl3d_colorbar_ax,
        )
        self._mpl3d_colorbar.set_label("S/m", color=palette.get("text", "#222"))
        self._mpl3d_colorbar.ax.tick_params(colors=palette.get("text", "#222"))

        self._stack.setCurrentWidget(self._mpl3d_host)
        self._controls.show()
        self._render_backend = "mpl3d"
        # Re-attach cached electrode patches if available — ax.clear()
        # earlier wiped the previous Poly3DCollection.
        self._mpl3d_electrode_collection = None
        if self._electrode_geometry is not None:
            self._build_mpl3d_electrode_collection(self._electrode_geometry)
        self._mpl3d_canvas.draw_idle()

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

        # Build the unstructured volume grid.  VTK expects
        # [n_pts, p0, p1, ...] rows; support both tetra meshes from the
        # CPU path and hex meshes from the CUDA-structured path.
        verts_per_cell = cells.shape[1]
        if verts_per_cell == 4:
            cell_type = pv.CellType.TETRA
        elif verts_per_cell == 8:
            cell_type = pv.CellType.HEXAHEDRON
        else:  # update_image() guards this; keep a defensive fallback.
            self._show_caption(t("sim.results.viewer3d_bad_mesh"), kind="error")
            return

        cell_array = np.empty((n_cells, verts_per_cell + 1), dtype=np.int64)
        cell_array[:, 0] = verts_per_cell
        cell_array[:, 1:] = cells
        cell_types = np.full(n_cells, cell_type, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cell_array.flatten(), cell_types, coords)
        if scalar_mode == "cell":
            grid.cell_data["sigma"] = cell_sigma
        else:
            grid.point_data["sigma"] = sigma

        sigma_min, sigma_max = _conductivity_color_limits(cell_sigma)

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
            inhom_mask = _cell_inhomogeneity_mask(cell_sigma)
            if np.any(inhom_mask):
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
        outline = grid.extract_surface(
            algorithm="dataset_surface"
        ).extract_feature_edges(
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
            self._wire_actor.SetVisibility(bool(self._wire_check.isChecked()))

        # Re-attach electrode patch actor if we already have geometry —
        # _discard_actors() above wiped the cached pointer.
        if self._electrode_geometry is not None:
            self._build_vtk_electrode_actor(self._electrode_geometry)

        plotter.reset_camera()
        plotter.render()

    # ------------------------------------------------------------------
    # Interactive controls — actor mutation only, never a full rebuild
    # ------------------------------------------------------------------

    def _apply_mpl3d_opacity(self, opacity: float) -> None:
        if (
            self._mpl3d_mesh_collection is not None
            and self._mpl3d_mesh_facecolors is not None
        ):
            self._mpl3d_mesh_facecolors[:, 3] = opacity
            self._mpl3d_mesh_collection.set_facecolors(self._mpl3d_mesh_facecolors)
        if (
            self._mpl3d_highlight_collection is not None
            and self._mpl3d_highlight_facecolors is not None
        ):
            self._mpl3d_highlight_facecolors[:, 3] = max(0.82, opacity)
            self._mpl3d_highlight_collection.set_facecolors(
                self._mpl3d_highlight_facecolors
            )

    def _apply_mpl3d_wire_visibility(self, checked: bool) -> None:
        palette = plot_palette()
        edge_color = palette.get("border", "#888") if checked else "none"
        if self._mpl3d_mesh_collection is not None:
            self._mpl3d_mesh_collection.set_edgecolor(edge_color)
        if self._mpl3d_highlight_collection is not None:
            highlight_edge = palette["highlight"] if checked else "none"
            self._mpl3d_highlight_collection.set_edgecolor(highlight_edge)

    def _on_opacity_changed(self, value: int) -> None:
        opacity = value / 100.0
        self._opacity_value.setText(f"{opacity:.2f}")
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_mesh_actor is not None:
                self._offscreen_mesh_actor.GetProperty().SetOpacity(opacity)
            self._refresh_offscreen_pixmap()
            return
        if self._render_backend == "mpl3d" and self._last_image is not None:
            self._apply_mpl3d_opacity(opacity)
            self._mpl3d_canvas.draw_idle()
            return
        if self._mesh_actor is None or self._plotter is None:
            return
        self._mesh_actor.GetProperty().SetOpacity(opacity)
        self._plotter.render()

    def _on_highlight_toggled(self, checked: bool) -> None:
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_highlight_actor is not None:
                self._offscreen_highlight_actor.SetVisibility(bool(checked))
            self._refresh_offscreen_pixmap()
            return
        if self._render_backend == "mpl3d" and self._last_image is not None:
            if self._mpl3d_highlight_collection is not None:
                self._mpl3d_highlight_collection.set_visible(bool(checked))
            self._mpl3d_canvas.draw_idle()
            return
        if self._highlight_actor is None or self._plotter is None:
            return
        self._highlight_actor.SetVisibility(bool(checked))
        self._plotter.render()

    def _on_wire_toggled(self, checked: bool) -> None:
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_wire_actor is not None:
                self._offscreen_wire_actor.SetVisibility(bool(checked))
            self._refresh_offscreen_pixmap()
            return
        if self._render_backend == "mpl3d" and self._last_image is not None:
            self._apply_mpl3d_wire_visibility(bool(checked))
            self._mpl3d_canvas.draw_idle()
            return
        if self._wire_actor is None or self._plotter is None:
            return
        self._wire_actor.SetVisibility(bool(checked))
        self._plotter.render()

    def _reset_camera(self) -> None:
        """Reset both data extent AND viewing angle across all backends.

        Each backend stores its camera state differently — this used
        to silently no-op for users on the matplotlib 3D fallback (the
        old code only called view_init but left their pan / zoom in
        place) and only half-reset for PyVista (reset_camera fits the
        actors but does not restore an isometric orientation).  Now
        every backend restores both the data range and a known
        canonical view direction.
        """
        if self._render_backend == "pyvista_offscreen":
            if self._offscreen_plotter is not None:
                self._offscreen_plotter.reset_camera()
                # Bring the orientation back to isometric so a click
                # always lands on the same canonical view, no matter
                # how the user has dragged the scene.
                try:
                    self._offscreen_plotter.view_isometric()
                except Exception:  # pragma: no cover — VTK quirk
                    pass
                self._refresh_offscreen_pixmap()
            return
        if self._render_backend == "mpl3d":
            self._mpl3d_ax.view_init(elev=22.0, azim=-45.0)
            if self._mpl3d_initial_bounds is not None:
                xlim, ylim, zlim = self._mpl3d_initial_bounds
                self._mpl3d_ax.set_xlim(*xlim)
                self._mpl3d_ax.set_ylim(*ylim)
                self._mpl3d_ax.set_zlim(*zlim)
            self._mpl3d_canvas.draw_idle()
            return
        if self._plotter is not None:
            self._plotter.reset_camera()
            try:
                self._plotter.view_isometric()
            except Exception:  # pragma: no cover — VTK quirk
                pass
            self._plotter.render()

    def _on_offscreen_dragged(self, dx: float, dy: float) -> None:
        if (
            self._render_backend != "pyvista_offscreen"
            or self._offscreen_plotter is None
        ):
            return
        camera = self._offscreen_plotter.camera
        camera.Azimuth(-dx * 0.45)
        camera.Elevation(dy * 0.45)
        camera.OrthogonalizeViewUp()
        # Mark drag-active so the optional render-scale profile applies;
        # reset shortly after the user pauses so any downsampled profile
        # gets one final full-resolution frame.
        self._is_dragging_offscreen = True
        self._drag_release_timer.start()
        self._schedule_offscreen_refresh()

    def _on_offscreen_zoomed(self, delta_y: float) -> None:
        if (
            self._render_backend != "pyvista_offscreen"
            or self._offscreen_plotter is None
        ):
            return
        self._offscreen_plotter.camera.Zoom(1.12 if delta_y > 0 else 0.89)
        self._is_dragging_offscreen = True
        self._drag_release_timer.start()
        self._schedule_offscreen_refresh()

    def _schedule_offscreen_refresh(self) -> None:
        """Coalesce rapid drag events into the configured refresh rate."""
        if not self._offscreen_render_timer.isActive():
            self._offscreen_render_timer.start()

    def _on_drag_idle(self) -> None:
        """User stopped dragging — render one final crisp frame."""
        if self._render_backend != "pyvista_offscreen":
            self._is_dragging_offscreen = False
            return
        self._is_dragging_offscreen = False
        self._refresh_offscreen_pixmap()

    # ------------------------------------------------------------------
    # Caption / theme handling
    # ------------------------------------------------------------------

    def _show_caption(self, text: str, *, kind: str) -> None:
        self._render_backend = "caption"
        self._discard_offscreen_plotter()
        self._remove_mpl3d_colorbar()
        self._mpl3d_mesh_collection = None
        self._mpl3d_highlight_collection = None
        self._mpl3d_mesh_facecolors = None
        self._mpl3d_highlight_facecolors = None
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
        # Hide the controls bar whenever the interactor isn't the
        # active page.  The controls only make sense against a live
        # VTK scene, and hiding them removes their contribution to
        # the widget's minimum-size floor.
        self._controls.hide()

    def _on_theme_mode_changed(self, _mode: str) -> None:
        if self._render_backend == "pyvista_offscreen" and self._last_image is not None:
            self._render_pyvista_offscreen_scene(*self._last_image[:3])
            return
        if self._render_backend == "mpl3d" and self._last_image is not None:
            self._render_matplotlib_scene(*self._last_image[:3])
            return
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
        self._electrode_check.setText(t("sim.results.electrodes_toggle"))
        self._reset_btn.setText(t("sim.results.viewer3d_reset"))
        if self._last_image is None and not self._caption_label.text():
            self._show_caption(t("sim.results.viewer3d_no_data"), kind="placeholder")

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
        self._discard_offscreen_plotter()
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:  # pragma: no cover — best-effort shutdown
                pass
            self._plotter = None
            self._plotter_ready = False
        super().closeEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        if self._render_backend == "pyvista_offscreen":
            QTimer.singleShot(0, self._refresh_offscreen_pixmap)
