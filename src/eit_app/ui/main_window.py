"""Main application window with tab-based layout."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget, QWidget

from eit_app.acquisition.acquisition_process import AcquisitionProcess
from eit_app.acquisition.ring_buffer import FrameRingBuffer
from eit_app.controllers.acquisition_controller import AcquisitionController
from eit_app.controllers.device_controller import DeviceController
from eit_app.controllers.reconstruction_controller import (
    ReconstructionController,
    ReconstructionRequest,
    get_single_step_cached_cache_key,
)
from eit_app.controllers.dataset_generator_controller import (
    DatasetGeneratorController,
    DatasetGeneratorRequest,
)
from eit_app.controllers.forward_solver_controller import (
    ForwardSolverController,
    ForwardSolverRequest,
    ForwardSolverResult,
)
from eit_app.controllers.batch_reconstruction_controller import (
    BatchReconstructionController,
    BatchReconstructionRequest,
)
from eit_app.controllers.database_controller import DatabaseController
from eit_app.controllers.recording_controller import RecordingController
from eit_app.hardware.connection_preflight import preflight_connection_target
from eit_app.hardware.factory import create_device_from_config, normalize_device_config
from eit_app.hardware.types import STIM_AMP_VALUES_UA, VOLTAGE_AMP_LABELS
from eit_app.i18n import current_language, set_language, t, translator
from eit_app.interop import (
    EidorsExportJob,
    EidorsScriptCaptureService,
    InteropBundleExporter,
    InteropBundleImporter,
    InteropSmokeValidator,
    ReconstructionPreset,
    build_geometry_payload_from_result,
)
from eit_app.measurement_layout import (
    measurement_layout_from_config,
)
from eit_app.models.app_state import (
    AcquisitionMode,
    AppState,
    ConnectionStatus,
    PowerStatus,
    RecordingStatus,
)
from eit_app.models.forward_model_config import (
    ForwardModelConfig,
    INTERACTIVE_3D_DEFAULT_HEIGHT,
    INTERACTIVE_3D_DEFAULT_RADIUS,
    electrode_level_fractions_for_rings,
)
from eit_app.models.simulation_state import (
    DatasetGeneratorConfig,
    SimulationState,
)
from eit_app.ui.database.database_tab import DatabaseTab
from eit_app.ui.hardware.hardware_tab import HardwareTab
from eit_app.ui.simulation.dataset_generator_tab import DatasetGeneratorTab
from eit_app.ui.simulation.simulation_tab import SimulationTab
from eit_app.ui.status_bar import EITStatusBar
from eit_app.models.precision import current_precision, set_precision
from eit_app.ui.theme import current_theme_mode, set_theme_mode

from eit_app.models.frame_model import FrameData

log = logging.getLogger(__name__)


def _is_wsl() -> bool:
    """Detect whether we are running inside a WSL distribution."""
    import os
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def _open_with_explorer_exe(folder_path: str) -> bool:
    """Launch Windows Explorer on a WSL / Windows path. Returns True on success."""
    import shutil
    import subprocess

    if shutil.which("explorer.exe") is None:
        return False
    try:
        # If we have wslpath, convert the POSIX path to a Windows UNC/local path
        if shutil.which("wslpath") is not None:
            try:
                win_path = subprocess.check_output(
                    ["wslpath", "-w", folder_path],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if win_path:
                    subprocess.Popen(["explorer.exe", win_path])
                    return True
            except Exception as exc:
                log.debug("wslpath conversion failed: %s", exc)
        # Fallback: let explorer.exe handle it as-is (works for Windows drive paths)
        subprocess.Popen(["explorer.exe", folder_path])
        return True
    except Exception as exc:
        log.debug("explorer.exe launch failed: %s", exc)
        return False


def _open_folder_in_file_manager(folder: str) -> bool:
    """Open *folder* in the native file manager. Returns True on success.

    Handles three environments:

    - **Windows**: ``os.startfile`` (then ``explorer.exe`` as a backup)
    - **macOS**: ``open``
    - **Pure Linux**: ``xdg-open`` → ``gio open`` → ``dbus-send`` → QDesktopServices
    - **WSL**: prefer ``explorer.exe`` (always available; converts path with
      ``wslpath -w``) → ``wslview`` → then normal Linux openers as last resort

    On WSL we intentionally skip ``xdg-open`` / ``gio open`` as the primary
    choice because they commonly fail when no Linux desktop is installed
    (the typical WSL case), producing the confusing
    "Failed to find default application for content type 'inode/directory'".
    """
    import os
    import shutil
    import subprocess
    import sys

    if not folder:
        return False
    folder_path = os.path.expanduser(str(folder))

    # ---- Windows native ----
    if sys.platform == "win32":
        try:
            os.startfile(folder_path)  # type: ignore[attr-defined]
            return True
        except Exception as exc:
            log.debug("os.startfile failed: %s", exc)
        try:
            subprocess.Popen(["explorer.exe", folder_path])
            return True
        except Exception as exc:
            log.debug("explorer.exe failed on win32: %s", exc)
        return _qt_open_url(folder_path)

    # ---- macOS ----
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", folder_path])
            return True
        except Exception as exc:
            log.debug("open (darwin) failed: %s", exc)
        return _qt_open_url(folder_path)

    # ---- WSL: go straight to explorer.exe (most reliable) ----
    if _is_wsl():
        if _open_with_explorer_exe(folder_path):
            return True
        if shutil.which("wslview") is not None:
            try:
                subprocess.Popen(["wslview", folder_path])
                return True
            except Exception as exc:
                log.debug("wslview failed: %s", exc)
        # Fall through to pure-Linux openers (may also fail but worth trying)

    # ---- Pure Linux ----
    for opener in ("xdg-open", "gio", "kde-open5", "kioclient5", "dolphin", "nautilus"):
        if shutil.which(opener) is None:
            continue
        try:
            if opener == "gio":
                subprocess.Popen(
                    ["gio", "open", folder_path],
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                )
            else:
                subprocess.Popen(
                    [opener, folder_path],
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                )
            return True
        except Exception as exc:
            log.debug("%s failed: %s", opener, exc)

    # ---- Qt cross-platform last resort ----
    return _qt_open_url(folder_path)


def _qt_open_url(folder_path: str) -> bool:
    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        url = QUrl.fromLocalFile(folder_path)
        if QDesktopServices.openUrl(url):
            return True
    except Exception as exc:
        log.debug("QDesktopServices failed: %s", exc)
    return False

def _voltage_gain_label(level: int) -> str:
    if 0 <= level < len(VOLTAGE_AMP_LABELS):
        return VOLTAGE_AMP_LABELS[level]
    return "?"


def _with_interactive_3d_geometry_defaults(
    config: ForwardModelConfig,
    *,
    enabled: bool,
) -> ForwardModelConfig:
    """Apply GUI-friendly 3D geometry only for the built-in interactive setup."""
    if not enabled or int(config.mesh_dimension) != 3:
        return config
    return config.with_overrides(
        radius=INTERACTIVE_3D_DEFAULT_RADIUS,
        height=INTERACTIVE_3D_DEFAULT_HEIGHT,
    )


class EITWorkstation(QMainWindow):
    """Main window for the EIT Workstation application."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Window title is set via _retranslate() so it follows the active
        # language (see end of __init__ for the signal wiring).
        #
        # Initial size caps to 90 % of the primary screen's available
        # area so the window never opens wider than the screen it's
        # launched on, AND the preferred size itself is 1280 × 800 —
        # tight enough that the splitter contents (which now total
        # ~1080 px wide by default) fully fit without anything being
        # clipped or pushed off-screen.  Larger preferred values
        # used to overflow 1366-px laptops or push the rightmost
        # context panel past the visible area.
        self.resize(self._preferred_initial_size(1280, 800))

        self._state = AppState(self)
        self._sim_state = SimulationState(self)
        self._device_ctrl = DeviceController(self)
        self._acq_ctrl = AcquisitionController(self)
        self._rec_ctrl = RecordingController(self)
        self._recon_ctrl = ReconstructionController(self)
        self._recon_prewarm_ctrl = ReconstructionController(self)
        self._hw_recon_ctrl = ReconstructionController(self)
        self._db_recon_ctrl = ReconstructionController(self)
        self._sim_recon_ctrl = ReconstructionController(self)
        self._db_ctrl = DatabaseController(self._default_db_path(), self)
        self._rec_ctrl.set_database_controller(self._db_ctrl)
        self._batch_recon_ctrl = BatchReconstructionController(self)
        self._batch_dialog = None  # lazily created
        # Phase 6: Difference dialog is modeless — retain a single
        # instance reference so Python GC doesn't collect the dialog
        # the moment _open_difference_dialog() returns, and so repeat
        # Tools→Difference clicks raise the existing window instead
        # of stacking a new one.
        self._difference_dialog = None
        self._fwd_ctrl = ForwardSolverController(self)
        self._dataset_ctrl = DatasetGeneratorController(self)
        self._last_fwd_result: ForwardSolverResult | None = None
        self._interop_capture_service = EidorsScriptCaptureService()
        self._interop_importer = InteropBundleImporter()
        self._interop_exporter = InteropBundleExporter()
        self._interop_smoke_validator = InteropSmokeValidator()
        self._sim_forward_model_config = ForwardModelConfig()
        self._dataset_forward_model_config = ForwardModelConfig()
        self._sim_use_interactive_3d_geometry_defaults = True
        self._dataset_use_interactive_3d_geometry_defaults = True
        self._interop_geometry_asset: dict | None = None
        self._interop_measurements_asset: dict[str, np.ndarray] | None = None
        self._last_imported_bundle = None

        self._transport_type = "serial"
        self._device_config = normalize_device_config("serial", {})
        self._ring_buffer: FrameRingBuffer | None = None
        self._acq_process: AcquisitionProcess | None = None
        self._scheduled_enabled = False
        self._scheduled_interval_sec = 5.0
        self._planned_acquisition_count = 0
        self._frequency_stepping_enabled = False
        self._planned_start_hz = int(self._device_config.get("frequency_hz", 1000))
        self._planned_end_hz = int(self._device_config.get("frequency_hz", 1000))
        self._plan_timer = QTimer(self)
        self._plan_timer.setSingleShot(True)
        self._plan_timer.timeout.connect(self._run_next_planned_acquisition)
        self._recon_prewarm_timer = QTimer(self)
        self._recon_prewarm_timer.setSingleShot(True)
        self._recon_prewarm_timer.timeout.connect(self._run_realtime_recon_prewarm)
        self._plan_active = False
        self._plan_completed_count = 0
        self._plan_frequencies: list[int] = []
        self._planned_step_pending = False
        self._latest_frame_timestamp = 0.0
        self._selected_reference_entry: dict | None = None
        self._selected_target_entry: dict | None = None
        self._record_requested = False
        self._single_frame_pending = False
        self._pending_power_commands: list[bool] = []

        # Auto-reconstruction pipeline state
        self._auto_reconstruct = False
        self._reference_frame: FrameData | None = None
        self._auto_recon_busy = False
        self._last_auto_ref_frame: FrameData | None = None
        self._last_auto_tgt_frame: FrameData | None = None
        self._recon_prewarm_busy = False
        self._recon_prewarm_active_signature: tuple[object, ...] | None = None
        self._recon_prewarm_requested_signature: tuple[object, ...] | None = None
        self._recon_prewarm_ready_signature: tuple[object, ...] | None = None

        # Database-driven reconstruction state
        self._pending_db_reconstruction: dict | None = None
        self._pending_auto_target_frame: FrameData | None = None

        self._build_ui()
        self._acq_panel.set_output_dir(self._default_output_dir())
        self._connect_signals()
        self._sync_sim_inhomogeneity_context()
        self._control_panel.set_enabled(False)
        self._refresh_expected_measurement_counts()
        self._refresh_session_summary()

        # Wire runtime language switching for chrome owned by this window.
        # Child widgets handle their own retranslation in later phases by
        # subscribing to translator().language_changed themselves.
        translator().language_changed.connect(self._on_language_changed)
        self._retranslate()

        # Kick off DB backfill shortly after startup so the UI shows
        # historical sessions without blocking window initialization.
        QTimer.singleShot(500, self._trigger_backfill)

    # --- Convenience accessors that delegate to the hardware tab ---

    @property
    def _conn_panel(self):
        return self._hw_tab.connection_panel

    @property
    def _control_panel(self):
        return self._hw_tab.control_panel

    @property
    def _acq_panel(self):
        return self._hw_tab.acquisition_panel

    @property
    def _summary_panel(self):
        return self._hw_tab.summary_panel

    @property
    def _workflow_toolbox(self):
        return self._hw_tab.workflow_toolbox

    @property
    def _live_plot(self):
        return self._hw_tab.live_plot

    @property
    def _recon_widget(self):
        return self._hw_tab.reconstruction_widget

    @property
    def _frame_browser(self):
        return self._hw_tab.frame_browser

    @property
    def _voltage_plot(self):
        return self._hw_tab.voltage_plot

    def _build_ui(self) -> None:
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        self._tab_widget.setDocumentMode(True)
        self.setCentralWidget(self._tab_widget)

        # Tab titles are assigned by _retranslate() so they follow the
        # active language.
        self._hw_tab = HardwareTab()
        self._tab_widget.addTab(self._hw_tab, "")

        self._sim_tab = SimulationTab()
        self._tab_widget.addTab(self._sim_tab, "")

        self._dataset_tab = DatasetGeneratorTab()
        self._tab_widget.addTab(self._dataset_tab, "")

        # Database tab — persistent archive of all recorded sessions.
        self._db_tab = DatabaseTab(self._db_ctrl)
        self._tab_widget.addTab(self._db_tab, "")

        self._status_bar = EITStatusBar(self)
        self.setStatusBar(self._status_bar)

        self._build_menus()

    def _build_menus(self) -> None:
        """Create the main menu bar and retain references for retranslation."""
        menu_bar = self.menuBar()

        # File menu --------------------------------------------------------
        self._menu_file = menu_bar.addMenu("")
        self._action_exit = self._menu_file.addAction("")
        # Ctrl+Q is the established quit binding across Linux DEs and
        # Windows Qt apps.  Set it explicitly — StandardKey.Quit returns
        # an empty sequence on some Qt builds (offscreen platform) and
        # leaves the action with no shortcut at all.
        self._action_exit.setShortcut(QKeySequence("Ctrl+Q"))
        self._action_exit.triggered.connect(self.close)

        # View menu --------------------------------------------------------
        # Light / Dark mode toggle.  The checked state reflects the
        # persisted preference (QSettings) so the initial paint matches
        # whatever the user last chose.
        self._menu_view = menu_bar.addMenu("")
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)

        self._action_theme_light = self._menu_view.addAction("")
        self._action_theme_light.setCheckable(True)
        self._action_theme_light.triggered.connect(
            lambda: self._on_theme_mode_selected("light")
        )
        self._theme_action_group.addAction(self._action_theme_light)

        self._action_theme_dark = self._menu_view.addAction("")
        self._action_theme_dark.setCheckable(True)
        self._action_theme_dark.triggered.connect(
            lambda: self._on_theme_mode_selected("dark")
        )
        self._theme_action_group.addAction(self._action_theme_dark)

        # Compute precision submenu — float32 trades a few decimals of
        # precision for ~half the memory + faster vectorised math, and
        # the ADC delivers ~7 effective bits, so float32 already covers
        # the input range with headroom to spare.
        self._menu_view.addSeparator()
        self._menu_precision = self._menu_view.addMenu("")
        self._precision_action_group = QActionGroup(self)
        self._precision_action_group.setExclusive(True)

        self._action_precision_float32 = self._menu_precision.addAction("")
        self._action_precision_float32.setCheckable(True)
        self._action_precision_float32.triggered.connect(
            lambda: self._on_precision_selected("float32")
        )
        self._precision_action_group.addAction(self._action_precision_float32)

        self._action_precision_float64 = self._menu_precision.addAction("")
        self._action_precision_float64.setCheckable(True)
        self._action_precision_float64.triggered.connect(
            lambda: self._on_precision_selected("float64")
        )
        self._precision_action_group.addAction(self._action_precision_float64)

        # Tools menu -------------------------------------------------------
        self._menu_tools = menu_bar.addMenu("")
        self._action_interop_hub = self._menu_tools.addAction("")
        # Ctrl+I — I as in Interop.  Not used elsewhere in the app.
        self._action_interop_hub.setShortcut(QKeySequence("Ctrl+I"))
        self._action_interop_hub.triggered.connect(self._open_interop_hub)
        self._menu_tools.addSeparator()

        # Reconstruction workflow entries — each opens the corresponding
        # dialog from a single, discoverable menu location rather than
        # hiding behind a button deep inside a tab.  Keyboard shortcuts
        # mirror the first letter of each dialog's name so they read
        # naturally: D for Difference, B for Batch, R for Reconstruct.
        self._action_difference = self._menu_tools.addAction("")
        self._action_difference.setShortcut(QKeySequence("Ctrl+D"))
        self._action_difference.triggered.connect(self._open_difference_dialog)

        self._action_batch_reconstruction = self._menu_tools.addAction("")
        self._action_batch_reconstruction.setShortcut(QKeySequence("Ctrl+B"))
        self._action_batch_reconstruction.triggered.connect(
            self._open_batch_reconstruction_from_menu
        )

        self._action_reconstruction = self._menu_tools.addAction("")
        self._action_reconstruction.setShortcut(QKeySequence("Ctrl+R"))
        self._action_reconstruction.triggered.connect(
            self._open_reconstruction_from_menu
        )

        # Language menu ----------------------------------------------------
        self._menu_language = menu_bar.addMenu("")
        self._lang_action_group = QActionGroup(self)
        self._lang_action_group.setExclusive(True)

        self._action_lang_zh = self._menu_language.addAction("")
        self._action_lang_zh.setCheckable(True)
        self._action_lang_zh.triggered.connect(lambda: set_language("zh"))
        self._lang_action_group.addAction(self._action_lang_zh)

        self._action_lang_en = self._menu_language.addAction("")
        self._action_lang_en.setCheckable(True)
        self._action_lang_en.triggered.connect(lambda: set_language("en"))
        self._lang_action_group.addAction(self._action_lang_en)

        # Tab-switching shortcuts — browser-style Ctrl+1..4 jump to the
        # Hardware / Simulation / Dataset / Database tabs respectively.
        # Registered as QShortcut on the main window so they fire even
        # when the menu bar isn't focused.
        self._tab_shortcuts: list[QShortcut] = []
        for index in range(self._tab_widget.count()):
            sc = QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self)
            sc.activated.connect(
                lambda idx=index: self._tab_widget.setCurrentIndex(idx)
            )
            self._tab_shortcuts.append(sc)

        # Simulation run shortcuts — F5 triggers the forward solve,
        # Ctrl+Enter triggers the inverse reconstruction.  Both are
        # gated to only fire when the Simulation tab is active, so the
        # user's F5 inside (say) the Database tab doesn't accidentally
        # kick off a solve on stale data.
        self._sim_forward_shortcut = QShortcut(QKeySequence("F5"), self)
        self._sim_forward_shortcut.activated.connect(self._sim_shortcut_run_forward)
        self._sim_inverse_shortcut_enter = QShortcut(
            QKeySequence("Ctrl+Return"), self
        )
        self._sim_inverse_shortcut_enter.activated.connect(
            self._sim_shortcut_run_inverse
        )
        # Alternate binding because some keyboards label the numpad-return
        # as Enter rather than Return, and Qt treats them distinctly.
        self._sim_inverse_shortcut_numpad = QShortcut(
            QKeySequence("Ctrl+Enter"), self
        )
        self._sim_inverse_shortcut_numpad.activated.connect(
            self._sim_shortcut_run_inverse
        )

    # ------------------------------------------------------------------
    # Shortcut slots
    # ------------------------------------------------------------------

    def _preferred_initial_size(self, preferred_w: int, preferred_h: int):
        """Cap the initial window size to 90 % of the available screen
        so the app never opens wider / taller than the display it
        launches on.  Falls back to the preferred size if Qt can't
        resolve a primary screen (headless / tests).
        """
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return QSize(preferred_w, preferred_h)
        avail = screen.availableGeometry()
        max_w = int(avail.width() * 0.9)
        max_h = int(avail.height() * 0.9)
        return QSize(min(preferred_w, max_w), min(preferred_h, max_h))

    def _on_theme_mode_selected(self, mode: str) -> None:
        """Handle View → Light/Dark action trigger."""
        app = QApplication.instance()
        if app is None:
            return
        set_theme_mode(app, mode)
        # Repolish all custom-styled chips/banners so they pick up the
        # new tone_palette values.  The status bar caches state and
        # refreshes on its own retranslate/apply helpers; session
        # summary's apply_state_banner re-runs from _refresh_session_summary.
        self._refresh_session_summary()
        self._status_bar._retranslate()

    def _on_precision_selected(self, mode: str) -> None:
        """Handle View → Compute Precision → float32/float64."""
        if mode == current_precision():
            return
        set_precision(mode)
        self._action_precision_float32.setChecked(mode == "float32")
        self._action_precision_float64.setChecked(mode == "float64")
        self._status_bar.showMessage(
            t("main.status.precision_changed", mode=mode), 5000
        )

    def _sim_shortcut_run_forward(self) -> None:
        """F5 handler — only acts when the Simulation tab is visible
        and the forward-solve button is currently enabled (i.e. we're
        not already mid-solve)."""
        if self._tab_widget.currentWidget() is not self._sim_tab:
            return
        btn = self._sim_tab.forward_problem_panel._solve_btn
        if btn.isEnabled():
            btn.click()

    def _sim_shortcut_run_inverse(self) -> None:
        """Ctrl+Enter handler for the inverse reconstruction button."""
        if self._tab_widget.currentWidget() is not self._sim_tab:
            return
        btn = self._sim_tab.inverse_problem_panel._recon_btn
        if btn.isEnabled():
            btn.click()

    # ------------------------------------------------------------------
    # i18n — retranslate chrome owned directly by the main window
    # ------------------------------------------------------------------

    @Slot(str)
    def _on_language_changed(self, _lang: str) -> None:
        """Slot for :attr:`Translator.language_changed`."""
        self._retranslate()
        # Session summary's banner + indicator chips are built from the
        # current app state, not from a static key set, so re-run the
        # summary refresh to pick up translated strings.
        self._refresh_session_summary()

    def _retranslate(self) -> None:
        """Refresh every user-visible string owned by :class:`EITWorkstation`.

        Child widgets are responsible for their own retranslation; they
        subscribe to :meth:`eit_app.i18n.translator.language_changed` in
        their own ``__init__``.
        """
        log.info(
            "[i18n] retranslating main window (language=%s)", current_language()
        )
        self.setWindowTitle(t("app.title"))

        self._tab_widget.setTabText(0, t("tab.hardware"))
        self._tab_widget.setTabText(1, t("tab.simulation"))
        self._tab_widget.setTabText(2, t("tab.dataset"))
        self._tab_widget.setTabText(3, t("tab.database"))

        self._menu_file.setTitle(t("menu.file"))
        self._action_exit.setText(t("menu.file.exit"))

        self._menu_view.setTitle(t("menu.view"))
        self._action_theme_light.setText(t("menu.view.theme_light"))
        self._action_theme_dark.setText(t("menu.view.theme_dark"))
        self._action_theme_light.setChecked(current_theme_mode() == "light")
        self._action_theme_dark.setChecked(current_theme_mode() == "dark")
        self._menu_precision.setTitle(t("menu.view.precision"))
        self._action_precision_float32.setText(t("menu.view.precision_float32"))
        self._action_precision_float64.setText(t("menu.view.precision_float64"))
        self._action_precision_float32.setChecked(current_precision() == "float32")
        self._action_precision_float64.setChecked(current_precision() == "float64")

        self._menu_tools.setTitle(t("menu.tools"))
        self._action_interop_hub.setText(t("menu.tools.interop_hub"))
        self._action_difference.setText(t("menu.tools.difference"))
        self._action_batch_reconstruction.setText(t("menu.tools.batch_reconstruction"))
        self._action_reconstruction.setText(t("menu.tools.reconstruction"))

        self._menu_language.setTitle(t("menu.language"))
        self._menu_language.setToolTip(t("menu.language.tooltip"))
        self._action_lang_zh.setText(t("menu.language.zh"))
        self._action_lang_en.setText(t("menu.language.en"))
        self._action_lang_zh.setChecked(current_language() == "zh")
        self._action_lang_en.setChecked(current_language() == "en")

    def _connect_signals(self) -> None:
        self._conn_panel.connect_requested.connect(self._on_connect_requested)
        self._conn_panel.disconnect_requested.connect(self._on_disconnect_requested)
        self._conn_panel.validation_failed.connect(self._on_error)

        self._device_ctrl.connected.connect(self._on_connected)
        self._device_ctrl.disconnected.connect(self._on_disconnected)
        self._device_ctrl.error.connect(self._on_error)
        self._device_ctrl.command_done.connect(self._on_device_command_done)
        self._device_ctrl.impedance_result.connect(self._on_impedance_result)

        self._control_panel.frequency_changed.connect(self._on_frequency_changed)
        self._control_panel.stim_amp_changed.connect(self._on_stim_amp_changed)
        self._control_panel.voltage_amp_changed.connect(self._on_voltage_amp_changed)
        self._control_panel.measurement_layout_changed.connect(self._on_measurement_layout_changed)
        self._control_panel.power_toggled.connect(self._on_power_toggled)
        self._control_panel.impedance_requested.connect(self._device_ctrl.measure_impedance)
        self._control_panel.single_point_requested.connect(self._on_single_point_requested)

        self._acq_panel.start_requested.connect(self._on_start_acquisition)
        self._acq_panel.single_frame_requested.connect(self._on_single_frame_requested)
        self._acq_panel.stop_requested.connect(self._on_stop_acquisition)
        self._acq_panel.recording_toggled.connect(self._on_recording_toggled)
        self._acq_panel.output_dir_changed.connect(self._on_output_dir_changed)
        self._acq_panel.acquisition_plan_changed.connect(self._on_acquisition_plan_changed)

        self._acq_ctrl.new_frame.connect(self._live_plot.update_frame)
        self._acq_ctrl.new_frame.connect(self._on_new_frame)
        self._acq_ctrl.fps_updated.connect(self._status_bar.on_fps_updated)
        self._acq_ctrl.error.connect(self._on_error)

        self._rec_ctrl.frame_saved.connect(self._on_frame_saved)
        self._rec_ctrl.recording_started.connect(self._on_recording_started)
        self._rec_ctrl.recording_stopped.connect(self._on_recording_stopped)
        self._rec_ctrl.error.connect(self._on_error)

        self._recon_ctrl.reconstruction_done.connect(self._on_auto_reconstruction_done)
        self._recon_ctrl.error.connect(self._on_auto_reconstruction_error)

        self._recon_prewarm_ctrl.reconstruction_done.connect(self._on_realtime_recon_prewarm_done)
        self._recon_prewarm_ctrl.error.connect(self._on_realtime_recon_prewarm_error)

        self._hw_recon_ctrl.reconstruction_done.connect(self._recon_widget.update_reconstruction)
        self._hw_recon_ctrl.reconstruction_done.connect(self._on_hardware_reconstruction_done)
        self._hw_recon_ctrl.progress.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._hw_recon_ctrl.error.connect(self._on_error)

        self._db_recon_ctrl.reconstruction_done.connect(self._on_db_reconstruction_done)
        self._db_recon_ctrl.progress.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._db_recon_ctrl.error.connect(self._on_error)

        self._sim_recon_ctrl.progress.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._sim_recon_ctrl.error.connect(self._on_error)

        self._frame_browser.reference_selected.connect(self._on_reference_selected)
        self._frame_browser.target_selected.connect(self._on_target_selected)
        self._frame_browser.frame_clicked.connect(self._on_frame_clicked)
        self._frame_browser.cleared.connect(self._on_frame_browser_cleared)

        # Database tab: user-driven reconstruction on historical data
        self._db_tab.reconstruct_requested.connect(self._on_db_reconstruct_requested)
        self._db_tab.open_containing_folder_requested.connect(
            self._on_open_session_folder
        )
        self._db_tab.batch_reconstruct_requested.connect(
            self._on_open_batch_dialog
        )

        self._state.connection_status_changed.connect(self._status_bar.on_connection_changed)
        self._state.power_status_changed.connect(self._status_bar.on_power_status_changed)
        self._state.power_status_changed.connect(self._control_panel.set_power_state)
        self._state.acquisition_mode_changed.connect(self._status_bar.on_acquisition_mode_changed)
        self._state.frame_count_changed.connect(self._status_bar.on_frame_count_changed)
        self._state.frame_count_changed.connect(self._acq_panel.set_frame_count)
        self._state.recording_active_changed.connect(self._status_bar.on_recording_changed)
        self._state.recording_status_changed.connect(self._status_bar.on_recording_status_changed)
        self._state.connection_status_changed.connect(lambda _value: self._refresh_session_summary())
        self._state.power_status_changed.connect(lambda _value: self._refresh_session_summary())
        self._state.acquisition_mode_changed.connect(lambda _value: self._refresh_session_summary())
        self._state.recording_status_changed.connect(lambda _value: self._refresh_session_summary())

        # Tab switching
        self._tab_widget.currentChanged.connect(self._status_bar.on_tab_changed)

        # --- Simulation signals ---
        sim = self._sim_tab
        sim.mesh_setup_panel.config_changed.connect(self._sync_sim_inhomogeneity_context)
        sim.forward_problem_panel.run_forward_requested.connect(self._on_run_forward)
        sim.inverse_problem_panel.run_inverse_requested.connect(self._on_run_sim_inverse)
        sim.inverse_problem_panel.save_requested.connect(self._on_save_sim_results)

        dataset = self._dataset_tab
        dataset.dataset_generator_panel.generate_requested.connect(self._on_generate_dataset)
        dataset.dataset_generator_panel.cancel_requested.connect(self._dataset_ctrl.cancel)

        self._fwd_ctrl.forward_done.connect(self._on_forward_done)
        self._fwd_ctrl.progress.connect(lambda msg: self._status_bar.showMessage(msg, 3000))
        self._fwd_ctrl.error.connect(self._on_error)

        self._dataset_ctrl.progress.connect(self._dataset_tab.set_progress)
        self._dataset_ctrl.generation_done.connect(self._on_dataset_done)
        self._dataset_ctrl.error.connect(self._on_error)

    @Slot(str, dict)
    def _on_connect_requested(self, transport_type: str, config: dict) -> None:
        prepared = self._prepare_connection_request(transport_type, dict(config))
        if prepared is None:
            return
        merged = dict(self._device_config)
        merged.update(prepared)
        merged["transport_type"] = transport_type
        self._transport_type = transport_type
        self._device_config = normalize_device_config(transport_type, merged)
        self._sync_state_device_config()
        self._device_ctrl.set_connection_profile(transport_type, self._device_config)
        self._state.set_connection_status(ConnectionStatus.CONNECTING)
        self._refresh_session_summary()
        self._status_bar.showMessage(self._connect_attempt_message(transport_type, self._device_config), 5000)
        self._device_ctrl.connect_device()

    def _prepare_connection_request(self, transport_type: str, config: dict) -> dict | None:
        if transport_type == "serial":
            port = str(config.get("port", "")).strip()
            if not port:
                self._conn_panel.refresh_serial_ports()
                config["port"] = self._conn_panel.selected_serial_port()
                config["port_display"] = self._conn_panel.selected_serial_display_name()
                port = str(config.get("port", "")).strip()

            if not port:
                self._conn_panel.set_serial_hint(t("main.status.port_not_found_scan"))
                self._on_error("Connection failed: No serial port detected.")
                return None

            preflight = preflight_connection_target("serial", config)
            if not preflight.ok:
                self._conn_panel.set_serial_hint(preflight.hint or preflight.summary)
                self._on_error(f"Connection failed: {preflight.summary}")
                return None
            if preflight.hint:
                self._conn_panel.set_serial_hint(preflight.hint)
            return config

        if transport_type == "relay":
            host = str(config.get("server_host", "")).strip()
            if not host:
                self._conn_panel.set_relay_hint(t("main.status.relay_host_empty"))
                self._on_error("Connection failed: Relay host is empty.")
                return None
            preflight = preflight_connection_target("relay", config)
            if not preflight.ok:
                self._conn_panel.set_relay_hint(preflight.hint or preflight.summary)
                self._on_error(f"Connection failed: {preflight.summary}")
                return None
            if preflight.hint:
                self._conn_panel.set_relay_hint(preflight.hint)
            return config

        return config

    @staticmethod
    def _connect_attempt_message(transport_type: str, config: dict) -> str:
        if transport_type == "serial":
            port = str(config.get("port_display", "")).strip() or str(config.get("port", "")).strip()
            baud = int(config.get("baudrate", 115200))
            if port.upper().startswith("COM"):
                return t("main.status.verifying.windows_bridge", port=port, baud=baud)
            return t("main.status.verifying.serial", port=port, baud=baud)
        if transport_type == "relay":
            host = str(config.get("server_host", "127.0.0.1"))
            port = int(config.get("server_port", 4555))
            return t("main.status.verifying.relay", host=host, port=port)
        return t("main.status.verifying.generic")

    @Slot()
    def _on_connected(self) -> None:
        self._state.set_connection_status(ConnectionStatus.CONNECTED)
        self._state.set_power_status(PowerStatus.UNKNOWN)
        self._state.set_acquisition_mode(AcquisitionMode.IDLE)
        self._state.set_recording_status(RecordingStatus.OFF)
        self._conn_panel.set_connected(True)
        self._control_panel.set_enabled(True)
        self._workflow_toolbox.setCurrentIndex(1)
        self._status_bar.showMessage(t("main.status.link_verified"), 4000)
        self._refresh_session_summary()
        self._schedule_realtime_recon_prewarm()

    @Slot()
    def _on_disconnected(self) -> None:
        self._recon_prewarm_timer.stop()
        self._recon_prewarm_busy = False
        self._recon_prewarm_active_signature = None
        self._recon_prewarm_requested_signature = None
        self._recon_prewarm_ready_signature = None
        self._pending_power_commands.clear()
        self._state.set_connection_status(ConnectionStatus.DISCONNECTED)
        self._state.set_power_status(PowerStatus.UNKNOWN)
        self._state.set_acquisition_mode(AcquisitionMode.IDLE)
        self._state.set_recording_status(RecordingStatus.OFF)
        self._conn_panel.set_connected(False)
        self._control_panel.set_enabled(False)
        self._workflow_toolbox.setCurrentIndex(0)
        self._refresh_session_summary()

    def _on_disconnect_requested(self) -> None:
        self._on_stop_acquisition()
        self._device_ctrl.disconnect_device()

    @Slot()
    def _on_start_acquisition(self) -> None:
        self._start_acquisition(single_frame=False)

    @Slot()
    def _on_single_frame_requested(self) -> None:
        self._start_acquisition(single_frame=True)

    def _start_acquisition(self, *, single_frame: bool) -> None:
        if self._state.connection_status is not ConnectionStatus.CONNECTED and self._transport_type != "simulator":
            self._on_error(t("main.error.connection_required"))
            return

        if self._transport_type != "simulator":
            released = self._device_ctrl.suspend_session(timeout_ms=3000)
            if not released:
                self._on_error(t("main.error.port_release_failed"))
                return

        self._single_frame_pending = single_frame
        self._latest_frame_timestamp = 0.0
        self._state.set_frame_count(0)
        self._auto_reconstruct = not single_frame
        self._reference_frame = None
        self._auto_recon_busy = False
        self._pending_auto_target_frame = None
        self._last_auto_ref_frame = None
        self._last_auto_tgt_frame = None
        if not single_frame:
            self._schedule_realtime_recon_prewarm(immediate=True)
        # Clear the voltage fit from any previous run so it shows nothing
        # until the second frame of this run is captured and reconstructed.
        try:
            self._voltage_plot.clear()
        except Exception:
            pass
        # Phase 4: flip the LivePlot to "waiting for device frames" until
        # the first frame arrives.  update_frame() clears the overlay.
        try:
            self._live_plot.set_loading(True)
        except Exception:
            pass
        if self._record_requested:
            if not self._ensure_recording_session(self._acq_panel.output_dir()):
                self._record_requested = False
                self._state.set_recording_status(RecordingStatus.OFF)

        if single_frame:
            self._rebuild_acquisition_pipeline()
            self._state.set_acquisition_mode(AcquisitionMode.SINGLE_SHOT)
            self._acq_ctrl.capture_one()
            self._status_bar.showMessage(t("main.status.single_frame_started"), 4000)
        elif self._planned_acquisition_count > 0 or self._frequency_stepping_enabled or self._scheduled_enabled:
            if self._planned_acquisition_count <= 0:
                self._on_error(t("main.error.acq_count_zero"))
                return
            self._start_planned_acquisition_run()
        else:
            self._rebuild_acquisition_pipeline()
            self._state.set_acquisition_mode(AcquisitionMode.CONTINUOUS)
            self._acq_ctrl.start()
            self._status_bar.showMessage(t("main.status.continuous_started"), 3000)

        self._state.set_power_status(PowerStatus.ON)
        self._acq_panel.set_acquiring(True)
        self._control_panel.set_enabled(False)
        self._workflow_toolbox.setCurrentIndex(2)
        self._refresh_session_summary()

    @Slot()
    def _on_stop_acquisition(self) -> None:
        self._plan_timer.stop()
        was_single_frame_mode = self._state.acquisition_mode is AcquisitionMode.SINGLE_SHOT
        was_plan_mode = self._plan_active or self._state.acquisition_mode in {
            AcquisitionMode.FINITE_RUN,
            AcquisitionMode.STEPPED_RUN,
            AcquisitionMode.SCHEDULED,
        }

        self._reset_acquisition_pipeline()
        self._single_frame_pending = False
        self._planned_step_pending = False
        self._plan_active = False
        self._plan_completed_count = 0
        self._plan_frequencies = []
        self._auto_reconstruct = False
        self._pending_auto_target_frame = None
        self._last_auto_ref_frame = None
        self._last_auto_tgt_frame = None
        self._state.set_acquisition_mode(AcquisitionMode.IDLE)
        self._acq_panel.set_acquiring(False)

        if self._state.connection_status is ConnectionStatus.CONNECTED:
            self._control_panel.set_enabled(True)
            self._workflow_toolbox.setCurrentIndex(2)

        if self._rec_ctrl.is_recording:
            self._rec_ctrl.stop_recording()
            self._state.set_recording_active(False)

        if self._record_requested:
            self._state.set_recording_status(RecordingStatus.ARMED)
        else:
            self._state.set_recording_status(RecordingStatus.OFF)

        if was_single_frame_mode:
            self._status_bar.showMessage(t("main.status.single_frame_done"), 4000)
        elif was_plan_mode:
            self._status_bar.showMessage(t("main.status.plan_stopped"), 4000)
        self._refresh_session_summary()

    @Slot(object)
    def _on_new_frame(self, frame: FrameData) -> None:
        self._latest_frame_timestamp = frame.timestamp
        self._state.set_frame_count(self._acq_ctrl.total_frames)
        if self._rec_ctrl.is_recording:
            self._rec_ctrl.save_frame(frame)
        if not self._auto_reconstruct:
            self._voltage_plot.update_hardware_voltages(frame.real, None)
        if self._single_frame_pending and self._state.frame_count >= 1:
            self._single_frame_pending = False
            QTimer.singleShot(0, self._on_stop_acquisition)

        # Auto-reconstruction: first frame becomes reference, subsequent
        # frames are reconstructed as difference against the reference.
        if self._auto_reconstruct:
            if self._reference_frame is None:
                self._reference_frame = frame
                self._frame_browser.set_reference_highlight(0)
                self._schedule_realtime_recon_prewarm(immediate=True)
                self._status_bar.showMessage(
                    f"Auto-reference set to frame #{frame.frame_index}", 3000
                )
            else:
                request_signature: tuple[object, ...] | None = None
                try:
                    request_signature = self._build_realtime_recon_prewarm_payload()[1]
                except Exception as exc:
                    log.debug("Failed to build realtime prewarm signature: %s", exc)
                if (
                    request_signature is not None
                    and request_signature != self._recon_prewarm_ready_signature
                ):
                    self._pending_auto_target_frame = frame
                    if not (
                        self._recon_prewarm_busy
                        and self._recon_prewarm_active_signature == request_signature
                    ):
                        self._schedule_realtime_recon_prewarm(immediate=True)
                elif not self._auto_recon_busy:
                    self._submit_auto_reconstruction(frame)
                else:
                    self._pending_auto_target_frame = frame

        if self._plan_active and self._planned_step_pending:
            self._planned_step_pending = False
            self._plan_completed_count += 1
            self._state.set_frame_count(self._plan_completed_count)
            self._reset_acquisition_pipeline()
            if self._plan_completed_count >= len(self._plan_frequencies):
                self._finish_planned_acquisition_run()
            elif self._scheduled_enabled:
                self._plan_timer.start(int(self._scheduled_interval_sec * 1000))
                self._status_bar.showMessage(
                    t(
                        "main.status.plan_step_done",
                        current=self._plan_completed_count,
                        total=len(self._plan_frequencies),
                        interval=self._scheduled_interval_sec,
                    ),
                    4000,
                )
            else:
                QTimer.singleShot(0, self._run_next_planned_acquisition)
            return

    @Slot(int, float, str)
    def _on_frame_saved(self, index: int, timestamp: float, path: str) -> None:
        self._frame_browser.add_frame_entry(index, timestamp, path)

    @Slot(str)
    def _on_recording_started(self, session_dir: str) -> None:
        self._state.set_recording_active(True)
        self._state.set_recording_status(RecordingStatus.RECORDING)
        self._status_bar.showMessage(
            t("main.status.recording_started", dir=session_dir), 5000
        )
        self._refresh_session_summary()

    @Slot(int)
    def _on_recording_stopped(self, count: int) -> None:
        self._state.set_recording_active(False)
        if self._record_requested:
            self._state.set_recording_status(RecordingStatus.ARMED)
        else:
            self._state.set_recording_status(RecordingStatus.OFF)
        self._status_bar.showMessage(
            t("main.status.recording_stopped", count=count), 5000
        )
        self._refresh_session_summary()

    # ---- Auto-reconstruction helpers ----

    def _submit_auto_reconstruction(self, target_frame: FrameData) -> None:
        """Submit a difference reconstruction request using the stored reference."""
        if self._reference_frame is None:
            return
        self._auto_recon_busy = True
        self._pending_auto_target_frame = None
        # Remember the frame pair so we can display measured/reconstructed
        # difference voltages even if the backend doesn't populate them.
        self._last_auto_ref_frame = self._reference_frame
        self._last_auto_tgt_frame = target_frame
        request = self._build_auto_reconstruction_request(
            target_frame,
            reference_frame=self._reference_frame,
            request_source="hardware_auto_live",
        )
        accepted = self._recon_ctrl.reconstruct(request)
        if not accepted:
            self._auto_recon_busy = False
            self._pending_auto_target_frame = target_frame
        return

    @staticmethod
    def _reconstruction_result_source(result: object) -> str:
        metadata = getattr(result, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("request_source", "")).strip().lower()

    @Slot(str)
    def _on_auto_reconstruction_error(self, msg: str) -> None:
        self._auto_recon_busy = False
        self._on_error(msg)
        if self._auto_reconstruct and self._pending_auto_target_frame is not None:
            QTimer.singleShot(0, self._submit_pending_auto_reconstruction)

    @Slot(object)
    def _on_auto_reconstruction_done(self, result) -> None:
        """Handle completed auto-reconstruction during acquisition.

        Display policy (difference imaging):
        - Reconstruction image: element-wise delta conductivity.
        - Voltage fit plot:
            * Measured diff = target.real - reference.real (always computed
              locally so it's reliable even if the backend omits it).
            * Recon fit = backend's simulated diff if provided; otherwise
              only the measured diff is shown.
        """
        if self._reconstruction_result_source(result) != "hardware_auto_live":
            return
        self._auto_recon_busy = False
        if getattr(result, "error_msg", None):
            if self._auto_reconstruct:
                self._auto_reconstruct = False
                self._status_bar.showMessage(
                    "Auto-reconstruction disabled: " + str(result.error_msg)[:80],
                    10000,
                )
                log.warning(
                    "Auto-reconstruction disabled after error: %s",
                    result.error_msg,
                )
            return

        # Always update the reconstruction image first
        try:
            self._recon_widget.update_reconstruction(result)
        except Exception as exc:
            log.warning("Reconstruction widget update failed: %s", exc)

        # Voltage fit: compute measured diff from the frame pair we submitted
        ref_frame = self._last_auto_ref_frame
        tgt_frame = self._last_auto_tgt_frame
        measured_diff: np.ndarray | None = None
        if ref_frame is not None and tgt_frame is not None:
            try:
                measured_diff = np.asarray(
                    tgt_frame.real - ref_frame.real, dtype=np.float64
                )
            except Exception as exc:
                log.debug("Failed to compute measured diff: %s", exc)

        # Prefer backend-provided measured diff if available and same shape
        backend_measured = getattr(result, "measured", None)
        if backend_measured is not None:
            try:
                backend_arr = np.asarray(backend_measured, dtype=np.float64).reshape(-1)
                if measured_diff is None or backend_arr.size == measured_diff.size:
                    measured_diff = backend_arr
            except Exception:
                pass

        simulated = getattr(result, "simulated", None)
        simulated_arr: np.ndarray | None = None
        if simulated is not None:
            try:
                simulated_arr = np.asarray(simulated, dtype=np.float64).reshape(-1)
                if (
                    measured_diff is not None
                    and simulated_arr.size != measured_diff.size
                ):
                    log.debug(
                        "Simulated size %d != measured %d; skipping recon curve",
                        simulated_arr.size, measured_diff.size,
                    )
                    simulated_arr = None
            except Exception:
                simulated_arr = None

        if measured_diff is not None and measured_diff.size > 0:
            self._voltage_plot.update_hardware_voltages(measured_diff, simulated_arr)
            log.debug(
                "Voltage fit updated: measured_diff=%d, simulated=%s",
                measured_diff.size,
                "yes" if simulated_arr is not None else "no",
            )

        if self._auto_reconstruct and self._pending_auto_target_frame is not None:
            QTimer.singleShot(0, self._submit_pending_auto_reconstruction)

    @Slot(object)
    def _on_realtime_recon_prewarm_done(self, result) -> None:
        if self._reconstruction_result_source(result) != "hardware_auto_prewarm":
            return
        active_signature = self._recon_prewarm_active_signature
        self._recon_prewarm_busy = False
        self._recon_prewarm_active_signature = None
        if getattr(result, "error_msg", None):
            return
        if active_signature is not None:
            self._recon_prewarm_ready_signature = active_signature
        self._status_bar.showMessage(t("main.status.prewarm_done"), 4000)
        if (
            self._recon_prewarm_requested_signature is not None
            and self._recon_prewarm_requested_signature != self._recon_prewarm_ready_signature
        ):
            QTimer.singleShot(0, self._run_realtime_recon_prewarm)
            return
        if self._auto_reconstruct and self._pending_auto_target_frame is not None:
            QTimer.singleShot(0, self._submit_pending_auto_reconstruction)

    @Slot(str)
    def _on_realtime_recon_prewarm_error(self, msg: str) -> None:
        self._recon_prewarm_busy = False
        self._recon_prewarm_active_signature = None
        log.warning("Realtime reconstruction prewarm failed: %s", msg)
        self._status_bar.showMessage(
            t(
                "main.status.prewarm_failed",
                reason=self._humanize_error_message(msg),
            ),
            10000,
        )

    def _submit_pending_auto_reconstruction(self) -> None:
        if not self._auto_reconstruct or self._auto_recon_busy:
            return
        pending_frame = self._pending_auto_target_frame
        if pending_frame is None:
            return
        self._submit_auto_reconstruction(pending_frame)

    @Slot(dict)
    def _on_reference_selected(self, entry: dict) -> None:
        self._selected_reference_entry = dict(entry)
        for row in range(self._frame_browser._model.rowCount()):
            current = self._frame_browser._model.get_entry(row)
            if current and current.get("file_path") == entry.get("file_path"):
                self._frame_browser.set_reference_highlight(row)
                break
        # Also update the auto-reconstruct reference frame
        file_path = entry.get("file_path", "")
        if file_path:
            try:
                from pyeidors.data.frame_io import read_frame_csv
                real, imag = read_frame_csv(file_path)
                self._reference_frame = FrameData(
                    real=real, imag=imag,
                    timestamp=entry.get("timestamp", 0.0),
                    frame_index=entry.get("frame_index", 0),
                )
                self._status_bar.showMessage(
                    t(
                        "main.status.reference_updated",
                        index=entry.get("frame_index", "?"),
                    ),
                    3000,
                )
            except Exception as exc:
                self._on_error(f"Failed to load reference frame: {exc}")
                return
        else:
            self._status_bar.showMessage(
                t(
                    "main.status.reference_selected",
                    index=entry.get("frame_index", "?"),
                ),
                3000,
            )

    @Slot(dict)
    def _on_target_selected(self, entry: dict) -> None:
        self._selected_target_entry = dict(entry)
        self._status_bar.showMessage(
            t(
                "main.status.target_selected",
                index=entry.get("frame_index", "?"),
            ),
            3000,
        )

    @Slot(dict)
    def _on_frame_clicked(self, entry: dict) -> None:
        """Load a recorded frame and display its waveform in the live plot."""
        file_path = entry.get("file_path", "")
        if not file_path:
            return
        try:
            from pyeidors.data.frame_io import read_frame_csv

            real, imag = read_frame_csv(file_path)
            from eit_app.models.frame_model import FrameData

            frame = FrameData(
                real=real,
                imag=imag,
                timestamp=entry.get("timestamp", 0.0),
                frame_index=entry.get("frame_index", 0),
            )
            self._live_plot.update_frame(frame)
            self._status_bar.showMessage(
                t(
                    "main.status.frame_preview",
                    index=entry.get("frame_index", "?"),
                ),
                3000,
            )
        except Exception as exc:
            self._on_error(f"Failed to load frame: {exc}")

    @Slot()
    def _on_frame_browser_cleared(self) -> None:
        self._selected_reference_entry = None
        self._selected_target_entry = None
        self._status_bar.showMessage(t("main.status.frames_cleared"), 3000)

    @Slot(bool, str)
    def _on_recording_toggled(self, active: bool, output_dir: str) -> None:
        normalized_output_dir = self._normalize_output_dir(output_dir)
        if normalized_output_dir:
            self._acq_panel.set_output_dir(normalized_output_dir)
        elif normalized_output_dir != output_dir:
            self._acq_panel.set_output_dir(normalized_output_dir)

        if active and not normalized_output_dir:
            normalized_output_dir = self._default_output_dir()
            self._acq_panel.set_output_dir(normalized_output_dir)

        self._record_requested = active
        if active:
            self._state.set_recording_status(RecordingStatus.ARMED)
            if self._state.acquisition_mode is AcquisitionMode.IDLE:
                target_dir = normalized_output_dir or self._default_output_dir()
                self._status_bar.showMessage(
                    t("main.status.record_enabled", dir=target_dir),
                    5000,
                )
                self._state.set_recording_active(False)
                return
            started = self._ensure_recording_session(normalized_output_dir)
            if not started:
                self._record_requested = False
                self._acq_panel.set_recording_active(False)
                self._state.set_recording_active(False)
                self._state.set_recording_status(RecordingStatus.OFF)
        else:
            if self._rec_ctrl.is_recording:
                self._rec_ctrl.stop_recording()
            self._state.set_recording_active(False)
            self._state.set_recording_status(RecordingStatus.OFF)
        self._refresh_session_summary()

    @Slot(str)
    def _on_output_dir_changed(self, _path: str) -> None:
        self._refresh_session_summary()

    @Slot(dict)
    def _on_acquisition_plan_changed(self, plan: dict) -> None:
        self._scheduled_enabled = bool(plan.get("timed_enabled", False))
        self._scheduled_interval_sec = float(plan.get("interval_sec", 5.0))
        self._planned_acquisition_count = int(plan.get("acquisition_count", 0))
        self._frequency_stepping_enabled = bool(plan.get("frequency_stepping", False))
        self._planned_start_hz = int(plan.get("start_hz", self._device_config.get("frequency_hz", 1000)))
        self._planned_end_hz = int(plan.get("end_hz", self._device_config.get("frequency_hz", 1000)))
        self._refresh_session_summary()

    @Slot(int)
    def _on_frequency_changed(self, hz: int) -> None:
        self._device_config["frequency_hz"] = hz
        self._sync_state_device_config()
        self._device_ctrl.set_connection_profile(self._transport_type, self._device_config)
        self._device_ctrl.set_frequency(hz)
        self._refresh_session_summary()
        self._schedule_realtime_recon_prewarm()

    @Slot(int)
    def _on_stim_amp_changed(self, level: int) -> None:
        self._device_config["stim_amp_level"] = level
        self._device_config["stim_amp_uA"] = STIM_AMP_VALUES_UA.get(level, level)
        self._sync_state_device_config()
        self._device_ctrl.set_connection_profile(self._transport_type, self._device_config)
        self._device_ctrl.set_stim_amplitude(level)
        self._refresh_session_summary()
        self._schedule_realtime_recon_prewarm()

    @Slot(int)
    def _on_voltage_amp_changed(self, level: int) -> None:
        self._device_config["voltage_amp_level_1"] = level
        self._device_config["voltage_amp_level_2"] = level
        self._device_config["contact_impedance_amp_level"] = level
        self._sync_state_device_config()
        self._device_ctrl.set_connection_profile(self._transport_type, self._device_config)
        self._device_ctrl.set_voltage_amp_levels(level, level)
        self._refresh_session_summary()
        self._schedule_realtime_recon_prewarm()

    @Slot(dict)
    def _on_measurement_layout_changed(self, layout: dict) -> None:
        self._device_config.update(layout)
        self._device_config = normalize_device_config(self._transport_type, self._device_config)
        self._sync_state_device_config()
        self._device_ctrl.set_connection_profile(self._transport_type, self._device_config)
        self._refresh_expected_measurement_counts()
        self._refresh_session_summary()
        points = int(self._device_config.get("points_per_frame", self._measurement_point_count()))
        self._status_bar.showMessage(
            t("main.status.layout_updated", points=points),
            3500,
        )
        self._schedule_realtime_recon_prewarm(immediate=True)

    @Slot(bool)
    def _on_power_toggled(self, on: bool) -> None:
        self._pending_power_commands.append(on)
        self._device_ctrl.power_control(on)
        self._refresh_session_summary()

    @Slot()
    def _on_single_point_requested(self) -> None:
        hz = int(self._device_config.get("frequency_hz", 1000))
        self._device_ctrl.single_point_test(hz)

    @Slot(str, object)
    def _on_device_command_done(self, name: str, result: object) -> None:
        if name == "capabilities" and isinstance(result, dict):
            protocol_version = str(result.get("protocol_version", "legacy-v1"))
            self._device_config["protocol_version"] = protocol_version
            for key in (
                "acquisition_mode",
                "supports_streaming",
                "supports_extended_impedance",
                "supports_3d_batch",
            ):
                if key in result:
                    self._device_config[key] = result[key]
            self._sync_state_device_config()
            self._status_bar.showMessage(
                t("main.status.protocol_caps", version=protocol_version), 3000
            )
            return

        if name == "single_point_test_at" and isinstance(result, tuple) and len(result) == 2:
            self._status_bar.showMessage(
                t("main.status.spt_result", real=result[0], imag=result[1]),
                5000,
            )
            return

        if name == "power_control":
            desired = self._pending_power_commands.pop(0) if self._pending_power_commands else None
            if desired is True:
                self._state.set_power_status(PowerStatus.ON)
                self._control_panel.set_power_state("on")
                self._status_bar.showMessage(t("main.status.power_on"), 4000)
            elif desired is False:
                self._state.set_power_status(PowerStatus.OFF)
                self._control_panel.set_power_state("off")
                self._status_bar.showMessage(t("main.status.power_off"), 4000)
            else:
                self._status_bar.showMessage(t("main.status.power_sent"), 3000)
            self._refresh_session_summary()
            return

        if name in {"set_frequency", "set_stim_amplitude", "set_voltage_amp_levels"}:
            self._status_bar.showMessage(
                t("main.status.command_sent", name=name), 3000
            )

    @Slot(object)
    def _on_impedance_result(self, result: object) -> None:
        try:
            values = list(result)
        except Exception:
            self._status_bar.showMessage(t("main.status.impedance_done"), 3000)
            return
        preview = ", ".join(f"{float(v):.4f}" for v in values[:4])
        self._status_bar.showMessage(
            t("main.status.impedance_result", values=preview), 5000
        )

    @Slot(object)
    def _on_hardware_reconstruction_done(self, result: object) -> None:
        # Phase 4: clear any "Reconstructing…" overlays regardless of
        # which result source this is — update_reconstruction already
        # repaints on success, and we want the overlay gone even for
        # ignored events below.
        self._recon_widget.set_loading(False)
        self._voltage_plot.set_loading(False)
        if self._tab_widget.currentWidget() is not self._hw_tab:
            return
        source = self._reconstruction_result_source(result)
        if source in {"hardware_auto_live", "db", "simulation"}:
            return
        # When auto-reconstruction is active, the dedicated handler
        # _on_auto_reconstruction_done owns the voltage plot update (so it
        # can draw the *difference* voltages). Skip here to avoid clobbering.
        if self._auto_reconstruct or self._last_auto_tgt_frame is not None:
            return
        measured = getattr(result, "measured", None)
        reconstructed = getattr(result, "simulated", None)
        if measured is None:
            return
        try:
            measured_arr = np.asarray(measured, dtype=float).reshape(-1)
        except Exception:
            return
        if measured_arr.size == 0:
            return
        reconstructed_arr = None
        if reconstructed is not None:
            try:
                reconstructed_arr = np.asarray(reconstructed, dtype=float).reshape(-1)
            except Exception:
                reconstructed_arr = None
        self._voltage_plot.update_hardware_voltages(measured_arr, reconstructed_arr)

    def _measurement_layout_config(self) -> dict[str, object]:
        layout = measurement_layout_from_config(self._device_config)
        return {
            "n_elec": int(layout["n_elec"]),
            "n_rings": int(layout["n_rings"]),
            "electrode_layout": layout["electrode_layout"],
            "measurement_protocol": layout["measurement_protocol"],
            "custom_pattern_json": layout["custom_pattern_json"],
            "custom_stim_matrix": layout["custom_stim_matrix"],
            "custom_meas_matrices": layout["custom_meas_matrices"],
            "stim_pattern": layout["stim_pattern"],
            "meas_pattern": layout["meas_pattern"],
            "use_meas_current": bool(layout["use_meas_current"]),
            "use_meas_current_next": int(layout["use_meas_current_next"]),
            "rotate_meas": bool(layout["rotate_meas"]),
            "stim_direction": layout["stim_direction"],
            "meas_direction": layout["meas_direction"],
            "stim_first_positive": bool(layout["stim_first_positive"]),
            "radius": float(layout["radius"]),
            "geometry_scale_to_m": float(layout["geometry_scale_to_m"]),
            "electrode_length_m_override": layout["electrode_length_m_override"],
            "electrode_coverage": float(layout["electrode_coverage"]),
            "contact_impedance": float(layout["contact_impedance"]),
            "points_per_frame": int(layout["points_per_frame"]),
            "total_electrodes": int(layout["total_electrodes"]),
        }

    def _hardware_reconstruction_drive_metadata(self) -> dict[str, object]:
        stim_level = int(self._device_config.get("stim_amp_level", 1))
        fallback_uA = float(STIM_AMP_VALUES_UA.get(stim_level, 100))
        try:
            stim_uA = float(self._device_config.get("stim_amp_uA", fallback_uA))
        except (TypeError, ValueError):
            stim_uA = fallback_uA
        if not np.isfinite(stim_uA) or stim_uA <= 0.0:
            stim_uA = fallback_uA
        return {
            "stim_amp_uA": int(round(stim_uA)),
            "drive_mode": "total_current",
            "drive_value": stim_uA * 1.0e-6,
        }

    def _build_auto_reconstruction_request(
        self,
        target_frame: FrameData,
        *,
        reference_frame: FrameData | None = None,
        request_source: str = "hardware_auto_live",
        warmup_only: bool = False,
    ) -> ReconstructionRequest:
        ref_frame = reference_frame or self._reference_frame or target_frame
        metadata = {
            **self._measurement_layout_config(),
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
            **self._hardware_reconstruction_drive_metadata(),
            "geometry_scale_to_m": float(self._device_config.get("geometry_scale_to_m", 1.0)),
            "reconstruction_runtime": "single_step_cached",
            "difference_lambda": 1.0e-2,
            "background_sigma": 1.0,
            "contact_impedance": float(self._device_config.get("contact_impedance", 0.01)),
            "electrode_length_m_override": self._device_config.get("electrode_length_m_override"),
            "electrode_coverage": float(self._device_config.get("electrode_coverage", 0.5)),
            "radius": float(self._device_config.get("radius", 1.0)),
            "mesh_height": float(self._device_config.get("height", 1.0)),
            "electrode_height_ratio": float(self._device_config.get("electrode_height_ratio", 0.2)),
            "z_center": float(self._device_config.get("z_center", 0.0)),
            "request_source": request_source,
        }
        if warmup_only:
            metadata["warmup_only"] = True
        return ReconstructionRequest(
            reference_frame=ref_frame,
            target_frame=target_frame,
            use_part="real",
            method="gn-difference",
            regularization_alpha=1.0,
            max_iterations=1,
            mesh_dimension=3 if int(self._device_config.get("mea_mode", 2)) == 3 else 2,
            mesh_refinement=int(self._state.reconstruction_config.mesh_refinement),
            metadata=metadata,
        )

    def _build_realtime_recon_prewarm_payload(
        self,
    ) -> tuple[ReconstructionRequest, tuple[object, ...]]:
        n_meas = max(1, self._measurement_point_count())
        zeros = np.zeros(n_meas, dtype=float)
        dummy_frame = FrameData(
            real=zeros.copy(),
            imag=zeros.copy(),
            timestamp=0.0,
            frame_index=-1,
        )
        request = self._build_auto_reconstruction_request(
            dummy_frame,
            reference_frame=dummy_frame,
            request_source="hardware_auto_prewarm",
            warmup_only=True,
        )
        return request, get_single_step_cached_cache_key(request)

    def _schedule_realtime_recon_prewarm(self, *, immediate: bool = False) -> None:
        if (
            self._transport_type != "simulator"
            and self._state.connection_status is not ConnectionStatus.CONNECTED
        ):
            self._recon_prewarm_timer.stop()
            return
        _request, signature = self._build_realtime_recon_prewarm_payload()
        self._recon_prewarm_requested_signature = signature
        if self._recon_prewarm_ready_signature == signature and not self._recon_prewarm_busy:
            return
        if self._recon_prewarm_busy and self._recon_prewarm_active_signature == signature:
            return
        self._recon_prewarm_timer.start(0 if immediate else 350)

    def _run_realtime_recon_prewarm(self) -> None:
        if self._recon_prewarm_busy:
            return
        if (
            self._transport_type != "simulator"
            and self._state.connection_status is not ConnectionStatus.CONNECTED
        ):
            return
        request, signature = self._build_realtime_recon_prewarm_payload()
        self._recon_prewarm_requested_signature = signature
        if self._recon_prewarm_ready_signature == signature:
            return
        self._recon_prewarm_busy = True
        self._recon_prewarm_active_signature = signature
        accepted = self._recon_prewarm_ctrl.reconstruct(request)
        if not accepted:
            self._recon_prewarm_busy = False
            self._recon_prewarm_active_signature = None
            return
        self._status_bar.showMessage(t("main.status.prewarming"), 3000)

    def _measurement_point_count(self) -> int:
        return int(self._measurement_layout_config()["points_per_frame"])

    def _rebuild_acquisition_pipeline(self) -> None:
        self._reset_acquisition_pipeline()
        n_meas = self._measurement_point_count()
        self._ring_buffer = FrameRingBuffer(capacity=256, n_meas=n_meas, create=True)
        self._acq_process = AcquisitionProcess(
            device_factory=create_device_from_config,
            device_config={
                "transport_type": self._transport_type,
                "config": dict(self._device_config),
            },
            buffer_name=self._ring_buffer.name,
            buffer_capacity=self._ring_buffer.capacity,
            n_meas=n_meas,
        )
        self._acq_ctrl.configure(
            self._acq_process,
            self._ring_buffer,
            frame_metadata=self._frame_metadata(),
        )

    def _reset_acquisition_pipeline(self) -> None:
        if self._acq_process is not None:
            self._acq_ctrl.shutdown()
            self._acq_process = None
        if self._ring_buffer is not None:
            try:
                self._ring_buffer.unlink()
            except FileNotFoundError:
                pass
            self._ring_buffer = None

    def _frame_metadata(self) -> dict:
        metadata = {
            "frequency_hz": int(self._device_config.get("frequency_hz", 1000)),
            "stim_amp_uA": int(self._device_config.get("stim_amp_uA", 100)),
            "voltage_amp_level_1": int(self._device_config.get("voltage_amp_level_1", 0)),
            "voltage_amp_level_2": int(self._device_config.get("voltage_amp_level_2", 0)),
            "mea_mode": int(self._device_config.get("mea_mode", 2)),
            "board_id": int(self._device_config.get("board_id", 1)),
            "user_id": int(self._device_config.get("user_id", 1)),
            "transport_type": self._transport_type,
            "protocol_version": str(self._device_config.get("protocol_version", "legacy-v1")),
        }
        metadata.update(self._measurement_layout_config())
        return metadata

    def _ensure_recording_session(self, output_dir: str) -> bool:
        target_dir = self._normalize_output_dir(output_dir) or self._default_output_dir()
        self._acq_panel.set_output_dir(target_dir)

        current_parent = None
        if self._rec_ctrl.session_dir is not None:
            current_parent = str(self._rec_ctrl.session_dir.parent)
        if self._rec_ctrl.is_recording and current_parent == target_dir:
            return True
        if self._rec_ctrl.is_recording:
            if self._rec_ctrl.frames_recorded == 0:
                self._rec_ctrl.stop_recording()
            else:
                self._status_bar.showMessage(
                    t("main.status.record_path_pending"),
                    5000,
                )
                return True

        started = self._rec_ctrl.start_recording(target_dir, session_metadata=self._frame_metadata())
        if not started:
            self._acq_panel.set_recording_active(False)
            return False
        return True

    def _default_output_dir(self) -> str:
        return str(self._acq_panel.default_output_dir())

    @staticmethod
    def _default_db_path() -> Path:
        """Return a platform-appropriate path for the frame database."""
        import os
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:
            base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        return base / "PyEidors" / "eit_frames.db"

    def _trigger_backfill(self) -> None:
        """Scan data/measurements/ and backfill the SQLite DB on startup."""
        try:
            data_dir = Path(self._default_output_dir()).parent
            if data_dir.exists():
                self._db_ctrl.start_backfill(data_dir)
        except Exception as exc:
            log.warning("Backfill trigger failed: %s", exc)

    @staticmethod
    def _normalize_output_dir(output_dir: str) -> str:
        raw = str(output_dir or "").strip()
        if not raw:
            return raw

        if raw.startswith("file://"):
            parsed = urlparse(raw)
            raw = parsed.path or raw

        normalized = raw.replace("\\", "/")
        for prefix in ("//wsl.localhost/", "//wsl$/"):
            if normalized.startswith(prefix):
                parts = normalized.split("/")
                if len(parts) >= 5:
                    return "/" + "/".join(parts[4:])

        if len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
            drive = raw[0].lower()
            tail = raw[2:].replace("\\", "/")
            return f"/mnt/{drive}{tail}"

        return normalized

    def _sync_state_device_config(self) -> None:
        for key, value in self._device_config.items():
            if hasattr(self._state.device_config, key):
                setattr(self._state.device_config, key, value)
        self._control_panel.set_measurement_layout(self._device_config)
        self._refresh_expected_measurement_counts()
        self._refresh_session_summary()

    def _refresh_expected_measurement_counts(self) -> None:
        hardware_count = self._measurement_point_count()
        self._live_plot.set_expected_point_count(hardware_count)
        self._voltage_plot.set_expected_point_count(hardware_count)
        self._recon_widget.configure_layout(
            n_elec=int(self._device_config.get("n_elec", 16)),
            radius=float(self._device_config.get("radius", 1.0)),
            electrode_coverage=float(self._device_config.get("electrode_coverage", 0.5)),
        )

    def _refresh_session_summary(self) -> None:
        stim_level = int(self._device_config.get("stim_amp_level", 1))
        stim_uA = int(self._device_config.get("stim_amp_uA", STIM_AMP_VALUES_UA.get(stim_level, 100)))
        gain_1 = int(self._device_config.get("voltage_amp_level_1", 3))
        gain_2 = int(self._device_config.get("voltage_amp_level_2", 5))
        title, detail, next_action, tone = self._summary_banner_state()

        self._summary_panel.set_status_banner(
            title=title,
            detail=detail,
            next_action=next_action,
            tone=tone,
        )
        self._summary_panel.set_indicator_states(
            {
                "link": self._indicator_link_state(),
                "power": self._indicator_power_state(),
                "record": self._indicator_record_state(),
                "acq": self._indicator_acq_state(),
            }
        )

        self._summary_panel.set_summary(
            {
                "identity": self._format_identity_summary(),
                "transport": self._format_transport_summary(),
                "layout": self._format_layout_summary(),
                "drive": (
                    f"{int(self._device_config.get('frequency_hz', 1000))} Hz | "
                    f"{stim_uA} uA (L{stim_level}) | "
                    f"V1 {_voltage_gain_label(gain_1)} | "
                    f"V2 {_voltage_gain_label(gain_2)}"
                ),
                "record": self._format_record_summary(),
                "plan": self._format_mode_summary(),
            }
        )

    def _format_identity_summary(self) -> str:
        board = int(self._device_config.get("board_id", 1))
        user = int(self._device_config.get("user_id", 1))
        mea_mode = int(self._device_config.get("mea_mode", 2))
        dimension = "3D" if mea_mode == 3 else "2D"
        return f"Board {board} | User {user} | {dimension}"

    def _format_transport_summary(self) -> str:
        if self._transport_type == "serial":
            port_display = str(self._device_config.get("port_display", "")).strip()
            port = port_display or str(self._device_config.get("port", "")).strip() or "not set"
            baud = int(self._device_config.get("baudrate", 115200))
            return f"Serial | {port} @ {baud}"
        if self._transport_type == "relay":
            host = str(self._device_config.get("server_host", "127.0.0.1"))
            port = int(self._device_config.get("server_port", 4555))
            board = int(self._device_config.get("board_id", 1))
            user = int(self._device_config.get("user_id", 1))
            return f"4G Relay | {host}:{port} | board {board} | user {user}"
        return "Simulator"

    def _format_layout_summary(self) -> str:
        layout = self._measurement_layout_config()
        mea_mode = int(self._device_config.get("mea_mode", 2))
        dimension = "3D" if mea_mode == 3 else "2D"
        rotate = "rotate" if bool(layout["rotate_meas"]) else "fixed"
        drive = "drive-included" if bool(layout["use_meas_current"]) else "drive-excluded"
        electrode_length = float(layout.get("electrode_length_m_override", 0.0) or 0.0)
        contact_impedance = float(layout.get("contact_impedance", 0.01) or 0.01)
        electrode_coverage = float(layout.get("electrode_coverage", 0.5) or 0.5)
        return (
            f"{dimension} | "
            f"{int(layout['n_elec'])}E x {int(layout['n_rings'])}R | "
            f"{layout['stim_pattern']} / {layout['meas_pattern']} | "
            f"{rotate} | {drive} | "
            f"+{int(layout['use_meas_current_next'])} skip | "
            f"{int(layout['points_per_frame'])} pts\n"
            f"CEM | L={electrode_length:.4f} | z={contact_impedance:.4g} | cov={electrode_coverage * 100.0:.1f}%"
        )

    def _format_record_summary(self) -> str:
        path = self._acq_panel.output_dir() or self._default_output_dir()
        status = {
            RecordingStatus.OFF: "Off",
            RecordingStatus.ARMED: "Armed",
            RecordingStatus.RECORDING: "Writing",
        }.get(self._state.recording_status, "Off")
        return f"{status} | {path}"

    def _format_mode_summary(self) -> str:
        mode = self._state.acquisition_mode
        current_hz = int(self._device_config.get("frequency_hz", self._planned_start_hz))
        if self._plan_active:
            run_label = "Stepped Run" if self._frequency_stepping_enabled else "Finite Run"
            freq_info = ""
            if self._frequency_stepping_enabled and self._plan_frequencies:
                freq_info = f" | {self._plan_frequencies[0]}→{self._plan_frequencies[-1]} Hz"
            elif self._plan_frequencies:
                freq_info = f" | {current_hz} Hz"
            if self._scheduled_enabled:
                return (
                    f"{run_label} | {self._plan_completed_count}/{len(self._plan_frequencies)}"
                    f" | every {self._scheduled_interval_sec:.1f}s{freq_info}"
                )
            return f"{run_label} | {self._plan_completed_count}/{len(self._plan_frequencies)}{freq_info}"
        if mode is AcquisitionMode.CONTINUOUS:
            return "Continuous"
        if mode is AcquisitionMode.FINITE_RUN:
            return "Finite Run"
        if mode is AcquisitionMode.STEPPED_RUN:
            return "Stepped Run"
        if mode is AcquisitionMode.SINGLE_SHOT:
            return "Single frame"
        if self._scheduled_enabled or self._planned_acquisition_count > 0 or self._frequency_stepping_enabled:
            freq_info = ""
            if self._frequency_stepping_enabled:
                freq_info = f" | {self._planned_start_hz}→{self._planned_end_hz} Hz"
                idle_label = "Idle | Stepped Run"
            elif self._planned_acquisition_count > 0:
                freq_info = f" | {current_hz} Hz"
                idle_label = "Idle | Finite Run"
            else:
                idle_label = "Idle | Finite Run"
            if self._scheduled_enabled:
                return (
                    f"{idle_label} {self._planned_acquisition_count}x"
                    f" | every {self._scheduled_interval_sec:.1f}s{freq_info}"
                )
            return f"{idle_label} {self._planned_acquisition_count}x{freq_info}"
        return "Idle | manual"

    def _banner(self, variant: str, tone: str) -> tuple[str, str, str, str]:
        """Resolve a banner variant to (title, detail, action, tone) via i18n."""
        return (
            t(f"hw.summary.banner.{variant}.title"),
            t(f"hw.summary.banner.{variant}.detail"),
            t(f"hw.summary.banner.{variant}.action"),
            tone,
        )

    def _summary_banner_state(self) -> tuple[str, str, str, str]:
        if self._state.connection_status is ConnectionStatus.ERROR:
            return self._banner("fault", "error")
        if self._state.connection_status is ConnectionStatus.CONNECTING:
            return self._banner("verifying", "warn")
        if self._state.connection_status is ConnectionStatus.DISCONNECTED:
            return self._banner("link_down", "idle")

        if self._state.acquisition_mode in {
            AcquisitionMode.CONTINUOUS,
            AcquisitionMode.SCHEDULED,
            AcquisitionMode.FINITE_RUN,
            AcquisitionMode.STEPPED_RUN,
            AcquisitionMode.SINGLE_SHOT,
        }:
            if self._state.recording_status is RecordingStatus.RECORDING:
                return self._banner("acquiring_recording", "active")
            return self._banner("acquiring", "active")

        if self._transport_type == "simulator":
            return self._banner("ready_simulator", "ready")

        if self._state.power_status is PowerStatus.ON:
            if self._state.recording_status is RecordingStatus.ARMED:
                return self._banner("ready_record_armed", "ready")
            return self._banner("ready", "ready")

        if self._state.recording_status is RecordingStatus.ARMED:
            return self._banner("link_verified_armed", "warn")

        return self._banner("link_verified", "warn")

    def _indicator_link_state(self) -> tuple[str, str]:
        mapping = {
            ConnectionStatus.DISCONNECTED: ("hw.summary.chip.link.down", "idle"),
            ConnectionStatus.CONNECTING: ("hw.summary.chip.link.check", "warn"),
            ConnectionStatus.CONNECTED: ("hw.summary.chip.link.ok", "ready"),
            ConnectionStatus.ERROR: ("hw.summary.chip.link.fault", "error"),
        }
        key, tone = mapping.get(
            self._state.connection_status, ("hw.summary.chip.link.unk", "idle")
        )
        return (t(key), tone)

    def _indicator_power_state(self) -> tuple[str, str]:
        mapping = {
            PowerStatus.UNKNOWN: ("hw.summary.chip.power.unk", "idle"),
            PowerStatus.OFF: ("hw.summary.chip.power.off", "warn"),
            PowerStatus.ON: ("hw.summary.chip.power.on", "ready"),
        }
        key, tone = mapping.get(
            self._state.power_status, ("hw.summary.chip.power.unk", "idle")
        )
        return (t(key), tone)

    def _indicator_record_state(self) -> tuple[str, str]:
        mapping = {
            RecordingStatus.OFF: ("hw.summary.chip.record.off", "idle"),
            RecordingStatus.ARMED: ("hw.summary.chip.record.arm", "ready"),
            RecordingStatus.RECORDING: ("hw.summary.chip.record.rec", "active"),
        }
        key, tone = mapping.get(
            self._state.recording_status, ("hw.summary.chip.record.off", "idle")
        )
        return (t(key), tone)

    def _indicator_acq_state(self) -> tuple[str, str]:
        mapping = {
            AcquisitionMode.IDLE: ("hw.summary.chip.acq.idle", "idle"),
            AcquisitionMode.CONTINUOUS: ("hw.summary.chip.acq.run", "active"),
            AcquisitionMode.SCHEDULED: ("hw.summary.chip.acq.sch", "active"),
            AcquisitionMode.FINITE_RUN: ("hw.summary.chip.acq.fin", "active"),
            AcquisitionMode.STEPPED_RUN: ("hw.summary.chip.acq.step", "active"),
            AcquisitionMode.SINGLE_SHOT: ("hw.summary.chip.acq.1fr", "active"),
        }
        key, tone = mapping.get(
            self._state.acquisition_mode, ("hw.summary.chip.acq.idle", "idle")
        )
        return (t(key), tone)

    def _build_planned_frequencies(self) -> list[int]:
        count = int(self._planned_acquisition_count)
        if count <= 0:
            return []
        if not self._frequency_stepping_enabled:
            hz = int(self._device_config.get("frequency_hz", self._planned_start_hz))
            return [hz] * count
        start_hz = int(self._planned_start_hz)
        end_hz = int(self._planned_end_hz)
        if count == 1:
            return [start_hz]
        return [
            int(round(start_hz + (end_hz - start_hz) * idx / (count - 1)))
            for idx in range(count)
        ]

    def _start_planned_acquisition_run(self) -> None:
        self._plan_timer.stop()
        self._plan_frequencies = self._build_planned_frequencies()
        self._plan_completed_count = 0
        self._plan_active = True
        self._planned_step_pending = False
        self._state.set_acquisition_mode(
            AcquisitionMode.STEPPED_RUN
            if self._frequency_stepping_enabled
            else AcquisitionMode.FINITE_RUN
        )
        self._acq_panel.set_acquiring(True)
        self._control_panel.set_enabled(False)
        self._workflow_toolbox.setCurrentIndex(2)
        self._status_bar.showMessage(
            t("main.status.plan_started", count=len(self._plan_frequencies)),
            3000,
        )
        if self._frequency_stepping_enabled:
            self._status_bar.showMessage(
                t("main.status.plan_sweep_note"),
                6000,
            )
        self._refresh_session_summary()
        self._run_next_planned_acquisition()

    @Slot()
    def _run_next_planned_acquisition(self) -> None:
        if not self._plan_active:
            return
        if self._plan_completed_count >= len(self._plan_frequencies):
            self._finish_planned_acquisition_run()
            return

        next_freq = int(self._plan_frequencies[self._plan_completed_count])
        self._device_config["frequency_hz"] = next_freq
        self._sync_state_device_config()
        self._control_panel.set_frequency_value(next_freq)
        self._rebuild_acquisition_pipeline()
        self._planned_step_pending = True
        self._acq_ctrl.capture_one()
        self._status_bar.showMessage(
            t(
                "main.status.plan_step_start",
                current=self._plan_completed_count + 1,
                total=len(self._plan_frequencies),
                hz=next_freq,
            ),
            4000,
        )

    def _finish_planned_acquisition_run(self) -> None:
        completed = self._plan_completed_count
        self._plan_timer.stop()
        self._plan_active = False
        self._planned_step_pending = False
        self._plan_frequencies = []
        self._state.set_acquisition_mode(AcquisitionMode.IDLE)
        self._acq_panel.set_acquiring(False)
        if self._state.connection_status is ConnectionStatus.CONNECTED:
            self._control_panel.set_enabled(True)
            self._workflow_toolbox.setCurrentIndex(2)
        if self._rec_ctrl.is_recording:
            self._rec_ctrl.stop_recording()
            self._state.set_recording_active(False)
        if self._record_requested:
            self._state.set_recording_status(RecordingStatus.ARMED)
        else:
            self._state.set_recording_status(RecordingStatus.OFF)
        self._status_bar.showMessage(
            t("main.status.plan_complete", count=completed), 5000
        )
        self._refresh_session_summary()

    def _open_difference_dialog(self) -> None:
        """Open the Difference dialog using current Hardware-tab frames.

        Wired to the Tools → Difference menu entry (Ctrl+D).  The
        dialog is modeless so the user can keep inspecting the frame
        browser behind it.  Repeated invocations raise the existing
        dialog and refresh its frame list rather than stacking a new
        copy on top.

        If fewer than 2 frames have been recorded the user sees a
        status-bar hint instead of an empty dialog, and the Hardware
        tab is brought forward so the hint is actionable.
        """
        from eit_app.ui.dialogs.difference_dialog import DifferenceDialog

        entries = []
        for row in range(self._frame_browser._model.rowCount()):
            entry = self._frame_browser._model.get_entry(row)
            if entry:
                entries.append(entry)

        if len(entries) < 2:
            self._status_bar.showMessage(
                t("main.status.need_frames_for_difference"), 5000
            )
            # Drop the user on the tab where they can actually fix it.
            self._tab_widget.setCurrentWidget(self._hw_tab)
            return

        ref_index = self._entry_index(entries, self._selected_reference_entry)
        tgt_index = self._entry_index(entries, self._selected_target_entry)
        if tgt_index == ref_index:
            tgt_index = None

        # Single-instance guard: if the dialog is already on screen,
        # refresh its frame list (new recordings may have arrived) and
        # raise it instead of stacking a second window.
        existing = getattr(self, "_difference_dialog", None)
        if existing is not None:
            existing.set_frame_entries(entries)
            existing.show()
            existing.raise_()
            existing.activateWindow()
            return

        dialog = DifferenceDialog(
            entries,
            self,
            default_ref_index=ref_index,
            default_tgt_index=tgt_index,
        )
        dialog.reconstruction_requested.connect(self._on_reconstruction_config)
        dialog.finished.connect(self._on_difference_dialog_finished)
        # Retain the reference so Python GC doesn't collect the
        # modeless dialog the moment this method returns.
        self._difference_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @Slot(int)
    def _on_difference_dialog_finished(self, _result: int) -> None:
        """Clear the single-instance slot when the dialog closes."""
        dialog = getattr(self, "_difference_dialog", None)
        if dialog is not None:
            dialog.deleteLater()
        self._difference_dialog = None

    def _open_batch_reconstruction_from_menu(self) -> None:
        """Tools → Batch Reconstruction menu launcher.

        Delegates to the existing _on_open_batch_dialog() slot with no
        pre-selected session; the dialog itself exposes Browse buttons
        so the user picks input/output folders interactively.
        """
        self._on_open_batch_dialog("")

    def _open_reconstruction_from_menu(self) -> None:
        """Tools → Reconstruction menu launcher.

        The single-frame reconstruction dialog is tied to a specific
        reference / target pair which the user selects in the Database
        tab.  Rather than opening an empty dialog with Run disabled,
        switch to the Database tab and surface a status-bar hint so
        the user knows what to do next.
        """
        self._tab_widget.setCurrentWidget(self._db_tab)
        self._status_bar.showMessage(t("main.status.reconstruction_hint"), 5000)

    @staticmethod
    def _entry_index(entries: list[dict], selected: dict | None) -> int:
        if not selected:
            return 0
        for index, entry in enumerate(entries):
            if entry.get("file_path") == selected.get("file_path"):
                return index
        return 0

    @Slot(dict)
    def _on_reconstruction_config(self, config: dict) -> None:
        ref_entry = config["ref_entry"]
        tgt_entry = config["tgt_entry"]

        try:
            from pyeidors.data.frame_io import read_frame_csv

            ref_real, ref_imag = read_frame_csv(ref_entry["file_path"])
            tgt_real, tgt_imag = read_frame_csv(tgt_entry["file_path"])
        except Exception as exc:
            self._on_error(f"Failed to load frames: {exc}")
            return

        from eit_app.models.frame_model import FrameData

        ref_frame = FrameData(real=ref_real, imag=ref_imag, timestamp=0.0, frame_index=0)
        tgt_frame = FrameData(real=tgt_real, imag=tgt_imag, timestamp=0.0, frame_index=1)

        rc = self._state.reconstruction_config
        request = ReconstructionRequest(
            reference_frame=ref_frame,
            target_frame=tgt_frame,
            use_part=config.get("use_part", rc.use_part),
            method=rc.method,
            regularization_alpha=rc.regularization_alpha,
            max_iterations=rc.max_iterations,
            mesh_dimension=rc.mesh_dimension,
            mesh_refinement=rc.mesh_refinement,
            metadata={
                **self._measurement_layout_config(),
                "difference_mode": config.get("mode", "raw"),
                "difference_orientation": config.get("orientation", "target_minus_reference"),
                **self._hardware_reconstruction_drive_metadata(),
                "geometry_scale_to_m": float(self._device_config.get("geometry_scale_to_m", 1.0)),
                "radius": float(self._device_config.get("radius", 1.0)),
                "contact_impedance": float(self._device_config.get("contact_impedance", 0.01)),
                "electrode_length_m_override": self._device_config.get("electrode_length_m_override"),
                "electrode_coverage": float(self._device_config.get("electrode_coverage", 0.5)),
                "request_source": "hardware_manual",
            },
        )
        # Phase 4: advertise the reconstruction + voltage fit plots as
        # busy while the worker runs.
        self._recon_widget.set_loading(True)
        self._voltage_plot.set_loading(True)
        self._hw_recon_ctrl.reconstruct(request)

    @Slot(dict)
    def _on_db_reconstruct_requested(self, config: dict) -> None:
        """User triggered a reconstruction from the Database tab."""
        target_entry = config.get("target_entry")
        if not target_entry:
            self._on_error("Reconstruction requires at least a target frame.")
            return

        ref_entry = config.get("reference_entry")
        method = config.get("method", "gn-difference")
        use_part = config.get("use_part", "real")

        try:
            from pyeidors.data.frame_io import read_frame_csv
            from eit_app.models.frame_model import FrameData

            tgt_path = target_entry.get("csv_path") or target_entry.get("file_path")
            tgt_real, tgt_imag = read_frame_csv(tgt_path)
            tgt_frame = FrameData(
                real=tgt_real,
                imag=tgt_imag,
                timestamp=float(target_entry.get("timestamp", 0.0)),
                frame_index=int(target_entry.get("frame_index", 0)),
            )

            if ref_entry is not None:
                ref_path = ref_entry.get("csv_path") or ref_entry.get("file_path")
                ref_real, ref_imag = read_frame_csv(ref_path)
                ref_frame = FrameData(
                    real=ref_real,
                    imag=ref_imag,
                    timestamp=float(ref_entry.get("timestamp", 0.0)),
                    frame_index=int(ref_entry.get("frame_index", 0)),
                )
            else:
                # Absolute method — reuse target as a placeholder reference
                # (the worker picks gn-absolute branch and ignores reference)
                ref_frame = tgt_frame
        except Exception as exc:
            self._on_error(f"Failed to load frames for reconstruction: {exc}")
            return

        rc = self._state.reconstruction_config
        request = ReconstructionRequest(
            reference_frame=ref_frame,
            target_frame=tgt_frame,
            use_part=use_part,
            method=method,
            regularization_alpha=float(config.get("regularization_alpha", 1.0)),
            max_iterations=int(config.get("max_iterations", 10)),
            mesh_dimension=rc.mesh_dimension,
            mesh_refinement=rc.mesh_refinement,
            metadata={
                **self._measurement_layout_config(),
                "difference_mode": "raw",
                "difference_orientation": "target_minus_reference",
                **self._hardware_reconstruction_drive_metadata(),
                "geometry_scale_to_m": float(
                    self._device_config.get("geometry_scale_to_m", 1.0)
                ),
                "radius": float(self._device_config.get("radius", 1.0)),
                "contact_impedance": float(
                    self._device_config.get("contact_impedance", 0.01)
                ),
                "electrode_length_m_override": self._device_config.get(
                    "electrode_length_m_override"
                ),
                "electrode_coverage": float(
                    self._device_config.get("electrode_coverage", 0.5)
                ),
                "db_reconstruction": True,
                "db_output_dir": config.get("output_dir"),
                "db_save_recon_image": bool(config.get("save_recon_image", False)),
                "db_save_voltage_fit": bool(config.get("save_voltage_fit", False)),
                "db_method_label": config.get("method_label", method),
                "request_source": "db",
            },
        )
        accepted = self._db_recon_ctrl.reconstruct(request)
        if not accepted:
            return
        self._pending_db_reconstruction = dict(config)
        self._status_bar.showMessage(
            f"Running {config.get('method_label', method)}…", 0
        )

    @Slot(object)
    def _on_db_reconstruction_done(self, result: object) -> None:
        """Persist DB-triggered reconstruction output if requested."""
        config = self._pending_db_reconstruction
        if config is None:
            return
        self._pending_db_reconstruction = None

        if getattr(result, "error_msg", None):
            self._status_bar.showMessage(
                f"Reconstruction failed: {result.error_msg}", 10000
            )
            return

        self._status_bar.showMessage(
            f"Reconstruction complete: {config.get('method_label', '')}", 6000
        )

        # Update the hardware-tab reconstruction display so the user sees it
        try:
            self._recon_widget.update_reconstruction(result)
        except Exception:
            pass

        output_dir = config.get("output_dir")
        if not output_dir:
            return

        try:
            from datetime import datetime

            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            method = str(config.get("method", "recon")).replace("-", "_")
            tgt_idx = (config.get("target_entry") or {}).get("frame_index", "?")
            prefix = f"{stamp}_{method}_tgt{tgt_idx}"

            if config.get("save_recon_image"):
                self._save_reconstruction_image(result, out / f"{prefix}_conductivity.png")
            if config.get("save_voltage_fit"):
                self._save_voltage_fit_plot(result, out / f"{prefix}_voltage_fit.png")

            self._status_bar.showMessage(f"Saved outputs to {out}", 8000)
            # Prompt to open the output folder
            self._offer_open_folder(str(out))
        except Exception as exc:
            log.warning("Failed to save reconstruction outputs: %s", exc)
            self._status_bar.showMessage(f"Save failed: {exc}", 8000)

    def _offer_open_folder(self, folder: str) -> None:
        """Show a non-blocking prompt offering to open the folder."""
        from PySide6.QtWidgets import QMessageBox

        msg = QMessageBox(self)
        msg.setWindowTitle("Reconstruction complete")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("Reconstruction saved successfully.")
        msg.setInformativeText(f"Output folder:\n{folder}")
        open_btn = msg.addButton("Open Folder", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        if msg.clickedButton() is open_btn:
            self._on_open_session_folder(folder)

    def _save_reconstruction_image(self, result, path: Path) -> None:
        """Render conductivity as PNG using matplotlib tripcolor."""
        import matplotlib
        matplotlib.use("Agg", force=False)
        from matplotlib import pyplot as plt
        from matplotlib.tri import Triangulation

        sigma = np.asarray(getattr(result, "conductivity", []), dtype=float).reshape(-1)
        coords = np.asarray(getattr(result, "node_coords", []), dtype=float)
        cells = np.asarray(getattr(result, "cell_connectivity", []), dtype=int)
        if sigma.size == 0 or coords.size == 0 or cells.size == 0:
            return

        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        fig.patch.set_facecolor("#f4f7fb")
        ax.set_facecolor("#fbfdff")
        tri = Triangulation(coords[:, 0], coords[:, 1], cells)
        if sigma.size == len(cells):
            tpc = ax.tripcolor(tri, sigma, shading="flat", cmap="viridis")
        else:
            tpc = ax.tripcolor(tri, sigma, shading="gouraud", cmap="viridis")
        ax.set_aspect("equal")
        ax.set_title("Conductivity reconstruction")
        fig.colorbar(tpc, ax=ax, label="S/m")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _save_voltage_fit_plot(self, result, path: Path) -> None:
        """Render measured vs reconstructed boundary voltages as PNG."""
        import matplotlib
        matplotlib.use("Agg", force=False)
        from matplotlib import pyplot as plt

        measured = getattr(result, "measured", None)
        simulated = getattr(result, "simulated", None)
        if measured is None:
            return
        measured = np.asarray(measured, dtype=float).reshape(-1)
        x = np.arange(1, measured.size + 1)

        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        fig.patch.set_facecolor("#f4f7fb")
        ax.set_facecolor("#fbfdff")
        ax.plot(x, measured, color="#4ecdc4", label="Measured")
        if simulated is not None:
            sim = np.asarray(simulated, dtype=float).reshape(-1)
            ax.plot(x, sim, color="#ff6b6b", linestyle="--", label="Reconstructed fit")
        ax.set_xlabel("Measurement index")
        ax.set_ylabel("Voltage (V)")
        ax.set_title("Boundary voltage fit")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    @Slot(str)
    def _on_open_batch_dialog(self, session_dir: str) -> None:
        """Open the batch reconstruction dialog prefilled with a session folder."""
        from eit_app.ui.dialogs.batch_reconstruction_dialog import (
            BatchReconstructionDialog,
        )

        src = Path(session_dir) if session_dir else None
        # Suggest a default output directory: <app>/results/{session_name}
        default_out = None
        if src is not None and src.exists():
            results_root = Path.cwd() / "results"
            try:
                results_root.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            default_out = results_root / src.name

        dialog = BatchReconstructionDialog(
            default_input=src,
            default_output=default_out,
            parent=self,
        )
        self._batch_dialog = dialog

        dialog.start_requested.connect(self._on_batch_start_requested)
        dialog.cancel_requested.connect(self._batch_recon_ctrl.cancel)
        self._batch_recon_ctrl.progress.connect(dialog.set_progress)
        self._batch_recon_ctrl.finished.connect(dialog.on_finished)
        self._batch_recon_ctrl.error.connect(dialog.on_error)

        dialog.exec()

        # Disconnect to avoid dangling connections with next dialog
        try:
            self._batch_recon_ctrl.progress.disconnect(dialog.set_progress)
            self._batch_recon_ctrl.finished.disconnect(dialog.on_finished)
            self._batch_recon_ctrl.error.disconnect(dialog.on_error)
        except (RuntimeError, TypeError):
            pass
        self._batch_dialog = None

    @Slot(dict)
    def _on_batch_start_requested(self, config: dict) -> None:
        """Launch a batch reconstruction job from the dialog's config."""
        try:
            req = BatchReconstructionRequest(
                input_folder=Path(config["input_folder"]),
                output_folder=Path(config["output_folder"]),
                method=config["method"],
                method_label=config.get("method_label", config["method"]),
                reference_csv=(
                    Path(config["reference_csv"])
                    if config.get("reference_csv")
                    else None
                ),
                use_part=config.get("use_part", "real"),
                regularization_alpha=float(config.get("regularization_alpha", 1.0)),
                max_iterations=int(config.get("max_iterations", 10)),
                save_recon_image=bool(config.get("save_recon_image", True)),
                save_voltage_fit=bool(config.get("save_voltage_fit", True)),
                metadata=self._measurement_layout_config(),
            )
            ok = self._batch_recon_ctrl.start(req)
            if not ok and self._batch_dialog is not None:
                self._batch_dialog.on_error("A batch job is already running.")
        except Exception as exc:
            log.exception("Batch start failed")
            if self._batch_dialog is not None:
                self._batch_dialog.on_error(str(exc))

    @Slot(str)
    def _on_open_session_folder(self, folder: str) -> None:
        """Open a session folder using the OS file manager.

        Handles WSL robustly: if xdg-open is missing, falls back to
        wslview, then explorer.exe with a Windows path, then gio open,
        then QDesktopServices.
        """
        if _open_folder_in_file_manager(folder):
            return
        # Everything failed — this is rare since explorer.exe works on WSL
        # and xdg-open on Linux desktop. Most likely cause: folder doesn't
        # exist or the user has no desktop/Explorer interop at all.
        self._on_error(
            f"Failed to open folder:\n{folder}\n\n"
            "Verify the path exists. On WSL without Windows integration, "
            "install wslu: sudo apt install wslu"
        )

    def _current_hardware_forward_model_config(self) -> ForwardModelConfig:
        return ForwardModelConfig.from_mapping(
            {
                **self._device_config,
                "mesh_dimension": 3 if int(self._device_config.get("mea_mode", 2)) == 3 else 2,
            }
        )

    def _current_sim_forward_model_config(self) -> ForwardModelConfig:
        mesh_cfg = self._sim_tab.mesh_setup_panel.get_config()
        # Pattern controls live on the mesh panel — include them here so
        # the forward solver and the inverse reconstruction share exactly
        # the same PatternConfig (otherwise we get the classic
        # "measurement vector has N columns but pattern expects M" error).
        config = self._sim_forward_model_config.with_overrides(
            mesh_dimension=mesh_cfg["mesh_dimension"],
            mesh_refinement=mesh_cfg["mesh_refinement"],
            mesh_family=mesh_cfg.get("mesh_family", "tetra"),
            n_elec=mesh_cfg["n_electrodes"],
            n_rings=int(mesh_cfg.get("n_rings", 1)),
            electrode_layout=mesh_cfg.get("electrode_layout", "ring_major"),
            background_conductivity=mesh_cfg["background_conductivity"],
            noise_level=self._sim_tab.forward_problem_panel.noise_level,
            measurement_protocol=mesh_cfg.get("measurement_protocol", "eidors_full_3d"),
            custom_pattern_json=mesh_cfg.get("custom_pattern_json", ""),
            stim_pattern=mesh_cfg.get("stim_pattern", "{ad}"),
            meas_pattern=mesh_cfg.get("meas_pattern", "{ad}"),
            rotate_meas=bool(mesh_cfg.get("rotate_meas", True)),
            use_meas_current=bool(mesh_cfg.get("use_meas_current", False)),
            use_meas_current_next=int(mesh_cfg.get("use_meas_current_next", 0)),
            electrode_level_fractions=(
                electrode_level_fractions_for_rings(int(mesh_cfg.get("n_rings", 1)))
                if int(mesh_cfg["mesh_dimension"]) == 3
                else self._sim_forward_model_config.electrode_level_fractions
            ),
        )
        return _with_interactive_3d_geometry_defaults(
            config,
            enabled=self._sim_use_interactive_3d_geometry_defaults,
        )

    @Slot()
    def _sync_sim_inhomogeneity_context(self) -> None:
        config = self._current_sim_forward_model_config()
        self._sim_tab.set_inhomogeneity_domain(
            mesh_dimension=int(config.mesh_dimension),
            radius=float(config.radius),
            height=float(config.height),
            z_center=float(config.z_center),
        )

    def _current_dataset_forward_model_config(self) -> ForwardModelConfig:
        mesh_cfg = self._dataset_tab.mesh_setup_panel.get_config()
        panel_cfg = self._dataset_tab.dataset_generator_panel.get_config()
        config = self._dataset_forward_model_config.with_overrides(
            mesh_dimension=mesh_cfg["mesh_dimension"],
            mesh_refinement=mesh_cfg["mesh_refinement"],
            mesh_family=mesh_cfg.get("mesh_family", "tetra"),
            n_elec=mesh_cfg["n_electrodes"],
            n_rings=int(mesh_cfg.get("n_rings", 1)),
            electrode_layout=mesh_cfg.get("electrode_layout", "ring_major"),
            background_conductivity=mesh_cfg["background_conductivity"],
            noise_level=panel_cfg["noise_level"],
            measurement_protocol=mesh_cfg.get("measurement_protocol", "eidors_full_3d"),
            custom_pattern_json=mesh_cfg.get("custom_pattern_json", ""),
            stim_pattern=mesh_cfg.get("stim_pattern", "{ad}"),
            meas_pattern=mesh_cfg.get("meas_pattern", "{ad}"),
            rotate_meas=bool(mesh_cfg.get("rotate_meas", True)),
            use_meas_current=bool(mesh_cfg.get("use_meas_current", False)),
            use_meas_current_next=int(mesh_cfg.get("use_meas_current_next", 0)),
            electrode_level_fractions=(
                electrode_level_fractions_for_rings(int(mesh_cfg.get("n_rings", 1)))
                if int(mesh_cfg["mesh_dimension"]) == 3
                else self._dataset_forward_model_config.electrode_level_fractions
            ),
        )
        return _with_interactive_3d_geometry_defaults(
            config,
            enabled=self._dataset_use_interactive_3d_geometry_defaults,
        )

    def _interop_reconstruction_preset(self) -> ReconstructionPreset:
        rc = self._state.reconstruction_config
        return ReconstructionPreset(
            method=rc.method,
            regularization_alpha=rc.regularization_alpha,
            max_iterations=rc.max_iterations,
            difference_mode="raw",
            difference_orientation="target_minus_reference",
        )

    def _simulation_measurement_export(self) -> dict[str, np.ndarray] | None:
        if self._last_fwd_result is None or self._last_fwd_result.error_msg:
            return None
        measurements = {
            "target": np.asarray(self._last_fwd_result.boundary_voltages, dtype=float).reshape(-1),
        }
        if self._last_fwd_result.homogeneous_voltages is not None:
            homogeneous = np.asarray(self._last_fwd_result.homogeneous_voltages, dtype=float).reshape(-1)
            measurements["homogeneous"] = homogeneous
            measurements["difference"] = measurements["target"] - homogeneous
        return measurements

    def _recording_measurement_export(self) -> dict[str, np.ndarray] | None:
        if not self._selected_reference_entry or not self._selected_target_entry:
            return None
        try:
            from pyeidors.data.frame_io import read_frame_csv

            ref_real, _ref_imag = read_frame_csv(self._selected_reference_entry["file_path"])
            tgt_real, _tgt_imag = read_frame_csv(self._selected_target_entry["file_path"])
        except Exception as exc:
            log.warning("Failed to build recording export payload: %s", exc)
            return None
        homogeneous = np.asarray(ref_real, dtype=float).reshape(-1)
        target = np.asarray(tgt_real, dtype=float).reshape(-1)
        return {
            "homogeneous": homogeneous,
            "target": target,
            "difference": target - homogeneous,
        }

    def _interop_export_snapshots(self) -> dict[str, dict[str, object]]:
        simulation_cfg = self._current_sim_forward_model_config()
        simulation_measurements = self._simulation_measurement_export()
        simulation_geometry = None
        simulation_notes: list[str] = []
        if self._last_fwd_result is not None and not self._last_fwd_result.error_msg:
            try:
                simulation_geometry = build_geometry_payload_from_result(
                    node_coords=self._last_fwd_result.node_coords,
                    cell_connectivity=self._last_fwd_result.cell_connectivity,
                    forward_model_config=simulation_cfg,
                    truth_elem_data=self._last_fwd_result.ground_truth_conductivity,
                    background=simulation_cfg.background_conductivity,
                    mesh_name="simulation_export",
                    scenario_name="simulation_forward_result",
                )
            except Exception as exc:
                simulation_notes.append(
                    t("main.interop.geometry_generate_failed", error=exc)
                )

        recording_notes: list[str] = []
        recording_measurements = self._recording_measurement_export()
        if recording_measurements is not None:
            recording_notes.append(t("main.interop.export_note_hw_real"))

        snapshots: dict[str, dict[str, object]] = {
            "simulation": {
                "name": "Current Simulation",
                "forward_model_config": simulation_cfg,
                "geometry_payload": simulation_geometry,
                "measurements": simulation_measurements,
                "reconstruction_preset": self._interop_reconstruction_preset(),
                "notes": simulation_notes,
            },
            "hardware": {
                "name": "Current Hardware Layout",
                "forward_model_config": self._current_hardware_forward_model_config(),
                "geometry_payload": self._interop_geometry_asset,
                "measurements": None,
                "reconstruction_preset": self._interop_reconstruction_preset(),
                "notes": [t("main.interop.export_note_hw_no_geom")],
            },
            "recording": {
                "name": "Current Recorded Frames",
                "forward_model_config": self._current_hardware_forward_model_config(),
                "geometry_payload": self._interop_geometry_asset,
                "measurements": recording_measurements,
                "reconstruction_preset": self._interop_reconstruction_preset(),
                "notes": recording_notes,
            },
        }
        return snapshots

    def _apply_reconstruction_preset(self, preset: ReconstructionPreset | None) -> None:
        if preset is None:
            return
        self._state.reconstruction_config.method = preset.method
        self._state.reconstruction_config.regularization_alpha = preset.regularization_alpha
        self._state.reconstruction_config.max_iterations = preset.max_iterations
        self._sim_tab.inverse_problem_panel.set_config(
            {
                "method": preset.method,
                "regularization_alpha": preset.regularization_alpha,
                "max_iterations": preset.max_iterations,
            }
        )

    def _apply_interop_import(self, target: str, loaded_bundle) -> str:
        preview = self._interop_importer.preview_loaded_package(loaded_bundle)
        config = preview.forward_model_config
        self._last_imported_bundle = loaded_bundle
        if loaded_bundle.geometry_payload is not None:
            self._interop_geometry_asset = loaded_bundle.geometry_payload
        if loaded_bundle.measurements is not None:
            self._interop_measurements_asset = loaded_bundle.measurements
        self._apply_reconstruction_preset(loaded_bundle.reconstruction_preset)

        if target == "hardware":
            self._device_config.update(
                {
                    "mea_mode": 3 if int(config.mesh_dimension) == 3 else 2,
                    "n_elec": int(config.n_elec),
                    "n_rings": int(config.n_rings),
                    "electrode_layout": config.electrode_layout,
                    "measurement_protocol": config.measurement_protocol,
                    "custom_pattern_json": config.custom_pattern_json,
                    "custom_stim_matrix": config.custom_stim_matrix,
                    "custom_meas_matrices": config.custom_meas_matrices,
                    "stim_pattern": config.stim_pattern,
                    "meas_pattern": config.meas_pattern,
                    "rotate_meas": bool(config.rotate_meas),
                    "use_meas_current": bool(config.use_meas_current),
                    "use_meas_current_next": int(config.use_meas_current_next),
                    "stim_direction": config.stim_direction,
                    "meas_direction": config.meas_direction,
                    "stim_first_positive": bool(config.stim_first_positive),
                }
            )
            self._device_config = normalize_device_config(self._transport_type, self._device_config)
            self._sync_state_device_config()
            self._tab_widget.setCurrentWidget(self._hw_tab)
            return t(
                "main.interop.applied_to_hw",
                dim=config.display_dimension(),
                n_elec=config.n_elec,
                points=config.point_count(),
            )

        if target == "simulation":
            self._sim_forward_model_config = config
            self._sim_use_interactive_3d_geometry_defaults = False
            self._sim_tab.mesh_setup_panel.set_config(
                {
                    "mesh_dimension": config.mesh_dimension,
                    "mesh_refinement": config.mesh_refinement,
                    "mesh_family": config.mesh_family,
                    "n_electrodes": config.n_elec,
                    "n_rings": int(config.n_rings),
                    "electrode_layout": config.electrode_layout,
                    "background_conductivity": config.background_conductivity,
                    "measurement_protocol": config.measurement_protocol,
                    "custom_pattern_json": config.custom_pattern_json,
                    "stim_pattern": config.stim_pattern,
                    "meas_pattern": config.meas_pattern,
                    "rotate_meas": bool(config.rotate_meas),
                    "use_meas_current": bool(config.use_meas_current),
                    "use_meas_current_next": int(config.use_meas_current_next),
                }
            )
            self._sim_tab.forward_problem_panel.set_noise_level(config.noise_level)
            self._sim_tab.results_widget.set_expected_point_count(config.point_count())
            self._tab_widget.setCurrentWidget(self._sim_tab)
            return t(
                "main.interop.applied_to_sim",
                dim=config.display_dimension(),
                n_elec=config.n_elec,
                points=config.point_count(),
            )

        if target == "dataset":
            self._dataset_forward_model_config = config
            self._dataset_use_interactive_3d_geometry_defaults = False
            self._dataset_tab.mesh_setup_panel.set_config(
                {
                    "mesh_dimension": config.mesh_dimension,
                    "mesh_refinement": config.mesh_refinement,
                    "mesh_family": config.mesh_family,
                    "n_electrodes": config.n_elec,
                    "n_rings": int(config.n_rings),
                    "electrode_layout": config.electrode_layout,
                    "background_conductivity": config.background_conductivity,
                    "measurement_protocol": config.measurement_protocol,
                    "custom_pattern_json": config.custom_pattern_json,
                    "stim_pattern": config.stim_pattern,
                    "meas_pattern": config.meas_pattern,
                    "rotate_meas": bool(config.rotate_meas),
                    "use_meas_current": bool(config.use_meas_current),
                    "use_meas_current_next": int(config.use_meas_current_next),
                }
            )
            self._dataset_tab.dataset_generator_panel.set_config({"noise_level": config.noise_level})
            self._tab_widget.setCurrentWidget(self._dataset_tab)
            return t(
                "main.interop.applied_to_dataset",
                dim=config.display_dimension(),
                n_elec=config.n_elec,
                points=config.point_count(),
            )

        if target == "measurements":
            if loaded_bundle.measurements is None:
                raise RuntimeError(t("main.interop.no_voltage_data"))
            return t("main.interop.voltage_cached")

        if target == "geometry":
            if loaded_bundle.geometry_payload is None:
                raise RuntimeError(t("main.interop.no_geometry"))
            return t("main.interop.geometry_cached")

        raise RuntimeError(t("main.interop.unknown_target", target=target))

    def _run_interop_smoke_validation(self, loaded_bundle) -> str:
        preset = loaded_bundle.reconstruction_preset or self._interop_reconstruction_preset()
        result = self._interop_smoke_validator.validate(
            loaded_bundle,
            reconstruction_preset=preset,
        )
        return str(result.get("message", t("main.interop.smoke_done")))

    def _open_interop_hub(self) -> None:
        from eit_app.ui.dialogs.interop_hub_dialog import InteropHubDialog

        dialog = InteropHubDialog(
            self,
            capture_service=self._interop_capture_service,
            importer=self._interop_importer,
            exporter=self._interop_exporter,
            export_snapshot_provider=self._interop_export_snapshots,
            apply_import_callback=self._apply_interop_import,
            smoke_validate_callback=self._run_interop_smoke_validation,
        )
        dialog.exec()

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        log.error(msg)
        if "power_control" in str(msg).lower():
            self._pending_power_commands.clear()
            self._control_panel.set_power_state(self._state.power_status.value)
        self._state.report_error(msg)
        if str(msg).lower().startswith("connection failed:"):
            self._state.set_connection_status(ConnectionStatus.ERROR)
            self._conn_panel.set_connected(False)
            self._control_panel.set_enabled(False)
            self._workflow_toolbox.setCurrentIndex(0)
            self._refresh_session_summary()
        summary = self._humanize_error_message(msg)
        self._apply_error_help(msg, summary)
        self._status_bar.showMessage(f"Error: {summary}", 15000)

    @staticmethod
    def _summarize_error_message(msg: str) -> str:
        lines = [line.strip() for line in str(msg).splitlines() if line.strip()]
        if not lines:
            return "Unknown error"
        for line in reversed(lines):
            if line.lower().startswith("runtimeerror:"):
                return line
        return lines[-1]

    def _humanize_error_message(self, msg: str) -> str:
        raw = self._summarize_error_message(msg)
        text = raw.lower()

        if "no serial port detected" in text:
            return t("main.hw_error.no_serial_ports")

        # Raw messages in Chinese are upstream-formatted; pass them through
        # untranslated so we don't double-localise already-localised content.
        if "windows 串口" in raw or "未找到串口设备" in raw or "串口 " in raw and "当前无法打开" in raw:
            return raw

        if "4g relay 服务器地址为空" in raw or "无法连接到 4g relay 服务器" in raw:
            return raw

        if "could not configure port" in text or "input/output error" in text:
            return t("main.hw_error.windows_port_invalid")

        if "windows serial bridge failed" in text:
            if (
                "access to the port" in text
                or "access is denied" in text
                or "denied" in text
                or "访问被拒绝" in raw
                or "拒绝访问" in raw
            ):
                return t("main.hw_error.windows_bridge_port_busy")
            if "cannot find the file" in text or "cannot find" in text:
                return t("main.hw_error.windows_bridge_port_missing")
            return t("main.hw_error.windows_bridge_generic")

        if "relay host is empty" in text:
            return t("main.hw_error.relay_host_empty")

        if "connection refused" in text:
            return t("main.hw_error.relay_refused")

        if "timed out" in text and "relay" in text:
            return t("main.hw_error.relay_timeout")

        if "permission denied" in text or "access is denied" in text:
            return t("main.hw_error.port_access_denied")

        return raw

    def _apply_error_help(self, msg: str, summary: str) -> None:
        lowered = str(msg).lower()
        if "serial" in self._transport_type:
            if "connection failed:" in lowered or "serial" in lowered or "com" in lowered:
                self._conn_panel.set_serial_hint(summary)
        if self._transport_type == "relay" and (
            "connection failed:" in lowered or "relay" in lowered
        ):
            self._conn_panel.set_relay_hint(summary)

    # ---- Simulation handlers ----

    @Slot()
    def _on_run_forward(self) -> None:
        mesh_cfg = self._sim_tab.mesh_setup_panel.get_config()
        inhomogeneities = self._sim_tab.inhomogeneity_editor.get_inhomogeneities()
        forward_cfg = self._current_sim_forward_model_config()

        request = ForwardSolverRequest(
            mesh_dimension=mesh_cfg["mesh_dimension"],
            mesh_refinement=mesh_cfg["mesh_refinement"],
            n_electrodes=mesh_cfg["n_electrodes"],
            background_conductivity=mesh_cfg["background_conductivity"],
            inhomogeneities=inhomogeneities,
            noise_level=forward_cfg.noise_level,
            forward_model_config=forward_cfg.to_mapping(),
        )
        self._sim_state.forward_running = True
        self._sim_tab.forward_problem_panel.set_running(True)
        self._sim_tab.inverse_problem_panel.set_save_enabled(False)
        # Phase 4: flag the conductivity image + voltage plot as busy
        # so the user sees a "Solving…" caption instead of a blank or
        # stale panel while the forward solver runs (5-60s range).
        self._sim_tab.results_widget.set_loading_forward(True)
        self._fwd_ctrl.solve(request)

    @Slot(object)
    def _on_forward_done(self, result: ForwardSolverResult) -> None:
        self._sim_state.forward_running = False
        self._sim_tab.forward_problem_panel.set_running(False)
        self._sim_tab.results_widget.set_loading_forward(False)

        if result.error_msg:
            self._sim_tab.forward_problem_panel.set_status(f"Error: {result.error_msg}")
            return

        self._last_fwd_result = result
        self._sim_tab.forward_problem_panel.set_status(
            f"Done: {result.n_elements} elements, {result.n_measurements} measurements"
        )
        self._sim_tab.metrics_panel.clear()
        self._sim_tab.results_widget.update_forward_result(result)

    @Slot()
    def _on_run_sim_inverse(self) -> None:
        if self._last_fwd_result is None or self._last_fwd_result.error_msg:
            self._on_error("Run the forward problem first.")
            return

        result = self._last_fwd_result
        inv_cfg = self._sim_tab.inverse_problem_panel.get_config()
        self._sim_state.inverse_running = True
        self._sim_tab.inverse_problem_panel.set_running(True)
        # Phase 4: show "Reconstructing…" captions on the reconstruction
        # image + voltage plot while the inverse solver runs.
        self._sim_tab.results_widget.set_loading_inverse(True)

        # Build a ReconstructionRequest using the forward result data.
        #
        # pyeidors's forward solver returns a REAL-VALUED measurement vector
        # (no I/Q encoding), so we store the whole vector in `real` and keep
        # `imag` as zeros.  The prior implementation split the vector in
        # half and treated the second half as fake imaginary, which halved
        # the effective measurement count and produced the
        # "got 104 columns, expected 208 columns" mismatch between
        # simulation forward output and the reconstruction pattern.
        from eit_app.models.frame_model import FrameData
        from eit_app.models.precision import compute_dtype
        import numpy as np

        n_meas = len(result.boundary_voltages)
        meas_dtype = compute_dtype()
        zero_imag = np.zeros(n_meas, dtype=meas_dtype)

        homog = (
            np.asarray(result.homogeneous_voltages, dtype=meas_dtype)
            if result.homogeneous_voltages is not None
            else np.zeros(n_meas, dtype=meas_dtype)
        )
        ref_frame = FrameData(
            real=homog,
            imag=zero_imag,
            timestamp=0.0,
            frame_index=0,
        )
        tgt_frame = FrameData(
            real=np.asarray(result.boundary_voltages, dtype=meas_dtype),
            imag=zero_imag,
            timestamp=0.0,
            frame_index=1,
        )

        forward_cfg = self._current_sim_forward_model_config()

        # Map the user's algorithm selection into the runtime path we
        # actually need.  The panel exposes raw eidors-style method keys
        # ("eidors_one_step_noser" / "eidors_abs_gn") for UX continuity,
        # but the reconstruction dispatcher keys off
        # (method + use_part + reconstruction_runtime) — without this
        # mapping a "single-step" selection falls through to the slow
        # iterative GN path, which is why forward→inverse took 40 seconds
        # and the reconstruction barely converged.
        raw_method = str(inv_cfg.get("method", "")).strip().lower()
        difference_preset = "eidors_one_step_noser"
        absolute_preset = "eidors_abs_gn"
        if "demo3d_tv" in raw_method:
            resolved_method = "gn-difference"
            reconstruction_runtime = "full_gn"
            difference_preset = "eidors_demo3d_tv"
        elif any(tag in raw_method for tag in ("one_step", "noser", "step", "gn-difference")):
            resolved_method = "gn-difference"
            reconstruction_runtime = "single_step_cached"
        elif any(tag in raw_method for tag in ("abs", "absolute")):
            resolved_method = "gn-absolute"
            reconstruction_runtime = "full_gn"
        else:
            # Unknown → safest fallback: iterative GN, no single-step cache.
            resolved_method = raw_method or "gn-difference"
            reconstruction_runtime = "full_gn"

        mesh_size = float(forward_cfg.mesh_refinement)
        is_3d_difference = int(forward_cfg.mesh_dimension) == 3 and resolved_method == "gn-difference"
        # Match the 3D paper/pre-experiment sphere scripts: EIDORS-style
        # difference reconstructions are solved in normalized measurement
        # space with one-step NOSER lambda_eff = 1e-2.
        difference_mode = "normalized" if is_3d_difference else "raw"
        difference_lambda = (
            1.0e-2
            if is_3d_difference and reconstruction_runtime == "single_step_cached"
            else None
        )
        metadata = {
            **forward_cfg.to_mapping(),
            **measurement_layout_from_config(forward_cfg.to_mapping()),
            "mesh_size": mesh_size,
            "difference_mode": difference_mode,
            "difference_orientation": "target_minus_reference",
            "difference_preset": difference_preset,
            "absolute_preset": absolute_preset,
            "request_source": "simulation",
            "reconstruction_runtime": reconstruction_runtime,
        }
        if difference_lambda is not None:
            metadata["difference_lambda"] = difference_lambda
        request = ReconstructionRequest(
            reference_frame=ref_frame,
            target_frame=tgt_frame,
            use_part="real",
            method=resolved_method,
            regularization_alpha=inv_cfg["regularization_alpha"],
            max_iterations=inv_cfg["max_iterations"],
            mesh_dimension=forward_cfg.mesh_dimension,
            mesh_refinement=mesh_size,
            metadata=metadata,
        )
        accepted = self._sim_recon_ctrl.reconstruct(request)
        if not accepted:
            self._sim_state.inverse_running = False
            self._sim_tab.inverse_problem_panel.set_running(False)
            self._sim_tab.results_widget.set_loading_inverse(False)
            return

        # Connect one-shot handler for simulation inverse result
        def _on_sim_recon_done(recon_result):
            self._sim_state.inverse_running = False
            self._sim_tab.inverse_problem_panel.set_running(False)
            self._sim_tab.results_widget.set_loading_inverse(False)

            if recon_result.error_msg:
                self._sim_tab.inverse_problem_panel.set_status(
                    f"Error: {recon_result.error_msg}"
                )
                return

            self._sim_tab.inverse_problem_panel.set_status("Reconstruction complete.")
            self._sim_tab.inverse_problem_panel.set_save_enabled(True)
            self._sim_tab.results_widget.update_inverse_result(
                reconstructed_conductivity=recon_result.conductivity,
                node_coords=recon_result.node_coords,
                cell_connectivity=recon_result.cell_connectivity,
            )
            # Surface the difference-voltage fit quality on the boundary
            # plot.  ReconstructionResult.measured = target − reference =
            # the *true* diff voltage induced by the inclusion.
            # ReconstructionResult.simulated = forward solve of the
            # reconstructed σ minus the reference solve = the *predicted*
            # diff voltage — what the reconstruction would produce if we
            # re-simulated it.  Plotting both side-by-side tells the user
            # how well their reconstructed σ matches the measured data.
            try:
                measured = getattr(recon_result, "measured", None)
                simulated = getattr(recon_result, "simulated", None)
                if measured is not None and simulated is not None:
                    plot_dtype = compute_dtype()
                    self._sim_tab.results_widget.voltage_plot.update_simulation_voltages(
                        ground_truth=np.asarray(measured, dtype=plot_dtype).reshape(-1),
                        reconstructed=np.asarray(simulated, dtype=plot_dtype).reshape(-1),
                    )
            except Exception as exc:
                log.warning(
                    "Failed to update simulation voltage plot with recon fit: %s", exc
                )
            self._sim_tab.metrics_panel.update_metrics(
                self._last_fwd_result.ground_truth_conductivity,
                recon_result.conductivity,
            )

        # Disconnect previous one-shot connections and reconnect
        try:
            self._sim_recon_ctrl.reconstruction_done.disconnect(self._sim_recon_handler)
        except (RuntimeError, AttributeError):
            pass
        self._sim_recon_handler = _on_sim_recon_done
        self._sim_recon_ctrl.reconstruction_done.connect(self._sim_recon_handler)

    @Slot()
    def _on_save_sim_results(self) -> None:
        if self._last_fwd_result is None:
            return

        from PySide6.QtWidgets import QFileDialog
        import numpy as np

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation Results", "", "NumPy archive (*.npz)"
        )
        if not path:
            return

        result = self._last_fwd_result
        np.savez(
            path,
            ground_truth=result.ground_truth_conductivity,
            boundary_voltages=result.boundary_voltages,
            homogeneous_voltages=result.homogeneous_voltages,
            node_coords=result.node_coords,
            cell_connectivity=result.cell_connectivity,
        )
        self._status_bar.showMessage(f"Saved to {path}", 5000)

    @Slot()
    def _on_generate_dataset(self) -> None:
        panel_cfg = self._dataset_tab.dataset_generator_panel.get_config()
        mesh_cfg = self._dataset_tab.mesh_setup_panel.get_config()

        if not panel_cfg["output_dir"]:
            self._on_error("Please specify an output directory for the dataset.")
            return

        forward_cfg = self._current_dataset_forward_model_config()
        config = DatasetGeneratorConfig(
            n_samples=panel_cfg["n_samples"],
            output_dir=panel_cfg["output_dir"],
            n_inhomogeneities_min=panel_cfg["n_inhomogeneities_min"],
            n_inhomogeneities_max=panel_cfg["n_inhomogeneities_max"],
            shapes=panel_cfg["shapes"],
            position_min=panel_cfg["position_min"],
            position_max=panel_cfg["position_max"],
            size_min=panel_cfg["size_min"],
            size_max=panel_cfg["size_max"],
            conductivity_min=panel_cfg["conductivity_min"],
            conductivity_max=panel_cfg["conductivity_max"],
            background_conductivity_min=panel_cfg["background_conductivity_min"],
            background_conductivity_max=panel_cfg["background_conductivity_max"],
            noise_level=forward_cfg.noise_level,
            mesh_dimension=forward_cfg.mesh_dimension,
            mesh_refinement=forward_cfg.mesh_refinement,
            n_electrodes=forward_cfg.n_elec,
        )
        self._sim_state.dataset_running = True
        self._dataset_tab.set_generating(True)
        self._dataset_tab.set_progress(0, panel_cfg["n_samples"])
        self._dataset_ctrl.generate(
            DatasetGeneratorRequest(
                config=config,
                forward_model_config=forward_cfg.to_mapping(),
            )
        )

    @Slot(int)
    def _on_dataset_done(self, total: int) -> None:
        self._sim_state.dataset_running = False
        self._dataset_tab.set_generating(False)
        if total > 0:
            self._dataset_tab.set_progress(total, total)
        else:
            self._dataset_tab.set_progress(0, 0)
        self._status_bar.showMessage(f"Dataset generation complete: {total} samples.", 10000)

    def closeEvent(self, event) -> None:
        self._db_tab.prepare_for_shutdown()
        self._on_stop_acquisition()
        self._recon_prewarm_timer.stop()
        if self._state.connection_status is ConnectionStatus.CONNECTED:
            try:
                self._device_ctrl.power_off_device()
            except Exception as exc:
                log.warning("Failed to power off device during shutdown: %s", exc)
        self._device_ctrl.shutdown()
        self._recon_ctrl.shutdown()
        self._recon_prewarm_ctrl.shutdown()
        self._hw_recon_ctrl.shutdown()
        self._db_recon_ctrl.shutdown()
        self._sim_recon_ctrl.shutdown()
        self._fwd_ctrl.shutdown()
        self._dataset_ctrl.shutdown()
        try:
            self._batch_recon_ctrl.shutdown()
        except Exception as exc:
            log.warning("Batch reconstruction shutdown failed: %s", exc)
        try:
            self._db_ctrl.shutdown()
        except Exception as exc:
            log.warning("Database shutdown failed: %s", exc)
        super().closeEvent(event)
