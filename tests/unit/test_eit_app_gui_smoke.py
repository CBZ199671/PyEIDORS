from __future__ import annotations

import gc
import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt, QThread
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import eit_app.ui.main_window as main_window_module
from eit_app.controllers import reconstruction_controller as rc
from eit_app.controllers.forward_solver_controller import ForwardSolverResult
from eit_app.hardware.connection_preflight import ConnectionPreflightResult
from eit_app.hardware.serial_port_discovery import SerialPortDescriptor
from eit_app.measurement_layout import estimate_measurement_point_count
from eit_app.models.app_state import ConnectionStatus
from eit_app.models.frame_model import FrameData
from eit_app.controllers.reconstruction_controller import ReconstructionResult
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.conductivity_image_widget import ConductivityImageWidget
from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget
from eit_app.ui.main_window import EITWorkstation
from pyeidors.data.frame_io import read_frame_yaml, read_session_metadata
from pyeidors.runtime_paths import pyeidors_cache_path


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _NoopSignal:
    def connect(self, _slot) -> None:
        return None


def _wait_until(predicate, *, timeout: float = 5.0, step: float = 0.02) -> bool:
    app = _get_app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(step)
    app.processEvents()
    return bool(predicate())


def _show_window(window: EITWorkstation) -> None:
    app = _get_app()
    window.show()
    app.processEvents()


def _click(widget) -> None:
    if hasattr(widget, "click"):
        widget.click()
    else:
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
    _get_app().processEvents()


def _fps_value(window: EITWorkstation) -> float:
    return float(window._status_bar._fps_label.text().split(": ", 1)[1])


def _first_electrode_arc_span(widget: ReconstructionWidget) -> float:
    x_data, y_data = widget._electrode_arc_item.getData()
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    split = np.flatnonzero(~finite)
    end = int(split[0]) if split.size else len(x)
    x = x[:end]
    y = y[:end]
    angles = np.unwrap(np.arctan2(y, x))
    return float(abs(angles[-1] - angles[0]))


def _boundary_radius(widget: ReconstructionWidget) -> float:
    x_data, y_data = widget._boundary_item.getData()
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    return float(np.nanmax(np.hypot(x, y)))


def _connect_simulator(window: EITWorkstation) -> None:
    # Drop the simulator's per-frame sleep: SimulatorDevice.read_frame()
    # calls time.sleep(1.0/fps) at the start of every frame (defaults to
    # fps=30 → 33 ms/frame).  Tests don't care about real-time pacing;
    # they just want frames to flow through the pipeline.  Setting fps
    # to 0 disables the sleep entirely (see the guard in read_frame).
    # This cuts ~3-5 seconds off each simulator-heavy smoke test.
    window._on_connect_requested("simulator", {"simulator_fps": 0})


def _splitter_has_center_priority(splitter) -> bool:
    sizes = splitter.sizes()
    return len(sizes) == 3 and sizes[1] > sizes[0] and sizes[1] > sizes[2]


def test_v361_main_window_save_helpers_preserve_display_dtype() -> None:
    recon_source = inspect.getsource(EITWorkstation._save_reconstruction_image)
    voltage_source = inspect.getsource(EITWorkstation._save_voltage_fit_plot)

    assert "dtype=float" not in recon_source
    assert "dtype=int" not in recon_source
    assert "dtype=float" not in voltage_source
    assert '_display_float_array(getattr(result, "conductivity", []))' in recon_source
    assert (
        '_display_int_array(getattr(result, "cell_connectivity", []))' in recon_source
    )
    assert "_display_float_array(measured)" in voltage_source
    assert "_display_float_array(simulated)" in voltage_source

    values = np.array([1.0, 2.0], dtype=np.float32)
    display_values = main_window_module._display_float_array(values)

    assert display_values.dtype == np.dtype(np.float32)
    assert np.shares_memory(display_values, values)

    cells = np.array([[0, 1, 2]], dtype=np.int32)
    display_cells = main_window_module._display_int_array(cells)

    assert display_cells.dtype == np.dtype(np.int32)
    assert np.shares_memory(display_cells, cells)


def test_v362_main_window_voltage_paths_preserve_display_dtype(monkeypatch) -> None:
    auto_source = inspect.getsource(EITWorkstation._on_auto_reconstruction_done)
    hardware_source = inspect.getsource(EITWorkstation._on_hardware_reconstruction_done)
    recording_source = inspect.getsource(EITWorkstation._recording_measurement_export)

    assert "dtype=np.float64" not in auto_source
    assert "dtype=float" not in hardware_source
    assert "dtype=float" not in recording_source
    assert "_display_float_array(backend_measured)" in auto_source
    assert "_display_float_array(simulated)" in auto_source
    assert "_display_float_array(measured)" in hardware_source
    assert "_display_float_array(reconstructed)" in hardware_source
    assert "_display_float_array(ref_real)" in recording_source
    assert "_display_float_array(tgt_real)" in recording_source

    class _CaptureVoltagePlot:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, np.ndarray | None]] = []

        def set_loading(self, _on: bool) -> None:
            return None

        def update_hardware_voltages(
            self, measured: np.ndarray, reconstructed: np.ndarray | None
        ) -> None:
            self.calls.append((measured, reconstructed))

    class _NoopWidget:
        def set_loading(self, _on: bool) -> None:
            return None

        def update_reconstruction(self, _result: object) -> None:
            return None

    ref_frame = SimpleNamespace(real=np.array([1.0, 2.0], dtype=np.float32))
    tgt_frame = SimpleNamespace(real=np.array([1.5, 2.75], dtype=np.float32))
    simulated = np.array([0.45, 0.7], dtype=np.float32)
    auto_plot = _CaptureVoltagePlot()
    auto_window = SimpleNamespace(
        _reconstruction_result_source=lambda _result: "hardware_auto_live",
        _auto_recon_busy=True,
        _auto_reconstruct=False,
        _pending_auto_target_frame=None,
        _recon_widget=_NoopWidget(),
        _equipotential_widget=_NoopWidget(),
        _last_auto_ref_frame=ref_frame,
        _last_auto_tgt_frame=tgt_frame,
        _voltage_plot=auto_plot,
        _status_bar=SimpleNamespace(showMessage=lambda *args, **kwargs: None),
    )

    EITWorkstation._on_auto_reconstruction_done(
        auto_window,
        SimpleNamespace(error_msg=None, measured=None, simulated=simulated),
    )

    measured_diff, simulated_arr = auto_plot.calls[-1]
    assert measured_diff.dtype == np.dtype(np.float32)
    assert simulated_arr is not None
    assert simulated_arr.dtype == np.dtype(np.float32)
    assert np.shares_memory(simulated_arr, simulated)

    measured = np.array([0.1, 0.2], dtype=np.float32)
    reconstructed = np.array([0.11, 0.18], dtype=np.float32)
    hardware_plot = _CaptureVoltagePlot()
    hw_tab = object()
    hardware_window = SimpleNamespace(
        _recon_widget=_NoopWidget(),
        _equipotential_widget=_NoopWidget(),
        _voltage_plot=hardware_plot,
        _tab_widget=SimpleNamespace(currentWidget=lambda: hw_tab),
        _hw_tab=hw_tab,
        _reconstruction_result_source=lambda _result: "hardware_manual",
        _auto_reconstruct=False,
        _last_auto_tgt_frame=None,
    )

    EITWorkstation._on_hardware_reconstruction_done(
        hardware_window,
        SimpleNamespace(measured=measured, simulated=reconstructed),
    )

    measured_arr, reconstructed_arr = hardware_plot.calls[-1]
    assert measured_arr.dtype == np.dtype(np.float32)
    assert reconstructed_arr is not None
    assert reconstructed_arr.dtype == np.dtype(np.float32)
    assert np.shares_memory(measured_arr, measured)
    assert np.shares_memory(reconstructed_arr, reconstructed)

    ref_csv = np.array([1.0, 2.0], dtype=np.float32)
    tgt_csv = np.array([1.1, 2.3], dtype=np.float32)

    def _fake_read_frame_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
        values = ref_csv if path == "ref.csv" else tgt_csv
        return values, np.zeros_like(values)

    import pyeidors.data.frame_io as frame_io

    monkeypatch.setattr(frame_io, "read_frame_csv", _fake_read_frame_csv)
    recording_window = SimpleNamespace(
        _selected_reference_entry={"file_path": "ref.csv"},
        _selected_target_entry={"file_path": "tgt.csv"},
    )

    payload = EITWorkstation._recording_measurement_export(recording_window)

    assert payload is not None
    assert payload["homogeneous"].dtype == np.dtype(np.float32)
    assert payload["target"].dtype == np.dtype(np.float32)
    assert payload["difference"].dtype == np.dtype(np.float32)
    assert np.shares_memory(payload["homogeneous"], ref_csv)
    assert np.shares_memory(payload["target"], tgt_csv)


def _close_window(window: EITWorkstation) -> None:
    window.close()
    assert _wait_until(lambda: not window._device_ctrl._thread.isRunning(), timeout=3.0)


def _as_wsl_unc(path: Path) -> str:
    posix_path = str(path)
    if not posix_path.startswith("/"):
        raise ValueError("Expected absolute POSIX path")
    return "\\\\wsl.localhost\\Ubuntu-22.04" + posix_path.replace("/", "\\")


def test_v126_default_db_path_honors_explicit_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "isolated.sqlite"
    monkeypatch.setenv("EIT_APP_DB_PATH", str(db_path))

    assert EITWorkstation._default_db_path() == db_path


@pytest.mark.gui
def test_v127_database_reconstruction_uses_advanced_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyeidors.data.frame_io as frame_io

    window = EITWorkstation()
    _show_window(window)
    captured: list[object] = []

    monkeypatch.setattr(
        frame_io,
        "read_frame_csv",
        lambda _path: (
            np.array([1.0, 2.0, 3.0], dtype=float),
            np.zeros(3, dtype=float),
        ),
    )
    monkeypatch.setattr(
        window._db_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_db_reconstruct_requested(
        {
            "method": "gn-absolute",
            "method_label": "Absolute GN",
            "target_entry": {
                "frame_index": 5,
                "csv_path": "/tmp/target.csv",
                "timestamp": 1.0,
            },
            "regularization_alpha": 0.2,
            "max_iterations": 7,
            "use_part": "real",
            "mesh_dimension": 3,
            "mesh_refinement": 0.12,
            "reconstruction_settings": {
                "mesh_dimension": 3,
                "mesh_refinement": 0.12,
                "mesh_size": 0.12,
                "n_elec": 32,
                "n_rings": 2,
                "stim_pattern": "{op}",
                "meas_pattern": "{ad}",
                "radius": 0.18,
                "mesh_height": 0.16,
                "contact_impedance": 0.02,
                "drive_mode": "total_current",
                "drive_value": 2.5e-5,
                "solver_mode": "strict",
            },
        }
    )

    try:
        assert captured
        request = captured[0]
        assert request.mesh_dimension == 3
        assert request.mesh_refinement == 0.12
        assert request.max_iterations == 7
        assert request.metadata["n_elec"] == 32
        assert request.metadata["n_rings"] == 2
        assert request.metadata["stim_pattern"] == "{op}"
        assert request.metadata["radius"] == 0.18
        assert request.metadata["mesh_height"] == 0.16
        assert request.metadata["contact_impedance"] == 0.02
        assert request.metadata["drive_value"] == 2.5e-5
        assert request.metadata["solver_mode"] == "strict"
    finally:
        _close_window(window)


def _sample_pixmap_unique_rgb_count(pixmap, *, samples_per_axis: int = 8) -> int:
    image = pixmap.toImage()
    if image.isNull():
        return 0
    width = image.width()
    height = image.height()
    xs = np.linspace(0, width - 1, min(samples_per_axis, width), dtype=int)
    ys = np.linspace(0, height - 1, min(samples_per_axis, height), dtype=int)
    return len({int(image.pixel(int(x), int(y))) & 0x00FFFFFF for x in xs for y in ys})


@pytest.fixture(autouse=True)
def _cleanup_top_level_widgets_after_test():
    _get_app()
    yield
    app = _get_app()
    app.processEvents()
    for widget in list(app.topLevelWidgets()):
        try:
            if isinstance(widget, EITWorkstation):
                _close_window(widget)
            else:
                widget.close()
        except Exception:
            pass
    app.processEvents()
    for obj in gc.get_objects():
        try:
            if isinstance(obj, QThread) and obj.isRunning():
                obj.requestInterruption()
                obj.quit()
                # 500ms is plenty for a cooperative shutdown; previously
                # 3000ms, which tripled smoke-suite runtime whenever a
                # worker thread sat in a blocking wait.  Any thread that
                # misses 500ms likely is deadlocked anyway and the test
                # has already failed — better to surface that fast.
                obj.wait(500)
        except Exception:
            pass
    try:
        from eit_app.ui.theme import _mode_listeners

        _mode_listeners.clear()
    except Exception:
        pass
    app.processEvents()


@pytest.mark.gui
def test_interop_hub_dialog_hot_swaps_language() -> None:
    """Interop Hub must refresh every labeled chrome when language flips.

    Covers the Phase 3d.4 migration: dialog title, tab titles, status
    box/row labels, step group titles, primary buttons, and the sentinel
    state text on the manual-input status rows.
    """
    from eit_app.i18n import set_language, t
    from eit_app.ui.dialogs.interop_hub_dialog import InteropHubDialog

    app = _get_app()
    set_language("en", persist=False)
    dialog = InteropHubDialog()
    dialog.show()
    app.processEvents()
    try:
        assert dialog.windowTitle() == t("dlg.interop.title")
        assert dialog._tabs.tabText(0) == t("dlg.interop.tabs.import")
        assert dialog._tabs.tabText(1) == t("dlg.interop.tabs.export")
        assert dialog._tabs.tabText(2) == t("dlg.interop.tabs.profiles")
        assert dialog._preview_btn.text() == t("dlg.interop.actions.preview_button")
        assert dialog._status_box.title() == t("dlg.interop.status.title")
        # "Not set" sentinel because no env was picked yet.
        assert dialog._status_value_labels["matlab"].text() == t(
            "dlg.interop.status.unspecified"
        )

        set_language("zh", persist=False)
        assert dialog.windowTitle() == t("dlg.interop.title")
        assert dialog._tabs.tabText(0) == t("dlg.interop.tabs.import")
        assert dialog._preview_btn.text() == t("dlg.interop.actions.preview_button")
        assert dialog._status_box.title() == t("dlg.interop.status.title")
        assert dialog._status_value_labels["matlab"].text() == t(
            "dlg.interop.status.unspecified"
        )
    finally:
        set_language("en", persist=False)
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_status_bar_reserves_error_tone_for_real_failures() -> None:
    """Tone-palette guard: 'error' (red) must only mark actual failures.

    Previously the recording-in-progress chip was painted red ("error"
    tone), which users reasonably read as "something is broken" — the
    exact opposite of the real state (capture running normally).

    The mode chip also used to cycle through four different tones per
    tab, implying (e.g.) the Dataset tab carried a warning and the
    Database tab was dormant.  Both are regular operating modes.
    """
    from eit_app.ui.status_bar import (
        _ACQ_KEYS,
        _LINK_KEYS,
        _MODE_KEYS,
        _POWER_KEYS,
        _RECORD_KEYS,
    )

    # recording == "recording" must be the 'active' tone, not 'error'.
    assert _RECORD_KEYS["recording"][1] == "active"
    # Mode chips are purely mode indicators — no per-tab threat levels.
    assert {tone for _, tone in _MODE_KEYS.values()} == {"active"}
    # Every tone we assign must resolve to a known palette entry.
    from eit_app.ui.theme import tone_palette

    known_tones = {"idle", "warn", "ready", "active", "error"}
    for mapping in (_LINK_KEYS, _POWER_KEYS, _ACQ_KEYS, _RECORD_KEYS, _MODE_KEYS):
        for _, tone in mapping.values():
            assert tone in known_tones, f"Unknown tone: {tone!r}"
            # tone_palette should return a 3-tuple (fg, bg, border).
            assert len(tone_palette(tone)) == 3


def test_runtime_diagnostics_are_hidden_for_regular_gui_users(monkeypatch) -> None:
    monkeypatch.delenv("EIT_APP_SHOW_RUNTIME_DIAGNOSTICS", raising=False)

    text = main_window_module._format_runtime_diagnostics(
        {
            "mesh_family": "tetra",
            "forward_backend_effective": "dolfinx",
            "petsc_device_effective": "cuda",
            "forward_solver_preset": "spd_gamg",
            "petsc_amgx_available": False,
            "forward_solver_policy_reason": "amgx_unavailable_downgraded_to_spd_gamg",
            "torch_device": "cuda",
        }
    )

    assert text == ""


def test_runtime_diagnostics_explains_amgx_cuda_downgrade_for_developers() -> None:
    text = main_window_module._format_runtime_diagnostics(
        {
            "mesh_family": "tetra",
            "forward_backend_effective": "dolfinx",
            "petsc_device_effective": "cuda",
            "forward_solver_preset": "spd_gamg",
            "petsc_amgx_available": False,
            "forward_solver_policy_reason": "amgx_unavailable_downgraded_to_spd_gamg",
            "torch_device": "cuda",
        },
        developer=True,
    )

    assert "solver=spd_gamg" in text
    assert "AmgX=false" in text
    assert "AmgX 不可用时使用 spd_gamg CUDA" in text


def test_every_pushbutton_in_ui_package_has_a_role_tag() -> None:
    """Lint-style check: no bare QPushButton without set_button_role(...).

    A button without a role inherits the neutral white fill, which
    breaks visual hierarchy — users can't tell "Run" from "Browse" at
    a glance.  This is a static source scan, not a Qt runtime check,
    so it runs fast and catches regressions right at authoring time.
    """
    import re
    from pathlib import Path

    ui_dir = Path(__file__).resolve().parents[2] / "src" / "eit_app" / "ui"
    missing: list[str] = []
    total = 0
    for py_file in ui_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for match in re.finditer(r"(self\.\w+)\s*=\s*QPushButton\(", text):
            total += 1
            name = match.group(1)
            role_pattern = (
                rf"set_button_role\(\s*{re.escape(name)}\s*,\s*['\"](\w+)['\"]\)"
            )
            if not re.search(role_pattern, text):
                missing.append(
                    f"{py_file.relative_to(ui_dir.parent.parent.parent)}: {name}"
                )

    assert not missing, (
        f"Found {len(missing)} QPushButton(s) without set_button_role "
        f"(total buttons: {total}):\n  " + "\n  ".join(missing)
    )


@pytest.mark.gui
def test_live_recon_voltage_widgets_expose_tri_state_overlay() -> None:
    """Phase 4: four plot widgets must accept set_loading(True/False)
    and swap their overlay to a 'loading' caption that also follows
    the active UI language (not get stuck on the English version).
    """
    from eit_app.i18n import set_language, t
    from eit_app.ui.simulation.simulation_results_widget import SimulationResultsWidget

    set_language("en", persist=False)
    window = EITWorkstation()
    _show_window(window)
    try:
        lp = window._live_plot
        rw = window._recon_widget
        vp = window._voltage_plot

        # All three start in the "empty" state (overlay visible, uses
        # the localized empty-placeholder text).
        assert lp._overlay_state == "empty"
        assert rw._overlay_mode == "empty"
        assert vp._overlay_state == "empty"

        # Flip all three to loading.
        lp.set_loading(True)
        rw.set_loading(True)
        vp.set_loading(True)
        _get_app().processEvents()
        assert lp._empty_overlay.text() == t("hw.live_plot.loading_overlay")
        assert rw._empty_overlay.text() == t("hw.reconstruction.loading_overlay")
        assert vp._empty_overlay.text() == t("voltage_plot.loading_overlay")

        # Language switch while loading must refresh every overlay.
        set_language("zh", persist=False)
        _get_app().processEvents()
        assert lp._empty_overlay.text() == t("hw.live_plot.loading_overlay")
        assert rw._empty_overlay.text() == t("hw.reconstruction.loading_overlay")
        assert vp._empty_overlay.text() == t("voltage_plot.loading_overlay")

        # Turning loading off reverts to the empty placeholder copy.
        lp.set_loading(False)
        rw.set_loading(False)
        vp.set_loading(False)
        _get_app().processEvents()
        assert lp._overlay_state == "empty"
        assert rw._overlay_mode == "empty"
        assert vp._overlay_state == "empty"

        # The SimulationResultsWidget forwards to its children without
        # crashing on back-to-back loading toggles.  Use English captions
        # here so matplotlib doesn't emit the "CJK glyph missing" warning
        # on Linux CI runners that lack CJK fonts.
        set_language("en", persist=False)
        _get_app().processEvents()
        sr: SimulationResultsWidget = window._sim_tab._results_widget
        sr.set_loading_forward(True)
        _get_app().processEvents()
        sr.set_loading_forward(False)
        sr.set_loading_inverse(True)
        _get_app().processEvents()
        sr.set_loading_inverse(False)
    finally:
        set_language("en", persist=False)
        _close_window(window)


@pytest.mark.gui
def test_workflow_shell_context_layouts_match_tab_roles() -> None:
    """Hardware / Dataset keep right context panels, Simulation does not.

    Simulation moved its metrics + run guide into the left rail so the
    central truth/reconstruction/voltage plots get the right-column width.
    """
    window = EITWorkstation()
    _show_window(window)
    try:
        expected_context_min = 220

        # Hardware → FrameBrowser is the context widget
        assert window._hw_tab._frame_browser.minimumWidth() == expected_context_min

        # Simulation → metrics and guide live in the left rail footer,
        # leaving only left controls + center visualizations in the splitter.
        sim_splitter = window._sim_tab._shell._main_splitter
        assert sim_splitter.count() == 2
        assert sim_splitter.widget(1) is window._sim_tab.results_widget
        assert (
            window._sim_tab.metrics_panel.parent() is window._sim_tab._left_status_panel
        )
        assert (
            window._sim_tab._run_guide_box.parent()
            is window._sim_tab._left_status_panel
        )
        window._tab_widget.setCurrentWidget(window._sim_tab)
        _get_app().processEvents()
        assert window._sim_tab._left_status_panel.isVisible()
        assert window._sim_tab._left_status_panel.height() > 0

        # Dataset → DatasetSummaryPanel is the context widget
        assert window._dataset_tab._summary_panel.minimumWidth() == expected_context_min

        # Tabs that still expose a right pane request the same default
        # so their context width doesn't visibly jump on tab switch.
        expected_default = 240
        for shell in (
            window._hw_tab._shell,
            window._dataset_tab._shell,
        ):
            assert shell._splitter_sizes[2] == expected_default
        assert window._sim_tab._shell._splitter_sizes == [460, 720]
    finally:
        _close_window(window)


@pytest.mark.gui
def test_simulation_metrics_panel_reports_mesh_sizes() -> None:
    from eit_app.i18n import current_language, set_language
    from eit_app.ui.simulation.metrics_panel import MetricsPanel

    _get_app()
    previous_language = current_language()
    set_language("en", persist=False)
    panel = MetricsPanel()
    try:
        node_coords = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        cells = np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.int32,
        )

        panel.update_mesh_stats(
            ground_truth_node_coords=node_coords,
            ground_truth_cell_connectivity=cells,
        )

        assert panel.title() == "Mesh & Metrics"
        assert panel._truth_mesh_label.text() == "5 nodes / 4 elements"
        assert panel._recon_mesh_label.text() == "\u2014"

        panel.update_metrics(
            np.ones(len(cells), dtype=float),
            np.ones(len(cells), dtype=float),
            ground_truth_node_coords=node_coords,
            ground_truth_cell_connectivity=cells,
            reconstructed_node_coords=node_coords,
            reconstructed_cell_connectivity=cells,
        )

        assert panel._truth_mesh_label.text() == "5 nodes / 4 elements"
        assert panel._recon_mesh_label.text() == "5 nodes / 4 elements"
        assert panel._l2_label.text() == "0.0000"
        set_language("zh", persist=False)
        assert panel._truth_mesh_label.text() == "\u8282\u70b9 5 / \u5143\u7d20 4"
        assert panel._recon_mesh_label.text() == "\u8282\u70b9 5 / \u5143\u7d20 4"
    finally:
        set_language(previous_language, persist=False)


@pytest.mark.gui
def test_batch_reconstruction_dialog_shows_eta_after_enough_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch progress label should grow an ETA after 2+ items finish
    and >= 1s of elapsed time.  Before that it stays on the plain
    current/total form so the user doesn't see wildly fluctuating
    estimates during the first hot-cache iteration.
    """
    import time as real_time
    from eit_app.ui.dialogs.batch_reconstruction_dialog import BatchReconstructionDialog

    _get_app()
    dialog = BatchReconstructionDialog(default_input=None, default_output=None)
    dialog.show()
    _get_app().processEvents()

    try:
        # Fake a steady clock so the assertions are deterministic.
        clock = {"now": 1000.0}

        def _fake_monotonic() -> float:
            return clock["now"]

        monkeypatch.setattr(real_time, "monotonic", _fake_monotonic)
        # Hit both the module-local import (inside _set_running /
        # _format_eta, which does ``import time as _time``).
        import eit_app.ui.dialogs.batch_reconstruction_dialog as mod

        monkeypatch.setattr(mod, "__name__", mod.__name__)

        dialog._set_running(True)
        assert dialog._run_started_at == 1000.0

        # Too early: only 1 item done, elapsed < 1s → no ETA suffix.
        clock["now"] = 1000.5
        dialog.set_progress(1, 10)
        assert (
            "ETA" not in dialog._progress_label.text()
            and "剩余" not in dialog._progress_label.text()
        )

        # 2 items done after 2s elapsed → rate 1/s → ETA for the
        # remaining 8 items ≈ 8s.  The label must contain a
        # localised "remaining" phrase.
        clock["now"] = 1002.0
        dialog.set_progress(2, 10)
        assert (
            "remaining" in dialog._progress_label.text()
            or "剩余" in dialog._progress_label.text()
        )

        # Non-empty message must bypass ETA decoration (used for
        # cancel / error paths).
        dialog.set_progress(3, 10, "Cancelling…")
        assert dialog._progress_label.text() == "Cancelling…"
    finally:
        dialog.close()
        dialog.deleteLater()
        _get_app().processEvents()


def test_frame_database_range_filters_on_n_elec_and_stim_amp(tmp_path: Path) -> None:
    """New n_elec_min/max and stim_amp_ua_min/max filters.

    Not gated on @pytest.mark.gui — FrameDatabase is a pure SQLite
    wrapper with no Qt widgets, so this runs fast as a normal unit.
    """
    from eit_app.models.frame_database import FrameDatabase

    db = FrameDatabase(tmp_path / "range.db")
    try:
        for name, n_elec, stim in [("A", 16, 100), ("B", 32, 500), ("C", 64, 1000)]:
            db.add_session(
                tmp_path / name,
                {
                    "name": name,
                    "started_at": f"2026-01-{int(n_elec / 16):02d}",
                    "n_elec": n_elec,
                    "frequency_hz": 1000,
                    "stim_amp_uA": stim,
                },
            )

        def names(rows):
            return sorted(r["name"] for r in rows)

        # n_elec inclusive range
        assert names(db.query_sessions(n_elec_min=32)) == ["B", "C"]
        assert names(db.query_sessions(n_elec_max=32)) == ["A", "B"]
        assert names(db.query_sessions(n_elec_min=32, n_elec_max=32)) == ["B"]

        # stim_amp_uA inclusive range
        assert names(db.query_sessions(stim_amp_ua_min=500)) == ["B", "C"]
        assert names(db.query_sessions(stim_amp_ua_max=500)) == ["A", "B"]

        # Combined
        assert names(
            db.query_sessions(n_elec_min=16, n_elec_max=32, stim_amp_ua_max=500)
        ) == ["A", "B"]

        # Empty result when ranges don't overlap any row
        assert names(db.query_sessions(n_elec_min=128)) == []
    finally:
        db.close()


@pytest.mark.gui
def test_bilingual_tab_capture_produces_stable_png_pixmaps() -> None:
    """Phase 10: verify the QWidget.grab() path used by the
    capture_bilingual_screenshots script produces non-trivial
    pixmaps in both languages.  Size-pin the window so the test
    actually exercises the layout (not a 100×100 default).
    """
    from eit_app.i18n import set_language

    window = EITWorkstation()
    window.resize(1280, 800)
    _show_window(window)
    app = _get_app()
    try:
        for lang in ("en", "zh"):
            set_language(lang, persist=False)
            app.processEvents()
            for index in range(window._tab_widget.count()):
                window._tab_widget.setCurrentIndex(index)
                for _ in range(3):
                    app.processEvents()
                pixmap = window.grab()
                assert not pixmap.isNull(), (
                    f"grab() returned null pixmap for tab {index} lang {lang}"
                )
                # Full-window grabs should be meaningful size; a 0×0 or
                # 1×1 result means the offscreen renderer broke.
                assert pixmap.width() >= 800
                assert pixmap.height() >= 500
    finally:
        set_language("en", persist=False)
        _close_window(window)


@pytest.mark.gui
def test_main_window_registers_keyboard_shortcuts() -> None:
    """Power users expect Ctrl+1..4 for tabs, F5 / Ctrl+Enter for solves.

    Qt lets each shortcut live on the menu action OR on an independent
    QShortcut; both are asserted here so a future refactor that moves
    one path to the other still trips the test.
    """
    window = EITWorkstation()
    _show_window(window)
    try:
        # Menu action shortcuts (Settings removed — File menu has only Exit)
        assert window._action_exit.shortcut().toString() == "Ctrl+Q"
        assert window._action_interop_hub.shortcut().toString() == "Ctrl+I"
        assert not hasattr(window, "_action_settings")

        # Tab jump shortcuts (Ctrl+1..Ctrl+4)
        assert [sc.key().toString() for sc in window._tab_shortcuts] == [
            f"Ctrl+{i}" for i in range(1, window._tab_widget.count() + 1)
        ]

        # Simulation action shortcuts
        assert window._sim_forward_shortcut.key().toString() == "F5"
        assert window._sim_inverse_shortcut_enter.key().toString() == "Ctrl+Return"

        # Activating Ctrl+3 should switch to the Dataset tab
        window._tab_widget.setCurrentIndex(0)
        window._tab_shortcuts[2].activated.emit()
        _get_app().processEvents()
        assert window._tab_widget.currentIndex() == 2

        # F5 outside the Simulation tab is a no-op.
        window._tab_widget.setCurrentIndex(0)  # Hardware
        # Temporarily wire a flag to the forward button click.
        fired = {"count": 0}
        window._sim_tab.forward_problem_panel._solve_btn.clicked.connect(
            lambda: fired.__setitem__("count", fired["count"] + 1)
        )
        window._sim_forward_shortcut.activated.emit()
        _get_app().processEvents()
        assert fired["count"] == 0, "F5 outside Simulation tab should not fire"

        # Flip to Simulation tab and try again
        window._tab_widget.setCurrentIndex(1)
        window._sim_forward_shortcut.activated.emit()
        _get_app().processEvents()
        assert fired["count"] == 1
    finally:
        _close_window(window)


@pytest.mark.gui
def test_forward_inverse_panels_toggle_busy_indicator_on_set_running() -> None:
    """set_running must reveal the busy bar AND lock adjacent inputs.

    Prior to this behavior the only feedback for a 30-40s solve was the
    disabled primary button, which is a weak cue.  The indeterminate
    QProgressBar (range 0-0) gives a visible "something's happening"
    hint, and disabling parameter editors prevents users from kicking
    off a second solve with different parameters mid-flight.
    """
    from eit_app.ui.simulation.forward_problem_panel import ForwardProblemPanel
    from eit_app.ui.simulation.inverse_problem_panel import InverseProblemPanel
    from PySide6.QtWidgets import QSizePolicy

    app = _get_app()

    fwd = ForwardProblemPanel()
    fwd.show()
    app.processEvents()
    assert fwd._busy_bar.isHidden()
    assert fwd._solve_btn.isEnabled()
    assert fwd._noise_spin.isEnabled()

    fwd.set_running(True)
    app.processEvents()
    assert not fwd._busy_bar.isHidden(), "busy bar should show while running"
    assert not fwd._solve_btn.isEnabled()
    assert not fwd._noise_spin.isEnabled()

    fwd.set_running(False)
    app.processEvents()
    assert fwd._busy_bar.isHidden()
    assert fwd._solve_btn.isEnabled()
    assert fwd._noise_spin.isEnabled()
    fwd.close()
    fwd.deleteLater()

    inv = InverseProblemPanel()
    inv.show()
    app.processEvents()
    assert inv._busy_bar.isHidden()
    assert inv._status_label.wordWrap()
    assert inv._status_label.minimumWidth() == 0
    assert (
        inv._status_label.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Ignored
    )
    assert inv._method_combo.isEnabled()
    assert not inv._alpha_spin.isEnabled()
    assert inv._alpha_spin.value() == pytest.approx(1.0e-2)
    assert inv._iter_spin.isHidden()
    assert not inv._iter_spin.isEnabled()

    inv.set_running(True)
    app.processEvents()
    assert not inv._busy_bar.isHidden()
    assert not inv._recon_btn.isEnabled()
    assert not inv._method_combo.isEnabled()
    assert not inv._alpha_spin.isEnabled()
    assert not inv._iter_spin.isEnabled()

    inv.set_running(False)
    app.processEvents()
    assert inv._busy_bar.isHidden()
    assert inv._recon_btn.isEnabled()
    assert inv._method_combo.isEnabled()
    assert not inv._alpha_spin.isEnabled()
    assert inv._iter_spin.isHidden()
    assert not inv._iter_spin.isEnabled()
    inv.close()
    inv.deleteLater()
    app.processEvents()


@pytest.mark.gui
def test_simulation_inverse_panel_uses_spec_route_labels() -> None:
    from eit_app.ui.simulation.inverse_problem_panel import (
        SIMULATION_INVERSE_METHODS,
        InverseProblemPanel,
        simulation_inverse_methods_for_mesh_dimension,
    )

    app = _get_app()
    inv = InverseProblemPanel()
    inv.show()
    app.processEvents()

    methods = [inv._method_combo.itemText(i) for i in range(inv._method_combo.count())]
    assert methods == simulation_inverse_methods_for_mesh_dimension(2)
    assert "eidors_one_step_noser" not in methods
    assert "eidors_abs_gn" not in methods
    assert "eidors_demo3d_tv" not in methods
    assert "absolute_gn" in methods
    assert "pseudo3d_noser_rm" not in methods
    assert inv.get_config()["method"] == "noser_rm"
    assert inv._method_combo.toolTip()

    inv.set_config({"method": "eidors_one_step_noser"})
    assert inv.get_config()["method"] == "debug_fine_mesh_noser"

    inv.set_config({"method": "eidors_abs_gn"})
    assert inv.get_config()["method"] == "absolute_gn"
    assert "Absolute" in inv._method_combo.toolTip()

    inv.set_config({"method": "noser_rm"})
    assert inv.get_config()["method"] == "noser_rm"
    assert "RM" in inv._method_combo.toolTip()

    inv.set_source_mesh_dimension(3)
    methods = [inv._method_combo.itemText(i) for i in range(inv._method_combo.count())]
    assert methods == SIMULATION_INVERSE_METHODS
    assert "pseudo3d_noser_rm" in methods

    inv.set_config({"method": "pseudo3d"})
    assert inv.get_config()["method"] == "pseudo3d_noser_rm"
    assert "Pseudo" in inv._method_combo.toolTip()

    inv.close()
    inv.deleteLater()
    app.processEvents()


@pytest.mark.gui
def test_simulation_pseudo3d_route_visible_only_for_3d_forward_mesh() -> None:
    window = EITWorkstation()
    _show_window(window)

    panel = window._sim_tab.inverse_problem_panel
    methods = [
        panel._method_combo.itemText(i) for i in range(panel._method_combo.count())
    ]
    assert "pseudo3d_noser_rm" not in methods

    window._sim_tab.mesh_setup_panel._dim_combo.setCurrentIndex(1)
    _get_app().processEvents()
    methods = [
        panel._method_combo.itemText(i) for i in range(panel._method_combo.count())
    ]
    assert "pseudo3d_noser_rm" in methods
    panel.set_config({"method": "pseudo3d_noser_rm"})
    assert panel.get_config()["method"] == "pseudo3d_noser_rm"

    window._sim_tab.mesh_setup_panel._dim_combo.setCurrentIndex(0)
    _get_app().processEvents()
    methods = [
        panel._method_combo.itemText(i) for i in range(panel._method_combo.count())
    ]
    assert "pseudo3d_noser_rm" not in methods
    assert panel.get_config()["method"] == "noser_rm"

    _close_window(window)


@pytest.mark.gui
def test_simulation_inverse_panel_hyperparameter_semantics_follow_route() -> None:
    from eit_app.ui.simulation.inverse_problem_panel import InverseProblemPanel

    app = _get_app()
    inv = InverseProblemPanel()
    inv.show()
    app.processEvents()

    assert inv.get_config()["method"] == "noser_rm"
    assert "λ_eff" in inv._lbl_alpha.text()
    assert not inv._alpha_spin.isEnabled()
    assert inv._alpha_spin.value() == pytest.approx(1.0e-2)
    assert inv.get_config()["regularization_alpha"] == pytest.approx(1.0e-2)
    assert not inv.get_config()["lambda_eff_custom_enabled"]
    assert not inv._custom_lambda_check.isHidden()
    assert inv._custom_lambda_check.isEnabled()
    assert not inv._custom_lambda_check.isChecked()
    assert "hp=0.1" in inv._alpha_spin.toolTip()

    inv.set_config({"method": "debug_full_gn", "regularization_alpha": 2.5})
    app.processEvents()
    assert inv._alpha_spin.isEnabled()
    assert inv._custom_lambda_check.isHidden()
    assert "α" in inv._lbl_alpha.text()
    assert inv._alpha_spin.value() == pytest.approx(2.5)
    assert inv.get_config()["regularization_alpha"] == pytest.approx(2.5)

    inv.set_config({"method": "greit", "regularization_alpha": 3.5})
    app.processEvents()
    assert inv._lbl_alpha.isHidden()
    assert inv._alpha_spin.isHidden()
    assert inv.get_config()["regularization_alpha"] == pytest.approx(3.5)
    assert not inv._greit_group.isHidden()
    assert inv.get_config()["greit_weight"] == pytest.approx(3.5)

    inv.close()
    inv.deleteLater()
    app.processEvents()


@pytest.mark.gui
def test_simulation_inverse_panel_greit_advanced_controls_are_serialized() -> None:
    from eit_app.ui.simulation.inverse_problem_panel import InverseProblemPanel

    app = _get_app()
    inv = InverseProblemPanel()
    inv.show()
    app.processEvents()

    assert inv.get_config()["method"] == "noser_rm"
    assert inv._greit_group.isHidden()

    inv.set_config(
        {
            "method": "greit",
            "greit_desired_image_mode": "adaptive_gauss",
            "greit_training_target_count": 128,
            "greit_target_size": 0.15,
            "greit_weight": 0.75,
            "greit_use_cached_rm": False,
            "greit_rebuild_rm": True,
        }
    )
    app.processEvents()

    cfg = inv.get_config()
    assert inv._greit_group.isVisible()
    assert cfg["regularization_alpha"] == pytest.approx(0.75)
    assert cfg["greit_desired_image_mode"] == "adaptive_gauss"
    assert cfg["greit_training_target_count"] == 128
    assert cfg["greit_target_size"] == pytest.approx(0.15)
    assert cfg["greit_weight_strategy"] == "fixed"
    assert cfg["greit_noise_figure"] is None
    assert cfg["greit_weight"] == pytest.approx(0.75)
    assert cfg["greit_use_cached_rm"] is False
    assert cfg["greit_rebuild_rm"] is True
    assert "冷构建" in inv._greit_cold_build_hint.text() or "cold" in (
        inv._greit_cold_build_hint.text().lower()
    )

    inv.set_running(True)
    app.processEvents()
    assert not inv._greit_group.isEnabled()

    inv.set_running(False)
    app.processEvents()
    assert inv._greit_group.isEnabled()

    inv.set_config(
        {
            "method": "greit",
            "greit_weight_strategy": "eidors_nf1",
            "greit_weight": 0.75,
        }
    )
    app.processEvents()
    cfg = inv.get_config()
    assert cfg["greit_weight_strategy"] == "eidors_nf1"
    assert cfg["greit_noise_figure"] == pytest.approx(1.0)
    assert cfg["regularization_alpha"] == pytest.approx(1.0)
    assert not inv._greit_weight_spin.isEnabled()

    inv.close()
    inv.deleteLater()
    app.processEvents()


@pytest.mark.gui
def test_simulation_inverse_panel_v117_custom_lambda_rebuild_entry() -> None:
    from eit_app.ui.simulation.inverse_problem_panel import InverseProblemPanel

    app = _get_app()
    inv = InverseProblemPanel()
    inv.show()
    app.processEvents()

    inv.set_config(
        {
            "method": "noser_rm",
            "lambda_eff_custom_enabled": True,
            "custom_lambda_eff": 0.04,
        }
    )
    app.processEvents()
    assert inv._custom_lambda_check.isChecked()
    assert inv._alpha_spin.isEnabled()
    assert "自定义" in inv._lbl_alpha.text() or "custom" in inv._lbl_alpha.text()
    assert inv._alpha_spin.value() == pytest.approx(0.04)
    assert inv.get_config()["lambda_eff_custom_enabled"] is True
    assert inv.get_config()["custom_lambda_eff"] == pytest.approx(0.04)

    inv._alpha_spin.setValue(0.09)
    app.processEvents()
    assert inv.get_config()["regularization_alpha"] == pytest.approx(0.09)
    assert inv.get_config()["custom_lambda_eff"] == pytest.approx(0.09)

    inv._custom_lambda_check.setChecked(False)
    app.processEvents()
    assert not inv.get_config()["lambda_eff_custom_enabled"]
    assert not inv._alpha_spin.isEnabled()
    assert inv._alpha_spin.value() == pytest.approx(1.0e-2)

    inv.set_config({"method": "greit", "lambda_eff_custom_enabled": True})
    app.processEvents()
    assert inv._custom_lambda_check.isHidden()
    assert not inv.get_config()["lambda_eff_custom_enabled"]

    inv.close()
    inv.deleteLater()
    app.processEvents()


@pytest.mark.gui
def test_simulation_inverse_panel_v116_iterations_only_for_absolute_route() -> None:
    from eit_app.ui.simulation.inverse_problem_panel import InverseProblemPanel

    app = _get_app()
    inv = InverseProblemPanel()
    inv.show()
    app.processEvents()

    assert inv.get_config()["method"] == "noser_rm"
    assert inv._iter_spin.isHidden()
    assert inv._lbl_iter.isHidden()
    assert not inv._iter_spin.isEnabled()

    inv.set_config({"method": "debug_full_gn", "max_iterations": 7})
    app.processEvents()
    assert inv.get_config()["method"] == "debug_full_gn"
    assert inv._iter_spin.isHidden()
    assert not inv._iter_spin.isEnabled()
    assert inv.get_config()["max_iterations"] == 7

    inv.set_config({"method": "absolute_gn", "max_iterations": 7})
    app.processEvents()
    assert inv.get_config()["method"] == "absolute_gn"
    assert not inv._iter_spin.isHidden()
    assert not inv._lbl_iter.isHidden()
    assert inv._iter_spin.isEnabled()
    assert inv._iter_spin.toolTip()

    inv.set_running(True)
    app.processEvents()
    assert not inv._iter_spin.isHidden()
    assert not inv._iter_spin.isEnabled()

    inv.set_running(False)
    app.processEvents()
    assert not inv._iter_spin.isHidden()
    assert inv._iter_spin.isEnabled()

    inv.close()
    inv.deleteLater()
    app.processEvents()


def test_theme_arrow_svg_is_hidpi_friendly_and_parses_via_qsvg() -> None:
    """All 8 spinbox/date-edit arrow URLs must be DPR-aware.

    The previous hardcoded SVGs set intrinsic width="10" height="10"
    on the <svg> root, which pins Qt's stylesheet rasterisation to
    exactly 10×10 px and produces blurry arrows on 2×/3× DPR displays.
    The fix removes the intrinsic size so Qt scales the viewBox to
    whatever device-pixel size the compositor requests.

    This test guards the contract at three levels:
      1. The decoded SVG payload must contain ``viewBox`` and must NOT
         contain ``width=`` / ``height=`` on the root element.
      2. The dark-mode arrow palette must use different fills from
         light (verified by parsing the fill color out of the SVG).
      3. Each URL must parse cleanly via QSvgRenderer — a smoke check
         that the base64 encoding round-trips and the SVG is valid XML.
    """
    import base64
    import re
    from PySide6.QtCore import QByteArray
    from PySide6.QtSvg import QSvgRenderer

    from eit_app.ui.theme import (
        _ARROW_URLS_DARK,
        _ARROW_URLS_LIGHT,
        _arrow_data_url,
    )

    _get_app()  # QSvgRenderer needs a QApplication

    def _decode(url: str) -> str:
        prefix = 'url("data:image/svg+xml;base64,'
        assert url.startswith(prefix), f"unexpected URL form: {url[:60]}"
        b64 = url[len(prefix) :].rstrip('")')
        return base64.b64decode(b64).decode("utf-8")

    for name, url in {
        **_ARROW_URLS_LIGHT,
        **{f"dk_{k}": v for k, v in _ARROW_URLS_DARK.items()},
    }.items():
        svg = _decode(url)
        # Level 1: viewBox-only, no intrinsic raster size.
        assert "viewBox=" in svg, f"{name}: missing viewBox"
        assert 'width="' not in svg, f"{name}: still has width= attribute"
        assert 'height="' not in svg, f"{name}: still has height= attribute"
        # Level 3: Qt accepts it as a valid SVG.
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        assert renderer.isValid(), f"{name}: QSvgRenderer rejected the payload"

    # Level 2: dark fills are brighter than light fills (higher
    # luminance approximated by the channel sum).  Compare idle up
    # arrows as the canonical representative.
    def _fill(url: str) -> str:
        m = re.search(r'fill="(#[0-9a-fA-F]{6})"', _decode(url))
        assert m is not None
        return m.group(1).lower()

    light_fill = _fill(_ARROW_URLS_LIGHT["up_idle"])
    dark_fill = _fill(_ARROW_URLS_DARK["up_idle"])
    assert light_fill != dark_fill
    # Dark palette chose #a7b2c2 (luma ≈ 175), light chose #5b6573
    # (luma ≈ 100).  Assert the dark variant has >30% higher sum
    # so chromatic drift alone can't pass the check.
    light_sum = sum(int(light_fill[i : i + 2], 16) for i in (1, 3, 5))
    dark_sum = sum(int(dark_fill[i : i + 2], 16) for i in (1, 3, 5))
    assert dark_sum > light_sum * 1.3, (
        f"Dark arrow fill {dark_fill} (sum {dark_sum}) should be noticeably "
        f"brighter than light {light_fill} (sum {light_sum})"
    )

    # Cross-check the factory helper produces the same output shape.
    url = _arrow_data_url("up", "#ff0000")
    svg = _decode(url)
    assert 'fill="#ff0000"' in svg
    assert "viewBox=" in svg
    assert 'width="' not in svg


@pytest.mark.gui
def test_inhomogeneity_editor_uses_explicit_column_widths_no_overlap() -> None:
    """Inhomogeneity column headers must not collide.

    Headers carry single-character labels (X / Y / Z / 长 / 宽 / 高 / σ);
    units now live in a hint line above the table so each numeric column
    can shrink to ~44 px without clipping.  The editor still has 2D and 3D
    layouts: 2D hides Z / depth, 3D reveals them.
    """
    from eit_app.ui.simulation.inhomogeneity_editor import InhomogeneityEditor

    _get_app()
    editor = InhomogeneityEditor()
    editor.resize(280, 240)
    editor.show()
    _get_app().processEvents()
    try:
        header = editor._table.horizontalHeader()
        assert not editor._table.verticalHeader().isVisible()
        widths = [header.sectionSize(c) for c in range(editor._model.columnCount())]
        # Shape column wider than the numeric columns; X/Y/W/H share size.
        assert widths[0] >= 70, f"Shape column too narrow: {widths[0]}"
        for c in (1, 2, 4, 5):
            assert widths[c] >= 40, f"Numeric column {c} too narrow: {widths[c]}"
        assert editor._table.isColumnHidden(3)
        assert editor._table.isColumnHidden(6)

        editor.set_domain_context(mesh_dimension=3, radius=0.18, height=0.16)
        _get_app().processEvents()
        assert not editor._table.isColumnHidden(3)
        assert not editor._table.isColumnHidden(6)
        editor._add_shape("circle")
        spec = editor.get_inhomogeneities()[0]
        assert spec.size_x < 0.18
        assert spec.size_z <= 0.16 * 0.5
        assert spec.center_z == pytest.approx(0.0)
        # σ column (last, stretched) takes whatever is left ≥ 40px.
        assert header.sectionSize(7) >= 40
    finally:
        editor.close()
        editor.deleteLater()
        _get_app().processEvents()


@pytest.mark.gui
def test_v108_inhomogeneity_editor_height_is_full_size_for_3d_paint() -> None:
    from eit_app.controllers.forward_solver_controller import _paint_shape
    from eit_app.ui.simulation.inhomogeneity_editor import InhomogeneityEditor

    _get_app()
    editor = InhomogeneityEditor()
    editor.set_domain_context(mesh_dimension=3, radius=0.18, height=0.16)
    editor.show()
    _get_app().processEvents()
    try:
        editor._add_shape("ellipse")
        height_index = editor._model.index(0, 6)
        spec = editor.get_inhomogeneities()[0]
        assert editor._model.data(height_index) == pytest.approx(spec.size_z * 2.0)

        assert editor._model.setData(height_index, 0.1)
        spec = editor.get_inhomogeneities()[0]
        assert spec.size_z == pytest.approx(0.05)

        centers = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.07]], dtype=float)
        values = np.ones(centers.shape[0], dtype=float)
        _paint_shape(values, centers, spec, mesh_dimension=3)

        assert values == pytest.approx([2.0, 1.0])
    finally:
        editor.close()
        editor.deleteLater()
        _get_app().processEvents()


@pytest.mark.gui
def test_v109_inhomogeneity_editor_sphere_size_columns_stay_linked() -> None:
    from eit_app.ui.simulation.inhomogeneity_editor import InhomogeneityEditor

    _get_app()
    editor = InhomogeneityEditor()
    editor.set_domain_context(mesh_dimension=3, radius=0.18, height=0.16)
    editor.show()
    _get_app().processEvents()
    try:
        editor._add_shape("circle")
        height_index = editor._model.index(0, 6)

        assert editor._model.setData(height_index, 0.1)
        spec = editor.get_inhomogeneities()[0]
        assert spec.size_x == pytest.approx(0.05)
        assert spec.size_y == pytest.approx(0.05)
        assert spec.size_z == pytest.approx(0.05)
        for col in (4, 5, 6):
            assert editor._model.data(editor._model.index(0, col)) == pytest.approx(0.1)
    finally:
        editor.close()
        editor.deleteLater()
        _get_app().processEvents()


def test_v119_inhomogeneity_boundary_warning_uses_visible_full_diameter() -> None:
    from eit_app.models.simulation_state import InhomogeneitySpec
    from eit_app.ui.simulation.inhomogeneity_editor import (
        inhomogeneity_boundary_violations,
    )

    centered_offset_sphere = InhomogeneitySpec(
        shape="circle",
        center_x=0.05,
        center_y=0.05,
        center_z=0.0,
        size_x=0.063,
        size_y=0.063,
        size_z=0.063,
    )
    mistaken_radius_sphere = InhomogeneitySpec(
        shape="circle",
        center_x=0.05,
        center_y=0.05,
        center_z=0.0,
        size_x=0.126,
        size_y=0.126,
        size_z=0.126,
    )

    assert (
        inhomogeneity_boundary_violations(
            [centered_offset_sphere],
            mesh_dimension=3,
            radius=0.18,
            height=0.5,
            z_center=0.0,
        )
        == []
    )
    assert inhomogeneity_boundary_violations(
        [mistaken_radius_sphere],
        mesh_dimension=3,
        radius=0.18,
        height=0.5,
        z_center=0.0,
    ) == [1]


@pytest.mark.gui
def test_v119_inhomogeneity_editor_shows_boundary_warning_label() -> None:
    from eit_app.ui.simulation.inhomogeneity_editor import InhomogeneityEditor

    _get_app()
    editor = InhomogeneityEditor()
    editor.set_domain_context(mesh_dimension=3, radius=0.18, height=0.5)
    editor.show()
    _get_app().processEvents()
    try:
        editor._add_shape("circle")
        assert editor._model.setData(editor._model.index(0, 1), 0.13)
        assert editor._model.setData(editor._model.index(0, 4), 0.126)
        _get_app().processEvents()

        assert editor._boundary_warning.isVisible()
        assert "1" in editor._boundary_warning.text()

        assert editor._model.setData(editor._model.index(0, 1), 0.05)
        assert editor._model.setData(editor._model.index(0, 2), 0.05)
        _get_app().processEvents()
        assert not editor._boundary_warning.isVisible()
    finally:
        editor.close()
        editor.deleteLater()
        _get_app().processEvents()


@pytest.mark.gui
def test_dark_stylesheet_uses_muted_section_chrome() -> None:
    from eit_app.ui.theme import _build_stylesheet

    css = _build_stylesheet("dark")
    section_chrome = css.split("/* === GroupBox / section panels === */", 1)[1]
    section_chrome = section_chrome.split("/* === Inputs === */", 1)[0]
    data_chrome = css.split("/* === Tables / trees / lists === */", 1)[1]
    data_chrome = data_chrome.split("/* === Scrollbars === */", 1)[0]

    assert "#8fc8ea" not in section_chrome
    assert "#b3d4ed" not in section_chrome
    assert "#4d6188" not in section_chrome
    assert "QGroupBox::title {\n    color: #a7b2c2;" in section_chrome
    assert "border: 1px solid #323a45;" in section_chrome
    assert "QHeaderView::section {\n    background: #252c36;" in data_chrome
    assert "QTableCornerButton::section {\n    background: #252c36;" in data_chrome
    assert (
        "QHeaderView::section:first {\n    border-top-left-radius: 0px;" in data_chrome
    )
    assert (
        "QHeaderView::section:last {\n    border-top-right-radius: 0px;" in data_chrome
    )
    assert "QAbstractScrollArea::corner {\n    background: #252c36;" in data_chrome
    assert "border-bottom: 1px solid #323a45;" in data_chrome


@pytest.mark.gui
def test_inline_card_stylesheets_follow_dark_mode_palette() -> None:
    """Mini cards painted via setStyleSheet (Session/Dataset summary
    panels, Database stats card, Database selection status, Database
    backfill subtitle) must re-paint when the user toggles dark mode.

    Before this fix each of those widgets used a hardcoded light
    background (#f5f9fd / #f7f9fc / #edf4fb) and stayed bright when
    the rest of the chrome flipped to dark.  All five paths now read
    from theme.card_palette() and re-apply on each theme_mode flip.
    """
    from eit_app.ui.theme import (
        card_palette,
        set_theme_mode,
    )
    from eit_app.ui.database.database_tab import DatabaseTab
    from eit_app.ui.hardware.session_summary_panel import SessionSummaryPanel
    from eit_app.ui.simulation.dataset_summary_panel import DatasetSummaryPanel

    app = _get_app()
    set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
    light_value_bg = card_palette()["value_bg"]

    session_summary = SessionSummaryPanel()
    dataset_summary = DatasetSummaryPanel()
    database_tab = DatabaseTab(
        SimpleNamespace(
            query_sessions=lambda **_filters: [],
            is_shutting_down=False,
            session_added=_NoopSignal(),
            frame_added=_NoopSignal(),
            backfill_progress=_NoopSignal(),
            backfill_done=_NoopSignal(),
        )
    )
    try:
        # Initial state: every card stylesheet contains the LIGHT bg color.
        sample_session_value = next(iter(session_summary._values.values()))
        sample_dataset_value = next(iter(dataset_summary._values.values()))
        assert light_value_bg in sample_session_value.styleSheet()
        assert light_value_bg in sample_dataset_value.styleSheet()
        assert light_value_bg.lower() != "#262d38"

        # Flip to dark and verify all 5 paths swap.
        set_theme_mode(app, "dark", persist=False, apply_stylesheet=False)
        _get_app().processEvents()
        dark_palette = card_palette()
        dark_value_bg = dark_palette["value_bg"]

        # SessionSummaryPanel: field value boxes + next-action banner
        for value in session_summary._values.values():
            assert dark_value_bg in value.styleSheet(), (
                "SessionSummaryPanel field value should follow dark mode"
            )
        assert (
            dark_palette["next_action_bg"] in session_summary._next_action.styleSheet()
        )

        # DatasetSummaryPanel: field value boxes
        for value in dataset_summary._values.values():
            assert dark_value_bg in value.styleSheet()

        # Database tab: stats card + selection status
        assert dark_palette["info_bg"] in database_tab._stats_card.styleSheet()
        assert dark_palette["info_accent"] in database_tab._count_label.styleSheet()
        assert (
            dark_palette["selection_bg"] in database_tab._selection_status.styleSheet()
        )

        # Light mode reverts the same paths.
        set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
        _get_app().processEvents()
        for value in session_summary._values.values():
            assert light_value_bg in value.styleSheet()
        assert "#edf4fb" in session_summary._next_action.styleSheet()
    finally:
        set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
        for widget in (session_summary, dataset_summary, database_tab):
            widget.close()
            widget.deleteLater()
        app.processEvents()


@pytest.mark.gui
def test_plot_widgets_repaint_canvas_when_dark_mode_toggles() -> None:
    """Dark mode must reach the four plot widgets (LivePlot, Voltage,
    Reconstruction, ConductivityImage), not just the QSS chrome.

    pyqtgraph + matplotlib paint their own canvases — they don't
    honour QSS — so the widgets subscribe to theme_mode_changed and
    re-pull from plot_palette() on each flip.  Regress that the
    background colors actually swap on toggle, and swap back when
    the user returns to light.
    """
    from eit_app.ui.theme import (
        plot_palette,
        set_theme_mode,
    )
    from eit_app.ui.boundary_voltage_plot_widget import BoundaryVoltagePlotWidget
    from eit_app.ui.hardware.live_plot_widget import LivePlotWidget
    from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget

    app = _get_app()
    set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
    light_palette = plot_palette()

    live_plot = LivePlotWidget()
    voltage_plot = BoundaryVoltagePlotWidget()
    recon_widget = ReconstructionWidget()
    gt = ConductivityImageWidget()
    try:
        # Initial widgets pulled the light palette.
        assert live_plot._plot_bg == light_palette["bg"]
        assert voltage_plot._plot_bg == light_palette["bg"]
        assert recon_widget._plot_bg == light_palette["panel_bg"]

        # The simulation results widget now wraps the matplotlib pane
        # in a dispatcher slot (_ConductivityViewSlot) that flips
        # between 2D matplotlib and 3D PyVista for 3D meshes — reach
        # through ._mpl to get back to the matplotlib Figure.
        light_facecolor = gt._figure.patch.get_facecolor()

        # Flip to dark.
        set_theme_mode(app, "dark", persist=False, apply_stylesheet=False)
        _get_app().processEvents()
        dark_palette = plot_palette()

        assert live_plot._plot_bg == dark_palette["bg"]
        assert voltage_plot._plot_bg == dark_palette["bg"]
        assert recon_widget._plot_bg == dark_palette["panel_bg"]
        # Matplotlib facecolor is a 4-tuple (r,g,b,a) of floats — not
        # the hex string we set — so compare via "is the value darker"
        # (sum of channels lower).
        dark_facecolor = gt._figure.patch.get_facecolor()
        assert sum(dark_facecolor[:3]) < sum(light_facecolor[:3]), (
            "ConductivityImage matplotlib figure background should darken"
        )

        # Toggle back to light: every widget reverts.
        set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
        _get_app().processEvents()
        assert live_plot._plot_bg == light_palette["bg"]
        assert voltage_plot._plot_bg == light_palette["bg"]
        assert recon_widget._plot_bg == light_palette["panel_bg"]
        # Facecolor returns to roughly the original light value.
        again = gt._figure.patch.get_facecolor()
        assert sum(again[:3]) > sum(dark_facecolor[:3])
    finally:
        set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
        for widget in (live_plot, voltage_plot, recon_widget, gt):
            widget.close()
            widget.deleteLater()
        app.processEvents()


@pytest.mark.gui
def test_dark_mode_toggle_swaps_stylesheet_and_tone_palette() -> None:
    """The View → Dark Theme action must:
    1. switch current_theme_mode() to 'dark'
    2. append the dark overlay QSS to the application stylesheet
    3. flip tone_palette('idle') to the dark-variant triplet
    4. survive another toggle back to light without leaking the
       overlay into the light stylesheet
    """
    from eit_app.ui.theme import (
        _build_stylesheet,
        current_theme_mode,
        set_theme_mode,
        tone_palette,
    )

    app = _get_app()
    # Start from known-good light state regardless of any persisted
    # preference on the dev machine.
    set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
    assert current_theme_mode() == "light"
    light_css = _build_stylesheet("light")
    light_tones = tone_palette("idle")

    set_theme_mode(app, "dark", persist=False, apply_stylesheet=False)
    assert current_theme_mode() == "dark"
    dark_css = _build_stylesheet("dark")
    dark_tones = tone_palette("idle")

    # The dark stylesheet is a strict superset of the light one (base
    # + overlay), so it's strictly longer.
    assert len(dark_css) > len(light_css)
    # Dark canvas color appears in the overlay but not the base.
    assert "#1a1f26" in dark_css
    assert "#1a1f26" not in light_css
    # Tone palette must swap to the dark triplet.
    assert dark_tones != light_tones
    assert dark_tones[0] == "#c7d0db"

    # Toggle back: stylesheet returns to the base exactly.
    set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
    assert _build_stylesheet("light") == light_css
    assert tone_palette("idle") == light_tones


@pytest.mark.gui
def test_theme_mode_subscribers_are_weakly_retained() -> None:
    from eit_app.ui.theme import _mode_listeners, set_theme_mode, subscribe_theme_mode

    app = _get_app()
    before = len(_mode_listeners)

    class ListenerProbe:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def on_theme_mode_changed(self, mode: str) -> None:
            self.calls.append(mode)

    probe = ListenerProbe()
    subscribe_theme_mode(probe.on_theme_mode_changed)
    assert len(_mode_listeners) == before + 1

    set_theme_mode(app, "dark", persist=False, apply_stylesheet=False)
    assert probe.calls[-1] == "dark"

    del probe
    gc.collect()
    set_theme_mode(app, "light", persist=False, apply_stylesheet=False)
    assert len(_mode_listeners) <= before


@pytest.mark.gui
def test_app_theme_publishes_accessibility_selectors() -> None:
    """Guard against accidental regression of the accessibility additions.

    These selectors make keyboard navigation and disabled state readable:
      - :focus on buttons / inputs / combos / tabs
      - :hover feedback on QToolBox tabs
      - :disabled with ~4.5:1 contrast
      - Latin + CJK font stack on the Qt application font
    """
    from eit_app.ui.fonts import configure_runtime_fonts
    from eit_app.ui.theme import apply_app_theme, _resolve_ui_font_families

    app = _get_app()
    configure_runtime_fonts(app)
    apply_app_theme(app)
    css = app.styleSheet()

    required_selectors = (
        "QPushButton:focus",
        'QPushButton[buttonRole="primary"]:focus',
        'QPushButton[buttonRole="danger"]:focus',
        "QLineEdit:focus,",  # shared selector — also matches spinbox/textedit
        "QLineEdit:disabled,",
        "QComboBox:focus",
        "QComboBox:disabled",
        "QTabBar::tab:focus",
        "QToolBox::tab:hover:!selected",
        "QCheckBox:focus",
    )
    missing = [sel for sel in required_selectors if sel not in css]
    assert not missing, f"Theme is missing accessibility selectors: {missing}"

    families = _resolve_ui_font_families()
    assert families[:3] == ["Segoe UI", "Noto Sans", "DejaVu Sans"], (
        "Latin font base should stay stable for Windows / Linux compat"
    )


@pytest.mark.gui
def test_reconstruction_widget_pre_renders_static_layout_and_refreshes_internal_image() -> (
    None
):
    _get_app()
    widget = ReconstructionWidget()
    widget.configure_layout(n_elec=8, radius=1.0)
    widget.show()
    _get_app().processEvents()

    coords = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    result = ReconstructionResult(
        conductivity=np.array([0.2, -0.3], dtype=float),
        node_coords=coords,
        cell_connectivity=cells,
        metadata={"n_elec": 8},
    )

    widget.update_reconstruction(result)
    _get_app().processEvents()

    assert widget._image_item.image is not None
    assert len(widget._electrode_label_items) == 8
    arc_x, arc_y = widget._electrode_arc_item.getData()
    assert arc_x is not None and arc_y is not None
    assert np.isnan(np.asarray(arc_x)).any()
    first_pos = widget._electrode_label_items[0].pos()
    second_pos = widget._electrode_label_items[1].pos()
    assert abs(first_pos.x()) < 0.2
    assert first_pos.y() > 0.9
    assert second_pos.x() < 0.0
    assert second_pos.y() > 0.5

    widget.update_reconstruction(
        ReconstructionResult(
            conductivity=np.array([-0.1, 0.4], dtype=float),
            node_coords=coords,
            cell_connectivity=cells,
            metadata={"n_elec": 8},
        )
    )
    _get_app().processEvents()

    assert widget._image_item.image is not None
    widget.clear()
    assert widget._empty_overlay.isHidden() is False
    widget.close()


@pytest.mark.gui
def test_conductivity_image_widget_recovers_from_orphaned_colorbar() -> None:
    _get_app()
    widget = ConductivityImageWidget()
    widget.show()
    _get_app().processEvents()

    coords = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    widget.update_image(np.array([1.0, 1.2], dtype=float), coords, cells)
    _get_app().processEvents()
    old_colorbar = widget._colorbar
    assert old_colorbar is not None
    old_colorbar_ax = old_colorbar.ax

    class _OrphanedColorbar:
        ax = old_colorbar_ax

        def remove(self) -> None:
            raise AttributeError("'NoneType' object has no attribute 'set_subplotspec'")

    widget._colorbar = _OrphanedColorbar()
    widget.update_image(np.array([0.9, 1.4], dtype=float), coords, cells)
    _get_app().processEvents()

    assert widget._colorbar is not None
    assert widget._colorbar is not old_colorbar
    assert old_colorbar_ax not in widget._figure.axes

    widget.clear()
    widget.close()


@pytest.mark.gui
def test_conductivity_image_widget_keeps_fixed_geometry_across_colorbar_ranges() -> (
    None
):
    app = _get_app()
    truth = ConductivityImageWidget("Truth")
    reconstruction = ConductivityImageWidget("Reconstruction")
    try:
        for widget in (truth, reconstruction):
            widget.resize(520, 520)
            widget.show()
        app.processEvents()

        coords = np.array(
            [
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=float,
        )
        cells = np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=int,
        )

        truth.update_image(np.array([1.0, 1.0, 2.0, 2.0], dtype=float), coords, cells)
        reconstruction.update_image(
            np.array([0.982, 1.013, 1.089, 1.177], dtype=float),
            coords,
            cells,
        )
        app.processEvents()

        assert truth._ax.get_position().bounds == pytest.approx(
            reconstruction._ax.get_position().bounds,
            abs=1.0e-9,
        )
        assert truth._colorbar.ax.get_position().bounds == pytest.approx(
            reconstruction._colorbar.ax.get_position().bounds,
            abs=1.0e-9,
        )
        assert truth._ax.get_window_extent().bounds[2:] == pytest.approx(
            reconstruction._ax.get_window_extent().bounds[2:],
            abs=1.0e-6,
        )
    finally:
        truth.clear()
        reconstruction.clear()
        truth.close()
        reconstruction.close()


@pytest.mark.gui
def test_conductivity_image_widget_smooths_cell_values_to_node_gouraud() -> None:
    _get_app()
    widget = ConductivityImageWidget()
    widget.show()
    _get_app().processEvents()

    coords = np.array(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    widget.update_image(np.array([1.0, 2.0], dtype=float), coords, cells)
    _get_app().processEvents()

    assert widget._last_render_mode == "cell_to_node_gouraud"
    assert widget._colorbar is not None

    widget.clear()
    widget.close()


@pytest.mark.gui
def test_conductivity_image_widget_renders_3d_tetra_projection() -> None:
    _get_app()
    widget = ConductivityImageWidget()
    widget.show()
    _get_app().processEvents()

    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

    widget.update_image(np.array([1.25], dtype=float), coords, cells)
    _get_app().processEvents()

    assert widget._colorbar is not None
    assert widget._last_caption is None

    widget.clear()
    widget.close()


@pytest.mark.gui
def test_connection_panel_auto_selects_unique_windows_serial_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIT_APP_SCAN_SERIAL_ON_STARTUP", "1")
    monkeypatch.setattr(
        "eit_app.ui.hardware.connection_panel.discover_serial_ports",
        lambda: [
            SerialPortDescriptor(
                device="COM4",
                display_name="COM4 - USB-SERIAL CH340",
                source="windows-com",
            )
        ],
    )

    window = EITWorkstation()
    _show_window(window)

    assert window._conn_panel.serial_port_count() == 1
    assert window._conn_panel.selected_serial_port() == "COM4"
    assert (
        window._conn_panel.selected_serial_display_name() == "COM4 - USB-SERIAL CH340"
    )
    assert "Auto-selected the only port" in window._conn_panel._port_hint.text()
    assert "Windows COM bridge" in window._conn_panel._port_hint.text()

    window._conn_panel._port_combo.setCurrentText(
        "COM4 -> /dev/ttyS3 - USB-SERIAL CH340"
    )
    assert window._conn_panel.selected_serial_port() == "COM4"

    _close_window(window)


@pytest.mark.gui
def test_connection_panel_defers_serial_scan_on_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EIT_APP_SCAN_SERIAL_ON_STARTUP", raising=False)
    calls = {"count": 0}

    def _fake_discover() -> list[SerialPortDescriptor]:
        calls["count"] += 1
        return [
            SerialPortDescriptor(
                device="COM7",
                display_name="COM7 - USB Serial",
                source="windows-com",
            )
        ]

    monkeypatch.setattr(
        "eit_app.ui.hardware.connection_panel.discover_serial_ports",
        _fake_discover,
    )

    window = EITWorkstation()
    _show_window(window)

    assert calls["count"] == 0
    assert window._conn_panel.serial_port_count() == 0

    window._conn_panel.refresh_serial_ports()
    assert calls["count"] == 1
    assert window._conn_panel.selected_serial_port() == "COM7"

    _close_window(window)


@pytest.mark.gui
def test_serial_connect_fails_fast_when_no_port_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "eit_app.ui.hardware.connection_panel.discover_serial_ports",
        lambda: [],
    )

    window = EITWorkstation()
    _show_window(window)

    _click(window._conn_panel._connect_btn)
    _get_app().processEvents()

    assert window._state.connection_status is ConnectionStatus.ERROR
    assert window._workflow_toolbox.currentIndex() == 0
    assert "No serial ports detected" in window._status_bar.currentMessage()
    assert "No serial ports detected" in window._conn_panel._port_hint.text()

    _close_window(window)


@pytest.mark.gui
def test_relay_preflight_failure_is_reported_before_device_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connect_calls: list[bool] = []

    def _fake_preflight(transport_type: str, config: dict) -> ConnectionPreflightResult:
        assert transport_type == "relay"
        assert config["server_host"] == "relay.example"
        return ConnectionPreflightResult(
            False,
            "无法连接到 4G Relay 服务器 relay.example:4555。",
            "relay.example:4555 当前不可达，请先确认服务已启动。",
        )

    monkeypatch.setattr(
        main_window_module, "preflight_connection_target", _fake_preflight
    )

    window = EITWorkstation()
    _show_window(window)
    monkeypatch.setattr(
        window._device_ctrl, "connect_device", lambda: connect_calls.append(True)
    )

    window._conn_panel._transport_combo.setCurrentIndex(1)
    window._conn_panel._server_host.setText("relay.example")
    window._conn_panel._server_port.setValue(4555)
    _click(window._conn_panel._connect_btn)
    _get_app().processEvents()

    assert connect_calls == []
    assert window._state.connection_status is ConnectionStatus.ERROR
    assert (
        "无法连接到 4G Relay 服务器 relay.example:4555"
        in window._status_bar.currentMessage()
    )
    assert "当前不可达" in window._conn_panel._transport_hint.text()

    _close_window(window)


@pytest.mark.gui
def test_dataset_generation_uses_dedicated_tab_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)
    window._tab_widget.setCurrentIndex(2)
    _get_app().processEvents()

    dataset_dir = tmp_path / "generated-dataset"
    window._dataset_tab.mesh_setup_panel._dim_combo.setCurrentIndex(1)
    window._dataset_tab.mesh_setup_panel._n_elec_spin.setValue(24)
    window._dataset_tab.dataset_generator_panel._n_samples_spin.setValue(12)
    window._dataset_tab.dataset_generator_panel._dir_edit.setText(str(dataset_dir))
    window._dataset_tab.dataset_generator_panel._ellipse_check.setChecked(True)
    window._dataset_tab.dataset_generator_panel._rect_check.setChecked(True)

    captured = {}

    def _fake_generate(request) -> None:
        captured["config"] = request.config

    monkeypatch.setattr(window._dataset_ctrl, "generate", _fake_generate)
    _click(window._dataset_tab.dataset_generator_panel._gen_btn)

    assert "config" in captured
    assert captured["config"].output_dir == str(dataset_dir)
    assert captured["config"].n_samples == 12
    assert captured["config"].mesh_dimension == 3
    assert captured["config"].n_electrodes == 24
    assert set(captured["config"].shapes) == {"circle", "ellipse", "rectangle"}
    assert window._dataset_tab.summary_panel._values["output_dir"].text() == str(
        dataset_dir
    )
    assert window._dataset_tab.summary_panel._values["samples"].text() == "12"
    assert window._dataset_tab.summary_panel._state_chip.text() == "Generating"

    _close_window(window)


@pytest.mark.gui
def test_simulator_continuous_acquisition_and_recording_smoke(tmp_path: Path) -> None:
    app = _get_app()
    window = EITWorkstation()
    _show_window(window)
    output_dir = tmp_path / "recordings"

    window._on_connect_requested("simulator", {"simulator_fps": 0})
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    window._on_recording_toggled(True, str(output_dir))
    window._on_start_acquisition()

    assert _wait_until(lambda: window._state.frame_count > 0, timeout=8.0)
    assert _wait_until(lambda: window._rec_ctrl.frames_recorded > 0, timeout=8.0)

    window._on_stop_acquisition()
    app.processEvents()

    csv_files = sorted(output_dir.rglob("*.csv"))
    yaml_files = sorted(output_dir.rglob("*.yaml"))
    session_dirs = sorted(output_dir.glob("session_*"))

    assert csv_files
    assert yaml_files
    assert session_dirs
    first_frame_meta = read_frame_yaml(yaml_files[0])
    session_meta = read_session_metadata(session_dirs[0] / "session_metadata.yaml")
    assert first_frame_meta["board_id"] == 1
    assert first_frame_meta["user_id"] == 1
    assert session_meta["board_id"] == 1
    assert session_meta["user_id"] == 1
    assert window._frame_browser._model.rowCount() >= 1
    assert window._acq_process is None
    assert window._ring_buffer is None
    _close_window(window)


@pytest.mark.gui
def test_simulator_scheduled_acquisition_smoke() -> None:
    app = _get_app()
    window = EITWorkstation()
    _show_window(window)

    window._on_connect_requested("simulator", {"simulator_fps": 0})
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    window._acq_panel.set_acquisition_plan(
        {
            "timed_enabled": True,
            "interval_sec": 0.2,
            "acquisition_count": 3,
            "frequency_stepping": True,
            "start_hz": 1000,
            "end_hz": 1200,
        }
    )
    window._on_acquisition_plan_changed(window._acq_panel.acquisition_plan())
    window._on_start_acquisition()

    assert window._plan_active is True
    assert window._workflow_toolbox.currentIndex() == 2
    assert window._status_bar._acq_label.text() == "Acq: Stepped Run"
    assert window._status_bar._power_label.text() == "Power: ON"
    assert window._summary_panel._state_badge.text() == "ACQUIRING"
    assert window._summary_panel._indicator_values["acq"].text() == "STEP"
    assert window._summary_panel._values["transport"].text() == "Simulator"
    assert (
        "Stepped Run | 0/3 | every 0.2s | 1000→1200 Hz"
        == window._summary_panel._values["plan"].text()
    )
    assert _wait_until(lambda: window._state.frame_count == 3, timeout=8.0)
    assert _wait_until(lambda: window._plan_active is False, timeout=8.0)
    assert window._control_panel._freq_spin.value() == 1200
    assert window._status_bar._acq_label.text() == "Acq: Idle"
    assert (
        window._summary_panel._values["plan"].text()
        == "Idle | Stepped Run 3x | every 0.2s | 1000→1200 Hz"
    )

    window._on_stop_acquisition()
    app.processEvents()

    assert window._acq_process is None
    assert window._state.frame_count == 3

    _close_window(window)


@pytest.mark.gui
def test_fixed_frequency_timed_run_uses_step2_drive_frequency_and_keeps_live_outputs() -> (
    None
):
    _get_app()
    window = EITWorkstation()
    _show_window(window)

    window._on_connect_requested("simulator", {"simulator_fps": 0})
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    window._on_frequency_changed(2500)
    window._acq_panel.set_acquisition_plan(
        {
            "timed_enabled": True,
            "interval_sec": 0.2,
            "acquisition_count": 3,
            "frequency_stepping": False,
            "start_hz": 1000,
            "end_hz": 1200,
        }
    )
    window._on_acquisition_plan_changed(window._acq_panel.acquisition_plan())

    assert window._build_planned_frequencies() == [2500, 2500, 2500]
    window._recon_prewarm_ready_signature = (
        window._build_realtime_recon_prewarm_payload()[1]
    )

    def _fake_reconstruct(request) -> bool:
        measured = np.asarray(
            request.target_frame.real - request.reference_frame.real,
            dtype=float,
        )
        result = ReconstructionResult(
            conductivity=np.array([0.1], dtype=float),
            node_coords=np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=float,
            ),
            cell_connectivity=np.array([[0, 1, 2]], dtype=int),
            measured=measured,
            simulated=measured * 0.5,
            metadata={
                **dict(request.metadata),
                "n_elec": 16,
                "electrode_coverage": 0.5,
            },
        )
        window._recon_ctrl.reconstruction_done.emit(result)
        return True

    window._recon_ctrl.reconstruct = _fake_reconstruct

    window._on_start_acquisition()

    assert window._auto_reconstruct is True
    assert window._status_bar._acq_label.text() == "Acq: Finite Run"
    assert (
        window._summary_panel._values["plan"].text()
        == "Finite Run | 0/3 | every 0.2s | 2500 Hz"
    )
    assert _wait_until(lambda: window._voltage_plot._has_data is True, timeout=8.0)
    assert _wait_until(lambda: window._state.frame_count == 3, timeout=8.0)
    assert window._summary_panel._values["drive"].text().startswith("2500 Hz")
    assert (
        window._summary_panel._values["plan"].text()
        == "Idle | Finite Run 3x | every 0.2s | 2500 Hz"
    )

    _close_window(window)


@pytest.mark.gui
def test_frequency_stepped_run_keeps_auto_reconstruction_enabled_and_updates_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_app()
    window = EITWorkstation()
    _show_window(window)

    submitted: list[object] = []

    def _fake_reconstruct(request) -> bool:
        submitted.append(request)
        measured = np.asarray(
            request.target_frame.real - request.reference_frame.real,
            dtype=float,
        )
        result = ReconstructionResult(
            conductivity=np.array([0.2], dtype=float),
            node_coords=np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=float,
            ),
            cell_connectivity=np.array([[0, 1, 2]], dtype=int),
            measured=measured,
            simulated=measured * 0.5,
            metadata={
                **dict(request.metadata),
                "n_elec": 16,
                "electrode_coverage": 0.5,
            },
        )
        window._recon_ctrl.reconstruction_done.emit(result)
        return True

    monkeypatch.setattr(window._recon_ctrl, "reconstruct", _fake_reconstruct)
    monkeypatch.setattr(window._acq_ctrl, "stop", lambda deactivate_device=False: None)
    monkeypatch.setattr(window, "_reset_acquisition_pipeline", lambda: None)
    monkeypatch.setattr(window, "_run_next_planned_acquisition", lambda: None)
    monkeypatch.setattr(window, "_finish_planned_acquisition_run", lambda: None)
    window._hw_recon_ctrl._busy = True
    window._db_recon_ctrl._busy = True
    window._sim_recon_ctrl._busy = True
    window._recon_prewarm_ready_signature = (
        window._build_realtime_recon_prewarm_payload()[1]
    )

    window._auto_reconstruct = True
    window._plan_active = True
    window._planned_step_pending = True
    window._scheduled_enabled = False
    window._plan_completed_count = 0
    window._plan_frequencies = [1000, 3000]

    first_frame = FrameData(
        real=np.array([10.0, 20.0, 30.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=1.0,
        frame_index=0,
    )
    window._acq_ctrl._total_frames = 1
    window._acq_ctrl.new_frame.emit(first_frame)
    _get_app().processEvents()

    assert window._reference_frame is first_frame
    assert submitted == []
    assert window._voltage_plot._has_data is False

    window._planned_step_pending = True
    window._plan_completed_count = 1
    second_frame = FrameData(
        real=np.array([14.0, 27.0, 41.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=2.0,
        frame_index=1,
    )
    window._acq_ctrl._total_frames = 2
    window._acq_ctrl.new_frame.emit(second_frame)
    _get_app().processEvents()

    assert len(submitted) == 1
    assert submitted[0].reference_frame is first_frame
    assert submitted[0].target_frame is second_frame
    assert submitted[0].metadata["drive_mode"] == "total_current"
    assert submitted[0].metadata["stim_amp_uA"] == 100
    assert submitted[0].metadata["drive_value"] == pytest.approx(100e-6)
    assert window._voltage_plot._has_data is True
    assert window._recon_widget._empty_overlay.isVisible() is False
    assert window._voltage_plot._curve_reconstructed.isVisible() is True

    measured_x, measured_y = window._voltage_plot._curve_primary.getData()
    recon_x, recon_y = window._voltage_plot._curve_reconstructed.getData()
    live_x, live_y = window._live_plot._curve_real.getData()

    expected_diff = np.array([4.0, 7.0, 11.0], dtype=float)
    assert measured_x is not None
    assert measured_y is not None
    assert recon_x is not None
    assert recon_y is not None
    assert live_x is not None
    assert live_y is not None
    assert np.allclose(measured_y, expected_diff)
    assert np.allclose(recon_y, expected_diff * 0.5)
    assert np.allclose(live_y, second_frame.real)
    assert not np.allclose(measured_y, live_y)

    _close_window(window)


@pytest.mark.gui
def test_realtime_auto_reconstruction_waits_for_prewarm_and_uses_latest_pending_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _get_app()
    window = EITWorkstation()
    _show_window(window)

    prewarm_requests: list[object] = []
    live_requests: list[object] = []

    monkeypatch.setattr(window, "_rebuild_acquisition_pipeline", lambda: None)
    monkeypatch.setattr(window._acq_ctrl, "start", lambda: None)

    def _fake_prewarm(request) -> bool:
        prewarm_requests.append(request)
        return True

    def _fake_live_reconstruct(request) -> bool:
        live_requests.append(request)
        return True

    monkeypatch.setattr(window._recon_prewarm_ctrl, "reconstruct", _fake_prewarm)
    monkeypatch.setattr(window._recon_ctrl, "reconstruct", _fake_live_reconstruct)

    window._transport_type = "simulator"
    window._state.set_connection_status(ConnectionStatus.CONNECTED)

    window._on_start_acquisition()
    _get_app().processEvents()

    assert len(prewarm_requests) == 1
    assert prewarm_requests[0].metadata["warmup_only"] is True
    assert prewarm_requests[0].metadata["request_source"] == "hardware_auto_prewarm"

    first_frame = FrameData(
        real=np.array([10.0, 20.0, 30.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=1.0,
        frame_index=0,
    )
    second_frame = FrameData(
        real=np.array([11.0, 21.0, 31.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=2.0,
        frame_index=1,
    )
    third_frame = FrameData(
        real=np.array([14.0, 24.0, 34.0], dtype=float),
        imag=np.zeros(3, dtype=float),
        timestamp=3.0,
        frame_index=2,
    )

    window._acq_ctrl._total_frames = 1
    window._acq_ctrl.new_frame.emit(first_frame)
    _get_app().processEvents()
    window._acq_ctrl._total_frames = 2
    window._acq_ctrl.new_frame.emit(second_frame)
    _get_app().processEvents()
    window._acq_ctrl._total_frames = 3
    window._acq_ctrl.new_frame.emit(third_frame)
    _get_app().processEvents()

    assert live_requests == []
    assert window._pending_auto_target_frame is third_frame

    warm_result = ReconstructionResult(
        conductivity=np.array([], dtype=float),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float),
        cell_connectivity=np.array([[0, 1, 2]], dtype=int),
        metadata=dict(prewarm_requests[0].metadata),
    )
    window._recon_prewarm_ctrl.reconstruction_done.emit(warm_result)
    _get_app().processEvents()

    assert len(live_requests) == 1
    assert live_requests[0].reference_frame is first_frame
    assert live_requests[0].target_frame is third_frame

    _close_window(window)


@pytest.mark.gui
def test_simulator_single_frame_capture_stops_automatically(tmp_path: Path) -> None:
    window = EITWorkstation()
    _show_window(window)

    window._on_connect_requested("simulator", {"simulator_fps": 0})
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    output_dir = tmp_path / "single-frame"
    window._acq_panel.set_output_dir(str(output_dir))
    _click(window._acq_panel._rec_check)
    assert window._status_bar._rec_label.text() == "Record: Armed"

    _click(window._acq_panel._single_frame_btn)

    assert _wait_until(lambda: window._state.frame_count == 1, timeout=8.0)
    assert _wait_until(lambda: window._acq_process is None, timeout=8.0)
    assert window._status_bar._acq_label.text() == "Acq: Idle"
    assert window._status_bar._rec_label.text() == "Record: Armed"
    assert "Single-frame acquisition complete" in window._status_bar.currentMessage()
    csv_files = sorted(output_dir.rglob("*.csv"))
    yaml_files = sorted(output_dir.rglob("*.yaml"))
    assert len(csv_files) == 1
    assert len(yaml_files) >= 2  # session metadata + frame metadata

    _close_window(window)


@pytest.mark.gui
def test_auto_close_combo_box_hides_popup_on_disable_clear_and_focus_loss() -> None:
    """Edge cases around the dropdown auto-hide:
    1. Calling setEnabled(False) while the popup is visible must hide it.
    2. Replacing the item list via clear() + addItems() must close a
       stale open popup so it doesn't show ghost entries.
    3. App focus moving to an unrelated widget must close the popup
       (keyboard Tab-away path, not just click-outside).
    """
    from PySide6.QtWidgets import QPushButton, QWidget, QVBoxLayout

    app = _get_app()
    host = QWidget()
    layout = QVBoxLayout(host)
    combo = AutoCloseComboBox()
    combo.addItems(["A", "B"])
    other = QPushButton("elsewhere")
    layout.addWidget(combo)
    layout.addWidget(other)
    host.show()
    app.processEvents()

    # Case 1: disable while open
    combo.showPopup()
    app.processEvents()
    assert combo._menu.isVisible()
    combo.setEnabled(False)
    app.processEvents()
    assert not combo._menu.isVisible(), "setEnabled(False) should hide popup"
    combo.setEnabled(True)
    app.processEvents()

    # Case 2: clear() while open
    combo.showPopup()
    app.processEvents()
    assert combo._menu.isVisible()
    combo.clear()
    app.processEvents()
    assert not combo._menu.isVisible(), "clear() should hide stale popup"
    combo.addItems(["X", "Y"])

    # Case 3: focus moves to an unrelated widget while popup is open
    combo.showPopup()
    app.processEvents()
    assert combo._menu.isVisible()
    # Simulate the Tab-away path by emitting focusChanged manually —
    # QMenu's internal Qt.Popup flag would intercept a real click, but
    # keyboard focus shifts may not always trigger the native dismiss.
    app.focusChanged.emit(combo._line_edit, other)
    app.processEvents()
    assert not combo._menu.isVisible(), (
        "focus moving to an unrelated widget should close the popup"
    )

    host.close()
    host.deleteLater()
    app.processEvents()


@pytest.mark.gui
def test_auto_close_combo_box_hides_popup_after_selection() -> None:
    app = _get_app()
    combo = AutoCloseComboBox()
    combo.addItems(["A", "B", "C"])
    combo.show()
    app.processEvents()

    combo.showPopup()
    app.processEvents()
    popup = combo._menu
    popup.actions()[1].trigger()

    assert _wait_until(lambda: combo.currentIndex() == 1, timeout=1.0)
    assert _wait_until(lambda: not popup.isVisible(), timeout=1.0)

    combo.showPopup()
    app.processEvents()
    assert len(combo._menu.actions()) == 3
    assert combo.itemText(1) == "B"
    combo.hidePopup()
    combo.close()


@pytest.mark.gui
def test_button_clicks_update_status_bar_and_device_profile() -> None:
    window = EITWorkstation()
    _show_window(window)

    _connect_simulator(window)

    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )
    assert window._status_bar._conn_label.text() == "Link: Verified"
    assert window._status_bar._power_label.text() == "Power: Unknown"
    assert window._status_bar._acq_label.text() == "Acq: Idle"
    assert window._workflow_toolbox.currentIndex() == 1
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["link"].text() == "OK"
    assert window._summary_panel._indicator_values["power"].text() == "UNK"
    assert window._summary_panel._values["transport"].text() == "Simulator"
    assert window._conn_panel._connect_btn.isEnabled() is False
    assert window._conn_panel._disconnect_btn.isEnabled() is True
    assert window._control_panel._freq_spin.isEnabled() is True
    assert window._control_panel._power_on_btn.isChecked() is False
    assert window._control_panel._power_off_btn.isChecked() is False

    window._control_panel._freq_spin.setValue(2500)
    _click(window._control_panel._freq_apply)
    assert _wait_until(
        lambda: (
            window._device_config["frequency_hz"] == 2500
            and "set_frequency" in window._status_bar.currentMessage()
        ),
        timeout=3.0,
    )
    assert "2500 Hz" in window._summary_panel._values["drive"].text()

    window._control_panel._stim_combo.setCurrentIndex(3)
    _click(window._control_panel._stim_apply)
    assert _wait_until(
        lambda: (
            window._device_config["stim_amp_level"] == 3
            and window._device_config["stim_amp_uA"] == 500
            and "set_stim_amplitude" in window._status_bar.currentMessage()
        ),
        timeout=3.0,
    )
    assert "500 uA" in window._summary_panel._values["drive"].text()

    window._control_panel._vamp_combo.setCurrentIndex(2)
    _click(window._control_panel._vamp_apply)
    assert _wait_until(
        lambda: (
            window._device_config["voltage_amp_level_1"] == 2
            and window._device_config["voltage_amp_level_2"] == 2
            and "set_voltage_amp_levels" in window._status_bar.currentMessage()
        ),
        timeout=3.0,
    )
    assert "0.32x" in window._summary_panel._values["drive"].text()

    _click(window._control_panel._imp_btn)
    assert _wait_until(
        lambda: window._status_bar.currentMessage().startswith("Contact impedance:"),
        timeout=3.0,
    )

    _click(window._control_panel._power_on_btn)
    assert _wait_until(
        lambda: (
            window._status_bar._power_label.text() == "Power: ON"
            and "Measurement power switched to ON"
            in window._status_bar.currentMessage()
        ),
        timeout=3.0,
    )
    assert window._workflow_toolbox.currentIndex() == 1
    assert window._summary_panel._indicator_values["power"].text() == "ON"
    assert window._control_panel._power_on_btn.isChecked() is True
    assert window._control_panel._power_off_btn.isChecked() is False

    _click(window._control_panel._spt_btn)
    assert window._workflow_toolbox.currentIndex() == 1
    assert _wait_until(
        lambda: "Single-point returned:" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["power"].text() == "ON"
    assert window._control_panel._power_on_btn.isChecked() is True
    assert window._control_panel._power_off_btn.isChecked() is False
    assert window._workflow_toolbox.currentIndex() == 1

    _click(window._control_panel._power_off_btn)
    assert _wait_until(
        lambda: (
            window._status_bar._power_label.text() == "Power: OFF"
            and "Measurement power switched to OFF"
            in window._status_bar.currentMessage()
        ),
        timeout=3.0,
    )
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["power"].text() == "OFF"
    assert window._control_panel._power_on_btn.isChecked() is False
    assert window._control_panel._power_off_btn.isChecked() is True

    _click(window._conn_panel._disconnect_btn)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.DISCONNECTED,
        timeout=3.0,
    )
    assert window._status_bar._conn_label.text() == "Link: Down"
    assert window._status_bar._power_label.text() == "Power: Unknown"
    assert window._summary_panel._indicator_values["link"].text() == "DOWN"
    assert window._control_panel._freq_spin.isEnabled() is False
    assert window._control_panel._power_on_btn.isChecked() is False
    assert window._control_panel._power_off_btn.isChecked() is False

    _close_window(window)


@pytest.mark.gui
def test_single_point_does_not_force_power_state_on_without_power_command() -> None:
    window = EITWorkstation()
    _show_window(window)

    _connect_simulator(window)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    assert window._status_bar._power_label.text() == "Power: Unknown"
    _click(window._control_panel._spt_btn)
    assert _wait_until(
        lambda: "Single-point returned:" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert window._status_bar._power_label.text() == "Power: Unknown"
    assert window._summary_panel._indicator_values["power"].text() == "UNK"
    assert window._workflow_toolbox.currentIndex() == 1

    _close_window(window)


@pytest.mark.gui
def test_gui_interaction_regression_for_fps_recording_and_frame_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    _connect_simulator(window)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    output_dir = tmp_path / "gui-recordings"
    window._acq_panel._dir_edit.setText(str(output_dir))
    _click(window._acq_panel._rec_check)
    assert _wait_until(
        lambda: window._acq_panel._rec_check.isChecked() is True, timeout=2.0
    )
    assert window._acq_panel._rec_check.isEnabled() is True
    assert (
        "Recording enabled; captures will be saved to"
        in window._status_bar.currentMessage()
    )
    assert window._status_bar._rec_label.text() == "Record: Armed"
    assert window._workflow_toolbox.currentIndex() == 1
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["record"].text() == "ARM"
    assert "Armed" in window._summary_panel._values["record"].text()
    assert str(output_dir) in window._summary_panel._values["record"].text()
    window._recon_prewarm_ready_signature = (
        window._build_realtime_recon_prewarm_payload()[1]
    )

    def _fake_reconstruct(request) -> bool:
        measured = np.asarray(
            request.target_frame.real - request.reference_frame.real,
            dtype=float,
        )
        result = ReconstructionResult(
            conductivity=np.array([0.1], dtype=float),
            node_coords=np.array(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=float,
            ),
            cell_connectivity=np.array([[0, 1, 2]], dtype=int),
            measured=measured,
            simulated=measured * 0.5,
            metadata={
                **dict(request.metadata),
                "n_elec": 16,
                "electrode_coverage": 0.5,
            },
        )
        window._recon_ctrl.reconstruction_done.emit(result)
        return True

    monkeypatch.setattr(window._recon_ctrl, "reconstruct", _fake_reconstruct)

    _click(window._acq_panel._start_btn)
    assert _wait_until(
        lambda: window._status_bar._rec_label.text() == "Record: Writing", timeout=2.0
    )
    assert window._workflow_toolbox.currentIndex() == 2
    assert window._status_bar._acq_label.text() == "Acq: Continuous"
    assert window._status_bar._power_label.text() == "Power: ON"
    assert window._summary_panel._state_badge.text() == "ACQUIRING + RECORDING"
    assert window._summary_panel._indicator_values["record"].text() == "REC"
    assert window._summary_panel._indicator_values["acq"].text() == "RUN"
    assert window._summary_panel._values["plan"].text() == "Continuous"
    assert _wait_until(lambda: window._state.frame_count > 0, timeout=8.0)
    assert _wait_until(lambda: _fps_value(window) > 0.0, timeout=8.0)
    assert _wait_until(
        lambda: window._frame_browser._model.rowCount() >= 2, timeout=8.0
    )

    assert window._status_bar._frame_label.text().startswith("Frames: ")
    assert window._acq_panel._frame_label.text() == str(window._state.frame_count)
    x_data, y_data = window._live_plot._curve_real.getData()
    assert x_data is not None
    assert y_data is not None
    assert len(y_data) == 208

    first_entry = window._frame_browser._model.get_entry(0)
    second_entry = window._frame_browser._model.get_entry(1)
    assert first_entry is not None
    assert second_entry is not None

    window._frame_browser._table.selectRow(0)
    _click(window._frame_browser._ref_btn)
    assert _wait_until(
        lambda: (
            window._selected_reference_entry is not None
            and window._selected_reference_entry.get("file_path")
            == first_entry["file_path"]
            and (
                "Reference frame selected" in window._status_bar.currentMessage()
                or "Reference frame updated" in window._status_bar.currentMessage()
            )
        ),
        timeout=2.0,
    )

    window._frame_browser.target_selected.emit(second_entry)
    assert _wait_until(
        lambda: (
            window._selected_target_entry is not None
            and window._selected_target_entry.get("file_path")
            == second_entry["file_path"]
            and "Target frame selected" in window._status_bar.currentMessage()
        ),
        timeout=2.0,
    )

    _click(window._acq_panel._stop_btn)
    assert _wait_until(lambda: window._acq_process is None, timeout=5.0)
    assert window._status_bar._fps_label.text() == "FPS: 0.0"
    assert window._status_bar._rec_label.text() == "Record: Armed"
    assert window._status_bar._acq_label.text() == "Acq: Idle"
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["record"].text() == "ARM"
    assert window._summary_panel._indicator_values["acq"].text() == "IDLE"
    assert "Idle | manual" == window._summary_panel._values["plan"].text()
    assert window._acq_panel._rec_check.isChecked() is True
    assert window._acq_panel._start_btn.isEnabled() is True
    assert window._acq_panel._stop_btn.isEnabled() is False

    _click(window._frame_browser._clear_btn)
    assert _wait_until(
        lambda: window._frame_browser._model.rowCount() == 0, timeout=2.0
    )
    assert window._selected_reference_entry is None
    assert window._selected_target_entry is None
    assert "Recorded frame list cleared" in window._status_bar.currentMessage()

    _close_window(window)


@pytest.mark.gui
def test_record_checkbox_reverts_when_recording_start_fails(tmp_path: Path) -> None:
    window = EITWorkstation()
    _show_window(window)

    def _fail_start_recording(*args, **kwargs) -> bool:
        return False

    _connect_simulator(window)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    window._acq_panel._dir_edit.setText(str(tmp_path / "failing-recordings"))
    window._rec_ctrl.start_recording = _fail_start_recording  # type: ignore[method-assign]

    _click(window._acq_panel._rec_check)
    _click(window._acq_panel._start_btn)

    assert _wait_until(
        lambda: window._acq_panel._rec_check.isChecked() is False, timeout=2.0
    )
    assert window._state.recording_active is False
    assert window._status_bar._rec_label.text() == "Record: Off"

    _close_window(window)


@pytest.mark.gui
def test_record_checkbox_prefills_default_output_dir_when_empty() -> None:
    window = EITWorkstation()
    _show_window(window)

    window._acq_panel.set_output_dir("")
    _click(window._acq_panel._rec_check)

    assert _wait_until(
        lambda: (
            window._acq_panel.output_dir()
            == str(window._acq_panel.default_output_dir())
        ),
        timeout=2.0,
    )
    assert (
        "Recording enabled; captures will be saved to"
        in window._status_bar.currentMessage()
    )
    assert window._status_bar._rec_label.text() == "Record: Armed"

    _close_window(window)


@pytest.mark.gui
def test_recording_supports_unc_output_path_and_uses_latest_directory(
    tmp_path: Path,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    _connect_simulator(window)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    output_dir = tmp_path / "recordings-unc"
    unc_path = _as_wsl_unc(output_dir)
    window._acq_panel._dir_edit.setText(unc_path)

    _click(window._acq_panel._rec_check)
    assert _wait_until(
        lambda: window._acq_panel._dir_edit.text() == str(output_dir), timeout=2.0
    )

    _click(window._acq_panel._start_btn)
    assert _wait_until(lambda: window._rec_ctrl.frames_recorded > 0, timeout=8.0)

    _click(window._acq_panel._stop_btn)
    csv_files = sorted(output_dir.rglob("*.csv"))
    yaml_files = sorted(output_dir.rglob("*.yaml"))
    assert csv_files
    assert yaml_files

    _close_window(window)


@pytest.mark.gui
def test_close_event_powers_off_connected_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    called = {"power_off": False, "suspend": False}

    def _fake_power_off(timeout_ms: int = 3000) -> bool:
        called["power_off"] = True
        return True

    def _fake_suspend(timeout_ms: int = 1500) -> bool:
        called["suspend"] = True
        return True

    monkeypatch.setattr(window._device_ctrl, "power_off_device", _fake_power_off)
    monkeypatch.setattr(window._device_ctrl, "suspend_session", _fake_suspend)
    window._state.set_connection_status(ConnectionStatus.CONNECTED)

    _close_window(window)

    assert called["power_off"] is True
    assert called["suspend"] is True


@pytest.mark.gui
def test_live_plot_can_toggle_imag_after_frame_arrives() -> None:
    window = EITWorkstation()
    _show_window(window)

    _connect_simulator(window)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    _click(window._acq_panel._start_btn)
    assert _wait_until(lambda: window._state.frame_count > 0, timeout=8.0)

    real_x, real_y = window._live_plot._curve_real.getData()
    imag_x, imag_y = window._live_plot._curve_imag.getData()
    assert real_x is not None
    assert real_y is not None
    assert imag_x is not None
    assert imag_y is not None
    assert len(real_y) == 208
    assert len(imag_y) == 208
    assert window._live_plot._plot_widget.getPlotItem().legend is None
    assert window._live_plot._curve_imag.isVisible() is False

    _click(window._live_plot._show_imag)
    assert window._live_plot._curve_imag.isVisible() is True
    imag_x, imag_y = window._live_plot._curve_imag.getData()
    assert imag_x is not None
    assert imag_y is not None
    assert len(imag_y) == 208

    _click(window._acq_panel._stop_btn)
    _close_window(window)


@pytest.mark.gui
def test_boundary_voltage_plot_uses_hardware_labels_without_homogeneous() -> None:
    window = EITWorkstation()
    _show_window(window)

    hw_plot = window._voltage_plot
    sim_plot = window._sim_tab.results_widget.voltage_plot

    assert hw_plot.legend_labels() == ["Measured", "Recon Fit"]
    assert sim_plot.legend_labels() == ["Ground Truth", "Recon Fit"]
    assert hw_plot.current_point_count() == 208
    assert sim_plot.current_point_count() == 208

    _close_window(window)


@pytest.mark.gui
def test_simulation_voltage_index_adapts_to_mesh_electrode_count() -> None:
    window = EITWorkstation()
    _show_window(window)

    sim_plot = window._sim_tab.results_widget.voltage_plot
    assert sim_plot.current_point_count() == 208

    window._sim_tab.mesh_setup_panel._n_elec_spin.setValue(32)
    _get_app().processEvents()

    assert sim_plot.current_point_count() == estimate_measurement_point_count(
        n_electrodes=32,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        use_meas_current=False,
        use_meas_current_next=0,
    )

    _close_window(window)


@pytest.mark.gui
def test_simulation_voltage_index_uses_3d_ring_count() -> None:
    window = EITWorkstation()
    _show_window(window)

    sim_plot = window._sim_tab.results_widget.voltage_plot
    mesh_panel = window._sim_tab.mesh_setup_panel
    mesh_panel._dim_combo.setCurrentIndex(1)
    _get_app().processEvents()

    cfg = mesh_panel.get_config()
    assert cfg["n_electrodes"] == 8
    assert cfg["n_rings"] == 2
    assert sim_plot.current_point_count() == 208

    _close_window(window)


@pytest.mark.gui
def test_simulation_forward_config_2d_default_uses_coverage_not_length_override() -> (
    None
):
    window = EITWorkstation()
    _show_window(window)

    cfg = window._current_sim_forward_model_config()

    assert cfg.mesh_dimension == 2
    assert cfg.n_elec == 16
    assert cfg.n_rings == 1
    assert cfg.electrode_length_m_override is None
    assert cfg.electrode_coverage == pytest.approx(0.5)
    assert cfg.drive_mode == "line_current_density"
    assert cfg.drive_value == pytest.approx(1.0)
    assert (
        window._sim_tab.mesh_setup_panel.get_config()["electrode_length_m_override"]
        is None
    )
    assert window._sim_tab.mesh_setup_panel._electrode_length_spin.value() == (
        pytest.approx(2.0 * np.pi * 0.5 / 16.0, abs=1.0e-6)
    )

    _close_window(window)


@pytest.mark.gui
def test_simulation_forward_config_preserves_3d_multiring_layout() -> None:
    window = EITWorkstation()
    _show_window(window)

    mesh_panel = window._sim_tab.mesh_setup_panel
    mesh_panel._dim_combo.setCurrentIndex(1)
    _get_app().processEvents()

    cfg = window._current_sim_forward_model_config()

    assert cfg.mesh_dimension == 3
    assert cfg.n_elec == 8
    assert cfg.n_rings == 2
    assert cfg.total_electrodes() == 16
    assert cfg.point_count() == 208
    assert cfg.radius == pytest.approx(0.18)
    assert cfg.height == pytest.approx(0.16)
    assert cfg.drive_mode == "total_current"
    assert cfg.drive_value == pytest.approx(100.0e-6)
    assert cfg.electrode_length_m_override == pytest.approx(
        2.0 * np.pi * 0.18 * 0.5 / 8.0
    )
    assert cfg.electrode_coverage == pytest.approx(0.5)
    assert cfg.electrode_area_m2_override == pytest.approx(
        cfg.electrode_length_m_override * 0.16 * 0.2,
        rel=1.0e-5,
    )
    assert cfg.electrode_height_ratio == pytest.approx(0.2, rel=1.0e-5)
    assert tuple(cfg.electrode_level_fractions) == (0.25, 0.75)

    _close_window(window)


@pytest.mark.gui
def test_simulation_drive_current_reaches_forward_request() -> None:
    window = EITWorkstation()
    _show_window(window)

    mesh_panel = window._sim_tab.mesh_setup_panel
    mesh_panel._dim_combo.setCurrentIndex(1)
    mesh_panel._drive_value_spin.setValue(250.0)
    _get_app().processEvents()

    cfg = window._current_sim_forward_model_config()
    request = window._build_sim_forward_request(request_source="test")

    assert cfg.drive_mode == "total_current"
    assert cfg.drive_value == pytest.approx(250.0e-6)
    assert request.forward_model_config["drive_mode"] == "total_current"
    assert request.forward_model_config["drive_value"] == pytest.approx(250.0e-6)

    _close_window(window)


@pytest.mark.gui
def test_v626_simulation_drive_value_clamps_zero_before_forward_request() -> None:
    window = EITWorkstation()
    _show_window(window)

    mesh_panel = window._sim_tab.mesh_setup_panel
    mesh_panel.set_config({"mesh_dimension": 3, "drive_value": 0.0})
    _get_app().processEvents()

    cfg = window._current_sim_forward_model_config()
    request = window._build_sim_forward_request(request_source="test")

    assert cfg.drive_mode == "total_current"
    assert cfg.drive_value == pytest.approx(100.0e-6)
    assert request.forward_model_config["drive_mode"] == "total_current"
    assert request.forward_model_config["drive_value"] == pytest.approx(100.0e-6)

    mesh_panel.set_config({"mesh_dimension": 2, "drive_value": 0.0})
    _get_app().processEvents()

    cfg_2d = window._current_sim_forward_model_config()
    assert cfg_2d.drive_mode == "line_current_density"
    assert cfg_2d.drive_value == pytest.approx(1.0)

    dataset_mesh_panel = window._dataset_tab.mesh_setup_panel
    dataset_mesh_panel.set_config({"mesh_dimension": 3, "drive_value": 0.0})
    _get_app().processEvents()

    dataset_cfg = window._current_dataset_forward_model_config()
    assert dataset_cfg.drive_mode == "total_current"
    assert dataset_cfg.drive_value == pytest.approx(100.0e-6)

    _close_window(window)


@pytest.mark.gui
def test_simulation_forward_config_clamps_dense_3d_ring_area_before_mesh_build() -> (
    None
):
    from pyeidors.geometry.mesh3d_generator import Cylinder3DMeshConfig

    window = EITWorkstation()
    _show_window(window)

    mesh_panel = window._sim_tab.mesh_setup_panel
    mesh_panel.set_config(
        {
            "mesh_dimension": 3,
            "radius": 0.18,
            "height": 0.16,
            "n_electrodes": 8,
            "n_rings": 8,
            "electrode_layout": "ring_major",
            "electrode_area_m2_override": 0.003,
        }
    )
    _get_app().processEvents()

    cfg = window._current_sim_forward_model_config()

    assert cfg.mesh_dimension == 3
    assert cfg.n_rings == 8
    assert cfg.electrode_height_ratio < min(
        right - left
        for left, right in zip(
            cfg.electrode_level_fractions[:-1],
            cfg.electrode_level_fractions[1:],
        )
    )
    assert cfg.electrode_area_m2_override == pytest.approx(
        cfg.electrode_length_m_override * cfg.height * cfg.electrode_height_ratio
    )
    Cylinder3DMeshConfig(
        radius=cfg.radius,
        height=cfg.height,
        electrode_height_ratio=cfg.electrode_height_ratio,
        electrode_level_fractions=cfg.electrode_level_fractions,
    )

    _close_window(window)


@pytest.mark.gui
def test_interop_imported_3d_geometry_is_not_replaced_by_interactive_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from eit_app.models.forward_model_config import ForwardModelConfig

    window = EITWorkstation()
    _show_window(window)

    imported = ForwardModelConfig(
        mesh_dimension=3,
        mesh_refinement=0.05,
        n_elec=12,
        n_rings=2,
        radius=0.72,
        height=0.44,
    )
    loaded_bundle = SimpleNamespace(
        geometry_payload=None,
        measurements=None,
        reconstruction_preset=None,
    )
    window._ensure_interop_services()
    monkeypatch.setattr(
        window._interop_importer,
        "preview_loaded_package",
        lambda _: SimpleNamespace(forward_model_config=imported),
    )

    window._apply_interop_import("simulation", loaded_bundle)
    cfg = window._current_sim_forward_model_config()

    assert cfg.mesh_dimension == 3
    assert cfg.n_elec == 12
    assert cfg.n_rings == 2
    assert cfg.radius == pytest.approx(0.72)
    assert cfg.height == pytest.approx(0.44)

    _close_window(window)


@pytest.mark.gui
def test_simulation_inverse_request_uses_forward_mesh_size_for_single_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.mesh_setup_panel._dim_combo.setCurrentIndex(1)
    _get_app().processEvents()
    window._sim_tab.mesh_setup_panel._refine_spin.setValue(0.1)
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "debug_fine_mesh_noser",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []

    def _capture_reconstruct(request) -> bool:
        captured.append(request)
        return True

    monkeypatch.setattr(window._sim_recon_ctrl, "reconstruct", _capture_reconstruct)
    monkeypatch.setattr(
        window,
        "_submit_scheduled_ui_task",
        lambda **kwargs: kwargs["callback"]() or SimpleNamespace(accepted=True),
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "gn-difference"
    assert request.reference_frame.real.size == n_meas
    assert request.target_frame.real.size == n_meas
    assert request.mesh_refinement == pytest.approx(0.1)
    assert request.metadata["mesh_size"] == pytest.approx(0.1)
    assert request.metadata["reconstruction_runtime"] == "single_step_cached"
    assert request.metadata["simulation_inverse_route"] == "debug_fine_mesh_noser"
    assert request.metadata["simulation_inverse_debug_route"] is True
    assert request.metadata["rm_route_requires_artifact"] is False
    assert request.metadata.get("online_hot_path") != "rm_matmul"
    assert request.metadata["difference_mode"] == "normalized"
    assert request.metadata["difference_lambda"] == pytest.approx(1.0e-2)
    assert request.regularization_alpha == pytest.approx(1.0e-2)
    assert request.metadata["hyperparameter_ui_name"] == "lambda_eff"
    assert request.metadata["hyperparameter_ui_locked"] is True
    assert request.metadata["hyperparameter_ui_value"] == pytest.approx(1.0e-2)
    assert request.metadata["regularization_alpha_applied"] is False
    assert request.metadata["lambda_eff"] == pytest.approx(1.0e-2)
    assert request.metadata["hp"] == pytest.approx(0.1)
    assert request.metadata["hp_squared"] == pytest.approx(1.0e-2)
    assert request.metadata["difference_lambda_semantics"] == (
        "lambda_eff_equals_hp_squared"
    )
    assert rc._prepare_single_step_cached_runtime(request).refinement == 2

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_2d_single_step_uses_canonical_noser_lambda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "debug_fine_mesh_noser",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )
    monkeypatch.setattr(
        window,
        "_submit_scheduled_ui_task",
        lambda **kwargs: kwargs["callback"]() or SimpleNamespace(accepted=True),
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.metadata["difference_mode"] == "raw"
    assert request.metadata["difference_lambda"] == pytest.approx(1.0e-2)
    assert request.metadata["simulation_inverse_route"] == "debug_fine_mesh_noser"
    assert request.metadata.get("online_hot_path") != "rm_matmul"
    assert request.regularization_alpha == pytest.approx(1.0e-2)
    assert request.metadata["hyperparameter_ui_name"] == "lambda_eff"
    assert request.metadata["hyperparameter_ui_locked"] is True
    assert request.metadata["regularization_alpha_applied"] is False
    assert request.metadata["hp"] == pytest.approx(0.1)

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_inverse_preserves_complex_forward_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 4
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.array(
            [1.0 + 0.10j, 1.2 + 0.20j, 1.4 + 0.30j, 1.6 + 0.40j],
            dtype=np.complex64,
        ),
        homogeneous_voltages=np.array(
            [0.8 + 0.05j, 1.0 + 0.10j, 1.2 + 0.15j, 1.4 + 0.20j],
            dtype=np.complex64,
        ),
        ground_truth_conductivity=np.ones(1, dtype=np.complex64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "debug_fine_mesh_noser",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )
    monkeypatch.setattr(
        window,
        "_submit_scheduled_ui_task",
        lambda **kwargs: kwargs["callback"]() or SimpleNamespace(accepted=True),
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.use_part == "complex"
    assert request.metadata["eit_value_mode"] == "complex_admittance"
    assert request.metadata["complex_reconstruction_dispatch"] == "native_complex"
    assert np.allclose(request.reference_frame.real, [0.8, 1.0, 1.2, 1.4])
    assert np.allclose(request.reference_frame.imag, [0.05, 0.10, 0.15, 0.20])
    assert np.allclose(request.target_frame.real, [1.0, 1.2, 1.4, 1.6])
    assert np.allclose(request.target_frame.imag, [0.10, 0.20, 0.30, 0.40])

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
@pytest.mark.parametrize("method", ["noser_rm", "laplace_rm", "curvature_rm"])
def test_simulation_2d_rm_routes_use_normalized_difference_mode(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """2D production RM routes must not use raw voltage scale."""
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": method,
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )
    monkeypatch.setattr(
        window,
        "_submit_scheduled_ui_task",
        lambda **kwargs: kwargs["callback"]() or SimpleNamespace(accepted=True),
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.metadata["simulation_inverse_route"] == method
    assert request.metadata["difference_mode"] == "normalized"
    assert request.metadata["mesh_dimension"] == 2
    assert request.metadata["electrode_length_m_override"] is None
    assert request.metadata["electrode_coverage"] == pytest.approx(0.5)

    window._sim_state.inverse_running = False
    _close_window(window)


def test_simulation_voltage_fit_restores_absolute_boundary_voltages() -> None:
    """V102: simulation fit plots absolute voltage, not normalized/raw diff."""

    forward = ForwardSolverResult(
        boundary_voltages=np.array([11.0, 16.0], dtype=np.float64),
        homogeneous_voltages=np.array([10.0, 20.0], dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=2,
    )
    normalized = ReconstructionResult(
        conductivity=np.ones(1, dtype=np.float64),
        node_coords=forward.node_coords,
        cell_connectivity=forward.cell_connectivity,
        simulated=np.array([0.1, -0.2], dtype=np.float64),
        metadata={
            "difference_mode": "normalized",
            "difference_orientation": "target_minus_reference",
        },
    )
    restored = EITWorkstation._simulation_reconstructed_voltage_fit(normalized, forward)
    assert restored is not None
    assert restored == pytest.approx([11.0, 16.0])

    raw_reverse = ReconstructionResult(
        conductivity=np.ones(1, dtype=np.float64),
        node_coords=forward.node_coords,
        cell_connectivity=forward.cell_connectivity,
        simulated=np.array([-1.0, 4.0], dtype=np.float64),
        metadata={
            "difference_mode": "raw",
            "difference_orientation": "reference_minus_target",
        },
    )
    restored = EITWorkstation._simulation_reconstructed_voltage_fit(
        raw_reverse, forward
    )
    assert restored is not None
    assert restored == pytest.approx([11.0, 16.0])

    complex_forward = ForwardSolverResult(
        boundary_voltages=np.array([11.0 + 1.5j, 16.0 + 2.0j], dtype=np.complex128),
        homogeneous_voltages=np.array([10.0 + 1.0j, 20.0 + 1.0j], dtype=np.complex128),
        ground_truth_conductivity=np.ones(1, dtype=np.complex128),
        node_coords=forward.node_coords,
        cell_connectivity=forward.cell_connectivity,
        n_elements=1,
        n_measurements=2,
    )
    complex_raw = ReconstructionResult(
        conductivity=np.ones(1, dtype=np.complex128),
        node_coords=forward.node_coords,
        cell_connectivity=forward.cell_connectivity,
        simulated=np.array([1.0 + 0.5j, -4.0 + 1.0j], dtype=np.complex128),
        metadata={
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
        },
    )
    restored = EITWorkstation._simulation_reconstructed_voltage_fit(
        complex_raw, complex_forward
    )
    assert restored is not None
    assert np.allclose(restored, np.array([11.0 + 1.5j, 16.0 + 2.0j]))


def test_v474_simulation_voltage_fit_finite_check_uses_bounded_scan() -> None:
    source = inspect.getsource(EITWorkstation._simulation_reconstructed_voltage_fit)
    helper_source = inspect.getsource(main_window_module._all_finite_values)

    assert "np.isfinite(reconstructed).all()" not in source
    assert "_all_finite_values(reconstructed)" in source
    assert "np.isfinite(chunk, out=chunk_mask)" in helper_source
    assert main_window_module._all_finite_values(np.array([1.0, 2.0]))
    assert not main_window_module._all_finite_values(np.array([1.0, np.nan]))

    forward = ForwardSolverResult(
        boundary_voltages=np.array([1.0, 2.0], dtype=np.float64),
        homogeneous_voltages=np.array([1.0, 2.0], dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0]], dtype=np.int32),
        n_elements=1,
        n_measurements=2,
    )
    result = ReconstructionResult(
        conductivity=np.ones(1, dtype=np.float64),
        node_coords=forward.node_coords,
        cell_connectivity=forward.cell_connectivity,
        simulated=np.array([0.0, np.nan], dtype=np.float64),
        metadata={"difference_mode": "raw"},
    )

    assert EITWorkstation._simulation_reconstructed_voltage_fit(result, forward) is None


@pytest.mark.gui
@pytest.mark.parametrize("method", ["noser_rm", "laplace_rm", "curvature_rm"])
def test_simulation_3d_rm_routes_use_normalized_difference_mode(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """3D RM-based routes must build/apply normalized dv and J."""
    window = EITWorkstation()
    _show_window(window)
    monkeypatch.setattr(
        window, "_schedule_sim_forward_prewarm", lambda *args, **kwargs: None
    )

    n_meas = 208
    forward_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    # Switch the simulation tab to the 3D mesh so is_3d_difference fires.
    window._sim_tab.mesh_setup_panel._dim_combo.setCurrentIndex(1)
    _get_app().processEvents()
    window._last_fwd_result = forward_result
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": method,
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )
    monkeypatch.setattr(
        window,
        "_submit_scheduled_ui_task",
        lambda **kwargs: kwargs["callback"]() or SimpleNamespace(accepted=True),
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.metadata["simulation_inverse_route"] == method
    assert request.metadata["difference_mode"] == "normalized"
    assert request.metadata["mesh_dimension"] == 3

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_pseudo3d_route_uses_layered_2d_inverse_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)
    monkeypatch.setattr(
        window, "_schedule_sim_forward_prewarm", lambda *args, **kwargs: None
    )

    n_meas = 208
    forward_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.mesh_setup_panel._dim_combo.setCurrentIndex(1)
    _get_app().processEvents()
    window._last_fwd_result = forward_result
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "pseudo3d_noser_rm",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )
    monkeypatch.setattr(
        window,
        "_submit_scheduled_ui_task",
        lambda **kwargs: kwargs["callback"]() or SimpleNamespace(accepted=True),
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.mesh_dimension == 2
    assert request.metadata["simulation_inverse_route"] == "pseudo3d_noser_rm"
    assert request.metadata["simulation_inverse_route_kind"] == "rm"
    assert request.metadata["rm_regularization"] == "noser"
    assert request.metadata["difference_mode"] == "normalized"
    assert request.metadata["mesh_dimension"] == 2
    assert request.metadata["n_elec"] == 8
    assert request.metadata["n_rings"] == 1
    assert request.metadata["pseudo3d_output"] is True
    assert request.metadata["pseudo3d_layered_output"] is True
    assert request.metadata["pseudo3d_source_mesh_dimension"] == 3
    assert request.metadata["pseudo3d_source_n_elec"] == 8
    assert request.metadata["pseudo3d_source_n_rings"] == 2
    assert request.metadata["pseudo3d_layer_count"] == 2
    assert request.metadata["pseudo3d_layer_n_elec"] == 8
    assert request.metadata["pseudo3d_display_mesh_dimension"] == 3
    assert request.metadata["petsc_device"] == "cpu"
    assert request.metadata["device"] == "cpu"
    assert request.metadata["rm_device"] == "cpu"
    assert request.metadata["forward_backend"] == "dolfinx"
    assert request.metadata["acceleration_profile"] == "default"

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
@pytest.mark.parametrize(
    ("method", "pending_task", "regularization", "auto_build", "display_mode"),
    [
        ("noser_rm", "", "noser", True, "absolute_sigma"),
        ("laplace_rm", "", "laplace", True, "absolute_sigma"),
        ("curvature_rm", "", "curvature", True, "absolute_sigma"),
        ("greit", "", "greit", True, ""),
    ],
)
def test_simulation_rm_routes_record_artifact_requirement(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    pending_task: str,
    regularization: str,
    auto_build: bool,
    display_mode: str,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": method,
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "gn-difference"
    assert request.metadata["reconstruction_runtime"] == "single_step_cached"
    assert request.metadata["simulation_inverse_route"] == method
    assert request.metadata["simulation_inverse_route_kind"] == "rm"
    assert request.metadata["simulation_inverse_debug_route"] is False
    assert request.metadata["rm_route_requires_artifact"] is True
    assert request.metadata["rm_auto_build"] is auto_build
    assert request.metadata["online_hot_path"] == "rm_matmul"
    assert request.metadata["rm_route_pending_task"] == pending_task
    assert request.metadata["rm_regularization"] == regularization
    assert request.metadata["difference_preset"] == method
    assert request.metadata["rm_output_display_mode"] == display_mode
    if method in {"noser_rm", "laplace_rm", "curvature_rm"}:
        assert request.metadata["difference_mode"] == "normalized"
        expected_form = "measurement" if method == "noser_rm" else "param"
        assert request.metadata["rm_form"] == expected_form
        assert request.metadata["rm_inverse_mesh_size"] <= request.metadata["mesh_size"]
        assert request.metadata["rm_artifact_dir"] == str(pyeidors_cache_path("gui_rm"))
        assert request.regularization_alpha == pytest.approx(1.0e-2)
        assert request.metadata["difference_lambda"] == pytest.approx(1.0e-2)
        assert request.metadata["hyperparameter_ui_name"] == "lambda_eff"
        assert request.metadata["hyperparameter_ui_locked"] is True
        assert request.metadata["lambda_eff_custom_enabled"] is False
        assert request.metadata["regularization_alpha_applied"] is False
        assert request.metadata["lambda_eff"] == pytest.approx(1.0e-2)
        assert request.metadata["hp"] == pytest.approx(0.1)
        assert request.metadata["hp_squared"] == pytest.approx(1.0e-2)
    if method == "greit":
        assert request.metadata["rm_form"] == "measurement"
        assert "greit_common_config" not in request.metadata
        assert "greit_common_config_dir" not in request.metadata
        assert "greit_common_config_auto_warm" not in request.metadata
        assert request.metadata["greit_registry_auto_resolve"] is True
        assert request.metadata["greit_registry_signature"]
        assert request.metadata["greit_registry_config"]["measurement_count"] == n_meas
        assert (
            request.metadata["greit_registry_config"]["builder_semantic_version"]
            == "native-greit-finite-target-v2"
        )
        assert request.metadata["greit_registry_config"]["target_size_semantics"] == (
            "fraction_of_tank_radius"
        )
        assert request.metadata["greit_registry_config"]["weight_strategy"] == "fixed"
        assert request.metadata["greit_weight_strategy"] == "fixed"
        assert (
            "config-driven registry artifact"
            in request.metadata["greit_common_config_unavailable_reason"]
        )
        assert (
            request.metadata["greit_official_fixture_scope"]
            == "requires registered EIDORS parity artifact"
        )
        assert request.metadata["greit_5936_protocol_scope"] == (
            "production route rejects deterministic fixtures"
        )
        assert request.metadata["greit_official_equivalence_claim_allowed"] is False
        assert "difference_lambda" not in request.metadata
        assert request.regularization_alpha == pytest.approx(1.0)
        assert request.metadata["hyperparameter_ui_name"] == "greit_weight"
        assert request.metadata["hyperparameter_ui_value"] == pytest.approx(1.0)
        assert request.metadata["hyperparameter_ui_locked"] is False
        assert request.metadata["hyperparameter_effective_source"] == (
            "greit_gui_advanced"
        )
        assert request.metadata["regularization_alpha_applied"] is False
        assert request.metadata["difference_lambda_semantics"] == (
            "unused_for_greit_artifact"
        )

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_rm_route_v117_custom_lambda_rebuilds_distinct_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "noser_rm",
            "regularization_alpha": 0.04,
            "lambda_eff_custom_enabled": True,
            "custom_lambda_eff": 0.04,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "gn-difference"
    assert request.regularization_alpha == pytest.approx(0.04)
    assert request.metadata["simulation_inverse_route"] == "noser_rm"
    assert request.metadata["difference_lambda"] == pytest.approx(0.04)
    assert request.metadata["lambda_eff"] == pytest.approx(0.04)
    assert request.metadata["hp"] == pytest.approx(0.2)
    assert request.metadata["hp_squared"] == pytest.approx(0.04)
    assert request.metadata["lambda_eff_custom_enabled"] is True
    assert request.metadata["hyperparameter_ui_locked"] is False
    assert request.metadata["hyperparameter_effective_source"] == "custom_rm_rebuild"
    assert request.metadata["regularization_alpha_applied"] is False
    assert request.metadata["rm_rebuild_required_by_custom_lambda"] is True
    assert request.metadata["difference_lambda_semantics"] == (
        "custom_lambda_eff_rebuilds_rm"
    )
    default_request = rc.ReconstructionRequest(
        reference_frame=request.reference_frame,
        target_frame=request.target_frame,
        use_part=request.use_part,
        method=request.method,
        regularization_alpha=1.0e-2,
        max_iterations=request.max_iterations,
        mesh_dimension=request.mesh_dimension,
        mesh_refinement=request.mesh_refinement,
        metadata={
            **request.metadata,
            "difference_lambda": 1.0e-2,
            "lambda_eff": 1.0e-2,
            "lambda_eff_custom_enabled": False,
        },
    )
    assert rc.get_single_step_cached_cache_key(
        request
    ) != rc.get_single_step_cached_cache_key(default_request)

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_default_noser_rm_hot_path_updates_gui_without_fragmentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyeidors.inverse import write_rm_artifact

    window = EITWorkstation()
    window.resize(1200, 800)
    _show_window(window)
    window._tab_widget.setCurrentWidget(window._sim_tab)
    _get_app().processEvents()

    node_coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    cells = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
        ],
        dtype=np.int32,
    )
    reference = np.full(4, 2.0, dtype=np.float64)
    target = reference + np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    expected_sigma = np.array([1.0, 1.2, 1.2, 1.0], dtype=np.float64)
    rm = np.diag([0.0, 0.4, 0.4, 0.0]).astype(np.float64)
    fit_jacobian = np.diag([0.0, 2.5, 2.5, 0.0]).astype(np.float64)
    artifact_path = write_rm_artifact(
        tmp_path / "gui_default_noser_rm.h5",
        rm,
        metadata={"algorithm": "one-step-noser", "rm_build_route": "noser_rm"},
        node_coords=node_coords,
        cell_connectivity=cells,
        jacobian=fit_jacobian,
    )
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=target,
        homogeneous_voltages=reference,
        ground_truth_conductivity=expected_sigma,
        node_coords=node_coords,
        cell_connectivity=cells,
        n_elements=len(cells),
        n_measurements=len(target),
        forward_model_config={
            "mesh_dimension": 2,
            "mesh_refinement": 0.1,
            "mesh_family": "tri",
            "n_elec": 4,
            "radius": 1.0,
            "height": 0.0,
            "background_conductivity": 1.0,
            "stim_pattern": "{ad}",
            "meas_pattern": "{ad}",
        },
    )
    window._sim_tab.results_widget.update_forward_result(window._last_fwd_result)
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.metadata["simulation_inverse_route"] == "noser_rm"
    assert request.metadata["simulation_inverse_route_kind"] == "rm"
    assert request.metadata["simulation_inverse_debug_route"] is False
    assert request.metadata["rm_route_requires_artifact"] is True
    assert request.metadata["online_hot_path"] == "rm_matmul"
    assert "rm_artifact_path" not in request.metadata

    def _unexpected_context(*_args, **_kwargs):
        raise AssertionError("Default NOSER GUI route must use RM hot path.")

    def _unexpected_runner():
        raise AssertionError("Default NOSER GUI route must not import dense GN runner.")

    monkeypatch.setattr(rc, "_ensure_single_step_cached_context", _unexpected_context)
    monkeypatch.setattr(rc, "_load_gn_difference_runner_module", _unexpected_runner)
    request.metadata["rm_artifact_path"] = str(artifact_path)
    request.metadata["rm_auto_build"] = False
    request.metadata["device"] = "cpu"
    result = rc._run_single_step_cached_request(request)

    np.testing.assert_allclose(result.conductivity, expected_sigma)
    assert result.metadata["rm_artifact_path"] == str(artifact_path)
    assert result.metadata["single_step_operator_space"] == "rm"
    assert result.metadata["online_hot_path"] == "rm_matmul"
    diagnostics = result.metadata["solver_diagnostics"]
    assert diagnostics["runtime"]["forward_solve_count"] == 0
    assert diagnostics["runtime"]["adjoint_solve_count"] == 0
    assert diagnostics["runtime"]["jacobian_rebuild_count"] == 0
    assert diagnostics["runtime"]["ksp_solve_count"] == 0

    delta = result.conductivity - 1.0
    assert delta[1:3].mean() > 0.15
    assert abs(delta[1] - delta[2]) <= 1.0e-12
    assert abs(delta[0] - delta[3]) <= 1.0e-12

    window._sim_recon_handler(result)
    _get_app().processEvents()

    recon_slot = window._sim_tab.results_widget._reconstruction_widget
    recon_widget = recon_slot._mpl
    assert recon_slot._stack.currentWidget() is recon_widget
    assert recon_widget._last_image is not None
    last_sigma, last_coords, last_cells, _title = recon_widget._last_image
    np.testing.assert_allclose(last_sigma, expected_sigma)
    np.testing.assert_allclose(last_coords, node_coords)
    np.testing.assert_array_equal(last_cells, cells)
    assert _sample_pixmap_unique_rgb_count(recon_slot.grab()) > 1
    assert (
        str(len(node_coords)) in window._sim_tab.metrics_panel._truth_mesh_label.text()
    )
    assert str(len(cells)) in window._sim_tab.metrics_panel._truth_mesh_label.text()
    assert (
        str(len(node_coords)) in window._sim_tab.metrics_panel._recon_mesh_label.text()
    )
    assert str(len(cells)) in window._sim_tab.metrics_panel._recon_mesh_label.text()
    assert window._sim_tab.metrics_panel._l2_label.text() == "0.0000"
    assert float(window._sim_tab.metrics_panel._corr_label.text()) >= 0.999

    voltage_plot = window._sim_tab.results_widget.voltage_plot
    assert voltage_plot._has_data is True
    assert voltage_plot._curve_primary.isVisible() is True
    assert voltage_plot._curve_reconstructed.isVisible() is True
    primary_x, primary_y = voltage_plot._curve_primary.getData()
    recon_x, recon_y = voltage_plot._curve_reconstructed.getData()
    assert primary_x is not None and primary_y is not None
    assert recon_x is not None and recon_y is not None
    np.testing.assert_allclose(primary_y, target)
    np.testing.assert_allclose(recon_y, target)
    assert len(recon_y) == len(target)
    plotted = np.concatenate(
        [np.asarray(primary_y, dtype=float), np.asarray(recon_y, dtype=float)]
    )
    _x_range, y_range = voltage_plot._plot_widget.getPlotItem().getViewBox().viewRange()
    assert y_range[0] <= float(np.nanmin(plotted))
    assert y_range[1] >= float(np.nanmax(plotted))
    assert (
        voltage_plot._curve_reconstructed_outline.zValue()
        < voltage_plot._curve_primary.zValue()
    )

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_greit3d_route_uses_registry_without_broad_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 5936
    forward_config = {
        "mesh_dimension": 3,
        "mesh_refinement": 0.1,
        "n_elec": 16,
        "n_rings": 3,
        "electrode_layout": "ring_major",
        "measurement_protocol": "eidors_full_3d",
        "radius": 0.18,
        "height": 0.16,
        "drive_mode": "total_current",
    }
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 0, 0, 0]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
        forward_model_config=forward_config,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "greit",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert "greit_common_config" not in request.metadata
    assert request.metadata["greit_registry_auto_resolve"] is True
    assert request.metadata["rm_auto_build"] is True
    assert request.metadata["greit_registry_config"]["measurement_count"] == n_meas
    assert request.metadata["greit_registry_config"]["n_rings"] == 3
    assert request.metadata["greit_registry_config"]["imgsz"] == (16, 16, 8)
    assert request.metadata["greit_registry_config"]["desired_solution_fn"] == "gauss"
    assert request.metadata["greit_registry_config"]["target_size"] == pytest.approx(
        0.20
    )
    assert request.metadata["greit_registry_signature"]
    assert request.metadata["greit_official_fixture_scope"] == (
        "requires registered EIDORS parity artifact"
    )
    assert request.metadata["greit_5936_protocol_scope"] == (
        "production route rejects deterministic fixtures"
    )
    assert request.metadata["greit_official_equivalence_claim_allowed"] is False
    assert (
        "EIDORS-parity artifacts matching the current geometry and protocol"
        in request.metadata["greit_official_equivalence_scope"]
    )
    assert "difference_lambda" not in request.metadata
    assert request.metadata["hyperparameter_ui_name"] == "greit_weight"
    assert request.metadata["hyperparameter_ui_locked"] is False
    assert request.metadata["regularization_alpha_applied"] is False

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_greit3d_route_uses_advanced_registry_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 5936
    forward_config = {
        "mesh_dimension": 3,
        "mesh_refinement": 0.1,
        "n_elec": 16,
        "n_rings": 3,
        "electrode_layout": "ring_major",
        "measurement_protocol": "eidors_full_3d",
        "radius": 0.18,
        "height": 0.5,
        "drive_mode": "total_current",
    }
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 0, 0, 0]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
        forward_model_config=forward_config,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "greit",
            "greit_desired_image_mode": "adaptive_gauss",
            "greit_training_target_count": 96,
            "greit_target_size": 0.15,
            "greit_weight": 0.75,
            "greit_use_cached_rm": False,
            "greit_rebuild_rm": True,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    registry_config = request.metadata["greit_registry_config"]
    assert request.regularization_alpha == pytest.approx(0.75)
    assert registry_config["desired_solution_fn"] == "adaptive_gauss"
    assert registry_config["desired_solution_params"]["desired_img_sampling"] == (
        "adaptive_gauss"
    )
    assert registry_config["greit_training_target_count_requested"] == 96
    assert np.prod(registry_config["imgsz"]) >= 96
    assert registry_config["target_size"] == pytest.approx(0.15)
    assert registry_config["weight_strategy"] == "fixed"
    assert registry_config["weight"] == pytest.approx(0.75)
    assert registry_config["greit_use_cached_rm"] is False
    assert registry_config["greit_rebuild_rm"] is True
    assert request.metadata["greit_desired_image_mode"] == "adaptive_gauss"
    assert request.metadata["greit_training_target_count"] == 96
    assert request.metadata["greit_weight_strategy"] == "fixed"
    assert request.metadata["greit_use_cached_rm"] is False
    assert request.metadata["greit_rebuild_rm"] is True
    assert request.metadata["hyperparameter_ui_value"] == pytest.approx(0.75)
    assert request.metadata["hyperparameter_effective_source"] == ("greit_gui_advanced")

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_greit_route_supports_2d_and_eidors_nf1_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    forward_config = {
        "mesh_dimension": 2,
        "mesh_refinement": 0.1,
        "n_elec": 16,
        "n_rings": 1,
        "electrode_layout": "ring_major",
        "measurement_protocol": "eidors_adjacent",
        "radius": 1.0,
        "height": 2.0,
        "drive_mode": "total_current",
    }
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
        forward_model_config=forward_config,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "greit",
            "greit_weight_strategy": "eidors_nf1",
            "greit_weight": 0.75,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    registry_config = request.metadata["greit_registry_config"]
    assert request.metadata["simulation_inverse_route"] == "greit"
    assert registry_config["mesh_dimension"] == 2
    assert registry_config["imgsz"] == (32, 32, 1)
    assert registry_config["rec_mask"] == "circular_fem_area_v1"
    assert registry_config["weight_strategy"] == "eidors_nf1"
    assert "weight" not in registry_config
    assert registry_config["noise_figure"] == pytest.approx(1.0)
    assert request.regularization_alpha == pytest.approx(1.0)
    assert request.metadata["hyperparameter_ui_name"] == "greit_noise_figure"
    assert request.metadata["hyperparameter_ui_locked"] is True
    assert request.metadata["regularization_alpha_applied"] is False

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_greit3d_route_2160_adad_gets_distinct_registry_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 2160
    forward_config = {
        "mesh_dimension": 3,
        "mesh_refinement": 0.1,
        "n_elec": 16,
        "n_rings": 3,
        "electrode_layout": "ring_major",
        "measurement_protocol": "eidors_full_3d",
        "radius": 0.18,
        "height": 0.16,
        "drive_mode": "total_current",
    }
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 0, 0, 0]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
        forward_model_config=forward_config,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "greit",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.metadata["simulation_inverse_route"] == "greit"
    assert "greit_common_config" not in request.metadata
    assert request.metadata["rm_auto_build"] is True
    assert request.metadata["greit_registry_auto_resolve"] is True
    assert request.metadata["greit_registry_config"]["measurement_count"] == n_meas
    assert request.metadata["greit_registry_config"]["imgsz"] == (16, 16, 8)
    assert request.metadata["greit_registry_signature"]
    assert (
        "config-driven registry artifact"
        in request.metadata["greit_common_config_unavailable_reason"]
    )

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_debug_full_gn_route_stays_explicit_debug_cold_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "debug_full_gn",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "gn-difference"
    assert request.metadata["reconstruction_runtime"] == "full_gn"
    assert request.metadata["simulation_inverse_route"] == "debug_full_gn"
    assert request.metadata["simulation_inverse_debug_route"] is True
    assert "difference_lambda" not in request.metadata
    assert request.metadata.get("online_hot_path") != "rm_matmul"
    assert request.regularization_alpha == pytest.approx(1.0)
    assert request.metadata["hyperparameter_ui_name"] == "alpha"
    assert request.metadata["hyperparameter_ui_locked"] is False
    assert request.metadata["hyperparameter_ui_value"] == pytest.approx(1.0)
    assert request.metadata["regularization_alpha_applied"] is True

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_absolute_gn_route_v116_uses_absolute_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "absolute_gn",
            "regularization_alpha": 1.25,
            "max_iterations": 7,
        }
    )
    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "gn-absolute"
    assert request.max_iterations == 7
    assert request.metadata["reconstruction_runtime"] == "full_gn"
    assert request.metadata["simulation_inverse_route"] == "absolute_gn"
    assert request.metadata["simulation_inverse_route_kind"] == "absolute"
    assert request.metadata["simulation_inverse_debug_route"] is False
    assert request.metadata["absolute_preset"] == "eidors_abs_gn"
    assert request.metadata.get("online_hot_path") != "rm_matmul"
    assert "difference_lambda" not in request.metadata
    assert request.regularization_alpha == pytest.approx(1.25)
    assert request.metadata["hyperparameter_ui_name"] == "alpha"
    assert request.metadata["hyperparameter_ui_locked"] is False
    assert request.metadata["hyperparameter_ui_value"] == pytest.approx(1.25)
    assert request.metadata["regularization_alpha_applied"] is True

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_inverse_uses_config_stored_with_forward_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    forward_config = {
        "mesh_dimension": 3,
        "mesh_refinement": 0.1,
        "mesh_family": "tetra",
        "n_elec": 8,
        "n_rings": 2,
        "electrode_layout": "zigzag",
        "measurement_protocol": "eidors_full_3d",
        "radius": 0.18,
        "height": 0.16,
        "drive_mode": "total_current",
    }
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 0, 0, 0]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
        forward_model_config=forward_config,
    )
    window._sim_tab.mesh_setup_panel.set_config(
        {
            "mesh_dimension": 3,
            "mesh_family": "hex",
            "mesh_refinement": 0.2,
            "n_electrodes": 16,
            "n_rings": 1,
            "electrode_layout": "ring_major",
        }
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "debug_fine_mesh_noser",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []

    def _capture_reconstruct(request) -> bool:
        captured.append(request)
        return True

    monkeypatch.setattr(window._sim_recon_ctrl, "reconstruct", _capture_reconstruct)
    monkeypatch.setattr(
        window,
        "_submit_scheduled_ui_task",
        lambda **kwargs: kwargs["callback"]() or SimpleNamespace(accepted=True),
    )

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.mesh_refinement == pytest.approx(0.1)
    assert request.metadata["mesh_family"] == "tetra"
    assert request.metadata["n_elec"] == 8
    assert request.metadata["n_rings"] == 2
    assert request.metadata["electrode_layout"] == "zigzag"
    assert request.metadata["electrode_length_m_override"] == pytest.approx(
        2.0 * np.pi * 0.18 * 0.5 / 16.0
    )

    window._sim_state.inverse_running = False
    _close_window(window)


@pytest.mark.gui
def test_simulation_inverse_request_propagates_compute_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eit_app.models.precision import current_precision, set_precision

    original_precision = current_precision()
    set_precision("float32", persist=False)
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
    )
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "noser_rm",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []

    def _capture_reconstruct(request) -> bool:
        captured.append(request)
        return True

    monkeypatch.setattr(window._sim_recon_ctrl, "reconstruct", _capture_reconstruct)
    monkeypatch.setattr(
        window,
        "_submit_scheduled_ui_task",
        lambda **kwargs: kwargs["callback"]() or SimpleNamespace(accepted=True),
    )
    try:
        window._on_run_sim_inverse()

        assert len(captured) == 1
        request = captured[0]
        assert request.reference_frame.real.dtype == np.float32
        assert request.target_frame.real.dtype == np.float32
        assert request.metadata["compute_precision"] == "float32"
        assert request.metadata["compute_dtype"] == "float32"
        assert request.metadata["rm_dtype"] == "float32"
        assert request.metadata["rm_matmul_dtype"] == "float32"
    finally:
        set_precision(original_precision, persist=False)
        _close_window(window)


@pytest.mark.gui
def test_simulation_forward_request_records_input_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    window._sim_tab.inhomogeneity_editor._add_shape("circle")
    captured: list[object] = []
    monkeypatch.setattr(
        window._fwd_ctrl,
        "solve",
        lambda request: captured.append(request),
    )

    window._on_run_forward()

    assert len(captured) == 1
    request = captured[0]
    signature = request.forward_model_config.get("simulation_input_signature")
    payload = request.forward_model_config.get("simulation_input_signature_payload")
    assert isinstance(signature, str) and len(signature) == 64
    assert payload["schema"] == "simulation_forward_inputs_v1"
    assert len(payload["inhomogeneities"]) == 1
    assert payload["inhomogeneities"][0]["shape"] == "circle"

    window._sim_state.forward_running = False
    _close_window(window)


@pytest.mark.gui
def test_v140_simulation_forward_prewarm_uses_same_request_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM", "1")
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM_START_MS", "25")
    window = EITWorkstation()
    _show_window(window)
    assert window._initial_sim_forward_prewarm_delay_ms() == 25

    captured: list[object] = []

    def _fake_solve(request) -> bool:
        captured.append(request)
        return True

    monkeypatch.setattr(window._fwd_prewarm_ctrl, "solve", _fake_solve)

    window._run_sim_forward_prewarm()

    assert len(captured) == 1
    request = captured[0]
    assert (
        request.forward_model_config["request_source"] == "simulation_forward_prewarm"
    )
    assert isinstance(
        request.forward_model_config["simulation_input_signature"],
        str,
    )
    assert window._fwd_prewarm_busy is True

    window._on_sim_forward_prewarm_done(
        ForwardSolverResult(
            boundary_voltages=np.array([1.0], dtype=np.float32),
            ground_truth_conductivity=np.array([1.0], dtype=np.float32),
            node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
            cell_connectivity=np.array([[0, 0, 0]], dtype=np.int32),
            n_elements=1,
            n_measurements=1,
        )
    )
    assert window._fwd_prewarm_busy is False
    assert window._fwd_prewarm_ready_signature is not None

    _close_window(window)


@pytest.mark.gui
def test_v610_3d_simulation_prewarm_defaults_to_setup_prime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM", "1")
    monkeypatch.delenv("EIT_APP_FORWARD_PREWARM_3D_MODE", raising=False)
    window = EITWorkstation()
    _show_window(window)
    window._sim_tab.mesh_setup_panel.set_config(
        {
            "mesh_dimension": 3,
            "mesh_family": "tetra",
            "mesh_refinement": 0.1,
            "n_electrodes": 16,
            "n_rings": 1,
            "electrode_layout": "ring_major",
        }
    )
    warmed: list[object] = []

    setup_flags: list[bool] = []

    def _fake_warm_backend(request, *, setup_prime=False):
        warmed.append(request)
        setup_flags.append(bool(setup_prime))

    monkeypatch.setattr(
        window, "_warm_sim_forward_backend_if_needed", _fake_warm_backend
    )
    monkeypatch.setattr(
        window._fwd_prewarm_ctrl,
        "solve",
        lambda _request: (_ for _ in ()).throw(
            AssertionError("3D default prewarm must not run a full solve")
        ),
    )

    window._schedule_sim_forward_prewarm()

    assert len(warmed) == 1
    assert warmed[0].mesh_dimension == 3
    assert setup_flags == [True]
    assert window._fwd_prewarm_busy is False

    _close_window(window)


@pytest.mark.gui
def test_v148_3d_simulation_prewarm_worker_mode_keeps_import_only_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM", "1")
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM_3D_MODE", "worker")
    window = EITWorkstation()
    _show_window(window)
    window._sim_tab.mesh_setup_panel.set_config(
        {
            "mesh_dimension": 3,
            "mesh_family": "tetra",
            "mesh_refinement": 0.1,
            "n_electrodes": 16,
            "n_rings": 1,
            "electrode_layout": "ring_major",
        }
    )
    setup_flags: list[bool] = []

    def _fake_warm_backend(request, *, setup_prime=False):
        assert request.mesh_dimension == 3
        setup_flags.append(bool(setup_prime))

    monkeypatch.setattr(
        window, "_warm_sim_forward_backend_if_needed", _fake_warm_backend
    )
    monkeypatch.setattr(
        window._fwd_prewarm_ctrl,
        "solve",
        lambda _request: (_ for _ in ()).throw(
            AssertionError("3D worker prewarm must not run a full solve")
        ),
    )

    window._schedule_sim_forward_prewarm()

    assert setup_flags == [False]
    assert window._fwd_prewarm_busy is False

    _close_window(window)


@pytest.mark.gui
def test_v317_3d_simulation_backend_warmup_reports_status_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM", "1")
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM_3D_MODE", "worker")
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM_START_MS", "5000")
    monkeypatch.setenv("EIT_APP_GUI_RUNTIME_PROFILE", "complex64-cuda")
    monkeypatch.setenv("EIT_APP_GUI_PROFILE", "gpu")

    class _ImmediateBackgroundScheduler:
        def __init__(self, *args, **kwargs) -> None:
            self.handles = []

        def submit(self, *, key, name, priority, fn, coalesce=True):
            handle = SimpleNamespace(
                key=key,
                name=name,
                priority=priority,
                accepted=True,
                reason="",
            )
            self.handles.append(handle)
            fn()
            return handle

        def shutdown(self, *, wait=True, timeout=2.0) -> None:
            return None

    monkeypatch.setattr(
        main_window_module,
        "BackgroundTaskScheduler",
        _ImmediateBackgroundScheduler,
    )
    warmed: list[tuple[Path, str]] = []

    def _fake_warm_backend_worker(*, repo, profile, progress_cb=None):
        warmed.append((Path(repo), str(profile)))
        if progress_cb is not None:
            progress_cb("fake backend warm progress")
        return SimpleNamespace(
            profile=str(profile),
            pid=4242,
            rss_bytes=2 * 1024 * 1024,
            rss_limit_bytes=4 * 1024 * 1024,
            primed_runtime=True,
            prime_command="prime_runtime",
            prime_duration_ms=8.0,
            prime_metadata={
                "petsc_cuda_probe": {
                    "petsc_cuda": True,
                    "probe_cache": {"hit": True, "layer": "disk"},
                },
            },
            recycled_after_request=False,
            recycle_reason="",
        )

    monkeypatch.setattr(
        "eit_app.backend_worker_pool.warm_persistent_backend_worker",
        _fake_warm_backend_worker,
    )
    window = EITWorkstation()
    _show_window(window)
    messages: list[str] = []
    window.sim_backend_warm_status.connect(messages.append)
    window._sim_tab.mesh_setup_panel.set_config(
        {
            "mesh_dimension": 3,
            "mesh_family": "tetra",
            "mesh_refinement": 0.1,
            "n_electrodes": 16,
            "n_rings": 1,
            "electrode_layout": "ring_major",
        }
    )

    window._schedule_sim_forward_prewarm()

    assert len(warmed) == 1
    assert warmed[0][1] == "cuda"
    report = window._fwd_backend_warm_reports["cuda"]
    assert report["pid"] == 4242
    assert report["primed_runtime"] is True
    assert report["prime_duration_ms"] == 8.0
    assert report["petsc_cuda_available"] is True
    assert report["petsc_cuda_probe_cache_hit"] is True
    assert report["petsc_cuda_probe_cache_layer"] == "disk"
    assert report["petsc_cuda_probe_status"] == "hit/disk"
    assert _wait_until(
        lambda: any("4242" in item and "2.0 MiB" in item for item in messages),
        timeout=1.0,
    )
    assert any("cuda" in item for item in messages)
    assert any("4242" in item and "2.0 MiB" in item for item in messages)
    assert any("PETSc probe=hit/disk" in item for item in messages)

    _close_window(window)


def test_v329_backend_worker_probe_summary_accepts_setup_metadata() -> None:
    summary = main_window_module._backend_worker_probe_summary(
        {
            "petsc_cuda": True,
            "petsc_cuda_probe_cache": {"hit": False, "layer": "disk"},
        }
    )

    assert summary["petsc_cuda"] is True
    assert summary["cache_hit"] is False
    assert summary["cache_layer"] == "disk"
    assert summary["status_text"] == "miss/disk"


@pytest.mark.gui
def test_v319_forward_done_records_gui_visualization_timing() -> None:
    window = EITWorkstation()
    _show_window(window)
    result = ForwardSolverResult(
        boundary_voltages=np.array([1.0], dtype=np.float32),
        ground_truth_conductivity=np.array([1.0], dtype=np.float32),
        node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 0, 0]], dtype=np.int32),
        n_elements=1,
        n_measurements=1,
        forward_model_config={
            "runtime_diagnostics": {},
            "forward_timing_ms": {"total": 4.0},
            "forward_timing_phase_order": ["total"],
        },
    )

    window._on_forward_done(result)

    timings = result.forward_model_config["forward_timing_ms"]
    assert timings["gui_visualization_update"] >= 0.0
    assert (
        result.forward_model_config["gui_forward_visualization_update_ms"]
        == timings["gui_visualization_update"]
    )
    assert (
        "gui_visualization_update"
        in result.forward_model_config["forward_timing_phase_order"]
    )

    _close_window(window)


@pytest.mark.gui
def test_v148_3d_simulation_prewarm_solve_mode_keeps_full_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM", "1")
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM_3D_MODE", "solve")
    window = EITWorkstation()
    _show_window(window)
    window._sim_tab.mesh_setup_panel.set_config(
        {
            "mesh_dimension": 3,
            "mesh_family": "tetra",
            "mesh_refinement": 0.1,
            "n_electrodes": 16,
            "n_rings": 1,
            "electrode_layout": "ring_major",
        }
    )
    captured: list[object] = []

    monkeypatch.setattr(
        window._fwd_prewarm_ctrl,
        "solve",
        lambda request: captured.append(request) or True,
    )

    window._run_sim_forward_prewarm()

    assert len(captured) == 1
    assert captured[0].mesh_dimension == 3
    assert window._fwd_prewarm_busy is True

    window._fwd_prewarm_busy = False
    _close_window(window)


@pytest.mark.gui
def test_v140_run_forward_promotes_matching_inflight_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM", "1")
    window = EITWorkstation()
    _show_window(window)

    signature, _payload = window._current_simulation_input_signature()
    window._fwd_prewarm_busy = True
    window._fwd_prewarm_active_signature = signature

    def _fail_solve(_request) -> bool:
        raise AssertionError("matching prewarm should be promoted, not restarted")

    monkeypatch.setattr(window._fwd_ctrl, "solve", _fail_solve)

    window._on_run_forward()

    assert window._fwd_prewarm_promote_to_user is True
    assert window._sim_state.forward_running is True

    result = ForwardSolverResult(
        boundary_voltages=np.array([1.0], dtype=np.float32),
        ground_truth_conductivity=np.array([1.0], dtype=np.float32),
        node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 0, 0]], dtype=np.int32),
        n_elements=1,
        n_measurements=1,
        forward_model_config={
            "simulation_input_signature": signature,
        },
    )
    window._on_sim_forward_prewarm_done(result)

    assert window._fwd_prewarm_promote_to_user is False
    assert window._sim_state.forward_running is False
    assert window._last_fwd_result is result

    _close_window(window)


@pytest.mark.gui
def test_v140_run_forward_reuses_completed_matching_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIT_APP_FORWARD_PREWARM", "1")
    window = EITWorkstation()
    _show_window(window)

    signature, _payload = window._current_simulation_input_signature()
    result = ForwardSolverResult(
        boundary_voltages=np.array([1.0], dtype=np.float32),
        ground_truth_conductivity=np.array([1.0], dtype=np.float32),
        node_coords=np.array([[0.0, 0.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 0, 0]], dtype=np.int32),
        n_elements=1,
        n_measurements=1,
        forward_model_config={
            "simulation_input_signature": signature,
            "request_source": "simulation_forward_prewarm",
        },
    )
    window._fwd_prewarm_ready_signature = signature
    window._fwd_prewarm_ready_result = result

    def _fail_solve(_request) -> bool:
        raise AssertionError("completed matching prewarm should be reused")

    monkeypatch.setattr(window._fwd_ctrl, "solve", _fail_solve)

    window._on_run_forward()

    assert window._last_fwd_result is result
    assert window._sim_state.forward_running is False
    assert window._fwd_prewarm_ready_result is None
    assert window._fwd_prewarm_ready_signature is None
    assert result.forward_model_config["request_source"] == "simulation_forward"
    assert result.forward_model_config["served_from_sim_forward_prewarm"] is True

    _close_window(window)


@pytest.mark.gui
def test_simulation_inverse_blocks_stale_forward_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    n_meas = 208
    signature, payload = window._current_simulation_input_signature()
    window._last_fwd_result = ForwardSolverResult(
        boundary_voltages=np.linspace(1.0, 2.0, n_meas, dtype=np.float64),
        homogeneous_voltages=np.linspace(0.8, 1.8, n_meas, dtype=np.float64),
        ground_truth_conductivity=np.ones(1, dtype=np.float64),
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        cell_connectivity=np.array([[0, 1, 2]], dtype=np.int32),
        n_elements=1,
        n_measurements=n_meas,
        forward_model_config={
            **window._current_sim_forward_model_config().to_mapping(),
            "simulation_input_signature": signature,
            "simulation_input_signature_payload": payload,
        },
    )
    window._sim_tab.inhomogeneity_editor._add_shape("circle")
    window._on_simulation_inputs_changed()
    live_status = window._sim_tab.inverse_problem_panel._status_label.text()
    assert live_status.splitlines() == [
        "Simulation inputs changed.",
        "Run the forward problem again before reconstruction.",
    ]

    captured: list[object] = []
    monkeypatch.setattr(
        window._sim_recon_ctrl,
        "reconstruct",
        lambda request: captured.append(request) or True,
    )

    window._on_run_sim_inverse()

    assert captured == []
    blocked_status = window._sim_tab.inverse_problem_panel._status_label.text()
    assert blocked_status.splitlines() == [
        "Simulation inputs changed after the last forward solve.",
        "Run the forward problem again before reconstruction.",
    ]

    _close_window(window)


@pytest.mark.gui
def test_live_plot_uses_positive_expected_measurement_index_range() -> None:
    window = EITWorkstation()
    _show_window(window)

    live_plot = window._live_plot
    assert live_plot.current_point_count() == 208
    x_min, x_max = live_plot._plot_widget.viewRange()[0]
    assert x_min >= 0.5
    assert x_max >= 208

    window._device_config["n_elec"] = 32
    window._sync_state_device_config()
    _get_app().processEvents()

    expected = estimate_measurement_point_count(
        n_electrodes=32,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        use_meas_current=False,
        use_meas_current_next=0,
    )
    assert live_plot.current_point_count() == expected
    x_min, x_max = live_plot._plot_widget.viewRange()[0]
    assert x_min >= 0.5
    assert x_max >= expected

    _close_window(window)


@pytest.mark.gui
def test_acquisition_pipeline_uses_dynamic_measurement_count() -> None:
    window = EITWorkstation()
    _show_window(window)

    window._device_config["n_elec"] = 32
    window._sync_state_device_config()
    window._rebuild_acquisition_pipeline()

    expected = estimate_measurement_point_count(
        n_electrodes=32,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        use_meas_current=False,
        use_meas_current_next=0,
    )
    assert window._ring_buffer is not None
    assert window._acq_process is not None
    assert window._ring_buffer._n_meas == expected
    assert window._acq_process._n_meas == expected
    assert window._acq_ctrl._frame_metadata["points_per_frame"] == expected
    assert window._acq_ctrl._frame_metadata["n_elec"] == 32

    window._reset_acquisition_pipeline()
    _close_window(window)


@pytest.mark.gui
def test_hardware_layout_controls_update_expected_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)
    monkeypatch.setattr(
        window, "_schedule_realtime_recon_prewarm", lambda *args, **kwargs: None
    )

    window._on_connected()
    initial_arc_span = _first_electrode_arc_span(window._recon_widget)
    initial_boundary_radius = _boundary_radius(window._recon_widget)

    window._control_panel._n_elec_spin.setValue(8)
    _get_app().processEvents()

    assert window._live_plot.current_point_count() == 40
    assert window._voltage_plot.current_point_count() == 40
    assert "expected 40 boundary samples" in window._control_panel._layout_hint.text()
    assert "8E x 1R" in window._summary_panel._values["layout"].text()
    assert "40 pts" in window._summary_panel._values["layout"].text()

    window._control_panel._n_elec_spin.setValue(16)
    window._control_panel._electrode_length_spin.setValue(0.020001)
    _get_app().processEvents()

    shortened_arc_span = _first_electrode_arc_span(window._recon_widget)
    assert shortened_arc_span < initial_arc_span * 0.2
    assert "CEM L=0.0200" in window._control_panel._layout_hint.text()
    assert "cov=5.1%" in window._control_panel._layout_hint.text()

    window._control_panel._radius_spin.setValue(1.5)
    _get_app().processEvents()

    resized_boundary_radius = _boundary_radius(window._recon_widget)
    resized_arc_span = _first_electrode_arc_span(window._recon_widget)
    assert resized_boundary_radius == pytest.approx(initial_boundary_radius * 1.5)
    assert resized_arc_span < shortened_arc_span
    assert "cov=3.4%" in window._control_panel._layout_hint.text()
    assert "L=0.0200" in window._summary_panel._values["layout"].text()

    _close_window(window)
