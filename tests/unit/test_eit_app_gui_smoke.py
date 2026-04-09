from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolBox

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from eit_app.models.app_state import ConnectionStatus, ReconstructionConfig
from eit_app.ui.auto_close_combo_box import AutoCloseComboBox
from eit_app.ui.dialogs.difference_dialog import DifferenceDialog
from eit_app.ui.dialogs.settings_dialog import SettingsDialog
from eit_app.ui.main_window import EITWorkstation
from pyeidors.data.frame_io import read_frame_yaml, read_session_metadata


def _get_app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


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


def _close_window(window: EITWorkstation) -> None:
    window.close()
    assert _wait_until(lambda: not window._device_ctrl._thread.isRunning(), timeout=3.0)


def _as_wsl_unc(path: Path) -> str:
    posix_path = str(path)
    if not posix_path.startswith("/"):
        raise ValueError("Expected absolute POSIX path")
    return "\\\\wsl.localhost\\Ubuntu-22.04" + posix_path.replace("/", "\\")


@pytest.mark.gui
def test_main_window_constructs_with_expected_default_state() -> None:
    app = _get_app()
    window = EITWorkstation()
    _show_window(window)
    app.processEvents()

    assert window.centralWidget() is not None
    assert window.windowTitle() == "EIT Workstation"
    assert window._conn_panel._transport_combo.count() == 3
    assert window._conn_panel.title() == "1. Link & Verify"
    assert window._control_panel.title() == "2. Setup & Diagnostics"
    assert window._acq_panel.title() == "3. Acquire & Record"
    assert window._summary_panel.title() == "4. Current Session Summary"
    assert isinstance(window._workflow_toolbox, QToolBox)
    assert window._workflow_toolbox.count() == 3
    assert window._workflow_toolbox.currentIndex() == 0
    assert window._state.connection_status is ConnectionStatus.DISCONNECTED
    assert window._control_panel._freq_spin.isEnabled() is False
    assert window._control_panel._vamp_combo_1.currentIndex() == 3
    assert window._control_panel._vamp_combo_2.currentIndex() == 5
    assert window._acq_panel.output_dir() == str(window._acq_panel.default_output_dir())
    assert window._acq_panel._start_btn.text() == "Start Continuous"
    assert window._acq_panel._single_frame_btn.text() == "Acquire One Frame"
    assert window._acq_panel._stop_btn.text() == "Stop Acquisition"
    assert window._frame_browser._model.rowCount() == 0
    assert window._status_bar._conn_label.text() == "Link: Down"
    assert window._status_bar._power_label.text() == "Power: Unknown"
    assert window._status_bar._acq_label.text() == "Acq: Idle"
    assert window._status_bar._rec_label.text() == "Record: Off"
    assert window._summary_panel._state_badge.text() == "LINK DOWN"
    assert "Connect & Verify" in window._summary_panel._next_action.text()
    assert window._summary_panel._indicator_values["link"].text() == "DOWN"
    assert window._summary_panel._indicator_values["power"].text() == "UNK"
    assert window._summary_panel._indicator_values["record"].text() == "OFF"
    assert window._summary_panel._indicator_values["acq"].text() == "IDLE"
    assert window._summary_panel._values["link"].text() == "Down"
    assert window._summary_panel._values["identity"].text() == "Board 1 | User 1 | 2D"
    assert "legacy-v1" in window._summary_panel._values["protocol"].text()
    assert "Serial" in window._summary_panel._values["transport"].text()
    assert window._summary_panel._values["frequency"].text() == "1000 Hz"
    assert "100 uA" in window._summary_panel._values["stim"].text()
    assert "data/measurements" in window._summary_panel._values["record"].text()
    assert window._summary_panel._values["mode"].text() == "Idle | manual"

    _close_window(window)


@pytest.mark.gui
def test_simulator_continuous_acquisition_and_recording_smoke(tmp_path: Path) -> None:
    app = _get_app()
    window = EITWorkstation()
    _show_window(window)
    output_dir = tmp_path / "recordings"

    window._on_connect_requested("simulator", {})
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
    assert window._scheduler is None

    _close_window(window)


@pytest.mark.gui
def test_simulator_scheduled_acquisition_smoke() -> None:
    app = _get_app()
    window = EITWorkstation()
    _show_window(window)

    window._on_connect_requested("simulator", {})
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    window._on_scheduled_mode_changed(True, 0.2, 1)
    window._on_start_acquisition()

    assert window._scheduler is not None
    assert window._workflow_toolbox.currentIndex() == 2
    assert window._status_bar._acq_label.text() == "Acq: Scheduled"
    assert window._status_bar._power_label.text() == "Power: ON"
    assert window._summary_panel._state_badge.text() == "ACQUIRING"
    assert window._summary_panel._indicator_values["acq"].text() == "SCH"
    assert "simulator" in window._summary_panel._values["protocol"].text()
    assert "Scheduled | every 0.2s | 1 frame/burst" == window._summary_panel._values["mode"].text()
    assert _wait_until(lambda: window._state.frame_count > 0, timeout=8.0)

    window._on_stop_acquisition()
    app.processEvents()

    assert window._scheduler is None
    assert window._acq_process is None
    assert window._state.frame_count >= 1

    _close_window(window)


@pytest.mark.gui
def test_simulator_single_frame_capture_stops_automatically(tmp_path: Path) -> None:
    window = EITWorkstation()
    _show_window(window)

    window._on_connect_requested("simulator", {})
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
    assert "单帧采集完成" in window._status_bar.currentMessage()
    csv_files = sorted(output_dir.rglob("*.csv"))
    yaml_files = sorted(output_dir.rglob("*.yaml"))
    assert len(csv_files) == 1
    assert len(yaml_files) >= 2  # session metadata + frame metadata

    _close_window(window)


@pytest.mark.gui
def test_difference_and_settings_dialog_smoke() -> None:
    _get_app()

    entries = [
        {"frame_index": 0, "timestamp": 1.0, "file_path": "/tmp/ref.csv"},
        {"frame_index": 1, "timestamp": 2.0, "file_path": "/tmp/tgt.csv"},
    ]
    dialog = DifferenceDialog(entries)
    emitted: list[dict] = []
    dialog.reconstruction_requested.connect(emitted.append)
    dialog._mode_combo.setCurrentText("normalized")
    dialog._orient_combo.setCurrentText("reference_minus_target")
    dialog._part_combo.setCurrentText("imag")
    dialog._on_accept()

    assert emitted
    assert emitted[0]["mode"] == "normalized"
    assert emitted[0]["orientation"] == "reference_minus_target"
    assert emitted[0]["use_part"] == "imag"

    config = ReconstructionConfig()
    settings = SettingsDialog(config)
    settings._method_combo.setCurrentText("sparse-bayes")
    settings._alpha_spin.setValue(0.25)
    settings._iter_spin.setValue(7)
    settings._dim_combo.setCurrentIndex(1)
    settings._refine_spin.setValue(5)
    settings._part_combo.setCurrentText("mag")
    settings._on_accept()

    new_config = settings.get_config()
    assert new_config.method == "sparse-bayes"
    assert new_config.regularization_alpha == pytest.approx(0.25)
    assert new_config.max_iterations == 7
    assert new_config.mesh_dimension == 3
    assert new_config.mesh_refinement == 5
    assert new_config.use_part == "mag"


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

    window._conn_panel._transport_combo.setCurrentIndex(2)
    _click(window._conn_panel._connect_btn)

    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )
    assert window._status_bar._conn_label.text() == "Link: Verified"
    assert window._status_bar._power_label.text() == "Power: Unknown"
    assert window._status_bar._acq_label.text() == "Acq: Idle"
    assert window._workflow_toolbox.currentIndex() == 1
    assert window._summary_panel._values["link"].text() == "Verified"
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["link"].text() == "OK"
    assert window._summary_panel._indicator_values["power"].text() == "UNK"
    assert window._conn_panel._connect_btn.isEnabled() is False
    assert window._conn_panel._disconnect_btn.isEnabled() is True
    assert window._control_panel._freq_spin.isEnabled() is True

    window._control_panel._freq_spin.setValue(2500)
    _click(window._control_panel._freq_apply)
    assert _wait_until(
        lambda: window._device_config["frequency_hz"] == 2500
        and "set_frequency" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert window._summary_panel._values["frequency"].text() == "2500 Hz"

    window._control_panel._stim_combo.setCurrentIndex(3)
    _click(window._control_panel._stim_apply)
    assert _wait_until(
        lambda: window._device_config["stim_amp_level"] == 3
        and window._device_config["stim_amp_uA"] == 500
        and "set_stim_amplitude" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert "500 uA" in window._summary_panel._values["stim"].text()

    window._control_panel._vamp_combo_1.setCurrentIndex(2)
    window._control_panel._vamp_combo_2.setCurrentIndex(5)
    _click(window._control_panel._vamp_apply)
    assert _wait_until(
        lambda: window._device_config["voltage_amp_level_1"] == 2
        and window._device_config["voltage_amp_level_2"] == 5
        and "set_voltage_amp_levels" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert "0.327x" in window._summary_panel._values["gain"].text()

    _click(window._control_panel._imp_btn)
    assert _wait_until(
        lambda: window._status_bar.currentMessage().startswith("接触阻抗:"),
        timeout=3.0,
    )

    _click(window._control_panel._power_on_btn)
    assert _wait_until(
        lambda: window._status_bar._power_label.text() == "Power: ON"
        and "测量电源命令已发送" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert window._summary_panel._values["power"].text() == "ON"
    assert window._workflow_toolbox.currentIndex() == 2
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["power"].text() == "ON"

    _click(window._control_panel._power_off_btn)
    assert _wait_until(
        lambda: window._status_bar._power_label.text() == "Power: OFF"
        and "测量电源命令已发送" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert window._summary_panel._values["power"].text() == "OFF"
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["power"].text() == "OFF"

    _click(window._conn_panel._disconnect_btn)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.DISCONNECTED,
        timeout=3.0,
    )
    assert window._status_bar._conn_label.text() == "Link: Down"
    assert window._status_bar._power_label.text() == "Power: Unknown"
    assert window._summary_panel._values["link"].text() == "Down"
    assert window._control_panel._freq_spin.isEnabled() is False

    _close_window(window)


@pytest.mark.gui
def test_gui_interaction_regression_for_fps_recording_and_frame_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = EITWorkstation()
    _show_window(window)

    window._conn_panel._transport_combo.setCurrentIndex(2)
    _click(window._conn_panel._connect_btn)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    output_dir = tmp_path / "gui-recordings"
    window._acq_panel._dir_edit.setText(str(output_dir))
    _click(window._acq_panel._rec_check)
    assert _wait_until(lambda: window._acq_panel._rec_check.isChecked() is True, timeout=2.0)
    assert window._acq_panel._rec_check.isEnabled() is True
    assert "开始采集后将保存到" in window._status_bar.currentMessage()
    assert window._status_bar._rec_label.text() == "Record: Armed"
    assert window._workflow_toolbox.currentIndex() == 1
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["record"].text() == "ARM"
    assert "Armed" in window._summary_panel._values["record"].text()
    assert str(output_dir) in window._summary_panel._values["record"].text()

    _click(window._acq_panel._start_btn)
    assert _wait_until(lambda: window._status_bar._rec_label.text() == "Record: Writing", timeout=2.0)
    assert window._workflow_toolbox.currentIndex() == 2
    assert window._status_bar._acq_label.text() == "Acq: Continuous"
    assert window._status_bar._power_label.text() == "Power: ON"
    assert window._summary_panel._state_badge.text() == "ACQUIRING + RECORDING"
    assert window._summary_panel._indicator_values["record"].text() == "REC"
    assert window._summary_panel._indicator_values["acq"].text() == "RUN"
    assert window._summary_panel._values["mode"].text() == "Continuous"
    assert _wait_until(lambda: window._state.frame_count > 0, timeout=8.0)
    assert _wait_until(lambda: _fps_value(window) > 0.0, timeout=8.0)
    assert _wait_until(lambda: window._frame_browser._model.rowCount() >= 2, timeout=8.0)

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
        lambda: window._selected_reference_entry is not None
        and window._selected_reference_entry.get("file_path") == first_entry["file_path"]
        and "参考帧已选择" in window._status_bar.currentMessage(),
        timeout=2.0,
    )

    window._frame_browser._table.selectRow(1)
    _click(window._frame_browser._tgt_btn)
    assert _wait_until(
        lambda: window._selected_target_entry is not None
        and window._selected_target_entry.get("file_path") == second_entry["file_path"]
        and "目标帧已选择" in window._status_bar.currentMessage(),
        timeout=2.0,
    )

    captured: dict[str, int] = {}

    def _fake_exec(self) -> int:
        captured["ref"] = self._ref_combo.currentIndex()
        captured["tgt"] = self._tgt_combo.currentIndex()
        return 0

    monkeypatch.setattr(DifferenceDialog, "exec", _fake_exec)
    window._open_difference_dialog()
    assert captured == {"ref": 0, "tgt": 1}

    _click(window._acq_panel._stop_btn)
    assert _wait_until(lambda: window._acq_process is None, timeout=5.0)
    assert window._status_bar._fps_label.text() == "FPS: 0.0"
    assert window._status_bar._rec_label.text() == "Record: Armed"
    assert window._status_bar._acq_label.text() == "Acq: Idle"
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["record"].text() == "ARM"
    assert window._summary_panel._indicator_values["acq"].text() == "IDLE"
    assert "Idle | manual" == window._summary_panel._values["mode"].text()
    assert window._acq_panel._rec_check.isChecked() is True
    assert window._acq_panel._start_btn.isEnabled() is True
    assert window._acq_panel._stop_btn.isEnabled() is False

    _click(window._frame_browser._clear_btn)
    assert _wait_until(lambda: window._frame_browser._model.rowCount() == 0, timeout=2.0)
    assert window._selected_reference_entry is None
    assert window._selected_target_entry is None
    assert "已清空录制帧列表" in window._status_bar.currentMessage()

    _close_window(window)


@pytest.mark.gui
def test_record_checkbox_reverts_when_recording_start_fails(tmp_path: Path) -> None:
    window = EITWorkstation()
    _show_window(window)

    def _fail_start_recording(*args, **kwargs) -> bool:
        return False

    window._conn_panel._transport_combo.setCurrentIndex(2)
    _click(window._conn_panel._connect_btn)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    window._acq_panel._dir_edit.setText(str(tmp_path / "failing-recordings"))
    window._rec_ctrl.start_recording = _fail_start_recording  # type: ignore[method-assign]

    _click(window._acq_panel._rec_check)
    _click(window._acq_panel._start_btn)

    assert _wait_until(lambda: window._acq_panel._rec_check.isChecked() is False, timeout=2.0)
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
        lambda: window._acq_panel.output_dir() == str(window._acq_panel.default_output_dir()),
        timeout=2.0,
    )
    assert "开始采集后将保存到" in window._status_bar.currentMessage()
    assert window._status_bar._rec_label.text() == "Record: Armed"

    _close_window(window)


@pytest.mark.gui
def test_recording_supports_unc_output_path_and_uses_latest_directory(tmp_path: Path) -> None:
    window = EITWorkstation()
    _show_window(window)

    window._conn_panel._transport_combo.setCurrentIndex(2)
    _click(window._conn_panel._connect_btn)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    output_dir = tmp_path / "recordings-unc"
    unc_path = _as_wsl_unc(output_dir)
    window._acq_panel._dir_edit.setText(unc_path)

    _click(window._acq_panel._rec_check)
    assert _wait_until(lambda: window._acq_panel._dir_edit.text() == str(output_dir), timeout=2.0)

    _click(window._acq_panel._start_btn)
    assert _wait_until(lambda: window._rec_ctrl.frames_recorded > 0, timeout=8.0)

    _click(window._acq_panel._stop_btn)
    csv_files = sorted(output_dir.rglob("*.csv"))
    yaml_files = sorted(output_dir.rglob("*.yaml"))
    assert csv_files
    assert yaml_files

    _close_window(window)


@pytest.mark.gui
def test_close_event_powers_off_connected_device(monkeypatch: pytest.MonkeyPatch) -> None:
    window = EITWorkstation()
    _show_window(window)

    called = {"power_off": False}

    def _fake_power_off(timeout_ms: int = 3000) -> bool:
        called["power_off"] = True
        return True

    monkeypatch.setattr(window._device_ctrl, "power_off_device", _fake_power_off)
    window._state.set_connection_status(ConnectionStatus.CONNECTED)

    _close_window(window)

    assert called["power_off"] is True


@pytest.mark.gui
def test_live_plot_can_toggle_imag_and_magnitude_after_frame_arrives() -> None:
    window = EITWorkstation()
    _show_window(window)

    window._conn_panel._transport_combo.setCurrentIndex(2)
    _click(window._conn_panel._connect_btn)
    assert _wait_until(
        lambda: window._state.connection_status is ConnectionStatus.CONNECTED,
        timeout=3.0,
    )

    _click(window._acq_panel._start_btn)
    assert _wait_until(lambda: window._state.frame_count > 0, timeout=8.0)

    real_x, real_y = window._live_plot._curve_real.getData()
    imag_x, imag_y = window._live_plot._curve_imag.getData()
    mag_x, mag_y = window._live_plot._curve_mag.getData()
    assert real_x is not None
    assert real_y is not None
    assert imag_x is not None
    assert imag_y is not None
    assert mag_x is not None
    assert mag_y is not None
    assert len(real_y) == 208
    assert len(imag_y) == 208
    assert len(mag_y) == 208
    assert window._live_plot._curve_imag.isVisible() is False
    assert window._live_plot._curve_mag.isVisible() is True
    assert window._live_plot._show_mag.isChecked() is True

    _click(window._live_plot._show_imag)
    assert window._live_plot._curve_imag.isVisible() is True
    imag_x, imag_y = window._live_plot._curve_imag.getData()
    assert imag_x is not None
    assert imag_y is not None
    assert len(imag_y) == 208

    _click(window._live_plot._show_mag)
    assert window._live_plot._curve_mag.isVisible() is False
    _click(window._live_plot._show_mag)
    assert window._live_plot._curve_mag.isVisible() is True
    mag_x, mag_y = window._live_plot._curve_mag.getData()
    assert mag_x is not None
    assert mag_y is not None
    assert len(mag_y) == 208

    _click(window._acq_panel._stop_btn)
    _close_window(window)
