from __future__ import annotations

import gc
import os
import time
from pathlib import Path

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
from eit_app.ui.dialogs.difference_dialog import DifferenceDialog
from eit_app.ui.hardware.reconstruction_widget import ReconstructionWidget
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
    window._on_connect_requested("simulator", {})


def _splitter_has_center_priority(splitter) -> bool:
    sizes = splitter.sizes()
    return len(sizes) == 3 and sizes[1] > sizes[0] and sizes[1] > sizes[2]


def _close_window(window: EITWorkstation) -> None:
    window.close()
    assert _wait_until(lambda: not window._device_ctrl._thread.isRunning(), timeout=3.0)


def _as_wsl_unc(path: Path) -> str:
    posix_path = str(path)
    if not posix_path.startswith("/"):
        raise ValueError("Expected absolute POSIX path")
    return "\\\\wsl.localhost\\Ubuntu-22.04" + posix_path.replace("/", "\\")


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
                obj.wait(3000)
        except Exception:
            pass
    app.processEvents()


@pytest.mark.gui
def test_reconstruction_widget_pre_renders_static_layout_and_refreshes_internal_image() -> None:
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
def test_connection_panel_auto_selects_unique_windows_serial_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert window._conn_panel.selected_serial_display_name() == "COM4 - USB-SERIAL CH340"
    assert "已自动选中唯一串口" in window._conn_panel._port_hint.text()
    assert "Windows 主机串口桥接" in window._conn_panel._port_hint.text()

    window._conn_panel._port_combo.setCurrentText("COM4 -> /dev/ttyS3 - USB-SERIAL CH340")
    assert window._conn_panel.selected_serial_port() == "COM4"

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
    assert "未检测到可用串口" in window._status_bar.currentMessage()
    assert "未检测到可用串口" in window._conn_panel._port_hint.text()

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

    monkeypatch.setattr(main_window_module, "preflight_connection_target", _fake_preflight)

    window = EITWorkstation()
    _show_window(window)
    monkeypatch.setattr(window._device_ctrl, "connect_device", lambda: connect_calls.append(True))

    window._conn_panel._transport_combo.setCurrentIndex(1)
    window._conn_panel._server_host.setText("relay.example")
    window._conn_panel._server_port.setValue(4555)
    _click(window._conn_panel._connect_btn)
    _get_app().processEvents()

    assert connect_calls == []
    assert window._state.connection_status is ConnectionStatus.ERROR
    assert "无法连接到 4G Relay 服务器 relay.example:4555" in window._status_bar.currentMessage()
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
    assert window._dataset_tab.summary_panel._values["output_dir"].text() == str(dataset_dir)
    assert window._dataset_tab.summary_panel._values["samples"].text() == "12"
    assert window._dataset_tab.summary_panel._state_chip.text() == "Generating"

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
    assert "Stepped Run | 0/3 | every 0.2s | 1000→1200 Hz" == window._summary_panel._values["plan"].text()
    assert _wait_until(lambda: window._state.frame_count == 3, timeout=8.0)
    assert _wait_until(lambda: window._plan_active is False, timeout=8.0)
    assert window._control_panel._freq_spin.value() == 1200
    assert window._status_bar._acq_label.text() == "Acq: Idle"
    assert window._summary_panel._values["plan"].text() == "Idle | Stepped Run 3x | every 0.2s | 1000→1200 Hz"

    window._on_stop_acquisition()
    app.processEvents()

    assert window._acq_process is None
    assert window._state.frame_count == 3

    _close_window(window)


@pytest.mark.gui
def test_fixed_frequency_timed_run_uses_step2_drive_frequency_and_keeps_live_outputs() -> None:
    _get_app()
    window = EITWorkstation()
    _show_window(window)

    window._on_connect_requested("simulator", {})
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
    window._recon_prewarm_ready_signature = window._build_realtime_recon_prewarm_payload()[1]

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
    assert window._summary_panel._values["plan"].text() == "Finite Run | 0/3 | every 0.2s | 2500 Hz"
    assert _wait_until(lambda: window._voltage_plot._has_data is True, timeout=8.0)
    assert _wait_until(lambda: window._state.frame_count == 3, timeout=8.0)
    assert window._summary_panel._values["drive"].text().startswith("2500 Hz")
    assert window._summary_panel._values["plan"].text() == "Idle | Finite Run 3x | every 0.2s | 2500 Hz"

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
    window._recon_prewarm_ready_signature = window._build_realtime_recon_prewarm_payload()[1]

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
        lambda: window._device_config["frequency_hz"] == 2500
        and "set_frequency" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert "2500 Hz" in window._summary_panel._values["drive"].text()

    window._control_panel._stim_combo.setCurrentIndex(3)
    _click(window._control_panel._stim_apply)
    assert _wait_until(
        lambda: window._device_config["stim_amp_level"] == 3
        and window._device_config["stim_amp_uA"] == 500
        and "set_stim_amplitude" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert "500 uA" in window._summary_panel._values["drive"].text()

    window._control_panel._vamp_combo.setCurrentIndex(2)
    _click(window._control_panel._vamp_apply)
    assert _wait_until(
        lambda: window._device_config["voltage_amp_level_1"] == 2
        and window._device_config["voltage_amp_level_2"] == 2
        and "set_voltage_amp_levels" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert "0.327x" in window._summary_panel._values["drive"].text()

    _click(window._control_panel._imp_btn)
    assert _wait_until(
        lambda: window._status_bar.currentMessage().startswith("接触阻抗:"),
        timeout=3.0,
    )

    _click(window._control_panel._power_on_btn)
    assert _wait_until(
        lambda: window._status_bar._power_label.text() == "Power: ON"
        and "测量电源已切换为 ON" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert window._workflow_toolbox.currentIndex() == 1
    assert window._summary_panel._indicator_values["power"].text() == "ON"
    assert window._control_panel._power_on_btn.isChecked() is True
    assert window._control_panel._power_off_btn.isChecked() is False

    _click(window._control_panel._spt_btn)
    assert window._workflow_toolbox.currentIndex() == 1
    assert _wait_until(
        lambda: "单点测试返回:" in window._status_bar.currentMessage(),
        timeout=3.0,
    )
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["power"].text() == "ON"
    assert window._control_panel._power_on_btn.isChecked() is True
    assert window._control_panel._power_off_btn.isChecked() is False
    assert window._workflow_toolbox.currentIndex() == 1

    _click(window._control_panel._power_off_btn)
    assert _wait_until(
        lambda: window._status_bar._power_label.text() == "Power: OFF"
        and "测量电源已切换为 OFF" in window._status_bar.currentMessage(),
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
        lambda: "单点测试返回:" in window._status_bar.currentMessage(),
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
    assert _wait_until(lambda: window._acq_panel._rec_check.isChecked() is True, timeout=2.0)
    assert window._acq_panel._rec_check.isEnabled() is True
    assert "开始采集后将保存到" in window._status_bar.currentMessage()
    assert window._status_bar._rec_label.text() == "Record: Armed"
    assert window._workflow_toolbox.currentIndex() == 1
    assert window._summary_panel._state_badge.text() == "READY FOR ACQUISITION"
    assert window._summary_panel._indicator_values["record"].text() == "ARM"
    assert "Armed" in window._summary_panel._values["record"].text()
    assert str(output_dir) in window._summary_panel._values["record"].text()
    window._recon_prewarm_ready_signature = window._build_realtime_recon_prewarm_payload()[1]

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
    assert _wait_until(lambda: window._status_bar._rec_label.text() == "Record: Writing", timeout=2.0)
    assert window._workflow_toolbox.currentIndex() == 2
    assert window._status_bar._acq_label.text() == "Acq: Continuous"
    assert window._status_bar._power_label.text() == "Power: ON"
    assert window._summary_panel._state_badge.text() == "ACQUIRING + RECORDING"
    assert window._summary_panel._indicator_values["record"].text() == "REC"
    assert window._summary_panel._indicator_values["acq"].text() == "RUN"
    assert window._summary_panel._values["plan"].text() == "Continuous"
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
        and (
            "参考帧已选择" in window._status_bar.currentMessage()
            or "参考帧已更新" in window._status_bar.currentMessage()
        ),
        timeout=2.0,
    )

    window._frame_browser.target_selected.emit(second_entry)
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
    assert "Idle | manual" == window._summary_panel._values["plan"].text()
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

    _connect_simulator(window)
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

    _connect_simulator(window)
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
    window._sim_tab.mesh_setup_panel._refine_spin.setValue(0.1)
    window._sim_tab.inverse_problem_panel.set_config(
        {
            "method": "eidors_one_step_noser",
            "regularization_alpha": 1.0,
            "max_iterations": 10,
        }
    )
    captured: list[object] = []

    def _capture_reconstruct(request) -> bool:
        captured.append(request)
        return True

    monkeypatch.setattr(window._sim_recon_ctrl, "reconstruct", _capture_reconstruct)

    window._on_run_sim_inverse()

    assert len(captured) == 1
    request = captured[0]
    assert request.method == "gn-difference"
    assert request.reference_frame.real.size == n_meas
    assert request.target_frame.real.size == n_meas
    assert request.mesh_refinement == pytest.approx(0.1)
    assert request.metadata["mesh_size"] == pytest.approx(0.1)
    assert request.metadata["reconstruction_runtime"] == "single_step_cached"
    assert rc._prepare_single_step_cached_runtime(request).refinement == 5

    window._sim_state.inverse_running = False
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
    monkeypatch.setattr(window, "_schedule_realtime_recon_prewarm", lambda *args, **kwargs: None)

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
