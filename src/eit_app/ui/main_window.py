"""Main application window with tab-based layout."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtWidgets import QMainWindow, QMessageBox, QTabWidget, QWidget

from eit_app.acquisition.acquisition_process import AcquisitionProcess
from eit_app.acquisition.ring_buffer import FrameRingBuffer
from eit_app.controllers.acquisition_controller import AcquisitionController
from eit_app.controllers.device_controller import DeviceController
from eit_app.controllers.reconstruction_controller import (
    ReconstructionController,
    ReconstructionRequest,
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
from eit_app.controllers.database_controller import DatabaseController
from eit_app.controllers.recording_controller import RecordingController
from eit_app.hardware.connection_preflight import preflight_connection_target
from eit_app.hardware.factory import create_device_from_config, normalize_device_config
from eit_app.hardware.types import STIM_AMP_VALUES_UA
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
from eit_app.models.forward_model_config import ForwardModelConfig
from eit_app.models.simulation_state import (
    DatasetGeneratorConfig,
    SimulationState,
)
from eit_app.ui.database.database_tab import DatabaseTab
from eit_app.ui.hardware.hardware_tab import HardwareTab
from eit_app.ui.simulation.dataset_generator_tab import DatasetGeneratorTab
from eit_app.ui.simulation.simulation_tab import SimulationTab
from eit_app.ui.status_bar import EITStatusBar

from eit_app.models.frame_model import FrameData

log = logging.getLogger(__name__)

_VOLTAGE_GAIN_LABELS = {
    0: "0.097x",
    1: "0.175x",
    2: "0.327x",
    3: "0.623x",
    4: "1.238x",
    5: "2.460x",
    6: "4.880x",
    7: "9.000x",
}


class EITWorkstation(QMainWindow):
    """Main window for the EIT Workstation application."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("EIT Workstation")
        self.resize(1500, 940)

        self._state = AppState(self)
        self._sim_state = SimulationState(self)
        self._device_ctrl = DeviceController(self)
        self._acq_ctrl = AcquisitionController(self)
        self._rec_ctrl = RecordingController(self)
        self._recon_ctrl = ReconstructionController(self)
        self._db_ctrl = DatabaseController(self._default_db_path(), self)
        self._rec_ctrl.set_database_controller(self._db_ctrl)
        self._fwd_ctrl = ForwardSolverController(self)
        self._dataset_ctrl = DatasetGeneratorController(self)
        self._last_fwd_result: ForwardSolverResult | None = None
        self._interop_capture_service = EidorsScriptCaptureService()
        self._interop_importer = InteropBundleImporter()
        self._interop_exporter = InteropBundleExporter()
        self._interop_smoke_validator = InteropSmokeValidator()
        self._sim_forward_model_config = ForwardModelConfig()
        self._dataset_forward_model_config = ForwardModelConfig()
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

        # Database-driven reconstruction state
        self._pending_db_reconstruction: dict | None = None
        self._pending_auto_target_frame: FrameData | None = None

        self._build_ui()
        self._acq_panel.set_output_dir(self._default_output_dir())
        self._connect_signals()
        self._control_panel.set_enabled(False)
        self._refresh_expected_measurement_counts()
        self._refresh_session_summary()

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

        # Hardware Measurement tab
        self._hw_tab = HardwareTab()
        self._tab_widget.addTab(self._hw_tab, "Hardware Measurement (\u5b9e\u6d4b)")

        # Simulation tab
        self._sim_tab = SimulationTab()
        self._tab_widget.addTab(self._sim_tab, "Simulation (\u4eff\u771f)")

        # Dataset generation tab
        self._dataset_tab = DatasetGeneratorTab()
        self._tab_widget.addTab(self._dataset_tab, "Dataset Generator (\u6570\u636e\u96c6\u751f\u6210)")

        # Database tab — persistent archive of all recorded sessions
        self._db_tab = DatabaseTab(self._db_ctrl)
        self._tab_widget.addTab(self._db_tab, "Database (\u6570\u636e\u5e93)")

        self._status_bar = EITStatusBar(self)
        self.setStatusBar(self._status_bar)

        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_menu.addAction("&Settings...", self._open_settings)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        tools_menu = menu.addMenu("&Tools")
        tools_menu.addAction("&Difference Reconstruction...", self._open_difference_dialog)
        tools_menu.addAction("EIDORS &Interop Hub...", self._open_interop_hub)

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

        self._recon_ctrl.reconstruction_done.connect(self._recon_widget.update_reconstruction)
        self._recon_ctrl.reconstruction_done.connect(self._on_hardware_reconstruction_done)
        self._recon_ctrl.reconstruction_done.connect(self._on_auto_reconstruction_done)
        self._recon_ctrl.reconstruction_done.connect(self._on_db_reconstruction_done)
        self._recon_ctrl.progress.connect(
            lambda msg: self._status_bar.showMessage(msg, 3000)
        )
        self._recon_ctrl.error.connect(self._on_error)

        self._frame_browser.reference_selected.connect(self._on_reference_selected)
        self._frame_browser.target_selected.connect(self._on_target_selected)
        self._frame_browser.frame_clicked.connect(self._on_frame_clicked)
        self._frame_browser.cleared.connect(self._on_frame_browser_cleared)

        # Database tab: user-driven reconstruction on historical data
        self._db_tab.reconstruct_requested.connect(self._on_db_reconstruct_requested)
        self._db_tab.open_containing_folder_requested.connect(
            self._on_open_session_folder
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
                self._conn_panel.set_serial_hint(
                    "未检测到可连接串口。请检查 USB 线、驱动、设备供电，然后点击 Scan 重新检测。"
                )
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
                self._conn_panel.set_relay_hint("4G Relay 服务器地址为空，请先填写可访问的 host。")
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
                return f"正在通过 Windows 主机串口 {port} 验证设备链路，波特率 {baud}。"
            return f"正在验证串口链路: {port} @ {baud}"
        if transport_type == "relay":
            host = str(config.get("server_host", "127.0.0.1"))
            port = int(config.get("server_port", 4555))
            return f"正在验证 4G Relay 链路: {host}:{port}"
        return "正在验证设备链路。"

    @Slot()
    def _on_connected(self) -> None:
        self._state.set_connection_status(ConnectionStatus.CONNECTED)
        self._state.set_power_status(PowerStatus.UNKNOWN)
        self._state.set_acquisition_mode(AcquisitionMode.IDLE)
        self._state.set_recording_status(RecordingStatus.OFF)
        self._conn_panel.set_connected(True)
        self._control_panel.set_enabled(True)
        self._workflow_toolbox.setCurrentIndex(1)
        self._status_bar.showMessage("链路连接与协议验证已完成，可按需开启测量电源并开始采集。", 4000)
        self._refresh_session_summary()

    @Slot()
    def _on_disconnected(self) -> None:
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
            self._on_error("请先完成设备连接验证。")
            return

        if self._transport_type != "simulator":
            released = self._device_ctrl.suspend_session(timeout_ms=3000)
            if not released:
                self._on_error("启动采集前未能释放控制串口，请重试或重新连接设备。")
                return

        self._single_frame_pending = single_frame
        self._latest_frame_timestamp = 0.0
        self._state.set_frame_count(0)
        self._auto_reconstruct = not single_frame
        self._reference_frame = None
        self._auto_recon_busy = False
        self._pending_auto_target_frame = None
        if self._record_requested:
            if not self._ensure_recording_session(self._acq_panel.output_dir()):
                self._record_requested = False
                self._state.set_recording_status(RecordingStatus.OFF)

        if single_frame:
            self._rebuild_acquisition_pipeline()
            self._state.set_acquisition_mode(AcquisitionMode.SINGLE_SHOT)
            self._acq_ctrl.capture_one()
            self._status_bar.showMessage("单帧采集已启动，采到 1 帧后将自动停止。", 4000)
        elif self._planned_acquisition_count > 0 or self._frequency_stepping_enabled or self._scheduled_enabled:
            if self._planned_acquisition_count <= 0:
                self._on_error("有限次采集或定时采集需要将 Acquisitions 设置为大于 0。")
                return
            self._start_planned_acquisition_run()
        else:
            self._rebuild_acquisition_pipeline()
            self._state.set_acquisition_mode(AcquisitionMode.CONTINUOUS)
            self._acq_ctrl.start()
            self._status_bar.showMessage("连续采集已启动。", 3000)

        self._state.set_power_status(PowerStatus.ON)
        self._acq_panel.set_acquiring(True)
        self._control_panel.set_enabled(False)
        self._workflow_toolbox.setCurrentIndex(2)
        self._refresh_session_summary()

    @Slot()
    def _on_stop_acquisition(self) -> None:
        self._plan_timer.stop()
        was_single_frame_mode = self._state.acquisition_mode is AcquisitionMode.SINGLE_SHOT
        was_plan_mode = self._plan_active

        if self._acq_process is not None:
            self._acq_ctrl.stop()

        self._reset_acquisition_pipeline()
        self._single_frame_pending = False
        self._planned_step_pending = False
        self._plan_active = False
        self._plan_completed_count = 0
        self._plan_frequencies = []
        self._auto_reconstruct = False
        self._pending_auto_target_frame = None
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
            self._status_bar.showMessage("单帧采集完成。", 4000)
        elif was_plan_mode:
            self._status_bar.showMessage("计划采集已停止。", 4000)
        self._refresh_session_summary()

    @Slot(object)
    def _on_new_frame(self, frame: FrameData) -> None:
        self._latest_frame_timestamp = frame.timestamp
        self._state.set_frame_count(self._acq_ctrl.total_frames)
        if self._rec_ctrl.is_recording:
            self._rec_ctrl.save_frame(frame)
        self._voltage_plot.update_hardware_voltages(frame.real, None)
        if self._plan_active and self._planned_step_pending:
            self._planned_step_pending = False
            self._plan_completed_count += 1
            self._state.set_frame_count(self._plan_completed_count)
            self._acq_ctrl.stop(deactivate_device=False)
            self._reset_acquisition_pipeline()
            if self._plan_completed_count >= len(self._plan_frequencies):
                self._finish_planned_acquisition_run()
            elif self._scheduled_enabled:
                self._plan_timer.start(int(self._scheduled_interval_sec * 1000))
                self._status_bar.showMessage(
                    f"第 {self._plan_completed_count}/{len(self._plan_frequencies)} 次采集完成，"
                    f"{self._scheduled_interval_sec:.1f}s 后开始下一次。",
                    4000,
                )
            else:
                QTimer.singleShot(0, self._run_next_planned_acquisition)
            return
        if self._single_frame_pending and self._state.frame_count >= 1:
            self._single_frame_pending = False
            QTimer.singleShot(0, self._on_stop_acquisition)

        # Auto-reconstruction: first frame becomes reference, subsequent
        # frames are reconstructed as difference against the reference.
        if self._auto_reconstruct:
            if self._reference_frame is None:
                self._reference_frame = frame
                self._frame_browser.set_reference_highlight(0)
                self._status_bar.showMessage(
                    f"Auto-reference set to frame #{frame.frame_index}", 3000
                )
            elif not self._auto_recon_busy:
                self._submit_auto_reconstruction(frame)
            else:
                self._pending_auto_target_frame = frame

    @Slot(int, float, str)
    def _on_frame_saved(self, index: int, timestamp: float, path: str) -> None:
        self._frame_browser.add_frame_entry(index, timestamp, path)

    @Slot(str)
    def _on_recording_started(self, session_dir: str) -> None:
        self._state.set_recording_active(True)
        self._state.set_recording_status(RecordingStatus.RECORDING)
        self._status_bar.showMessage(f"开始录制: {session_dir}", 5000)
        self._refresh_session_summary()

    @Slot(int)
    def _on_recording_stopped(self, count: int) -> None:
        self._state.set_recording_active(False)
        if self._record_requested:
            self._state.set_recording_status(RecordingStatus.ARMED)
        else:
            self._state.set_recording_status(RecordingStatus.OFF)
        self._status_bar.showMessage(f"录制已停止，共保存 {count} 帧。", 5000)
        self._refresh_session_summary()

    # ---- Auto-reconstruction helpers ----

    def _submit_auto_reconstruction(self, target_frame: FrameData) -> None:
        """Submit a difference reconstruction request using the stored reference."""
        if self._reference_frame is None:
            return
        self._auto_recon_busy = True
        self._pending_auto_target_frame = None
        layout_meta = self._measurement_layout_config()
        request = ReconstructionRequest(
            reference_frame=self._reference_frame,
            target_frame=target_frame,
            use_part="real",
            method="gn-difference",
            regularization_alpha=1.0,
            max_iterations=1,
            mesh_dimension=3 if int(self._device_config.get("mea_mode", 2)) == 3 else 2,
            mesh_refinement=int(self._state.reconstruction_config.mesh_refinement),
            metadata={
                **layout_meta,
                "difference_mode": "raw",
                "difference_orientation": "target_minus_reference",
                "drive_mode": "total_current",
                "drive_value": 1.0e-5,
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
            },
        )
        self._recon_ctrl.reconstruct(request)

    @Slot(object)
    def _on_auto_reconstruction_done(self, result) -> None:
        """Handle completed auto-reconstruction during acquisition."""
        self._auto_recon_busy = False
        if result.error_msg:
            # Disable auto-reconstruction on fatal errors (e.g. missing DOLFINx)
            # to prevent spamming the same error every frame.
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
        self._recon_widget.update_reconstruction(result)
        if hasattr(result, "measured") and result.measured is not None:
            self._voltage_plot.update_hardware_voltages(
                result.measured,
                result.simulated if hasattr(result, "simulated") else None,
            )
        if self._auto_reconstruct and self._pending_auto_target_frame is not None:
            QTimer.singleShot(0, self._submit_pending_auto_reconstruction)

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
                    f"参考帧已更新: #{entry.get('frame_index', '?')}", 3000
                )
            except Exception as exc:
                self._on_error(f"Failed to load reference frame: {exc}")
                return
        else:
            self._status_bar.showMessage(
                f"参考帧已选择: #{entry.get('frame_index', '?')}", 3000
            )

    @Slot(dict)
    def _on_target_selected(self, entry: dict) -> None:
        self._selected_target_entry = dict(entry)
        self._status_bar.showMessage(
            f"目标帧已选择: #{entry.get('frame_index', '?')}",
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
                f"显示帧 #{entry.get('frame_index', '?')} 的波形数据",
                3000,
            )
        except Exception as exc:
            self._on_error(f"Failed to load frame: {exc}")

    @Slot()
    def _on_frame_browser_cleared(self) -> None:
        self._selected_reference_entry = None
        self._selected_target_entry = None
        self._status_bar.showMessage("已清空录制帧列表。", 3000)

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
                    f"已启用录制，开始采集后将保存到 {target_dir}",
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

    @Slot(int)
    def _on_stim_amp_changed(self, level: int) -> None:
        self._device_config["stim_amp_level"] = level
        self._device_config["stim_amp_uA"] = STIM_AMP_VALUES_UA.get(level, level)
        self._sync_state_device_config()
        self._device_ctrl.set_connection_profile(self._transport_type, self._device_config)
        self._device_ctrl.set_stim_amplitude(level)
        self._refresh_session_summary()

    @Slot(int)
    def _on_voltage_amp_changed(self, level: int) -> None:
        self._device_config["voltage_amp_level_1"] = level
        self._device_config["voltage_amp_level_2"] = level
        self._device_config["contact_impedance_amp_level"] = level
        self._sync_state_device_config()
        self._device_ctrl.set_connection_profile(self._transport_type, self._device_config)
        self._device_ctrl.set_voltage_amp_levels(level, level)
        self._refresh_session_summary()

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
            f"硬件布局已更新：{points} 个边界电压点。",
            3500,
        )

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
            self._status_bar.showMessage(f"协议能力: {protocol_version}", 3000)
            return

        if name == "single_point_test_at" and isinstance(result, tuple) and len(result) == 2:
            self._status_bar.showMessage(
                f"单点测试返回: real={result[0]:.4f} V, imag={result[1]:.4f} V",
                5000,
            )
            return

        if name == "power_control":
            desired = self._pending_power_commands.pop(0) if self._pending_power_commands else None
            if desired is True:
                self._state.set_power_status(PowerStatus.ON)
                self._control_panel.set_power_state("on")
                self._status_bar.showMessage("测量电源已切换为 ON。", 4000)
            elif desired is False:
                self._state.set_power_status(PowerStatus.OFF)
                self._control_panel.set_power_state("off")
                self._status_bar.showMessage("测量电源已切换为 OFF。", 4000)
            else:
                self._status_bar.showMessage("测量电源命令已发送。", 3000)
            self._refresh_session_summary()
            return

        if name in {"set_frequency", "set_stim_amplitude", "set_voltage_amp_levels"}:
            self._status_bar.showMessage(f"命令已发送: {name}", 3000)

    @Slot(object)
    def _on_impedance_result(self, result: object) -> None:
        try:
            values = list(result)
        except Exception:
            self._status_bar.showMessage("接触阻抗测量完成。", 3000)
            return
        preview = ", ".join(f"{float(v):.4f}" for v in values[:4])
        self._status_bar.showMessage(f"接触阻抗: {preview}", 5000)

    @Slot(object)
    def _on_hardware_reconstruction_done(self, result: object) -> None:
        if self._tab_widget.currentWidget() is not self._hw_tab:
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
                    "当前录制已开始，新保存路径将在下次采集时生效。",
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
                    f"V1 {_VOLTAGE_GAIN_LABELS.get(gain_1, '?')} | "
                    f"V2 {_VOLTAGE_GAIN_LABELS.get(gain_2, '?')}"
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
            freq_info = ""
            if self._frequency_stepping_enabled and self._plan_frequencies:
                freq_info = f" | {self._plan_frequencies[0]}→{self._plan_frequencies[-1]} Hz"
            elif self._plan_frequencies:
                freq_info = f" | {current_hz} Hz"
            if self._scheduled_enabled:
                return (
                    f"Timed run | {self._plan_completed_count}/{len(self._plan_frequencies)}"
                    f" | every {self._scheduled_interval_sec:.1f}s{freq_info}"
                )
            return (
                f"Finite run | {self._plan_completed_count}/{len(self._plan_frequencies)}"
                f"{freq_info}"
            )
        if mode is AcquisitionMode.CONTINUOUS:
            return "Continuous"
        if mode is AcquisitionMode.SINGLE_SHOT:
            return "Single frame"
        if self._scheduled_enabled or self._planned_acquisition_count > 0 or self._frequency_stepping_enabled:
            freq_info = ""
            if self._frequency_stepping_enabled:
                freq_info = f" | {self._planned_start_hz}→{self._planned_end_hz} Hz"
            elif self._planned_acquisition_count > 0:
                freq_info = f" | {current_hz} Hz"
            if self._scheduled_enabled:
                return (
                    f"Idle | timed {self._planned_acquisition_count}x"
                    f" | every {self._scheduled_interval_sec:.1f}s{freq_info}"
                )
            return f"Idle | run {self._planned_acquisition_count}x{freq_info}"
        return "Idle | manual"

    def _summary_banner_state(self) -> tuple[str, str, str, str]:
        if self._state.connection_status is ConnectionStatus.ERROR:
            return (
                "FAULT",
                "The link is in an error state and requires operator attention.",
                "Next: Disconnect the link, check transport settings, and verify again.",
                "error",
            )

        if self._state.connection_status is ConnectionStatus.CONNECTING:
            return (
                "VERIFYING LINK",
                "The workstation is probing the device and reading its protocol capabilities.",
                "Next: Wait for link verification to finish.",
                "warn",
            )

        if self._state.connection_status is ConnectionStatus.DISCONNECTED:
            return (
                "LINK DOWN",
                "No verified device link is active.",
                "Next: Select a transport and click Connect & Verify.",
                "idle",
            )

        if self._state.acquisition_mode in {
            AcquisitionMode.CONTINUOUS,
            AcquisitionMode.SCHEDULED,
            AcquisitionMode.SINGLE_SHOT,
        }:
            if self._state.recording_status is RecordingStatus.RECORDING:
                return (
                    "ACQUIRING + RECORDING",
                    "Frames are being captured and written to the active session.",
                    "Next: Monitor incoming frames or stop acquisition when the run is complete.",
                    "active",
                )
            return (
                "ACQUIRING",
                "Frames are being captured from the active transport.",
                "Next: Monitor the live plot and stop acquisition when the run is complete.",
                "active",
            )

        if self._transport_type == "simulator":
            return (
                "READY FOR ACQUISITION",
                "The simulator link is verified and can start generating frames immediately.",
                "Next: Start continuous or single-frame acquisition.",
                "ready",
            )

        if self._state.power_status is PowerStatus.ON:
            if self._state.recording_status is RecordingStatus.ARMED:
                return (
                    "READY + RECORD ARMED",
                    "The device link is verified, measurement power is ON, and the next run will be saved.",
                    "Next: Start acquisition to capture and record the next session.",
                    "ready",
                )
            return (
                "READY FOR ACQUISITION",
                "The device link is verified and measurement power is ON.",
                "Next: Start continuous or single-frame acquisition.",
                "ready",
            )

        if self._state.recording_status is RecordingStatus.ARMED:
            return (
                "LINK VERIFIED",
                "The link is verified and recording is armed, but measurement power is not confirmed ON.",
                "Next: Turn measurement power ON when the hardware is ready, then start acquisition.",
                "warn",
            )

        return (
            "LINK VERIFIED",
            "The device link is verified and waiting for measurement power or the next setup change.",
            "Next: Turn measurement power ON when the hardware is ready, then start acquisition.",
            "warn",
        )

    def _indicator_link_state(self) -> tuple[str, str]:
        mapping = {
            ConnectionStatus.DISCONNECTED: ("DOWN", "idle"),
            ConnectionStatus.CONNECTING: ("CHECK", "warn"),
            ConnectionStatus.CONNECTED: ("OK", "ready"),
            ConnectionStatus.ERROR: ("FAULT", "error"),
        }
        return mapping.get(self._state.connection_status, ("UNK", "idle"))

    def _indicator_power_state(self) -> tuple[str, str]:
        mapping = {
            PowerStatus.UNKNOWN: ("UNK", "idle"),
            PowerStatus.OFF: ("OFF", "warn"),
            PowerStatus.ON: ("ON", "ready"),
        }
        return mapping.get(self._state.power_status, ("UNK", "idle"))

    def _indicator_record_state(self) -> tuple[str, str]:
        mapping = {
            RecordingStatus.OFF: ("OFF", "idle"),
            RecordingStatus.ARMED: ("ARM", "ready"),
            RecordingStatus.RECORDING: ("REC", "active"),
        }
        return mapping.get(self._state.recording_status, ("OFF", "idle"))

    def _indicator_acq_state(self) -> tuple[str, str]:
        mapping = {
            AcquisitionMode.IDLE: ("IDLE", "idle"),
            AcquisitionMode.CONTINUOUS: ("RUN", "active"),
            AcquisitionMode.SCHEDULED: ("SCH", "active"),
            AcquisitionMode.SINGLE_SHOT: ("1FR", "active"),
        }
        return mapping.get(self._state.acquisition_mode, ("IDLE", "idle"))

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
            AcquisitionMode.SCHEDULED if self._scheduled_enabled else AcquisitionMode.CONTINUOUS
        )
        self._acq_panel.set_acquiring(True)
        self._control_panel.set_enabled(False)
        self._workflow_toolbox.setCurrentIndex(2)
        self._status_bar.showMessage(
            f"计划采集已启动，共 {len(self._plan_frequencies)} 次。",
            3000,
        )
        if self._frequency_stepping_enabled:
            self._status_bar.showMessage(
                "变频采集已启动：将按交频差实时更新波形、边界电压与重构显示。",
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
            f"开始第 {self._plan_completed_count + 1}/{len(self._plan_frequencies)} 次采集：{next_freq} Hz",
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
        self._status_bar.showMessage(f"计划采集完成，共 {completed} 次。", 5000)
        self._refresh_session_summary()

    def _open_difference_dialog(self) -> None:
        from eit_app.ui.dialogs.difference_dialog import DifferenceDialog

        entries = []
        for row in range(self._frame_browser._model.rowCount()):
            entry = self._frame_browser._model.get_entry(row)
            if entry:
                entries.append(entry)

        if len(entries) < 2:
            QMessageBox.information(self, "Info", "Need at least 2 recorded frames.")
            return

        ref_index = self._entry_index(entries, self._selected_reference_entry)
        tgt_index = self._entry_index(entries, self._selected_target_entry)
        if tgt_index == ref_index:
            tgt_index = None

        dialog = DifferenceDialog(
            entries,
            self,
            default_ref_index=ref_index,
            default_tgt_index=tgt_index,
        )
        dialog.reconstruction_requested.connect(self._on_reconstruction_config)
        dialog.exec()

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
                "drive_mode": "total_current",
                "drive_value": 1.0e-5,
                "geometry_scale_to_m": float(self._device_config.get("geometry_scale_to_m", 1.0)),
                "radius": float(self._device_config.get("radius", 1.0)),
                "contact_impedance": float(self._device_config.get("contact_impedance", 0.01)),
                "electrode_length_m_override": self._device_config.get("electrode_length_m_override"),
                "electrode_coverage": float(self._device_config.get("electrode_coverage", 0.5)),
            },
        )
        self._recon_ctrl.reconstruct(request)

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
                "drive_mode": "total_current",
                "drive_value": 1.0e-5,
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
            },
        )
        self._pending_db_reconstruction = dict(config)
        self._status_bar.showMessage(
            f"Running {config.get('method_label', method)}…", 0
        )
        self._recon_ctrl.reconstruct(request)

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
        except Exception as exc:
            log.warning("Failed to save reconstruction outputs: %s", exc)
            self._status_bar.showMessage(f"Save failed: {exc}", 8000)

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
    def _on_open_session_folder(self, folder: str) -> None:
        """Open a session folder using the OS file manager."""
        import subprocess
        import sys
        try:
            if sys.platform == "win32":
                import os
                os.startfile(folder)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as exc:
            self._on_error(f"Failed to open folder: {exc}")

    def _open_settings(self) -> None:
        from eit_app.ui.dialogs.settings_dialog import SettingsDialog

        dialog = SettingsDialog(self._state.reconstruction_config, self)
        if dialog.exec():
            self._state.reconstruction_config = dialog.get_config()

    def _current_hardware_forward_model_config(self) -> ForwardModelConfig:
        return ForwardModelConfig.from_mapping(
            {
                **self._device_config,
                "mesh_dimension": 3 if int(self._device_config.get("mea_mode", 2)) == 3 else 2,
            }
        )

    def _current_sim_forward_model_config(self) -> ForwardModelConfig:
        mesh_cfg = self._sim_tab.mesh_setup_panel.get_config()
        return self._sim_forward_model_config.with_overrides(
            mesh_dimension=mesh_cfg["mesh_dimension"],
            mesh_refinement=mesh_cfg["mesh_refinement"],
            n_elec=mesh_cfg["n_electrodes"],
            background_conductivity=mesh_cfg["background_conductivity"],
            noise_level=self._sim_tab.forward_problem_panel.noise_level,
        )

    def _current_dataset_forward_model_config(self) -> ForwardModelConfig:
        mesh_cfg = self._dataset_tab.mesh_setup_panel.get_config()
        panel_cfg = self._dataset_tab.dataset_generator_panel.get_config()
        return self._dataset_forward_model_config.with_overrides(
            mesh_dimension=mesh_cfg["mesh_dimension"],
            mesh_refinement=mesh_cfg["mesh_refinement"],
            n_elec=mesh_cfg["n_electrodes"],
            noise_level=panel_cfg["noise_level"],
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
                simulation_notes.append(f"无法自动生成 simulation geometry.mat：{exc}")

        recording_notes: list[str] = []
        recording_measurements = self._recording_measurement_export()
        if recording_measurements is not None:
            recording_notes.append("当前录制导出默认使用实部边界电压，以便与 EIDORS 常见差分工作流兼容。")

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
                "notes": ["硬件页当前默认导出布局模板；若需要几何，请先从仿真结果或 bridge 包导入 geometry 资产。"],
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
            return (
                f"已将 bridge 配置导入到硬件页：{config.display_dimension()} | "
                f"{config.n_elec} 电极/环 | {config.point_count()} 点。"
            )

        if target == "simulation":
            self._sim_forward_model_config = config
            self._sim_tab.mesh_setup_panel.set_config(
                {
                    "mesh_dimension": config.mesh_dimension,
                    "mesh_refinement": config.mesh_refinement,
                    "n_electrodes": config.n_elec,
                    "background_conductivity": config.background_conductivity,
                }
            )
            self._sim_tab.forward_problem_panel.set_noise_level(config.noise_level)
            self._sim_tab.results_widget.set_expected_point_count(config.point_count())
            self._tab_widget.setCurrentWidget(self._sim_tab)
            return (
                f"已将 bridge 配置导入到仿真页：{config.display_dimension()} | "
                f"{config.n_elec} 电极/环 | {config.point_count()} 点。"
            )

        if target == "dataset":
            self._dataset_forward_model_config = config
            self._dataset_tab.mesh_setup_panel.set_config(
                {
                    "mesh_dimension": config.mesh_dimension,
                    "mesh_refinement": config.mesh_refinement,
                    "n_electrodes": config.n_elec,
                    "background_conductivity": config.background_conductivity,
                }
            )
            self._dataset_tab.dataset_generator_panel.set_config({"noise_level": config.noise_level})
            self._tab_widget.setCurrentWidget(self._dataset_tab)
            return (
                f"已将 bridge 配置导入到数据集页：{config.display_dimension()} | "
                f"{config.n_elec} 电极/环 | {config.point_count()} 点。"
            )

        if target == "measurements":
            if loaded_bundle.measurements is None:
                raise RuntimeError("这个 bridge 包里没有可导入的边界电压数据。")
            return "已缓存边界电压数据资产，后续可用于导出、对照或重构冒烟。"

        if target == "geometry":
            if loaded_bundle.geometry_payload is None:
                raise RuntimeError("这个 bridge 包里没有 geometry.mat。")
            return "已缓存 geometry 资产，后续导出到 EIDORS 时可直接复用。"

        raise RuntimeError(f"未知导入目标：{target}")

    def _run_interop_smoke_validation(self, loaded_bundle) -> str:
        preset = loaded_bundle.reconstruction_preset or self._interop_reconstruction_preset()
        result = self._interop_smoke_validator.validate(
            loaded_bundle,
            reconstruction_preset=preset,
        )
        return str(result.get("message", "互通烟测已完成。"))

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
            return "未检测到可用串口。请检查 USB 连接、驱动和设备供电后重新 Scan。"

        if "windows 串口" in raw or "未找到串口设备" in raw or "串口 " in raw and "当前无法打开" in raw:
            return raw

        if "4g relay 服务器地址为空" in raw or "无法连接到 4g relay 服务器" in raw:
            return raw

        if "could not configure port" in text or "input/output error" in text:
            return (
                "串口无法配置。当前环境中该端口不可用；请优先从下拉框选择自动检测到的 COM 口，"
                "不要手动填写 /dev/ttyS*。"
            )

        if "windows serial bridge failed" in text:
            if (
                "access to the port" in text
                or "access is denied" in text
                or "denied" in text
                or "访问被拒绝" in raw
                or "拒绝访问" in raw
            ):
                return (
                    "Windows 串口桥接失败：该 COM 口可能仍被其他程序占用；"
                    "如果你刚关闭本软件，请等待 1-2 秒后重试。"
                )
            if "cannot find the file" in text or "cannot find" in text:
                return "Windows 串口桥接失败：当前找不到这个 COM 口，请重新插拔设备后再 Scan。"
            return "Windows 主机串口桥接启动失败，请重新 Scan 后重试。"

        if "relay host is empty" in text:
            return "4G Relay 服务器地址为空，请填写可访问的 host。"

        if "connection refused" in text:
            return "4G Relay 服务器拒绝连接，请检查 host/port 是否正确以及服务是否已启动。"

        if "timed out" in text and "relay" in text:
            return "4G Relay 连接超时，请检查网络、服务器地址和目标设备是否在线。"

        if "permission denied" in text or "access is denied" in text:
            return "串口访问被拒绝，可能被其他程序占用。请关闭占用进程后重试。"

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
        self._fwd_ctrl.solve(request)

    @Slot(object)
    def _on_forward_done(self, result: ForwardSolverResult) -> None:
        self._sim_state.forward_running = False
        self._sim_tab.forward_problem_panel.set_running(False)

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

        # Build a ReconstructionRequest using the forward result data
        from eit_app.models.frame_model import FrameData
        import numpy as np

        n_meas = len(result.boundary_voltages)
        half = n_meas // 2

        # Treat homogeneous as reference, inhomogeneous as target
        ref_frame = FrameData(
            real=result.homogeneous_voltages[:half] if result.homogeneous_voltages is not None else np.zeros(half),
            imag=result.homogeneous_voltages[half:] if result.homogeneous_voltages is not None else np.zeros(n_meas - half),
            timestamp=0.0,
            frame_index=0,
        )
        tgt_frame = FrameData(
            real=result.boundary_voltages[:half],
            imag=result.boundary_voltages[half:] if n_meas > half else np.zeros(n_meas - half),
            timestamp=0.0,
            frame_index=1,
        )

        forward_cfg = self._current_sim_forward_model_config()
        request = ReconstructionRequest(
            reference_frame=ref_frame,
            target_frame=tgt_frame,
            use_part="real",
            method=inv_cfg["method"],
            regularization_alpha=inv_cfg["regularization_alpha"],
            max_iterations=inv_cfg["max_iterations"],
            mesh_dimension=forward_cfg.mesh_dimension,
            mesh_refinement=int(1.0 / max(forward_cfg.mesh_refinement, 1e-6)),
            metadata={
                **forward_cfg.to_mapping(),
                **measurement_layout_from_config(forward_cfg.to_mapping()),
                "difference_mode": "raw",
                "difference_orientation": "target_minus_reference",
            },
        )
        self._recon_ctrl.reconstruct(request)

        # Connect one-shot handler for simulation inverse result
        def _on_sim_recon_done(recon_result):
            self._sim_state.inverse_running = False
            self._sim_tab.inverse_problem_panel.set_running(False)

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
            self._sim_tab.metrics_panel.update_metrics(
                self._last_fwd_result.ground_truth_conductivity,
                recon_result.conductivity,
            )

        # Disconnect previous one-shot connections and reconnect
        try:
            self._recon_ctrl.reconstruction_done.disconnect(self._sim_recon_handler)
        except (RuntimeError, AttributeError):
            pass
        self._sim_recon_handler = _on_sim_recon_done
        self._recon_ctrl.reconstruction_done.connect(self._sim_recon_handler)

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
        self._on_stop_acquisition()
        if self._state.connection_status is ConnectionStatus.CONNECTED:
            try:
                self._device_ctrl.power_off_device()
            except Exception as exc:
                log.warning("Failed to power off device during shutdown: %s", exc)
        self._device_ctrl.shutdown()
        self._recon_ctrl.shutdown()
        self._fwd_ctrl.shutdown()
        self._dataset_ctrl.shutdown()
        try:
            self._db_ctrl.shutdown()
        except Exception as exc:
            log.warning("Database shutdown failed: %s", exc)
        super().closeEvent(event)
