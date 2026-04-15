"""GUI-facing device controller that executes I/O in a worker thread."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QEventLoop, QObject, Qt, QThread, QTimer, Signal, Slot

from eit_app.hardware.factory import create_device_from_config, normalize_device_config
from eit_app.hardware.types import STIM_AMP_VALUES_UA

log = logging.getLogger(__name__)


class _DeviceWorker(QObject):
    """Runs blocking device operations in a background QThread."""

    connected = Signal()
    disconnected = Signal()
    suspended = Signal()
    error = Signal(str)
    command_done = Signal(str, object)
    impedance_result = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._transport_type = "simulator"
        self._device_config: dict[str, Any] = normalize_device_config("simulator", {})
        self._device = None

    def set_connection_profile(self, transport_type: str, config: dict[str, Any]) -> None:
        self._transport_type = transport_type
        self._device_config = normalize_device_config(transport_type, config)
        self._apply_profile_to_live_device()

    def current_config(self) -> dict[str, Any]:
        return dict(self._device_config)

    @Slot(str, dict)
    def update_connection_profile(self, transport_type: str, config: dict[str, Any]) -> None:
        self.set_connection_profile(transport_type, config)

    @Slot()
    def do_connect(self) -> None:
        try:
            device = self._ensure_connected_device()
            query = getattr(device, "try_query_capabilities", None)
            capabilities = query() if callable(query) else device.capabilities()
            self.command_done.emit("capabilities", capabilities)
            self.connected.emit()
        except Exception as exc:
            self.error.emit(f"Connection failed: {exc}")

    @Slot()
    def do_disconnect(self) -> None:
        self._disconnect_live_device()
        self.disconnected.emit()

    @Slot()
    def do_suspend(self) -> None:
        try:
            self._disconnect_live_device()
            self.suspended.emit()
        except Exception as exc:
            self.error.emit(f"Suspend failed: {exc}")

    @Slot(str, dict)
    def do_command(self, name: str, kwargs: dict[str, Any]) -> None:
        try:
            device = self._ensure_connected_device()
            method = getattr(device, name, None)
            if method is None:
                self.error.emit(f"Unknown command: {name}")
                return
            result = method(**kwargs)
            self.command_done.emit(name, result)
        except Exception as exc:
            self.error.emit(f"Command '{name}' failed: {exc}")

    @Slot()
    def do_measure_impedance(self) -> None:
        try:
            device = self._ensure_connected_device()
            result = device.measure_contact_impedance()
            self.impedance_result.emit(result)
        except Exception as exc:
            self.error.emit(f"Impedance measurement failed: {exc}")

    def _ensure_connected_device(self):
        if self._device is None:
            self._device = create_device_from_config(
                self._transport_type,
                self._device_config,
            )
        if not getattr(self._device, "is_connected", False):
            self._device.connect()
            self._apply_profile_to_live_device()
        return self._device

    def _disconnect_live_device(self) -> None:
        if self._device is None:
            return
        try:
            if getattr(self._device, "is_connected", False):
                self._device.disconnect()
        finally:
            self._device = None

    def _apply_profile_to_live_device(self) -> None:
        if self._device is None:
            return
        config = getattr(self._device, "_config", None)
        if isinstance(config, dict):
            config.update(self._device_config)


class DeviceController(QObject):
    """Background-thread controller for short-lived device transactions."""

    request_profile_update = Signal(str, dict)
    request_connect = Signal()
    request_disconnect = Signal()
    request_suspend = Signal()
    request_command = Signal(str, dict)
    request_impedance = Signal()

    connected = Signal()
    disconnected = Signal()
    suspended = Signal()
    error = Signal(str)
    command_done = Signal(str, object)
    impedance_result = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread = QThread()
        self._worker = _DeviceWorker()
        self._worker.moveToThread(self._thread)
        self._transport_type = "simulator"
        self._config: dict[str, Any] = normalize_device_config("simulator", {})

        self.request_profile_update.connect(
            self._worker.update_connection_profile,
            Qt.ConnectionType.QueuedConnection,
        )
        self.request_connect.connect(self._worker.do_connect, Qt.ConnectionType.QueuedConnection)
        self.request_disconnect.connect(
            self._worker.do_disconnect,
            Qt.ConnectionType.QueuedConnection,
        )
        self.request_suspend.connect(
            self._worker.do_suspend,
            Qt.ConnectionType.QueuedConnection,
        )
        self.request_command.connect(
            self._worker.do_command,
            Qt.ConnectionType.QueuedConnection,
        )
        self.request_impedance.connect(
            self._worker.do_measure_impedance,
            Qt.ConnectionType.QueuedConnection,
        )

        self._worker.connected.connect(self.connected)
        self._worker.disconnected.connect(self.disconnected)
        self._worker.suspended.connect(self.suspended)
        self._worker.error.connect(self.error)
        self._worker.command_done.connect(self.command_done)
        self._worker.impedance_result.connect(self.impedance_result)

        self._thread.start()
        self.request_profile_update.emit(self._transport_type, dict(self._config))

    def set_connection_profile(self, transport_type: str, config: dict[str, Any]) -> None:
        self._transport_type = transport_type
        self._config = normalize_device_config(transport_type, config)
        self.request_profile_update.emit(self._transport_type, dict(self._config))

    def current_config(self) -> dict[str, Any]:
        return dict(self._config)

    def connect_device(self) -> None:
        self.request_connect.emit()

    def disconnect_device(self) -> None:
        self.request_disconnect.emit()

    def suspend_session(self, timeout_ms: int = 3000) -> bool:
        if not self._thread.isRunning():
            return False

        loop = QEventLoop()
        completed = {"ok": False}

        def _on_suspended() -> None:
            completed["ok"] = True
            if loop.isRunning():
                loop.quit()

        def _on_error(_msg: str) -> None:
            if loop.isRunning():
                loop.quit()

        self.suspended.connect(_on_suspended)
        self.error.connect(_on_error)
        try:
            self.request_suspend.emit()
            QTimer.singleShot(timeout_ms, loop.quit)
            loop.exec()
        finally:
            self.suspended.disconnect(_on_suspended)
            self.error.disconnect(_on_error)
        return completed["ok"]

    def send_command(self, name: str, **kwargs: Any) -> None:
        config = dict(self._config)
        if name == "set_frequency" and "hz" in kwargs:
            config["frequency_hz"] = int(kwargs["hz"])
        elif name == "set_stim_amplitude" and "level" in kwargs:
            level = int(kwargs["level"])
            config["stim_amp_level"] = level
            config["stim_amp_uA"] = STIM_AMP_VALUES_UA.get(level, level)
        elif name == "set_voltage_amp_levels":
            config["voltage_amp_level_1"] = int(kwargs["level_1"])
            config["voltage_amp_level_2"] = int(kwargs["level_2"])
        elif name == "set_voltage_amp" and "level" in kwargs:
            config["voltage_amp_level_1"] = int(kwargs["level"])
            config["voltage_amp_level_2"] = int(kwargs["level"])
        self._transport_type = str(config.get("transport_type", self._transport_type))
        self._config = normalize_device_config(self._transport_type, config)
        self.request_profile_update.emit(self._transport_type, dict(self._config))
        self.request_command.emit(name, kwargs)

    def set_frequency(self, hz: int) -> None:
        self.send_command("set_frequency", hz=hz)

    def set_stim_amplitude(self, level: int) -> None:
        self.send_command("set_stim_amplitude", level=level)

    def set_voltage_amp_levels(self, level_1: int, level_2: int) -> None:
        self.send_command("set_voltage_amp_levels", level_1=level_1, level_2=level_2)

    def set_voltage_amp(self, level: int) -> None:
        self.send_command("set_voltage_amp", level=level)

    def power_control(self, on: bool) -> None:
        self.send_command("power_control", on=on)

    def single_point_test(self, hz: int) -> None:
        self.send_command("single_point_test_at", hz=hz)

    def run_sweep(self, start_hz: int, end_hz: int, n_points: int) -> None:
        self.send_command(
            "run_sweep",
            start_hz=start_hz,
            end_hz=end_hz,
            n_points=n_points,
        )

    def measure_impedance(self) -> None:
        self.request_impedance.emit()

    def power_off_device(self, timeout_ms: int = 3000) -> bool:
        if not self._thread.isRunning():
            return False

        loop = QEventLoop()
        completed = {"ok": False}

        def _on_command_done(name: str, _result: object) -> None:
            if name != "power_control":
                return
            completed["ok"] = True
            if loop.isRunning():
                loop.quit()

        def _on_error(msg: str) -> None:
            if "power_control" not in msg:
                return
            if loop.isRunning():
                loop.quit()

        self.command_done.connect(_on_command_done)
        self.error.connect(_on_error)
        try:
            self.power_control(False)
            QTimer.singleShot(timeout_ms, loop.quit)
            loop.exec()
        finally:
            self.command_done.disconnect(_on_command_done)
            self.error.disconnect(_on_error)
        return completed["ok"]

    def shutdown(self) -> None:
        if self._thread.isRunning():
            try:
                self.suspend_session(timeout_ms=1500)
            except Exception:
                pass
        self._thread.quit()
        self._thread.wait(3000)
