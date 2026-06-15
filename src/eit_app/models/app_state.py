"""Observable application state for the EIT Workstation.

Uses PySide6 signals so that UI widgets can react to state changes
without tight coupling. All state mutations go through setter methods
that emit the corresponding signal.
"""

import math
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QObject, Signal


class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class AcquisitionMode(Enum):
    IDLE = "idle"
    CONTINUOUS = "continuous"
    SCHEDULED = "scheduled"
    FINITE_RUN = "finite_run"
    STEPPED_RUN = "stepped_run"
    SINGLE_SHOT = "single_shot"


class PowerStatus(Enum):
    UNKNOWN = "unknown"
    OFF = "off"
    ON = "on"


class RecordingStatus(Enum):
    OFF = "off"
    ARMED = "armed"
    RECORDING = "recording"


@dataclass
class DeviceConfig:
    """Persisted device connection settings."""

    port: str = ""
    transport_type: str = "serial"  # "serial", "relay", "simulator"
    baudrate: int = 115200
    server_host: str = "127.0.0.1"
    server_port: int = 4555
    board_id: int = 1
    user_id: int = 1
    mea_mode: int = 2  # 2D = 2, 3D = 3 (reserved interface)
    frequency_hz: int = 1000
    stim_amp_level: int = 1
    stim_amp_uA: int = 100
    voltage_amp_level_1: int = 3
    voltage_amp_level_2: int = 5
    contact_impedance_amp_level: int = 3
    protocol_version: str = "legacy-v1"
    n_elec: int = 16
    n_rings: int = 1
    stim_pattern: str = "{ad}"
    meas_pattern: str = "{ad}"
    rotate_meas: bool = True
    use_meas_current: bool = False
    use_meas_current_next: int = 0
    stim_direction: str = "ccw"
    meas_direction: str = "ccw"
    stim_first_positive: bool = False
    radius: float = 1.0
    geometry_scale_to_m: float = 1.0
    electrode_coverage: float = 0.5
    electrode_length_m_override: float = math.pi / 16.0
    contact_impedance: float = 0.01
    points_per_frame: int = 208


@dataclass
class RecordingConfig:
    """Recording session settings."""

    output_dir: str = ""
    session_name: str = ""
    auto_record: bool = False


@dataclass
class ReconstructionConfig:
    """Reconstruction parameters exposed to the user."""

    method: str = "gn-difference"
    regularization_alpha: float = 1.0
    max_iterations: int = 10
    mesh_dimension: int = 2
    mesh_refinement: int = 4
    use_part: str = "real"  # "real", "imag", "mag"


class AppState(QObject):
    """Central observable state for the EIT Workstation application.

    Widgets connect to signals to stay in sync. Controllers call setters
    to push state changes.
    """

    # Signals
    connection_status_changed = Signal(str)
    acquisition_mode_changed = Signal(str)
    power_status_changed = Signal(str)
    frame_count_changed = Signal(int)
    recording_active_changed = Signal(bool)
    recording_status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._connection_status = ConnectionStatus.DISCONNECTED
        self._acquisition_mode = AcquisitionMode.IDLE
        self._power_status = PowerStatus.UNKNOWN
        self._frame_count: int = 0
        self._recording_active: bool = False
        self._recording_status = RecordingStatus.OFF
        self.device_config = DeviceConfig()
        self.recording_config = RecordingConfig()
        self.reconstruction_config = ReconstructionConfig()

    # -- Properties with signal emission --

    @property
    def connection_status(self) -> ConnectionStatus:
        return self._connection_status

    def set_connection_status(self, status: ConnectionStatus) -> None:
        if self._connection_status != status:
            self._connection_status = status
            self.connection_status_changed.emit(status.value)

    @property
    def acquisition_mode(self) -> AcquisitionMode:
        return self._acquisition_mode

    def set_acquisition_mode(self, mode: AcquisitionMode) -> None:
        if self._acquisition_mode != mode:
            self._acquisition_mode = mode
            self.acquisition_mode_changed.emit(mode.value)

    @property
    def power_status(self) -> PowerStatus:
        return self._power_status

    def set_power_status(self, status: PowerStatus) -> None:
        if self._power_status != status:
            self._power_status = status
            self.power_status_changed.emit(status.value)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def set_frame_count(self, count: int) -> None:
        self._frame_count = count
        self.frame_count_changed.emit(count)

    @property
    def recording_active(self) -> bool:
        return self._recording_active

    def set_recording_active(self, active: bool) -> None:
        if self._recording_active != active:
            self._recording_active = active
            self.recording_active_changed.emit(active)

    @property
    def recording_status(self) -> RecordingStatus:
        return self._recording_status

    def set_recording_status(self, status: RecordingStatus) -> None:
        if self._recording_status != status:
            self._recording_status = status
            self.recording_status_changed.emit(status.value)

    def report_error(self, message: str) -> None:
        self.error_occurred.emit(message)
