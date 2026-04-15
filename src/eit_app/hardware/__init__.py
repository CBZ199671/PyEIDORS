"""Public hardware interfaces for the EIT Workstation."""

from .connection_preflight import ConnectionPreflightResult, preflight_connection_target
from .factory import create_device_from_config, normalize_device_config
from .relay_transport import RelayTransport
from .serial_transport import C8051Device, SerialTransport
from .simulator import SimulatorDevice
from .windows_serial_transport import WindowsSerialTransport

__all__ = [
    "C8051Device",
    "ConnectionPreflightResult",
    "RelayTransport",
    "SerialTransport",
    "SimulatorDevice",
    "WindowsSerialTransport",
    "create_device_from_config",
    "normalize_device_config",
    "preflight_connection_target",
]
