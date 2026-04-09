"""Public hardware interfaces for the EIT Workstation."""

from .factory import create_device_from_config, normalize_device_config
from .relay_transport import RelayTransport
from .serial_transport import C8051Device, SerialTransport
from .simulator import SimulatorDevice

__all__ = [
    "C8051Device",
    "RelayTransport",
    "SerialTransport",
    "SimulatorDevice",
    "create_device_from_config",
    "normalize_device_config",
]
