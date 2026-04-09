"""Hardware device factory helpers shared by GUI and acquisition process."""

from __future__ import annotations

from typing import Any

from .relay_transport import RelayTransport
from .serial_transport import C8051Device, SerialTransport
from .simulator import SimulatorDevice
from .types import (
    DEFAULT_BOARD_ID,
    DEFAULT_MEA_MODE,
    DEFAULT_SERVER_PORT,
    DEFAULT_USER_ID,
    STIM_AMP_VALUES_UA,
)


def normalize_device_config(transport_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Fill in missing device configuration with runtime defaults."""
    normalized = dict(config)
    normalized.setdefault("transport_type", transport_type)
    normalized.setdefault("server_host", "127.0.0.1")
    normalized.setdefault("baudrate", 115200)
    normalized.setdefault("server_port", DEFAULT_SERVER_PORT)
    normalized.setdefault("board_id", DEFAULT_BOARD_ID)
    normalized.setdefault("user_id", DEFAULT_USER_ID)
    normalized.setdefault("mea_mode", DEFAULT_MEA_MODE)
    normalized.setdefault("start_variant", "3byte")
    normalized.setdefault("power_on_settle_sec", 0.8)
    normalized.setdefault("apply_profile_on_start", False)
    normalized.setdefault("frequency_hz", 1000)
    normalized.setdefault("stim_amp_level", 1)
    normalized["stim_amp_uA"] = int(
        STIM_AMP_VALUES_UA.get(
            int(normalized["stim_amp_level"]),
            int(normalized.get("stim_amp_uA", 100)),
        )
    )
    normalized.setdefault("voltage_amp_level_1", 3)
    normalized.setdefault("voltage_amp_level_2", 5)
    normalized.setdefault("contact_impedance_amp_level", normalized["voltage_amp_level_1"])
    normalized.setdefault("protocol_version", "legacy-v1")
    normalized.setdefault("command_retries", 2)
    normalized.setdefault("legacy_frame_timeout_sec", 20.0)
    normalized.setdefault("legacy_frame_retries", 2)
    return normalized


def create_device_from_config(
    transport_type: str,
    config: dict[str, Any],
):
    """Create a hardware device for the given transport type."""
    normalized = normalize_device_config(transport_type, config)

    if transport_type == "simulator":
        return SimulatorDevice(
            fps=float(normalized.get("simulator_fps", 30.0)),
            noise_std=float(normalized.get("simulator_noise_std", 0.002)),
            seed=normalized.get("simulator_seed"),
        )

    if transport_type == "serial":
        transport = SerialTransport(
            normalized.get("port", ""),
            int(normalized.get("baudrate", 115200)),
        )
        return C8051Device(transport=transport, device_config=normalized)

    if transport_type == "relay":
        transport = RelayTransport(
            host=normalized.get("server_host", ""),
            port=int(normalized.get("server_port", DEFAULT_SERVER_PORT)),
            board_id=int(normalized.get("board_id", DEFAULT_BOARD_ID)),
            user_id=int(normalized.get("user_id", DEFAULT_USER_ID)),
        )
        return C8051Device(transport=transport, device_config=normalized)

    raise ValueError(f"Unsupported transport type: {transport_type!r}")
