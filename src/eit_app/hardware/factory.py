"""Hardware device factory helpers shared by GUI and acquisition process."""

from __future__ import annotations

import re
from typing import Any

from eit_app.measurement_layout import measurement_layout_from_config

from .relay_transport import RelayTransport
from .serial_port_discovery import resolve_serial_port_name, running_in_wsl
from .serial_transport import C8051Device, SerialTransport
from .simulator import SimulatorDevice
from .types import (
    DEFAULT_BOARD_ID,
    DEFAULT_MEA_MODE,
    DEFAULT_SERVER_PORT,
    DEFAULT_USER_ID,
    STIM_AMP_VALUES_UA,
)
from .windows_serial_transport import WindowsSerialTransport

_WINDOWS_COM_RE = re.compile(r"COM\d+$", re.IGNORECASE)


def normalize_device_config(
    transport_type: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Fill in missing device configuration with runtime defaults."""
    normalized = dict(config)
    raw_port = str(normalized.get("port", "")).strip()
    normalized.setdefault("transport_type", transport_type)
    normalized.setdefault("server_host", "127.0.0.1")
    normalized.setdefault("baudrate", 115200)
    normalized.setdefault("server_port", DEFAULT_SERVER_PORT)
    normalized.setdefault("board_id", DEFAULT_BOARD_ID)
    normalized.setdefault("user_id", DEFAULT_USER_ID)
    normalized.setdefault("mea_mode", DEFAULT_MEA_MODE)
    # Prefer the legacy-compatible start command by default. ``auto`` tries the
    # 2-byte START_MEA first, which matches the historical Windows upper
    # computer behavior, then falls back to the 3-byte variant for newer
    # firmware.
    normalized.setdefault("start_variant", "auto")
    normalized.setdefault("power_on_settle_sec", 0.8)
    normalized.setdefault("apply_profile_on_start", True)
    normalized.setdefault("frequency_hz", 1000)
    normalized.setdefault("stim_amp_level", 1)
    normalized["stim_amp_uA"] = int(
        STIM_AMP_VALUES_UA.get(
            int(normalized["stim_amp_level"]),
            int(normalized.get("stim_amp_uA", 100)),
        )
    )
    normalized.setdefault("voltage_amp_level_1", 7)
    normalized.setdefault("voltage_amp_level_2", 7)
    normalized.setdefault(
        "contact_impedance_amp_level", normalized["voltage_amp_level_1"]
    )
    normalized.setdefault("protocol_version", "legacy-v1")
    normalized.setdefault("command_retries", 2)
    normalized.setdefault("legacy_frame_timeout_sec", 20.0)
    normalized.setdefault("legacy_frame_retries", 2)
    normalized.setdefault("radius", 1.0)
    normalized.setdefault("geometry_scale_to_m", 1.0)
    normalized.setdefault("contact_impedance", 0.01)
    normalized.update(measurement_layout_from_config(normalized))
    if transport_type == "serial":
        normalized["port"] = resolve_serial_port_name(raw_port)
        if "port_display" not in normalized and raw_port:
            normalized["port_display"] = raw_port
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
            n_electrodes=int(
                normalized.get("n_elec", normalized.get("n_electrodes", 16))
            ),
            n_rings=int(normalized.get("n_rings", 1)),
            stim_pattern=normalized.get("stim_pattern", "{ad}"),
            meas_pattern=normalized.get("meas_pattern", "{ad}"),
            use_meas_current=bool(normalized.get("use_meas_current", False)),
            use_meas_current_next=int(normalized.get("use_meas_current_next", 0)),
            rotate_meas=bool(normalized.get("rotate_meas", True)),
            stim_direction=str(normalized.get("stim_direction", "ccw")),
            meas_direction=str(normalized.get("meas_direction", "ccw")),
            stim_first_positive=bool(normalized.get("stim_first_positive", False)),
        )

    if transport_type == "serial":
        port_name = str(normalized.get("port", "")).strip()
        if running_in_wsl() and _WINDOWS_COM_RE.fullmatch(port_name):
            transport = WindowsSerialTransport(
                port_name,
                int(normalized.get("baudrate", 115200)),
            )
        else:
            transport = SerialTransport(
                port_name,
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
