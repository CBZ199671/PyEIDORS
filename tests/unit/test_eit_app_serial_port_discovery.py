from __future__ import annotations

import eit_app.hardware.factory as factory_module
import eit_app.hardware.serial_port_discovery as discovery
from eit_app.hardware.factory import create_device_from_config, normalize_device_config
from eit_app.hardware.serial_port_discovery import (
    SerialPortDescriptor,
    discover_serial_ports,
    resolve_serial_port_name,
)
from eit_app.hardware.windows_serial_transport import WindowsSerialTransport


def test_resolve_serial_port_name_preserves_manual_windows_com_input() -> None:
    assert resolve_serial_port_name("COM7") == "COM7"
    assert resolve_serial_port_name(" com7 ") == "com7"


def test_discover_serial_ports_uses_windows_fallback_when_wsl_pyserial_is_empty(
    monkeypatch,
) -> None:
    monkeypatch.setattr(discovery, "_discover_pyserial_ports", lambda: [])
    monkeypatch.setattr(
        discovery,
        "_discover_windows_serial_ports",
        lambda: [
            SerialPortDescriptor(
                device="COM7",
                display_name="COM7 - USB-SERIAL CH340",
                source="windows-com",
            )
        ],
    )

    ports = discover_serial_ports()

    assert len(ports) == 1
    assert ports[0].device == "COM7"
    assert ports[0].display_name == "COM7 - USB-SERIAL CH340"


def test_normalize_device_config_preserves_manual_com_input_and_display() -> None:
    config = normalize_device_config(
        "serial",
        {
            "port": "COM4",
            "baudrate": 115200,
            "port_display": "COM4 - USB-SERIAL CH340",
        },
    )

    assert config["port"] == "COM4"
    assert config["port_display"] == "COM4 - USB-SERIAL CH340"


def test_create_device_uses_windows_serial_bridge_for_com_ports_in_wsl(
    monkeypatch,
) -> None:
    monkeypatch.setattr(factory_module, "running_in_wsl", lambda: True)

    device = create_device_from_config(
        "serial",
        {"port": "COM4", "baudrate": 115200},
    )

    assert isinstance(device._transport, WindowsSerialTransport)
