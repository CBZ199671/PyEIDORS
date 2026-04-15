from __future__ import annotations

import socket

import eit_app.hardware.connection_preflight as preflight
from eit_app.hardware.serial_port_discovery import SerialPortDescriptor
from eit_app.hardware.connection_preflight import preflight_connection_target


def test_preflight_serial_wsl_windows_bridge_success(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "running_in_wsl", lambda: True)
    monkeypatch.setattr(
        preflight,
        "discover_serial_ports",
        lambda: [
            SerialPortDescriptor(
                device="COM4",
                display_name="COM4 - USB-SERIAL CH340",
                source="windows-com",
            )
        ],
    )

    result = preflight_connection_target(
        "serial",
        {"port": "COM4", "port_display": "COM4 - USB-SERIAL CH340", "baudrate": 115200},
    )

    assert result.ok is True
    assert "连接时会自动通过 Windows 主机串口桥接打开" in result.hint


def test_preflight_serial_reports_missing_windows_com_port(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "running_in_wsl", lambda: True)
    monkeypatch.setattr(preflight, "discover_serial_ports", lambda: [])

    result = preflight_connection_target(
        "serial",
        {"port": "COM4", "port_display": "COM4 - USB-SERIAL CH340", "baudrate": 115200},
    )

    assert result.ok is False
    assert "Windows 当前未检测到串口 COM4 - USB-SERIAL CH340" in result.summary
    assert "等待 1-2 秒再重试" in result.hint


def test_preflight_serial_reports_missing_direct_device() -> None:
    result = preflight_connection_target(
        "serial",
        {"port": "/dev/ttyDOES_NOT_EXIST", "baudrate": 115200},
    )

    assert result.ok is False
    assert "未找到串口设备 /dev/ttyDOES_NOT_EXIST" in result.summary


def test_preflight_relay_success(monkeypatch) -> None:
    events: list[str] = []

    class _FakeSocket:
        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda address, timeout: _FakeSocket(),
    )

    result = preflight_connection_target(
        "relay",
        {"server_host": "127.0.0.1", "server_port": 4555},
    )

    assert result.ok is True
    assert "服务器 127.0.0.1:4555 可达" in result.hint
    assert events == ["close"]


def test_preflight_relay_failure_mentions_service_state(monkeypatch) -> None:
    def _raise_refused(address, timeout):
        raise ConnectionRefusedError("Connection refused")

    monkeypatch.setattr(socket, "create_connection", _raise_refused)

    result = preflight_connection_target(
        "relay",
        {"server_host": "relay.example", "server_port": 4555},
    )

    assert result.ok is False
    assert "无法连接到 4G Relay 服务器 relay.example:4555" in result.summary
    assert "拒绝连接" in result.hint
