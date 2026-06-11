"""Fast connection preflight checks for serial and 4G relay transports."""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .serial_port_discovery import discover_serial_ports, running_in_wsl
from .serial_transport import SerialTransport

_WINDOWS_COM_RE = re.compile(r"COM\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class ConnectionPreflightResult:
    """User-facing preflight result for a connection attempt."""

    ok: bool
    summary: str
    hint: str


def preflight_connection_target(
    transport_type: str,
    config: dict[str, Any],
    *,
    timeout_sec: float = 2.5,
) -> ConnectionPreflightResult:
    """Perform a quick availability check before the real protocol handshake."""
    if transport_type == "serial":
        return _preflight_serial(config)
    if transport_type == "relay":
        return _preflight_relay(config, timeout_sec=timeout_sec)
    return ConnectionPreflightResult(True, "已跳过连接前预检。", "")


def _preflight_serial(config: dict[str, Any]) -> ConnectionPreflightResult:
    port = str(config.get("port", "")).strip()
    if not port:
        return ConnectionPreflightResult(
            False,
            "未选择串口。",
            "请先点击 Scan，或从下拉框选择自动检测到的串口。",
        )

    baudrate = int(config.get("baudrate", 115200))
    display_name = str(config.get("port_display", "")).strip() or port

    if running_in_wsl() and _WINDOWS_COM_RE.fullmatch(port):
        if not _windows_port_present(port):
            return ConnectionPreflightResult(
                False,
                f"Windows 当前未检测到串口 {display_name}。",
                "请确认设备仍然插着，必要时重新插拔 USB 后点击 Scan；如果你刚关闭软件，请等待 1-2 秒再重试。",
            )
        return ConnectionPreflightResult(
            True,
            f"已检测到 Windows 串口 {display_name}。",
            (
                f"已检测到 {display_name}，连接时会自动通过 Windows 主机串口桥接打开；"
                "如果刚关闭软件，程序也会自动做短暂重试。"
            ),
        )

    if port.startswith("/dev/") and not Path(port).exists():
        return ConnectionPreflightResult(
            False,
            f"未找到串口设备 {port}。",
            "请检查 USB 线和设备供电；如果软件运行在 WSL 中，优先从下拉框选择自动检测到的 Windows COM 口。",
        )

    try:
        _probe_transport(SerialTransport(port, baudrate))
    except Exception as exc:
        return ConnectionPreflightResult(
            False,
            f"串口 {display_name} 当前无法打开。",
            _serial_direct_failure_hint(port, exc),
        )

    return ConnectionPreflightResult(
        True,
        f"已确认串口 {display_name} 可打开。",
        f"已检测到 {display_name}，串口预检通过，正在继续验证设备协议。",
    )


def _preflight_relay(
    config: dict[str, Any],
    *,
    timeout_sec: float,
) -> ConnectionPreflightResult:
    host = str(config.get("server_host", "")).strip()
    port = int(config.get("server_port", 4555))

    if not host:
        return ConnectionPreflightResult(
            False,
            "4G Relay 服务器地址为空。",
            "请先填写可访问的服务器 host，再发起连接。",
        )

    try:
        sock = socket.create_connection((host, port), timeout=timeout_sec)
    except OSError as exc:
        return ConnectionPreflightResult(
            False,
            f"无法连接到 4G Relay 服务器 {host}:{port}。",
            _relay_failure_hint(host, port, exc),
        )

    sock.close()
    return ConnectionPreflightResult(
        True,
        f"已连通 4G Relay 服务器 {host}:{port}。",
        f"服务器 {host}:{port} 可达，接下来会继续验证设备握手和链路能力。",
    )


def _probe_transport(transport) -> None:
    opened = False
    try:
        transport.open()
        opened = True
    finally:
        if opened:
            transport.close()


def _windows_port_present(port: str) -> bool:
    target = port.strip().upper()
    for descriptor in discover_serial_ports():
        if str(descriptor.device).strip().upper() == target:
            return True
    return False


def _serial_direct_failure_hint(port: str, exc: Exception) -> str:
    text = str(exc).lower()
    if "permission denied" in text or "access is denied" in text:
        return f"{port} 访问被拒绝，可能已被其他程序占用。请关闭占用进程后重试。"
    if "no such file" in text or "cannot open" in text or "could not open" in text:
        return f"{port} 当前不存在或未就绪。请检查 USB 连接并重新 Scan。"
    if "input/output error" in text or "could not configure port" in text:
        return (
            f"{port} 当前不可配置。若你在 WSL 中运行，请不要手动填写 /dev/ttyS*，"
            "优先选择自动检测到的 Windows COM 口。"
        )
    return f"{port} 当前无法打开。请确认串口号、驱动和波特率设置无误。"


def _relay_failure_hint(host: str, port: int, exc: Exception) -> str:
    text = str(exc).lower()
    if "refused" in text:
        return f"{host}:{port} 拒绝连接。请确认 relay 服务已启动，并检查 host/port 是否填写正确。"
    if "timed out" in text:
        return f"连接 {host}:{port} 超时。请检查网络、服务器地址和目标设备是否在线。"
    if "name or service not known" in text or "getaddrinfo failed" in text:
        return f"无法解析服务器地址 {host}。请检查 host 拼写或 DNS 配置。"
    return f"{host}:{port} 当前不可达。请检查网络、服务状态和防火墙设置。"
