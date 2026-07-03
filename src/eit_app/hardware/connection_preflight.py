"""Fast connection preflight checks for serial and 4G relay transports."""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eit_app.i18n import t

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
    return ConnectionPreflightResult(True, t("preflight.skipped"), "")


def _preflight_serial(config: dict[str, Any]) -> ConnectionPreflightResult:
    port = str(config.get("port", "")).strip()
    if not port:
        return ConnectionPreflightResult(
            False,
            t("preflight.serial.no_port.summary"),
            t("preflight.serial.no_port.hint"),
        )

    baudrate = int(config.get("baudrate", 115200))
    display_name = str(config.get("port_display", "")).strip() or port

    if running_in_wsl() and _WINDOWS_COM_RE.fullmatch(port):
        if not _windows_port_present(port):
            return ConnectionPreflightResult(
                False,
                t("preflight.serial.win_absent.summary", name=display_name),
                t("preflight.serial.win_absent.hint"),
            )
        return ConnectionPreflightResult(
            True,
            t("preflight.serial.win_present.summary", name=display_name),
            t("preflight.serial.win_present.hint", name=display_name),
        )

    if port.startswith("/dev/") and not Path(port).exists():
        return ConnectionPreflightResult(
            False,
            t("preflight.serial.dev_missing.summary", port=port),
            t("preflight.serial.dev_missing.hint"),
        )

    try:
        _probe_transport(SerialTransport(port, baudrate))
    except Exception as exc:
        return ConnectionPreflightResult(
            False,
            t("preflight.serial.cannot_open.summary", name=display_name),
            _serial_direct_failure_hint(port, exc),
        )

    return ConnectionPreflightResult(
        True,
        t("preflight.serial.ok.summary", name=display_name),
        t("preflight.serial.ok.hint", name=display_name),
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
            t("preflight.relay.no_host.summary"),
            t("preflight.relay.no_host.hint"),
        )

    try:
        sock = socket.create_connection((host, port), timeout=timeout_sec)
    except OSError as exc:
        return ConnectionPreflightResult(
            False,
            t("preflight.relay.fail.summary", host=host, port=port),
            _relay_failure_hint(host, port, exc),
        )

    sock.close()
    return ConnectionPreflightResult(
        True,
        t("preflight.relay.ok.summary", host=host, port=port),
        t("preflight.relay.ok.hint", host=host, port=port),
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
        return t("preflight.serial.fail.denied", port=port)
    if "no such file" in text or "cannot open" in text or "could not open" in text:
        return t("preflight.serial.fail.missing", port=port)
    if "input/output error" in text or "could not configure port" in text:
        return t("preflight.serial.fail.unconfigurable", port=port)
    return t("preflight.serial.fail.generic", port=port)


def _relay_failure_hint(host: str, port: int, exc: Exception) -> str:
    text = str(exc).lower()
    if "refused" in text:
        return t("preflight.relay.fail.refused", host=host, port=port)
    if "timed out" in text:
        return t("preflight.relay.fail.timeout", host=host, port=port)
    if "name or service not known" in text or "getaddrinfo failed" in text:
        return t("preflight.relay.fail.dns", host=host)
    return t("preflight.relay.fail.generic", host=host, port=port)
