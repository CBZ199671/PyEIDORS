"""Host-aware serial port discovery helpers."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SerialPortDescriptor:
    """A serial port option suitable for UI display and connection."""

    device: str
    display_name: str
    source: str


def running_in_wsl() -> bool:
    """Return True when the current Python process is running inside WSL."""
    if os.getenv("WSL_DISTRO_NAME"):
        return True
    try:
        return (
            "microsoft"
            in Path("/proc/sys/kernel/osrelease").read_text(encoding="utf-8").lower()
        )
    except OSError:
        return False


def resolve_serial_port_name(port_name: str) -> str:
    """Normalize a user-entered port name without forcing host-side remapping."""
    return port_name.strip()


def discover_serial_ports() -> list[SerialPortDescriptor]:
    """Return serial ports discoverable from the current runtime context."""
    by_device: dict[str, SerialPortDescriptor] = {}

    for descriptor in _discover_pyserial_ports():
        by_device[descriptor.device] = descriptor

    for descriptor in _discover_windows_serial_ports():
        existing = by_device.get(descriptor.device)
        if existing is None or existing.source == "pyserial":
            by_device[descriptor.device] = descriptor

    return list(by_device.values())


def _discover_pyserial_ports() -> list[SerialPortDescriptor]:
    try:
        from serial.tools.list_ports import comports
    except ImportError:
        return []

    descriptors: list[SerialPortDescriptor] = []
    for info in comports():
        device = str(getattr(info, "device", "")).strip()
        if not device:
            continue
        description = str(getattr(info, "description", "")).strip()
        if description and description != device:
            display_name = f"{device} - {description}"
        else:
            display_name = device
        descriptors.append(
            SerialPortDescriptor(
                device=device,
                display_name=display_name,
                source="pyserial",
            )
        )
    return descriptors


def _discover_windows_serial_ports() -> list[SerialPortDescriptor]:
    if not running_in_wsl():
        return []

    payload = _query_windows_serial_ports()
    if not payload:
        return []

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        records = [parsed]
    elif isinstance(parsed, list):
        records = [item for item in parsed if isinstance(item, dict)]
    else:
        return []

    descriptors: list[SerialPortDescriptor] = []
    for record in records:
        port_name = str(record.get("port", "")).strip().upper()
        if not port_name:
            continue

        name = str(record.get("name", "")).strip()
        compact_name = _compact_windows_name(name, port_name)
        display_name = port_name
        if compact_name:
            display_name = f"{display_name} - {compact_name}"

        descriptors.append(
            SerialPortDescriptor(
                device=port_name,
                display_name=display_name,
                source="windows-com",
            )
        )
    return descriptors


def _query_windows_serial_ports() -> str:
    script = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$items = @(
    Get-CimInstance Win32_PnPEntity |
        Where-Object { $_.Name -match '\(COM\d+\)' } |
        ForEach-Object {
            if ($_.Name -match '\((COM\d+)\)') {
                [PSCustomObject]@{
                    port = $matches[1]
                    name = $_.Name
                    device_id = $_.DeviceID
                }
            }
        }
)
$items | ConvertTo-Json -Compress
"""

    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=6,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _compact_windows_name(name: str, port_name: str) -> str:
    compact = name.strip()
    if not compact:
        return ""
    suffix = f"({port_name})"
    compact = compact.replace(suffix, "").strip()
    compact = compact.rstrip("- ").strip()
    return compact
