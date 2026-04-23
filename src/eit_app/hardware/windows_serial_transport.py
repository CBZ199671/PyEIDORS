"""Windows-hosted serial transport bridged into WSL over localhost TCP."""

from __future__ import annotations

import os
import selectors
import socket
import subprocess
import time
from base64 import b64encode

from .base_transport import AbstractTransport

_BRIDGE_BOOT_TIMEOUT_SEC = 8.0
_BRIDGE_OPEN_RETRY_DELAYS_SEC = (0.0, 0.25, 0.55, 1.0)
_BRIDGE_CLOSE_GRACE_SEC = 0.35
_BRIDGE_ATTACH_TIMEOUT_SEC = 4.0

_WINDOWS_BRIDGE_SCRIPT_TEMPLATE = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$portName = '__EIT_PORT_NAME__'
$baudRate = __EIT_BAUD_RATE__
try {
    $serial = [System.IO.Ports.SerialPort]::new(
        $portName,
        $baudRate,
        [System.IO.Ports.Parity]::None,
        8,
        [System.IO.Ports.StopBits]::One
    )
    $serial.ReadTimeout = 50
    $serial.WriteTimeout = 2000
    $serial.Handshake = [System.IO.Ports.Handshake]::None
    $serial.Open()

    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $listenPort = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
    [Console]::WriteLine("READY $listenPort")
    [Console]::Out.Flush()

    $attachDeadline = [DateTime]::UtcNow.AddSeconds(__EIT_ATTACH_TIMEOUT_SEC__)
    while (-not $listener.Pending()) {
        if ([DateTime]::UtcNow -ge $attachDeadline) {
            throw "Bridge client attach timeout"
        }
        Start-Sleep -Milliseconds 20
    }

    $client = $listener.AcceptTcpClient()
    $client.NoDelay = $true
    $stream = $client.GetStream()
    $inBuffer = New-Object byte[] 4096
    $outBuffer = New-Object byte[] 4096

    while ($serial.IsOpen) {
        if ($client.Client.Poll(1000, [System.Net.Sockets.SelectMode]::SelectRead) -and $client.Client.Available -eq 0) {
            break
        }

        while ($stream.DataAvailable) {
            $read = $stream.Read($inBuffer, 0, $inBuffer.Length)
            if ($read -le 0) { throw "Socket closed" }
            $serial.Write($inBuffer, 0, $read)
        }

        $available = $serial.BytesToRead
        if ($available -gt 0) {
            $chunk = [Math]::Min($available, $outBuffer.Length)
            $read = $serial.Read($outBuffer, 0, $chunk)
            if ($read -gt 0) {
                $stream.Write($outBuffer, 0, $read)
                $stream.Flush()
            }
        } else {
            Start-Sleep -Milliseconds 5
        }
    }
} catch {
    [Console]::WriteLine("ERROR $($_.Exception.Message)")
    [Console]::Out.Flush()
} finally {
    if ($stream) { $stream.Dispose() }
    if ($client) { $client.Close() }
    if ($listener) { $listener.Stop() }
    if ($serial) {
        if ($serial.IsOpen) { $serial.Close() }
        $serial.Dispose()
    }
}
"""


class WindowsSerialTransport(AbstractTransport):
    """Bridge a Windows COM port into WSL via a localhost TCP proxy."""

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self._port = port.strip().upper()
        self._baudrate = int(baudrate)
        self._proc: subprocess.Popen[bytes] | None = None
        self._sock: socket.socket | None = None
        self._rx_buffer = bytearray()

    def open(self) -> None:
        last_exc: Exception | None = None
        for attempt, delay in enumerate(_BRIDGE_OPEN_RETRY_DELAYS_SEC, start=1):
            self.close()
            if delay > 0:
                time.sleep(delay)
            try:
                self._open_once()
                return
            except Exception as exc:
                last_exc = exc
                self.close()
                if attempt >= len(
                    _BRIDGE_OPEN_RETRY_DELAYS_SEC
                ) or not self._is_retryable_open_error(exc):
                    raise
        if last_exc is not None:
            raise last_exc

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None:
            if self._proc.poll() is None:
                try:
                    self._proc.wait(timeout=_BRIDGE_CLOSE_GRACE_SEC)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    try:
                        self._proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        pass
            self._proc = None
        self._rx_buffer.clear()

    def write(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("Transport not open")
        self._sock.sendall(data)

    def read(self, size: int, timeout: float = 2.0) -> bytes:
        if self._sock is None:
            raise RuntimeError("Transport not open")
        if size <= 0:
            return b""

        deadline = time.monotonic() + timeout
        while len(self._rx_buffer) < size and time.monotonic() < deadline:
            self._recv_into_buffer(deadline - time.monotonic())

        data = bytes(self._rx_buffer[:size])
        del self._rx_buffer[: len(data)]
        return data

    def read_until(self, terminator: bytes, timeout: float = 2.0) -> bytes:
        if self._sock is None:
            raise RuntimeError("Transport not open")
        deadline = time.monotonic() + timeout
        while True:
            index = self._rx_buffer.find(terminator)
            if index >= 0:
                end = index + len(terminator)
                data = bytes(self._rx_buffer[:end])
                del self._rx_buffer[:end]
                return data

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return b""
            self._recv_into_buffer(remaining)

    @property
    def is_open(self) -> bool:
        return self._sock is not None

    def _recv_into_buffer(self, timeout: float) -> None:
        if self._sock is None:
            raise RuntimeError("Transport not open")
        timeout = max(0.05, timeout)
        self._sock.settimeout(timeout)
        try:
            chunk = self._sock.recv(4096)
        except socket.timeout:
            return
        if not chunk:
            raise RuntimeError("Windows serial bridge closed")
        self._rx_buffer.extend(chunk)

    def _open_once(self) -> None:
        self._proc = self._start_bridge_process()
        try:
            line = self._read_bridge_line(self._proc, timeout=_BRIDGE_BOOT_TIMEOUT_SEC)
        except Exception:
            self.close()
            raise

        if not line.startswith("READY "):
            self.close()
            raise RuntimeError(f"Windows serial bridge failed: {line or 'no response'}")

        try:
            listen_port = int(line.split(" ", 1)[1].strip())
        except (IndexError, ValueError) as exc:
            self.close()
            raise RuntimeError(f"Invalid bridge bootstrap response: {line!r}") from exc

        try:
            self._sock = socket.create_connection(
                ("127.0.0.1", listen_port), timeout=5.0
            )
        except OSError:
            self.close()
            raise
        self._sock.settimeout(2.0)
        self._rx_buffer.clear()

    @staticmethod
    def _is_retryable_open_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "access is denied",
                "access to the port",
                "the port is closed",
                "used by another process",
                "访问被拒绝",
                "拒绝访问",
                "占用",
            )
        )

    def _start_bridge_process(self) -> subprocess.Popen[bytes]:
        escaped_port = self._port.replace("'", "''")
        script = (
            _WINDOWS_BRIDGE_SCRIPT_TEMPLATE.replace("__EIT_PORT_NAME__", escaped_port)
            .replace("__EIT_BAUD_RATE__", str(self._baudrate))
            .replace("__EIT_ATTACH_TIMEOUT_SEC__", f"{_BRIDGE_ATTACH_TIMEOUT_SEC:.1f}")
        )
        encoded = b64encode(script.encode("utf-16-le")).decode("ascii")
        return subprocess.Popen(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @staticmethod
    def _read_bridge_line(proc: subprocess.Popen[bytes], *, timeout: float) -> str:
        if proc.stdout is None:
            return ""

        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        buffer = bytearray()
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                if proc.poll() is not None and not buffer:
                    break
                wait_time = max(0.05, deadline - time.monotonic())
                events = selector.select(wait_time)
                if not events:
                    continue
                chunk = os.read(proc.stdout.fileno(), 4096)
                if not chunk:
                    break
                buffer.extend(chunk)
                if b"\n" in buffer:
                    line = buffer.splitlines()[0]
                    return line.decode("utf-8", errors="replace").strip()
        finally:
            selector.close()
        return buffer.decode("utf-8", errors="replace").strip()
