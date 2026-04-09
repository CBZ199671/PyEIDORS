"""Legacy 4G relay transport compatible with the old server forwarding protocol."""

from __future__ import annotations

import socket
import struct
import time

from .base_transport import AbstractTransport
from .protocol import (
    build_relay_registration,
    build_relay_transmit,
    parse_relay_response,
    parse_response,
    relay_device_payload_to_frame,
)
from .types import (
    DEFAULT_BOARD_ID,
    DEFAULT_SERVER_PORT,
    DEFAULT_USER_ID,
    FRAME_HEAD,
    RelayStatus,
)


class RelayTransport(AbstractTransport):
    """TCP transport that speaks the old server registration/forwarding protocol."""

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_SERVER_PORT,
        *,
        board_id: int = DEFAULT_BOARD_ID,
        user_id: int = DEFAULT_USER_ID,
    ) -> None:
        self._host = host
        self._port = port
        self._board_id = board_id
        self._user_id = user_id
        self._sock: socket.socket | None = None
        self._rx_buffer = bytearray()
        self._device_buffer = bytearray()

    def open(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        self._sock.connect((self._host, self._port))
        self._rx_buffer.clear()
        self._device_buffer.clear()
        self._sock.sendall(build_relay_registration(self._user_id))
        status = self._wait_for_server_ack(timeout=5.0)
        if status is not RelayStatus.SUCCESS:
            raise RuntimeError(f"Relay registration failed: {status.name}")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None
        self._rx_buffer.clear()
        self._device_buffer.clear()

    def write(self, data: bytes) -> None:
        if self._sock is None:
            raise RuntimeError("Transport not open")

        result = parse_response(data)
        if result is None:
            raise ValueError("RelayTransport.write expects a device frame")

        packet = build_relay_transmit(
            result.cmd,
            result.data,
            board_id=self._board_id,
            user_id=self._user_id,
        )
        self._sock.sendall(packet)
        status = self._wait_for_server_ack(timeout=5.0)
        if status is not RelayStatus.SUCCESS:
            raise RuntimeError(f"Relay command rejected: {status.name}")

    def read(self, size: int, timeout: float = 2.0) -> bytes:
        if self._sock is None:
            raise RuntimeError("Transport not open")
        if size <= 0:
            return b""

        deadline = time.monotonic() + timeout
        while len(self._device_buffer) < size:
            remaining = max(0.05, deadline - time.monotonic())
            if remaining <= 0:
                break
            self._device_buffer.extend(self._recv_device_frame(timeout=remaining))

        data = bytes(self._device_buffer[:size])
        del self._device_buffer[: len(data)]
        return data

    def read_until(self, terminator: bytes, timeout: float = 2.0) -> bytes:
        if self._sock is None:
            raise RuntimeError("Transport not open")
        deadline = time.monotonic() + timeout
        while True:
            index = self._device_buffer.find(terminator)
            if index >= 0:
                end = index + len(terminator)
                data = bytes(self._device_buffer[:end])
                del self._device_buffer[:end]
                return data

            remaining = max(0.05, deadline - time.monotonic())
            if remaining <= 0:
                break
            self._device_buffer.extend(self._recv_device_frame(timeout=remaining))

        return b""

    @property
    def is_open(self) -> bool:
        return self._sock is not None

    def _recv_server_frame(self, timeout: float) -> bytes:
        if self._sock is None:
            raise RuntimeError("Transport not open")
        while True:
            head_idx = self._rx_buffer.find(FRAME_HEAD)
            if head_idx > 0:
                del self._rx_buffer[:head_idx]
            elif head_idx < 0:
                keep = max(0, len(FRAME_HEAD) - 1)
                if len(self._rx_buffer) > keep:
                    del self._rx_buffer[:-keep]

            if len(self._rx_buffer) >= len(FRAME_HEAD) + 2:
                len_field = struct.unpack_from(">H", self._rx_buffer, len(FRAME_HEAD))[0]
                frame_total = len_field + 7
                if len_field < 3:
                    del self._rx_buffer[:1]
                    continue
                if len(self._rx_buffer) >= frame_total:
                    frame = bytes(self._rx_buffer[:frame_total])
                    del self._rx_buffer[:frame_total]
                    return frame

            self._sock.settimeout(timeout)
            chunk = self._sock.recv(4096)
            if not chunk:
                raise RuntimeError("Relay socket closed")
            self._rx_buffer.extend(chunk)

    def _wait_for_server_ack(self, timeout: float) -> RelayStatus:
        frame = self._recv_server_frame(timeout=timeout)
        parsed = parse_relay_response(frame)
        if parsed is None or not parsed.valid_crc or parsed.status is None:
            raise RuntimeError("Expected relay status frame")
        return parsed.status

    def _recv_device_frame(self, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        while True:
            remaining = max(0.05, deadline - time.monotonic())
            if remaining <= 0:
                raise RuntimeError("Timed out while waiting for relay device frame")

            frame = self._recv_server_frame(timeout=remaining)
            parsed = parse_relay_response(frame)
            if parsed is None or not parsed.valid_crc:
                raise RuntimeError("Invalid relay frame received")
            if parsed.status is not None:
                if parsed.status is not RelayStatus.SUCCESS:
                    raise RuntimeError(f"Relay server error: {parsed.status.name}")
                continue
            if parsed.device_cmd is None:
                continue
            return relay_device_payload_to_frame(parsed.device_cmd, parsed.device_payload)
