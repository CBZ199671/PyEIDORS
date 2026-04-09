from __future__ import annotations

from collections import deque

from eit_app.hardware.protocol import build_frame
from eit_app.hardware.relay_transport import RelayTransport
from eit_app.hardware.types import Command, FRAME_END, RelayCommand


def test_relay_transport_recv_device_frame_skips_server_ack() -> None:
    transport = RelayTransport("127.0.0.1", 4555, board_id=10, user_id=1)
    transport._sock = object()  # type: ignore[assignment]

    frames = deque(
        [
            build_frame(0x00, b"\x00"),
            build_frame(
                RelayCommand.TRANSMIT,
                bytes([10, 1, Command.SINGLE_POINT_TEST, 0x00, 0x01, 0x00, 0x02]),
            ),
        ]
    )

    transport._recv_server_frame = lambda timeout: frames.popleft()  # type: ignore[method-assign]

    device_frame = transport._recv_device_frame(timeout=1.0)

    assert device_frame == build_frame(Command.SINGLE_POINT_TEST, b"\x00\x01\x00\x02")


def test_relay_transport_read_exposes_device_frame_bytes() -> None:
    transport = RelayTransport("127.0.0.1", 4555, board_id=10, user_id=1)
    transport._sock = object()  # type: ignore[assignment]

    device_frame = build_frame(Command.SINGLE_POINT_TEST, b"\x00\x01\x00\x02")
    frames = deque([device_frame])
    transport._recv_device_frame = lambda timeout: frames.popleft()  # type: ignore[method-assign]

    first = transport.read(1, timeout=1.0)
    rest = transport.read(len(device_frame) - 1, timeout=1.0)

    assert first + rest == device_frame


def test_relay_transport_read_until_consumes_device_frame_buffer() -> None:
    transport = RelayTransport("127.0.0.1", 4555, board_id=10, user_id=1)
    transport._sock = object()  # type: ignore[assignment]

    device_frame = build_frame(Command.START_MEA, bytes(range(16)))
    frames = deque([device_frame])
    transport._recv_device_frame = lambda timeout: frames.popleft()  # type: ignore[method-assign]

    result = transport.read_until(FRAME_END, timeout=1.0)

    assert result == device_frame
    assert transport._device_buffer == bytearray()
