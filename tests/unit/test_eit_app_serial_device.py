from __future__ import annotations

from collections import deque

from eit_app.measurement_layout import measurement_layout_from_config
from eit_app.hardware.factory import normalize_device_config
from eit_app.hardware.protocol import (
    build_frame,
    build_start_measurement,
    build_start_measurement_with_mode,
)
from eit_app.hardware.serial_transport import C8051Device
from eit_app.hardware.types import Command


class _FakeTransport:
    def __init__(self, responses: list[bytes]) -> None:
        self.responses = deque(bytearray(resp) for resp in responses)
        self.writes: list[bytes] = []
        self.is_open = False

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def read(self, size: int, timeout: float = 2.0) -> bytes:
        if not self.responses:
            return b""
        current = self.responses[0]
        if not current:
            self.responses.popleft()
            return b""
        chunk = bytes(current[:size])
        del current[:size]
        if not current:
            self.responses.popleft()
        return chunk

    def read_until(self, terminator: bytes, timeout: float = 2.0) -> bytes:
        if not self.responses:
            raise RuntimeError("no response queued")
        current = bytes(self.responses.popleft())
        return current


def test_factory_defaults_to_auto_start_variant_for_legacy_serial_devices() -> None:
    normalized = normalize_device_config("serial", {"port": "COM4"})

    assert normalized["start_variant"] == "auto"


def test_c8051_device_auto_fallback_supports_reserved_3d_start_format() -> None:
    transport = _FakeTransport(
        responses=[
            build_frame(Command.SINGLE_POINT_TEST, b"\x00\x00\x00\x00"),
            build_frame(Command.START_MEA, bytes(208 * 4)),
        ]
    )
    device = C8051Device(
        transport=transport,
        device_config={
            "frequency_hz": 1000,
            "mea_mode": 3,
            "start_variant": "auto",
            "stim_amp_level": 1,
            "voltage_amp_level_1": 0,
            "voltage_amp_level_2": 0,
        },
    )

    device.connect()
    device.start_measurement()
    frame = device.read_frame()
    device.disconnect()

    assert frame.real.shape == (208,)
    assert frame.imag.shape == (208,)
    assert frame.metadata["mea_mode"] == 3
    assert transport.writes[-2] == build_start_measurement(1000)
    assert transport.writes[-1] == build_start_measurement_with_mode(1000, mea_mode=3)


def test_c8051_device_reads_full_frame_even_if_payload_contains_footer_bytes() -> None:
    payload = bytearray(208 * 4)
    for index in range(len(payload)):
        payload[index] = index % 251
    payload[120] = 0xFD
    payload[121] = 0xFC

    transport = _FakeTransport(
        responses=[build_frame(Command.START_MEA, bytes(payload))]
    )
    device = C8051Device(
        transport=transport,
        device_config={
            "frequency_hz": 1000,
            "mea_mode": 2,
            "start_variant": "3byte",
            "stim_amp_level": 1,
            "voltage_amp_level_1": 0,
            "voltage_amp_level_2": 0,
        },
    )

    device.connect()
    device.start_measurement()
    frame = device.read_frame()
    device.disconnect()

    assert frame.real.shape == (208,)
    assert frame.imag.shape == (208,)
    assert transport.writes[-1] == build_start_measurement_with_mode(1000, mea_mode=2)


def test_c8051_device_accepts_reserved_non_208_frame_spec() -> None:
    points = int(measurement_layout_from_config({"n_elec": 32})["points_per_frame"])
    transport = _FakeTransport(
        responses=[build_frame(Command.START_MEA, bytes(points * 4))]
    )
    device = C8051Device(
        transport=transport,
        device_config={
            "frequency_hz": 1000,
            "mea_mode": 2,
            "start_variant": "3byte",
            "stim_amp_level": 1,
            "voltage_amp_level_1": 0,
            "voltage_amp_level_2": 0,
            "power_on_settle_sec": 0.0,
            "n_elec": 32,
            "points_per_frame": points,
        },
    )

    device.connect()
    device.start_measurement()
    frame = device.read_frame()
    device.disconnect()

    assert frame.real.shape == (points,)
    assert frame.imag.shape == (points,)
    assert frame.metadata["points_per_frame"] == points
    assert frame.metadata["n_elec"] == 32
