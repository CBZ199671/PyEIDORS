from __future__ import annotations

import numpy as np

from eit_app.hardware.protocol import (
    build_frame,
    build_relay_registration,
    build_relay_transmit,
    build_start_measurement,
    build_start_measurement_with_mode,
    parse_measurement_frame,
    parse_relay_response,
    parse_response,
    relay_device_payload_to_frame,
)
from eit_app.hardware.types import Command, RelayCommand


def test_build_frame_matches_legacy_length_contract() -> None:
    frame = build_frame(Command.START_MEA, b"\x00\x0a")

    assert frame[:3] == b"\x88\xfb\xfa"
    assert frame[3:5] == b"\x00\x05"
    assert frame[-2:] == b"\xfd\xfc"

    parsed = parse_response(frame)
    assert parsed is not None
    assert parsed.valid_crc is True
    assert parsed.cmd == Command.START_MEA
    assert parsed.data == b"\x00\x0a"


def test_build_start_measurement_reserves_three_d_mode_interface() -> None:
    cmd_2d = build_start_measurement(1000)
    cmd_reserved_3d = build_start_measurement_with_mode(1000, mea_mode=3)

    parsed_2d = parse_response(cmd_2d)
    parsed_3d = parse_response(cmd_reserved_3d)

    assert parsed_2d is not None
    assert parsed_3d is not None
    assert parsed_2d.data == b"\x00\x0a"
    assert parsed_3d.data == b"\x03\x00\x0a"


def test_relay_registration_and_transmit_roundtrip() -> None:
    reg = build_relay_registration(user_id=7)
    reg_parsed = parse_relay_response(reg)

    assert reg_parsed is not None
    assert reg_parsed.valid_crc is True
    assert reg_parsed.cmd == RelayCommand.REGISTER
    assert reg_parsed.data == b"\x02\x07"

    tx = build_relay_transmit(
        Command.START_MEA,
        b"\x00\x0a",
        board_id=10,
        user_id=7,
    )
    tx_parsed = parse_relay_response(tx)

    assert tx_parsed is not None
    assert tx_parsed.valid_crc is True
    assert tx_parsed.cmd == RelayCommand.TRANSMIT
    assert tx_parsed.board_id == 10
    assert tx_parsed.user_id == 7
    assert tx_parsed.device_cmd == Command.START_MEA
    assert tx_parsed.device_payload == b"\x00\x0a"

    device_frame = relay_device_payload_to_frame(
        tx_parsed.device_cmd,
        tx_parsed.device_payload,
    )
    device_parsed = parse_response(device_frame)
    assert device_parsed is not None
    assert device_parsed.valid_crc is True
    assert device_parsed.cmd == Command.START_MEA
    assert device_parsed.data == b"\x00\x0a"


def test_parse_measurement_frame_keeps_2d_layout() -> None:
    from eit_app.models.precision import compute_dtype

    real, imag = parse_measurement_frame(bytes(208 * 4))

    expected_dtype = compute_dtype()
    assert real.shape == (208,)
    assert imag.shape == (208,)
    assert real.dtype == expected_dtype
    assert imag.dtype == expected_dtype
