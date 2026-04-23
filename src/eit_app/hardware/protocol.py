"""Legacy EIT device and 4G relay communication helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from .types import (
    FRAME_END,
    FRAME_HEAD,
    ADCParams,
    Command,
    DEFAULT_ADC_PARAMS,
    DEFAULT_BOARD_ID,
    DEFAULT_FRAME_SPEC,
    DEFAULT_MEA_MODE,
    DEFAULT_USER_ID,
    FrameSpec,
    RelayCommand,
    RelayStatus,
    VOLTAGE_AMP_FACTORS,
)


def crc16_modbus(data: bytes) -> int:
    """Compute CRC-16/MODBUS over *data*."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def _build_raw_frame(cmd: int, data: bytes = b"") -> bytes:
    len_bytes = struct.pack(">H", len(data) + 3)
    crc_region = len_bytes + bytes([cmd & 0xFF]) + data
    crc = crc16_modbus(crc_region)
    crc_bytes = struct.pack("<H", crc)
    return FRAME_HEAD + crc_region + crc_bytes + FRAME_END


def build_frame(cmd: int, data: bytes = b"") -> bytes:
    """Build a device frame.

    Layout: ``HEAD + LEN(2B BE) + CMD(1B) + DATA + CRC16(2B LE) + END``.
    """
    return _build_raw_frame(cmd, data)


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing a device frame."""

    cmd: int
    data: bytes
    valid_crc: bool


@dataclass(frozen=True)
class RelayParseResult:
    """Result of parsing a legacy 4G relay frame."""

    cmd: int
    data: bytes
    valid_crc: bool
    status: RelayStatus | None = None
    board_id: int | None = None
    user_id: int | None = None
    device_cmd: int | None = None
    device_payload: bytes = b""


def _parse_payload(buf: bytes) -> tuple[int, bytes, bool] | None:
    head_len = len(FRAME_HEAD)
    end_len = len(FRAME_END)
    head_idx = buf.find(FRAME_HEAD)
    if head_idx < 0:
        return None

    remaining = buf[head_idx:]
    if len(remaining) < head_len + 2 + 1 + 2 + end_len:
        return None

    len_field = struct.unpack_from(">H", remaining, head_len)[0]
    frame_total = len_field + 7
    if len_field < 3 or len(remaining) < frame_total:
        return None
    if remaining[frame_total - end_len : frame_total] != FRAME_END:
        return None

    cmd_offset = head_len + 2
    cmd = remaining[cmd_offset]
    data_len = len_field - 3
    data_start = cmd_offset + 1
    data_end = data_start + data_len
    data = bytes(remaining[data_start:data_end])
    crc_region_end = head_len + len_field
    expected_crc = crc16_modbus(remaining[head_len:crc_region_end])
    actual_crc = struct.unpack_from("<H", remaining, crc_region_end)[0]
    return cmd, data, expected_crc == actual_crc


def parse_response(buf: bytes) -> ParseResult | None:
    """Parse a device frame from *buf*."""
    parsed = _parse_payload(buf)
    if parsed is None:
        return None
    cmd, data, valid_crc = parsed
    return ParseResult(cmd=cmd, data=data, valid_crc=valid_crc)


def parse_relay_response(buf: bytes) -> RelayParseResult | None:
    """Parse a legacy 4G relay frame from *buf*."""
    parsed = _parse_payload(buf)
    if parsed is None:
        return None
    cmd, data, valid_crc = parsed

    if cmd == 0x00:
        status = RelayStatus(data[0]) if data else RelayStatus.DEFEATED
        return RelayParseResult(
            cmd=cmd,
            data=data,
            valid_crc=valid_crc,
            status=status,
        )

    if cmd == RelayCommand.TRANSMIT and len(data) >= 3:
        return RelayParseResult(
            cmd=cmd,
            data=data,
            valid_crc=valid_crc,
            board_id=data[0],
            user_id=data[1],
            device_cmd=data[2],
            device_payload=bytes(data[3:]),
        )

    return RelayParseResult(cmd=cmd, data=data, valid_crc=valid_crc)


def _encode_freq_word(hz: int) -> bytes:
    if hz <= 0:
        raise ValueError(f"Frequency must be positive, got {hz}")
    word = int(round(hz / 100.0))
    if not 0 <= word <= 0xFFFF:
        raise ValueError(f"Frequency word out of range: {hz} Hz -> {word}")
    return struct.pack(">H", word)


def build_power_control(on: bool) -> bytes:
    return build_frame(Command.POWER_CONTROL, bytes([0xFF if on else 0x00]))


def build_freq_set(hz: int) -> bytes:
    return build_frame(Command.FREQ_SET, _encode_freq_word(hz))


def build_stim_amp_set(current_uA: int) -> bytes:
    if not 0 <= current_uA <= 0xFFFF:
        raise ValueError(f"Stimulus current out of range: {current_uA}")
    return build_frame(Command.STI_AMP_SET, struct.pack(">H", current_uA))


def build_voltage_amp_set(level_1: int, level_2: int | None = None) -> bytes:
    level_2 = level_1 if level_2 is None else level_2
    for level in (level_1, level_2):
        if not 0 <= level < len(VOLTAGE_AMP_FACTORS):
            raise ValueError(f"Invalid voltage amp level: {level}")
    return build_frame(Command.VOLTAGE_AMP_SET, bytes([level_1, level_2]))


def build_sweep_set(
    sti_1: int,
    sti_2: int,
    mea_1: int,
    mea_2: int,
    amp_level: int,
) -> bytes:
    return build_frame(
        Command.SWEEP_SET,
        bytes(
            [sti_1 & 0xFF, sti_2 & 0xFF, mea_1 & 0xFF, mea_2 & 0xFF, amp_level & 0xFF]
        ),
    )


def build_single_point_test(hz: int) -> bytes:
    return build_frame(Command.SINGLE_POINT_TEST, _encode_freq_word(hz))


def build_start_measurement(hz: int) -> bytes:
    """Build the old 2-byte host-style acquisition command."""
    return build_frame(Command.START_MEA, _encode_freq_word(hz))


def build_start_measurement_with_mode(
    hz: int, mea_mode: int = DEFAULT_MEA_MODE
) -> bytes:
    """Build the 3-byte acquisition command expected by the checked-in C8051 code."""
    return build_frame(
        Command.START_MEA, bytes([mea_mode & 0xFF]) + _encode_freq_word(hz)
    )


def build_contact_impedance_mea(voltage_amp_level: int = 0) -> bytes:
    return build_frame(Command.CONTACT_IMPEDANCE_MEA, bytes([voltage_amp_level & 0xFF]))


def build_capability_query() -> bytes:
    return build_frame(Command.CAPABILITY_QUERY)


def build_stream_start(hz: int, mea_mode: int = DEFAULT_MEA_MODE) -> bytes:
    return build_frame(
        Command.STREAM_START, bytes([mea_mode & 0xFF]) + _encode_freq_word(hz)
    )


def build_stream_stop() -> bytes:
    return build_frame(Command.STREAM_STOP)


def build_relay_registration(user_id: int = DEFAULT_USER_ID) -> bytes:
    return _build_raw_frame(RelayCommand.REGISTER, bytes([0x02, user_id & 0xFF]))


def build_relay_transmit(
    device_cmd: int,
    device_payload: bytes,
    *,
    board_id: int = DEFAULT_BOARD_ID,
    user_id: int = DEFAULT_USER_ID,
) -> bytes:
    payload = (
        bytes([board_id & 0xFF, user_id & 0xFF, device_cmd & 0xFF]) + device_payload
    )
    return _build_raw_frame(RelayCommand.TRANSMIT, payload)


def relay_device_payload_to_frame(device_cmd: int, device_payload: bytes) -> bytes:
    """Rebuild a synthetic device frame from relay payload for upper-layer parsing."""
    return build_frame(device_cmd, device_payload)


def _component_scale(gain: float, params: ADCParams) -> float:
    if gain <= 0:
        return params.component_scale * params.amplitude_scale
    return (params.component_scale * params.amplitude_scale) / gain


def adc_to_voltage(
    adc0_h: int,
    adc0_l: int,
    adc1_h: int,
    adc1_l: int,
    *,
    gain: float = 1.0,
    params: ADCParams = DEFAULT_ADC_PARAMS,
) -> tuple[float, float]:
    """Convert one 4-byte ADC sample to calibrated real/imag voltage components."""
    adc0 = (adc0_h << 8) | adc0_l
    adc1 = (adc1_h << 8) | adc1_l
    real = (
        (adc0 * params.vref / params.max_value) - params.offset_v - params.real_offset
    )
    imag = (
        (adc1 * params.vref / params.max_value) - params.offset_v - params.imag_offset
    )
    scale = _component_scale(gain, params)
    return real * scale, imag * scale


def parse_measurement_frame(
    data: bytes,
    *,
    gain_level_1: int = 7,
    gain_level_2: int | None = None,
    spec: FrameSpec = DEFAULT_FRAME_SPEC,
    params: ADCParams = DEFAULT_ADC_PARAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse one calibrated measurement frame into real/imag arrays.

    Uses a single uniform gain for all measurement points.
    ``gain_level_2`` is accepted for backward compatibility but ignored;
    ``gain_level_1`` is applied to every point.
    """
    expected = spec.points_per_frame * spec.bytes_per_point
    if len(data) != expected:
        raise ValueError(f"Expected {expected} bytes, got {len(data)}")

    from eit_app.models.precision import compute_dtype

    gain = VOLTAGE_AMP_FACTORS[gain_level_1]
    dtype = compute_dtype()
    real = np.empty(spec.points_per_frame, dtype=dtype)
    imag = np.empty(spec.points_per_frame, dtype=dtype)

    for i in range(spec.points_per_frame):
        off = i * spec.bytes_per_point
        r, im = adc_to_voltage(
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
            gain=gain,
            params=params,
        )
        real[i] = r
        imag[i] = im

    return real, imag


def parse_single_point_response(
    data: bytes,
    *,
    gain_level: int = 0,
    params: ADCParams = DEFAULT_ADC_PARAMS,
) -> tuple[float, float]:
    if len(data) < 4:
        raise ValueError(f"Single point response too short: {len(data)}")
    gain = VOLTAGE_AMP_FACTORS[gain_level]
    return adc_to_voltage(data[0], data[1], data[2], data[3], gain=gain, params=params)


def parse_contact_impedance_response(
    data: bytes,
    *,
    gain_level: int = 0,
    params: ADCParams = DEFAULT_ADC_PARAMS,
) -> np.ndarray:
    """Parse legacy or extended contact impedance frames."""
    if len(data) % 4 != 0 or len(data) == 0:
        raise ValueError(f"Invalid contact impedance payload length: {len(data)}")
    from eit_app.models.precision import compute_dtype

    gain = VOLTAGE_AMP_FACTORS[gain_level]
    result = np.empty(len(data) // 4, dtype=compute_dtype())
    for idx in range(result.shape[0]):
        off = idx * 4
        real, imag = adc_to_voltage(
            data[off],
            data[off + 1],
            data[off + 2],
            data[off + 3],
            gain=gain,
            params=params,
        )
        result[idx] = float(np.sqrt(real**2 + imag**2))
    return result
