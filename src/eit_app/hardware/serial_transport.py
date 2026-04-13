"""Serial transport and legacy C8051 device implementation."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base_transport import AbstractHardwareDevice, AbstractTransport, RawFrame
from .protocol import (
    build_capability_query,
    build_contact_impedance_mea,
    build_freq_set,
    build_power_control,
    build_single_point_test,
    build_start_measurement,
    build_start_measurement_with_mode,
    build_stim_amp_set,
    build_stream_start,
    build_stream_stop,
    build_voltage_amp_set,
    parse_contact_impedance_response,
    parse_measurement_frame,
    parse_response,
    parse_single_point_response,
)
from .types import (
    FRAME_HEAD,
    FRAME_END,
    AcquisitionMode,
    Command,
    DEFAULT_FRAME_SPEC,
    DEFAULT_MEA_MODE,
    FrameSpec,
    STIM_AMP_VALUES_UA,
)

log = logging.getLogger(__name__)

try:
    import serial as _serial
except ImportError:  # pragma: no cover
    _serial = None  # type: ignore[assignment]


class SerialTransport(AbstractTransport):
    """pyserial-based transport for direct USB/UART connection."""

    def __init__(self, port: str, baudrate: int = 115200) -> None:
        self._port = port
        self._baudrate = baudrate
        self._serial: _serial.Serial | None = None  # type: ignore[name-defined]

    def open(self) -> None:
        if _serial is None:
            raise ImportError("pyserial is required: pip install pyserial")
        self._serial = _serial.Serial(
            self._port, self._baudrate, timeout=2.0, write_timeout=2.0
        )

    def close(self) -> None:
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
        self._serial = None

    def write(self, data: bytes) -> None:
        if self._serial is None:
            raise RuntimeError("Transport not open")
        self._serial.write(data)

    def read(self, size: int, timeout: float = 2.0) -> bytes:
        if self._serial is None:
            raise RuntimeError("Transport not open")
        self._serial.timeout = timeout
        return self._serial.read(size)

    def read_until(self, terminator: bytes, timeout: float = 2.0) -> bytes:
        if self._serial is None:
            raise RuntimeError("Transport not open")
        self._serial.timeout = timeout
        return self._serial.read_until(terminator)

    def reset_input_buffer(self) -> None:
        if self._serial is None:
            raise RuntimeError("Transport not open")
        self._serial.reset_input_buffer()

    def reset_output_buffer(self) -> None:
        if self._serial is None:
            raise RuntimeError("Transport not open")
        self._serial.reset_output_buffer()

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open


class C8051Device(AbstractHardwareDevice):
    """Legacy-compatible C8051F060 EIT measurement board."""

    def __init__(
        self,
        transport: AbstractTransport,
        device_config: dict[str, Any] | None = None,
    ) -> None:
        self._transport = transport
        self._config = dict(device_config or {})
        self._connected = False
        self._measuring = False
        self._prepared = False
        self._start_variant = str(self._config.get("start_variant", "auto"))
        mode_name = str(self._config.get("acquisition_mode", "legacy_one_shot")).lower()
        self._acquisition_mode = (
            AcquisitionMode.STREAMING
            if mode_name == "streaming"
            else AcquisitionMode.LEGACY_ONE_SHOT
        )

    def connect(self) -> None:
        self._transport.open()
        self._connected = True
        self._prepared = False

    def disconnect(self) -> None:
        self._measuring = False
        self._prepared = False
        self._transport.close()
        self._connected = False

    def power_control(self, on: bool) -> None:
        self._transport.write(build_power_control(on))
        if not on:
            self._prepared = False

    def set_frequency(self, hz: int) -> None:
        self._config["frequency_hz"] = int(hz)
        self._transport.write(build_freq_set(int(hz)))

    def set_stim_amplitude(self, level: int) -> None:
        level = int(level)
        current_uA = STIM_AMP_VALUES_UA.get(level, level)
        self._config["stim_amp_level"] = level
        self._config["stim_amp_uA"] = current_uA
        self._transport.write(build_stim_amp_set(current_uA))

    def set_voltage_amp(self, level: int) -> None:
        self.set_voltage_amp_levels(level, level)

    def set_voltage_amp_levels(self, level_1: int, level_2: int) -> None:
        self._config["voltage_amp_level_1"] = int(level_1)
        self._config["voltage_amp_level_2"] = int(level_2)
        self._config.setdefault("contact_impedance_amp_level", int(level_1))
        self._transport.write(build_voltage_amp_set(int(level_1), int(level_2)))

    def start_measurement(self) -> None:
        self._prepare_measurement_state()
        if self.acquisition_mode is AcquisitionMode.STREAMING:
            self._transport.write(
                build_stream_start(
                    int(self._config.get("frequency_hz", 1000)),
                    int(self._config.get("mea_mode", DEFAULT_MEA_MODE)),
                )
            )
        self._measuring = True

    def stop_measurement(self) -> None:
        if self.acquisition_mode is AcquisitionMode.STREAMING:
            self._transport.write(build_stream_stop())
        self._measuring = False

    def read_frame(self) -> RawFrame:
        if not self._connected:
            raise RuntimeError("Device is not connected")
        if not self._measuring:
            raise RuntimeError("Measurement has not been started")

        if self.acquisition_mode is AcquisitionMode.LEGACY_ONE_SHOT:
            return self._read_legacy_one_shot_frame()

        result = self._read_device_response(timeout=self._stream_response_timeout())
        if result.cmd != Command.START_MEA:
            raise RuntimeError(f"Unexpected frame command: {result.cmd}")
        real, imag = parse_measurement_frame(
            result.data,
            gain_level_1=int(self._config.get("voltage_amp_level_1", 3)),
            gain_level_2=int(self._config.get("voltage_amp_level_2", 5)),
            spec=self._frame_spec(),
        )
        return RawFrame(
            real=real,
            imag=imag,
            timestamp=time.time(),
            metadata=self._frame_metadata(),
        )

    def measure_contact_impedance(self) -> np.ndarray:
        self._ensure_device_ready()
        amp_level = int(self._config.get("contact_impedance_amp_level", 0))
        retries = max(1, int(self._config.get("command_retries", 1)))
        errors: list[str] = []
        for attempt in range(1, retries + 1):
            try:
                self._clear_transport_input()
                self._transport.write(build_contact_impedance_mea(amp_level))
                result = self._read_device_response(timeout=self._command_response_timeout())
                if result.cmd != Command.CONTACT_IMPEDANCE_MEA:
                    raise RuntimeError(f"Unexpected impedance response: {result.cmd}")
                return parse_contact_impedance_response(result.data, gain_level=amp_level)
            except Exception as exc:
                errors.append(f"{attempt}/{retries}: {exc}")
                if attempt < retries:
                    time.sleep(0.25)
        raise RuntimeError(f"Contact impedance failed: {'; '.join(errors)}")

    def single_point_test(self) -> tuple[float, float]:
        hz = int(self._config.get("frequency_hz", 1000))
        return self.single_point_test_at(hz)

    def single_point_test_at(self, hz: int) -> tuple[float, float]:
        self._ensure_device_ready()
        retries = max(1, int(self._config.get("command_retries", 1)))
        errors: list[str] = []
        for attempt in range(1, retries + 1):
            try:
                self._clear_transport_input()
                self._transport.write(build_single_point_test(hz))
                result = self._read_device_response(timeout=self._command_response_timeout())
                if result.cmd != Command.SINGLE_POINT_TEST:
                    raise RuntimeError(f"Unexpected single-point response: {result.cmd}")
                return parse_single_point_response(
                    result.data,
                    gain_level=int(self._config.get("voltage_amp_level_1", 3)),
                )
            except Exception as exc:
                errors.append(f"{attempt}/{retries}: {exc}")
                if attempt < retries:
                    time.sleep(0.25)
        raise RuntimeError(f"Single-point test failed: {'; '.join(errors)}")

    def run_sweep(self, start_hz: int, end_hz: int, n_points: int) -> list[dict[str, float]]:
        if n_points < 2:
            raise ValueError("Sweep requires at least 2 points")
        frequencies = np.linspace(start_hz, end_hz, n_points, dtype=float)
        results: list[dict[str, float]] = []
        for freq in frequencies:
            real, imag = self.single_point_test_at(int(round(float(freq))))
            results.append(
                {
                    "frequency_hz": float(freq),
                    "real": float(real),
                    "imag": float(imag),
                }
            )
        return results

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def acquisition_mode(self) -> AcquisitionMode:
        return self._acquisition_mode

    def capabilities(self) -> dict[str, Any]:
        return {
            "protocol_version": self._config.get("protocol_version", "legacy-v1"),
            "acquisition_mode": (
                "streaming"
                if self.acquisition_mode is AcquisitionMode.STREAMING
                else "legacy_one_shot"
            ),
            "supports_streaming": self.acquisition_mode is AcquisitionMode.STREAMING,
            "supports_extended_impedance": False,
            "supports_3d_batch": False,
            "supported_mea_modes": [2, 3],
        }

    def try_query_capabilities(self) -> dict[str, Any]:
        """Best-effort capability query for v2 firmware."""
        self._transport.write(build_capability_query())
        try:
            result = self._read_device_response(timeout=1.0)
        except Exception:
            return self.capabilities()
        if result.cmd != Command.CAPABILITY_QUERY or len(result.data) < 4:
            return self.capabilities()

        caps = {
            "protocol_version": f"v{result.data[0]}",
            "supports_streaming": bool(result.data[1] & 0x01),
            "supports_extended_impedance": bool(result.data[1] & 0x02),
            "supports_3d_batch": bool(result.data[1] & 0x04),
            "acquisition_mode": "streaming" if (result.data[1] & 0x01) else "legacy_one_shot",
        }
        self._config.update(caps)
        self._acquisition_mode = (
            AcquisitionMode.STREAMING
            if caps["supports_streaming"]
            else AcquisitionMode.LEGACY_ONE_SHOT
        )
        return caps

    def _prepare_measurement_state(self) -> None:
        self.power_control(True)
        settle_sec = float(self._config.get("power_on_settle_sec", 0.8))
        if settle_sec > 0:
            time.sleep(settle_sec)

        if bool(self._config.get("apply_profile_on_start", True)):
            self.set_stim_amplitude(int(self._config.get("stim_amp_level", 1)))
            self.set_voltage_amp_levels(
                int(self._config.get("voltage_amp_level_1", 3)),
                int(self._config.get("voltage_amp_level_2", 5)),
            )
        self._prepared = True

    def _read_legacy_one_shot_frame(self) -> RawFrame:
        hz = int(self._config.get("frequency_hz", 1000))
        variants = self._start_command_variants(hz)
        errors: list[str] = []
        retries = max(1, int(self._config.get("legacy_frame_retries", 1)))
        if self._start_variant == "auto":
            retries = 1

        for variant_name, packet in variants:
            for attempt in range(1, retries + 1):
                try:
                    self._clear_transport_input()
                    self._transport.write(packet)
                    result = self._read_device_response(timeout=self._legacy_frame_timeout())
                    if result.cmd != Command.START_MEA:
                        raise RuntimeError(f"Unexpected frame command: {result.cmd}")
                    self._start_variant = variant_name
                    real, imag = parse_measurement_frame(
                        result.data,
                        gain_level_1=int(self._config.get("voltage_amp_level_1", 0)),
                        gain_level_2=int(self._config.get("voltage_amp_level_2", 0)),
                        spec=self._frame_spec(),
                    )
                    return RawFrame(
                        real=real,
                        imag=imag,
                        timestamp=time.time(),
                        metadata=self._frame_metadata(),
                    )
                except Exception as exc:
                    errors.append(f"{variant_name}[{attempt}/{retries}]: {exc}")
                    if attempt < retries:
                        time.sleep(0.25)

        raise RuntimeError(f"Legacy acquisition failed: {'; '.join(errors)}")

    def _start_command_variants(self, hz: int) -> Sequence[tuple[str, bytes]]:
        if self._start_variant == "2byte":
            return [("2byte", build_start_measurement(hz))]
        if self._start_variant == "3byte":
            return [
                (
                    "3byte",
                    build_start_measurement_with_mode(
                        hz,
                        int(self._config.get("mea_mode", DEFAULT_MEA_MODE)),
                    ),
                )
            ]
        return [
            ("2byte", build_start_measurement(hz)),
            (
                "3byte",
                build_start_measurement_with_mode(
                    hz,
                    int(self._config.get("mea_mode", DEFAULT_MEA_MODE)),
                ),
            ),
        ]

    def _read_device_response(self, timeout: float) -> Any:
        buf = self._read_frame_packet(timeout)
        result = parse_response(buf)
        if result is None or not result.valid_crc:
            raise RuntimeError("Invalid frame received from device")
        return result

    def _read_frame_packet(self, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        buf = bytearray()
        head_len = len(FRAME_HEAD)

        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            chunk = self._transport.read(1, timeout=remaining)
            if not chunk:
                continue
            buf.extend(chunk)

            head_idx = buf.find(FRAME_HEAD)
            if head_idx < 0:
                keep = 2
                if len(buf) > keep:
                    del buf[:-keep]
                continue
            if head_idx > 0:
                del buf[:head_idx]

            if len(buf) < head_len + 2:
                continue

            len_field = int.from_bytes(buf[head_len: head_len + 2], "big")
            if len_field < 3:
                del buf[:1]
                continue

            frame_total = len_field + 7
            while len(buf) < frame_total and time.monotonic() < deadline:
                remaining = max(0.05, deadline - time.monotonic())
                chunk = self._transport.read(frame_total - len(buf), timeout=remaining)
                if not chunk:
                    continue
                buf.extend(chunk)

            if len(buf) >= frame_total:
                return bytes(buf[:frame_total])

        raise RuntimeError("Timed out while reading device frame")

    def _frame_metadata(self) -> dict[str, Any]:
        frame_spec = self._frame_spec()
        return {
            "frequency_hz": int(self._config.get("frequency_hz", 1000)),
            "stim_amp_uA": int(self._config.get("stim_amp_uA", 100)),
            "voltage_amp_level_1": int(self._config.get("voltage_amp_level_1", 3)),
            "voltage_amp_level_2": int(self._config.get("voltage_amp_level_2", 5)),
            "mea_mode": int(self._config.get("mea_mode", DEFAULT_MEA_MODE)),
            "board_id": int(self._config.get("board_id", 1)),
            "user_id": int(self._config.get("user_id", 1)),
            "transport_type": self._config.get("transport_type", "serial"),
            "protocol_version": self._config.get("protocol_version", "legacy-v1"),
            "n_elec": int(self._config.get("n_elec", frame_spec.n_electrodes)),
            "n_rings": int(self._config.get("n_rings", 1)),
            "stim_pattern": self._config.get("stim_pattern", "{ad}"),
            "meas_pattern": self._config.get("meas_pattern", "{ad}"),
            "use_meas_current": bool(self._config.get("use_meas_current", False)),
            "use_meas_current_next": int(self._config.get("use_meas_current_next", 0)),
            "points_per_frame": int(frame_spec.points_per_frame),
        }

    def _frame_spec(self) -> FrameSpec:
        total_electrodes = max(int(self._config.get("n_elec", DEFAULT_FRAME_SPEC.n_electrodes)), 1)
        total_electrodes *= max(int(self._config.get("n_rings", 1)), 1)
        points_per_frame = max(
            int(self._config.get("points_per_frame", DEFAULT_FRAME_SPEC.points_per_frame)),
            1,
        )
        return FrameSpec(
            n_electrodes=total_electrodes,
            points_per_frame=points_per_frame,
            bytes_per_point=DEFAULT_FRAME_SPEC.bytes_per_point,
        )

    def _clear_transport_input(self) -> None:
        reset_input = getattr(self._transport, "reset_input_buffer", None)
        if callable(reset_input):
            try:
                reset_input()
            except Exception:
                pass

    def _ensure_device_ready(self) -> None:
        if not self._prepared:
            self._prepare_measurement_state()

    def _command_response_timeout(self) -> float:
        return float(self._config.get("command_timeout_sec", 5.0))

    def _legacy_frame_timeout(self) -> float:
        return float(self._config.get("legacy_frame_timeout_sec", 20.0))

    def _stream_response_timeout(self) -> float:
        return float(self._config.get("stream_timeout_sec", 5.0))
