"""Simulated EIT device for offline development and testing."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base_transport import AbstractHardwareDevice, AbstractTransport, RawFrame
from .types import AcquisitionMode, DEFAULT_FRAME_SPEC


class SimulatorTransport(AbstractTransport):
    """In-memory transport that echoes nothing (used by SimulatorDevice)."""

    def __init__(self) -> None:
        self._open = False

    def open(self) -> None:
        self._open = True

    def close(self) -> None:
        self._open = False

    def write(self, data: bytes) -> None:
        pass  # swallow outgoing bytes

    def read(self, size: int, timeout: float = 2.0) -> bytes:
        return b""

    def read_until(self, terminator: bytes, timeout: float = 2.0) -> bytes:
        return b""

    @property
    def is_open(self) -> bool:
        return self._open


class SimulatorDevice(AbstractHardwareDevice):
    """Generates synthetic 208-point measurement frames.

    Simulates a circular conductivity anomaly producing a characteristic
    voltage pattern. Useful for GUI development without physical hardware.

    Args:
        fps: Target frames per second (controls sleep between reads).
        noise_std: Standard deviation of additive Gaussian noise.
        seed: RNG seed for reproducible noise.
    """

    def __init__(
        self,
        fps: float = 30.0,
        noise_std: float = 0.002,
        seed: int | None = None,
    ) -> None:
        self._fps = fps
        self._noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._connected = False
        self._measuring = False
        self._frame_index = 0
        self._frequency_hz = 1000
        self._stim_amp_uA = 100
        self._voltage_amp_level_1 = 0
        self._voltage_amp_level_2 = 0
        self._base_real = self._generate_base_pattern()
        self._base_imag = self._base_real * 0.1  # small imaginary component

    def _generate_base_pattern(self) -> np.ndarray:
        """Create a synthetic voltage pattern mimicking adjacent drive."""
        n = DEFAULT_FRAME_SPEC.points_per_frame
        x = np.linspace(0, 4 * np.pi, n)
        # Simulate a background + localized anomaly
        base = 0.5 + 0.1 * np.sin(x)
        # Add anomaly signature around measurement 50-80
        anomaly = 0.15 * np.exp(-0.5 * ((np.arange(n) - 65) / 10) ** 2)
        return base + anomaly

    def connect(self) -> None:
        self._connected = True
        self._frame_index = 0

    def disconnect(self) -> None:
        self._connected = False
        self._measuring = False

    def power_control(self, on: bool) -> None:
        pass

    def set_frequency(self, hz: int) -> None:
        self._frequency_hz = hz

    def set_stim_amplitude(self, level: int) -> None:
        self._stim_amp_uA = level

    def set_voltage_amp(self, level: int) -> None:
        self._voltage_amp_level_1 = level
        self._voltage_amp_level_2 = level

    def set_voltage_amp_levels(self, level_1: int, level_2: int) -> None:
        self._voltage_amp_level_1 = level_1
        self._voltage_amp_level_2 = level_2

    def start_measurement(self) -> None:
        self._measuring = True

    def stop_measurement(self) -> None:
        self._measuring = False

    def read_frame(self) -> RawFrame:
        """Return a synthetic frame, sleeping to match target FPS."""
        if self._fps > 0:
            time.sleep(1.0 / self._fps)

        noise_r = self._rng.normal(0, self._noise_std, DEFAULT_FRAME_SPEC.points_per_frame)
        noise_i = self._rng.normal(0, self._noise_std, DEFAULT_FRAME_SPEC.points_per_frame)

        self._frame_index += 1
        return RawFrame(
            real=(self._base_real + noise_r).copy(),
            imag=(self._base_imag + noise_i).copy(),
            timestamp=time.time(),
            metadata={
                "frequency_hz": self._frequency_hz,
                "stim_amp_uA": self._stim_amp_uA,
                "voltage_amp_level_1": self._voltage_amp_level_1,
                "voltage_amp_level_2": self._voltage_amp_level_2,
                "mea_mode": 2,
                "transport_type": "simulator",
                "protocol_version": "simulator",
            },
        )

    def measure_contact_impedance(self) -> np.ndarray:
        return self._rng.uniform(50.0, 200.0, size=DEFAULT_FRAME_SPEC.n_electrodes)

    def single_point_test(self) -> tuple[float, float]:
        return 0.5 + self._rng.normal(0, 0.01), 0.05 + self._rng.normal(0, 0.001)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def acquisition_mode(self) -> AcquisitionMode:
        return AcquisitionMode.STREAMING

    def capabilities(self) -> dict[str, Any]:
        return {
            "protocol_version": "simulator",
            "supports_streaming": True,
            "supports_extended_impedance": True,
            "supports_3d_batch": False,
            "acquisition_mode": "streaming",
        }
