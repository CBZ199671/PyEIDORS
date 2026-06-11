"""Simulated EIT device for offline development and testing."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from eit_app.measurement_layout import measurement_layout_from_config

from .base_transport import AbstractHardwareDevice, RawFrame
from .types import AcquisitionMode


class SimulatorDevice(AbstractHardwareDevice):
    """Generates synthetic measurement frames for the configured layout.

    Simulates a circular conductivity anomaly producing a characteristic
    voltage pattern. Useful for GUI development without physical hardware.

    Args:
        fps: Target frames per second (controls sleep between reads).
        noise_std: Standard deviation of additive Gaussian noise.
        seed: RNG seed for reproducible noise.
        n_electrodes: Electrodes per ring.
        n_rings: Number of rings.
    """

    def __init__(
        self,
        fps: float = 30.0,
        noise_std: float = 0.002,
        seed: int | None = None,
        *,
        n_electrodes: int = 16,
        n_rings: int = 1,
        stim_pattern: str = "{ad}",
        meas_pattern: str = "{ad}",
        use_meas_current: bool = False,
        use_meas_current_next: int = 0,
        rotate_meas: bool = True,
        stim_direction: str = "ccw",
        meas_direction: str = "ccw",
        stim_first_positive: bool = False,
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
        self._power_on = False
        self._layout = measurement_layout_from_config(
            {
                "n_electrodes": n_electrodes,
                "n_rings": n_rings,
                "stim_pattern": stim_pattern,
                "meas_pattern": meas_pattern,
                "use_meas_current": use_meas_current,
                "use_meas_current_next": use_meas_current_next,
                "rotate_meas": rotate_meas,
                "stim_direction": stim_direction,
                "meas_direction": meas_direction,
                "stim_first_positive": stim_first_positive,
            }
        )
        self._point_count = int(self._layout["points_per_frame"])
        self._base_real = self._generate_base_pattern()
        self._base_imag = self._base_real * 0.1  # small imaginary component

    def _generate_base_pattern(self) -> np.ndarray:
        """Create a synthetic voltage pattern mimicking adjacent drive."""
        n = self._point_count
        x = np.linspace(0, 4 * np.pi, n)
        # Simulate a background + localized anomaly
        base = 0.5 + 0.1 * np.sin(x)
        center = max(n // 3, 1)
        width = max(n / 20.0, 1.0)
        anomaly = 0.15 * np.exp(-0.5 * ((np.arange(n) - center) / width) ** 2)
        return base + anomaly

    def connect(self) -> None:
        self._connected = True
        self._frame_index = 0
        self._power_on = False

    def disconnect(self) -> None:
        self._connected = False
        self._measuring = False
        self._power_on = False

    def power_control(self, on: bool) -> None:
        self._power_on = bool(on)

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

        noise_r = self._rng.normal(0, self._noise_std, self._point_count)
        noise_i = self._rng.normal(0, self._noise_std, self._point_count)

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
                "n_elec": int(self._layout["n_elec"]),
                "n_rings": int(self._layout["n_rings"]),
                "stim_pattern": self._layout["stim_pattern"],
                "meas_pattern": self._layout["meas_pattern"],
                "use_meas_current": bool(self._layout["use_meas_current"]),
                "use_meas_current_next": int(self._layout["use_meas_current_next"]),
                "points_per_frame": self._point_count,
            },
        )

    def measure_contact_impedance(self) -> np.ndarray:
        return self._rng.uniform(
            50.0, 200.0, size=int(self._layout["total_electrodes"])
        )

    def single_point_test(self) -> tuple[float, float]:
        return 0.5 + self._rng.normal(0, 0.01), 0.05 + self._rng.normal(0, 0.001)

    def single_point_test_at(self, hz: int) -> tuple[float, float]:
        self._frequency_hz = int(hz)
        return self.single_point_test()

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
            "points_per_frame": self._point_count,
            "n_elec": int(self._layout["n_elec"]),
            "n_rings": int(self._layout["n_rings"]),
        }
