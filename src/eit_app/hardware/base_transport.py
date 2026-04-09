"""Abstract transport and device interfaces for EIT hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import AcquisitionMode


@dataclass
class RawFrame:
    """Raw measurement frame from hardware after ADC conversion."""

    real: np.ndarray  # (208,) float64 voltages
    imag: np.ndarray  # (208,) float64 voltages
    timestamp: float  # time.time() at reception
    metadata: dict[str, Any] | None = None


class AbstractTransport(ABC):
    """Low-level byte transport (serial, TCP, etc.)."""

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def write(self, data: bytes) -> None: ...

    @abstractmethod
    def read(self, size: int, timeout: float = 2.0) -> bytes: ...

    @abstractmethod
    def read_until(self, terminator: bytes, timeout: float = 2.0) -> bytes: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...


class AbstractHardwareDevice(ABC):
    """High-level EIT device interface."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def power_control(self, on: bool) -> None: ...

    @abstractmethod
    def set_frequency(self, hz: int) -> None: ...

    @abstractmethod
    def set_stim_amplitude(self, level: int) -> None: ...

    @abstractmethod
    def set_voltage_amp(self, level: int) -> None: ...

    @abstractmethod
    def start_measurement(self) -> None: ...

    @abstractmethod
    def stop_measurement(self) -> None: ...

    @abstractmethod
    def read_frame(self) -> RawFrame: ...

    @abstractmethod
    def measure_contact_impedance(self) -> np.ndarray: ...

    @abstractmethod
    def single_point_test(self) -> tuple[float, float]: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @property
    @abstractmethod
    def acquisition_mode(self) -> AcquisitionMode: ...

    @abstractmethod
    def capabilities(self) -> dict[str, Any]: ...
