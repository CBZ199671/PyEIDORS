"""Universal frame data container shared across acquisition, recording, and display."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameData:
    """Single EIT measurement frame.

    This is the universal data unit passed between acquisition, recording,
    and visualization. It holds voltage-converted measurements (not raw ADC).

    Attributes:
        real: Real-part voltages, shape (n_meas,), typically (208,).
        imag: Imaginary-part voltages, shape (n_meas,).
        timestamp: Unix epoch seconds at acquisition time.
        frame_index: Monotonically increasing counter from the acquisition session.
        metadata: Additional per-frame info (frequency, amplitude, etc.).
    """

    real: np.ndarray
    imag: np.ndarray
    timestamp: float
    frame_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_meas(self) -> int:
        return self.real.shape[0]

    def to_measurement_vector(self, use_part: str = "real") -> np.ndarray:
        """Extract measurement vector compatible with MeasurementDataset.

        Args:
            use_part: "real", "imag", or "mag" (magnitude).

        Returns:
            1-D float64 array of length n_meas.
        """
        if use_part == "real":
            return self.real.copy()
        if use_part == "imag":
            return self.imag.copy()
        if use_part == "mag":
            return np.abs(self.real + 1j * self.imag)
        raise ValueError(f"Unknown use_part: {use_part!r}. Expected 'real', 'imag', or 'mag'.")

    def amplitude(self) -> np.ndarray:
        """Compute calibrated amplitude: 2 * sqrt(R^2 + I^2) * 1.10."""
        return 2.0 * np.sqrt(self.real ** 2 + self.imag ** 2) * 1.10

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for YAML sidecar writing)."""
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "n_meas": self.n_meas,
            **self.metadata,
        }
