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
        real: Real-part voltages, shape (n_meas,).
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
            use_part: "real", "imag", "mag" (magnitude), or "complex".

        Returns:
            1-D array of length n_meas.
        """
        if use_part == "real":
            return self.real.copy()
        if use_part == "imag":
            return self.imag.copy()
        if use_part == "mag":
            return np.hypot(self.real, self.imag)
        if use_part == "complex":
            real = np.asarray(self.real)
            imag = np.asarray(self.imag)
            out = np.empty(
                real.shape, dtype=np.result_type(real.dtype, imag.dtype, np.complex64)
            )
            out.real = real
            out.imag = imag
            return out
        raise ValueError(
            f"Unknown use_part: {use_part!r}. "
            "Expected 'real', 'imag', 'mag', or 'complex'."
        )

    def amplitude(self) -> np.ndarray:
        """Compute calibrated amplitude: 2 * sqrt(R^2 + I^2) * 1.10."""
        amplitude = np.hypot(self.real, self.imag)
        amplitude *= 2.2
        return amplitude

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for YAML sidecar writing)."""
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "n_meas": self.n_meas,
            **self.metadata,
        }
