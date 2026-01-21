"""Measurement dataset helper utilities.

This module provides `MeasurementDataset` for converting measurement matrices
and metadata conforming to the specification into `EITData` objects used
internally by PyEIDORS. This decouples hardware/host software format adaptation
from the inverse problem workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

import numpy as np

from .structures import PatternConfig, EITData
from ..electrodes.patterns import StimMeasPatternManager


def _parse_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {value}")
    return bool(value)


def _parse_direction(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text not in {"cw", "ccw"}:
        raise ValueError(f"Direction must be 'cw' or 'ccw', got: {value}")
    return text


@dataclass
class MeasurementDataset:
    """Encapsulates measurement matrix and metadata conforming to specification.

    Args:
        measurements: Array of shape ``(n_frames, n_meas_total)`` or ``(n_meas_total,)``.
        pattern_config: Configuration for stimulation/measurement pattern generation.
        metadata: Original metadata dictionary, mainly for tracking and debugging.
        data_type: ``type`` label passed to ``EITData``, e.g., ``"real"``/``"difference"``.
    """

    measurements: np.ndarray
    pattern_config: PatternConfig
    stim_matrix: np.ndarray
    n_elec: int
    n_stim: int
    n_meas_total: int
    n_meas_per_stim: Sequence[int]
    metadata: Mapping[str, Any]
    data_type: str = "real"

    # ---------------------------- Construction Interface ----------------------------
    @classmethod
    def from_metadata(
        cls,
        measurements: Union[np.ndarray, Sequence[Sequence[float]]],
        metadata: Mapping[str, Any],
        data_type: str = "real",
    ) -> "MeasurementDataset":
        """Construct measurement dataset from metadata.

        This method will:
        1. Build ``PatternConfig``;
        2. Create ``StimMeasPatternManager`` and compute measurement counts;
        3. Validate measurement matrix dimensions;
        4. Return the encapsulated dataset object.
        """

        measurements_array = cls._normalize_measurements(measurements)
        pattern_config = cls._pattern_config_from_metadata(metadata)
        pattern_manager = StimMeasPatternManager(pattern_config)

        expected_meas = pattern_manager.n_meas_total
        if measurements_array.shape[1] != expected_meas:
            raise ValueError(
                "Measurement matrix columns do not match stimulation/measurement pattern: "
                f"got {measurements_array.shape[1]} columns, "
                f"expected {expected_meas} columns."
            )

        # n_frames: actual frame count in CSV (each frame has 2 columns: real + imaginary)
        # Validation: loaded frames cannot exceed n_frames
        expected_frames = metadata.get("n_frames")
        if expected_frames is not None and measurements_array.shape[0] > expected_frames:
            raise ValueError(
                "Loaded frame count exceeds declared n_frames in metadata: "
                f"n_frames={expected_frames}, actual loaded={measurements_array.shape[0]}"
            )

        return cls(
            measurements=measurements_array,
            pattern_config=pattern_config,
            stim_matrix=pattern_manager.stim_matrix.copy(),
            n_elec=pattern_config.n_elec,
            n_stim=pattern_manager.n_stim,
            n_meas_total=pattern_manager.n_meas_total,
            n_meas_per_stim=tuple(pattern_manager.n_meas_per_stim),
            metadata=dict(metadata),
            data_type=data_type,
        )

    # ---------------------------- Public API ----------------------------
    def to_eit_data(self, frame_index: int = 0, data_type: Optional[str] = None) -> EITData:
        """Convert specified frame to ``EITData`` object.

        Args:
            frame_index: Selected frame index, defaults to first frame.
            data_type: Override default ``data_type``, e.g., ``"difference"``.
        """

        frame = self._get_frame(frame_index)
        return EITData(
            meas=frame.copy(),
            stim_pattern=self.stim_matrix.copy(),
            n_elec=self.n_elec,
            n_stim=self.n_stim,
            n_meas=self.n_meas_total,
            type=data_type or self.data_type,
        )

    def iter_frames(self, data_type: Optional[str] = None) -> Iterator[EITData]:
        """Generate ``EITData`` objects frame by frame."""

        for idx in range(self.measurements.shape[0]):
            yield self.to_eit_data(frame_index=idx, data_type=data_type)

    def summary(self) -> Dict[str, Any]:
        """Return summary of measurement configuration and data dimensions."""

        return {
            "n_frames": int(self.measurements.shape[0]),
            "n_elec": self.n_elec,
            "n_stim": self.n_stim,
            "n_meas_total": self.n_meas_total,
            "n_meas_per_stim": list(self.n_meas_per_stim),
            "data_type": self.data_type,
        }

    # ---------------------------- Internal Utilities ----------------------------
    @staticmethod
    def _normalize_measurements(
        measurements: Union[np.ndarray, Sequence[Sequence[float]]]
    ) -> np.ndarray:
        array = np.asarray(measurements, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(
                "measurements must be a 1D or 2D array, "
                f"got {array.ndim} dimensions"
            )
        return array

    @staticmethod
    def _pattern_config_from_metadata(metadata: Mapping[str, Any]) -> PatternConfig:
        required_fields = ["n_elec", "stim_pattern", "meas_pattern"]
        missing = [field for field in required_fields if field not in metadata]
        if missing:
            raise KeyError(f"metadata missing required fields: {', '.join(missing)}")

        return PatternConfig(
            n_elec=int(metadata["n_elec"]),
            n_rings=int(metadata.get("n_rings", 1)),
            stim_pattern=metadata.get("stim_pattern", "{ad}"),
            meas_pattern=metadata.get("meas_pattern", "{ad}"),
            amplitude=float(metadata.get("amplitude", 1.0)),
            use_meas_current=_parse_bool(metadata.get("use_meas_current"), False),
            use_meas_current_next=int(metadata.get("use_meas_current_next", 0)),
            rotate_meas=_parse_bool(metadata.get("rotate_meas"), True),
            stim_direction=_parse_direction(metadata.get("stim_direction"), "ccw"),
            meas_direction=_parse_direction(metadata.get("meas_direction"), "ccw"),
            stim_first_positive=_parse_bool(metadata.get("stim_first_positive"), False),
        )

    def _get_frame(self, frame_index: int) -> np.ndarray:
        if not 0 <= frame_index < self.measurements.shape[0]:
            raise IndexError(
                f"frame_index out of range: {frame_index}, "
                f"valid indices are [0, {self.measurements.shape[0] - 1}]"
            )
        return self.measurements[frame_index]
