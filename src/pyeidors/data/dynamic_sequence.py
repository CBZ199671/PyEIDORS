"""Dynamic multi-frame measurement sequence contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from pyeidors.data.channels import normalize_bad_channel_mask
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact, write_hdf5_artifact
from pyeidors.utils.numeric_ops import all_finite_values


DYNAMIC_MEASUREMENT_SEQUENCE_SCHEMA = "pyeidors-dynamic-measurement-sequence-v1"


@dataclass(frozen=True)
class DynamicMeasurementSequence:
    """Multi-frame measurement array plus dynamic acquisition metadata."""

    frames: np.ndarray
    t: np.ndarray
    dt: np.ndarray
    frame_id: np.ndarray
    sampling_rate_hz: float
    reference_policy: str
    stim_meas_signature: str
    bad_channel_mask: np.ndarray
    measurement_weights: np.ndarray
    measurement_weight_kind: str
    frequency_hz: np.ndarray
    context_metadata: MappingProxyType
    metadata: MappingProxyType
    data_type: str = "real"

    @classmethod
    def from_arrays(
        cls,
        frames: Any,
        *,
        t: Any | None = None,
        dt: Any | None = None,
        sampling_rate_hz: float | None = None,
        frame_id: Any | None = None,
        reference_policy: str = "none",
        stim_meas_signature: str = "unspecified",
        bad_channel_mask: Any | None = None,
        measurement_weights: Any | None = None,
        frequency_hz: Any | None = None,
        context_metadata: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        data_type: str = "real",
    ) -> "DynamicMeasurementSequence":
        """Create and validate a dynamic measurement sequence."""

        frame_batch = _frame_batch(frames)
        n_frames, n_measurements = frame_batch.shape
        sample_rate = _sampling_rate(sampling_rate_hz)
        timestamps = _timestamps(t, n_frames=n_frames, sampling_rate_hz=sample_rate)
        deltas = _dt(dt, timestamps=timestamps, sampling_rate_hz=sample_rate)
        ids = _frame_ids(frame_id, n_frames=n_frames)
        mask = _bad_channel_mask_frames(
            bad_channel_mask,
            n_frames=n_frames,
            n_measurements=n_measurements,
        )
        weights, weight_kind = _measurement_weights(
            measurement_weights,
            n_frames=n_frames,
            n_measurements=n_measurements,
        )
        frequencies = _frequency_hz(frequency_hz, n_frames=n_frames)
        ref_policy = _nonempty_string(reference_policy, name="reference_policy")
        signature = _nonempty_string(
            stim_meas_signature,
            name="stim_meas_signature",
        )
        return cls(
            frames=_readonly(frame_batch),
            t=_readonly(timestamps),
            dt=_readonly(deltas),
            frame_id=_readonly(ids),
            sampling_rate_hz=sample_rate,
            reference_policy=ref_policy,
            stim_meas_signature=signature,
            bad_channel_mask=_readonly(mask),
            measurement_weights=_readonly(weights),
            measurement_weight_kind=weight_kind,
            frequency_hz=_readonly(frequencies),
            context_metadata=MappingProxyType(dict(context_metadata or {})),
            metadata=MappingProxyType(dict(metadata or {})),
            data_type=str(data_type).strip() or "real",
        )

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_measurements(self) -> int:
        return int(self.frames.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(v) for v in self.frames.shape)

    def frame_metadata(self, frame_index: int) -> MappingProxyType:
        """Return all per-frame dynamic metadata for ``frame_index``."""

        idx = int(frame_index)
        if not 0 <= idx < self.n_frames:
            raise IndexError(
                f"frame_index out of range: {idx}, valid range [0, {self.n_frames - 1}]"
            )
        return MappingProxyType(
            {
                "frame_index": idx,
                "frame_id": int(self.frame_id[idx]),
                "t": float(self.t[idx]),
                "dt": float(self.dt[idx]),
                "sampling_rate_hz": float(self.sampling_rate_hz),
                "reference_policy": self.reference_policy,
                "stim_meas_signature": self.stim_meas_signature,
                "bad_channel_mask": tuple(bool(v) for v in self.bad_channel_mask[idx]),
                "measurement_weight_kind": self.measurement_weight_kind,
                "frequency_hz": float(self.frequency_hz[idx]),
                "context_metadata": dict(self.context_metadata),
                "data_type": self.data_type,
            }
        )

    def summary(self) -> dict[str, Any]:
        """Return compact sequence dimensions and dynamic metadata."""

        return {
            "schema": DYNAMIC_MEASUREMENT_SEQUENCE_SCHEMA,
            "n_frames": self.n_frames,
            "n_measurements": self.n_measurements,
            "sampling_rate_hz": float(self.sampling_rate_hz),
            "reference_policy": self.reference_policy,
            "stim_meas_signature": self.stim_meas_signature,
            "measurement_weight_kind": self.measurement_weight_kind,
            "data_type": self.data_type,
        }


def write_dynamic_measurement_sequence(
    path: str | Path,
    sequence: DynamicMeasurementSequence,
    *,
    compression: str | None = "gzip",
) -> Path:
    """Write a dynamic measurement sequence as an HDF5 package."""

    arrays = {
        "frames": sequence.frames,
        "t": sequence.t,
        "dt": sequence.dt,
        "frame_id": sequence.frame_id,
        "bad_channel_mask": sequence.bad_channel_mask.astype(np.uint8),
        "measurement_weights": sequence.measurement_weights,
        "frequency_hz": sequence.frequency_hz,
    }
    metadata = {
        "artifact_format": "hdf5",
        "package_role": "dynamic_measurement_sequence",
        "data_type": sequence.data_type,
        "n_frames": sequence.n_frames,
        "n_measurements": sequence.n_measurements,
        "sampling_rate_hz": float(sequence.sampling_rate_hz),
        "reference_policy": sequence.reference_policy,
        "stim_meas_signature": sequence.stim_meas_signature,
        "measurement_weight_kind": sequence.measurement_weight_kind,
        "context_metadata": dict(sequence.context_metadata),
        "metadata": dict(sequence.metadata),
    }
    return write_hdf5_artifact(
        path,
        arrays,
        metadata,
        schema=DYNAMIC_MEASUREMENT_SEQUENCE_SCHEMA,
        compression=compression,
    )


def read_dynamic_measurement_sequence(path: str | Path) -> DynamicMeasurementSequence:
    """Read a dynamic measurement sequence HDF5 package."""

    artifact = read_hdf5_artifact(path)
    if artifact.schema != DYNAMIC_MEASUREMENT_SEQUENCE_SCHEMA:
        raise ValueError(
            "HDF5 package schema mismatch: "
            f"{artifact.schema!r} != {DYNAMIC_MEASUREMENT_SEQUENCE_SCHEMA!r}."
        )
    arrays = artifact.arrays
    metadata = artifact.metadata
    required = {
        "frames",
        "t",
        "dt",
        "frame_id",
        "bad_channel_mask",
        "measurement_weights",
        "frequency_hz",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise KeyError(f"dynamic sequence package missing arrays: {', '.join(missing)}")
    weight_kind = str(metadata.get("measurement_weight_kind", "identity"))
    weights = None if weight_kind == "identity" else arrays["measurement_weights"]
    return DynamicMeasurementSequence.from_arrays(
        arrays["frames"],
        t=arrays["t"],
        dt=arrays["dt"],
        sampling_rate_hz=float(metadata.get("sampling_rate_hz", 0.0)),
        frame_id=arrays["frame_id"],
        reference_policy=str(metadata.get("reference_policy", "none")),
        stim_meas_signature=str(metadata.get("stim_meas_signature", "unspecified")),
        bad_channel_mask=np.asarray(arrays["bad_channel_mask"], dtype=bool),
        measurement_weights=weights,
        frequency_hz=arrays["frequency_hz"],
        context_metadata=metadata.get("context_metadata", {}),
        metadata=metadata.get("metadata", {}),
        data_type=str(metadata.get("data_type", "real")),
    )


def _frame_batch(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("frames must be a 1D vector or 2D frame batch.")
    if 0 in arr.shape:
        raise ValueError("frames must be non-empty.")
    if not all_finite_values(arr):
        raise FloatingPointError("frames contain non-finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _sampling_rate(value: float | None) -> float:
    if value is None:
        return 0.0
    rate = float(value)
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("sampling_rate_hz must be finite and non-negative.")
    return rate


def _timestamps(
    value: Any | None,
    *,
    n_frames: int,
    sampling_rate_hz: float,
) -> np.ndarray:
    if value is None:
        if sampling_rate_hz > 0.0:
            values = np.arange(n_frames, dtype=np.float64) / sampling_rate_hz
        else:
            values = np.zeros(n_frames, dtype=np.float64)
    else:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size != int(n_frames):
        raise ValueError(f"t length {values.size} does not match {n_frames}.")
    if not all_finite_values(values):
        raise FloatingPointError("t contains non-finite values.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _dt(
    value: Any | None,
    *,
    timestamps: np.ndarray,
    sampling_rate_hz: float,
) -> np.ndarray:
    n_frames = int(timestamps.size)
    if value is None:
        if n_frames == 1:
            values = np.zeros(1, dtype=np.float64)
        else:
            values = np.empty(n_frames, dtype=np.float64)
            values[0] = 0.0
            values[1:] = np.diff(timestamps)
    else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            values = np.full(n_frames, float(arr), dtype=np.float64)
        else:
            values = arr.reshape(-1)
    if values.size != n_frames:
        raise ValueError(f"dt length {values.size} does not match {n_frames}.")
    if not all_finite_values(values):
        raise FloatingPointError("dt contains non-finite values.")
    if values.size and float(np.min(values)) < 0.0:
        raise ValueError("dt must be non-negative.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _frame_ids(value: Any | None, *, n_frames: int) -> np.ndarray:
    if value is None:
        ids = np.arange(n_frames, dtype=np.int64)
    else:
        ids = np.asarray(value, dtype=np.int64).reshape(-1)
    if ids.size != int(n_frames):
        raise ValueError(f"frame_id length {ids.size} does not match {n_frames}.")
    return np.ascontiguousarray(ids, dtype=np.int64)


def _bad_channel_mask_frames(
    value: Any | None,
    *,
    n_frames: int,
    n_measurements: int,
) -> np.ndarray:
    if value is None:
        mask = np.zeros((n_frames, n_measurements), dtype=bool)
    else:
        arr = np.asarray(value)
        if arr.ndim == 1:
            base = normalize_bad_channel_mask(arr, n_measurements=n_measurements)
            mask = np.empty((n_frames, n_measurements), dtype=bool)
            np.copyto(mask, base.reshape(1, -1), casting="no")
        elif arr.ndim == 2 and arr.shape == (n_frames, n_measurements):
            mask = arr.astype(bool, copy=False)
        else:
            raise ValueError(
                "bad_channel_mask must be length-n or frame-aligned n_frames-by-n."
            )
    return np.ascontiguousarray(mask, dtype=bool)


def _measurement_weights(
    value: Any | None,
    *,
    n_frames: int,
    n_measurements: int,
) -> tuple[np.ndarray, str]:
    if value is None:
        return np.ones(n_measurements, dtype=np.float64), "identity"
    arr = np.asarray(value, dtype=np.float64)
    if not all_finite_values(arr):
        raise FloatingPointError("measurement_weights contain non-finite values.")
    if arr.size and float(np.min(arr)) < 0.0:
        raise ValueError("measurement_weights entries must be non-negative.")
    if arr.ndim == 1:
        if arr.size != n_measurements:
            raise ValueError(
                f"measurement_weights length {arr.size} does not match {n_measurements}."
            )
        return np.ascontiguousarray(arr, dtype=np.float64), "diagonal"
    if arr.ndim == 2:
        if arr.shape == (n_frames, n_measurements):
            return np.ascontiguousarray(arr, dtype=np.float64), "diagonal_per_frame"
        if arr.shape == (n_measurements, n_measurements):
            _validate_symmetric(arr, name="measurement_weights")
            return np.ascontiguousarray(arr, dtype=np.float64), "full"
    if arr.ndim == 3 and arr.shape == (n_frames, n_measurements, n_measurements):
        for idx, matrix in enumerate(arr):
            _validate_symmetric(matrix, name=f"measurement_weights[{idx}]")
        return np.ascontiguousarray(arr, dtype=np.float64), "full_per_frame"
    raise ValueError(
        "measurement_weights must be diagonal, frame-diagonal, full, or frame-full."
    )


def _validate_symmetric(matrix: np.ndarray, *, name: str) -> None:
    if not np.allclose(matrix, matrix.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(f"{name} matrix must be symmetric.")


def _frequency_hz(value: Any | None, *, n_frames: int) -> np.ndarray:
    if value is None:
        values = np.zeros(n_frames, dtype=np.float64)
    else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            values = np.full(n_frames, float(arr), dtype=np.float64)
        else:
            values = arr.reshape(-1)
    if values.size != int(n_frames):
        raise ValueError(
            f"frequency_hz length {values.size} does not match {n_frames}."
        )
    if not all_finite_values(values):
        raise FloatingPointError("frequency_hz contains non-finite values.")
    if values.size and float(np.min(values)) < 0.0:
        raise ValueError("frequency_hz must be non-negative.")
    return np.ascontiguousarray(values, dtype=np.float64)


def _nonempty_string(value: str, *, name: str) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must be non-empty.")
    return resolved


def _readonly(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values)
    out.setflags(write=False)
    return out


__all__ = [
    "DYNAMIC_MEASUREMENT_SEQUENCE_SCHEMA",
    "DynamicMeasurementSequence",
    "read_dynamic_measurement_sequence",
    "write_dynamic_measurement_sequence",
]
