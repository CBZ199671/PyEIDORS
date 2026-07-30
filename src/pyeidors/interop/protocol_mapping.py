"""Provable Bridge v3 hardware/model protocol and current mapping."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


def _real_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if np.iscomplexobj(matrix) and not np.allclose(np.imag(matrix), 0.0):
        raise ValueError(f"{name} must be real")
    matrix = np.asarray(np.real(matrix), dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(matrix)


def _real_measurement_matrices(
    values: Sequence[Any],
    *,
    name: str,
    n_stim: int,
    n_elec: int,
) -> tuple[np.ndarray, ...]:
    if len(values) != n_stim:
        raise ValueError(f"{name} must contain one matrix per stimulation")
    matrices: list[np.ndarray] = []
    for index, value in enumerate(values):
        matrix = _real_matrix(value, name=f"{name}[{index}]")
        if matrix.shape[1] != n_elec:
            raise ValueError(f"{name}[{index}] electrode width mismatch")
        matrices.append(matrix)
    return tuple(matrices)


def _proportional_scale(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> float | None:
    denominator = float(np.dot(reference, reference))
    if denominator <= atol**2:
        raise ValueError("Protocol rows must not be zero")
    scale = float(np.dot(candidate, reference) / denominator)
    if not np.isfinite(scale) or abs(scale) <= atol:
        return None
    if not np.allclose(candidate, scale * reference, rtol=rtol, atol=atol):
        return None
    return scale


def _runtime_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_value(metadata: Mapping[str, Any], key: str) -> Any | None:
    if key not in metadata:
        return None
    value = metadata[key]
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return value


@dataclass(frozen=True)
class ActualCurrentResolution:
    """Actual stimulation physics selected by explicit metadata priority."""

    source: str
    stim_matrix: np.ndarray
    row_scales: tuple[float, ...]
    runtime_physics_hash: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ActualCurrentResolution:
        """Restore and integrity-check recorded actual-current metadata."""

        payload = dict(value)
        if payload.get("schema") != "pyeidors_actual_current_resolution_v3":
            raise ValueError("Unsupported actual-current resolution schema")
        source = str(payload.get("source", "")).strip()
        if source not in {"frame", "session", "device"}:
            raise ValueError("Actual-current source must be frame, session, or device")
        stim_matrix = _real_matrix(
            payload.get("stim_matrix", ()),
            name="actual_current.stim_matrix",
        )
        row_scales = tuple(float(item) for item in payload.get("row_scales", ()))
        if stim_matrix.shape[0] != len(row_scales) or not row_scales:
            raise ValueError("Actual-current stimulation/scaling lengths differ")
        if not np.all(np.isfinite(row_scales)) or any(
            abs(scale) <= 1.0e-12 for scale in row_scales
        ):
            raise ValueError("Actual-current scales must be finite and non-zero")
        proof_payload = {
            "source": source,
            "stim_matrix": stim_matrix.tolist(),
            "row_scales": list(row_scales),
        }
        runtime_physics_hash = str(payload.get("runtime_physics_hash", ""))
        if runtime_physics_hash != _runtime_hash(proof_payload):
            raise ValueError("Actual-current resolution fingerprint mismatch")
        return cls(
            source=source,
            stim_matrix=stim_matrix,
            row_scales=row_scales,
            runtime_physics_hash=runtime_physics_hash,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "pyeidors_actual_current_resolution_v3",
            "source": self.source,
            "stim_matrix": self.stim_matrix.tolist(),
            "row_scales": list(self.row_scales),
            "runtime_physics_hash": self.runtime_physics_hash,
        }


@dataclass(frozen=True)
class ProtocolChannelMapping:
    """Unique hardware-to-model stimulation and channel mapping proof."""

    stimulation_permutation: tuple[int, ...]
    stimulation_scales: tuple[float, ...]
    channel_permutation: tuple[int, ...]
    channel_signs: tuple[int, ...]
    runtime_stim_matrix: np.ndarray
    runtime_fingerprint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ProtocolChannelMapping:
        """Restore and integrity-check a persisted mapping proof."""

        payload = dict(value)
        if payload.get("schema") != "pyeidors_protocol_channel_mapping_v3":
            raise ValueError("Unsupported protocol channel mapping schema")
        stimulation_permutation = tuple(
            int(item) for item in payload.get("stimulation_permutation", ())
        )
        stimulation_scales = tuple(
            float(item) for item in payload.get("stimulation_scales", ())
        )
        channel_permutation = tuple(
            int(item) for item in payload.get("channel_permutation", ())
        )
        channel_signs = tuple(int(item) for item in payload.get("channel_signs", ()))
        runtime_stim_matrix = _real_matrix(
            payload.get("runtime_stim_matrix", ()),
            name="runtime_stim_matrix",
        )
        if not stimulation_permutation:
            raise ValueError("Protocol mapping has no stimulation rows")
        if len(stimulation_permutation) != len(stimulation_scales):
            raise ValueError("Protocol mapping stimulation lengths differ")
        if runtime_stim_matrix.shape[0] != len(stimulation_permutation):
            raise ValueError("Protocol mapping runtime stimulation row mismatch")
        if sorted(stimulation_permutation) != list(range(len(stimulation_permutation))):
            raise ValueError("Protocol stimulation permutation is not one-to-one")
        if not channel_permutation:
            raise ValueError("Protocol mapping has no measurement channels")
        if sorted(channel_permutation) != list(range(len(channel_permutation))):
            raise ValueError("Protocol channel permutation is not one-to-one")
        if len(channel_permutation) != len(channel_signs):
            raise ValueError("Protocol mapping channel lengths differ")
        if any(sign not in {-1, 1} for sign in channel_signs):
            raise ValueError("Protocol channel signs must be +1 or -1")
        if not np.all(np.isfinite(stimulation_scales)) or any(
            abs(scale) <= 1.0e-12 for scale in stimulation_scales
        ):
            raise ValueError("Protocol stimulation scales must be finite and non-zero")

        proof_payload = {
            "stimulation_permutation": list(stimulation_permutation),
            "stimulation_scales": list(stimulation_scales),
            "channel_permutation": list(channel_permutation),
            "channel_signs": list(channel_signs),
            "runtime_stim_matrix": runtime_stim_matrix.tolist(),
        }
        runtime_fingerprint = str(payload.get("runtime_fingerprint", ""))
        if runtime_fingerprint != _runtime_hash(proof_payload):
            raise ValueError("Protocol channel mapping fingerprint mismatch")
        return cls(
            stimulation_permutation=stimulation_permutation,
            stimulation_scales=stimulation_scales,
            channel_permutation=channel_permutation,
            channel_signs=channel_signs,
            runtime_stim_matrix=runtime_stim_matrix,
            runtime_fingerprint=runtime_fingerprint,
        )

    def validate_for_model(
        self,
        *,
        model_stim_matrix: Any,
        measurement_count: int,
        rtol: float = 1.0e-9,
        atol: float = 1.0e-12,
    ) -> None:
        """Verify that a persisted proof still belongs to the loaded model."""

        model = _real_matrix(model_stim_matrix, name="model_stim_matrix")
        if self.runtime_stim_matrix.shape != model.shape:
            raise ValueError("Runtime/model stimulation matrix shapes differ")
        if len(self.stimulation_permutation) != model.shape[0]:
            raise ValueError("Runtime/model stimulation row counts differ")
        if len(self.channel_permutation) != int(measurement_count):
            raise ValueError("Runtime/model measurement channel counts differ")
        for index, (actual, reference) in enumerate(
            zip(self.runtime_stim_matrix, model, strict=True)
        ):
            scale = _proportional_scale(
                actual,
                reference,
                rtol=rtol,
                atol=atol,
            )
            if scale is None or not np.isclose(
                scale,
                self.stimulation_scales[index],
                rtol=rtol,
                atol=atol,
            ):
                raise ValueError(
                    f"Runtime stimulation row {index} is not the proven "
                    "model-current scale"
                )

    def apply(self, values: Any) -> np.ndarray:
        array = np.asarray(values)
        if array.shape[-1] != len(self.channel_permutation):
            raise ValueError(
                "Hardware measurement width does not match the proven "
                f"mapping: {array.shape[-1]} != {len(self.channel_permutation)}"
            )
        mapped = np.take(
            array,
            np.asarray(self.channel_permutation, dtype=np.int64),
            axis=-1,
        )
        return mapped * np.asarray(self.channel_signs)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "pyeidors_protocol_channel_mapping_v3",
            "stimulation_permutation": list(self.stimulation_permutation),
            "stimulation_scales": list(self.stimulation_scales),
            "channel_permutation": list(self.channel_permutation),
            "channel_signs": list(self.channel_signs),
            "runtime_stim_matrix": self.runtime_stim_matrix.tolist(),
            "runtime_fingerprint": self.runtime_fingerprint,
            "proof": "unique_exact_stimulation_and_measurement_rows",
        }


def resolve_actual_stimulation(
    model_stim_matrix: Any,
    *,
    frame_metadata: Mapping[str, Any] | None = None,
    session_metadata: Mapping[str, Any] | None = None,
    device_config: Mapping[str, Any] | None = None,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
) -> ActualCurrentResolution:
    """Resolve actual current using frame > session > device priority."""

    model = _real_matrix(model_stim_matrix, name="model_stim_matrix")
    sources = (
        ("frame", dict(frame_metadata or {})),
        ("session", dict(session_metadata or {})),
        ("device", dict(device_config or {})),
    )
    matrix_keys = (
        "actual_stim_matrix",
        "effective_stim_matrix",
        "stim_matrix_effective",
        "stim_matrix",
    )
    amp_keys = (
        ("actual_current_a", 1.0),
        ("stim_current_a", 1.0),
        ("injection_current_a", 1.0),
        ("stim_amp_uA", 1.0e-6),
    )
    for source_name, metadata in sources:
        for key in matrix_keys:
            value = _metadata_value(metadata, key)
            if value is None:
                continue
            actual = _real_matrix(
                value,
                name=f"{source_name}.{key}",
            )
            if actual.shape != model.shape:
                raise ValueError(
                    f"{source_name}.{key} shape does not match model stimulation"
                )
            break
        else:
            actual = None
            for key, unit_scale in amp_keys:
                value = _metadata_value(metadata, key)
                if value is None:
                    continue
                current = np.asarray(value, dtype=np.float64).reshape(-1)
                if current.size == 1:
                    current = np.repeat(current, model.shape[0])
                if current.size != model.shape[0]:
                    raise ValueError(
                        f"{source_name}.{key} must be scalar or per-stimulation"
                    )
                current = current * unit_scale
                positive = np.sum(np.maximum(model, 0.0), axis=1)
                if np.any(positive <= atol):
                    raise ValueError(
                        "Model stimulation rows need positive injected current"
                    )
                actual = model * (current / positive)[:, None]
                break
        if actual is None:
            continue

        scales: list[float] = []
        for index, (candidate, reference) in enumerate(zip(actual, model, strict=True)):
            scale = _proportional_scale(
                candidate,
                reference,
                rtol=rtol,
                atol=atol,
            )
            if scale is None:
                raise ValueError(
                    f"{source_name} stimulation row {index} is not a finite "
                    "non-zero real multiple of the model row"
                )
            scales.append(scale)
        payload = {
            "source": source_name,
            "stim_matrix": actual.tolist(),
            "row_scales": scales,
        }
        return ActualCurrentResolution(
            source=source_name,
            stim_matrix=actual,
            row_scales=tuple(scales),
            runtime_physics_hash=_runtime_hash(payload),
        )
    raise ValueError(
        "Actual stimulation current is missing from frame, session, and device "
        "metadata."
    )


def prove_protocol_mapping(
    *,
    model_stim_matrix: Any,
    model_meas_matrices: Sequence[Any],
    hardware_stim_matrix: Any,
    hardware_meas_matrices: Sequence[Any],
    rtol: float = 1.0e-9,
    atol: float = 1.0e-12,
) -> ProtocolChannelMapping:
    """Prove a unique row-level hardware protocol mapping or fail closed."""

    model_stim = _real_matrix(model_stim_matrix, name="model_stim_matrix")
    hardware_stim = _real_matrix(
        hardware_stim_matrix,
        name="hardware_stim_matrix",
    )
    if hardware_stim.shape != model_stim.shape:
        raise ValueError("Hardware/model stimulation matrix shapes differ")
    n_stim, n_elec = model_stim.shape
    model_meas = _real_measurement_matrices(
        model_meas_matrices,
        name="model_meas_matrices",
        n_stim=n_stim,
        n_elec=n_elec,
    )
    hardware_meas = _real_measurement_matrices(
        hardware_meas_matrices,
        name="hardware_meas_matrices",
        n_stim=n_stim,
        n_elec=n_elec,
    )

    stimulation_permutation: list[int] = []
    stimulation_scales: list[float] = []
    for model_index, model_row in enumerate(model_stim):
        candidates: list[tuple[int, float]] = []
        for hardware_index, hardware_row in enumerate(hardware_stim):
            scale = _proportional_scale(
                hardware_row,
                model_row,
                rtol=rtol,
                atol=atol,
            )
            if scale is not None:
                candidates.append((hardware_index, scale))
        if not candidates:
            raise ValueError(
                f"Missing hardware stimulation match for model row {model_index}"
            )
        if len(candidates) != 1:
            raise ValueError(
                f"Ambiguous hardware stimulation matches for model row {model_index}"
            )
        hardware_index, scale = candidates[0]
        stimulation_permutation.append(hardware_index)
        stimulation_scales.append(scale)
    if len(set(stimulation_permutation)) != n_stim:
        raise ValueError("Hardware stimulation mapping is not one-to-one")

    hardware_offsets = np.cumsum(
        [0, *(int(matrix.shape[0]) for matrix in hardware_meas)]
    )
    channel_permutation: list[int] = []
    channel_signs: list[int] = []
    for model_stim_index, model_matrix in enumerate(model_meas):
        hardware_stim_index = stimulation_permutation[model_stim_index]
        hardware_matrix = hardware_meas[hardware_stim_index]
        if hardware_matrix.shape[0] != model_matrix.shape[0]:
            raise ValueError(
                "Hardware/model measurement counts differ for matched "
                f"stimulation {model_stim_index}"
            )
        used_rows: set[int] = set()
        for model_meas_index, model_row in enumerate(model_matrix):
            candidates: list[tuple[int, int]] = []
            for hardware_meas_index, hardware_row in enumerate(hardware_matrix):
                if np.allclose(hardware_row, model_row, rtol=rtol, atol=atol):
                    candidates.append((hardware_meas_index, 1))
                if np.allclose(hardware_row, -model_row, rtol=rtol, atol=atol):
                    candidates.append((hardware_meas_index, -1))
            if not candidates:
                raise ValueError(
                    "Missing hardware measurement match for model row "
                    f"{model_stim_index}:{model_meas_index}"
                )
            if len(candidates) != 1:
                raise ValueError(
                    "Ambiguous hardware measurement matches for model row "
                    f"{model_stim_index}:{model_meas_index}"
                )
            hardware_meas_index, sign = candidates[0]
            if hardware_meas_index in used_rows:
                raise ValueError("Hardware measurement mapping is not one-to-one")
            used_rows.add(hardware_meas_index)
            channel_permutation.append(
                int(hardware_offsets[hardware_stim_index] + hardware_meas_index)
            )
            channel_signs.append(sign)

    runtime_stim = hardware_stim[np.asarray(stimulation_permutation, dtype=np.int64)]
    payload = {
        "stimulation_permutation": stimulation_permutation,
        "stimulation_scales": stimulation_scales,
        "channel_permutation": channel_permutation,
        "channel_signs": channel_signs,
        "runtime_stim_matrix": runtime_stim.tolist(),
    }
    return ProtocolChannelMapping(
        stimulation_permutation=tuple(stimulation_permutation),
        stimulation_scales=tuple(stimulation_scales),
        channel_permutation=tuple(channel_permutation),
        channel_signs=tuple(channel_signs),
        runtime_stim_matrix=np.ascontiguousarray(runtime_stim),
        runtime_fingerprint=_runtime_hash(payload),
    )


__all__ = [
    "ActualCurrentResolution",
    "ProtocolChannelMapping",
    "prove_protocol_mapping",
    "resolve_actual_stimulation",
]
