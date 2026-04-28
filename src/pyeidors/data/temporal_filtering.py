"""Measurement-domain temporal filtering for online EIT frames."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ._temporal_core import (
    as_frame_batch as _as_frame_batch,
    positive_int as _positive_int,
    unit_interval as _unit_interval,
)


TemporalMeasurementHook = Callable[[np.ndarray, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class MeasurementTemporalFilterResult:
    """Filtered measurement frames plus causal filter state metadata."""

    values: np.ndarray
    metadata: MappingProxyType

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


def filter_measurement_frames(
    frames: Any,
    *,
    temporal: str = "none",
    moving_window: int = 3,
    exponential_alpha: float = 0.5,
    initial_state: Mapping[str, Any] | None = None,
    timestamps: Any | None = None,
    sample_rate_hz: float | None = None,
    hook: TemporalMeasurementHook | None = None,
    hook_kind: str | None = None,
    return_metadata: bool = False,
) -> np.ndarray | MeasurementTemporalFilterResult:
    """Causally filter raw or difference-voltage measurement frames.

    ``timestamps`` are carried through metadata only. They are never averaged,
    interpolated, or otherwise smoothed.
    """

    batch, was_vector = _as_frame_batch(frames)
    mode = _normalize_temporal_mode(temporal)
    state = _state_mapping(initial_state)
    _validate_state_mode(state, mode)
    resolved_timestamps = _timestamps(timestamps, n_frames=batch.shape[0])
    resolved_sample_rate = _optional_positive_float(
        sample_rate_hz,
        name="sample_rate_hz",
    )
    previous_count = _state_frame_count(state)

    if mode == "none":
        temporal_values = batch.copy()
        final_state = _final_state(
            mode=mode,
            frame_count=previous_count + batch.shape[0],
            last_output=temporal_values[-1],
            history_tail=np.empty((0, batch.shape[1]), dtype=np.float64),
            sample_rate_hz=resolved_sample_rate,
        )
    elif mode == "moving_average":
        temporal_values, history_tail = _moving_average(
            batch,
            window=moving_window,
            initial_state=state,
        )
        final_state = _final_state(
            mode=mode,
            frame_count=previous_count + batch.shape[0],
            last_output=temporal_values[-1],
            history_tail=history_tail,
            sample_rate_hz=resolved_sample_rate,
        )
    elif mode == "exponential":
        temporal_values = _exponential_smooth(
            batch,
            alpha=exponential_alpha,
            initial_state=state,
        )
        final_state = _final_state(
            mode=mode,
            frame_count=previous_count + batch.shape[0],
            last_output=temporal_values[-1],
            history_tail=np.empty((0, batch.shape[1]), dtype=np.float64),
            sample_rate_hz=resolved_sample_rate,
        )
    else:  # pragma: no cover - protected by _normalize_temporal_mode
        raise AssertionError(f"unexpected temporal mode: {mode}")

    values = temporal_values
    applied_hook_kind = "none"
    hook_metadata: dict[str, Any] = {}
    if hook is not None:
        applied_hook_kind = _normalize_hook_kind(hook_kind)
        context = MappingProxyType(
            {
                "temporal": mode,
                "hook_kind": applied_hook_kind,
                "frame_count": int(batch.shape[0]),
                "n_measurements": int(batch.shape[1]),
                "timestamps": resolved_timestamps,
                "sample_rate_hz": resolved_sample_rate,
                "timestamp_policy": "metadata_only_no_smoothing",
            }
        )
        values, hook_metadata = _apply_hook(
            hook,
            values,
            context=context,
            expected_shape=values.shape,
        )
    elif hook_kind is not None:
        raise ValueError("hook_kind requires a hook callable.")

    if was_vector:
        values_out = values.reshape(-1)
    else:
        values_out = values
    metadata = MappingProxyType(
        {
            "schema": "pyeidors-measurement-temporal-filter-v1",
            "temporal": mode,
            "moving_window": int(moving_window),
            "exponential_alpha": float(exponential_alpha),
            "frame_count": int(batch.shape[0]),
            "n_measurements": int(batch.shape[1]),
            "was_vector": bool(was_vector),
            "input_shape": tuple(int(v) for v in batch.shape),
            "output_shape": tuple(int(v) for v in values_out.shape),
            "initial_state_used": bool(state),
            "final_state": final_state,
            "timestamps": resolved_timestamps,
            "timestamps_present": resolved_timestamps is not None,
            "timestamp_policy": "metadata_only_no_smoothing",
            "sample_rate_hz": resolved_sample_rate,
            "hook_applied": hook is not None,
            "hook_kind": applied_hook_kind,
            "hook_metadata": MappingProxyType(hook_metadata),
        }
    )
    result = MeasurementTemporalFilterResult(
        values=np.ascontiguousarray(values_out, dtype=np.float64),
        metadata=metadata,
    )
    return result if return_metadata else result.values


def _moving_average(
    batch: np.ndarray,
    *,
    window: int,
    initial_state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    width = _positive_int(window, "moving_window")
    prior_tail = _state_history_tail(
        initial_state,
        n_measurements=batch.shape[1],
    )
    n_prior = prior_tail.shape[0]
    n_frames = batch.shape[0]
    combined = np.concatenate([prior_tail, batch], axis=0) if n_prior else batch
    n_combined = combined.shape[0]
    out = np.empty_like(batch, dtype=np.float64)
    if n_frames > 0:
        csum = np.empty((n_combined + 1, batch.shape[1]), dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(combined, axis=0, dtype=np.float64, out=csum[1:])
        indices = np.arange(n_prior, n_combined)
        starts = np.maximum(0, indices + 1 - width)
        denom = (indices + 1 - starts).astype(np.float64).reshape(-1, 1)
        out = (csum[indices + 1] - csum[starts]) / denom
    tail_count = max(width - 1, 0)
    if tail_count == 0 or n_combined == 0:
        return out, np.empty((0, batch.shape[1]), dtype=np.float64)
    return out, np.ascontiguousarray(combined[-tail_count:], dtype=np.float64)


def _exponential_smooth(
    batch: np.ndarray,
    *,
    alpha: float,
    initial_state: Mapping[str, Any],
) -> np.ndarray:
    alpha_value = _unit_interval(alpha, "exponential_alpha")
    previous = _state_last_output(initial_state, n_measurements=batch.shape[1])
    if previous is None:
        previous = batch[0].copy()
    out = np.empty_like(batch, dtype=np.float64)
    for idx, frame in enumerate(batch):
        current = alpha_value * frame + (1.0 - alpha_value) * previous
        out[idx] = current
        previous = current
    return out


def _apply_hook(
    hook: TemporalMeasurementHook,
    values: np.ndarray,
    *,
    context: Mapping[str, Any],
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = hook(values.copy(), context)
    hook_metadata: dict[str, Any] = {}
    if isinstance(raw, tuple):
        if len(raw) != 2:
            raise ValueError("temporal hook tuple result must be (frames, metadata).")
        raw_values, raw_metadata = raw
        if raw_metadata is not None:
            if not isinstance(raw_metadata, Mapping):
                raise ValueError("temporal hook metadata must be a mapping.")
            hook_metadata = dict(raw_metadata)
    else:
        raw_values = raw
    out = np.asarray(raw_values, dtype=np.float64)
    if out.shape != expected_shape:
        raise ValueError(
            f"temporal hook output shape {out.shape} does not match {expected_shape}."
        )
    if not np.isfinite(out).all():
        raise FloatingPointError("temporal hook output contains non-finite values.")
    return np.ascontiguousarray(out, dtype=np.float64), hook_metadata


def _normalize_temporal_mode(value: str | None) -> str:
    resolved = str(value or "none").strip().lower()
    aliases = {
        "off": "none",
        "identity": "none",
        "ma": "moving_average",
        "moving-average": "moving_average",
        "ema": "exponential",
        "exp": "exponential",
    }
    resolved = aliases.get(resolved, resolved)
    if resolved not in {"none", "moving_average", "exponential"}:
        raise ValueError("temporal must be one of: none, moving_average, exponential.")
    return resolved


def _normalize_hook_kind(value: str | None) -> str:
    resolved = str(value or "custom").strip().lower().replace("-", "_")
    if resolved not in {"custom", "bandpass", "lockin", "lock_in"}:
        raise ValueError("hook_kind must be one of: custom, bandpass, lockin.")
    return "lockin" if resolved == "lock_in" else resolved


def _state_mapping(initial_state: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if initial_state is None:
        return MappingProxyType({})
    if not isinstance(initial_state, Mapping):
        raise ValueError("initial_state must be a mapping.")
    return initial_state


def _validate_state_mode(state: Mapping[str, Any], mode: str) -> None:
    if not state:
        return
    state_mode = str(state.get("mode", mode)).strip().lower()
    if state_mode != mode:
        raise ValueError(f"initial_state mode {state_mode!r} does not match {mode!r}.")


def _state_frame_count(state: Mapping[str, Any]) -> int:
    if not state:
        return 0
    count = int(state.get("frame_count", 0))
    if count < 0:
        raise ValueError("initial_state frame_count must be non-negative.")
    return count


def _state_history_tail(
    state: Mapping[str, Any],
    *,
    n_measurements: int,
) -> np.ndarray:
    raw = state.get("history_tail", ()) if state else ()
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, int(n_measurements)), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != int(n_measurements):
        raise ValueError("initial_state history_tail has incompatible shape.")
    if not np.isfinite(arr).all():
        raise FloatingPointError(
            "initial_state history_tail contains non-finite values."
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _state_last_output(
    state: Mapping[str, Any],
    *,
    n_measurements: int,
) -> np.ndarray | None:
    if not state or "last_output" not in state:
        return None
    arr = np.asarray(state["last_output"], dtype=np.float64).reshape(-1)
    if arr.size != int(n_measurements):
        raise ValueError("initial_state last_output has incompatible length.")
    if not np.isfinite(arr).all():
        raise FloatingPointError(
            "initial_state last_output contains non-finite values."
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _timestamps(value: Any | None, *, n_frames: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != int(n_frames):
        raise ValueError(f"timestamps length {arr.size} does not match {n_frames}.")
    if not np.isfinite(arr).all():
        raise FloatingPointError("timestamps contain non-finite values.")
    return tuple(float(v) for v in arr)


def _final_state(
    *,
    mode: str,
    frame_count: int,
    last_output: np.ndarray,
    history_tail: np.ndarray,
    sample_rate_hz: float | None,
) -> MappingProxyType:
    return MappingProxyType(
        {
            "mode": mode,
            "frame_count": int(frame_count),
            "last_output": _tuple_vector(last_output),
            "history_tail": _tuple_rows(history_tail),
            "sample_rate_hz": sample_rate_hz,
        }
    )


def _tuple_vector(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1))


def _tuple_rows(values: np.ndarray) -> tuple[tuple[float, ...], ...]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return ()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return tuple(_tuple_vector(row) for row in arr)


def _optional_positive_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


__all__ = [
    "MeasurementTemporalFilterResult",
    "TemporalMeasurementHook",
    "filter_measurement_frames",
]
