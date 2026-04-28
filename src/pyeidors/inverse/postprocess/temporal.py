"""Temporal smoothing pipeline for RM-produced reconstruction frames."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from pyeidors.data._temporal_core import (
    as_frame_batch as _as_frame_batch,
    positive_int as _positive_int,
    unit_interval as _unit_interval,
)

from .tv import TVRefinementResult, refine_tv_pdhg


@dataclass(frozen=True)
class TemporalTVPipelineResult:
    """Postprocessed frame batch plus temporal/TV diagnostics."""

    values: np.ndarray
    metadata: MappingProxyType

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


def moving_average_frames(frames: Any, *, window: int = 3) -> np.ndarray:
    """Causal moving-average smoothing over reconstruction frames."""

    batch, _ = _as_frame_batch(frames)
    width = _positive_int(window, "window")
    out = np.empty_like(batch, dtype=np.float64)
    for idx in range(batch.shape[0]):
        start = max(0, idx + 1 - width)
        out[idx] = np.mean(batch[start : idx + 1], axis=0)
    return out


def exponential_smooth_frames(
    frames: Any,
    *,
    alpha: float = 0.5,
    initial: Any | None = None,
) -> np.ndarray:
    """Causal exponential smoothing over reconstruction frames."""

    batch, _ = _as_frame_batch(frames)
    alpha_value = _unit_interval(alpha, "alpha")
    out = np.empty_like(batch, dtype=np.float64)
    if initial is None:
        previous = batch[0].copy()
    else:
        previous = np.asarray(initial, dtype=np.float64).reshape(-1)
        if previous.size != batch.shape[1]:
            raise ValueError(
                f"initial length {previous.size} does not match {batch.shape[1]}."
            )
        if not np.isfinite(previous).all():
            raise FloatingPointError("initial contains non-finite values.")
    for idx, frame in enumerate(batch):
        current = alpha_value * frame + (1.0 - alpha_value) * previous
        out[idx] = current
        previous = current
    return out


def postprocess_rm_frames(
    frames: Any,
    mesh: Any,
    *,
    temporal: str = "none",
    moving_window: int = 3,
    exponential_alpha: float = 0.5,
    exponential_initial: Any | None = None,
    apply_tv: bool = True,
    roi_mask: Any | None = None,
    tv_weight: float = 1.0e-2,
    tv_max_iterations: int = 100,
    tv_tolerance: float = 1.0e-6,
    graph_weight: str = "unit",
    return_metadata: bool = False,
) -> np.ndarray | TemporalTVPipelineResult:
    """Apply temporal smoothing and optional ROI TV refinement to RM frames."""

    batch, was_vector = _as_frame_batch(frames)
    temporal_mode = str(temporal).strip().lower()
    if temporal_mode in {"none", "off", "identity"}:
        smoothed = batch.copy()
        temporal_mode = "none"
    elif temporal_mode in {"moving_average", "moving-average", "ma"}:
        smoothed = moving_average_frames(batch, window=moving_window)
        temporal_mode = "moving_average"
    elif temporal_mode in {"exponential", "ema", "exp"}:
        smoothed = exponential_smooth_frames(
            batch,
            alpha=exponential_alpha,
            initial=exponential_initial,
        )
        temporal_mode = "exponential"
    else:
        raise ValueError("temporal must be one of: none, moving_average, exponential.")

    if apply_tv:
        refined_rows: list[np.ndarray] = []
        tv_metadata: list[dict[str, Any]] = []
        for frame in smoothed:
            result = refine_tv_pdhg(
                frame,
                mesh,
                roi_mask=roi_mask,
                tv_weight=tv_weight,
                max_iterations=tv_max_iterations,
                tolerance=tv_tolerance,
                graph_weight=graph_weight,
                return_metadata=True,
                seed_source=f"temporal_{temporal_mode}",
            )
            assert isinstance(result, TVRefinementResult)
            refined_rows.append(np.asarray(result.values, dtype=np.float64))
            tv_metadata.append(dict(result.metadata))
        values = np.vstack(refined_rows)
    else:
        values = smoothed
        tv_metadata = []

    if was_vector:
        values_out = values.reshape(-1)
    else:
        values_out = values
    metadata = MappingProxyType(
        {
            "schema": "pyeidors-rm-postprocess-v1",
            "temporal": temporal_mode,
            "moving_window": int(moving_window),
            "exponential_alpha": float(exponential_alpha),
            "apply_tv": bool(apply_tv),
            "tv_frame_count": int(len(tv_metadata)),
            "frame_count": int(batch.shape[0]),
            "n_parameters": int(batch.shape[1]),
            "was_vector": bool(was_vector),
            "tv": tuple(tv_metadata),
        }
    )
    result = TemporalTVPipelineResult(values=values_out, metadata=metadata)
    return result if return_metadata else result.values


__all__ = [
    "TemporalTVPipelineResult",
    "exponential_smooth_frames",
    "moving_average_frames",
    "postprocess_rm_frames",
]
