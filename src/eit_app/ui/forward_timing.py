"""Lightweight helpers for GUI forward timing metadata."""

from __future__ import annotations

from typing import Any


def _record_forward_visualization_timing(
    result: Any,
    *,
    visual_ms: float,
) -> None:
    if not isinstance(result.forward_model_config, dict):
        return
    elapsed = max(0.0, float(visual_ms))
    timings = dict(result.forward_model_config.get("forward_timing_ms") or {})
    timings["gui_visualization_update"] = elapsed
    phase_order = list(
        result.forward_model_config.get("forward_timing_phase_order") or []
    )
    if "gui_visualization_update" not in phase_order:
        phase_order.append("gui_visualization_update")
    result.forward_model_config["forward_timing_ms"] = timings
    result.forward_model_config["forward_timing_phase_order"] = phase_order
    result.forward_model_config["gui_forward_visualization_update_ms"] = elapsed
