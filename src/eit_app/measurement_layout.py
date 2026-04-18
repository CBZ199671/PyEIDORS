"""Helpers for estimating and normalizing EIT measurement layouts."""

from __future__ import annotations

import math
import json
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_MEASUREMENT_LAYOUT: dict[str, Any] = {
    "n_elec": 16,
    "n_rings": 1,
    "stim_pattern": "{ad}",
    "meas_pattern": "{ad}",
    "use_meas_current": False,
    "use_meas_current_next": 0,
    "rotate_meas": True,
    "stim_direction": "ccw",
    "meas_direction": "ccw",
    "stim_first_positive": False,
    "electrode_layout": "ring_major",
    "measurement_protocol": "eidors_full_3d",
    "custom_pattern_json": "",
    "custom_stim_matrix": None,
    "custom_meas_matrices": None,
    "radius": 1.0,
    "geometry_scale_to_m": 1.0,
    "electrode_coverage": 0.5,
    "electrode_length_m_override": None,
    "contact_impedance": 0.01,
}


def _is_adjacent_pattern(pattern: str | Sequence[int]) -> bool:
    return isinstance(pattern, str) and pattern.strip().lower() == "{ad}"


def _coerce_scalar_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return float(default)
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        return _coerce_scalar_float(value[0], default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_custom_pattern_payload(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value.strip())
        except (TypeError, json.JSONDecodeError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _resolve_electrode_length_override(
    source: Mapping[str, Any],
    *,
    n_elec: int,
    explicit_length: bool,
    explicit_coverage: bool,
) -> float:
    if explicit_length:
        return max(
            _coerce_scalar_float(source.get("electrode_length_m_override"), math.pi / 16.0),
            1e-9,
        )

    coverage = max(
        _coerce_scalar_float(
            source.get("electrode_coverage"),
            0.5 if explicit_coverage else DEFAULT_MEASUREMENT_LAYOUT["electrode_coverage"],
        ),
        1e-6,
    )
    radius = max(_coerce_scalar_float(source.get("radius"), 1.0), 1e-9)
    geometry_scale_to_m = max(_coerce_scalar_float(source.get("geometry_scale_to_m"), 1.0), 1e-9)
    circumference_m = 2.0 * math.pi * radius * geometry_scale_to_m
    return max(circumference_m * coverage / max(int(n_elec), 1), 1e-9)


def _resolve_electrode_coverage(
    source: Mapping[str, Any],
    *,
    n_elec: int,
    electrode_length_m_override: float,
    explicit_length: bool,
    explicit_coverage: bool,
) -> float:
    if explicit_length:
        radius = max(_coerce_scalar_float(source.get("radius"), 1.0), 1e-9)
        geometry_scale_to_m = max(_coerce_scalar_float(source.get("geometry_scale_to_m"), 1.0), 1e-9)
        pitch_m = 2.0 * math.pi * radius * geometry_scale_to_m / max(int(n_elec), 1)
        if pitch_m <= 0.0:
            return 0.5
        coverage = float(electrode_length_m_override) / pitch_m
        return min(max(coverage, 1e-6), 1.0)

    if explicit_coverage:
        return min(max(_coerce_scalar_float(source["electrode_coverage"], 0.5), 1e-6), 1.0)

    radius = max(_coerce_scalar_float(source.get("radius"), 1.0), 1e-9)
    geometry_scale_to_m = max(_coerce_scalar_float(source.get("geometry_scale_to_m"), 1.0), 1e-9)
    pitch_m = 2.0 * math.pi * radius * geometry_scale_to_m / max(int(n_elec), 1)
    if pitch_m <= 0.0:
        return 0.5
    coverage = float(electrode_length_m_override) / pitch_m
    return min(max(coverage, 1e-6), 1.0)


def estimate_measurement_point_count(
    *,
    n_electrodes: int,
    stim_pattern: str | Sequence[int] = "{ad}",
    meas_pattern: str | Sequence[int] = "{ad}",
    n_rings: int = 1,
    use_meas_current: bool = False,
    use_meas_current_next: int = 0,
    rotate_meas: bool = True,
    stim_direction: str = "ccw",
    meas_direction: str = "ccw",
    stim_first_positive: bool = False,
    electrode_layout: str = "ring_major",
    measurement_protocol: str = "eidors_full_3d",
    custom_stim_matrix: Any | None = None,
    custom_meas_matrices: Any | None = None,
) -> int:
    """Estimate boundary-voltage sample count for the configured pattern layout."""
    total_electrodes = max(int(n_electrodes), 1) * max(int(n_rings), 1)
    protocol = str(measurement_protocol or "eidors_full_3d").strip().lower().replace("-", "_")
    if (
        protocol in {"eidors_full_3d", "full_3d", "true_3d"}
        and
        _is_adjacent_pattern(stim_pattern)
        and _is_adjacent_pattern(meas_pattern)
        and not bool(use_meas_current)
    ):
        # The current hardware excludes the driven pair plus the paired adjacent
        # measurement sample, which yields 13*16 = 208 for the default board.
        excluded = 3 + 2 * max(int(use_meas_current_next), 0)
        return max(total_electrodes * max(total_electrodes - excluded, 1), 1)

    try:
        from pyeidors.data.structures import PatternConfig
        from pyeidors.electrodes.patterns import StimMeasPatternManager

        pattern = PatternConfig(
            n_elec=max(int(n_electrodes), 1),
            n_rings=max(int(n_rings), 1),
            stim_pattern=stim_pattern,
            meas_pattern=meas_pattern,
            electrode_layout=str(electrode_layout or "ring_major"),
            measurement_protocol=str(measurement_protocol or "eidors_full_3d"),
            custom_stim_matrix=custom_stim_matrix,
            custom_meas_matrices=custom_meas_matrices,
            use_meas_current=use_meas_current,
            use_meas_current_next=max(int(use_meas_current_next), 0),
            rotate_meas=rotate_meas,
            stim_direction=stim_direction,
            meas_direction=meas_direction,
            stim_first_positive=stim_first_positive,
        )
        manager = StimMeasPatternManager(pattern)
        return max(int(manager.n_meas_total), 1)
    except Exception:
        if use_meas_current:
            return total_electrodes * total_electrodes
        excluded = 3 + 2 * max(int(use_meas_current_next), 0)
        return max(total_electrodes * max(total_electrodes - excluded, 1), 1)


def measurement_layout_from_config(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize measurement-layout keys and derive the expected point count.

    The hardware path still defaults to the current 16-electrode adjacent layout,
    but callers can override these values to prepare simulation, storage, and
    future transports for different acquisition geometries.
    """
    raw = dict(config or {})
    source = dict(DEFAULT_MEASUREMENT_LAYOUT)
    source.update(raw)

    if "n_elec" in raw:
        n_elec = int(raw["n_elec"])
    elif "n_electrodes" in raw:
        n_elec = int(raw["n_electrodes"])
    else:
        n_elec = int(source.get("n_elec", 16))
    n_rings = int(raw.get("n_rings", source.get("n_rings", 1)))
    mesh_dimension = int(
        raw.get("mesh_dimension", raw.get("mea_mode", source.get("mesh_dimension", 2)))
    )
    electrode_layout = str(source.get("electrode_layout", "ring_major")).strip().lower()
    electrodes_per_circumference = max(n_elec, 1)
    if mesh_dimension == 3 and n_rings > 1 and electrode_layout == "zigzag":
        electrodes_per_circumference = max(n_elec, 1) * max(n_rings, 1)
    radius = max(_coerce_scalar_float(source.get("radius"), 1.0), 1e-9)
    geometry_scale_to_m = max(
        _coerce_scalar_float(source.get("geometry_scale_to_m"), 1.0),
        1e-9,
    )
    explicit_length = (
        "electrode_length_m_override" in raw
        and raw["electrode_length_m_override"] not in (None, "")
    )
    explicit_coverage = "electrode_coverage" in raw and raw["electrode_coverage"] not in (None, "")
    electrode_length_m_override = _resolve_electrode_length_override(
        source,
        n_elec=electrodes_per_circumference,
        explicit_length=explicit_length,
        explicit_coverage=explicit_coverage,
    )
    electrode_coverage = _resolve_electrode_coverage(
        source,
        n_elec=electrodes_per_circumference,
        electrode_length_m_override=electrode_length_m_override,
        explicit_length=explicit_length,
        explicit_coverage=explicit_coverage,
    )
    layout = {
        "n_elec": max(n_elec, 1),
        "n_rings": max(n_rings, 1),
        "stim_pattern": source.get("stim_pattern", "{ad}"),
        "meas_pattern": source.get("meas_pattern", "{ad}"),
        "electrode_layout": str(source.get("electrode_layout", "ring_major")),
        "measurement_protocol": str(source.get("measurement_protocol", "eidors_full_3d")),
        "custom_pattern_json": str(source.get("custom_pattern_json", "")),
        "custom_stim_matrix": source.get("custom_stim_matrix"),
        "custom_meas_matrices": source.get("custom_meas_matrices"),
        "use_meas_current": bool(source.get("use_meas_current", False)),
        "use_meas_current_next": max(int(source.get("use_meas_current_next", 0)), 0),
        "rotate_meas": bool(source.get("rotate_meas", True)),
        "stim_direction": str(source.get("stim_direction", "ccw")),
        "meas_direction": str(source.get("meas_direction", "ccw")),
        "stim_first_positive": bool(source.get("stim_first_positive", False)),
        "radius": radius,
        "geometry_scale_to_m": geometry_scale_to_m,
        "electrode_length_m_override": electrode_length_m_override,
        "electrode_coverage": electrode_coverage,
        "contact_impedance": max(_coerce_scalar_float(source.get("contact_impedance"), 0.01), 0.0),
    }
    explicit_points = int(raw.get("points_per_frame_override", 0) or 0)
    custom_payload = _parse_custom_pattern_payload(layout["custom_pattern_json"])
    custom_stim_matrix = layout["custom_stim_matrix"]
    custom_meas_matrices = layout["custom_meas_matrices"]
    if custom_stim_matrix is None:
        custom_stim_matrix = custom_payload.get("stim_matrix")
    if custom_meas_matrices is None:
        custom_meas_matrices = custom_payload.get("meas_matrices")
    layout["points_per_frame"] = (
        explicit_points
        if explicit_points > 0
        else estimate_measurement_point_count(
            n_electrodes=layout["n_elec"],
            stim_pattern=layout["stim_pattern"],
            meas_pattern=layout["meas_pattern"],
            n_rings=layout["n_rings"],
            use_meas_current=layout["use_meas_current"],
            use_meas_current_next=layout["use_meas_current_next"],
            rotate_meas=layout["rotate_meas"],
            stim_direction=layout["stim_direction"],
            meas_direction=layout["meas_direction"],
            stim_first_positive=layout["stim_first_positive"],
            electrode_layout=layout["electrode_layout"],
            measurement_protocol=layout["measurement_protocol"],
            custom_stim_matrix=custom_stim_matrix,
            custom_meas_matrices=custom_meas_matrices,
        )
    )
    layout["total_electrodes"] = layout["n_elec"] * layout["n_rings"]
    return layout
