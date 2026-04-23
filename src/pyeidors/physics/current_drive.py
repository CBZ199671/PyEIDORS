"""Current drive semantics and unit conversion helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import numpy as np

from ..data.structures import PatternConfig

DriveMode = Literal["line_current_density", "total_current", "normalized"]
_VALID_DRIVE_MODES: tuple[DriveMode, ...] = (
    "line_current_density",
    "total_current",
    "normalized",
)


def normalize_drive_mode(mode: str) -> DriveMode:
    normalized = str(mode).strip().lower()
    if normalized not in _VALID_DRIVE_MODES:
        allowed = ", ".join(_VALID_DRIVE_MODES)
        raise ValueError(
            f"Unsupported drive_mode={mode!r}. Expected one of: {allowed}."
        )
    return normalized  # type: ignore[return-value]


def validate_drive_config(
    *,
    drive_mode: str,
    drive_value: float,
    geometry_scale_to_m: float,
    mesh_tdim: int | None = None,
) -> DriveMode:
    """Validate drive configuration values.

    Returns:
        Normalized drive mode string.
    """
    normalized_mode = normalize_drive_mode(drive_mode)
    value = float(drive_value)
    if value <= 0.0:
        raise ValueError(f"drive_value must be positive, got {drive_value!r}")

    scale = float(geometry_scale_to_m)
    if scale <= 0.0:
        raise ValueError(
            f"geometry_scale_to_m must be positive, got {geometry_scale_to_m!r}"
        )

    if (
        normalized_mode == "line_current_density"
        and mesh_tdim is not None
        and int(mesh_tdim) != 2
    ):
        raise ValueError(
            "drive_mode='line_current_density' is supported for 2D meshes only "
            f"(got topological dim={mesh_tdim})."
        )

    return normalized_mode


def normalize_pattern_config_for_mesh(
    pattern_config: PatternConfig,
    *,
    mesh_tdim: int,
) -> tuple[PatternConfig, dict[str, str]]:
    """Normalize drive-mode semantics to a mesh-compatible configuration."""

    requested_mode = normalize_drive_mode(pattern_config.drive_mode)
    effective_mode = requested_mode
    if int(mesh_tdim) == 3 and requested_mode == "line_current_density":
        effective_mode = "total_current"
    normalized = (
        pattern_config
        if effective_mode == requested_mode
        else replace(
            pattern_config,
            drive_mode=effective_mode,
        )
    )
    return normalized, {
        "drive_mode_requested": requested_mode,
        "drive_mode_effective": effective_mode,
    }


def _as_positive_array(
    values: Sequence[float], *, n_elec: int, name: str
) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != n_elec:
        raise ValueError(f"{name} length mismatch: expected {n_elec}, got {arr.size}.")
    bad_idx = np.nonzero(arr <= 0.0)[0]
    if bad_idx.size:
        first = int(bad_idx[0])
        raise ValueError(f"{name}[{first}] must be positive, got {arr[first]!r}.")
    return arr


def resolve_electrode_lengths_m(
    *,
    electrode_lengths_mesh: Sequence[float],
    geometry_scale_to_m: float,
    electrode_length_m_override: float | Sequence[float] | None,
    n_elec: int,
) -> np.ndarray:
    """Resolve per-electrode physical lengths in meters."""
    if n_elec <= 0:
        raise ValueError(f"n_elec must be positive, got {n_elec!r}.")

    scale = float(geometry_scale_to_m)
    if scale <= 0.0:
        raise ValueError(
            f"geometry_scale_to_m must be positive, got {geometry_scale_to_m!r}"
        )

    if electrode_length_m_override is None:
        mesh_lengths = _as_positive_array(
            electrode_lengths_mesh,
            n_elec=n_elec,
            name="electrode_lengths_mesh",
        )
        return mesh_lengths * scale

    if isinstance(electrode_length_m_override, (int, float)):
        value = float(electrode_length_m_override)
        if value <= 0.0:
            raise ValueError(
                "electrode_length_m_override must be positive when scalar, "
                f"got {electrode_length_m_override!r}."
            )
        return np.full(n_elec, value, dtype=float)

    return _as_positive_array(
        electrode_length_m_override,
        n_elec=n_elec,
        name="electrode_length_m_override",
    )


def build_stim_currents(
    *,
    drive_mode: str,
    drive_value: float,
    inj_indices: Sequence[int],
    inj_weights: Sequence[float],
    electrode_lengths_m: Sequence[float] | None,
) -> np.ndarray:
    """Compute stimulation currents for one injection pattern."""
    normalized_mode = normalize_drive_mode(drive_mode)
    value = float(drive_value)
    weights = np.asarray(inj_weights, dtype=float).reshape(-1)
    indices = np.asarray(inj_indices, dtype=int).reshape(-1)
    if indices.size != weights.size:
        raise ValueError(
            "inj_indices and inj_weights length mismatch: "
            f"{indices.size} vs {weights.size}."
        )

    if normalized_mode in {"total_current", "normalized"}:
        return value * weights

    if electrode_lengths_m is None:
        raise ValueError(
            "drive_mode='line_current_density' requires electrode_lengths_m "
            "to compute physical currents."
        )

    lengths = np.asarray(electrode_lengths_m, dtype=float).reshape(-1)
    if lengths.size == 0:
        raise ValueError("electrode_lengths_m cannot be empty.")

    out = np.zeros_like(weights, dtype=float)
    for i, idx in enumerate(indices):
        if idx < 0 or idx >= lengths.size:
            raise ValueError(
                f"Injection electrode index {idx} is out of range for "
                f"electrode_lengths_m size {lengths.size}."
            )
        length = float(lengths[idx])
        if length <= 0.0:
            raise ValueError(
                f"electrode_lengths_m[{idx}] must be positive, got {length!r}."
            )
        out[i] = value * length * weights[i]
    return out
