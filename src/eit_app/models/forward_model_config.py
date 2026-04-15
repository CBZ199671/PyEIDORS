"""Shared forward-model configuration used by UI interop and runtime adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eit_app.measurement_layout import measurement_layout_from_config


def _to_float_list(values: Any) -> list[float] | None:
    if values is None or values == "":
        return None
    if isinstance(values, (int, float)):
        return [float(values)]
    if isinstance(values, str):
        parts = [part.strip() for part in values.replace(";", ",").split(",")]
        floats = [float(part) for part in parts if part]
        return floats or None
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return None


@dataclass
class ForwardModelConfig:
    """Portable forward-model configuration across Hardware / Simulation / Dataset."""

    mesh_dimension: int = 2
    mesh_refinement: float = 0.1
    background_conductivity: float = 1.0
    noise_level: float = 0.0

    n_elec: int = 16
    n_rings: int = 1
    stim_pattern: str = "{ad}"
    meas_pattern: str = "{ad}"
    rotate_meas: bool = True
    use_meas_current: bool = False
    use_meas_current_next: int = 0
    stim_direction: str = "ccw"
    meas_direction: str = "ccw"
    stim_first_positive: bool = False

    drive_mode: str = "line_current_density"
    drive_value: float = 1.0
    geometry_scale_to_m: float = 1.0
    electrode_length_m_override: float | list[float] | None = None
    contact_impedance: float | list[float] | None = None

    radius: float = 1.0
    height: float = 1.0
    electrode_height_ratio: float = 0.2
    electrode_level_fractions: tuple[float, ...] = (0.25, 0.75)
    z_center: float = 0.0
    mesh_family: str = "tetra"
    geometry_version: str = "geomv2"

    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None = None) -> "ForwardModelConfig":
        raw = dict(mapping or {})
        level_fractions = raw.get("electrode_level_fractions", (0.25, 0.75))
        if isinstance(level_fractions, str):
            parsed = _to_float_list(level_fractions)
            level_fractions = tuple(parsed or [0.25, 0.75])
        elif isinstance(level_fractions, (list, tuple)):
            level_fractions = tuple(float(value) for value in level_fractions)
        else:
            level_fractions = (0.25, 0.75)

        elec_override = raw.get("electrode_length_m_override")
        contact_impedance = raw.get("contact_impedance")

        return cls(
            mesh_dimension=int(raw.get("mesh_dimension", raw.get("mea_mode", 2))),
            mesh_refinement=float(raw.get("mesh_refinement", raw.get("mesh_size", 0.1))),
            background_conductivity=float(raw.get("background_conductivity", 1.0)),
            noise_level=float(raw.get("noise_level", 0.0)),
            n_elec=int(raw.get("n_elec", raw.get("n_electrodes", 16))),
            n_rings=int(raw.get("n_rings", 1)),
            stim_pattern=str(raw.get("stim_pattern", "{ad}")),
            meas_pattern=str(raw.get("meas_pattern", "{ad}")),
            rotate_meas=bool(raw.get("rotate_meas", True)),
            use_meas_current=bool(raw.get("use_meas_current", False)),
            use_meas_current_next=int(raw.get("use_meas_current_next", 0)),
            stim_direction=str(raw.get("stim_direction", "ccw")),
            meas_direction=str(raw.get("meas_direction", "ccw")),
            stim_first_positive=bool(raw.get("stim_first_positive", False)),
            drive_mode=str(raw.get("drive_mode", "line_current_density")),
            drive_value=float(raw.get("drive_value", 1.0)),
            geometry_scale_to_m=float(raw.get("geometry_scale_to_m", 1.0)),
            electrode_length_m_override=(
                _to_float_list(elec_override)
                if not isinstance(elec_override, (int, float))
                else float(elec_override)
            ),
            contact_impedance=(
                _to_float_list(contact_impedance)
                if not isinstance(contact_impedance, (int, float))
                else float(contact_impedance)
            ),
            radius=float(raw.get("radius", 1.0)),
            height=float(raw.get("height", 1.0)),
            electrode_height_ratio=float(raw.get("electrode_height_ratio", 0.2)),
            electrode_level_fractions=level_fractions,
            z_center=float(raw.get("z_center", 0.0)),
            mesh_family=str(raw.get("mesh_family", "tetra")),
            geometry_version=str(raw.get("geometry_version", "geomv2")),
            notes=[str(item) for item in raw.get("notes", [])],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "mesh_dimension": int(self.mesh_dimension),
            "mesh_refinement": float(self.mesh_refinement),
            "background_conductivity": float(self.background_conductivity),
            "noise_level": float(self.noise_level),
            "n_elec": int(self.n_elec),
            "n_rings": int(self.n_rings),
            "stim_pattern": self.stim_pattern,
            "meas_pattern": self.meas_pattern,
            "rotate_meas": bool(self.rotate_meas),
            "use_meas_current": bool(self.use_meas_current),
            "use_meas_current_next": int(self.use_meas_current_next),
            "stim_direction": self.stim_direction,
            "meas_direction": self.meas_direction,
            "stim_first_positive": bool(self.stim_first_positive),
            "drive_mode": self.drive_mode,
            "drive_value": float(self.drive_value),
            "geometry_scale_to_m": float(self.geometry_scale_to_m),
            "electrode_length_m_override": self.electrode_length_m_override,
            "contact_impedance": self.contact_impedance,
            "radius": float(self.radius),
            "height": float(self.height),
            "electrode_height_ratio": float(self.electrode_height_ratio),
            "electrode_level_fractions": list(self.electrode_level_fractions),
            "z_center": float(self.z_center),
            "mesh_family": self.mesh_family,
            "geometry_version": self.geometry_version,
            "notes": list(self.notes),
        }

    def measurement_layout(self) -> dict[str, Any]:
        return measurement_layout_from_config(self.to_mapping())

    def point_count(self) -> int:
        return int(self.measurement_layout()["points_per_frame"])

    def display_dimension(self) -> str:
        return "3D" if int(self.mesh_dimension) == 3 else "2D"

    def summary(self) -> dict[str, str]:
        layout = self.measurement_layout()
        return {
            "dimension": self.display_dimension(),
            "electrodes": f"{int(self.n_elec)}E x {int(self.n_rings)}R",
            "patterns": f"{self.stim_pattern} / {self.meas_pattern}",
            "rotation": "rotate" if self.rotate_meas else "fixed",
            "drive_related": "include drive electrodes" if self.use_meas_current else "exclude drive electrodes",
            "extra_skip": f"+{int(self.use_meas_current_next)} extra skip",
            "points": f"{int(layout['points_per_frame'])}",
        }

    def with_overrides(self, **overrides: Any) -> "ForwardModelConfig":
        payload = self.to_mapping()
        payload.update(overrides)
        return ForwardModelConfig.from_mapping(payload)
