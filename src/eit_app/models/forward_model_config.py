"""Shared forward-model configuration used by UI interop and runtime adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from eit_app.measurement_layout import measurement_layout_from_config


INTERACTIVE_3D_DEFAULT_ELECTRODES_PER_RING = 8
INTERACTIVE_3D_DEFAULT_RINGS = 2
INTERACTIVE_3D_DEFAULT_RADIUS = 0.18
INTERACTIVE_3D_DEFAULT_HEIGHT = 0.16
ELECTRODE_HEIGHT_RATIO_WINDOW_MARGIN = 0.95


def drive_mode_for_mesh_dimension(drive_mode: Any, mesh_dimension: int) -> str:
    """Return a drive mode that is valid for the selected mesh dimension."""
    raw = str(drive_mode or "").strip().lower()
    if not raw:
        return "total_current" if int(mesh_dimension) == 3 else "line_current_density"
    if int(mesh_dimension) == 3 and raw == "line_current_density":
        return "total_current"
    return raw


def electrode_level_fractions_for_rings(n_rings: int) -> tuple[float, ...]:
    """Return stable 3D electrode level fractions for the requested ring count."""
    rings = max(int(n_rings), 1)
    if rings <= 2:
        # The current 3D generator expects at least two vertical windows.
        return (0.25, 0.75)
    lo, hi = 0.15, 0.85
    step = (hi - lo) / float(rings - 1)
    return tuple(lo + step * idx for idx in range(rings))


def max_non_overlapping_electrode_height_ratio(
    electrode_level_fractions: tuple[float, ...] | list[float],
) -> float:
    """Largest GUI-safe 3D electrode height ratio for the level spacing."""

    levels = sorted(float(value) for value in electrode_level_fractions)
    if len(levels) < 2:
        return ELECTRODE_HEIGHT_RATIO_WINDOW_MARGIN
    min_gap = min(right - left for left, right in zip(levels[:-1], levels[1:]))
    return max(min(float(min_gap) * ELECTRODE_HEIGHT_RATIO_WINDOW_MARGIN, 1.0), 1e-6)


def max_electrode_height_ratio_for_rings(n_rings: int) -> float:
    """Largest GUI-safe 3D electrode height ratio for a ring count."""

    return max_non_overlapping_electrode_height_ratio(
        electrode_level_fractions_for_rings(n_rings)
    )


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


def _parse_custom_pattern_payload(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return dict(loaded) if isinstance(loaded, dict) else {}
    return {}


@dataclass
class ForwardModelConfig:
    """Portable forward-model configuration across Hardware / Simulation / Dataset."""

    mesh_dimension: int = 2
    mesh_refinement: float = 0.1
    potential_order: int = 1
    background_conductivity: float = 1.0
    noise_level: float = 0.0

    n_elec: int = 16
    n_rings: int = 1
    electrode_layout: str = "ring_major"
    measurement_protocol: str = "eidors_full_3d"
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
    electrode_coverage: float = 0.5
    electrode_area_m2_override: float | None = None
    contact_impedance: float | list[float] | None = None
    custom_pattern_json: str = ""
    custom_stim_matrix: Any | None = None
    custom_meas_matrices: Any | None = None

    radius: float = 1.0
    height: float = 1.0
    electrode_height_ratio: float = 0.2
    electrode_level_fractions: tuple[float, ...] = (0.25, 0.75)
    z_center: float = 0.0
    mesh_family: str = "tetra"
    geometry_version: str = "geomv2"

    solver_mode: str = "auto"
    line_search_mode: str = "auto"
    linear_solver: str = "auto"
    preconditioner: str = "auto"
    fast_linear_path: str = "auto"
    forward_solver_preset: str = "auto"
    forward_mat_solve: str = "auto"
    petsc_device: str = "auto"
    device: str = "auto"
    forward_backend: str = "dolfinx"
    acceleration_profile: str = "default"

    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mesh_dimension = int(self.mesh_dimension)
        self.potential_order = max(1, int(self.potential_order))
        self.drive_mode = drive_mode_for_mesh_dimension(
            self.drive_mode,
            self.mesh_dimension,
        )

    @classmethod
    def from_mapping(
        cls, mapping: dict[str, Any] | None = None
    ) -> "ForwardModelConfig":
        raw = dict(mapping or {})
        mesh_dimension = int(raw.get("mesh_dimension", raw.get("mea_mode", 2)))
        raw_drive_mode = raw.get(
            "drive_mode",
            "total_current" if mesh_dimension == 3 else "line_current_density",
        )
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
        custom_payload = _parse_custom_pattern_payload(raw.get("custom_pattern_json"))
        if not custom_payload:
            custom_payload = _parse_custom_pattern_payload(raw.get("custom_pattern"))
        custom_stim_matrix = raw.get(
            "custom_stim_matrix", custom_payload.get("stim_matrix")
        )
        custom_meas_matrices = raw.get(
            "custom_meas_matrices", custom_payload.get("meas_matrices")
        )
        layout = measurement_layout_from_config(raw)

        return cls(
            mesh_dimension=mesh_dimension,
            mesh_refinement=float(
                raw.get("mesh_refinement", raw.get("mesh_size", 0.1))
            ),
            potential_order=int(
                raw.get(
                    "potential_order",
                    raw.get(
                        "fem_potential_order",
                        raw.get("potential_degree", raw.get("p_order", 1)),
                    ),
                )
            ),
            background_conductivity=float(raw.get("background_conductivity", 1.0)),
            noise_level=float(raw.get("noise_level", 0.0)),
            n_elec=int(raw.get("n_elec", raw.get("n_electrodes", 16))),
            n_rings=int(raw.get("n_rings", 1)),
            electrode_layout=str(raw.get("electrode_layout", "ring_major"))
            .strip()
            .lower()
            or "ring_major",
            measurement_protocol=str(
                raw.get(
                    "measurement_protocol",
                    raw.get("acquisition_protocol", "eidors_full_3d"),
                )
            )
            .strip()
            .lower()
            or "eidors_full_3d",
            stim_pattern=str(raw.get("stim_pattern", "{ad}")),
            meas_pattern=str(raw.get("meas_pattern", "{ad}")),
            rotate_meas=bool(raw.get("rotate_meas", True)),
            use_meas_current=bool(raw.get("use_meas_current", False)),
            use_meas_current_next=int(raw.get("use_meas_current_next", 0)),
            stim_direction=str(raw.get("stim_direction", "ccw")),
            meas_direction=str(raw.get("meas_direction", "ccw")),
            stim_first_positive=bool(raw.get("stim_first_positive", False)),
            drive_mode=drive_mode_for_mesh_dimension(raw_drive_mode, mesh_dimension),
            drive_value=float(raw.get("drive_value", 1.0)),
            geometry_scale_to_m=float(raw.get("geometry_scale_to_m", 1.0)),
            electrode_length_m_override=(
                _to_float_list(elec_override)
                if not isinstance(elec_override, (int, float))
                else float(elec_override)
            ),
            electrode_coverage=float(layout.get("electrode_coverage", 0.5)),
            electrode_area_m2_override=(
                None
                if layout.get("electrode_area_m2_override") in (None, "")
                else float(layout.get("electrode_area_m2_override"))
            ),
            contact_impedance=(
                _to_float_list(contact_impedance)
                if not isinstance(contact_impedance, (int, float))
                else float(contact_impedance)
            ),
            custom_pattern_json=str(raw.get("custom_pattern_json", "")),
            custom_stim_matrix=custom_stim_matrix,
            custom_meas_matrices=custom_meas_matrices,
            radius=float(raw.get("radius", 1.0)),
            height=float(raw.get("height", 1.0)),
            electrode_height_ratio=float(raw.get("electrode_height_ratio", 0.2)),
            electrode_level_fractions=level_fractions,
            z_center=float(raw.get("z_center", 0.0)),
            mesh_family=str(raw.get("mesh_family", "tetra")),
            geometry_version=str(raw.get("geometry_version", "geomv2")),
            solver_mode=str(raw.get("solver_mode", "auto")),
            line_search_mode=str(raw.get("line_search_mode", "auto")),
            linear_solver=str(raw.get("linear_solver", "auto")),
            preconditioner=str(raw.get("preconditioner", "auto")),
            fast_linear_path=str(raw.get("fast_linear_path", "auto")),
            forward_solver_preset=str(raw.get("forward_solver_preset", "auto")),
            forward_mat_solve=str(raw.get("forward_mat_solve", "auto")),
            petsc_device=str(raw.get("petsc_device", "auto")),
            device=str(raw.get("device", "auto")),
            forward_backend=str(raw.get("forward_backend", "dolfinx")),
            acceleration_profile=str(raw.get("acceleration_profile", "default")),
            notes=[str(item) for item in raw.get("notes", [])],
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "mesh_dimension": int(self.mesh_dimension),
            "mesh_refinement": float(self.mesh_refinement),
            "potential_order": int(self.potential_order),
            "background_conductivity": float(self.background_conductivity),
            "noise_level": float(self.noise_level),
            "n_elec": int(self.n_elec),
            "n_rings": int(self.n_rings),
            "electrode_layout": self.electrode_layout,
            "measurement_protocol": self.measurement_protocol,
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
            "electrode_coverage": float(self.electrode_coverage),
            "electrode_area_m2_override": self.electrode_area_m2_override,
            "contact_impedance": self.contact_impedance,
            "custom_pattern_json": self.custom_pattern_json,
            "custom_stim_matrix": self.custom_stim_matrix,
            "custom_meas_matrices": self.custom_meas_matrices,
            "radius": float(self.radius),
            "height": float(self.height),
            "electrode_height_ratio": float(self.electrode_height_ratio),
            "electrode_level_fractions": list(self.electrode_level_fractions),
            "z_center": float(self.z_center),
            "mesh_family": self.mesh_family,
            "geometry_version": self.geometry_version,
            "solver_mode": self.solver_mode,
            "line_search_mode": self.line_search_mode,
            "linear_solver": self.linear_solver,
            "preconditioner": self.preconditioner,
            "fast_linear_path": self.fast_linear_path,
            "forward_solver_preset": self.forward_solver_preset,
            "forward_mat_solve": self.forward_mat_solve,
            "petsc_device": self.petsc_device,
            "device": self.device,
            "forward_backend": self.forward_backend,
            "acceleration_profile": self.acceleration_profile,
            "notes": list(self.notes),
        }

    def measurement_layout(self) -> dict[str, Any]:
        return measurement_layout_from_config(self.to_mapping())

    def point_count(self) -> int:
        return int(self.measurement_layout()["points_per_frame"])

    def total_electrodes(self) -> int:
        return max(int(self.n_elec), 1) * max(int(self.n_rings), 1)

    def display_dimension(self) -> str:
        return "3D" if int(self.mesh_dimension) == 3 else "2D"

    def summary(self) -> dict[str, str]:
        layout = self.measurement_layout()
        return {
            "dimension": self.display_dimension(),
            "electrodes": f"{int(self.n_elec)}E x {int(self.n_rings)}R",
            "layout": self.electrode_layout,
            "protocol": self.measurement_protocol,
            "patterns": f"{self.stim_pattern} / {self.meas_pattern}",
            "rotation": "rotate" if self.rotate_meas else "fixed",
            "drive_related": "include drive electrodes"
            if self.use_meas_current
            else "exclude drive electrodes",
            "extra_skip": f"+{int(self.use_meas_current_next)} extra skip",
            "points": f"{int(layout['points_per_frame'])}",
        }

    def with_overrides(self, **overrides: Any) -> "ForwardModelConfig":
        payload = self.to_mapping()
        payload.update(overrides)
        return ForwardModelConfig.from_mapping(payload)
