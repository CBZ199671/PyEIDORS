"""PyEIDORS Data Structure Definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh, MeshTags


@dataclass
class PatternConfig:
    """Stimulation and measurement pattern configuration."""

    n_elec: int
    n_rings: int = 1
    stim_pattern: str | list[int] = "{ad}"
    meas_pattern: str | list[int] = "{ad}"
    electrode_layout: Literal["ring_major", "zigzag"] = "ring_major"
    measurement_protocol: Literal[
        "layer_local_2p5d",
        "eidors_full_3d",
        "cross_layer_full",
        "hybrid_full_3d",
        "custom",
    ] = "eidors_full_3d"
    custom_stim_matrix: Any | None = None
    custom_meas_matrices: Any | None = None
    drive_mode: Literal["line_current_density", "total_current", "normalized"] = (
        "line_current_density"
    )
    drive_value: float = 1.0
    geometry_scale_to_m: float = 1.0
    electrode_length_m_override: float | list[float] | None = None
    use_meas_current: bool = False
    use_meas_current_next: int = 0
    rotate_meas: bool = True
    stim_direction: str = "ccw"  # 'ccw' or 'cw'
    meas_direction: str = "ccw"
    stim_first_positive: bool = False


@dataclass
class EITData:
    """EIT data container."""

    meas: np.ndarray
    stim_pattern: np.ndarray
    n_elec: int
    n_stim: int
    n_meas: int
    type: str = "real"
    reference_meas: np.ndarray | None = None
    target_meas: np.ndarray | None = None
    difference_mode: str = "raw"
    difference_orientation: str = "target_minus_reference"


@dataclass
class EITImage:
    """EIT image container."""

    elem_data: np.ndarray
    fwd_model: Any
    type: str = "conductivity"
    name: str = ""

    def get_conductivity(self) -> np.ndarray:
        if self.type == "resistivity":
            return 1.0 / self.elem_data
        return self.elem_data


@dataclass
class MeshConfig:
    """Mesh configuration parameters."""

    dimension: int = 2
    radius: float = 1.0
    height: float = 1.0
    electrode_height_ratio: float = 0.2
    electrode_level_fractions: tuple[float, ...] = (0.25, 0.75)
    z_center: float = 0.0
    refinement: int = 8
    electrode_vertices: int = 8
    gap_vertices: int = 4
    mesh_size: float = 0.1
    mesh_family: str = "tetra"
    geometry_version: str = "geomv2"
    electrode_layout: str = "ring_major"
    generator_revision: str | None = None


@dataclass
class ElectrodePosition:
    """Electrode position information."""

    L: int  # Number of electrodes
    positions: list[tuple[float, float]]  # Electrode angular positions (start, end)

    @classmethod
    def create_circular(
        cls, n_elec: int = 16, radius: float = 1.0
    ) -> "ElectrodePosition":
        """Create circular electrode positions."""
        import math

        # Each electrode covers 1/4 of its angular segment
        electrode_width = 2 * math.pi / n_elec / 4

        positions = []
        for i in range(n_elec):
            center_angle = 2 * math.pi * i / n_elec
            start_angle = center_angle - electrode_width / 2
            end_angle = center_angle + electrode_width / 2
            positions.append((start_angle, end_angle))

        return cls(L=n_elec, positions=positions)


@dataclass
class EITMesh:
    """Strongly-typed mesh container for the DOLFINx-only runtime."""

    mesh: "Mesh"
    facet_tags: "MeshTags"
    cell_tags: "MeshTags | None" = None
    association_table: dict[str, int] = field(default_factory=dict)
    physical_groups: dict[str, Any] = field(default_factory=dict)
    radius: float = 0.0
    mesh_file: str | None = None
    electrode_vertices: list[np.ndarray] | None = None
    mesh_family: str | None = None
    geometry_version: str | None = None
    generator_revision: str | None = None
    structured_sidecar_file: str | None = None
    structured_sidecar_version: str | None = None

    @property
    def comm(self):
        return self.mesh.comm

    @property
    def geometry(self):
        return self.mesh.geometry

    @property
    def topology(self):
        return self.mesh.topology

    def coordinates(self) -> np.ndarray:
        """Return geometry coordinates with shape ``(n_points, gdim)``."""
        return self.mesh.geometry.x[:, : self.mesh.geometry.dim]

    def num_vertices(self) -> int:
        index_map = self.mesh.topology.index_map(0)
        return int(index_map.size_local if index_map is not None else 0)

    def num_cells(self) -> int:
        tdim = self.mesh.topology.dim
        index_map = self.mesh.topology.index_map(tdim)
        return int(index_map.size_local if index_map is not None else 0)

    def cells(self) -> np.ndarray:
        """Return local cell-to-vertex connectivity."""
        tdim = self.mesh.topology.dim
        self.mesh.topology.create_connectivity(tdim, 0)
        c2v = self.mesh.topology.connectivity(tdim, 0)
        if c2v is None:
            return np.empty((0, 0), dtype=np.int32)

        n_cells = self.num_cells()
        if n_cells == 0:
            return np.empty((0, 0), dtype=np.int32)

        first = c2v.links(0)
        verts_per_cell = len(first)
        data = np.zeros((n_cells, verts_per_cell), dtype=np.int32)
        data[0] = first
        for idx in range(1, n_cells):
            data[idx] = c2v.links(idx)
        return data

    def get_info(self) -> dict[str, Any]:
        coords = self.coordinates()
        center = (
            coords.mean(axis=0) if coords.size else np.zeros((self.mesh.geometry.dim,))
        )
        n_elec = sum(
            1
            for k in self.association_table
            if isinstance(k, str) and k.lower().startswith("electrode")
        )
        return {
            "num_vertices": self.num_vertices(),
            "num_cells": self.num_cells(),
            "num_electrodes": n_elec,
            "radius": float(self.radius),
            "center": center.tolist(),
            "association_table": dict(self.association_table),
            "mesh_file": self.mesh_file,
            "mesh_family": self.mesh_family,
            "geometry_version": self.geometry_version,
            "generator_revision": self.generator_revision,
            "structured_sidecar_file": self.structured_sidecar_file,
            "structured_sidecar_version": self.structured_sidecar_version,
        }


@dataclass
class FrameMetadata:
    """Per-frame metadata for the streaming acquisition format."""

    timestamp: str = ""
    frame_index: int = 0
    n_meas: int = 208
    frequency_hz: float = 0.0
    stim_amplitude: float = 0.0
    voltage_amplitude: float = 0.0
    board_id: int = 1
    user_id: int = 1
    transport_type: str = "serial"
    protocol_version: str = "legacy-v1"
    sampling_rate_hz: float = 0.0
    adc_params: dict[str, float] | None = None
    notes: str = ""
