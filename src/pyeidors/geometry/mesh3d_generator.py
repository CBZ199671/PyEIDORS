"""3D cylindrical EIT mesh generation for tetra and tensor-product hex paths."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from math import atan2, cos, pi, sin
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from ..data.structures import EITMesh
from ..electrodes.layout import (
    ELECTRODE_LAYOUT_RING_MAJOR,
    ELECTRODE_LAYOUT_ZIGZAG,
    normalize_electrode_layout,
)
from ..femx import build_eit_mesh, estimate_radius
from ..perf.policy import (
    DEFAULT_3D_GENERATOR_REVISION,
    DEFAULT_3D_GEOMETRY_VERSION,
    DEFAULT_MESH_FAMILY,
    LEGACY_3D_GENERATOR_REVISION,
    SQUARE_TO_DISK_3D_GENERATOR_REVISION,
    normalize_mesh_family,
)
from ._helpers import association_from_mesh_data, write_association_table

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional in isolated unit stubs
    import gmsh

    GMSH_AVAILABLE = True
except ImportError:  # pragma: no cover
    gmsh = None  # type: ignore[assignment]
    GMSH_AVAILABLE = False

try:  # pragma: no cover - optional in some unit stubs
    import meshio

    MESHIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    meshio = None  # type: ignore[assignment]
    MESHIO_AVAILABLE = False


STRUCTURED_SIDECAR_VERSION = "cuda-structured-v1"
DEFAULT_ZIGZAG_LEVEL_FRACTIONS = (0.25, 0.75)


def normalize_electrode_level_fractions(
    value: Sequence[float] | float | None,
    *,
    default: Sequence[float] = DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
) -> tuple[float, ...]:
    if value is None:
        seq = tuple(float(v) for v in default)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        seq = (float(value),)
    else:
        seq = tuple(float(v) for v in value)
    if not seq:
        raise ValueError("electrode_level_fractions must contain at least one entry")
    for frac in seq:
        if not 0.0 < float(frac) < 1.0:
            raise ValueError(
                "electrode_level_fractions entries must be in (0, 1), "
                f"got {frac!r}."
            )
    return seq


@dataclass
class Cylinder3DMeshConfig:
    """Configuration for cylindrical 3D EIT mesh generation."""

    radius: float = 1.0
    height: float = 1.0
    z_center: float = 0.0
    refinement: int = 8
    electrode_vertices: int = 6
    gap_vertices: int = 1
    electrode_height_ratio: float = 0.2
    electrode_level_fractions: Tuple[float, ...] = DEFAULT_ZIGZAG_LEVEL_FRACTIONS
    electrode_layout: str = ELECTRODE_LAYOUT_RING_MAJOR

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius!r}")
        if self.height <= 0.0:
            raise ValueError(f"height must be positive, got {self.height!r}")
        if self.refinement <= 0:
            raise ValueError(f"refinement must be positive, got {self.refinement!r}")
        if self.electrode_vertices < 2:
            raise ValueError(
                f"electrode_vertices must be >= 2, got {self.electrode_vertices!r}"
            )
        if self.gap_vertices < 0:
            raise ValueError(f"gap_vertices must be >= 0, got {self.gap_vertices!r}")
        if not 0.0 < self.electrode_height_ratio <= 1.0:
            raise ValueError(
                "electrode_height_ratio must be in (0, 1], "
                f"got {self.electrode_height_ratio!r}"
            )
        self.electrode_level_fractions = normalize_electrode_level_fractions(
            self.electrode_level_fractions,
            default=DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
        )
        self.electrode_layout = normalize_electrode_layout(self.electrode_layout)
        if len(self.electrode_level_fractions) < 2:
            raise ValueError(
                "3D cylindrical meshes require at least two electrode_level_fractions "
                "entries for multi-layer electrode placement."
            )
        sorted_windows = sorted(_electrode_vertical_windows(self), key=lambda w: w[0])
        if any(
            right[0] - left[1] <= 1e-10
            for left, right in zip(sorted_windows[:-1], sorted_windows[1:])
        ):
            raise ValueError(
                "electrode windows overlap; reduce electrode_height_ratio or "
                "separate electrode_level_fractions further."
            )

    @property
    def mesh_size(self) -> float:
        return self.radius / (self.refinement * 2.0)

    @property
    def z_min(self) -> float:
        return float(self.z_center - 0.5 * self.height)

    @property
    def z_max(self) -> float:
        return float(self.z_center + 0.5 * self.height)


@dataclass
class ElectrodeArcConfig:
    """Electrode arc placement around cylinder sidewall."""

    n_elec: int
    coverage: float = 0.5
    rotation: float = 0.0
    anticlockwise: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.n_elec, int) or self.n_elec <= 0:
            raise ValueError("n_elec must be a positive integer")
        if not 0.0 < self.coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")

    @property
    def positions(self) -> List[Tuple[float, float]]:
        electrode_size = 2 * pi / self.n_elec * self.coverage
        gap_size = 2 * pi / self.n_elec * (1.0 - self.coverage)

        first_center = pi / 2 + self.rotation
        first_start = first_center - electrode_size / 2

        positions: List[Tuple[float, float]] = []
        for i in range(self.n_elec):
            step = i * (electrode_size + gap_size)
            start = first_start + step
            end = start + electrode_size
            positions.append((start, end))

        if not self.anticlockwise:
            positions[1:] = positions[1:][::-1]
        return positions


def _normalize_angle(theta: float) -> float:
    value = float(theta) % (2.0 * pi)
    return value if value >= 0.0 else value + 2.0 * pi


def _angle_in_arc(theta: float, start: float, end: float, *, tol: float = 1e-10) -> bool:
    theta_n = _normalize_angle(theta)
    start_n = _normalize_angle(start)
    end_n = _normalize_angle(end)
    if end_n < start_n:
        end_n += 2.0 * pi
        if theta_n < start_n:
            theta_n += 2.0 * pi
    return (start_n - tol) <= theta_n <= (end_n + tol)


def _classify_theta(theta: float, positions: Sequence[Tuple[float, float]]) -> int | None:
    for idx, (start, end) in enumerate(positions, start=1):
        if _angle_in_arc(theta, start, end):
            return int(idx)
    return None




def structured_sidecar_path_for_mesh(mesh_file: str | Path) -> Path:
    mesh_path = Path(mesh_file)
    return mesh_path.with_name(f"{mesh_path.stem}_structured.json")


def validate_structured_sidecar_payload(payload: dict) -> dict:
    required = {
        "version",
        "mesh_family",
        "geometry_version",
        "generator_revision",
        "block_topology",
        "blocks",
        "structured_node_to_mesh_node",
        "structured_cell_to_block",
        "structured_cell_local_ijk",
        "boundary_faces",
        "field_tags",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"structured sidecar missing required keys: {missing}")
    if str(payload.get("version")).strip().lower() != STRUCTURED_SIDECAR_VERSION:
        raise ValueError(
            "structured sidecar version mismatch: "
            f"expected {STRUCTURED_SIDECAR_VERSION!r}, got {payload.get('version')!r}"
        )
    if str(payload.get("mesh_family")).strip().lower() != "hex":
        raise ValueError("structured sidecar currently supports mesh_family='hex' only")
    if str(payload.get("geometry_version")).strip().lower() != "geomv2":
        raise ValueError("structured sidecar currently supports geometry_version='geomv2' only")
    if not isinstance(payload.get("blocks"), list) or not payload["blocks"]:
        raise ValueError("structured sidecar must include at least one block")
    if not isinstance(payload.get("structured_node_to_mesh_node"), list) or not payload["structured_node_to_mesh_node"]:
        raise ValueError("structured sidecar must include structured_node_to_mesh_node entries")
    if not isinstance(payload.get("structured_cell_to_block"), list) or not payload["structured_cell_to_block"]:
        raise ValueError("structured sidecar must include structured_cell_to_block entries")
    if not isinstance(payload.get("structured_cell_local_ijk"), list):
        raise ValueError("structured sidecar must include structured_cell_local_ijk entries")
    if len(payload["structured_cell_to_block"]) != len(payload["structured_cell_local_ijk"]):
        raise ValueError(
            "structured sidecar cell metadata length mismatch between "
            "structured_cell_to_block and structured_cell_local_ijk"
        )
    if not isinstance(payload.get("boundary_faces"), list):
        raise ValueError("structured sidecar must include boundary_faces as a list")
    if not isinstance(payload.get("field_tags"), dict) or not payload["field_tags"]:
        raise ValueError("structured sidecar must include non-empty field_tags")
    return payload


def load_structured_sidecar(sidecar_path: str | Path) -> dict:
    path = Path(sidecar_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_structured_sidecar_payload(payload)


def _write_structured_sidecar(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_structured_sidecar_payload(payload)
    path.write_text(json.dumps(validated, indent=2), encoding="utf-8")


def _prepare_output_paths(
    *,
    output_dir: Optional[Path],
    mesh_name: Optional[str],
    prefix: str,
) -> tuple[Path, str, Path, Path]:
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if mesh_name is None:
        mesh_name = f"{prefix}_{int(time.time() * 1e6) % 1000000}"

    msh_path = output_dir / f"{mesh_name}.msh"
    assoc_path = output_dir / f"{mesh_name}_association_table.ini"
    return output_dir, mesh_name, msh_path, assoc_path


def _electrode_vertical_windows(config: Cylinder3DMeshConfig) -> list[tuple[float, float]]:
    half_height = 0.5 * config.height * config.electrode_height_ratio
    windows: list[tuple[float, float]] = []
    for frac in config.electrode_level_fractions:
        center = config.z_min + frac * config.height
        z_lower = max(config.z_min, center - half_height)
        z_upper = min(config.z_max, center + half_height)
        if z_upper - z_lower <= 1e-10:
            raise ValueError(
                "Resolved electrode window collapsed; adjust "
                "electrode_height_ratio or electrode_level_fractions."
            )
        windows.append((z_lower, z_upper))
    return windows


def _window_contains(
    z_value: float,
    window: tuple[float, float],
    *,
    tol: float = 1e-10,
) -> bool:
    return window[0] - tol <= z_value <= window[1] + tol


def _find_electrode_window_index(
    z_value: float,
    config: Cylinder3DMeshConfig,
) -> int | None:
    for idx, window in enumerate(_electrode_vertical_windows(config)):
        if _window_contains(z_value, window):
            return idx
    return None


def _total_3d_electrode_count(
    *,
    config: Cylinder3DMeshConfig,
    electrodes: ElectrodeArcConfig,
) -> int:
    """Return the number of physical CEM electrode tags on the 3D mesh."""
    if normalize_electrode_layout(config.electrode_layout) == ELECTRODE_LAYOUT_RING_MAJOR:
        return int(electrodes.n_elec) * len(_electrode_vertical_windows(config))
    return int(electrodes.n_elec)


def _ring_major_electrode_index(
    *,
    window_idx: int,
    electrode_idx: int,
    electrodes_per_ring: int,
) -> int:
    return int(window_idx) * int(electrodes_per_ring) + int(electrode_idx)


def _build_z_stage_breakpoints(config: Cylinder3DMeshConfig) -> list[float]:
    points = [config.z_min, config.z_max]
    for z_lower, z_upper in _electrode_vertical_windows(config):
        points.extend([z_lower, z_upper])
    ordered = sorted(points)
    unique: list[float] = []
    for value in ordered:
        if not unique or abs(value - unique[-1]) > 1e-10:
            unique.append(value)
    return unique


def _z_stage_intervals(config: Cylinder3DMeshConfig) -> list[tuple[float, float]]:
    breaks = _build_z_stage_breakpoints(config)
    return [
        (start, stop)
        for start, stop in zip(breaks[:-1], breaks[1:])
        if stop - start > 1e-12
    ]


def _classify_sidewall_patch(
    *,
    theta: float,
    z_center: float,
    positions: Sequence[tuple[float, float]],
    config: Cylinder3DMeshConfig,
) -> tuple[str, int | None]:
    window_idx = _find_electrode_window_index(z_center, config)
    if window_idx is None:
        return "blank_side", None

    electrode_idx = _classify_theta(theta, positions)
    if electrode_idx is None:
        return "gaps", None

    layout = normalize_electrode_layout(config.electrode_layout)
    if layout == ELECTRODE_LAYOUT_RING_MAJOR:
        return (
            "electrode",
            _ring_major_electrode_index(
                window_idx=window_idx,
                electrode_idx=int(electrode_idx),
                electrodes_per_ring=len(positions),
            ),
        )

    if window_idx == (electrode_idx - 1) % len(_electrode_vertical_windows(config)):
        return "electrode", int(electrode_idx)
    return "gaps", None


def _top_surface_from_extrusion(extruded: Sequence[tuple[int, int]]) -> int:
    candidates = [int(tag) for dim, tag in extruded if int(dim) == 2]
    if not candidates:
        raise RuntimeError("Expected a top surface from extrusion, but none were returned.")
    top_tag = int(candidates[0])
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, top_tag)
    top_z = float(0.5 * (zmin + zmax))
    top_span = float(zmax - zmin)
    for surf_tag in candidates[1:]:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)
        z_span = float(zmax - zmin)
        z_mid = float(0.5 * (zmin + zmax))
        if z_mid > top_z + 1e-10 or (abs(z_mid - top_z) <= 1e-10 and z_span < top_span):
            top_z = z_mid
            top_tag = int(surf_tag)
            top_span = z_span
    return top_tag


def _lateral_surfaces_from_extrusion(extruded: Sequence[tuple[int, int]], top_surface: int) -> list[int]:
    surfaces: list[int] = []
    for dim, tag in extruded:
        if int(dim) != 2 or int(tag) == int(top_surface):
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, int(tag))
        if float(zmax - zmin) > 1e-8:
            surfaces.append(int(tag))
    return surfaces


class _LegacyTetraCylinder3DMeshGenerator:
    """Preserve the pre-geomv2 tetra generator for explicit legacy use."""

    def __init__(self, config: Cylinder3DMeshConfig, electrodes: ElectrodeArcConfig) -> None:
        self.config = config
        self.electrodes = electrodes

    def generate(
        self,
        *,
        output_dir: Optional[Path] = None,
        mesh_name: Optional[str] = None,
    ) -> EITMesh:
        if not GMSH_AVAILABLE:
            raise ImportError("gmsh Python bindings are required to generate 3D meshes.")

        _, mesh_name, msh_path, assoc_path = _prepare_output_paths(
            output_dir=output_dir,
            mesh_name=mesh_name,
            prefix="mesh3d",
        )

        initialized_here = False
        if not gmsh.isInitialized():
            gmsh.initialize()
            initialized_here = True
        gmsh.clear()
        gmsh.model.add(mesh_name)

        try:
            geometry = self._create_geometry()
            self._set_physical_groups(geometry)
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)
            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_path))
            mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3)
        finally:
            if initialized_here:
                gmsh.finalize()
            else:
                gmsh.clear()

        association_table = association_from_mesh_data(mesh_data)
        write_association_table(assoc_path, association_table)

        return build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=estimate_radius(mesh_data.mesh),
            mesh_file=str(msh_path),
            mesh_family="tetra",
            geometry_version="legacy",
            generator_revision=LEGACY_3D_GENERATOR_REVISION,
        )

    def _create_geometry(self) -> dict[str, object]:
        r = float(self.config.radius)
        z0 = float(self.config.z_min)
        h = float(self.config.height)
        positions = self.electrodes.positions

        boundary_points: List[int] = []
        electrode_ranges: List[Tuple[int, int]] = []
        n_in = int(self.config.electrode_vertices)
        n_out = int(self.config.gap_vertices)

        for i, (start, end) in enumerate(positions):
            start_idx = len(boundary_points)
            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                boundary_points.append(gmsh.model.occ.addPoint(x, y, z0))
            electrode_ranges.append((start_idx, len(boundary_points) - 1))

            if i < len(positions) - 1:
                gap_start = end
                gap_end = positions[i + 1][0]
            else:
                gap_start = end
                gap_end = positions[0][0] + 2 * pi

            if n_out > 0:
                for theta in np.linspace(gap_start, gap_end, n_out + 2)[1:-1]:
                    x, y = r * cos(theta), r * sin(theta)
                    boundary_points.append(gmsh.model.occ.addPoint(x, y, z0))

        lines: List[int] = []
        for i in range(len(boundary_points)):
            j = (i + 1) % len(boundary_points)
            lines.append(gmsh.model.occ.addLine(boundary_points[i], boundary_points[j]))

        loop = gmsh.model.occ.addCurveLoop(lines)
        base_surface = gmsh.model.occ.addPlaneSurface([loop])
        extruded = gmsh.model.occ.extrude([(2, base_surface)], 0.0, 0.0, h)
        gmsh.model.occ.synchronize()

        volume_tags = [tag for dim, tag in extruded if dim == 3]
        if not volume_tags:
            raise RuntimeError("Failed to create 3D volume during extrusion.")

        side_surfaces = self._resolve_side_surfaces(lines, h)
        side_by_line = self._map_side_surfaces_to_lines(side_surfaces, lines)

        return {
            "volume_tag": int(volume_tags[0]),
            "lines": lines,
            "side_surfaces": side_surfaces,
            "side_by_line": side_by_line,
            "electrode_ranges": electrode_ranges,
        }

    def _resolve_side_surfaces(self, lines: Sequence[int], height: float) -> List[int]:
        line_set = set(int(line) for line in lines)
        side_surfaces: List[int] = []
        for _, surf_tag in gmsh.model.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)
            if (zmax - zmin) < max(1e-9, 0.5 * height):
                continue
            boundary = gmsh.model.getBoundary([(2, int(surf_tag))], oriented=False, recursive=False)
            if any(int(dim) == 1 and int(tag) in line_set for dim, tag in boundary):
                side_surfaces.append(int(surf_tag))
        return sorted(set(side_surfaces))

    def _map_side_surfaces_to_lines(
        self,
        side_surfaces: Sequence[int],
        lines: Sequence[int],
    ) -> dict[int, int]:
        line_set = set(int(line) for line in lines)
        side_by_line: dict[int, int] = {}
        for surf_tag in side_surfaces:
            boundary = gmsh.model.getBoundary([(2, int(surf_tag))], oriented=False, recursive=False)
            for dim, tag in boundary:
                if int(dim) != 1:
                    continue
                line_tag = int(tag)
                if line_tag not in line_set:
                    continue
                side_by_line[line_tag] = int(surf_tag)
        return side_by_line

    def _set_physical_groups(self, geometry: dict[str, object]) -> None:
        n_elec = int(self.electrodes.n_elec)
        volume_tag = int(geometry["volume_tag"])
        lines: List[int] = list(geometry["lines"])  # type: ignore[assignment]
        electrode_ranges: List[Tuple[int, int]] = list(geometry["electrode_ranges"])  # type: ignore[assignment]
        side_surfaces: List[int] = list(geometry["side_surfaces"])  # type: ignore[assignment]
        side_by_line: dict[int, int] = dict(geometry["side_by_line"])  # type: ignore[assignment]

        gmsh.model.addPhysicalGroup(3, [volume_tag], 1, name="domain")

        used_surfaces: set[int] = set()
        for i, (start, end) in enumerate(electrode_ranges):
            electrode_surfaces: List[int] = []
            for j in range(start, end):
                line_tag = int(lines[j % len(lines)])
                surf_tag = side_by_line.get(line_tag)
                if surf_tag is not None:
                    electrode_surfaces.append(surf_tag)
            electrode_surfaces = sorted(set(electrode_surfaces))
            if not electrode_surfaces:
                raise RuntimeError(f"Failed to resolve side surfaces for electrode_{i + 1}.")
            gmsh.model.addPhysicalGroup(
                2,
                electrode_surfaces,
                i + 2,
                name=f"electrode_{i + 1}",
            )
            used_surfaces.update(electrode_surfaces)

        gap_surfaces = sorted(set(side_surfaces) - used_surfaces)
        if gap_surfaces:
            gmsh.model.addPhysicalGroup(2, gap_surfaces, n_elec + 2, name="gaps")


class _GeomV2TetraCylinder3DMeshGenerator:
    """Generate a tetra cylinder with finite-height sidewall electrode patches."""

    def __init__(
        self,
        config: Cylinder3DMeshConfig,
        electrodes: ElectrodeArcConfig,
        *,
        generator_revision: str = DEFAULT_3D_GENERATOR_REVISION,
    ) -> None:
        self.config = config
        self.electrodes = electrodes
        self.generator_revision = (
            str(generator_revision).strip().lower() or DEFAULT_3D_GENERATOR_REVISION
        )

    def _create_base_surface(self) -> tuple[int, list[int]]:
        r = float(self.config.radius)
        z0 = float(self.config.z_min)
        positions = self.electrodes.positions
        n_in = int(self.config.electrode_vertices)
        n_out = int(self.config.gap_vertices)

        boundary_points: list[int] = []
        for i, (start, end) in enumerate(positions):
            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                boundary_points.append(gmsh.model.occ.addPoint(x, y, z0))

            if i < len(positions) - 1:
                gap_start = end
                gap_end = positions[i + 1][0]
            else:
                gap_start = end
                gap_end = positions[0][0] + 2 * pi

            if n_out > 0:
                for theta in np.linspace(gap_start, gap_end, n_out + 2)[1:-1]:
                    x, y = r * cos(theta), r * sin(theta)
                    boundary_points.append(gmsh.model.occ.addPoint(x, y, z0))

        lines: list[int] = []
        for idx in range(len(boundary_points)):
            nxt = (idx + 1) % len(boundary_points)
            lines.append(gmsh.model.occ.addLine(boundary_points[idx], boundary_points[nxt]))

        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])
        return int(surface), lines

    def generate(
        self,
        *,
        output_dir: Optional[Path] = None,
        mesh_name: Optional[str] = None,
    ) -> EITMesh:
        if not GMSH_AVAILABLE:
            raise ImportError("gmsh Python bindings are required to generate 3D meshes.")

        _, mesh_name, msh_path, assoc_path = _prepare_output_paths(
            output_dir=output_dir,
            mesh_name=mesh_name,
            prefix="mesh3d",
        )

        initialized_here = False
        if not gmsh.isInitialized():
            gmsh.initialize()
            initialized_here = True
        gmsh.clear()
        gmsh.model.add(mesh_name)

        all_volumes: list[int] = []
        electrode_candidate_surfaces: list[int] = []

        try:
            current_surface, _ = self._create_base_surface()
            for z_start, z_stop in _z_stage_intervals(self.config):
                delta_z = float(z_stop - z_start)
                if delta_z <= 1e-12:
                    continue
                extruded = gmsh.model.occ.extrude([(2, current_surface)], 0.0, 0.0, float(delta_z))
                gmsh.model.occ.synchronize()
                top_surface = _top_surface_from_extrusion(extruded)
                lateral_surfaces = _lateral_surfaces_from_extrusion(extruded, top_surface)
                all_volumes.extend(int(tag) for dim, tag in extruded if int(dim) == 3)
                if _find_electrode_window_index(0.5 * float(z_start + z_stop), self.config) is not None:
                    electrode_candidate_surfaces.extend(lateral_surfaces)
                current_surface = int(top_surface)

            if not all_volumes:
                raise RuntimeError("Failed to create 3D geomv2 tetra volume.")

            gmsh.model.addPhysicalGroup(3, sorted(set(all_volumes)), 1, name="domain")

            total_electrodes = _total_3d_electrode_count(
                config=self.config,
                electrodes=self.electrodes,
            )
            groups: dict[int, list[int]] = {idx: [] for idx in range(1, total_electrodes + 1)}
            gap_surfaces: list[int] = []
            positions = self.electrodes.positions
            for surf_tag in sorted(set(electrode_candidate_surfaces)):
                com = gmsh.model.occ.getCenterOfMass(2, int(surf_tag))
                theta = atan2(float(com[1]), float(com[0]))
                face_kind, electrode_idx = _classify_sidewall_patch(
                    theta=theta,
                    z_center=float(com[2]),
                    positions=positions,
                    config=self.config,
                )
                if face_kind != "electrode" or electrode_idx is None:
                    gap_surfaces.append(int(surf_tag))
                else:
                    groups[int(electrode_idx)].append(int(surf_tag))

            for idx in range(1, total_electrodes + 1):
                surfaces = sorted(set(groups[idx]))
                if not surfaces:
                    raise RuntimeError(f"Failed to resolve geomv2 side surfaces for electrode_{idx}.")
                gmsh.model.addPhysicalGroup(2, surfaces, idx + 1, name=f"electrode_{idx}")

            if gap_surfaces:
                gmsh.model.addPhysicalGroup(
                    2,
                    sorted(set(gap_surfaces)),
                    total_electrodes + 2,
                    name="gaps",
                )

            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)
            gmsh.model.mesh.generate(3)
            gmsh.write(str(msh_path))
            mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3)
        finally:
            if initialized_here:
                gmsh.finalize()
            else:
                gmsh.clear()

        association_table = association_from_mesh_data(mesh_data)
        write_association_table(assoc_path, association_table)

        return build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=estimate_radius(mesh_data.mesh),
            mesh_file=str(msh_path),
            mesh_family="tetra",
            geometry_version="geomv2",
            generator_revision=self.generator_revision,
        )


def _square_to_disk(u: float, v: float) -> tuple[float, float]:
    if abs(u) < 1e-14 and abs(v) < 1e-14:
        return 0.0, 0.0
    if abs(u) > abs(v):
        r = float(u)
        phi = (pi / 4.0) * (float(v) / float(u))
    else:
        r = float(v)
        phi = (pi / 2.0) - (pi / 4.0) * (float(u) / float(v))
    return float(r * cos(phi)), float(r * sin(phi))


class _GeomV2HexCylinder3DMeshGenerator:
    """Generate a structured tensor-product hex cylinder and write gmsh22 via meshio."""

    def __init__(
        self,
        config: Cylinder3DMeshConfig,
        electrodes: ElectrodeArcConfig,
        *,
        generator_revision: str = DEFAULT_3D_GENERATOR_REVISION,
    ) -> None:
        self.config = config
        self.electrodes = electrodes
        self.generator_revision = str(generator_revision).strip().lower() or DEFAULT_3D_GENERATOR_REVISION

    def _z_levels(self) -> np.ndarray:
        intervals = _z_stage_intervals(self.config)
        total_layers = max(len(intervals), max(6, self.config.refinement * 3))
        total_height = max(self.config.height, 1e-12)
        counts = [
            max(1, int(round(total_layers * (z_stop - z_start) / total_height)))
            for z_start, z_stop in intervals
        ]
        while sum(counts) > total_layers:
            idx = int(np.argmax(np.asarray(counts, dtype=np.int32)))
            counts[idx] -= 1
        while sum(counts) < total_layers:
            interval_lengths = np.asarray(
                [z_stop - z_start for z_start, z_stop in intervals],
                dtype=np.float64,
            )
            counts[int(np.argmax(interval_lengths))] += 1

        levels = [intervals[0][0]]
        for (z_start, z_stop), n_interval in zip(intervals, counts):
            segment = np.linspace(z_start, z_stop, n_interval + 1, dtype=np.float64)[1:]
            levels.extend(segment.tolist())
        return np.asarray(levels, dtype=np.float64)

    def _structured_geometry_square_to_disk(self) -> tuple[np.ndarray, np.ndarray, dict]:
        n_side = max(8, self.electrodes.n_elec * max(1, self.config.refinement))
        x_grid = np.linspace(-1.0, 1.0, n_side + 1, dtype=np.float64)
        y_grid = np.linspace(-1.0, 1.0, n_side + 1, dtype=np.float64)
        z_grid = self._z_levels()

        points: list[list[float]] = []
        index: dict[tuple[int, int, int], int] = {}
        for k, z_val in enumerate(z_grid):
            for j, y_val in enumerate(y_grid):
                for i, x_val in enumerate(x_grid):
                    xd, yd = _square_to_disk(x_val, y_val)
                    r = self.config.radius
                    points.append([r * xd, r * yd, float(z_val)])
                    index[(i, j, k)] = len(points) - 1

        cells: list[list[int]] = []
        for k in range(len(z_grid) - 1):
            for j in range(len(y_grid) - 1):
                for i in range(len(x_grid) - 1):
                    cells.append(
                        [
                            index[(i, j, k)],
                            index[(i + 1, j, k)],
                            index[(i + 1, j + 1, k)],
                            index[(i, j + 1, k)],
                            index[(i, j, k + 1)],
                            index[(i + 1, j, k + 1)],
                            index[(i + 1, j + 1, k + 1)],
                            index[(i, j + 1, k + 1)],
                        ]
                    )
        metadata = {
            "block_topology": ["square_to_disk"],
            "blocks": [
                {
                    "id": 0,
                    "name": "square_to_disk",
                    "logical_cells": [n_side, n_side, len(z_grid) - 1],
                    "logical_nodes": [n_side + 1, n_side + 1, len(z_grid)],
                }
            ],
            "structured_node_to_mesh_node": list(range(len(points))),
            "structured_cell_to_block": [0] * len(cells),
            "structured_cell_local_ijk": [
                [i, j, k]
                for k in range(len(z_grid) - 1)
                for j in range(len(y_grid) - 1)
                for i in range(len(x_grid) - 1)
            ],
            "z_levels": z_grid.tolist(),
        }
        return np.asarray(points, dtype=np.float64), np.asarray(cells, dtype=np.int32), metadata

    @staticmethod
    def _signed_quad_area(points2d: np.ndarray) -> float:
        x = np.asarray(points2d[:, 0], dtype=np.float64)
        y = np.asarray(points2d[:, 1], dtype=np.float64)
        return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

    def _structured_geometry_o_grid(self) -> tuple[np.ndarray, np.ndarray, dict]:
        refinement_i = max(1, self.config.refinement)
        n_core = max(8, 4 * refinement_i + 4)
        n_ring = max(6, 2 * refinement_i + 4)
        core_half = 0.45

        base_points: list[list[float]] = []
        base_index: dict[tuple[float, float], int] = {}
        base_quads: list[list[int]] = []
        base_quad_block: list[int] = []
        base_quad_local_ij: list[list[int]] = []
        blocks: list[dict[str, object]] = []

        def _add_base_point(x: float, y: float) -> int:
            key = (round(x, 12), round(y, 12))
            if key in base_index:
                return base_index[key]
            base_index[key] = len(base_points)
            base_points.append([x, y])
            return base_index[key]

        def _add_block(name: str, x_grid: np.ndarray, y_grid: np.ndarray) -> int:
            ids = np.zeros(x_grid.shape, dtype=np.int32)
            for j in range(x_grid.shape[0]):
                for i in range(x_grid.shape[1]):
                    ids[j, i] = _add_base_point(float(x_grid[j, i]), float(y_grid[j, i]))
            block_id = len(blocks)
            blocks.append(
                {
                    "id": block_id,
                    "name": str(name),
                    "logical_cells": [int(x_grid.shape[1] - 1), int(x_grid.shape[0] - 1), 0],
                    "logical_nodes": [int(x_grid.shape[1]), int(x_grid.shape[0]), 0],
                }
            )
            for j in range(x_grid.shape[0] - 1):
                for i in range(x_grid.shape[1] - 1):
                    quad = [
                        int(ids[j, i]),
                        int(ids[j, i + 1]),
                        int(ids[j + 1, i + 1]),
                        int(ids[j + 1, i]),
                    ]
                    coords = np.asarray([base_points[idx] for idx in quad], dtype=np.float64)
                    if self._signed_quad_area(coords) < 0.0:
                        quad = [quad[0], quad[3], quad[2], quad[1]]
                    base_quads.append(quad)
                    base_quad_block.append(block_id)
                    base_quad_local_ij.append([int(i), int(j)])
            return block_id

        core_axis = np.linspace(-core_half, core_half, n_core + 1, dtype=np.float64)
        x_core, y_core = np.meshgrid(core_axis, core_axis, indexing="xy")
        _add_block("core", x_core, y_core)

        eta = np.linspace(-1.0, 1.0, n_core + 1, dtype=np.float64)
        rho = np.linspace(0.0, 1.0, n_ring + 1, dtype=np.float64)

        def _add_ring_block(name: str, inner_fn, angle_fn) -> None:
            x_grid = np.zeros((eta.size, rho.size), dtype=np.float64)
            y_grid = np.zeros((eta.size, rho.size), dtype=np.float64)
            for j, t_val in enumerate(eta):
                inner_x, inner_y = inner_fn(float(t_val))
                theta = float(angle_fn(float(t_val)))
                outer_x = float(cos(theta))
                outer_y = float(sin(theta))
                for i, s_val in enumerate(rho):
                    x_grid[j, i] = (1.0 - float(s_val)) * inner_x + float(s_val) * outer_x
                    y_grid[j, i] = (1.0 - float(s_val)) * inner_y + float(s_val) * outer_y
            _add_block(name, x_grid, y_grid)

        _add_ring_block(
            "east",
            lambda t: (core_half, core_half * t),
            lambda t: (pi / 4.0) * t,
        )
        _add_ring_block(
            "north",
            lambda t: (-core_half * t, core_half),
            lambda t: (pi / 2.0) + (pi / 4.0) * t,
        )
        _add_ring_block(
            "west",
            lambda t: (-core_half, -core_half * t),
            lambda t: pi + (pi / 4.0) * t,
        )
        _add_ring_block(
            "south",
            lambda t: (core_half * t, -core_half),
            lambda t: (-pi / 2.0) + (pi / 4.0) * t,
        )

        base_points_arr = np.asarray(base_points, dtype=np.float64)
        base_points_arr *= float(self.config.radius)
        base_quads_arr = np.asarray(base_quads, dtype=np.int32)

        z_grid = self._z_levels()
        points: list[list[float]] = []
        for z_val in z_grid:
            for x_val, y_val in base_points_arr:
                points.append([float(x_val), float(y_val), float(z_val)])

        n_base = base_points_arr.shape[0]
        hexes: list[list[int]] = []
        cell_blocks: list[int] = []
        cell_local_ijk: list[list[int]] = []
        for k in range(len(z_grid) - 1):
            offset0 = k * n_base
            offset1 = (k + 1) * n_base
            for quad_index, quad in enumerate(base_quads_arr):
                v0, v1, v2, v3 = [int(v) for v in quad]
                hexes.append(
                    [
                        offset0 + v0,
                        offset0 + v1,
                        offset0 + v2,
                        offset0 + v3,
                        offset1 + v0,
                        offset1 + v1,
                        offset1 + v2,
                        offset1 + v3,
                    ]
                )
                cell_blocks.append(base_quad_block[quad_index])
                i_local, j_local = base_quad_local_ij[quad_index]
                cell_local_ijk.append([i_local, j_local, k])

        for block in blocks:
            block["logical_cells"][2] = len(z_grid) - 1  # type: ignore[index]
            block["logical_nodes"][2] = len(z_grid)  # type: ignore[index]

        metadata = {
            "block_topology": [str(block["name"]) for block in blocks],
            "blocks": blocks,
            "structured_node_to_mesh_node": list(range(len(points))),
            "structured_cell_to_block": cell_blocks,
            "structured_cell_local_ijk": cell_local_ijk,
            "z_levels": z_grid.tolist(),
        }
        return np.asarray(points, dtype=np.float64), np.asarray(hexes, dtype=np.int32), metadata

    def _structured_geometry(self) -> tuple[np.ndarray, np.ndarray, dict]:
        if self.generator_revision == SQUARE_TO_DISK_3D_GENERATOR_REVISION:
            return self._structured_geometry_square_to_disk()
        return self._structured_geometry_o_grid()

    @staticmethod
    def _cell_faces(cell: np.ndarray) -> list[list[int]]:
        return [
            [int(cell[0]), int(cell[1]), int(cell[2]), int(cell[3])],
            [int(cell[4]), int(cell[5]), int(cell[6]), int(cell[7])],
            [int(cell[0]), int(cell[1]), int(cell[5]), int(cell[4])],
            [int(cell[1]), int(cell[2]), int(cell[6]), int(cell[5])],
            [int(cell[2]), int(cell[3]), int(cell[7]), int(cell[6])],
            [int(cell[3]), int(cell[0]), int(cell[4]), int(cell[7])],
        ]

    def _boundary_quads(
        self,
        points: np.ndarray,
        hexes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[dict[str, object]]]:
        face_map: dict[tuple[int, ...], list[int]] = {}
        ordered_faces: dict[tuple[int, ...], list[int]] = {}
        for cell in hexes:
            for face in self._cell_faces(cell):
                key = tuple(sorted(int(v) for v in face))
                face_map[key] = face_map.get(key, []) + [1]
                ordered_faces.setdefault(key, face)

        z_min = float(self.config.z_min)
        z_max = float(self.config.z_max)
        total_electrodes = _total_3d_electrode_count(
            config=self.config,
            electrodes=self.electrodes,
        )
        quad_cells: list[list[int]] = []
        quad_tags: list[int] = []
        field_data: dict[str, np.ndarray] = {"domain": np.array([1, 3], dtype=np.int32)}
        for idx in range(1, total_electrodes + 1):
            field_data[f"electrode_{idx}"] = np.array([idx + 1, 2], dtype=np.int32)
        gap_tag = total_electrodes + 2
        top_tag = total_electrodes + 3
        bottom_tag = total_electrodes + 4
        blank_tag = total_electrodes + 5
        field_data["gaps"] = np.array([gap_tag, 2], dtype=np.int32)
        field_data["top"] = np.array([top_tag, 2], dtype=np.int32)
        field_data["bottom"] = np.array([bottom_tag, 2], dtype=np.int32)
        field_data["blank_side"] = np.array([blank_tag, 2], dtype=np.int32)
        boundary_faces: list[dict[str, object]] = []

        for key, count_marks in face_map.items():
            if len(count_marks) != 1:
                continue
            face = ordered_faces[key]
            coords = points[np.asarray(face, dtype=np.int32)]
            z_coords = coords[:, 2]
            quad_cells.append(list(face))
            face_kind = "blank_side"
            face_tag = blank_tag
            electrode_idx: int | None = None
            if np.allclose(z_coords, z_min, atol=1e-10):
                face_kind = "bottom"
                face_tag = bottom_tag
            elif np.allclose(z_coords, z_max, atol=1e-10):
                face_kind = "top"
                face_tag = top_tag
            else:
                z_center = float(coords[:, 2].mean())
                theta = atan2(float(coords[:, 1].mean()), float(coords[:, 0].mean()))
                face_kind, electrode_idx = _classify_sidewall_patch(
                    theta=theta,
                    z_center=z_center,
                    positions=self.electrodes.positions,
                    config=self.config,
                )
                if face_kind == "gaps":
                    face_tag = gap_tag
                elif face_kind == "electrode" and electrode_idx is not None:
                    face_tag = int(electrode_idx + 1)
                else:
                    face_kind = "blank_side"
                    face_tag = blank_tag
            quad_tags.append(face_tag)
            boundary_faces.append(
                {
                    "kind": face_kind,
                    "tag": int(face_tag),
                    "electrode_index": None if electrode_idx is None else int(electrode_idx),
                    "vertices": [int(v) for v in face],
                }
            )

        return (
            np.asarray(quad_cells, dtype=np.int32),
            np.asarray(quad_tags, dtype=np.int32),
            field_data,
            boundary_faces,
        )

    def generate(
        self,
        *,
        output_dir: Optional[Path] = None,
        mesh_name: Optional[str] = None,
    ) -> EITMesh:
        if not MESHIO_AVAILABLE:
            raise ImportError("meshio is required to generate tensor-product hex meshes.")

        _, _, msh_path, assoc_path = _prepare_output_paths(
            output_dir=output_dir,
            mesh_name=mesh_name,
            prefix="mesh3d",
        )

        points, hexes, structured_meta = self._structured_geometry()
        quads, quad_tags, field_data, boundary_faces = self._boundary_quads(points, hexes)
        hex_tags = np.ones(hexes.shape[0], dtype=np.int32)

        mesh = meshio.Mesh(
            points=points,
            cells=[
                ("hexahedron", hexes),
                ("quad", quads),
            ],
            cell_data={
                "gmsh:physical": [hex_tags, quad_tags],
                "gmsh:geometrical": [hex_tags, quad_tags],
            },
            field_data=field_data,
        )
        meshio.write(msh_path, mesh, file_format="gmsh22")
        sidecar_path = structured_sidecar_path_for_mesh(msh_path)
        _write_structured_sidecar(
            sidecar_path,
            {
                "version": STRUCTURED_SIDECAR_VERSION,
                "mesh_family": "hex",
                "geometry_version": "geomv2",
                "generator_revision": self.generator_revision,
                "mesh_name": msh_path.stem,
                "mesh_file": str(msh_path),
                "n_elec": int(_total_3d_electrode_count(config=self.config, electrodes=self.electrodes)),
                "electrodes_per_ring": int(self.electrodes.n_elec),
                "electrode_layout": normalize_electrode_layout(self.config.electrode_layout),
                "field_tags": {
                    name: int(value[0]) for name, value in field_data.items()
                },
                "boundary_faces": boundary_faces,
                **structured_meta,
            },
        )

        mesh_data = gmshio.read_from_msh(str(msh_path), MPI.COMM_WORLD, rank=0, gdim=3)
        association_table = association_from_mesh_data(mesh_data)
        write_association_table(assoc_path, association_table)

        return build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=estimate_radius(mesh_data.mesh),
            mesh_file=str(msh_path),
            mesh_family="hex",
            geometry_version="geomv2",
            generator_revision=self.generator_revision,
            structured_sidecar_file=str(sidecar_path),
            structured_sidecar_version=STRUCTURED_SIDECAR_VERSION,
        )


def create_cylinder_3d_eit_mesh(
    *,
    n_elec: int = 16,
    radius: float = 1.0,
    height: float = 1.0,
    refinement: int = 8,
    electrode_coverage: float = 0.5,
    electrode_height_ratio: float = 0.2,
    electrode_level_fractions: Optional[Sequence[float]] = None,
    z_center: float = 0.0,
    output_dir: Optional[str] = None,
    mesh_name: Optional[str] = None,
    mesh_family: str = DEFAULT_MESH_FAMILY,
    geometry_version: str = DEFAULT_3D_GEOMETRY_VERSION,
    generator_revision: str = DEFAULT_3D_GENERATOR_REVISION,
    electrode_layout: str = ELECTRODE_LAYOUT_RING_MAJOR,
) -> EITMesh:
    """Create a 3D cylindrical EIT mesh with explicit cell-family selection."""

    refinement_i = int(refinement)
    electrode_vertices = max(3, min(6, refinement_i + 2))
    gap_vertices = 0 if refinement_i <= 2 else 1
    resolved_family = normalize_mesh_family(mesh_family, default=DEFAULT_MESH_FAMILY)
    resolved_geometry = str(geometry_version).strip().lower() or DEFAULT_3D_GEOMETRY_VERSION
    resolved_generator_revision = (
        str(generator_revision).strip().lower() or DEFAULT_3D_GENERATOR_REVISION
    )
    resolved_layout = normalize_electrode_layout(electrode_layout)
    if resolved_geometry == "legacy":
        # The legacy extruded generator only has one continuous sidewall band.
        # Keep old scripts working by using the historical single-sequence
        # numbering there; ring-major multi-level numbering lives in geomv2.
        resolved_layout = ELECTRODE_LAYOUT_ZIGZAG
    level_fractions = (
        normalize_electrode_level_fractions(
            electrode_level_fractions,
            default=DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
        )
        if electrode_level_fractions is not None
        else DEFAULT_ZIGZAG_LEVEL_FRACTIONS
    )
    if resolved_layout == ELECTRODE_LAYOUT_RING_MAJOR:
        n_levels = len(level_fractions)
        if int(n_elec) % n_levels != 0:
            raise ValueError(
                "ring_major 3D meshes require n_elec to be the total physical "
                f"electrode count and divisible by the number of rings/levels ({n_levels}); "
                f"got n_elec={n_elec}."
            )
        electrodes_per_ring = max(int(n_elec) // n_levels, 1)
    else:
        electrodes_per_ring = max(int(n_elec), 1)

    config = Cylinder3DMeshConfig(
        radius=radius,
        height=height,
        z_center=z_center,
        refinement=refinement_i,
        electrode_vertices=electrode_vertices,
        gap_vertices=gap_vertices,
        electrode_height_ratio=electrode_height_ratio,
        electrode_level_fractions=level_fractions,
        electrode_layout=resolved_layout,
    )
    electrodes = ElectrodeArcConfig(n_elec=electrodes_per_ring, coverage=electrode_coverage)
    output_path = Path(output_dir) if output_dir else None

    if resolved_family == "hex":
        if resolved_geometry != "geomv2":
            raise ValueError("mesh_family='hex' currently supports geometry_version='geomv2' only.")
        generator = _GeomV2HexCylinder3DMeshGenerator(
            config=config,
            electrodes=electrodes,
            generator_revision=resolved_generator_revision,
        )
    elif resolved_geometry == "legacy":
        generator = _LegacyTetraCylinder3DMeshGenerator(config=config, electrodes=electrodes)
    else:
        generator = _GeomV2TetraCylinder3DMeshGenerator(
            config=config,
            electrodes=electrodes,
            generator_revision=resolved_generator_revision,
        )
    mesh = generator.generate(output_dir=output_path, mesh_name=mesh_name)
    mesh.generator_revision = (
        LEGACY_3D_GENERATOR_REVISION if resolved_geometry == "legacy" else resolved_generator_revision
    )
    return mesh
