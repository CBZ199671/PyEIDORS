"""Optimized EIT mesh generation using Gmsh + DOLFINx native mesh import."""

from __future__ import annotations

import logging
import re
import tempfile
import time
from configparser import ConfigParser
from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from mpi4py import MPI
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
import ufl

from ..data.structures import EITMesh
from ..femx import build_eit_mesh, estimate_radius
from ..perf.policy import (
    DEFAULT_3D_GEOMETRY_VERSION,
    DEFAULT_3D_GENERATOR_REVISION,
    DEFAULT_MESH_FAMILY,
    LEGACY_3D_GENERATOR_REVISION,
    normalize_mesh_family,
)
from .mesh3d_generator import (
    DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
    STRUCTURED_SIDECAR_VERSION,
    create_cylinder_3d_eit_mesh,
    load_structured_sidecar,
    normalize_electrode_level_fractions,
    structured_sidecar_path_for_mesh,
)

logger = logging.getLogger(__name__)

try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:  # pragma: no cover
    gmsh = None  # type: ignore[assignment]
    GMSH_AVAILABLE = False


@dataclass
class ElectrodePosition:
    """Electrode position configuration."""

    L: int
    coverage: float = 0.5
    rotation: float = 0.0
    anticlockwise: bool = True

    def __post_init__(self):
        if not isinstance(self.L, int) or self.L <= 0:
            raise ValueError("Number of electrodes must be a positive integer")
        if not 0 < self.coverage <= 1:
            raise ValueError("Coverage must be in range (0, 1]")

    @property
    def positions(self) -> List[Tuple[float, float]]:
        electrode_size = 2 * pi / self.L * self.coverage
        gap_size = 2 * pi / self.L * (1 - self.coverage)

        first_electrode_center = pi / 2 + self.rotation
        first_electrode_start = first_electrode_center - electrode_size / 2

        positions: List[Tuple[float, float]] = []
        for i in range(self.L):
            total_space = electrode_size + gap_size
            start = first_electrode_start + i * total_space
            end = start + electrode_size
            positions.append((start, end))

        if not self.anticlockwise:
            positions[1:] = positions[1:][::-1]

        return positions


@dataclass
class OptimizedMeshConfig:
    radius: float = 1.0
    refinement: int = 8
    electrode_vertices: int = 6
    gap_vertices: int = 1

    @property
    def mesh_size(self) -> float:
        return self.radius / (self.refinement * 2)


class OptimizedMeshGenerator:
    """Optimized mesh generator with stable physical tagging."""

    def __init__(self, config: OptimizedMeshConfig, electrodes: ElectrodePosition):
        self.config = config
        self.electrodes = electrodes
        self.mesh_data: Dict[str, object] = {}

    def generate(self, output_dir: Optional[Path] = None, mesh_name: Optional[str] = None) -> EITMesh:
        if not GMSH_AVAILABLE:
            raise ImportError("gmsh Python bindings are required to generate meshes.")

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if mesh_name is None:
            mesh_name = f"mesh_{int(time.time() * 1e6) % 1000000}"

        msh_path = output_dir / f"{mesh_name}.msh"
        association_path = output_dir / f"{mesh_name}_association_table.ini"

        initialized_here = False
        if not gmsh.isInitialized():
            gmsh.initialize()
            initialized_here = True
        gmsh.clear()
        gmsh.model.add(mesh_name)
        try:
            self._create_geometry()
            self._set_physical_groups()
            self._generate_mesh()
            gmsh.write(str(msh_path))
            self._extract_electrode_vertices()

            mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2)
        finally:
            if initialized_here:
                gmsh.finalize()
            else:
                gmsh.clear()

        association_table = _association_from_mesh_data(mesh_data)
        _write_association_table(association_path, association_table)

        electrode_vertices = [np.asarray(v, dtype=float) for v in self.mesh_data.get("electrode_vertices", [])]
        mesh = build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=self.config.radius,
            mesh_file=str(msh_path),
            electrode_vertices=electrode_vertices,
        )
        return mesh

    def _create_geometry(self):
        positions = self.electrodes.positions
        n_in = self.config.electrode_vertices
        n_out = self.config.gap_vertices
        r = self.config.radius

        boundary_points = []
        electrode_ranges = []

        for i, (start, end) in enumerate(positions):
            start_idx = len(boundary_points)

            for theta in np.linspace(start, end, n_in):
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)

            electrode_ranges.append((start_idx, len(boundary_points) - 1))

            if i < len(positions) - 1:
                gap_start = end
                gap_end = positions[i + 1][0]
            else:
                gap_start = end
                gap_end = positions[0][0] + 2 * pi

            gap_points = np.linspace(gap_start, gap_end, n_out + 2)[1:-1]
            for theta in gap_points:
                x, y = r * cos(theta), r * sin(theta)
                tag = gmsh.model.occ.addPoint(x, y, 0.0)
                boundary_points.append(tag)

        lines = []
        for i in range(len(boundary_points)):
            next_i = (i + 1) % len(boundary_points)
            line = gmsh.model.occ.addLine(boundary_points[i], boundary_points[next_i])
            lines.append(line)

        loop = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([loop])

        mesh_size_center = 0.095 * r
        cp_distance = 0.1 * r
        center_points = [
            gmsh.model.occ.addPoint(x, y, 0.0, meshSize=mesh_size_center)
            for x, y in [
                (-cp_distance, cp_distance),
                (cp_distance, cp_distance),
                (-cp_distance, -cp_distance),
                (cp_distance, -cp_distance),
            ]
        ]

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.embed(0, center_points, 2, surface)

        self.mesh_data["boundary_points"] = boundary_points
        self.mesh_data["electrode_ranges"] = electrode_ranges
        self.mesh_data["lines"] = lines
        self.mesh_data["surface"] = surface

    def _set_physical_groups(self):
        surface = self.mesh_data["surface"]
        lines = self.mesh_data["lines"]
        electrode_ranges = self.mesh_data["electrode_ranges"]

        gmsh.model.addPhysicalGroup(2, [surface], 1, name="domain")

        electrode_lines = []
        for i, (start, end) in enumerate(electrode_ranges):
            lines_for_electrode = []
            for j in range(start, end):
                line_idx = j % len(lines)
                lines_for_electrode.append(lines[line_idx])

            if lines_for_electrode:
                gmsh.model.addPhysicalGroup(1, lines_for_electrode, i + 2, name=f"electrode_{i + 1}")
                electrode_lines.extend(lines_for_electrode)

        gap_lines = [line for line in lines if line not in electrode_lines]
        if gap_lines:
            gmsh.model.addPhysicalGroup(1, gap_lines, self.electrodes.L + 2, name="gaps")

    def _generate_mesh(self):
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.config.mesh_size)
        gmsh.model.mesh.generate(2)

    def _extract_electrode_vertices(self):
        positions = self.electrodes.positions
        r = self.config.radius
        n_in = self.config.electrode_vertices

        electrode_vertices = []
        for start, end in positions:
            vertices = []
            for theta in np.linspace(start, end, n_in):
                vertices.append([r * cos(theta), r * sin(theta)])
            electrode_vertices.append(vertices)

        self.mesh_data["electrode_vertices"] = electrode_vertices


class OptimizedMeshConverter:
    """Read a ``.msh`` file into DOLFINx mesh and tags."""

    def __init__(self, mesh_file: str, output_dir: str, gdim: int = 2):
        self.mesh_file = Path(mesh_file)
        self.output_dir = Path(output_dir)
        self.prefix = self.mesh_file.stem
        self.gdim = int(gdim)

    def convert(self) -> tuple[EITMesh, object, Dict[str, int]]:
        mesh_data = gmshio.read_from_msh(str(self.mesh_file), MPI.COMM_WORLD, rank=0, gdim=self.gdim)
        association_table = _association_from_mesh_data(mesh_data)

        association_file = self.output_dir / f"{self.prefix}_association_table.ini"
        _write_association_table(association_file, association_table)

        mesh = build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=estimate_radius(mesh_data.mesh),
            mesh_file=str(self.mesh_file),
        )
        return mesh, mesh_data.facet_tags, association_table


def _write_association_table(path: Path, association_table: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = ConfigParser()
    config["ASSOCIATION TABLE"] = {str(k): str(v) for k, v in association_table.items()}
    with path.open("w", encoding="utf-8") as f:
        config.write(f)


def _association_from_mesh_data(mesh_data) -> Dict[str, int]:
    association_table: Dict[str, int] = {}
    for name, group in (mesh_data.physical_groups or {}).items():
        association_table[name] = int(group.tag)
    return association_table


# Convenience functions

def create_eit_mesh(
    n_elec: int = 16,
    radius: float = 1.0,
    refinement: int = 6,
    electrode_coverage: float = 0.5,
    output_dir: str = None,
    mesh_name: Optional[str] = None,
):
    mesh_config = OptimizedMeshConfig(
        radius=radius,
        refinement=refinement,
        electrode_vertices=6,
        gap_vertices=1,
    )

    electrode_config = ElectrodePosition(
        L=n_elec,
        coverage=electrode_coverage,
        rotation=0.0,
        anticlockwise=True,
    )

    generator = OptimizedMeshGenerator(mesh_config, electrode_config)
    return generator.generate(output_dir=Path(output_dir) if output_dir else None, mesh_name=mesh_name)


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def _build_cache_name(n_elec: int, radius: float, refinement: int, electrode_coverage: float) -> str:
    radius_str = _format_float(radius)
    coverage_str = _format_float(electrode_coverage)
    return f"mesh_{n_elec}e_r{radius_str}_ref{refinement}_cov{coverage_str}"


def _build_cache_name_3d(
    n_elec: int,
    radius: float,
    height: float,
    refinement: int,
    electrode_coverage: float,
    electrode_height_ratio: float,
    electrode_level_fractions: tuple[float, ...],
    z_center: float,
    mesh_family: str,
    geometry_version: str,
    generator_revision: str,
) -> str:
    levels_str = "-".join(_format_float(float(value)) for value in electrode_level_fractions)
    return (
        "mesh3d_"
        f"{n_elec}e_r{_format_float(radius)}_h{_format_float(height)}_"
        f"ref{refinement}_cov{_format_float(electrode_coverage)}_"
        f"ehr{_format_float(electrode_height_ratio)}_"
        f"lev{levels_str}_"
        f"zc{_format_float(z_center)}_"
        f"cf{str(mesh_family).strip().lower()}_{str(geometry_version).strip().lower()}_"
        f"{str(generator_revision).strip().lower()}"
    )


def _infer_geometry_version(mesh_name: str) -> str:
    lowered = str(mesh_name).strip().lower()
    return "geomv2" if "geomv2" in lowered else "legacy"


def _infer_generator_revision(mesh_name: str) -> str:
    lowered = str(mesh_name).strip().lower()
    match = re.search(r"(g3d\d+)", lowered)
    if match is not None:
        return str(match.group(1))
    return LEGACY_3D_GENERATOR_REVISION


def _infer_mesh_family_from_mesh(mesh: EITMesh) -> str | None:
    if int(mesh.topology.dim) != 3:
        return None
    cells = mesh.cells()
    if cells.ndim != 2 or cells.shape[0] == 0:
        return None
    verts_per_cell = int(cells.shape[1])
    if verts_per_cell == 8:
        return "hex"
    if verts_per_cell == 4:
        return "tetra"
    return None


def _cached_3d_cem_mesh_is_complete(mesh: EITMesh, *, n_elec: int) -> bool:
    if int(mesh.topology.dim) != 3:
        return True
    association = dict(getattr(mesh, "association_table", {}) or {})
    if not association or "domain" not in association or "gaps" not in association:
        return False
    electrode_keys = [f"electrode_{idx}" for idx in range(1, int(n_elec) + 1)]
    if any(key not in association for key in electrode_keys):
        return False
    if mesh.facet_tags is None:
        return False
    try:
        ds = ufl.Measure("ds", domain=mesh.mesh, subdomain_data=mesh.facet_tags)
        one = fem.Constant(mesh.mesh, 1.0)
        measures = []
        for key in electrode_keys:
            value_local = fem.assemble_scalar(fem.form(one * ds(int(association[key]))))
            value = mesh.comm.allreduce(value_local, op=MPI.SUM)
            measures.append(float(value))
    except Exception as exc:
        logger.warning("Skipping cached 3D mesh %s due to CEM validation failure: %s", mesh.mesh_file, exc)
        return False
    arr = np.asarray(measures, dtype=float)
    if not bool(arr.size == int(n_elec) and np.all(np.isfinite(arr)) and float(np.min(arr)) > 0.0):
        return False
    mesh_family = str(getattr(mesh, "mesh_family", None) or "").strip().lower()
    geometry_version = str(getattr(mesh, "geometry_version", None) or "").strip().lower()
    generator_revision = str(getattr(mesh, "generator_revision", None) or "").strip().lower()
    if mesh_family == "hex" and geometry_version == "geomv2" and generator_revision == DEFAULT_3D_GENERATOR_REVISION:
        mesh_file = getattr(mesh, "mesh_file", None)
        if not mesh_file:
            return False
        sidecar_path = structured_sidecar_path_for_mesh(mesh_file)
        if not sidecar_path.exists():
            logger.warning("Skipping cached mesh %s because structured sidecar is missing", mesh.mesh_file)
            return False
        try:
            load_structured_sidecar(sidecar_path)
        except Exception as exc:
            logger.warning(
                "Skipping cached mesh %s because structured sidecar validation failed: %s",
                mesh.mesh_file,
                exc,
            )
            return False
    return True


def _load_cached_mesh(mesh_dir: Path, mesh_name: str, *, gdim: int = 2, n_elec: int = 16):
    msh_file = mesh_dir / f"{mesh_name}.msh"
    association_file = mesh_dir / f"{mesh_name}_association_table.ini"

    if not msh_file.exists():
        return None

    try:
        mesh_data = gmshio.read_from_msh(str(msh_file), MPI.COMM_WORLD, rank=0, gdim=int(gdim))
    except Exception as exc:
        logger.warning(
            "Skipping cached mesh %s due to gdim=%d load failure: %s",
            msh_file,
            int(gdim),
            exc,
        )
        return None

    if association_file.exists():
        association = ConfigParser()
        association.read(association_file)
        if "ASSOCIATION TABLE" in association:
            section = association["ASSOCIATION TABLE"]
            association_table = {key: int(value) for key, value in section.items()}
        else:
            association_table = {}
    else:
        association_table = _association_from_mesh_data(mesh_data)

    sidecar_path = structured_sidecar_path_for_mesh(msh_file)
    geometry_version = _infer_geometry_version(mesh_name)
    generator_revision = _infer_generator_revision(mesh_name)
    if sidecar_path.exists():
        try:
            sidecar = load_structured_sidecar(sidecar_path)
            geometry_version = str(sidecar.get("geometry_version", geometry_version)).strip().lower() or geometry_version
            generator_revision = str(sidecar.get("generator_revision", generator_revision)).strip().lower() or generator_revision
        except Exception:
            pass

    mesh = build_eit_mesh(
        mesh_data.mesh,
        facet_tags=mesh_data.facet_tags,
        cell_tags=mesh_data.cell_tags,
        association_table=association_table,
        physical_groups=mesh_data.physical_groups,
        radius=estimate_radius(mesh_data.mesh),
        mesh_file=str(msh_file),
        geometry_version=geometry_version,
        generator_revision=generator_revision,
        structured_sidecar_file=(
            str(sidecar_path)
            if sidecar_path.exists()
            else None
        ),
        structured_sidecar_version=(
            STRUCTURED_SIDECAR_VERSION
            if sidecar_path.exists()
            else None
        ),
    )
    mesh.mesh_family = _infer_mesh_family_from_mesh(mesh)
    if int(gdim) == 3 and not _cached_3d_cem_mesh_is_complete(mesh, n_elec=int(n_elec)):
        logger.warning("Skipping cached mesh %s because 3D CEM tags/measures are incomplete", mesh_name)
        return None
    return mesh


def load_or_create_mesh(
    mesh_dir: str = "eit_meshes",
    mesh_name: str = None,
    n_elec: int = 16,
    dimension: int = 2,
    **kwargs,
):
    mesh_dir_path = Path(mesh_dir)
    mesh_dir_path.mkdir(parents=True, exist_ok=True)

    params = dict(kwargs)
    radius = params.pop("radius", 1.0)
    refinement = params.pop("refinement", 6)
    electrode_coverage = params.pop("electrode_coverage", 0.5)
    height = params.pop("height", 1.0)
    electrode_height_ratio = params.pop("electrode_height_ratio", 0.2)
    if "electrode_layout" in params:
        raise ValueError(
            "3D electrode_layout has been removed from PyEIDORS. "
            "Use zigzag electrode_level_fractions instead."
        )
    electrode_level_fractions = normalize_electrode_level_fractions(
        params.pop("electrode_level_fractions", DEFAULT_ZIGZAG_LEVEL_FRACTIONS),
        default=DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
    )
    z_center = params.pop("z_center", 0.0)
    mesh_family = normalize_mesh_family(
        params.pop("mesh_family", DEFAULT_MESH_FAMILY),
        default=DEFAULT_MESH_FAMILY,
    )
    geometry_version = (
        str(params.pop("geometry_version", DEFAULT_3D_GEOMETRY_VERSION)).strip().lower()
        or DEFAULT_3D_GEOMETRY_VERSION
    )
    generator_revision = str(
        params.pop("generator_revision", DEFAULT_3D_GENERATOR_REVISION)
    ).strip().lower() or DEFAULT_3D_GENERATOR_REVISION
    gdim = int(dimension)
    if gdim not in {2, 3}:
        raise ValueError(f"dimension must be 2 or 3, got {dimension!r}")

    if mesh_name:
        cache_name = mesh_name
    elif gdim == 2:
        cache_name = _build_cache_name(n_elec, radius, refinement, electrode_coverage)
    else:
        cache_name = _build_cache_name_3d(
            n_elec=n_elec,
            radius=radius,
            height=height,
            refinement=refinement,
            electrode_coverage=electrode_coverage,
            electrode_height_ratio=electrode_height_ratio,
            electrode_level_fractions=electrode_level_fractions,
            z_center=z_center,
            mesh_family=mesh_family,
            geometry_version=geometry_version,
            generator_revision=generator_revision,
        )

    cached_mesh = _load_cached_mesh(mesh_dir_path, cache_name, gdim=gdim, n_elec=n_elec)
    if cached_mesh is not None:
        logger.info("Loaded cached mesh: %s", cache_name)
        return cached_mesh

    logger.info("Cached mesh not found, generating: %s", cache_name)
    if params:
        logger.debug("Unused mesh parameters: %s", params)

    if gdim == 2:
        return create_eit_mesh(
            n_elec=n_elec,
            radius=radius,
            refinement=refinement,
            electrode_coverage=electrode_coverage,
            output_dir=str(mesh_dir_path),
            mesh_name=cache_name,
        )
    return create_cylinder_3d_eit_mesh(
        n_elec=n_elec,
        radius=radius,
        height=height,
        refinement=refinement,
        electrode_coverage=electrode_coverage,
        electrode_height_ratio=electrode_height_ratio,
        electrode_level_fractions=electrode_level_fractions,
        z_center=z_center,
        output_dir=str(mesh_dir_path),
        mesh_name=cache_name,
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        generator_revision=generator_revision,
    )
