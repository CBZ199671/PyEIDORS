"""EIT mesh generator based on Gmsh + DOLFINx native import."""

from __future__ import annotations

import importlib.util as _import_util
import logging
import tempfile
import time
from contextlib import contextmanager
from math import cos, pi, sin
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from mpi4py import MPI

from ..data.structures import EITMesh, ElectrodePosition, MeshConfig
from ._helpers import (
    add_named_physical_group,
    assert_unique_physical_group_ownership,
    validate_mesh_data_tags,
)
from .dolfinx_mesh_cache import write_dolfinx_mesh_cache, xdmf_cache_path_for_mesh

build_eit_mesh = None


def _ensure_femx() -> None:
    """Defer pyeidors.femx (and transitive dolfinx.io.gmsh) until cache miss."""
    global build_eit_mesh
    if build_eit_mesh is not None:
        return
    from ..femx import build_eit_mesh as _build

    build_eit_mesh = _build


logger = logging.getLogger(__name__)

GMSH_AVAILABLE: bool = _import_util.find_spec("gmsh") is not None
gmsh = None
gmshio = None
_GMSH_LOADED = False


def _ensure_gmsh() -> bool:
    """Import gmsh + dolfinx.io.gmsh on demand. Return False iff unavailable."""
    global gmsh, gmshio, _GMSH_LOADED
    if not GMSH_AVAILABLE:
        return False
    if _GMSH_LOADED:
        return gmsh is not None and gmshio is not None
    if gmsh is not None and gmshio is not None:
        _GMSH_LOADED = True
        return True
    try:
        import gmsh as _gmsh_mod
        from dolfinx.io import gmsh as _gmshio_mod
    except (ImportError, OSError):
        return False

    gmsh = _gmsh_mod
    gmshio = _gmshio_mod
    _GMSH_LOADED = True
    return True


class MeshGenerator:
    """Generate EIT meshes with electrode physical groups."""

    def __init__(self, config: MeshConfig, electrodes: ElectrodePosition):
        self.config = config
        self.electrodes = electrodes
        self.mesh_data: Dict[str, Any] = {}

    @contextmanager
    def gmsh_context(self, model_name: str = "EIT_Mesh"):
        if not _ensure_gmsh():
            raise ImportError("gmsh Python bindings are required to generate meshes.")
        gmsh.initialize()
        gmsh.model.add(model_name)
        try:
            yield
        finally:
            gmsh.finalize()

    def generate(
        self,
        output_dir: Optional[Path] = None,
        return_metadata: bool = False,
        save_msh: bool = True,
        mesh_name: Optional[str] = None,
    ) -> Union[EITMesh, Dict[str, Any]]:
        """Generate mesh.

        Args:
            output_dir: Directory for optional ``.msh`` output.
            return_metadata: If True, return metadata dict including generated EITMesh.
            save_msh: Whether to persist a ``.msh`` source file. The
                DOLFINx XDMF/HDF5 cache is always written for reload.
            mesh_name: Optional base name for generated files.
        """

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        if mesh_name is None:
            mesh_name = f"mesh_{int(time.time() * 1e6) % 1000000}"
        mesh_file = output_dir / f"{mesh_name}.msh"

        with self.gmsh_context(model_name=mesh_name):
            self._create_geometry()
            self._set_physical_groups()
            self._generate_mesh()

            if save_msh:
                gmsh.write(str(mesh_file))

            self._extract_electrode_vertices()
            assert_unique_physical_group_ownership(gmsh.model)
            mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, rank=0, gdim=2)

        electrode_names = [
            f"electrode_{idx}" for idx in range(1, self.electrodes.L + 1)
        ]
        facet_names = [*electrode_names, "gaps"]
        association_table = validate_mesh_data_tags(
            mesh_data,
            gdim=2,
            required_names=["domain", *facet_names],
            required_facet_names=facet_names,
        )
        write_dolfinx_mesh_cache(
            mesh_data,
            source_msh_file=mesh_file if save_msh else None,
            cache_base_file=mesh_file,
            association_table=association_table,
            gdim=2,
        )

        _ensure_femx()

        electrode_vertices = [
            np.asarray(v, dtype=float)
            for v in self.mesh_data.get("electrode_vertices", [])
        ]
        persisted_mesh_file = (
            mesh_file if save_msh else xdmf_cache_path_for_mesh(mesh_file)
        )
        mesh = build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=self.config.radius,
            mesh_file=str(persisted_mesh_file),
            electrode_vertices=electrode_vertices,
        )

        if not return_metadata:
            return mesh

        return {
            "mesh": mesh,
            "mesh_file": mesh.mesh_file,
            "association_table": association_table,
            "radius": self.config.radius,
            "electrodes": self.electrodes,
            "vertex_data": self.mesh_data.get("electrode_vertices", []),
        }

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

        add_named_physical_group(gmsh.model, 2, [surface], 1, "domain")

        electrode_lines = []
        for i, (start, end) in enumerate(electrode_ranges):
            lines_for_electrode = []
            for j in range(start, end):
                line_idx = j % len(lines)
                lines_for_electrode.append(lines[line_idx])

            if lines_for_electrode:
                add_named_physical_group(
                    gmsh.model,
                    1,
                    lines_for_electrode,
                    i + 2,
                    f"electrode_{i + 1}",
                )
                electrode_lines.extend(lines_for_electrode)

        gap_lines = [line for line in lines if line not in electrode_lines]
        if gap_lines:
            add_named_physical_group(
                gmsh.model,
                1,
                gap_lines,
                self.electrodes.L + 2,
                "gaps",
            )

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
