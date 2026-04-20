"""Mesh converter for Gmsh -> DOLFINx mesh import."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from ..data.structures import EITMesh
from ..femx import build_eit_mesh
from ._helpers import validate_mesh_data_tags, write_association_table
from .dolfinx_mesh_cache import write_dolfinx_mesh_cache

logger = logging.getLogger(__name__)


class MeshConverter:
    """Convert ``.msh`` file to DOLFINx mesh with facet tags."""

    def __init__(self, mesh_file: str, output_dir: str, gdim: int = 2):
        self.mesh_file = Path(mesh_file)
        self.output_dir = Path(output_dir)
        self.prefix = self.mesh_file.stem
        self.gdim = int(gdim)
        if self.gdim not in {2, 3}:
            raise ValueError(f"gdim must be 2 or 3, got {gdim!r}")

    def convert(self) -> Tuple[EITMesh, object, Dict[str, int]]:
        mesh_data = gmshio.read_from_msh(
            str(self.mesh_file),
            MPI.COMM_WORLD,
            rank=0,
            gdim=self.gdim,
        )

        association_table = validate_mesh_data_tags(mesh_data, gdim=self.gdim)
        self._write_association_table(association_table)
        write_dolfinx_mesh_cache(
            mesh_data,
            source_msh_file=self.mesh_file,
            association_table=association_table,
            gdim=self.gdim,
        )

        mesh = build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            mesh_file=str(self.mesh_file),
        )
        return mesh, mesh_data.facet_tags, association_table

    def _write_association_table(self, association_table: Dict[str, int]) -> None:
        file_path = self.output_dir / f"{self.prefix}_association_table.ini"
        write_association_table(file_path, association_table)
        logger.debug("Association table saved: %s", file_path)
