"""Mesh converter for Gmsh -> DOLFINx mesh import.

T78 Path C consolidates the canonical ``MeshConverter`` body so the
:class:`OptimizedMeshConverter` variant in
:mod:`pyeidors.geometry.optimized_mesh_generator` can be a thin
subclass that only adds a ``radius_provider`` (``estimate_radius``).
The shared body covers Gmsh ``.msh`` parsing, physical group
validation, association-table persistence and DOLFINx mesh cache
writes — exactly the steps both historical converters duplicated.
"""

from __future__ import annotations

import importlib.util as _import_util
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from mpi4py import MPI

from ..data.structures import EITMesh
from ..femx import build_eit_mesh
from ._helpers import validate_mesh_data_tags, write_association_table
from .dolfinx_mesh_cache import write_dolfinx_mesh_cache

logger = logging.getLogger(__name__)

GMSH_AVAILABLE: bool = _import_util.find_spec("gmsh") is not None
gmshio = None
_GMSHIO_LOADED = False


def _ensure_gmsh() -> bool:
    """Import dolfinx.io.gmsh on demand. Return False iff unavailable."""
    global gmshio, _GMSHIO_LOADED
    if _GMSHIO_LOADED:
        return GMSH_AVAILABLE
    _GMSHIO_LOADED = True
    if not GMSH_AVAILABLE:
        return False
    from dolfinx.io import gmsh as _gmshio_mod

    gmshio = _gmshio_mod
    return True


RadiusProvider = Callable[[object], float]


class MeshConverter:
    """Convert ``.msh`` file to DOLFINx mesh with facet tags.

    ``radius_provider`` is optional callable invoked on the parsed
    DOLFINx mesh to derive an :attr:`EITMesh.radius` value (used by
    e.g. :class:`OptimizedMeshConverter` to reuse
    :func:`pyeidors.femx.estimate_radius`). When ``None``, the produced
    :class:`EITMesh` carries ``radius=None`` — the legacy plain-converter
    behavior.
    """

    def __init__(
        self,
        mesh_file: str,
        output_dir: str,
        gdim: int = 2,
        *,
        radius_provider: Optional[RadiusProvider] = None,
    ):
        self.mesh_file = Path(mesh_file)
        self.output_dir = Path(output_dir)
        self.prefix = self.mesh_file.stem
        self.gdim = int(gdim)
        if self.gdim not in {2, 3}:
            raise ValueError(f"gdim must be 2 or 3, got {gdim!r}")
        self._radius_provider = radius_provider

    def convert(self) -> Tuple[EITMesh, object, Dict[str, int]]:
        if not _ensure_gmsh():
            raise ImportError("gmsh Python bindings are required to convert meshes.")
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

        radius = (
            self._radius_provider(mesh_data.mesh)
            if self._radius_provider is not None
            else None
        )
        mesh = build_eit_mesh(
            mesh_data.mesh,
            facet_tags=mesh_data.facet_tags,
            cell_tags=mesh_data.cell_tags,
            association_table=association_table,
            physical_groups=mesh_data.physical_groups,
            radius=radius,
            mesh_file=str(self.mesh_file),
        )
        return mesh, mesh_data.facet_tags, association_table

    def _write_association_table(self, association_table: Dict[str, int]) -> None:
        file_path = self.output_dir / f"{self.prefix}_association_table.ini"
        write_association_table(file_path, association_table)
        logger.debug("Association table saved: %s", file_path)
