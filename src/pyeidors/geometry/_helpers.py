"""Shared helpers for geometry mesh modules."""

from __future__ import annotations

import re
from configparser import ConfigParser
from pathlib import Path
from typing import Dict

from ..data.structures import EITMesh
from ..perf.policy import LEGACY_3D_GENERATOR_REVISION


def association_from_mesh_data(mesh_data) -> Dict[str, int]:
    """Extract an association table from DOLFINx mesh data physical groups."""
    return {
        name: int(group.tag)
        for name, group in (mesh_data.physical_groups or {}).items()
    }


def write_association_table(path: Path, association_table: Dict[str, int]) -> None:
    """Persist an association table as an INI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    config = ConfigParser()
    config["ASSOCIATION TABLE"] = {str(k): str(v) for k, v in association_table.items()}
    with path.open("w", encoding="utf-8") as f:
        config.write(f)


def infer_mesh_family_from_mesh(mesh: EITMesh) -> str | None:
    """Guess mesh family (hex/tetra) from cell vertex count."""
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


def infer_geometry_version(mesh_name: str) -> str:
    """Infer geometry version from a mesh name string."""
    lowered = str(mesh_name).strip().lower()
    return "geomv2" if "geomv2" in lowered else "legacy"


def infer_generator_revision(mesh_name: str) -> str:
    """Infer generator revision from a mesh name string."""
    lowered = str(mesh_name).strip().lower()
    match = re.search(r"(g3d\d+)", lowered)
    if match is not None:
        return str(match.group(1))
    return LEGACY_3D_GENERATOR_REVISION
