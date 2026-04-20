"""Shared helpers for geometry mesh modules."""

from __future__ import annotations

import re
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict

from ..data.structures import EITMesh
from ..perf.policy import LEGACY_3D_GENERATOR_REVISION


def _physical_group_tag(group: Any) -> int:
    tag = getattr(group, "tag", None)
    if tag is not None:
        return int(tag)
    if isinstance(group, (tuple, list)) and group:
        return int(group[-1])
    return int(group)


def _physical_group_dim(group: Any) -> int | None:
    dim = getattr(group, "dim", None)
    if dim is not None:
        return int(dim)
    if isinstance(group, (tuple, list)) and len(group) >= 2:
        return int(group[0])
    return None


def add_named_physical_group(
    model: Any,
    dim: int,
    entities: list[int] | tuple[int, ...],
    tag: int,
    name: str,
) -> int:
    """Add a named Gmsh physical group with stable, sorted entity ownership."""
    unique_entities = sorted({int(entity) for entity in entities})
    if not unique_entities:
        raise ValueError(f"Physical group {name!r} has no entities")

    try:
        group_tag = model.addPhysicalGroup(
            int(dim),
            unique_entities,
            int(tag),
            name=str(name),
        )
    except TypeError:
        group_tag = model.addPhysicalGroup(int(dim), unique_entities, int(tag))
        if hasattr(model, "setPhysicalName"):
            model.setPhysicalName(int(dim), int(group_tag or tag), str(name))
    else:
        if hasattr(model, "setPhysicalName"):
            model.setPhysicalName(int(dim), int(group_tag or tag), str(name))
    return int(group_tag or tag)


def assert_unique_physical_group_ownership(model: Any) -> None:
    """Fail if a same-dimensional Gmsh entity belongs to multiple physical groups."""
    if not hasattr(model, "getPhysicalGroups") or not hasattr(
        model, "getEntitiesForPhysicalGroup"
    ):
        return

    owners: dict[tuple[int, int], str] = {}
    for dim, group_tag in model.getPhysicalGroups():
        dim_i = int(dim)
        tag_i = int(group_tag)
        if hasattr(model, "getPhysicalName"):
            name = str(model.getPhysicalName(dim_i, tag_i) or f"{dim_i}:{tag_i}")
        else:
            name = f"{dim_i}:{tag_i}"
        for entity in model.getEntitiesForPhysicalGroup(dim_i, tag_i):
            key = (dim_i, int(entity))
            previous = owners.get(key)
            if previous is not None:
                raise RuntimeError(
                    f"Gmsh entity dim={dim_i} tag={int(entity)} belongs to both "
                    f"{previous!r} and {name!r}"
                )
            owners[key] = name


def association_from_mesh_data(mesh_data) -> Dict[str, int]:
    """Extract an association table from DOLFINx mesh data physical groups."""
    return {
        name: _physical_group_tag(group)
        for name, group in (mesh_data.physical_groups or {}).items()
    }


def physical_group_dimensions_from_mesh_data(mesh_data) -> Dict[str, int]:
    """Extract physical group dimensions from DOLFINx mesh data when available."""
    dimensions: Dict[str, int] = {}
    for name, group in (mesh_data.physical_groups or {}).items():
        dim = _physical_group_dim(group)
        if dim is not None:
            dimensions[str(name)] = int(dim)
    return dimensions


def validate_mesh_data_tags(
    mesh_data,
    *,
    gdim: int,
    required_names: list[str] | tuple[str, ...] = (),
    required_facet_names: list[str] | tuple[str, ...] = (),
) -> Dict[str, int]:
    """Validate DOLFINx Gmsh import tags and return the association table."""
    association = association_from_mesh_data(mesh_data)
    missing = sorted(
        str(name) for name in required_names if str(name) not in association
    )
    if missing:
        raise RuntimeError(
            "DOLFINx Gmsh import is missing physical groups: " + ", ".join(missing)
        )

    dimensions = physical_group_dimensions_from_mesh_data(mesh_data)
    if dimensions.get("domain") is not None and dimensions["domain"] != int(gdim):
        raise RuntimeError(
            f"Physical group 'domain' has dim={dimensions['domain']}; expected {int(gdim)}"
        )

    expected_facet_dim = int(gdim) - 1
    bad_facets = [
        name
        for name in required_facet_names
        if dimensions.get(str(name)) is not None
        and dimensions[str(name)] != expected_facet_dim
    ]
    if bad_facets:
        details = ", ".join(
            f"{name}: dim={dimensions[str(name)]}" for name in bad_facets
        )
        raise RuntimeError(
            f"Facet physical groups have unexpected dimensions ({details}); "
            f"expected dim={expected_facet_dim}"
        )

    if required_facet_names and getattr(mesh_data, "facet_tags", None) is None:
        raise RuntimeError("DOLFINx Gmsh import did not produce facet_tags")

    return association


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
