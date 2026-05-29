"""Shared helpers for geometry mesh modules."""

from __future__ import annotations

import re
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..data.structures import EITMesh
from ..electrodes.layout import ELECTRODE_LAYOUT_RING_MAJOR, normalize_electrode_layout
from ..perf.policy import LEGACY_3D_GENERATOR_REVISION

build_eit_mesh = None
estimate_radius = None


def _ensure_femx() -> None:
    """Defer pyeidors.femx (and transitive dolfinx.io.gmsh) until cache miss."""
    global build_eit_mesh, estimate_radius
    if build_eit_mesh is not None and estimate_radius is not None:
        return
    from ..femx import build_eit_mesh as _build, estimate_radius as _estimate

    if build_eit_mesh is None:
        build_eit_mesh = _build
    if estimate_radius is None:
        estimate_radius = _estimate


def format_float_compact(value: float) -> str:
    """Format ``value`` for cache filename use: 6dp, trim trailing 0/dot, ``.``→``p``."""
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace(".", "p")


def geometry_dtype_cache_suffix(geometry_dtype: Any | None) -> str:
    """Return a cache-name suffix for non-default mesh coordinate precision."""
    if geometry_dtype is None:
        return ""
    dtype = np.dtype(geometry_dtype)
    if dtype == np.dtype(np.float64):
        return ""
    if dtype == np.dtype(np.float32):
        return "_f32"
    return f"_{dtype.name.replace('.', 'p')}"


def build_mesh_cache_name(
    n_elec: int,
    radius: float,
    refinement: int,
    electrode_coverage: float,
    geometry_dtype: Any | None = None,
) -> str:
    """Cache-stable 2D mesh name keyed on ``(n_elec, radius, refinement, coverage)``."""
    radius_str = format_float_compact(radius)
    coverage_str = format_float_compact(electrode_coverage)
    return (
        f"mesh_{int(n_elec)}e_r{radius_str}_ref{int(refinement)}_cov{coverage_str}"
        f"{geometry_dtype_cache_suffix(geometry_dtype)}"
    )


def build_mesh_cache_name_3d(
    n_elec: int,
    radius: float,
    height: float,
    refinement: int,
    electrode_coverage: float,
    electrode_height_ratio: float,
    electrode_level_fractions: Sequence[float],
    z_center: float,
    mesh_family: str,
    geometry_version: str,
    generator_revision: str,
    electrode_layout: str = ELECTRODE_LAYOUT_RING_MAJOR,
    geometry_dtype: Any | None = None,
) -> str:
    """Cache-stable 3D cylinder mesh name keyed on full geometry signature."""
    levels_str = "-".join(
        format_float_compact(float(value)) for value in electrode_level_fractions
    )
    layout_str = normalize_electrode_layout(electrode_layout)
    return (
        "mesh3d_"
        f"{int(n_elec)}e_r{format_float_compact(radius)}_h{format_float_compact(height)}_"
        f"ref{int(refinement)}_cov{format_float_compact(electrode_coverage)}_"
        f"ehr{format_float_compact(electrode_height_ratio)}_"
        f"lev{levels_str}_"
        f"zc{format_float_compact(z_center)}_"
        f"el{layout_str}_"
        f"cf{str(mesh_family).strip().lower()}_{str(geometry_version).strip().lower()}_"
        f"{str(generator_revision).strip().lower()}"
        f"{geometry_dtype_cache_suffix(geometry_dtype)}"
    )


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


def finalize_3d_cylinder_mesh(
    mesh_data,
    *,
    msh_path: Path,
    assoc_path: Path,
    total_electrodes: int,
    mesh_family: str,
    geometry_version: str,
    generator_revision: str,
    structured_sidecar_file: Optional[Path] = None,
    structured_sidecar_version: Optional[str] = None,
) -> EITMesh:
    """Run the post-mesh-creation tail shared by all 3D cylinder generators.

    The legacy tetra, geomv2 tetra and geomv2 hex cylinder generators
    historically duplicated the same ~30-line tail: build the
    ``electrode_*`` + ``gaps`` facet name list, validate physical
    groups via :func:`validate_mesh_data_tags`, persist the association
    table, write the DOLFINx mesh cache (XDMF + HDF5 + optional
    structured sidecar), then return an :class:`EITMesh` via
    :func:`pyeidors.femx.build_eit_mesh`. T78 phase 2 consolidates that
    tail here so the three generators only differ in their geometry
    construction (``_create_geometry`` / ``_set_physical_groups`` /
    ``_structured_geometry`` etc.) and in the
    ``mesh_family`` / ``geometry_version`` / ``generator_revision``
    triple.

    The optional ``structured_sidecar_file`` / ``structured_sidecar_version``
    pair is only populated by the geomv2 hex variant; the helper
    forwards both ``None`` for the tetra paths so the existing
    :func:`write_dolfinx_mesh_cache` and :func:`build_eit_mesh`
    signatures see the same defaults they always have.
    """
    # Local import keeps the module-level dependency tree clean — only
    # callers that finalize a mesh pull these heavyweight modules in.
    _ensure_femx()
    from .dolfinx_mesh_cache import write_dolfinx_mesh_cache

    electrode_names = [
        f"electrode_{idx}" for idx in range(1, int(total_electrodes) + 1)
    ]
    facet_names = [*electrode_names, "gaps"]
    association_table = validate_mesh_data_tags(
        mesh_data,
        gdim=3,
        required_names=["domain", *facet_names],
        required_facet_names=facet_names,
    )
    write_association_table(assoc_path, association_table)

    cache_kwargs: Dict[str, Any] = {
        "source_msh_file": msh_path,
        "association_table": association_table,
        "gdim": 3,
        "mesh_family": mesh_family,
        "geometry_version": geometry_version,
        "generator_revision": generator_revision,
    }
    if structured_sidecar_file is not None:
        cache_kwargs["structured_sidecar_file"] = structured_sidecar_file
    if structured_sidecar_version is not None:
        cache_kwargs["structured_sidecar_version"] = structured_sidecar_version
    write_dolfinx_mesh_cache(mesh_data, **cache_kwargs)

    build_kwargs: Dict[str, Any] = {
        "facet_tags": mesh_data.facet_tags,
        "cell_tags": mesh_data.cell_tags,
        "association_table": association_table,
        "physical_groups": mesh_data.physical_groups,
        "radius": estimate_radius(mesh_data.mesh),
        "mesh_file": str(msh_path),
        "mesh_family": mesh_family,
        "geometry_version": geometry_version,
        "generator_revision": generator_revision,
    }
    if structured_sidecar_file is not None:
        build_kwargs["structured_sidecar_file"] = str(structured_sidecar_file)
    if structured_sidecar_version is not None:
        build_kwargs["structured_sidecar_version"] = structured_sidecar_version

    return build_eit_mesh(mesh_data.mesh, **build_kwargs)
