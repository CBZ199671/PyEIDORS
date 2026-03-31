"""DOLFINx helper utilities for PyEIDORS."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from dolfinx import fem
from dolfinx.mesh import Mesh, MeshTags
import ufl

from ..data.structures import EITMesh


def mesh_coordinates(mesh: Mesh) -> np.ndarray:
    """Return geometry coordinates as ``(n_points, gdim)`` array."""
    return mesh.geometry.x[:, : mesh.geometry.dim]


def mesh_num_vertices(mesh: Mesh) -> int:
    index_map = mesh.topology.index_map(0)
    return int(index_map.size_local if index_map is not None else 0)


def mesh_num_cells(mesh: Mesh) -> int:
    tdim = mesh.topology.dim
    index_map = mesh.topology.index_map(tdim)
    return int(index_map.size_local if index_map is not None else 0)


def mesh_num_edges(mesh: Mesh) -> int:
    mesh.topology.create_entities(1)
    index_map = mesh.topology.index_map(1)
    return int(index_map.size_local if index_map is not None else 0)


def mesh_cell_vertices(mesh: Mesh) -> np.ndarray:
    """Return local cell->vertex connectivity array."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    connectivity = mesh.topology.connectivity(tdim, 0)
    if connectivity is None:
        return np.empty((0, 0), dtype=np.int32)

    num_cells = mesh_num_cells(mesh)
    if num_cells == 0:
        return np.empty((0, 0), dtype=np.int32)

    first = connectivity.links(0)
    vertices_per_cell = len(first)
    data = np.zeros((num_cells, vertices_per_cell), dtype=np.int32)
    data[0, :] = first
    for cell in range(1, num_cells):
        data[cell, :] = connectivity.links(cell)
    return data


def mesh_facet_vertices(mesh: Mesh) -> np.ndarray:
    """Return local facet->vertex connectivity array."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, 0)
    connectivity = mesh.topology.connectivity(fdim, 0)
    if connectivity is None:
        return np.empty((0, 0), dtype=np.int32)

    index_map = mesh.topology.index_map(fdim)
    if index_map is None:
        return np.empty((0, 0), dtype=np.int32)

    num_facets = int(index_map.size_local)
    if num_facets == 0:
        return np.empty((0, 0), dtype=np.int32)

    first = connectivity.links(0)
    verts_per_facet = len(first)
    data = np.zeros((num_facets, verts_per_facet), dtype=np.int32)
    data[0, :] = first
    for facet in range(1, num_facets):
        data[facet, :] = connectivity.links(facet)
    return data


def cell_midpoints(mesh: Mesh) -> np.ndarray:
    """Compute cell centroids from local connectivity."""
    coords = mesh_coordinates(mesh)
    c2v = mesh_cell_vertices(mesh)
    if c2v.size == 0:
        return np.empty((0, mesh.geometry.dim), dtype=coords.dtype)
    return coords[c2v].mean(axis=1)


def estimate_radius(mesh: Mesh) -> float:
    coords = mesh_coordinates(mesh)
    if coords.size == 0:
        return 0.0
    center = coords.mean(axis=0)
    return float(np.linalg.norm(coords - center, axis=1).max())


def build_eit_mesh(
    mesh: Mesh,
    *,
    facet_tags: MeshTags,
    cell_tags: Optional[MeshTags] = None,
    association_table: Optional[Dict[str, int]] = None,
    physical_groups: Optional[Dict[str, Any]] = None,
    radius: Optional[float] = None,
    mesh_file: Optional[str] = None,
    electrode_vertices: Optional[list[np.ndarray]] = None,
    mesh_family: Optional[str] = None,
    geometry_version: Optional[str] = None,
    generator_revision: Optional[str] = None,
    structured_sidecar_file: Optional[str] = None,
    structured_sidecar_version: Optional[str] = None,
) -> EITMesh:
    """Build a strongly-typed :class:`EITMesh` container."""
    resolved_radius = float(radius) if radius is not None else estimate_radius(mesh)
    return EITMesh(
        mesh=mesh,
        facet_tags=facet_tags,
        cell_tags=cell_tags,
        association_table=association_table or {},
        physical_groups=physical_groups or {},
        radius=resolved_radius,
        mesh_file=mesh_file,
        electrode_vertices=electrode_vertices,
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        generator_revision=generator_revision,
        structured_sidecar_file=structured_sidecar_file,
        structured_sidecar_version=structured_sidecar_version,
    )


def create_ds_measure(mesh: Mesh, facet_tags: MeshTags) -> ufl.Measure:
    """Create boundary measure for electrode assembly."""
    return ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)


def function_get_array(fn: fem.Function) -> np.ndarray:
    return fn.x.array


def function_set_array(fn: fem.Function, values: np.ndarray) -> None:
    fn.x.array[:] = np.asarray(values).ravel()


def function_size(fn: fem.Function) -> int:
    return int(fn.x.array.size)
