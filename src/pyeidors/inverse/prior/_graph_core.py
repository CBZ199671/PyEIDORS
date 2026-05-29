"""Shared graph-operator primitives for inverse priors and regularizers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.inverse.dual_mesh import CellMesh, VoxelGrid


def resolve_graph_weight(weight: str) -> str:
    """Normalize and validate supported graph edge weighting modes."""

    resolved = str(weight).strip().lower()
    if resolved not in {"unit", "volume"}:
        raise ValueError("weight must be 'unit' or 'volume'.")
    return resolved


def graph_edges_and_volumes(mesh: Any) -> tuple[int, list[tuple[int, int]], np.ndarray]:
    """Return ``(n_cells, edges, volumes)`` for supported inverse meshes."""

    if isinstance(mesh, VoxelGrid):
        n_cells = mesh.num_cells()
        volumes = np.full(n_cells, float(np.prod(mesh.spacing)), dtype=np.float64)
        return n_cells, voxel_edges(mesh.shape), volumes

    cells = extract_cells(mesh)
    n_cells = int(cells.shape[0])
    coords = extract_coordinates(mesh)
    return (
        n_cells,
        shared_facet_edges(cells, coordinates=coords),
        cell_volumes(mesh, cells, n_cells),
    )


def extract_cells(mesh: Any) -> np.ndarray:
    """Extract dense cell connectivity from CellMesh-like objects."""

    if isinstance(mesh, CellMesh):
        cells = mesh.cells
    else:
        attr = getattr(mesh, "cells", None)
        cells = attr() if callable(attr) else attr
    if cells is None:
        raise TypeError(f"Cannot extract cells from mesh type {type(mesh)!r}.")
    cells = np.asarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[0] == 0 or cells.shape[1] == 0:
        raise ValueError("mesh cells must be a non-empty 2D array.")
    if int(np.min(cells, initial=0)) < 0:
        raise ValueError("mesh cells contain negative vertex indices.")
    return cells


def extract_coordinates(mesh: Any) -> np.ndarray | None:
    """Extract vertex coordinates when available."""

    if isinstance(mesh, CellMesh):
        return np.asarray(mesh.coordinates, dtype=np.float64)
    attr = getattr(mesh, "coordinates", None)
    if callable(attr):
        return np.asarray(attr(), dtype=np.float64)
    if attr is not None:
        return np.asarray(attr, dtype=np.float64)
    geometry = getattr(mesh, "geometry", None)
    if geometry is not None and hasattr(geometry, "x"):
        dim = int(getattr(geometry, "dim", np.asarray(geometry.x).shape[1]))
        return np.asarray(geometry.x, dtype=np.float64)[:, :dim]
    return None


def voxel_edges(shape: tuple[int, ...]) -> list[tuple[int, int]]:
    """Build face-neighbour edges for a structured voxel grid."""

    edges: list[tuple[int, int]] = []
    ranges = [range(int(v)) for v in shape]
    for multi in np.ndindex(*[len(r) for r in ranges]):
        idx = int(np.ravel_multi_index(multi, shape, order="C"))
        for axis in range(len(shape)):
            if multi[axis] + 1 >= shape[axis]:
                continue
            neighbour = list(multi)
            neighbour[axis] += 1
            jdx = int(np.ravel_multi_index(tuple(neighbour), shape, order="C"))
            edges.append((idx, jdx))
    return edges


def shared_facet_edges(
    cells: np.ndarray,
    *,
    coordinates: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Build cell edges from shared facets in cell connectivity."""

    required_shared = _minimum_shared_facet_vertices(cells, coordinates=coordinates)
    facets: dict[tuple[int, ...], int] = {}
    edges: set[tuple[int, int]] = set()
    for cell_idx, cell in enumerate(cells):
        vertices = tuple(int(v) for v in cell)
        if len(vertices) <= 1:
            continue
        if required_shared >= len(vertices):
            keys = [tuple(sorted(vertices))]
        else:
            keys = [
                tuple(sorted(face))
                for face in _vertex_combinations(vertices, required_shared)
            ]
        for key in keys:
            previous = facets.get(key)
            if previous is None:
                facets[key] = int(cell_idx)
            else:
                edges.add(tuple(sorted((previous, int(cell_idx)))))
    return sorted(edges)


def _minimum_shared_facet_vertices(
    cells: np.ndarray,
    *,
    coordinates: np.ndarray | None = None,
) -> int:
    vertices_per_cell = int(cells.shape[1])
    if vertices_per_cell <= 2:
        return 1
    if coordinates is not None:
        coords = np.asarray(coordinates)
        if coords.ndim == 2 and coords.shape[1] > 0:
            dim = int(coords.shape[1])
            if dim <= 1:
                return 1
            if dim == 2:
                return 2
            if vertices_per_cell >= 8:
                return 4
            return 3
    if vertices_per_cell >= 6:
        return 3
    return max(1, vertices_per_cell - 1)


def _vertex_combinations(
    vertices: tuple[int, ...],
    size: int,
) -> list[tuple[int, ...]]:
    if size <= 0:
        return [()]
    if size > len(vertices):
        return []
    if size == len(vertices):
        return [vertices]
    out: list[tuple[int, ...]] = []

    def visit(start: int, prefix: tuple[int, ...]) -> None:
        remaining = size - len(prefix)
        if remaining == 0:
            out.append(prefix)
            return
        stop = len(vertices) - remaining + 1
        for idx in range(start, stop):
            visit(idx + 1, prefix + (vertices[idx],))

    visit(0, ())
    return out


def cell_volumes(mesh: Any, cells: np.ndarray, n_cells: int) -> np.ndarray:
    """Estimate per-cell volumes used by volume-weighted graph operators."""

    coords = extract_coordinates(mesh)
    if coords is None:
        return np.ones(n_cells, dtype=np.float64)
    if int(cells.max()) >= coords.shape[0]:
        raise ValueError("mesh cells reference missing coordinates.")
    volumes = np.ones(n_cells, dtype=np.float64)
    coord_dim = int(coords.shape[1])
    max_local_dim = min(max(int(cells.shape[1]) - 1, 0), coord_dim)
    cell_vertices = np.empty((cells.shape[1], coord_dim), dtype=np.float64)
    basis = np.empty((coord_dim, max_local_dim), dtype=np.float64)
    gram = np.empty((max_local_dim, max_local_dim), dtype=np.float64)
    for idx, cell in enumerate(cells):
        local_dim = int(cell.size - 1)
        if local_dim <= 0 or local_dim > coord_dim:
            volumes[idx] = 1.0
            continue
        _fill_cell_vertices(cell_vertices, coords, cell)
        active_gram = _simplex_gram_from_vertices(
            cell_vertices,
            local_dim=local_dim,
            basis=basis,
            gram=gram,
        )
        det = max(float(np.linalg.det(active_gram)), 0.0)
        volumes[idx] = np.sqrt(det)
    return np.maximum(volumes, np.finfo(np.float64).eps)


def _fill_cell_vertices(out: np.ndarray, coords: np.ndarray, cell: np.ndarray) -> None:
    for local_idx, vertex_idx in enumerate(cell):
        out[local_idx, :] = coords[int(vertex_idx)]


def _simplex_gram_from_vertices(
    vertices: np.ndarray,
    *,
    local_dim: int,
    basis: np.ndarray,
    gram: np.ndarray,
) -> np.ndarray:
    coord_dim = int(vertices.shape[1])
    active_basis = basis[:coord_dim, :local_dim]
    origin = vertices[0, :coord_dim]
    for basis_idx in range(local_dim):
        np.subtract(
            vertices[basis_idx + 1, :coord_dim],
            origin,
            out=active_basis[:, basis_idx],
        )
    active_gram = gram[:local_dim, :local_dim]
    np.matmul(active_basis.T, active_basis, out=active_gram)
    return active_gram


def dolfinx_facet_edges(mesh: Any) -> list[tuple[int, int]]:
    """Build local facet-adjacent cell edges from a DOLFINx-like mesh."""

    tdim = mesh.topology.dim
    fdim = tdim - 1

    mesh.topology.create_connectivity(fdim, tdim)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    facet_map = mesh.topology.index_map(fdim)
    if facet_to_cell is None or facet_map is None:
        return []

    edges: list[tuple[int, int]] = []
    for facet in range(int(facet_map.size_local)):
        adjacent_cells = facet_to_cell.links(facet)
        if len(adjacent_cells) != 2:
            continue
        edges.append((int(adjacent_cells[0]), int(adjacent_cells[1])))
    return edges


def dolfinx_cell_difference_operator(mesh: Any, n_elements: int) -> sparse.csr_matrix:
    """Build the facet-adjacent difference operator for DOLFINx-like meshes."""

    n_cells = int(n_elements)
    edges = dolfinx_facet_edges(mesh)
    if not edges:
        return sparse.csr_matrix((0, n_cells), dtype=np.float64)
    return difference_from_edges(
        n_cells,
        edges,
        volumes=np.ones(n_cells, dtype=np.float64),
        weight="unit",
    )


def laplacian_from_edges(
    n_cells: int,
    edges: list[tuple[int, int]],
    *,
    volumes: np.ndarray,
    weight: str,
) -> sparse.csr_matrix:
    """Build a graph Laplacian from weighted undirected edges."""

    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    degree = np.zeros(n_cells, dtype=np.float64)
    for i, j in edges:
        edge_weight = _edge_weight(i, j, volumes=volumes, weight=weight)
        degree[i] += edge_weight
        degree[j] += edge_weight
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-edge_weight, -edge_weight])
    rows.extend(range(n_cells))
    cols.extend(range(n_cells))
    data.extend(degree.tolist())
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))


def difference_from_edges(
    n_cells: int,
    edges: list[tuple[int, int]],
    *,
    volumes: np.ndarray,
    weight: str,
) -> sparse.csr_matrix:
    """Build an oriented graph difference operator from weighted edges."""

    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for row_idx, (i, j) in enumerate(edges):
        scale = float(np.sqrt(_edge_weight(i, j, volumes=volumes, weight=weight)))
        rows.extend([row_idx, row_idx])
        cols.extend([int(i), int(j)])
        data.extend([scale, -scale])
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(edges), n_cells),
        dtype=np.float64,
    )


def _edge_weight(i: int, j: int, *, volumes: np.ndarray, weight: str) -> float:
    if weight == "volume":
        return 2.0 / (float(volumes[i]) + float(volumes[j]))
    return 1.0
