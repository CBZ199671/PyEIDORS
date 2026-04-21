"""Graph-Laplacian priors for coarse inverse meshes."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from pyeidors.inverse.dual_mesh import CellMesh, VoxelGrid


def graph_laplacian(mesh: Any, *, weight: str = "unit") -> sparse.csr_matrix:
    """Build a cell-neighbour graph Laplacian for an inverse mesh.

    The operator is defined on coarse inverse cells. For simplex-like cell
    meshes, cells are adjacent when they share a facet
    (``vertices_per_cell - 1`` common vertices). For ``VoxelGrid`` inputs,
    face-neighbour adjacency is generated directly from the grid shape.
    """

    resolved_weight = str(weight).strip().lower()
    if resolved_weight not in {"unit", "volume"}:
        raise ValueError("weight must be 'unit' or 'volume'.")
    if isinstance(mesh, VoxelGrid):
        edges = _voxel_edges(mesh.shape)
        n_cells = mesh.num_cells()
        volumes = np.full(n_cells, float(np.prod(mesh.spacing)), dtype=np.float64)
    else:
        cells = _cells(mesh)
        n_cells = int(cells.shape[0])
        edges = _cell_edges_from_shared_facets(cells)
        volumes = _cell_volumes(mesh, cells, n_cells)
    return _laplacian_from_edges(
        n_cells, edges, volumes=volumes, weight=resolved_weight
    )


def _cells(mesh: Any) -> np.ndarray:
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
    if np.any(cells < 0):
        raise ValueError("mesh cells contain negative vertex indices.")
    return cells


def _coordinates(mesh: Any) -> np.ndarray | None:
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


def _voxel_edges(shape: tuple[int, ...]) -> list[tuple[int, int]]:
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


def _cell_edges_from_shared_facets(cells: np.ndarray) -> list[tuple[int, int]]:
    required_shared = max(1, int(cells.shape[1] - 1))
    facets: dict[tuple[int, ...], int] = {}
    edges: set[tuple[int, int]] = set()
    for cell_idx, cell in enumerate(cells):
        vertices = tuple(int(v) for v in cell)
        if len(vertices) <= 1:
            continue
        if required_shared == len(vertices):
            keys = [tuple(sorted(vertices))]
        else:
            keys = []
            for drop_idx in range(len(vertices)):
                face = vertices[:drop_idx] + vertices[drop_idx + 1 :]
                keys.append(tuple(sorted(face)))
        for key in keys:
            previous = facets.get(key)
            if previous is None:
                facets[key] = int(cell_idx)
            else:
                edges.add(tuple(sorted((previous, int(cell_idx)))))
    return sorted(edges)


def _cell_volumes(mesh: Any, cells: np.ndarray, n_cells: int) -> np.ndarray:
    coords = _coordinates(mesh)
    if coords is None:
        return np.ones(n_cells, dtype=np.float64)
    if int(cells.max()) >= coords.shape[0]:
        raise ValueError("mesh cells reference missing coordinates.")
    volumes = np.ones(n_cells, dtype=np.float64)
    for idx, cell in enumerate(cells):
        vertices = coords[cell]
        local_dim = int(vertices.shape[0] - 1)
        if local_dim <= 0 or local_dim > vertices.shape[1]:
            volumes[idx] = 1.0
            continue
        basis = (vertices[1:] - vertices[0]).T
        gram = basis.T @ basis
        det = max(float(np.linalg.det(gram)), 0.0)
        volumes[idx] = np.sqrt(det)
    return np.maximum(volumes, np.finfo(np.float64).eps)


def _laplacian_from_edges(
    n_cells: int,
    edges: list[tuple[int, int]],
    *,
    volumes: np.ndarray,
    weight: str,
) -> sparse.csr_matrix:
    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    degree = np.zeros(n_cells, dtype=np.float64)
    for i, j in edges:
        if weight == "volume":
            edge_weight = 2.0 / (float(volumes[i]) + float(volumes[j]))
        else:
            edge_weight = 1.0
        degree[i] += edge_weight
        degree[j] += edge_weight
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([-edge_weight, -edge_weight])
    rows.extend(range(n_cells))
    cols.extend(range(n_cells))
    data.extend(degree.tolist())
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))


__all__ = ["graph_laplacian"]
