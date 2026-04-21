"""Dual-mesh utilities for EIDORS-style 3D difference reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

OutsidePolicy = Literal["nearest", "raise"]


def _as_coordinates(values: Any, *, name: str) -> np.ndarray:
    coords = np.asarray(values, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"{name} coordinates must be a 2D array.")
    if coords.shape[0] == 0 or coords.shape[1] == 0:
        raise ValueError(f"{name} coordinates must be non-empty.")
    if not np.isfinite(coords).all():
        raise ValueError(f"{name} coordinates contain non-finite values.")
    return np.ascontiguousarray(coords, dtype=np.float64)


def _as_cells(values: Any, *, name: str) -> np.ndarray:
    cells = np.asarray(values, dtype=np.int64)
    if cells.ndim != 2:
        raise ValueError(f"{name} cells must be a 2D connectivity array.")
    if cells.shape[0] == 0 or cells.shape[1] == 0:
        raise ValueError(f"{name} cells must be non-empty.")
    if np.any(cells < 0):
        raise ValueError(f"{name} cells contain negative vertex indices.")
    return np.ascontiguousarray(cells, dtype=np.int64)


@dataclass
class CellMesh:
    """Small array-backed cell mesh used by dual-mesh tests and adapters."""

    coordinates: np.ndarray
    cells: np.ndarray
    name: str = "cell-mesh"

    def __post_init__(self) -> None:
        self.coordinates = _as_coordinates(self.coordinates, name=self.name)
        self.cells = _as_cells(self.cells, name=self.name)
        if int(self.cells.max()) >= self.coordinates.shape[0]:
            raise ValueError(f"{self.name} cells reference missing coordinates.")

    def num_cells(self) -> int:
        return int(self.cells.shape[0])

    def cell_centers(self) -> np.ndarray:
        return self.coordinates[self.cells].mean(axis=1)


@dataclass
class VoxelGrid:
    """Axis-aligned coarse inverse voxel grid.

    ``shape`` is interpreted in axis-major order, so
    ``np.ravel_multi_index((ix, iy, iz), shape, order="C")`` defines the
    coarse parameter index.
    """

    origin: np.ndarray
    spacing: np.ndarray
    shape: tuple[int, ...]
    name: str = "voxel-grid"

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        self.spacing = np.asarray(self.spacing, dtype=np.float64).reshape(-1)
        self.shape = tuple(int(v) for v in self.shape)
        if not self.shape:
            raise ValueError("VoxelGrid shape must be non-empty.")
        if len(self.origin) != len(self.shape) or len(self.spacing) != len(self.shape):
            raise ValueError(
                "VoxelGrid origin, spacing, and shape dimensions must match."
            )
        if any(v <= 0 for v in self.shape):
            raise ValueError("VoxelGrid shape entries must be positive.")
        if np.any(~np.isfinite(self.origin)) or np.any(~np.isfinite(self.spacing)):
            raise ValueError("VoxelGrid origin and spacing must be finite.")
        if np.any(self.spacing <= 0.0):
            raise ValueError("VoxelGrid spacing entries must be positive.")

    @classmethod
    def from_bounds(
        cls,
        lower: Any,
        upper: Any,
        shape: tuple[int, ...],
        *,
        name: str = "voxel-grid",
    ) -> "VoxelGrid":
        lower_arr = np.asarray(lower, dtype=np.float64).reshape(-1)
        upper_arr = np.asarray(upper, dtype=np.float64).reshape(-1)
        shape_tuple = tuple(int(v) for v in shape)
        if lower_arr.shape != upper_arr.shape:
            raise ValueError("VoxelGrid bounds must have matching dimensions.")
        if len(shape_tuple) != lower_arr.size:
            raise ValueError("VoxelGrid bounds and shape dimensions must match.")
        if np.any(upper_arr <= lower_arr):
            raise ValueError("VoxelGrid upper bounds must exceed lower bounds.")
        spacing = (upper_arr - lower_arr) / np.asarray(shape_tuple, dtype=np.float64)
        return cls(origin=lower_arr, spacing=spacing, shape=shape_tuple, name=name)

    @property
    def dimension(self) -> int:
        return int(len(self.shape))

    def num_cells(self) -> int:
        return int(np.prod(self.shape))

    def cell_centers(self) -> np.ndarray:
        axes = [
            self.origin[axis]
            + (np.arange(size, dtype=np.float64) + 0.5) * self.spacing[axis]
            for axis, size in enumerate(self.shape)
        ]
        grids = np.meshgrid(*axes, indexing="ij")
        return np.stack([grid.ravel(order="C") for grid in grids], axis=1)

    def locate_points(
        self,
        points: Any,
        *,
        outside: OutsidePolicy = "nearest",
    ) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2:
            raise ValueError("points must be a 2D array.")
        if pts.shape[1] < self.dimension:
            raise ValueError("points have fewer dimensions than the voxel grid.")
        pts = pts[:, : self.dimension]
        scaled = np.floor((pts - self.origin) / self.spacing).astype(np.int64)
        upper = np.asarray(self.shape, dtype=np.int64)
        inside = np.all((scaled >= 0) & (scaled < upper), axis=1)

        indices = np.empty(pts.shape[0], dtype=np.int64)
        if np.any(inside):
            inside_scaled = scaled[inside].T
            indices[inside] = np.ravel_multi_index(
                tuple(inside_scaled),
                self.shape,
                order="C",
            )
        if np.any(~inside):
            if outside == "raise":
                raise ValueError(
                    "Some fine-cell centers lie outside the coarse voxel grid."
                )
            if outside != "nearest":
                raise ValueError("outside must be 'nearest' or 'raise'.")
            tree = cKDTree(self.cell_centers())
            _, nearest = tree.query(pts[~inside])
            indices[~inside] = np.asarray(nearest, dtype=np.int64)
        return indices


@dataclass
class DualMesh:
    """Fine forward mesh + coarse inverse mesh with a sparse projection map."""

    fine_mesh: Any
    coarse_mesh: Any
    projection: sparse.spmatrix | None = None
    method: str = "piecewise_constant"
    outside: OutsidePolicy = "nearest"

    def __post_init__(self) -> None:
        if self.projection is None:
            projection = coarse2fine(
                self.fine_mesh,
                self.coarse_mesh,
                method=self.method,
                outside=self.outside,
            )
        else:
            projection = sparse.csr_matrix(self.projection, dtype=np.float64)
        n_fine = _num_cells(self.fine_mesh)
        n_coarse = _num_cells(self.coarse_mesh)
        if projection.shape != (n_fine, n_coarse):
            raise ValueError(
                "coarse2fine projection shape mismatch: "
                f"got {projection.shape}, expected {(n_fine, n_coarse)}."
            )
        self.projection = projection.tocsr()

    @property
    def n_fine_cells(self) -> int:
        return int(self.projection.shape[0])

    @property
    def n_coarse_cells(self) -> int:
        return int(self.projection.shape[1])

    @property
    def coarse2fine(self) -> sparse.csr_matrix:
        return self.projection

    def project_to_fine(self, coarse_values: Any) -> np.ndarray:
        values = np.asarray(coarse_values, dtype=np.float64).reshape(-1)
        if values.size != self.n_coarse_cells:
            raise ValueError(
                f"Expected {self.n_coarse_cells} coarse values, got {values.size}."
            )
        return np.asarray(self.projection @ values, dtype=np.float64)

    def restrict_to_coarse(
        self, fine_values: Any, *, average: bool = True
    ) -> np.ndarray:
        values = np.asarray(fine_values, dtype=np.float64).reshape(-1)
        if values.size != self.n_fine_cells:
            raise ValueError(
                f"Expected {self.n_fine_cells} fine values, got {values.size}."
            )
        restricted = np.asarray(self.projection.T @ values, dtype=np.float64)
        if average:
            counts = np.asarray(self.projection.sum(axis=0)).reshape(-1)
            restricted = restricted / np.maximum(counts, 1.0)
        return restricted

    def summary(self) -> dict[str, Any]:
        return {
            "n_fine_cells": self.n_fine_cells,
            "n_coarse_cells": self.n_coarse_cells,
            "projection_nnz": int(self.projection.nnz),
            "method": self.method,
            "outside": self.outside,
        }


def coarse2fine(
    mesh_fine: Any,
    mesh_coarse: Any,
    *,
    method: str = "piecewise_constant",
    outside: OutsidePolicy = "nearest",
    containment_tol: float = 1e-9,
) -> sparse.csr_matrix:
    """Build a sparse map from coarse inverse parameters to fine cells.

    Each fine cell receives exactly one coarse parameter in the current
    piecewise-constant implementation. This keeps the first v1 dual-model
    slice explicit and predictable; higher-order interpolation can be added
    later without changing the public surface.
    """

    if method != "piecewise_constant":
        raise ValueError("Only method='piecewise_constant' is currently supported.")
    fine_centers = _cell_centers(mesh_fine)
    if fine_centers.shape[0] == 0:
        raise ValueError("mesh_fine must contain at least one cell.")

    if isinstance(mesh_coarse, VoxelGrid):
        coarse_indices = mesh_coarse.locate_points(fine_centers, outside=outside)
        n_coarse = mesh_coarse.num_cells()
    else:
        coarse_indices = _locate_points_in_cell_mesh(
            fine_centers,
            mesh_coarse,
            outside=outside,
            containment_tol=containment_tol,
        )
        n_coarse = _num_cells(mesh_coarse)

    rows = np.arange(fine_centers.shape[0], dtype=np.int64)
    data = np.ones(fine_centers.shape[0], dtype=np.float64)
    return sparse.csr_matrix(
        (data, (rows, coarse_indices)),
        shape=(fine_centers.shape[0], n_coarse),
        dtype=np.float64,
    )


def _coordinates(mesh: Any) -> np.ndarray:
    if isinstance(mesh, VoxelGrid):
        return mesh.cell_centers()
    if isinstance(mesh, CellMesh):
        return mesh.coordinates

    attr = getattr(mesh, "coordinates", None)
    if callable(attr):
        return _as_coordinates(attr(), name=type(mesh).__name__)
    if attr is not None:
        return _as_coordinates(attr, name=type(mesh).__name__)

    geometry = getattr(mesh, "geometry", None)
    if geometry is not None and hasattr(geometry, "x"):
        dim = int(getattr(geometry, "dim", np.asarray(geometry.x).shape[1]))
        return _as_coordinates(
            np.asarray(geometry.x)[:, :dim], name=type(mesh).__name__
        )

    raise TypeError(f"Cannot extract coordinates from mesh type {type(mesh)!r}.")


def _cells(mesh: Any) -> np.ndarray:
    if isinstance(mesh, VoxelGrid):
        raise TypeError("VoxelGrid does not expose vertex connectivity.")
    if isinstance(mesh, CellMesh):
        return mesh.cells

    attr = getattr(mesh, "cells", None)
    if callable(attr):
        cells = attr()
        return _as_cells(cells, name=type(mesh).__name__)
    if attr is not None:
        return _as_cells(attr, name=type(mesh).__name__)

    topology = getattr(mesh, "topology", None)
    if topology is not None:
        tdim = int(topology.dim)
        create = getattr(topology, "create_connectivity", None)
        if callable(create):
            create(tdim, 0)
        connectivity = topology.connectivity(tdim, 0)
        if connectivity is None:
            raise ValueError("Mesh topology has no cell-to-vertex connectivity.")
        index_map = topology.index_map(tdim)
        n_cells = int(index_map.size_local if index_map is not None else 0)
        if n_cells <= 0:
            raise ValueError("Mesh topology has no local cells.")
        first = np.asarray(connectivity.links(0), dtype=np.int64)
        cells = np.empty((n_cells, first.size), dtype=np.int64)
        cells[0] = first
        for idx in range(1, n_cells):
            cells[idx] = np.asarray(connectivity.links(idx), dtype=np.int64)
        return _as_cells(cells, name=type(mesh).__name__)

    raise TypeError(f"Cannot extract cells from mesh type {type(mesh)!r}.")


def _num_cells(mesh: Any) -> int:
    if isinstance(mesh, VoxelGrid):
        return mesh.num_cells()
    attr = getattr(mesh, "num_cells", None)
    if callable(attr):
        return int(attr())
    return int(_cells(mesh).shape[0])


def _cell_centers(mesh: Any) -> np.ndarray:
    attr = getattr(mesh, "cell_centers", None)
    if callable(attr):
        centers = attr()
        return _as_coordinates(centers, name=f"{type(mesh).__name__}.cell_centers")
    coords = _coordinates(mesh)
    cells = _cells(mesh)
    if int(cells.max()) >= coords.shape[0]:
        raise ValueError("Mesh cells reference missing coordinates.")
    return np.asarray(coords[cells].mean(axis=1), dtype=np.float64)


def _locate_points_in_cell_mesh(
    points: np.ndarray,
    mesh: Any,
    *,
    outside: OutsidePolicy,
    containment_tol: float,
) -> np.ndarray:
    coords = _coordinates(mesh)
    if points.shape[1] < coords.shape[1]:
        raise ValueError(
            "Fine-cell centers have fewer dimensions than the coarse mesh."
        )
    search_points = points[:, : coords.shape[1]]
    cells = _cells(mesh)
    if int(cells.max()) >= coords.shape[0]:
        raise ValueError("Coarse mesh cells reference missing coordinates.")
    centers = _cell_centers(mesh)
    tree = cKDTree(centers)
    out = np.empty(search_points.shape[0], dtype=np.int64)
    cell_vertices = coords[cells]
    mins = cell_vertices.min(axis=1) - containment_tol
    maxs = cell_vertices.max(axis=1) + containment_tol

    for point_idx, point in enumerate(search_points):
        candidates = np.flatnonzero(np.all((point >= mins) & (point <= maxs), axis=1))
        match = -1
        for cell_idx in candidates:
            if _point_in_simplex(point, cell_vertices[cell_idx], tol=containment_tol):
                match = int(cell_idx)
                break
        if match >= 0:
            out[point_idx] = match
            continue
        if outside == "raise":
            raise ValueError("Some fine-cell centers lie outside the coarse cell mesh.")
        if outside != "nearest":
            raise ValueError("outside must be 'nearest' or 'raise'.")
        _, nearest = tree.query(point)
        out[point_idx] = int(nearest)
    return out


def _point_in_simplex(point: np.ndarray, vertices: np.ndarray, *, tol: float) -> bool:
    local_dim = int(vertices.shape[0] - 1)
    if local_dim <= 0:
        return False
    gdim = int(vertices.shape[1])
    if local_dim > gdim:
        return False
    origin = vertices[0]
    basis = (vertices[1:] - origin).T
    rhs = point[:gdim] - origin
    try:
        bary_tail, *_ = np.linalg.lstsq(basis, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return False
    reconstructed = origin + basis @ bary_tail
    scale = max(
        1.0, float(np.linalg.norm(point[:gdim])), float(np.linalg.norm(vertices))
    )
    if float(np.linalg.norm(reconstructed - point[:gdim])) > tol * scale * 10.0:
        return False
    bary = np.concatenate(([1.0 - float(np.sum(bary_tail))], bary_tail))
    return bool(np.all(bary >= -tol) and np.all(bary <= 1.0 + tol))


__all__ = [
    "CellMesh",
    "DualMesh",
    "VoxelGrid",
    "coarse2fine",
]
