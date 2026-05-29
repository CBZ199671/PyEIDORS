"""HDF5 cache for mesh-derived geometry arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from pyeidors.cache.keys import hash_array
from pyeidors.io.hdf5_artifacts import (
    read_hdf5_artifact,
    write_large_cache_hdf5_artifact,
)


MESH_DERIVED_SCHEMA = "pyeidors-mesh-derived-hdf5-v1"
DEFAULT_VISUAL_POINT_CLOUD_MAX_POINTS = 60_000


@dataclass(frozen=True)
class MeshDerivedArrays:
    """Mesh arrays that are expensive or repetitive to derive."""

    node_coords: np.ndarray
    cell_connectivity: np.ndarray
    cell_centers: np.ndarray
    cell_measures: np.ndarray
    metadata: MappingProxyType
    path: str | None = None
    electrode_vertices: np.ndarray | None = None
    electrode_vertex_offsets: np.ndarray | None = None
    visual_point_cloud_centers: np.ndarray | None = None
    visual_point_cloud_cell_indices: np.ndarray | None = None

    @property
    def n_cells(self) -> int:
        return int(self.cell_connectivity.shape[0])

    @property
    def gdim(self) -> int:
        return int(self.node_coords.shape[1]) if self.node_coords.ndim == 2 else 0


def mesh_derived_signature_payload(mesh: Any) -> dict[str, Any]:
    """Return the content signature payload for derived mesh arrays."""

    coords, cells, tdim = _mesh_geometry_arrays(mesh)
    electrode_vertices, electrode_offsets = _mesh_electrode_vertex_arrays(mesh, coords)
    return _mesh_derived_signature_payload_from_arrays(
        coords,
        cells,
        tdim,
        electrode_vertices=electrode_vertices,
        electrode_vertex_offsets=electrode_offsets,
    )


def _mesh_derived_signature_payload_from_arrays(
    coords: np.ndarray,
    cells: np.ndarray,
    tdim: int,
    *,
    electrode_vertices: np.ndarray | None = None,
    electrode_vertex_offsets: np.ndarray | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": MESH_DERIVED_SCHEMA,
        "tdim": int(tdim),
        "gdim": int(coords.shape[1]) if coords.ndim == 2 else 0,
        "geometry_dtype": str(coords.dtype),
        "cell_dtype": str(cells.dtype),
        "n_nodes": int(coords.shape[0]),
        "n_cells": int(cells.shape[0]),
        "vertices_per_cell": int(cells.shape[1]) if cells.ndim == 2 else 0,
        "geometry_hash": hash_array(coords),
        "cell_connectivity_hash": hash_array(cells),
    }
    if electrode_vertices is not None and electrode_vertices.size:
        payload.update(
            {
                "electrode_vertices_hash": hash_array(electrode_vertices),
                "electrode_vertex_offsets_hash": hash_array(
                    np.asarray(electrode_vertex_offsets, dtype=np.int64)
                ),
                "electrode_vertex_count": int(electrode_vertices.shape[0]),
                "electrode_patch_count": max(
                    0, int(np.asarray(electrode_vertex_offsets).size) - 1
                ),
            }
        )
    return payload


def mesh_derived_signature(mesh: Any) -> str:
    """Return a stable SHA-256 signature for mesh-derived geometry arrays."""

    return _mesh_derived_signature_from_payload(mesh_derived_signature_payload(mesh))


def _mesh_derived_signature_from_payload(payload: Mapping[str, Any]) -> str:
    from pyeidors.cache.object_signature import stable_signature_hash

    return stable_signature_hash(payload)


def mesh_derived_cache_path(
    cache_dir: str | Path,
    signature: str,
) -> Path:
    """Return the canonical HDF5 path for a derived mesh signature."""

    return Path(cache_dir) / "mesh_derived" / f"{str(signature)}.h5"


def build_mesh_derived_arrays(
    mesh: Any,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> MeshDerivedArrays:
    """Build derived mesh arrays from a DOLFINx-like mesh or EITMesh."""

    coords, cells, tdim = _mesh_geometry_arrays(mesh)
    electrode_vertices, electrode_offsets = _mesh_electrode_vertex_arrays(mesh, coords)
    payload = _mesh_derived_signature_payload_from_arrays(
        coords,
        cells,
        tdim,
        electrode_vertices=electrode_vertices,
        electrode_vertex_offsets=electrode_offsets,
    )
    signature = _mesh_derived_signature_from_payload(payload)
    return _build_mesh_derived_arrays_from_arrays(
        coords,
        cells,
        tdim,
        payload=payload,
        signature=signature,
        metadata=metadata,
        electrode_vertices=electrode_vertices,
        electrode_vertex_offsets=electrode_offsets,
    )


def _build_mesh_derived_arrays_from_arrays(
    coords: np.ndarray,
    cells: np.ndarray,
    tdim: int,
    *,
    payload: Mapping[str, Any],
    signature: str,
    metadata: Mapping[str, Any] | None,
    electrode_vertices: np.ndarray | None = None,
    electrode_vertex_offsets: np.ndarray | None = None,
) -> MeshDerivedArrays:
    if cells.size and int(cells.max()) >= coords.shape[0]:
        raise ValueError("Mesh cells reference missing coordinates.")
    centers = _cell_centers(coords, cells)
    measures = _cell_measures(coords, cells, tdim=tdim)
    point_centers, point_indices = _visual_point_cloud_arrays(
        centers,
        max_points=DEFAULT_VISUAL_POINT_CLOUD_MAX_POINTS,
    )
    meta = {
        "artifact_schema": MESH_DERIVED_SCHEMA,
        "artifact_format": "hdf5",
        "signature": signature,
        "signature_payload": dict(payload),
        "visual_point_cloud_max_points": DEFAULT_VISUAL_POINT_CLOUD_MAX_POINTS,
        "visual_point_cloud_count": int(point_indices.size),
        "electrode_patch_count": max(
            0, int(np.asarray(electrode_vertex_offsets).size) - 1
        )
        if electrode_vertex_offsets is not None
        else 0,
    }
    if metadata:
        meta.update(dict(metadata))
    return MeshDerivedArrays(
        node_coords=np.asarray(coords),
        cell_connectivity=np.asarray(cells, dtype=np.int32),
        cell_centers=np.asarray(centers, dtype=coords.dtype),
        cell_measures=np.asarray(measures, dtype=coords.dtype),
        metadata=MappingProxyType(meta),
        electrode_vertices=electrode_vertices,
        electrode_vertex_offsets=electrode_vertex_offsets,
        visual_point_cloud_centers=point_centers,
        visual_point_cloud_cell_indices=point_indices,
    )


def write_mesh_derived_artifact(
    path: str | Path,
    derived: MeshDerivedArrays,
) -> Path:
    """Persist derived mesh arrays as a chunked HDF5 artifact."""

    arrays = {
        "node_coords": derived.node_coords,
        "cell_connectivity": derived.cell_connectivity,
        "cell_centers": derived.cell_centers,
        "cell_measures": derived.cell_measures,
    }
    if derived.electrode_vertices is not None:
        arrays["electrode_vertices"] = derived.electrode_vertices
    if derived.electrode_vertex_offsets is not None:
        arrays["electrode_vertex_offsets"] = derived.electrode_vertex_offsets
    if derived.visual_point_cloud_centers is not None:
        arrays["visual_point_cloud_centers"] = derived.visual_point_cloud_centers
    if derived.visual_point_cloud_cell_indices is not None:
        arrays["visual_point_cloud_cell_indices"] = (
            derived.visual_point_cloud_cell_indices
        )
    return write_large_cache_hdf5_artifact(
        path,
        arrays,
        dict(derived.metadata),
        schema=MESH_DERIVED_SCHEMA,
    )


def load_mesh_derived_artifact(path: str | Path) -> MeshDerivedArrays:
    """Load a derived mesh HDF5 artifact eagerly."""

    artifact = read_hdf5_artifact(path, verify_checksums=True)
    arrays = artifact.arrays
    required = {"node_coords", "cell_connectivity", "cell_centers", "cell_measures"}
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"Mesh-derived artifact missing arrays: {', '.join(missing)}")
    metadata = dict(artifact.metadata)
    metadata.setdefault("artifact_schema", artifact.schema)
    return MeshDerivedArrays(
        node_coords=np.asarray(arrays["node_coords"]),
        cell_connectivity=np.asarray(arrays["cell_connectivity"], dtype=np.int32),
        cell_centers=np.asarray(arrays["cell_centers"]),
        cell_measures=np.asarray(arrays["cell_measures"]),
        metadata=MappingProxyType(metadata),
        path=str(Path(path)),
        electrode_vertices=np.asarray(arrays["electrode_vertices"])
        if "electrode_vertices" in arrays
        else None,
        electrode_vertex_offsets=np.asarray(arrays["electrode_vertex_offsets"])
        if "electrode_vertex_offsets" in arrays
        else None,
        visual_point_cloud_centers=np.asarray(arrays["visual_point_cloud_centers"])
        if "visual_point_cloud_centers" in arrays
        else None,
        visual_point_cloud_cell_indices=np.asarray(
            arrays["visual_point_cloud_cell_indices"], dtype=np.int64
        )
        if "visual_point_cloud_cell_indices" in arrays
        else None,
    )


def load_or_build_mesh_derived_artifact(
    mesh: Any,
    *,
    cache_dir: str | Path = ".pyeidors_cache/v2",
    refresh: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> MeshDerivedArrays:
    """Load the derived mesh artifact for ``mesh`` or build it once."""

    coords, cells, tdim = _mesh_geometry_arrays(mesh)
    electrode_vertices, electrode_offsets = _mesh_electrode_vertex_arrays(mesh, coords)
    payload = _mesh_derived_signature_payload_from_arrays(
        coords,
        cells,
        tdim,
        electrode_vertices=electrode_vertices,
        electrode_vertex_offsets=electrode_offsets,
    )
    signature = _mesh_derived_signature_from_payload(payload)
    path = mesh_derived_cache_path(cache_dir, signature)
    if path.exists() and not refresh:
        return load_mesh_derived_artifact(path)
    derived = _build_mesh_derived_arrays_from_arrays(
        coords,
        cells,
        tdim,
        payload=payload,
        signature=signature,
        metadata=metadata,
        electrode_vertices=electrode_vertices,
        electrode_vertex_offsets=electrode_offsets,
    )
    written = write_mesh_derived_artifact(path, derived)
    return load_mesh_derived_artifact(written)


def _mesh_electrode_vertex_arrays(
    mesh: Any,
    coords: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    vertices = getattr(mesh, "electrode_vertices", None)
    if vertices is None:
        return None, None
    rows: list[np.ndarray] = []
    offsets = [0]
    gdim = int(coords.shape[1]) if coords.ndim == 2 else 0
    for item in vertices:
        arr = np.asarray(item, dtype=coords.dtype)
        if arr.size == 0:
            offsets.append(offsets[-1])
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            continue
        if gdim > 0:
            if arr.shape[1] < gdim:
                padded = np.zeros((arr.shape[0], gdim), dtype=coords.dtype)
                padded[:, : arr.shape[1]] = arr
                arr = padded
            elif arr.shape[1] > gdim:
                arr = arr[:, :gdim]
        rows.append(np.ascontiguousarray(arr))
        offsets.append(offsets[-1] + int(arr.shape[0]))
    if not rows:
        return None, None
    return (
        np.ascontiguousarray(np.vstack(rows), dtype=coords.dtype),
        np.asarray(offsets, dtype=np.int64),
    )


def _visual_point_cloud_arrays(
    centers: np.ndarray,
    *,
    max_points: int = DEFAULT_VISUAL_POINT_CLOUD_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.asarray(centers)
    n_cells = int(centers.shape[0]) if centers.ndim == 2 else 0
    if n_cells <= 0:
        return (
            np.empty(
                (0, centers.shape[1] if centers.ndim == 2 else 0), dtype=centers.dtype
            ),
            np.empty((0,), dtype=np.int64),
        )
    limit = max(0, int(max_points))
    if limit == 0 or n_cells <= limit:
        indices = np.arange(n_cells, dtype=np.int64)
    else:
        indices = np.linspace(0, n_cells - 1, limit, dtype=np.int64)
        indices = np.unique(indices)
    return np.ascontiguousarray(centers[indices]), np.ascontiguousarray(indices)


def _mesh_geometry_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray, int]:
    coords = _mesh_coordinates(mesh)
    cells = _mesh_cells(mesh)
    return coords, cells, _mesh_topology_dim(mesh, cells, coords=coords)


def _mesh_coordinates(mesh: Any) -> np.ndarray:
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


def _mesh_cells(mesh: Any) -> np.ndarray:
    attr = getattr(mesh, "cells", None)
    if callable(attr):
        return _as_cells(attr(), name=type(mesh).__name__)
    if attr is not None:
        return _as_cells(attr, name=type(mesh).__name__)
    topology = getattr(mesh, "topology", None)
    if topology is None:
        raise TypeError(f"Cannot extract cells from mesh type {type(mesh)!r}.")
    tdim = int(topology.dim)
    create = getattr(topology, "create_connectivity", None)
    if callable(create):
        create(tdim, 0)
    connectivity = topology.connectivity(tdim, 0)
    if connectivity is None:
        return np.empty((0, 0), dtype=np.int32)
    index_map = topology.index_map(tdim)
    n_cells = int(index_map.size_local if index_map is not None else 0)
    if n_cells <= 0:
        return np.empty((0, 0), dtype=np.int32)
    flat = getattr(connectivity, "array", None)
    offsets = getattr(connectivity, "offsets", None)
    if flat is not None and offsets is not None:
        offset_arr = np.asarray(offsets)
        if offset_arr.size >= n_cells + 1:
            widths = np.diff(offset_arr[: n_cells + 1])
            if widths.size and np.all(widths == widths[0]):
                vertices_per_cell = int(widths[0])
                start = int(offset_arr[0])
                stop = int(offset_arr[n_cells])
                if (
                    vertices_per_cell >= 0
                    and stop - start == n_cells * vertices_per_cell
                ):
                    flat_arr = np.asarray(flat)
                    cells = np.asarray(flat_arr[start:stop], dtype=np.int32).reshape(
                        n_cells,
                        vertices_per_cell,
                    )
                    return _as_cells(cells, name=type(mesh).__name__)
    first = np.asarray(connectivity.links(0), dtype=np.int32)
    cells = np.empty((n_cells, first.size), dtype=np.int32)
    cells[0] = first
    for idx in range(1, n_cells):
        cells[idx] = np.asarray(connectivity.links(idx), dtype=np.int32)
    return _as_cells(cells, name=type(mesh).__name__)


def _mesh_topology_dim(
    mesh: Any, cells: np.ndarray, *, coords: np.ndarray | None = None
) -> int:
    topology = getattr(mesh, "topology", None)
    if topology is not None and getattr(topology, "dim", None) is not None:
        return int(topology.dim)
    if coords is None:
        coords = _mesh_coordinates(mesh)
    if cells.ndim == 2 and cells.shape[1] in {4, 8} and coords.shape[1] >= 3:
        return 3
    if cells.ndim == 2 and cells.shape[1] in {3, 4}:
        return 2
    return int(coords.shape[1]) if coords.ndim == 2 else 0


def _as_coordinates(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"{name} coordinates must be a 2D array.")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} coordinates must expose at least one dimension.")
    return np.ascontiguousarray(arr)


def _as_cells(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"{name} cells must be a 2D connectivity array.")
    if arr.size and int(arr.min()) < 0:
        raise ValueError(f"{name} cells must not contain negative vertex indices.")
    return np.ascontiguousarray(arr)


def _cell_centers(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    if cells.size == 0:
        return np.empty((0, coords.shape[1]), dtype=coords.dtype)
    centers = np.zeros((cells.shape[0], coords.shape[1]), dtype=coords.dtype)
    work = np.empty_like(centers)
    for local_idx in range(cells.shape[1]):
        np.take(coords, cells[:, local_idx], axis=0, out=work)
        centers += work
    centers /= float(cells.shape[1])
    return centers


def _cell_measures(coords: np.ndarray, cells: np.ndarray, *, tdim: int) -> np.ndarray:
    if cells.size == 0:
        return np.empty((0,), dtype=coords.dtype)
    measures = np.empty((cells.shape[0],), dtype=coords.dtype)
    if int(tdim) == 1:
        for cell_idx, cell in enumerate(cells):
            first_idx = int(cell[0])
            last_idx = int(cell[-1])
            measures[cell_idx] = np.linalg.norm(coords[last_idx] - coords[first_idx])
        return measures
    if int(tdim) == 2:
        cell_vertices = np.empty((cells.shape[1], coords.shape[1]), dtype=np.float64)
        for cell_idx, cell in enumerate(cells):
            _fill_cell_vertices(cell_vertices, coords, cell)
            measures[cell_idx] = _polygon_area(cell_vertices)
        return measures
    if int(tdim) == 3:
        if cells.shape[1] == 4:
            for cell_idx, cell in enumerate(cells):
                measures[cell_idx] = _tetra_volume_from_indices(coords, cell)
        elif cells.shape[1] == 8:
            cell_vertices = np.empty(
                (cells.shape[1], coords.shape[1]), dtype=np.float64
            )
            for cell_idx, cell in enumerate(cells):
                volume = _axis_aligned_hexa_volume_from_indices(coords, cell)
                if np.isfinite(volume):
                    measures[cell_idx] = volume
                else:
                    _fill_cell_vertices(cell_vertices, coords, cell)
                    measures[cell_idx] = _polyhedron_volume(cell_vertices)
        else:
            cell_vertices = np.empty(
                (cells.shape[1], coords.shape[1]), dtype=np.float64
            )
            for cell_idx, cell in enumerate(cells):
                _fill_cell_vertices(cell_vertices, coords, cell)
                measures[cell_idx] = _polyhedron_volume(cell_vertices)
        return measures
    measures.fill(1)
    return measures


def _fill_cell_vertices(out: np.ndarray, coords: np.ndarray, cell: np.ndarray) -> None:
    for local_idx, vertex_idx in enumerate(cell):
        out[local_idx, :] = coords[int(vertex_idx)]


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    if pts.shape[1] == 2:
        x = pts[:, 0]
        y = pts[:, 1]
        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    origin = pts[0]
    area = 0.0
    for idx in range(1, pts.shape[0] - 1):
        area += 0.5 * np.linalg.norm(np.cross(pts[idx] - origin, pts[idx + 1] - origin))
    return float(area)


def _tetra_volume(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] != 4 or pts.shape[1] < 3:
        return 0.0
    mat = np.empty((3, 3), dtype=np.float64)
    mat[0, :] = pts[1, :3] - pts[0, :3]
    mat[1, :] = pts[2, :3] - pts[0, :3]
    mat[2, :] = pts[3, :3] - pts[0, :3]
    return float(abs(np.linalg.det(mat)) / 6.0)


def _tetra_volume_from_indices(coords: np.ndarray, cell: np.ndarray) -> float:
    if cell.size != 4 or coords.shape[1] < 3:
        return 0.0
    p0 = coords[int(cell[0])]
    p1 = coords[int(cell[1])]
    p2 = coords[int(cell[2])]
    p3 = coords[int(cell[3])]
    ax = float(p1[0] - p0[0])
    ay = float(p1[1] - p0[1])
    az = float(p1[2] - p0[2])
    bx = float(p2[0] - p0[0])
    by = float(p2[1] - p0[1])
    bz = float(p2[2] - p0[2])
    cx = float(p3[0] - p0[0])
    cy = float(p3[1] - p0[1])
    cz = float(p3[2] - p0[2])
    det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    return abs(det) / 6.0


def _axis_aligned_hexa_volume_from_indices(
    coords: np.ndarray, cell: np.ndarray
) -> float:
    if cell.size != 8 or coords.shape[1] < 3:
        return float("nan")

    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for vertex_idx in cell:
        point = coords[int(vertex_idx)]
        for axis in range(3):
            value = float(point[axis])
            if not np.isfinite(value):
                return float("nan")
            mins[axis] = min(mins[axis], value)
            maxs[axis] = max(maxs[axis], value)

    extents = [maxs[axis] - mins[axis] for axis in range(3)]
    if any(extent < 0.0 for extent in extents):
        return float("nan")
    if any(extent == 0.0 for extent in extents):
        return 0.0

    scale = max(
        max(abs(value) for pair in zip(mins, maxs, strict=True) for value in pair), 1.0
    )
    tol = max(1.0e-12, scale * 1.0e-6)
    seen = 0
    for vertex_idx in cell:
        point = coords[int(vertex_idx)]
        code = 0
        for axis in range(3):
            value = float(point[axis])
            if abs(value - mins[axis]) <= tol:
                bit = 0
            elif abs(value - maxs[axis]) <= tol:
                bit = 1
            else:
                return float("nan")
            code |= bit << axis
        seen |= 1 << code
    if seen != 0xFF:
        return float("nan")
    return extents[0] * extents[1] * extents[2]


def _polyhedron_volume(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == 4:
        return _tetra_volume(pts)
    if pts.shape[0] < 4 or pts.shape[1] < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(pts[:, :3]).volume)
    except Exception:
        extents = pts[:, :3].max(axis=0) - pts[:, :3].min(axis=0)
        return float(abs(np.prod(extents)))


__all__ = [
    "DEFAULT_VISUAL_POINT_CLOUD_MAX_POINTS",
    "MESH_DERIVED_SCHEMA",
    "MeshDerivedArrays",
    "build_mesh_derived_arrays",
    "load_mesh_derived_artifact",
    "load_or_build_mesh_derived_artifact",
    "mesh_derived_cache_path",
    "mesh_derived_signature",
    "mesh_derived_signature_payload",
    "write_mesh_derived_artifact",
]
