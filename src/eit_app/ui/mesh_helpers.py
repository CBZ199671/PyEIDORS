"""Shared numpy / mesh helpers used by the conductivity display widgets.

Each helper is small but was previously duplicated across at least two
of ``conductivity_3d_widget``, ``conductivity_image_widget``, and
``hardware/equipotential_plot_widget``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


_TETRA_FACE_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 1, 3),
    (0, 2, 3),
    (1, 2, 3),
)
_FINITE_SCAN_CHUNK_ITEMS = 1_048_576


def _all_finite(values: np.ndarray) -> bool:
    arr = np.asarray(values).reshape(-1)
    chunk_items = max(1, int(_FINITE_SCAN_CHUNK_ITEMS))
    work = np.empty(min(chunk_items, max(arr.size, 1)), dtype=bool)
    for start in range(0, arr.size, chunk_items):
        chunk = arr[start : start + chunk_items]
        chunk_mask = work[: chunk.size]
        np.isfinite(chunk, out=chunk_mask)
        if not bool(chunk_mask.all()):
            return False
    return True


def _finite_sum_count(values: np.ndarray) -> tuple[float, int]:
    arr = np.asarray(values).reshape(-1)
    chunk_items = max(1, int(_FINITE_SCAN_CHUNK_ITEMS))
    work = np.empty(min(chunk_items, max(arr.size, 1)), dtype=bool)
    total = 0.0
    count = 0
    for start in range(0, arr.size, chunk_items):
        chunk = arr[start : start + chunk_items]
        chunk_mask = work[: chunk.size]
        np.isfinite(chunk, out=chunk_mask)
        if not bool(chunk_mask.any()):
            continue
        total += float(np.sum(chunk, where=chunk_mask, initial=0.0))
        count += int(np.count_nonzero(chunk_mask))
    return total, count


def _integer_cells(cell_connectivity: np.ndarray) -> np.ndarray:
    cells = np.asarray(cell_connectivity)
    if not np.issubdtype(cells.dtype, np.integer):
        cells = np.asarray(cells, dtype=np.intp)
    return cells


def extract_boundary_triangles(
    cell_connectivity: np.ndarray,
    *,
    return_sources: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Pull boundary triangles out of a 2D triangle or 3D tetra mesh.

    For a 3-vertex (triangle) mesh every cell IS already a boundary
    face — passes through unchanged.  For a 4-vertex (tetra) mesh the
    boundary is the set of triangular faces that appear in exactly one
    cell, computed via the standard sorted-tuple key trick.

    Args:
        cell_connectivity: ``(M, 3)`` or ``(M, 4)`` int array.
        return_sources: When True the second return value is an
            int32 array mapping each output triangle back to its
            source cell index — useful for colouring boundary
            triangles by the source cell's scalar.  When False the
            second return value is ``None`` and we don't bother
            tracking provenance (the cheaper path).

    Returns:
        ``(triangles, sources_or_none)`` where ``triangles`` is an
        ``(F, 3) int32`` array of node indices.  ``sources`` is
        ``(F,) int32`` (or None) with the source-cell index for each
        triangle.
    """
    cells = _integer_cells(cell_connectivity)
    if cells.ndim != 2 or cells.shape[0] == 0:
        empty = np.empty((0, 3), dtype=np.int32)
        return (
            (empty, np.empty((0,), dtype=np.int32)) if return_sources else (empty, None)
        )

    if cells.shape[1] == 3:
        triangles = cells.astype(np.int32, copy=False)
        if return_sources:
            return triangles, np.arange(len(cells), dtype=np.int32)
        return triangles, None

    if cells.shape[1] != 4:
        empty = np.empty((0, 3), dtype=np.int32)
        return (
            (empty, np.empty((0,), dtype=np.int32)) if return_sources else (empty, None)
        )

    # Tetra path: dict-of-sorted-keys to detect faces shared by exactly
    # one cell.  Python loop because mesh sizes are typically O(10k)
    # cells and the per-cell branching makes vectorisation awkward.
    faces: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {}
    for cell_idx, cell in enumerate(cells):
        for offsets in _TETRA_FACE_OFFSETS:
            face = (int(cell[offsets[0]]), int(cell[offsets[1]]), int(cell[offsets[2]]))
            key = tuple(sorted(face))
            faces[key] = None if key in faces else (face, cell_idx)
    kept_count = sum(1 for payload in faces.values() if payload is not None)
    if kept_count <= 0:
        empty = np.empty((0, 3), dtype=np.int32)
        return (
            (empty, np.empty((0,), dtype=np.int32)) if return_sources else (empty, None)
        )

    triangles = np.empty((kept_count, 3), dtype=np.int32)
    if return_sources:
        sources = np.empty(kept_count, dtype=np.int32)
    else:
        sources = None
    kept_idx = 0
    for payload in faces.values():
        if payload is None:
            continue
        face, cell_idx = payload
        triangles[kept_idx] = face
        if sources is not None:
            sources[kept_idx] = int(cell_idx)
        kept_idx += 1

    if return_sources:
        assert sources is not None
        return triangles, sources
    return triangles, None


def cell_to_node_average(
    cell_values: np.ndarray, cells: np.ndarray, n_nodes: int
) -> np.ndarray:
    """Vectorised area-uniform per-cell → per-node averaging.

    For each node, returns the unweighted mean of the values from all
    cells that touch it.  Orphan nodes (touched by no cell, which can
    happen at the mesh boundary after cell extraction) get the global
    mean so downstream triangulators don't trip on NaN.

    Uses one scatter-add pass per local cell vertex.  This avoids
    materialising the larger ``np.repeat(values, vertices_per_cell)``
    temporary on large 3D display meshes.
    """
    cells_i = np.asarray(cells)
    if not np.issubdtype(cells_i.dtype, np.integer):
        cells_i = np.asarray(cells_i, dtype=np.intp)
    values = np.asarray(cell_values)
    if np.iscomplexobj(values):
        values = np.real(values)
    if np.issubdtype(values.dtype, np.floating):
        dtype = np.result_type(values.dtype, np.float32)
        sigma = np.asarray(values, dtype=dtype).reshape(-1)
    else:
        sigma = np.asarray(values, dtype=np.float32).reshape(-1)
        dtype = sigma.dtype
    if cells_i.ndim != 2 or sigma.size != cells_i.shape[0]:
        raise ValueError(
            f"cell_to_node_average: cells {cells_i.shape} vs values {sigma.shape}"
        )

    node_sum = np.zeros(n_nodes, dtype=dtype)
    node_count = np.zeros(n_nodes, dtype=dtype)
    for local_idx in range(cells_i.shape[1]):
        np.add.at(node_sum, cells_i[:, local_idx], sigma)
        np.add.at(node_count, cells_i[:, local_idx], 1.0)

    touched = node_count > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        node_values = np.divide(node_sum, node_count, out=node_sum, where=touched)
    if not bool(touched.all()):
        if _all_finite(sigma):
            touched_count = int(np.count_nonzero(touched))
            mean = (
                float(np.sum(node_values, where=touched, initial=0.0))
                / float(touched_count)
                if touched_count
                else 0.0
            )
            np.logical_not(touched, out=touched)
            np.copyto(node_values, mean, where=touched)
            return node_values
        np.logical_not(touched, out=touched)
        np.copyto(node_values, np.nan, where=touched)
    np.isnan(node_values, out=touched)
    if bool(touched.any()):
        finite_sum, finite_count = _finite_sum_count(node_values)
        mean = finite_sum / float(finite_count) if finite_count else 0.0
        np.copyto(node_values, mean, where=touched)
    return node_values
