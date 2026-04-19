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
    cells = np.asarray(cell_connectivity, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[0] == 0:
        empty = np.empty((0, 3), dtype=np.int32)
        return (empty, np.empty((0,), dtype=np.int32)) if return_sources else (empty, None)

    if cells.shape[1] == 3:
        triangles = cells.astype(np.int32, copy=False)
        if return_sources:
            return triangles, np.arange(len(cells), dtype=np.int32)
        return triangles, None

    if cells.shape[1] != 4:
        empty = np.empty((0, 3), dtype=np.int32)
        return (empty, np.empty((0,), dtype=np.int32)) if return_sources else (empty, None)

    # Tetra path: dict-of-sorted-keys to detect faces shared by exactly
    # one cell.  Python loop because mesh sizes are typically O(10k)
    # cells and the per-cell branching makes vectorisation awkward.
    faces: dict[tuple[int, int, int], tuple[tuple[int, int, int], int] | None] = {}
    for cell_idx, cell in enumerate(cells):
        for offsets in _TETRA_FACE_OFFSETS:
            face = (int(cell[offsets[0]]), int(cell[offsets[1]]), int(cell[offsets[2]]))
            key = tuple(sorted(face))
            faces[key] = None if key in faces else (face, cell_idx)
    kept = [payload for payload in faces.values() if payload is not None]
    if not kept:
        empty = np.empty((0, 3), dtype=np.int32)
        return (empty, np.empty((0,), dtype=np.int32)) if return_sources else (empty, None)
    triangles = np.asarray([face for face, _ in kept], dtype=np.int32)
    if return_sources:
        sources = np.asarray([idx for _, idx in kept], dtype=np.int32)
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

    Roughly 30–50× faster than the previous nested-Python-loop form
    on a typical 10k-cell mesh (np.add.at is the standard idiom for
    scatter-add accumulations).
    """
    cells_i = np.asarray(cells, dtype=np.int64)
    sigma = np.asarray(cell_values, dtype=np.float64).reshape(-1)
    if cells_i.ndim != 2 or sigma.size != cells_i.shape[0]:
        raise ValueError(
            f"cell_to_node_average: cells {cells_i.shape} vs values {sigma.shape}"
        )

    flat_idx = cells_i.ravel()
    repeats = np.repeat(sigma, cells_i.shape[1])
    node_sum = np.zeros(n_nodes, dtype=np.float64)
    node_count = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(node_sum, flat_idx, repeats)
    np.add.at(node_count, flat_idx, 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        node_values = np.where(node_count > 0, node_sum / node_count, np.nan)
    if np.any(np.isnan(node_values)):
        finite = node_values[np.isfinite(node_values)]
        mean = float(finite.mean()) if finite.size else 0.0
        node_values = np.where(np.isnan(node_values), mean, node_values)
    return node_values
