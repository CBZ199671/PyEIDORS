"""Mesh-related utility functions."""

from __future__ import annotations

import numpy as np


def cell_to_node(mesh, cell_values: np.ndarray) -> np.ndarray:
    """Convert cell values to node values by averaging adjacent cells.

    Args:
        mesh: ``EITMesh`` or raw DOLFINx mesh-like object exposing ``cells()`` and ``num_vertices()``.
        cell_values: Values for each cell, shape (num_cells,).

    Returns:
        Values for each node, shape (num_vertices,).
    """
    node_vals = np.zeros(mesh.num_vertices())
    counts = np.zeros(mesh.num_vertices())

    for ci, cell in enumerate(mesh.cells()):
        for v in cell:
            node_vals[v] += cell_values[ci]
            counts[v] += 1

    counts[counts == 0] = 1  # Avoid division by zero
    node_vals /= counts
    return node_vals


def get_mesh_info(mesh) -> dict:
    """Get basic mesh information.

    Args:
        mesh: ``EITMesh`` or raw DOLFINx mesh-like object.

    Returns:
        Dictionary containing mesh information.
    """
    raw_mesh = mesh.mesh if hasattr(mesh, "mesh") else mesh
    topology = raw_mesh.topology
    topology.create_entities(1)
    edge_map = topology.index_map(1)
    num_edges = int(edge_map.size_local if edge_map is not None else 0)
    return {
        "num_cells": mesh.num_cells(),
        "num_vertices": mesh.num_vertices(),
        "num_edges": num_edges,
        "geometric_dimension": int(raw_mesh.geometry.dim),
        "topology_dimension": int(raw_mesh.topology.dim),
    }
