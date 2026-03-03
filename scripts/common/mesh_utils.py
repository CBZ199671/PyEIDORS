"""Mesh-related utility functions."""

from __future__ import annotations

import numpy as np


def cell_to_node(mesh, cell_values: np.ndarray) -> np.ndarray:
    """Convert cell values to node values by averaging adjacent cells.

    Args:
        mesh: FEniCS mesh object.
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
        mesh: FEniCS mesh object.

    Returns:
        Dictionary containing mesh information.
    """
    return {
        "num_cells": mesh.num_cells(),
        "num_vertices": mesh.num_vertices(),
        "num_edges": mesh.num_edges(),
        "geometric_dimension": mesh.geometric_dimension(),
        "topology_dimension": mesh.topology().dim(),
    }
