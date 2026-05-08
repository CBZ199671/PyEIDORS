"""Graph priors for coarse inverse meshes."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from ._graph_core import (
    difference_from_edges,
    graph_edges_and_volumes,
    laplacian_from_edges,
    resolve_graph_weight,
)
from .rtr import RtRPrior, as_rtr_prior


def graph_laplacian(mesh: Any, *, weight: str = "unit") -> sparse.csr_matrix:
    """Build a cell-neighbour graph Laplacian for an inverse mesh.

    The operator is defined on coarse inverse cells. For simplex-like cell
    meshes, cells are adjacent when they share a facet
    (``vertices_per_cell - 1`` common vertices). For ``VoxelGrid`` inputs,
    face-neighbour adjacency is generated directly from the grid shape.
    EIDORS ``prior_laplace`` contributes ``[2, -2; -2, 2]`` per shared
    interior face, i.e. twice the plain graph Laplacian.
    """

    resolved_weight = resolve_graph_weight(weight)
    n_cells, edges, volumes = graph_edges_and_volumes(mesh)
    return (
        2.0
        * laplacian_from_edges(
            n_cells,
            edges,
            volumes=volumes,
            weight=resolved_weight,
        )
    ).tocsr()


def graph_difference_operator(mesh: Any, *, weight: str = "unit") -> sparse.csr_matrix:
    """Build oriented cell-neighbour difference operator for inverse meshes."""

    resolved_weight = resolve_graph_weight(weight)
    n_cells, edges, volumes = graph_edges_and_volumes(mesh)
    return difference_from_edges(
        n_cells, edges, volumes=volumes, weight=resolved_weight
    )


def graph_ltl(mesh: Any, *, weight: str = "unit") -> sparse.csr_matrix:
    """Build a squared-Laplacian curvature prior.

    ``graph_laplacian`` already returns the EIDORS Laplace RtR payload.  The
    curvature variant applies the Laplace operator twice, so it is distinct
    from the Laplace prior rather than a renamed graph incidence ``D.T @ D``.
    """

    laplace = graph_laplacian(mesh, weight=weight).tocsr()
    n_cells = int(laplace.shape[1])
    if laplace.shape[0] == 0 or laplace.nnz == 0:
        return sparse.csr_matrix((n_cells, n_cells), dtype=np.float64)
    return (laplace.T @ laplace).tocsr()


def graph_ltl_prior(mesh: Any, *, weight: str = "unit") -> RtRPrior:
    """Build a named squared-Laplacian ``graph_ltl`` RtR prior."""

    laplace = graph_laplacian(mesh, weight=weight).tocsr()
    n_cells = int(laplace.shape[1])
    matrix = (
        sparse.csr_matrix((n_cells, n_cells), dtype=np.float64)
        if laplace.shape[0] == 0 or laplace.nnz == 0
        else (laplace.T @ laplace).tocsr()
    )
    return as_rtr_prior(
        matrix,
        name="graph_ltl",
        metadata={
            "prior_family": "graph_ltl",
            "graph_weight": str(weight).strip().lower(),
            "laplace_operator_shape": tuple(int(v) for v in laplace.shape),
            "regularization_source": "graph_laplacian_squared",
            "signature_hint": "graph_ltl",
        },
    )


def graph_curvature_prior(mesh: Any, *, weight: str = "unit") -> RtRPrior:
    """Build a named ``curvature`` RtR prior with ``L.T @ L`` payload."""

    prior = graph_ltl_prior(mesh, weight=weight)
    metadata = {
        key: value
        for key, value in dict(prior.metadata).items()
        if key not in {"kind", "name", "nnz", "shape", "signature_hash"}
    }
    return as_rtr_prior(
        prior.as_RtR(dense=False),
        name="curvature",
        metadata={
            **metadata,
            "alias": "curvature",
            "signature_hint": "graph_ltl",
        },
    )


__all__ = [
    "graph_curvature_prior",
    "graph_difference_operator",
    "graph_laplacian",
    "graph_ltl",
    "graph_ltl_prior",
]
