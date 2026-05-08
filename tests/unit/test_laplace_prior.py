"""Tests for coarse inverse mesh Laplace priors."""

from __future__ import annotations

import numpy as np

from pyeidors.inverse.dual_mesh import CellMesh, VoxelGrid
from pyeidors.inverse.prior import as_rtr_prior
from pyeidors.inverse.prior.laplace import (
    graph_curvature_prior,
    graph_laplacian,
    graph_ltl,
    graph_ltl_prior,
)


def test_graph_laplacian_for_voxel_grid_uses_face_neighbours() -> None:
    grid = VoxelGrid(
        origin=np.array([0.0, 0.0]),
        spacing=np.array([1.0, 1.0]),
        shape=(2, 2),
    )

    laplace = graph_laplacian(grid)

    expected = np.array(
        [
            [4.0, -2.0, -2.0, 0.0],
            [-2.0, 4.0, 0.0, -2.0],
            [-2.0, 0.0, 4.0, -2.0],
            [0.0, -2.0, -2.0, 4.0],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(laplace.toarray(), expected)


def test_graph_laplacian_for_tetra_cells_uses_shared_facets() -> None:
    mesh = CellMesh(
        coordinates=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        cells=np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=int),
        name="coarse-tetra",
    )

    laplace = graph_laplacian(mesh)

    np.testing.assert_allclose(
        laplace.toarray(),
        np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=float),
    )


def test_graph_laplacian_for_hexa_cells_uses_shared_faces() -> None:
    mesh = CellMesh(
        coordinates=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [2.0, 0.0, 1.0],
                [2.0, 1.0, 1.0],
            ],
            dtype=float,
        ),
        cells=np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 8, 9, 2, 5, 10, 11, 6],
            ],
            dtype=int,
        ),
        name="coarse-hexa",
    )

    laplace = graph_laplacian(mesh)

    np.testing.assert_allclose(
        laplace.toarray(),
        np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=float),
    )


def test_graph_laplacian_volume_weight_keeps_row_sum_zero() -> None:
    grid = VoxelGrid(
        origin=np.array([0.0]),
        spacing=np.array([2.0]),
        shape=(3,),
    )

    laplace = graph_laplacian(grid, weight="volume")

    np.testing.assert_allclose(np.asarray(laplace.sum(axis=1)).reshape(-1), 0.0)
    assert laplace.nnz > 0


def test_graph_ltl_squares_laplacian_and_keeps_named_prior_identity() -> None:
    grid = VoxelGrid(
        origin=np.array([0.0, 0.0]),
        spacing=np.array([1.0, 1.0]),
        shape=(2, 2),
    )

    laplace = graph_laplacian(grid)
    ltl = graph_ltl(grid)
    laplace_prior = as_rtr_prior(
        laplace,
        name="laplace",
        metadata={"signature_hint": "laplace"},
    )
    graph_prior = graph_ltl_prior(grid)
    curvature_prior = graph_curvature_prior(grid)

    np.testing.assert_allclose(ltl.toarray(), (laplace @ laplace).toarray())
    assert not np.allclose(ltl.toarray(), laplace.toarray())
    assert graph_prior.signature_hash != laplace_prior.signature_hash
    assert curvature_prior.signature_hash == graph_prior.signature_hash
    assert graph_prior.metadata["laplace_operator_shape"] == (4, 4)
    assert graph_prior.metadata["regularization_source"] == "graph_laplacian_squared"
    assert curvature_prior.metadata["alias"] == "curvature"
    assert curvature_prior.metadata["name"] == "curvature"
