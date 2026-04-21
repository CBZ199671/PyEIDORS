"""Tests for coarse inverse mesh Laplace priors."""

from __future__ import annotations

import numpy as np

from pyeidors.inverse.dual_mesh import CellMesh, VoxelGrid
from pyeidors.inverse.prior.laplace import graph_laplacian


def test_graph_laplacian_for_voxel_grid_uses_face_neighbours() -> None:
    grid = VoxelGrid(
        origin=np.array([0.0, 0.0]),
        spacing=np.array([1.0, 1.0]),
        shape=(2, 2),
    )

    laplace = graph_laplacian(grid)

    expected = np.array(
        [
            [2.0, -1.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0, -1.0],
            [-1.0, 0.0, 2.0, -1.0],
            [0.0, -1.0, -1.0, 2.0],
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
        np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float),
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
