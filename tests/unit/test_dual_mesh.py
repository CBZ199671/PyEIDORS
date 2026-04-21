"""Tests for dual-mesh coarse-to-fine projection helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pyeidors.inverse.dual_mesh import CellMesh, DualMesh, VoxelGrid, coarse2fine


def _cell_mesh_from_centers(centers: np.ndarray, *, name: str = "fine") -> CellMesh:
    centers = np.asarray(centers, dtype=float)
    coords: list[np.ndarray] = []
    cells: list[list[int]] = []
    eps = 1e-3
    if centers.shape[1] == 2:
        offsets = np.array([[-eps, -eps], [eps, 0.0], [0.0, eps]], dtype=float)
    elif centers.shape[1] == 3:
        offsets = np.array(
            [
                [-eps, -eps, -eps],
                [eps, 0.0, 0.0],
                [0.0, eps, 0.0],
                [0.0, 0.0, eps],
            ],
            dtype=float,
        )
    else:
        raise ValueError("test helper only supports 2D or 3D centers")

    for center in centers:
        start = len(coords)
        coords.extend(center + offsets)
        cells.append(list(range(start, start + offsets.shape[0])))
    return CellMesh(np.asarray(coords, dtype=float), np.asarray(cells), name=name)


def test_coarse2fine_maps_fine_cells_to_voxel_cells() -> None:
    fine = _cell_mesh_from_centers(
        np.array(
            [
                [0.25, 0.25],
                [1.25, 0.25],
                [0.25, 1.25],
                [1.25, 1.25],
            ],
            dtype=float,
        )
    )
    coarse = VoxelGrid(
        origin=np.array([0.0, 0.0]), spacing=np.array([1.0, 1.0]), shape=(2, 2)
    )

    projection = coarse2fine(fine, coarse)

    assert sparse.isspmatrix_csr(projection)
    assert projection.shape == (4, 4)
    assert projection.nnz == 4
    np.testing.assert_allclose(
        projection.toarray(),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )


def test_coarse2fine_uses_tetra_containment_before_nearest() -> None:
    coarse = CellMesh(
        coordinates=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        cells=np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=int),
        name="coarse-tetra",
    )
    fine = _cell_mesh_from_centers(
        np.array([[0.1, 0.1, 0.1], [1.2, 0.1, 0.1]], dtype=float),
        name="fine-tetra",
    )

    projection = coarse2fine(fine, coarse)

    np.testing.assert_allclose(
        projection.toarray(),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )


def test_coarse2fine_falls_back_to_nearest_or_raises_for_outside_points() -> None:
    fine = _cell_mesh_from_centers(np.array([[3.0, 0.5]], dtype=float))
    coarse = VoxelGrid(
        origin=np.array([0.0, 0.0]), spacing=np.array([1.0, 1.0]), shape=(2, 1)
    )

    projection = coarse2fine(fine, coarse)
    np.testing.assert_allclose(
        projection.toarray(), np.array([[0.0, 1.0]], dtype=float)
    )

    with pytest.raises(ValueError, match="outside the coarse voxel grid"):
        coarse2fine(fine, coarse, outside="raise")


def test_dual_mesh_projects_and_restricts_cell_values() -> None:
    fine = _cell_mesh_from_centers(
        np.array([[0.25, 0.25], [1.25, 0.25], [1.25, 0.75]], dtype=float)
    )
    coarse = VoxelGrid(
        origin=np.array([0.0, 0.0]), spacing=np.array([1.0, 1.0]), shape=(2, 1)
    )
    dual = DualMesh(fine_mesh=fine, coarse_mesh=coarse)

    assert dual.summary() == {
        "n_fine_cells": 3,
        "n_coarse_cells": 2,
        "projection_nnz": 3,
        "method": "piecewise_constant",
        "outside": "nearest",
    }
    np.testing.assert_allclose(
        dual.project_to_fine(np.array([10.0, 20.0])), [10.0, 20.0, 20.0]
    )
    np.testing.assert_allclose(
        dual.restrict_to_coarse(np.array([1.0, 3.0, 5.0])), [1.0, 4.0]
    )


def test_dual_mesh_validates_projection_shape_and_value_lengths() -> None:
    fine = _cell_mesh_from_centers(np.array([[0.25, 0.25], [1.25, 0.25]], dtype=float))
    coarse = VoxelGrid(
        origin=np.array([0.0, 0.0]), spacing=np.array([1.0, 1.0]), shape=(2, 1)
    )

    with pytest.raises(ValueError, match="projection shape mismatch"):
        DualMesh(fine, coarse, projection=sparse.eye(1, 2, format="csr"))

    dual = DualMesh(fine, coarse)
    with pytest.raises(ValueError, match="Expected 2 coarse values"):
        dual.project_to_fine(np.ones(1))
    with pytest.raises(ValueError, match="Expected 2 fine values"):
        dual.restrict_to_coarse(np.ones(1))
