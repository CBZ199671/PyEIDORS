"""Tests for dual-mesh matrix-free Jacobian operators."""

from __future__ import annotations

import numpy as np

from pyeidors.inverse import CellMesh, DualMesh, DualMeshJacobianOperator, VoxelGrid


def _cell_mesh_from_centers(centers: np.ndarray, *, name: str) -> CellMesh:
    centers = np.asarray(centers, dtype=float)
    offsets = np.array(
        [
            [-1e-3, -1e-3, -1e-3],
            [1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 1e-3],
        ],
        dtype=float,
    )
    coordinates: list[np.ndarray] = []
    cells: list[list[int]] = []
    for center in centers:
        start = len(coordinates)
        coordinates.extend(center + offsets)
        cells.append(list(range(start, start + offsets.shape[0])))
    return CellMesh(np.asarray(coordinates), np.asarray(cells), name=name)


def test_dual_mesh_matrix_free_jv_jtr_matches_dense_reference() -> None:
    coarse = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        shape=(2, 1, 1),
    )
    fine = _cell_mesh_from_centers(
        np.array(
            [
                [0.25, 0.5, 0.5],
                [0.75, 0.5, 0.5],
                [1.25, 0.5, 0.5],
                [1.75, 0.5, 0.5],
            ],
            dtype=float,
        ),
        name="fine-cem-surrogate",
    )
    dual = DualMesh(fine, coarse)
    fine_j = np.array(
        [
            [1.0, 2.0, 0.0, -1.0],
            [0.5, -0.25, 3.0, 1.5],
            [-2.0, 1.0, 0.75, 0.25],
        ],
        dtype=float,
    )
    op = DualMeshJacobianOperator(dual, fine_j)
    spec_op = DualMeshJacobianOperator(fine_j, dual.coarse2fine)
    dense = fine_j @ dual.coarse2fine.toarray()

    v = np.array([1.25, -0.5], dtype=float)
    r = np.array([0.5, -2.0, 1.0], dtype=float)
    weights = np.array([2.0, 0.5, 1.5], dtype=float)
    regularization = np.diag([3.0, 4.0])

    np.testing.assert_allclose(op.Jv(v), dense @ v)
    np.testing.assert_allclose(spec_op.Jv(v), dense @ v)
    np.testing.assert_allclose(op.JTr(r), dense.T @ r)
    np.testing.assert_allclose(spec_op.JTr(r), dense.T @ r)
    np.testing.assert_allclose(op.to_dense(), dense)
    np.testing.assert_allclose(op.as_linear_operator().matvec(v), dense @ v)
    np.testing.assert_allclose(op.as_linear_operator().rmatvec(r), dense.T @ r)
    np.testing.assert_allclose(
        op.normal_matvec(
            v,
            measurement_weights=weights,
            alpha=0.1,
            regularization=regularization,
        ),
        dense.T @ (weights * (dense @ v)) + 0.1 * (regularization @ v),
    )


def test_dual_mesh_matrix_free_accepts_callable_actions() -> None:
    projection = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=float)
    fine_j = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]], dtype=float)
    op = DualMeshJacobianOperator(
        projection,
        fine_forward_action=lambda fine: fine_j @ fine,
        fine_adjoint_action=lambda residual: fine_j.T @ residual,
        n_measurements=2,
    )
    dense = fine_j @ projection

    np.testing.assert_allclose(op.Jv(np.array([1.0, 4.0])), dense @ [1.0, 4.0])
    np.testing.assert_allclose(op.JTr(np.array([0.25, -2.0])), dense.T @ [0.25, -2.0])
