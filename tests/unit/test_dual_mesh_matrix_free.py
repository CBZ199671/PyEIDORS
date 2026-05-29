"""Tests for dual-mesh matrix-free Jacobian operators."""

from __future__ import annotations

import inspect

import numpy as np

from pyeidors.inverse import CellMesh, DualMesh, DualMeshJacobianOperator, VoxelGrid
from pyeidors.inverse.matrix_free import dual_mesh as dual_mesh_operator_module


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


def test_v482_dual_mesh_matrix_free_finite_guards_use_bounded_scanner() -> None:
    checked_functions = (
        dual_mesh_operator_module._optional_matrix,
        dual_mesh_operator_module._as_vector,
    )
    old_payload_scans = (
        "np.isfinite(matrix).all()",
        "np.isfinite(vector).all()",
    )

    for func in checked_functions:
        source = inspect.getsource(func)
        assert "all_finite_values(" in source
        for old_payload_scan in old_payload_scans:
            assert old_payload_scan not in source


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


def test_v281_dual_mesh_to_dense_direct_fills_columns(monkeypatch) -> None:
    projection = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=float)
    fine_j = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]], dtype=float)
    op = DualMeshJacobianOperator(
        projection,
        fine_forward_action=lambda fine: fine_j @ fine,
        fine_adjoint_action=lambda residual: fine_j.T @ residual,
        n_measurements=2,
    )
    expected = fine_j @ projection

    def _fail_dense_helper(*_args, **_kwargs):
        raise AssertionError("dual-mesh dense materialization must direct-fill")

    monkeypatch.setattr(
        dual_mesh_operator_module.np, "column_stack", _fail_dense_helper
    )
    monkeypatch.setattr(dual_mesh_operator_module.np, "eye", _fail_dense_helper)

    np.testing.assert_allclose(op.to_dense(), expected)
    source = inspect.getsource(
        dual_mesh_operator_module.DualMeshJacobianOperator.to_dense
    )
    assert "np.column_stack" not in source
    assert "np.eye" not in source
