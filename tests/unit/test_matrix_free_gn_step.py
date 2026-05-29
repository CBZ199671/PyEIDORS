"""Tests for matrix-free IRGNM / LM step wrapper."""

from __future__ import annotations

import inspect

import numpy as np
from scipy import sparse

from pyeidors.inverse.solvers import matrix_free_gn as matrix_free_gn_module
from pyeidors.inverse import (
    CellMesh,
    DualMesh,
    DualMeshJacobianOperator,
    VoxelGrid,
    solve_matrix_free_gn_step,
)


def test_v482_matrix_free_gn_finite_guards_use_bounded_scanner() -> None:
    checked_functions = (
        matrix_free_gn_module._as_vector,
        matrix_free_gn_module._as_regularization,
        matrix_free_gn_module._sqrt_measurement_weights,
    )
    old_payload_scans = (
        "np.isfinite(vector).all()",
        "np.isfinite(matrix.data).all()",
        "np.isfinite(diag).all()",
        "np.any(diag < 0.0)",
    )

    for func in checked_functions:
        source = inspect.getsource(func)
        assert "all_finite_values(" in source
        for old_payload_scan in old_payload_scans:
            assert old_payload_scan not in source


def test_v507_matrix_free_weight_diagonal_detection_avoids_dense_diag_copy() -> None:
    source = inspect.getsource(matrix_free_gn_module._sqrt_measurement_weights)
    helper_source = inspect.getsource(
        matrix_free_gn_module._dense_matrix_is_effectively_diagonal
    )

    assert "np.diag(np.diag(weights))" not in source
    assert "np.diag(weights)" not in source
    assert "_dense_matrix_is_effectively_diagonal(weights" in source
    assert "weights.diagonal()" in source
    assert "np.abs(chunk, out=abs_chunk)" in helper_source

    weights = np.diag([4.0, 9.0, 16.0])
    sqrt_weights, kind = matrix_free_gn_module._sqrt_measurement_weights(
        weights,
        n_measurements=3,
    )

    np.testing.assert_allclose(sqrt_weights, np.array([2.0, 3.0, 4.0]))
    assert kind == "diagonal-matrix"
    assert matrix_free_gn_module._dense_matrix_is_effectively_diagonal(weights)

    off_diagonal = weights.copy()
    off_diagonal[0, 1] = 1e-2
    assert not matrix_free_gn_module._dense_matrix_is_effectively_diagonal(off_diagonal)


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


def _dual_mesh_case() -> tuple[DualMeshJacobianOperator, np.ndarray]:
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
            [0.25, 0.5, -0.5, 1.25],
        ],
        dtype=float,
    )
    dense = fine_j @ dual.coarse2fine.toarray()
    return DualMeshJacobianOperator(dual, fine_j), dense


def test_matrix_free_irgnm_step_matches_dense_weighted_reference() -> None:
    operator, dense = _dual_mesh_case()
    residual = np.array([0.08, -0.03, 0.05, -0.02], dtype=float)
    current = np.array([0.2, -0.1], dtype=float)
    prior = np.array([0.05, 0.02], dtype=float)
    weights = np.array([2.0, 0.5, 1.5, 0.75], dtype=float)
    reg_diag = np.array([1.0, 1.8], dtype=float)
    regularization = sparse.diags(reg_diag, 0, format="csr")
    alpha = 0.09

    result = solve_matrix_free_gn_step(
        operator,
        residual,
        current=current,
        prior=prior,
        regularization=regularization,
        alpha=alpha,
        method="irgnm",
        measurement_weights=weights,
        matrix_free_ksp_backend="scipy",
    )

    w = np.diag(weights)
    hessian = dense.T @ w @ dense + alpha * regularization.toarray()
    rhs = -(
        dense.T @ w @ residual + alpha * (regularization.toarray() @ (current - prior))
    )
    expected = np.linalg.solve(hessian, rhs)
    np.testing.assert_allclose(result.delta, expected, rtol=1e-5, atol=1e-7)
    assert result.metadata["gn_family_method"] == "irgnm"
    assert result.metadata["jacobian_representation"] == "dual_mesh_jacobian_operator"
    assert result.metadata["dense_jacobian_materialized"] is False
    assert result.metadata["matrix_free_ksp_backend_requested"] == "scipy"
    assert result.metadata["measurement_weight_kind"] == "diagonal"


def test_matrix_free_lm_step_damps_hessian_without_damping_prior_gradient() -> None:
    operator, dense = _dual_mesh_case()
    residual = np.array([0.06, -0.01, 0.02, -0.04], dtype=float)
    current = np.array([0.12, -0.07], dtype=float)
    weights = np.array([1.25, 0.8, 1.4, 0.6], dtype=float)
    reg_diag = np.array([1.0, 2.0], dtype=float)
    regularization = sparse.diags(reg_diag, 0, format="csr")
    alpha = 0.05
    damping = 0.3

    result = solve_matrix_free_gn_step(
        operator,
        residual,
        current=current,
        regularization=regularization,
        alpha=alpha,
        damping=damping,
        method="lm",
        measurement_weights=np.diag(weights),
        matrix_free_ksp_backend="scipy",
    )

    w = np.diag(weights)
    hessian = (
        dense.T @ w @ dense + alpha * regularization.toarray() + damping * np.eye(2)
    )
    rhs = -(dense.T @ w @ residual + alpha * (regularization.toarray() @ current))
    expected = np.linalg.solve(hessian, rhs)
    np.testing.assert_allclose(result.delta, expected, rtol=1e-5, atol=1e-7)
    assert result.metadata["gn_family_method"] == "lm"
    assert result.metadata["damping"] == damping
    assert result.metadata["measurement_weight_kind"] == "diagonal-matrix"


def test_matrix_free_step_preserves_native_complex_dense_solve() -> None:
    jacobian = np.array(
        [
            [1.0 + 0.4j, 0.5 - 0.2j],
            [-0.25 + 0.1j, 1.5 + 0.3j],
            [0.75 - 0.5j, -0.4 + 0.8j],
        ],
        dtype=np.complex64,
    )
    true_delta = np.array([0.2 + 0.3j, -0.1 + 0.4j], dtype=np.complex64)
    residual = -(jacobian @ true_delta)

    result = solve_matrix_free_gn_step(
        jacobian,
        residual,
        regularization=np.eye(2, dtype=np.float64),
        alpha=0.0,
        method="irgnm",
        fast_linear_path="auto",
    )

    assert result.delta.dtype == np.complex64
    np.testing.assert_allclose(result.delta, true_delta, rtol=1e-5, atol=1e-6)
    assert result.metadata["path"] == "native-complex-dense"
    assert result.metadata["native_complex_linear_algebra"] is True
