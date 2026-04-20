"""Fast-mode linear solver unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator

import pyeidors.inverse.solvers.gauss_newton_runtime as gn_runtime
from pyeidors.inverse.jacobian.linearized import JacobianLinearization
from pyeidors.inverse.solvers.gauss_newton_runtime import _solve_linear_system_fast


def _dummy_reconstructor(
    linear_solver: str,
    *,
    preconditioner: str = "diag",
    fast_linear_path: str = "auto",
):
    return SimpleNamespace(
        R_matrix=diags([1.0, 2.0, 3.0], 0, format="csr"),
        R_diag=np.array([1.0, 2.0, 3.0], dtype=float),
        use_prior_term=True,
        performance_mode="aggressive",
        linear_solver=linear_solver,
        preconditioner=preconditioner,
        fast_linear_path=fast_linear_path,
        cholmod_max_n=12000,
        cholmod_max_memory_gib=4.0,
    )


def _expected_solution(J: np.ndarray, residual: np.ndarray, de: np.ndarray, lam: float) -> np.ndarray:
    R = np.diag([1.0, 2.0, 3.0])
    A = J.T @ J + lam * R
    rhs = -(J.T @ residual + lam * (R @ de))
    return np.linalg.solve(A, rhs)


def _make_linearization_from_dense(J: np.ndarray) -> JacobianLinearization:
    # Use a single spatial component so the synthetic gradients encode J exactly.
    n_meas, n_param = J.shape
    grad_u = np.ones((n_param, 1), dtype=float)
    adjoint_gradients = tuple(np.asarray(row, dtype=float).reshape(n_param, 1) for row in J)
    return JacobianLinearization(
        grad_u_all=(grad_u,),
        adjoint_gradients=adjoint_gradients,
        cell_areas=np.ones(n_param, dtype=float),
        n_meas_per_stim=(n_meas,),
        sign=1.0,
    )


def test_fast_solver_auto_woodbury_matches_dense_reference():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15

    delta, _, _ = _solve_linear_system_fast(
        _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="auto"),
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )
    expected = _expected_solution(J, residual, de, lam)
    assert np.allclose(delta, expected, rtol=1e-6, atol=1e-8)


def test_fast_solver_explicit_pcg_matches_dense_reference():
    J = np.array(
        [
            [0.9, 0.1, -0.3],
            [-0.5, 0.7, 0.4],
            [0.3, -0.2, 0.6],
            [0.2, 0.8, -0.1],
            [0.1, -0.4, 0.7],
        ],
        dtype=float,
    )
    residual = np.array([0.06, -0.03, 0.01, 0.05, -0.02], dtype=float)
    de = np.array([0.05, -0.08, 0.12], dtype=float)
    lam = 0.05

    delta, _, _ = _solve_linear_system_fast(
        _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="pcg"),
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=1,
    )
    expected = _expected_solution(J, residual, de, lam)
    assert np.allclose(delta, expected, rtol=1e-5, atol=1e-7)


def test_fast_solver_linear_operator_pcg_matches_dense_reference():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )

    recon = _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="pcg")
    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=2,
    )

    np.testing.assert_allclose(delta, _expected_solution(J, residual, de, lam), rtol=1e-5, atol=1e-7)
    meta = getattr(recon, "_last_fast_linear_meta", {})
    assert meta["jacobian_representation"] == "linear_operator"
    assert meta["dense_jacobian_materialized"] is False
    assert isinstance(meta["linear_iterations"], int)


def test_fast_solver_auto_hessian_diag_from_jacobian_linearization():
    """Operator path with no explicit diag attrs derives NOSER diag for free."""
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    linearization = _make_linearization_from_dense(J)

    recon = SimpleNamespace(
        R_matrix=diags([1.0, 2.0, 3.0], 0, format="csr"),
        use_prior_term=True,
        performance_mode="aggressive",
        linear_solver="auto",
        preconditioner="diag",
        fast_linear_path="pcg",
        cholmod_max_n=12000,
        cholmod_max_memory_gib=4.0,
    )

    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=linearization,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )

    expected = _expected_solution(J, residual, de, lam)
    np.testing.assert_allclose(delta, expected, rtol=1e-5, atol=1e-7)

    meta = recon._last_fast_linear_meta
    assert meta["jacobian_representation"] == "jacobian_linearization"
    assert meta["dense_jacobian_materialized"] is False
    assert meta["matrix_free_pc_source"] == "auto_linearization_diag"
    # diag(J^T J) is strictly positive for this synthetic problem.
    assert meta["matrix_free_pc_min"] > 0.0


def test_auto_hessian_diag_respects_measurement_weights():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    linearization = _make_linearization_from_dense(J)
    weights = np.array([1.5, 0.5, 1.0, 0.8], dtype=float)

    expected = (weights[:, None] * J * J).sum(axis=0)
    actual = linearization.hessian_diag(measurement_weights=weights)
    np.testing.assert_allclose(actual, expected)


def test_explicit_noser_diag_still_wins_over_auto_linearization_diag():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    linearization = _make_linearization_from_dense(J)

    recon = SimpleNamespace(
        R_matrix=diags([1.0, 2.0, 3.0], 0, format="csr"),
        R_diag=np.array([1.0, 2.0, 3.0], dtype=float),
        use_prior_term=True,
        performance_mode="aggressive",
        linear_solver="auto",
        preconditioner="diag",
        fast_linear_path="pcg",
        cholmod_max_n=12000,
        cholmod_max_memory_gib=4.0,
    )

    _solve_linear_system_fast(
        recon,
        J_weighted_np=linearization,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )

    meta = recon._last_fast_linear_meta
    assert meta["matrix_free_pc_source"] != "auto_linearization_diag"


def test_matrix_free_ksp_backend_defaults_to_scipy_when_unset():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )

    recon = _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="pcg")
    _solve_linear_system_fast(
        recon,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )
    meta = recon._last_fast_linear_meta
    assert meta["matrix_free_ksp_backend_requested"] == "scipy"
    assert meta["matrix_free_ksp_backend_effective"] == "scipy"


def test_matrix_free_ksp_backend_petsc_matches_scipy_reference():
    if gn_runtime._PETSc is None:
        pytest.skip("petsc4py unavailable in this environment")
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )

    recon = _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="pcg")
    recon.matrix_free_ksp_backend = "petsc"

    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )

    expected = _expected_solution(J, residual, de, lam)
    np.testing.assert_allclose(delta, expected, rtol=1e-5, atol=1e-7)

    meta = recon._last_fast_linear_meta
    assert meta["matrix_free_ksp_backend_requested"] == "petsc"
    assert meta["matrix_free_ksp_backend_effective"] == "petsc"
    assert isinstance(meta["linear_iterations"], int)
    assert meta["linear_iterations"] > 0


def test_matrix_free_ksp_backend_petsc_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setattr(gn_runtime, "_PETSc", None)

    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )

    recon = _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="pcg")
    recon.matrix_free_ksp_backend = "petsc"

    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )

    expected = _expected_solution(J, residual, de, lam)
    np.testing.assert_allclose(delta, expected, rtol=1e-5, atol=1e-7)

    meta = recon._last_fast_linear_meta
    assert meta["matrix_free_ksp_backend_requested"] == "petsc"
    assert meta["matrix_free_ksp_backend_effective"] == "scipy"
    assert meta["matrix_free_ksp_backend_fallback_reason"] == "petsc_backend_unavailable"


def test_solve_matrix_free_hessian_via_petsc_direct_call():
    if gn_runtime._PETSc is None:
        pytest.skip("petsc4py unavailable in this environment")
    n = 4
    rng = np.random.default_rng(1)
    M = rng.standard_normal((n, n))
    A = M @ M.T + 0.5 * np.eye(n)  # SPD
    b = rng.standard_normal(n)

    h_op = LinearOperator(
        (n, n),
        matvec=lambda x: A @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )

    delta, iterations, converged, reason = gn_runtime._solve_matrix_free_hessian_via_petsc(
        h_op,
        b,
        None,
        rtol=1e-10,
        maxiter=200,
    )

    np.testing.assert_allclose(delta, np.linalg.solve(A, b), rtol=1e-6, atol=1e-8)
    assert converged
    assert reason is None
    assert iterations > 0


def test_matrix_free_noser_preconditioner_contract_clamps_positive_diag():
    recon = SimpleNamespace(
        R_diag=np.array([0.0, np.nan, 4.0], dtype=float),
        regularization_type="noser",
        matrix_free_pc_floor=1e-6,
    )

    diag, meta = gn_runtime._operator_diag_preconditioner(
        recon,
        3,
        0.5,
        preferred="noser",
    )

    np.testing.assert_allclose(diag, np.array([1e-6, 1e-6, 2.0], dtype=float))
    assert np.isfinite(diag).all()
    assert np.all(diag >= 1e-6)
    assert meta["matrix_free_pc_source"] == "noser"
    assert meta["matrix_free_pc_mode"] == "noser"
    assert "noser_diag_clamped" in str(meta["matrix_free_pc_reason"])


def test_fast_solver_matrix_free_noser_and_prior_pc_metadata():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )

    recon_noser = _dummy_reconstructor("auto", preconditioner="noser", fast_linear_path="pcg")
    recon_noser.regularization_type = "noser"
    delta_noser, _, _ = _solve_linear_system_fast(
        recon_noser,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=4,
    )
    np.testing.assert_allclose(delta_noser, _expected_solution(J, residual, de, lam), rtol=1e-5, atol=1e-7)
    meta_noser = getattr(recon_noser, "_last_fast_linear_meta", {})
    assert meta_noser["resolved_preconditioner"] == "noser"
    assert meta_noser["matrix_free_pc_source"] == "noser"
    assert meta_noser["dense_jacobian_materialized"] is False

    recon_prior = _dummy_reconstructor("auto", preconditioner="prior", fast_linear_path="pcg")
    delta_prior, _, _ = _solve_linear_system_fast(
        recon_prior,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=5,
    )
    np.testing.assert_allclose(delta_prior, _expected_solution(J, residual, de, lam), rtol=1e-5, atol=1e-7)
    meta_prior = getattr(recon_prior, "_last_fast_linear_meta", {})
    assert meta_prior["resolved_preconditioner"] == "prior"
    assert meta_prior["matrix_free_pc_source"] == "prior"
    assert meta_prior["matrix_free_pmat_available"] is False


def test_fast_solver_matrix_free_petsc_gamg_requires_pmat_fallback(monkeypatch: pytest.MonkeyPatch):
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )
    monkeypatch.setattr(
        gn_runtime,
        "detect_performance_capabilities",
        lambda: {
            "pyamg": False,
            "cholmod": False,
            "petsc_mat_solve": False,
            "petsc_gamg": True,
        },
    )

    recon = _dummy_reconstructor("auto", preconditioner="petsc-gamg", fast_linear_path="pcg")
    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=6,
    )

    np.testing.assert_allclose(delta, _expected_solution(J, residual, de, lam), rtol=1e-5, atol=1e-7)
    meta = getattr(recon, "_last_fast_linear_meta", {})
    assert meta["resolved_preconditioner"] == "diag"
    assert meta["matrix_free_pmat_available"] is False
    assert "petsc_gamg_not_supported_in_matrix_free" in str(meta["fallback_reason"])


def test_fast_solver_matrix_free_sparse_pmat_smoke_matches_dense_reference():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )

    recon = _dummy_reconstructor("auto", preconditioner="pmat", fast_linear_path="pcg")
    recon.matrix_free_pmat = diags([1.2, 1.8, 2.6], 0, format="csr")
    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=7,
    )

    np.testing.assert_allclose(delta, _expected_solution(J, residual, de, lam), rtol=1e-5, atol=1e-7)
    meta = getattr(recon, "_last_fast_linear_meta", {})
    assert meta["resolved_preconditioner"] == "pmat"
    assert meta["path"] == "pcg-pmat-precond"
    assert meta["matrix_free_pmat_available"] is True
    assert meta["matrix_free_pc_source"] == "pmat"
    assert meta["matrix_free_pmat_kind"] == "sparse-diagonal"


def test_fast_solver_matrix_free_coarse_pmat_and_custom_pc_smokes():
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )
    expected = _expected_solution(J, residual, de, lam)

    recon_coarse = _dummy_reconstructor("auto", preconditioner="coarse", fast_linear_path="pcg")
    recon_coarse.matrix_free_coarse_pmat = np.diag([1.2, 1.8, 2.6])
    delta_coarse, _, _ = _solve_linear_system_fast(
        recon_coarse,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=8,
    )
    np.testing.assert_allclose(delta_coarse, expected, rtol=1e-5, atol=1e-7)
    meta_coarse = getattr(recon_coarse, "_last_fast_linear_meta", {})
    assert meta_coarse["resolved_preconditioner"] == "coarse"
    assert meta_coarse["path"] == "pcg-coarse-pmat-precond"
    assert meta_coarse["matrix_free_pc_source"] == "coarse-pmat"

    recon_custom = _dummy_reconstructor("auto", preconditioner="custom", fast_linear_path="pcg")
    recon_custom.matrix_free_pc_action = lambda x: np.asarray(x, dtype=float) / np.array([1.2, 1.8, 2.6])
    delta_custom, _, _ = _solve_linear_system_fast(
        recon_custom,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=9,
    )
    np.testing.assert_allclose(delta_custom, expected, rtol=1e-5, atol=1e-7)
    meta_custom = getattr(recon_custom, "_last_fast_linear_meta", {})
    assert meta_custom["resolved_preconditioner"] == "custom"
    assert meta_custom["path"] == "pcg-custom-pcshell-precond"
    assert meta_custom["matrix_free_pc_source"] == "custom-pcshell"


def test_fast_solver_matrix_free_petsc_gamg_with_pmat_uses_compatible_pmat(monkeypatch: pytest.MonkeyPatch):
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15
    op = LinearOperator(
        J.shape,
        matvec=lambda x: J @ np.asarray(x, dtype=float),
        rmatvec=lambda x: J.T @ np.asarray(x, dtype=float),
        dtype=np.float64,
    )
    monkeypatch.setattr(
        gn_runtime,
        "detect_performance_capabilities",
        lambda: {
            "pyamg": False,
            "cholmod": False,
            "petsc_mat_solve": False,
            "petsc_gamg": True,
        },
    )

    recon = _dummy_reconstructor("auto", preconditioner="petsc-gamg", fast_linear_path="pcg")
    recon.matrix_free_pmat = diags([1.2, 1.8, 2.6], 0, format="csr")
    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=op,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=10,
    )

    np.testing.assert_allclose(delta, _expected_solution(J, residual, de, lam), rtol=1e-5, atol=1e-7)
    meta = getattr(recon, "_last_fast_linear_meta", {})
    assert meta["resolved_preconditioner"] == "pmat"
    assert meta["matrix_free_pc_source"] == "pmat"
    assert meta["matrix_free_pmat_requested_preconditioner"] == "petsc-gamg"
    assert "petsc_gamg_not_supported_in_matrix_free" not in str(meta.get("fallback_reason"))


def test_fast_solver_jacobian_linearization_supports_weights_and_callable_regularization():
    J = np.array(
        [
            [0.4, -0.1, 0.3],
            [0.2, 0.8, -0.4],
            [-0.5, 0.6, 0.1],
        ],
        dtype=float,
    )
    weights = np.array([2.0, 0.5, 1.5], dtype=float)
    residual = np.array([0.03, -0.02, 0.04], dtype=float)
    weighted_residual = weights * residual
    de = np.array([0.1, -0.05, 0.2], dtype=float)
    lam = 0.07
    reg_diag = np.array([1.0, 2.0, 3.0], dtype=float)
    J_weighted = J * weights[:, None]
    expected = np.linalg.solve(
        J_weighted.T @ J_weighted + lam * np.diag(reg_diag),
        -(J_weighted.T @ weighted_residual + lam * (reg_diag * de)),
    )

    recon = _dummy_reconstructor("auto", preconditioner="diag", fast_linear_path="pcg")
    recon.R_matrix = lambda x: reg_diag * np.asarray(x, dtype=float)
    recon.R_diag = reg_diag
    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=_make_linearization_from_dense(J),
        measurement_weight_np=weights,
        weighted_residual_np=weighted_residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=3,
    )

    np.testing.assert_allclose(delta, expected, rtol=1e-5, atol=1e-7)
    meta = getattr(recon, "_last_fast_linear_meta", {})
    assert meta["jacobian_representation"] == "jacobian_linearization"
    assert meta["jacobian_shape"] == [3, 3]
    assert meta["dense_jacobian_materialized"] is False
    assert isinstance(meta["linear_iterations"], int)


def test_fast_solver_cholmod_preconditioner_and_fallback(monkeypatch: pytest.MonkeyPatch):
    J = np.array(
        [
            [0.8, -0.2, 0.5],
            [0.1, 0.7, -0.3],
            [0.6, 0.4, 0.2],
            [-0.2, 0.3, 0.9],
        ],
        dtype=float,
    )
    residual = np.array([0.1, -0.04, 0.03, 0.05], dtype=float)
    de = np.array([0.2, -0.1, 0.08], dtype=float)
    lam = 0.15

    class _FakeFactor:
        def __init__(self, mat):
            dense = mat.toarray()
            dense = 0.5 * (dense + dense.T)
            self._dense = dense

        def solve_A(self, rhs):
            return np.linalg.solve(self._dense, rhs)

    monkeypatch.setattr(
        gn_runtime,
        "detect_performance_capabilities",
        lambda: {
            "pyamg": False,
            "cholmod": True,
            "petsc_mat_solve": False,
            "petsc_gamg": False,
        },
    )

    recon = _dummy_reconstructor("auto", preconditioner="cholmod", fast_linear_path="pcg")
    monkeypatch.setattr(gn_runtime, "cholmod_cholesky", lambda mat: _FakeFactor(mat))
    delta, _, _ = _solve_linear_system_fast(
        recon,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=0,
    )
    expected = _expected_solution(J, residual, de, lam)
    assert np.allclose(delta, expected, rtol=1e-5, atol=1e-7)
    meta = getattr(recon, "_last_fast_linear_meta", {})
    assert "cholmod-precond" in str(meta.get("path", ""))

    monkeypatch.setattr(
        gn_runtime,
        "cholmod_cholesky",
        lambda _mat: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    recon2 = _dummy_reconstructor("auto", preconditioner="cholmod", fast_linear_path="pcg")
    delta2, _, _ = _solve_linear_system_fast(
        recon2,
        J_weighted_np=J,
        weighted_residual_np=residual,
        de_current_np=de,
        lambda_eff=lam,
        iteration=1,
    )
    assert np.allclose(delta2, expected, rtol=5e-4, atol=1e-5)
    meta2 = getattr(recon2, "_last_fast_linear_meta", {})
    assert "cholmod" in str(meta2.get("fallback_reason", ""))
