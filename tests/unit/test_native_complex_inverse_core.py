from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
from scipy.sparse.linalg import LinearOperator

from pyeidors.inverse.jacobian.linearized import (
    JacobianLinearization,
    LazyAdjointJacobianLinearization,
)
from pyeidors.inverse.solvers.gauss_newton_linear_system import (
    _as_jacobian_action_bundle,
    _regularization_matrix_for_native_complex,
    _solve_linear_system_fast,
)
from pyeidors.inverse.solvers.gauss_newton_measurement_space import (
    _configure_measurement_space,
    _extract_measured_vector,
)
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.data.difference import build_difference_vector


def test_measurement_dataset_normalization_preserves_complex_values() -> None:
    values = np.array([[1.0 + 0.25j, 2.0 - 0.5j]], dtype=np.complex64)

    normalized = MeasurementDataset._normalize_measurements(values)

    assert normalized.dtype == np.complex64
    assert np.allclose(normalized, values)


def test_difference_vector_preserves_complex64_dtype() -> None:
    target = np.array([1.0 + 0.25j, 2.0 - 0.5j], dtype=np.complex64)
    reference = np.array([0.8 + 0.1j, 1.8 - 0.2j], dtype=np.complex64)

    diff = build_difference_vector(target, reference, mode="raw")

    assert diff.dtype == np.complex64
    assert np.allclose(diff, target - reference)


def test_measurement_space_preserves_complex64_phasors() -> None:
    measured = SimpleNamespace(
        meas=np.array([1.0 + 0.25j, 2.0 - 0.5j], dtype=np.complex64),
        type="difference",
        reference_meas=np.array([0.8 + 0.1j, 1.8 - 0.2j], dtype=np.complex64),
        target_meas=np.array([1.0 + 0.25j, 2.0 - 0.5j], dtype=np.complex64),
        difference_mode="raw",
        difference_orientation="target_minus_reference",
    )
    reconstructor = SimpleNamespace(
        difference_mode="raw",
        difference_orientation="target_minus_reference",
    )

    vector = _extract_measured_vector(measured)
    _configure_measurement_space(reconstructor, measured)

    assert vector.dtype == np.complex64
    assert reconstructor._difference_reference_meas.dtype == np.complex64
    assert reconstructor._difference_target_meas.dtype == np.complex64


def test_dense_jacobian_action_uses_hermitian_rmatvec_for_complex() -> None:
    jacobian = np.array(
        [
            [1.0 + 2.0j, 0.5 - 0.25j],
            [0.25 + 0.5j, -1.0 + 0.75j],
        ],
        dtype=np.complex128,
    )
    residual = np.array([0.2 - 0.1j, -0.4 + 0.3j], dtype=np.complex128)

    bundle = _as_jacobian_action_bundle(jacobian)

    assert np.allclose(bundle.rmatvec(residual), jacobian.conj().T @ residual)


def test_native_complex_linear_operator_regularization_direct_fills_dense_matrix(
    monkeypatch,
) -> None:
    matrix = np.array(
        [[2.0, -0.5, 0.25], [0.1, 1.5, -0.2], [0.0, 0.3, 1.0]],
        dtype=np.float64,
    )
    op = LinearOperator(
        matrix.shape,
        matvec=lambda vector: matrix @ np.asarray(vector, dtype=np.float64),
        dtype=np.float64,
    )

    def _fail_eye(*_args, **_kwargs):
        raise AssertionError("LinearOperator path must not allocate dense eye")

    def _fail_column_stack(*_args, **_kwargs):
        raise AssertionError("LinearOperator path must not use column_stack")

    import pyeidors.inverse.solvers.gauss_newton_linear_system as linear_system

    monkeypatch.setattr(linear_system.np, "eye", _fail_eye)
    monkeypatch.setattr(linear_system.np, "column_stack", _fail_column_stack)

    actual = _regularization_matrix_for_native_complex(op, 3)

    np.testing.assert_allclose(actual, matrix)


def test_v520_native_complex_regularization_identity_and_diagonal_direct_fill(
    monkeypatch,
) -> None:
    import pyeidors.inverse.solvers.gauss_newton_linear_system as linear_system

    source = inspect.getsource(linear_system._regularization_matrix_for_native_complex)
    assert "np.eye(int(n_param)" not in source
    assert "np.diag(arr)" not in source
    assert "_dense_identity_matrix(int(n_param))" in source
    assert "_dense_diagonal_matrix(arr)" in source

    def _fail_eye(*_args, **_kwargs):
        raise AssertionError(
            "native complex identity regularization must not use np.eye"
        )

    def _fail_diag(*_args, **_kwargs):
        raise AssertionError(
            "native complex diagonal regularization must not use np.diag"
        )

    monkeypatch.setattr(linear_system.np, "eye", _fail_eye)
    monkeypatch.setattr(linear_system.np, "diag", _fail_diag)

    identity = _regularization_matrix_for_native_complex(None, 3)
    diagonal = _regularization_matrix_for_native_complex(
        np.array([2.0, 3.0, 4.0], dtype=np.float64),
        3,
    )

    expected_identity = np.zeros((3, 3), dtype=np.float64)
    expected_identity.reshape(-1)[::4] = 1.0
    expected_diagonal = np.zeros((3, 3), dtype=np.float64)
    expected_diagonal.reshape(-1)[::4] = np.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(identity, expected_identity)
    np.testing.assert_allclose(diagonal, expected_diagonal)


def test_v510_native_complex_normal_step_adds_diagonal_regularization_in_place() -> (
    None
):
    from pyeidors.inverse.solvers.gauss_newton_linear_system import (
        solve_native_complex_normal_step,
    )
    import pyeidors.inverse.solvers.gauss_newton_linear_system as linear_system

    source = inspect.getsource(linear_system.solve_native_complex_normal_step)
    payload_source = inspect.getsource(
        linear_system._regularization_payload_for_native_complex
    )

    assert "J_h @ J + float(lambda_eff) * reg" not in source
    assert "float(lambda_eff) * (reg @ prior)" not in source
    assert "add_scaled_diagonal_in_place(" in source
    assert "add_scaled_values_in_place(" in source
    assert "np.asarray(diag, dtype=dtype)" in payload_source

    jacobian = np.array(
        [
            [1.0 + 0.5j, 0.5 - 0.25j],
            [0.25 + 0.5j, -1.0 + 0.75j],
            [0.75 - 0.1j, 0.3 + 0.2j],
        ],
        dtype=np.complex128,
    )
    residual = np.array([0.2 - 0.1j, -0.4 + 0.3j, 0.1 + 0.2j])
    diagonal = np.array([2.0, 3.0], dtype=np.float64)

    diagonal_delta, diagonal_meta = solve_native_complex_normal_step(
        jacobian=jacobian,
        residual=residual,
        lambda_eff=0.25,
        regularization=diagonal,
    )
    dense_delta, dense_meta = solve_native_complex_normal_step(
        jacobian=jacobian,
        residual=residual,
        lambda_eff=0.25,
        regularization=np.diag(diagonal),
    )
    identity_delta, identity_meta = solve_native_complex_normal_step(
        jacobian=jacobian,
        residual=residual,
        lambda_eff=0.25,
        regularization=None,
    )
    dense_identity_delta, _ = solve_native_complex_normal_step(
        jacobian=jacobian,
        residual=residual,
        lambda_eff=0.25,
        regularization=np.eye(2, dtype=np.float64),
    )

    np.testing.assert_allclose(diagonal_delta, dense_delta)
    np.testing.assert_allclose(identity_delta, dense_identity_delta)
    assert diagonal_meta["regularization_kind"] == "diagonal"
    assert dense_meta["regularization_kind"] == "matrix"
    assert identity_meta["regularization_kind"] == "identity"


def test_linearized_jacobian_complex_actions_match_dense_hermitian() -> None:
    grad_u = np.array(
        [[1.0 + 0.5j, 2.0 - 0.25j], [0.5 - 1.0j, -1.0 + 0.75j]],
        dtype=np.complex128,
    )
    adjoints = (
        np.array([[0.25 + 1.0j, 1.5 - 0.5j], [1.0 - 0.25j, 0.5 + 0.5j]]),
        np.array([[-1.0 + 0.25j, 0.75 + 0.5j], [0.5 + 1.5j, -0.25 + 0.75j]]),
    )
    lin = JacobianLinearization(
        grad_u_all=(grad_u,),
        adjoint_gradients=adjoints,
        cell_areas=np.array([0.2, 0.4], dtype=np.float64),
        n_meas_per_stim=(2,),
    )
    dense = lin.to_dense()
    vector = np.array([0.2 + 0.1j, -0.3 + 0.4j], dtype=np.complex128)
    residual = np.array([0.5 - 0.2j, -0.1 + 0.3j], dtype=np.complex128)

    assert np.iscomplexobj(dense)
    assert np.allclose(lin.matvec(vector), dense @ vector)
    assert np.allclose(lin.rmatvec(residual), dense.conj().T @ residual)


def test_linearized_complex_hessian_diag_matches_hermitian_normal_diagonal() -> None:
    grad_u = np.array(
        [[1.0 + 0.5j, 2.0 - 0.25j], [0.5 - 1.0j, -1.0 + 0.75j]],
        dtype=np.complex128,
    )
    adjoints = (
        np.array([[0.25 + 1.0j, 1.5 - 0.5j], [1.0 - 0.25j, 0.5 + 0.5j]]),
        np.array([[-1.0 + 0.25j, 0.75 + 0.5j], [0.5 + 1.5j, -0.25 + 0.75j]]),
    )
    lin = JacobianLinearization(
        grad_u_all=(grad_u,),
        adjoint_gradients=adjoints,
        cell_areas=np.array([0.2, 0.4], dtype=np.float64),
        n_meas_per_stim=(2,),
    )
    weights = np.array([2.0, 0.5], dtype=np.float64)
    alpha = 0.1
    regularization_diag = np.array([0.25, 0.75], dtype=np.float64)

    dense = lin.to_dense()
    expected = (
        np.real(np.diag(dense.conj().T @ np.diag(weights) @ dense))
        + alpha * regularization_diag
    )

    np.testing.assert_allclose(
        lin.hessian_diag(
            measurement_weights=weights,
            alpha=alpha,
            regularization_diag=regularization_diag,
        ),
        expected,
    )


def test_linearized_jacobian_preserves_complex64_precision() -> None:
    grad_u = np.array(
        [[1.0 + 0.5j], [0.5 - 1.0j]],
        dtype=np.complex64,
    )
    adjoints = (
        np.array([[0.25 + 1.0j], [1.0 - 0.25j]], dtype=np.complex64),
        np.array([[-1.0 + 0.25j], [0.5 + 1.5j]], dtype=np.complex64),
    )
    lin = JacobianLinearization(
        grad_u_all=(grad_u,),
        adjoint_gradients=adjoints,
        cell_areas=np.array([0.2, 0.4], dtype=np.float64),
        n_meas_per_stim=(2,),
    )
    vector = np.array([0.2 + 0.1j, -0.3 + 0.4j], dtype=np.complex64)
    residual = np.array([0.5 - 0.2j, -0.1 + 0.3j], dtype=np.complex64)

    assert lin.dtype == np.dtype(np.complex64)
    assert lin.to_dense().dtype == np.complex64
    assert lin.matvec(vector).dtype == np.complex64
    assert lin.rmatvec(residual).dtype == np.complex64


def test_lazy_adjoint_complex_rmatvec_matches_dense_hermitian() -> None:
    class FakeLazy(LazyAdjointJacobianLinearization):
        def _gradients_for_patterns(self, patterns, *, rhs_kind):
            self.captured_patterns = np.asarray(patterns)
            combo = np.einsum("se,eag->sag", patterns, self.base_gradients)
            return tuple(combo)

    grad_u = np.array([[1.0 + 0.5j], [0.5 - 1.0j]], dtype=np.complex128)
    base_gradients = np.array(
        [
            [[0.25 + 1.0j], [1.0 - 0.25j]],
            [[1.5 - 0.5j], [0.5 + 0.5j]],
        ],
        dtype=np.complex128,
    )
    meas_matrix = np.array([[1.0, -0.5], [0.25, 1.5]], dtype=np.float64)
    lin = FakeLazy(
        fwd_model=SimpleNamespace(n_elec=2),
        sigma_values=np.ones(2, dtype=np.complex128),
        u_all=(np.ones(2, dtype=np.complex128),),
        grad_u_all=(grad_u,),
        cell_areas=np.array([0.2, 0.4], dtype=np.float64),
        n_meas_per_stim=(2,),
        meas_matrices=(meas_matrix,),
        gradient_callback=lambda _fields: (),
    )
    lin.base_gradients = base_gradients
    adjoint_per_measurement = np.einsum("me,eag->mag", meas_matrix, base_gradients)
    dense = (
        np.einsum("eg,meg->me", grad_u, adjoint_per_measurement, optimize=True)
        * lin.cell_areas[None, :]
        * lin.sign
    )
    residual = np.array([0.5 - 0.2j, -0.1 + 0.3j], dtype=np.complex128)

    assert np.allclose(lin.rmatvec(residual), dense.conj().T @ residual)
    assert np.allclose(lin.captured_patterns, meas_matrix.T @ np.conj(residual))


def test_fast_dense_solver_routes_complex_to_native_hermitian_system() -> None:
    jacobian = np.array(
        [
            [1.0 + 2.0j, 0.5 - 0.25j],
            [0.25 + 0.5j, -1.0 + 0.75j],
            [1.5 - 0.5j, 0.25 + 1.0j],
        ],
        dtype=np.complex128,
    )
    true_delta = np.array([0.2 + 0.3j, -0.1 + 0.4j], dtype=np.complex128)
    residual = -(jacobian @ true_delta)
    reconstructor = SimpleNamespace(
        R_matrix=np.eye(2, dtype=np.float64),
        R_diag=np.ones(2, dtype=np.float64),
        use_prior_term=False,
        performance_mode="aggressive",
        linear_solver="auto",
        preconditioner="auto",
        fast_linear_path="auto",
        solver_mode="fast",
        regularization=None,
    )

    delta, delta_norm, jtr_norm = _solve_linear_system_fast(
        reconstructor,
        J_weighted_np=jacobian,
        weighted_residual_np=residual,
        de_current_np=np.zeros(2, dtype=np.complex128),
        lambda_eff=0.0,
        iteration=0,
    )

    assert np.allclose(delta, true_delta)
    assert delta_norm > 0.0
    assert jtr_norm > 0.0
    assert reconstructor._last_fast_linear_meta["path"] == "native-complex-dense"
    assert reconstructor._last_fast_linear_meta["native_complex_linear_algebra"] is True


def test_gn_runtime_complex_helpers_use_hermitian_and_vdot() -> None:
    import torch

    from pyeidors.inverse.solvers.gauss_newton_runtime import (
        _build_linear_system,
        _hermitian_transpose,
        _vdot_real_torch,
    )

    jacobian = torch.tensor(
        [
            [1.0 + 0.5j, 0.25 - 0.75j],
            [-0.5 + 1.0j, 0.75 + 0.25j],
        ],
        dtype=torch.complex64,
    )
    residual = torch.tensor([0.2 - 0.1j, -0.4 + 0.3j], dtype=torch.complex64)
    current = torch.tensor([0.1 + 0.2j, -0.3 + 0.1j], dtype=torch.complex64)
    reconstructor = SimpleNamespace(
        R_torch=torch.eye(2, dtype=torch.complex64),
        use_prior_term=True,
    )

    jacobian_h = _hermitian_transpose(jacobian)
    A, b = _build_linear_system(
        reconstructor,
        jacobian_h @ jacobian,
        jacobian_h @ residual,
        current,
        lambda_eff=0.25,
        iteration=0,
    )

    assert torch.allclose(jacobian_h, jacobian.conj().transpose(0, 1))
    assert (
        _vdot_real_torch(residual, residual)
        == torch.vdot(
            residual.reshape(-1),
            residual.reshape(-1),
        ).real.item()
    )
    assert torch.allclose(A, jacobian.conj().T @ jacobian + 0.25 * torch.eye(2))
    assert torch.allclose(
        b,
        -(jacobian.conj().T @ residual + 0.25 * current),
    )


def test_native_complex_normal_step_preserves_complex64_precision() -> None:
    from pyeidors.inverse.solvers.gauss_newton_linear_system import (
        solve_native_complex_normal_step,
    )

    jacobian = np.array(
        [[1.0 + 0.5j, 0.5 - 0.25j], [0.25 + 0.5j, -1.0 + 0.75j]],
        dtype=np.complex64,
    )
    true_delta = np.array([0.2 + 0.3j, -0.1 + 0.4j], dtype=np.complex64)
    residual = -(jacobian @ true_delta)

    delta, meta = solve_native_complex_normal_step(
        jacobian=jacobian,
        residual=residual,
        lambda_eff=0.0,
        regularization=np.eye(2, dtype=np.float64),
    )

    assert delta.dtype == np.complex64
    assert meta["jacobian_dtype"] == "complex64"
    assert np.allclose(delta, true_delta, rtol=1e-5, atol=1e-6)
