"""T80 entrance gate: shared base-class surface for the two Jacobian linearizations.

Freezes the cross-class API contract that ``JacobianLinearization``
(eager, EIDORS-runtime sign ``+1.0``) and
``LazyAdjointJacobianLinearization`` (lazy adjoint, EIDORS-canonical
sign ``-1.0``) must both honor.  Both expose the same public matrix-free
operator surface:

* ``n_parameters`` / ``n_measurements`` / ``shape`` derived from
  ``cell_areas`` + ``n_meas_per_stim``.
* ``assert_compatible`` permissive on empty fingerprints (V9).
* ``as_linear_operator`` returning a SciPy ``LinearOperator`` of
  ``(n_measurements, n_parameters)`` shape and ``float64`` dtype.

The eager path additionally has to honor V7
(``matvec == to_dense @ v`` / ``rmatvec == to_dense.T @ r``) and V8
(``hessian_diag`` formula + ``sign²`` + floor) — the eager class is the
one cited by V7/V8 because it owns the dense-equivalent kernel.

These tests block the T80 fusion (``_LinearizationBase`` extraction)
from regressing the shared surface or the V7/V8/V9 contracts.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator

from pyeidors.inverse.jacobian.linearized import (
    JacobianLinearization,
    LazyAdjointJacobianLinearization,
    _weighted_contrib_power_sum,
)


def _eager_fixture(rng_seed: int = 7) -> JacobianLinearization:
    rng = np.random.default_rng(rng_seed)
    n_elem = 6
    cell_areas = np.linspace(0.1, 1.0, n_elem, dtype=np.float64)
    n_meas_per_stim = (3, 2)
    n_meas_total = sum(n_meas_per_stim)
    grad_u_all = tuple(rng.standard_normal((n_elem, 2)) for _ in n_meas_per_stim)
    adjoint_gradients = tuple(
        rng.standard_normal((n_elem, 2)) for _ in range(n_meas_total)
    )
    return JacobianLinearization(
        grad_u_all=grad_u_all,
        adjoint_gradients=adjoint_gradients,
        cell_areas=cell_areas,
        n_meas_per_stim=n_meas_per_stim,
        sign=1.0,
        sigma_fingerprint="abc123",
    )


def _lazy_fixture(rng_seed: int = 11) -> LazyAdjointJacobianLinearization:
    """Build a Lazy linearization that exercises only base-class API.

    The fwd_model is intentionally a sentinel — none of the base-class
    methods exercised here trigger ``forward_solve`` /
    ``solve_full_rhs``, so a placeholder object is enough.
    """
    rng = np.random.default_rng(rng_seed)
    n_elem = 5
    n_meas_per_stim = (2, 4)
    sigma_values = rng.standard_normal(n_elem)
    u_all = tuple(rng.standard_normal(7) for _ in n_meas_per_stim)
    grad_u_all = tuple(rng.standard_normal((n_elem, 2)) for _ in n_meas_per_stim)
    cell_areas = np.linspace(0.2, 1.5, n_elem, dtype=np.float64)
    meas_matrices = tuple(
        rng.standard_normal((n_meas, 4)) for n_meas in n_meas_per_stim
    )
    fwd_model = SimpleNamespace(
        V_sigma=SimpleNamespace(),
        n_elec=4,
        dofs=7,
        V=SimpleNamespace(),
        phi=SimpleNamespace(),
    )
    return LazyAdjointJacobianLinearization(
        fwd_model=fwd_model,
        sigma_values=sigma_values,
        u_all=u_all,
        grad_u_all=grad_u_all,
        cell_areas=cell_areas,
        n_meas_per_stim=n_meas_per_stim,
        meas_matrices=meas_matrices,
        gradient_callback=lambda fields: tuple(np.zeros((n_elem, 2)) for _ in fields),
        sign=-1.0,
        sigma_fingerprint="lazy-fp",
    )


def test_shared_shape_properties_match_cell_areas_and_n_meas_per_stim() -> None:
    eager = _eager_fixture()
    lazy = _lazy_fixture()

    assert eager.n_parameters == 6
    assert eager.n_measurements == 5
    assert eager.shape == (5, 6)

    assert lazy.n_parameters == 5
    assert lazy.n_measurements == 6
    assert lazy.shape == (6, 5)


def test_assert_compatible_permissive_on_empty_fingerprint() -> None:
    """V9: permissive when either stored or provided fingerprint is empty."""
    eager = _eager_fixture()
    lazy = _lazy_fixture()

    eager.assert_compatible(None)
    eager.assert_compatible("")
    lazy.assert_compatible(None)
    lazy.assert_compatible("")

    empty_fp_eager = JacobianLinearization(
        grad_u_all=eager.grad_u_all,
        adjoint_gradients=eager.adjoint_gradients,
        cell_areas=eager.cell_areas,
        n_meas_per_stim=eager.n_meas_per_stim,
        sign=eager.sign,
        sigma_fingerprint="",
    )
    empty_fp_eager.assert_compatible("anything")  # stored empty -> skip


def test_assert_compatible_raises_on_fingerprint_mismatch() -> None:
    """V9: both fingerprints non-empty + different -> ValueError."""
    eager = _eager_fixture()
    lazy = _lazy_fixture()

    with pytest.raises(ValueError, match="JacobianLinearization"):
        eager.assert_compatible("different-fingerprint")

    with pytest.raises(ValueError, match="LazyAdjointJacobianLinearization"):
        lazy.assert_compatible("different-fingerprint")


def test_as_linear_operator_returns_scipy_operator_with_correct_shape_dtype() -> None:
    eager = _eager_fixture()
    lazy = _lazy_fixture()

    op_eager = eager.as_linear_operator()
    op_lazy = lazy.as_linear_operator()

    assert isinstance(op_eager, LinearOperator)
    assert isinstance(op_lazy, LinearOperator)
    assert op_eager.shape == eager.shape
    assert op_lazy.shape == lazy.shape
    assert op_eager.dtype == np.float64
    assert op_lazy.dtype == np.float64


def test_eager_matvec_and_rmatvec_match_to_dense_action() -> None:
    """V7: eager ``matvec(v) == to_dense() @ v``, ``rmatvec(r) == to_dense().T @ r``."""
    eager = _eager_fixture()
    dense = eager.to_dense()
    rng = np.random.default_rng(31)

    for _ in range(3):
        v = rng.standard_normal(eager.n_parameters)
        np.testing.assert_allclose(eager.matvec(v), dense @ v, rtol=1e-12, atol=1e-14)

        r = rng.standard_normal(eager.n_measurements)
        np.testing.assert_allclose(
            eager.rmatvec(r), dense.T @ r, rtol=1e-12, atol=1e-14
        )


def test_eager_hessian_diag_matches_jt_w_j_diagonal_with_floor() -> None:
    """V8: ``hessian_diag == diag(J^T W J) + alpha * R_diag`` clamped >= floor."""
    eager = _eager_fixture()
    dense = eager.to_dense()
    weights = np.linspace(0.5, 2.0, eager.n_measurements, dtype=np.float64)
    reg_diag = np.linspace(0.1, 0.4, eager.n_parameters, dtype=np.float64)
    alpha = 0.7
    floor = 0.05

    expected = np.diag(dense.T @ np.diag(weights) @ dense) + alpha * reg_diag
    expected = np.maximum(expected, floor)

    actual = eager.hessian_diag(
        measurement_weights=weights,
        alpha=alpha,
        regularization_diag=reg_diag,
        floor=floor,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_v228_lazy_hessian_diag_chunk_reduction_avoids_broadcast_matrices() -> None:
    exact_source = inspect.getsource(
        LazyAdjointJacobianLinearization._exact_hessian_diag_chunked
    )
    sampled_source = inspect.getsource(
        LazyAdjointJacobianLinearization._sampled_hessian_diag
    )

    assert "block_weights[:, None]" not in exact_source
    assert "selected_weights[:, None]" not in sampled_source
    assert "self.cell_areas[None" not in exact_source
    assert "self.cell_areas[None" not in sampled_source
    assert "_weighted_contrib_power_sum" in exact_source
    assert "_weighted_contrib_power_sum" in sampled_source

    contrib = np.array(
        [
            [1.0 + 2.0j, -0.5 + 0.25j, 2.0 - 1.0j],
            [0.25 - 0.5j, 1.5 + 0.0j, -1.0 + 0.75j],
        ],
        dtype=np.complex128,
    )
    weights = np.array([0.5, 2.0], dtype=np.float64)
    cell_areas = np.array([2.0, 0.5, 1.5], dtype=np.float64)
    expected = (
        np.real(np.conj(contrib) * contrib)
        * weights[:, None]
        * (cell_areas[None, :] ** 2)
    ).sum(axis=0)

    np.testing.assert_allclose(
        _weighted_contrib_power_sum(contrib, weights, cell_areas * cell_areas),
        expected,
    )


def test_v229_eager_to_dense_fills_output_blocks_in_place() -> None:
    source = inspect.getsource(JacobianLinearization.to_dense)

    assert "out=block_view" in source
    assert "self.cell_areas[None" not in source
    assert "block_view *= self.sign" in source
    assert "block_view *= self.cell_areas[start:end]" in source

    eager = _eager_fixture()
    dense_default = eager.to_dense()
    dense_blocked = eager.to_dense(block_size=2)
    np.testing.assert_allclose(dense_blocked, dense_default)


def test_eager_normal_matvec_matches_jtwjv_plus_alpha_rv() -> None:
    """``normal_matvec`` returns ``J^T W J v + alpha * R v`` for any reg adapter."""
    eager = _eager_fixture()
    dense = eager.to_dense()
    rng = np.random.default_rng(101)

    weights = np.linspace(0.4, 1.6, eager.n_measurements, dtype=np.float64)
    R = rng.standard_normal((eager.n_parameters, eager.n_parameters))
    R = R @ R.T  # SPD-ish
    alpha = 0.3

    v = rng.standard_normal(eager.n_parameters)
    expected = dense.T @ (weights * (dense @ v)) + alpha * (R @ v)

    out_dense = eager.normal_matvec(
        v, measurement_weights=weights, alpha=alpha, regularization=R
    )
    np.testing.assert_allclose(out_dense, expected, rtol=1e-10, atol=1e-12)

    out_callable = eager.normal_matvec(
        v,
        measurement_weights=weights,
        alpha=alpha,
        regularization=lambda x: R @ x,
    )
    np.testing.assert_allclose(out_callable, expected, rtol=1e-10, atol=1e-12)

    out_sparse = eager.normal_matvec(
        v,
        measurement_weights=weights,
        alpha=alpha,
        regularization=csr_matrix(R),
    )
    np.testing.assert_allclose(out_sparse, expected, rtol=1e-10, atol=1e-12)

    out_linop = eager.normal_matvec(
        v,
        measurement_weights=weights,
        alpha=alpha,
        regularization=LinearOperator(
            R.shape, matvec=lambda x: R @ x, dtype=np.float64
        ),
    )
    np.testing.assert_allclose(out_linop, expected, rtol=1e-10, atol=1e-12)


def test_default_sign_conventions_match_v73_pairing() -> None:
    """V73: eager defaults to ``sign=+1.0``, lazy defaults to ``sign=-1.0``."""
    eager = JacobianLinearization(
        grad_u_all=(np.zeros((1, 1)),),
        adjoint_gradients=(np.zeros((1, 1)),),
        cell_areas=np.ones(1),
        n_meas_per_stim=(1,),
    )
    assert float(eager.sign) == 1.0

    lazy = LazyAdjointJacobianLinearization(
        fwd_model=SimpleNamespace(),
        sigma_values=np.zeros(1),
        u_all=(np.zeros(1),),
        grad_u_all=(np.zeros((1, 1)),),
        cell_areas=np.ones(1),
        n_meas_per_stim=(1,),
        meas_matrices=(np.eye(1),),
        gradient_callback=lambda fields: tuple(np.zeros((1, 1)) for _ in fields),
    )
    assert float(lazy.sign) == -1.0


def test_eager_as_petsc_mat_creates_python_mat_with_correct_shape() -> None:
    """``as_petsc_mat`` produces a PETSc Python matrix of ``(n_meas, n_param)``.

    Skipped when ``petsc4py`` is unavailable to keep this gate runnable
    in CI shards that do not provision PETSc.
    """
    petsc4py = pytest.importorskip("petsc4py")  # noqa: F841

    eager = _eager_fixture()
    mat = eager.as_petsc_mat()
    try:
        assert mat.getSize() == eager.shape
    finally:
        destroy = getattr(mat, "destroy", None)
        if callable(destroy):
            destroy()
