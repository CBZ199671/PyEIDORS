"""Regularization operator tests for sparse/linear-operator pathways."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, diags, isspmatrix
from scipy.sparse.linalg import LinearOperator

from pyeidors.inverse.regularization.base_regularization import BaseRegularization
from pyeidors.inverse.regularization.smoothness import (
    NOSERRegularization,
    TotalVariationRegularization,
)


def _fake_model(n_elements: int = 6):
    index_map = SimpleNamespace(size_local=n_elements)
    dofmap = SimpleNamespace(index_map=index_map, index_map_bs=1)
    v_sigma = SimpleNamespace(dofmap=dofmap)
    return SimpleNamespace(mesh=object(), V_sigma=v_sigma)


class _DummyRegularization(BaseRegularization):
    def __init__(self, fwd_model, payload):
        super().__init__(fwd_model)
        self._payload = payload

    def create_matrix(self):
        return self._payload


def test_noser_returns_sparse_diagonal_when_baseline_is_ready():
    fwd_model = _fake_model(5)
    reg = NOSERRegularization(
        fwd_model,
        jacobian_calculator=SimpleNamespace(),
        base_conductivity=1.0,
        alpha=2.0,
        exponent=0.5,
    )
    reg._baseline_diag = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=float)

    matrix = reg.get_regularization_matrix(cache=False)
    assert isspmatrix(matrix)
    assert matrix.shape == (5, 5)
    assert np.allclose(matrix.diagonal(), 2.0 * np.sqrt(reg._baseline_diag))
    assert np.isfinite(matrix.data).all()


def test_as_linear_operator_supports_dense_sparse_and_linearop():
    fwd_model = _fake_model(4)
    dense_reg = _DummyRegularization(fwd_model, np.eye(4, dtype=float))
    sparse_reg = _DummyRegularization(fwd_model, diags([1.0, 2.0, 3.0, 4.0], 0, format="csr"))
    base_linear = LinearOperator((4, 4), matvec=lambda v: np.asarray(v, dtype=float) * 3.0)
    lop_reg = _DummyRegularization(fwd_model, base_linear)

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    dense_op = dense_reg.as_linear_operator(dense_reg.get_regularization_matrix())
    sparse_op = sparse_reg.as_linear_operator(sparse_reg.get_regularization_matrix())
    linear_op = lop_reg.as_linear_operator(lop_reg.get_regularization_matrix())

    assert np.allclose(dense_op.matvec(x), x)
    assert np.allclose(sparse_op.matvec(x), np.array([1.0, 4.0, 9.0, 16.0], dtype=float))
    assert np.allclose(linear_op.matvec(x), x * 3.0)

    csr_payload = csr_matrix(np.eye(4))
    assert np.allclose(dense_reg.as_linear_operator(csr_payload).matvec(x), x)


def test_total_variation_regularization_returns_sparse_matrix(eit_system):
    reg = TotalVariationRegularization(
        eit_system.fwd_model,
        alpha=1.5,
        epsilon=1e-6,
        reference_conductivity=1.0,
    )
    matrix = reg.get_regularization_matrix(cache=False)
    assert isspmatrix(matrix)
    assert matrix.shape == (eit_system.reconstructor.n_elements, eit_system.reconstructor.n_elements)
    diag = np.asarray(matrix.diagonal(), dtype=float)
    assert np.isfinite(diag).all()
    assert np.all(diag >= 0.0)
