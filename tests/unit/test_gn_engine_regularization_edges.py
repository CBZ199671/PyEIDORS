"""Additional regularization-matrix edge tests for the GN engine."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

import pyeidors.inverse.solvers.gauss_newton_engine as gn_engine


def _make_reconstructor(
    matrix,
    *,
    n_elements: int = 2,
    solver_mode: str = "fast",
    line_search_mode: str = "fast",
    as_linear_operator=None,
):
    reconstructor = gn_engine.GaussNewtonReconstructor.__new__(gn_engine.GaussNewtonReconstructor)
    reconstructor.n_elements = int(n_elements)
    reconstructor.solver_mode = solver_mode
    reconstructor.line_search_mode = line_search_mode
    reconstructor.device = torch.device("cpu")
    reconstructor._torch_dtype = torch.float64
    reconstructor.R_matrix = None
    reconstructor.R_linear_operator = None
    reconstructor.R_diag = None
    reconstructor.R_torch = None

    regularization = SimpleNamespace(get_regularization_matrix=lambda: matrix)
    if as_linear_operator is not None:
        regularization.as_linear_operator = as_linear_operator
    reconstructor.regularization = regularization
    return reconstructor


def test_validate_option_rejects_invalid_value():
    with pytest.raises(ValueError, match="Unsupported mode='bad'"):
        gn_engine._validate_option("mode", "bad", {"good", "ok"})


def test_ensure_regularization_ready_rejects_shape_mismatch():
    reconstructor = _make_reconstructor(np.eye(3, dtype=float), n_elements=2)
    with pytest.raises(RuntimeError, match="Regularization matrix shape mismatch"):
        reconstructor.ensure_regularization_ready()


def test_ensure_regularization_ready_rejects_empty_and_nonfinite_sparse():
    empty_sparse = _make_reconstructor(sparse.csr_matrix((2, 2), dtype=float))
    with pytest.raises(FloatingPointError, match="sparse matrix is empty"):
        empty_sparse.ensure_regularization_ready()

    nonfinite_sparse = _make_reconstructor(sparse.csr_matrix(np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=float)))
    with pytest.raises(FloatingPointError, match="contains non-finite values"):
        nonfinite_sparse.ensure_regularization_ready()


def test_ensure_regularization_ready_extracts_diag_from_noncsr_sparse():
    reconstructor = _make_reconstructor(sparse.csc_matrix(np.diag([2.0, 3.0])))
    reconstructor.ensure_regularization_ready()
    np.testing.assert_allclose(reconstructor.R_diag, np.array([2.0, 3.0], dtype=float))
    assert reconstructor.R_torch is None


def test_ensure_regularization_ready_handles_linear_operator_modes():
    finite_op = LinearOperator((2, 2), matvec=lambda x: np.asarray(x, dtype=float))
    reconstructor = _make_reconstructor(finite_op, solver_mode="fast")
    reconstructor.ensure_regularization_ready()
    assert reconstructor.R_diag is None
    assert reconstructor.R_torch is None

    bad_op = LinearOperator((2, 2), matvec=lambda _x: np.array([np.nan, 0.0], dtype=float))
    with pytest.raises(FloatingPointError, match="LinearOperator produces non-finite"):
        _make_reconstructor(bad_op, solver_mode="fast").ensure_regularization_ready()

    with pytest.raises(RuntimeError, match="LinearOperator is not supported"):
        _make_reconstructor(finite_op, solver_mode="strict").ensure_regularization_ready()


def test_ensure_regularization_ready_rejects_nonfinite_dense_and_transfer_failures(monkeypatch: pytest.MonkeyPatch):
    nonfinite_dense = _make_reconstructor(np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=float), solver_mode="strict")
    with pytest.raises(FloatingPointError, match="Regularization matrix contains non-finite values"):
        nonfinite_dense.ensure_regularization_ready()

    reconstructor = _make_reconstructor(np.eye(2, dtype=float), solver_mode="strict")
    monkeypatch.setattr(gn_engine.torch, "isfinite", lambda _tensor: torch.tensor([False]))
    with pytest.raises(FloatingPointError, match="contains non-finite values after transfer"):
        reconstructor.ensure_regularization_ready()


def test_ensure_regularization_ready_dense_fast_mode_keeps_tensor_none():
    reconstructor = _make_reconstructor(
        np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float),
        solver_mode="fast",
        line_search_mode="fast",
    )
    reconstructor.ensure_regularization_ready()
    np.testing.assert_allclose(reconstructor.R_diag, np.array([2.0, 3.0], dtype=float))
    assert reconstructor.R_torch is None
