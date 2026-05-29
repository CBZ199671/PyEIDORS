"""Tests for the generic RtR prior contract."""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import LinearOperator

import pyeidors.inverse.prior.rtr as rtr_module
from pyeidors.inverse.prior import (
    as_rtr_prior,
    load_rtr_prior_artifact,
    write_rtr_prior_artifact,
)


def _legacy_rtr_signature(payload, *, metadata: dict) -> str:
    semantic = json.dumps(
        {
            "schema": rtr_module.RTR_PRIOR_SCHEMA,
            "signature_hint": metadata.get("signature_hint"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if sparse.issparse(payload):
        mat = payload.tocsr()
        encoded = (
            semantic
            + b"|"
            + str(mat.dtype).encode()
            + b"|"
            + json.dumps(list(mat.shape)).encode()
            + b"|"
            + np.ascontiguousarray(mat.indptr, dtype=np.int64).tobytes()
            + b"|"
            + np.ascontiguousarray(mat.indices, dtype=np.int64).tobytes()
            + b"|"
            + np.ascontiguousarray(mat.data, dtype=np.float64).tobytes()
        )
        return hashlib.sha256(encoded).hexdigest()
    arr = np.ascontiguousarray(payload, dtype=np.float64)
    encoded = (
        semantic
        + b"|"
        + str(arr.dtype).encode()
        + b"|"
        + json.dumps(list(arr.shape)).encode()
        + b"|"
        + arr.tobytes()
    )
    return hashlib.sha256(encoded).hexdigest()


def test_rtr_prior_accepts_dense_sparse_operator_and_callable() -> None:
    x = np.array([1.0, -2.0], dtype=float)
    dense = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=float)
    dense_prior = as_rtr_prior(dense, name="dense-demo")
    np.testing.assert_allclose(dense_prior.apply(x), dense @ x)
    np.testing.assert_allclose(dense_prior.diag(), np.array([2.0, 3.0]))
    np.testing.assert_allclose(dense_prior.as_RtR(dense=True), dense)
    assert dense_prior.signature_hash
    assert dense_prior.metadata["name"] == "dense-demo"

    sparse_prior = as_rtr_prior(
        sparse.diags([4.0, 5.0], 0, format="csr"), name="sparse-demo"
    )
    np.testing.assert_allclose(sparse_prior.apply(x), np.array([4.0, -10.0]))
    np.testing.assert_allclose(sparse_prior.diag(), np.array([4.0, 5.0]))
    assert sparse.issparse(sparse_prior.as_RtR(dense=False))

    op = LinearOperator((2, 2), matvec=lambda v: 6.0 * np.asarray(v, dtype=float))
    op_prior = as_rtr_prior(
        op,
        metadata={"diag": [6.0, 6.0], "signature_hint": "scale-six"},
        name="operator-demo",
    )
    np.testing.assert_allclose(op_prior.apply(x), 6.0 * x)
    np.testing.assert_allclose(op_prior.diag(), np.array([6.0, 6.0]))
    assert isinstance(op_prior.as_RtR(dense=False), LinearOperator)

    call_prior = as_rtr_prior(
        lambda v: 7.0 * np.asarray(v, dtype=float),
        shape=(2, 2),
        metadata={"diag": [7.0, 7.0], "signature_hint": "scale-seven"},
        name="callable-demo",
    )
    np.testing.assert_allclose(call_prior.apply(x), 7.0 * x)
    np.testing.assert_allclose(call_prior.diag(), np.array([7.0, 7.0]))
    assert isinstance(call_prior.as_RtR(dense=False), LinearOperator)


def test_rtr_prior_infers_diagonal_shape_and_guards_dense_materialization() -> None:
    diag_prior = as_rtr_prior(np.array([1.0, 2.0, 3.0], dtype=float))
    assert diag_prior.shape == (3, 3)
    np.testing.assert_allclose(diag_prior.diag(), np.array([1.0, 2.0, 3.0]))

    call_prior = as_rtr_prior(
        lambda v: np.asarray(v, dtype=float),
        shape=(3, 3),
        metadata={"signature_hint": "identity-callable"},
    )
    with pytest.raises(ValueError, match="Refusing to materialize RtR dense matrix"):
        call_prior.as_RtR(dense=True, max_dense_n=2)
    np.testing.assert_allclose(call_prior.as_RtR(dense=True), np.eye(3))


def test_v288_rtr_prior_dense_materialization_direct_fills(monkeypatch) -> None:
    expected = np.diag([2.0, 3.0, 4.0])
    prior = as_rtr_prior(
        lambda v: np.array([2.0, 3.0, 4.0], dtype=float) * np.asarray(v, dtype=float),
        shape=(3, 3),
        metadata={"signature_hint": "diag-callable"},
    )

    def _fail_dense_helper(*_args, **_kwargs):
        raise AssertionError("RtR dense materialization must direct-fill columns")

    monkeypatch.setattr(rtr_module.np, "eye", _fail_dense_helper)
    monkeypatch.setattr(rtr_module.np, "column_stack", _fail_dense_helper)

    np.testing.assert_allclose(prior.as_RtR(dense=True), expected)
    source = inspect.getsource(rtr_module.RtRPrior.as_RtR)
    assert "np.eye" not in source
    assert "np.column_stack" not in source


def test_rtr_prior_signature_hint_distinguishes_semantically_named_explicit_priors() -> (
    None
):
    matrix = sparse.diags([1.0, 2.0, 3.0], 0, format="csr")
    laplace = as_rtr_prior(
        matrix,
        name="laplace",
        metadata={"signature_hint": "laplace"},
    )
    graph_ltl = as_rtr_prior(
        matrix,
        name="curvature",
        metadata={"signature_hint": "graph_ltl"},
    )

    assert laplace.signature_hash != graph_ltl.signature_hash


def test_rtr_prior_signatures_stream_payloads_without_tobytes_copy() -> None:
    dense = np.array([[2.0, 0.5], [0.5, 3.0]], dtype=float)
    dense_metadata = {"signature_hint": "dense-stream"}
    dense_prior = as_rtr_prior(dense, metadata=dense_metadata)
    assert dense_prior.signature_hash == _legacy_rtr_signature(
        dense, metadata=dense_metadata
    )

    sparse_matrix = sparse.csr_matrix(
        np.array(
            [[4.0, 0.0, -1.0], [0.0, 5.0, 0.25], [-1.0, 0.25, 6.0]],
            dtype=float,
        )
    )
    sparse_metadata = {"signature_hint": "sparse-stream"}
    sparse_prior = as_rtr_prior(sparse_matrix, metadata=sparse_metadata)
    assert sparse_prior.signature_hash == _legacy_rtr_signature(
        sparse_matrix, metadata=sparse_metadata
    )

    source = inspect.getsource(rtr_module._signature_for_payload)
    assert "update_digest_with_array_payload" in source
    assert ".tobytes(" not in source
    assert "np.ascontiguousarray" not in source


def test_v488_rtr_prior_guards_use_bounded_finite_scans() -> None:
    apply_source = inspect.getsource(rtr_module.RtRPrior.apply)
    diag_source = inspect.getsource(rtr_module.RtRPrior.diag)
    as_prior_source = inspect.getsource(rtr_module.as_rtr_prior)
    sparse_source = inspect.getsource(rtr_module._validate_sparse_payload)
    vector_source = inspect.getsource(rtr_module._as_vector)

    assert "all_finite_values(result)" in apply_source
    assert "np.isfinite(result).all()" not in apply_source
    assert "all_finite_values(diag)" in diag_source
    assert "np.isfinite(diag).all()" not in diag_source
    assert "np.diag(self._payload)" not in diag_source
    assert "self._payload.diagonal()" in diag_source
    assert "all_finite_values(array)" in as_prior_source
    assert "np.isfinite(array).all()" not in as_prior_source
    assert "all_finite_values(matrix.data)" in sparse_source
    assert "np.isfinite(matrix.data).all()" not in sparse_source
    assert "all_finite_values(vector)" in vector_source
    assert "np.isfinite(vector).all()" not in vector_source


def test_rtr_prior_hdf5_round_trips_explicit_priors(tmp_path) -> None:
    vector = np.array([2.0, -1.0, 0.5], dtype=float)
    sparse_prior = as_rtr_prior(
        sparse.diags([1.0, 3.0, 5.0], 0, format="csr"), name="persist-sparse"
    )
    sparse_path = write_rtr_prior_artifact(tmp_path / "sparse_prior.h5", sparse_prior)
    loaded_sparse = load_rtr_prior_artifact(sparse_path)
    assert loaded_sparse.signature_hash == sparse_prior.signature_hash
    np.testing.assert_allclose(loaded_sparse.apply(vector), sparse_prior.apply(vector))

    dense = np.array(
        [[2.0, 0.25, 0.0], [0.25, 4.0, -0.5], [0.0, -0.5, 3.0]],
        dtype=float,
    )
    dense_prior = as_rtr_prior(dense, name="persist-dense")
    dense_path = write_rtr_prior_artifact(tmp_path / "dense_prior.h5", dense_prior)
    loaded_dense = load_rtr_prior_artifact(dense_path)
    assert loaded_dense.signature_hash == dense_prior.signature_hash
    np.testing.assert_allclose(loaded_dense.as_RtR(dense=True), dense)


def test_rtr_prior_hdf5_rejects_matrix_free_priors(tmp_path) -> None:
    prior = as_rtr_prior(
        lambda v: np.asarray(v, dtype=float),
        shape=(2, 2),
        metadata={"signature_hint": "callable-not-persisted"},
    )
    with pytest.raises(ValueError, match="Only dense/sparse matrix-backed"):
        write_rtr_prior_artifact(tmp_path / "callable_prior.h5", prior)
