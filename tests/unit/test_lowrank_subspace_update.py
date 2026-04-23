"""Tests for low-rank Jacobian subspace extraction."""

from __future__ import annotations

import numpy as np

from pyeidors.inverse.reduced.lowrank_subspace import build_lowrank_subspace


def test_tsvd_subspace_shape_and_orthogonality():
    rng = np.random.default_rng(2)
    j_mat = rng.standard_normal((40, 120))
    basis, s_val = build_lowrank_subspace(j_mat, rank=12, energy=0.99, method="tsvd")
    assert basis.shape[0] == 120
    assert 1 <= basis.shape[1] <= 12
    assert s_val.shape[0] == basis.shape[1]
    gram = basis.T @ basis
    assert np.allclose(gram, np.eye(basis.shape[1]), atol=1e-7)


def test_randomized_subspace_shape():
    rng = np.random.default_rng(3)
    j_mat = rng.standard_normal((32, 96))
    basis, s_val = build_lowrank_subspace(
        j_mat, rank=10, energy=0.98, method="randomized"
    )
    assert basis.shape[0] == 96
    assert 1 <= basis.shape[1] <= 10
    assert s_val.shape[0] == basis.shape[1]
