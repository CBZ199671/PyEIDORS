"""Unit tests for POD basis builders."""

from __future__ import annotations

import numpy as np

from pyeidors.inverse.reduced.pod_basis import compute_pod_basis, merge_orthonormal_bases
from pyeidors.inverse.reduced.snapshot_bank import SnapshotBank, select_snapshot_matrix


def test_compute_pod_basis_returns_orthonormal_columns():
    rng = np.random.default_rng(0)
    snapshots = rng.standard_normal((64, 12))
    basis = compute_pod_basis(snapshots, rank=8, energy=0.99)
    assert basis.shape[0] == 64
    assert 1 <= basis.shape[1] <= 8
    with np.errstate(all="ignore"):
        gram = basis.T @ basis
    assert np.allclose(gram, np.eye(basis.shape[1]), atol=1e-7)


def test_merge_orthonormal_bases_applies_rank_cap():
    rng = np.random.default_rng(1)
    a = np.linalg.qr(rng.standard_normal((48, 8)))[0]
    b = np.linalg.qr(rng.standard_normal((48, 6)))[0]
    merged = merge_orthonormal_bases(a, b, rank_cap=10)
    assert merged.shape == (48, 10)
    with np.errstate(all="ignore"):
        gram = merged.T @ merged
    assert np.allclose(gram, np.eye(10), atol=1e-7)


def test_snapshot_bank_and_selection_policy():
    bank = SnapshotBank(max_snapshots=4)
    for idx in range(6):
        bank.add(np.array([1.0 + idx, 2.0, 3.0], dtype=float))
    mat = bank.matrix()
    assert mat.shape == (3, 4)

    synthetic = np.column_stack(
        [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        ]
    )
    selected = select_snapshot_matrix(
        "hybrid",
        n_param=3,
        bank_matrix=mat,
        synthetic_matrix=synthetic,
        cached_matrix=None,
    )
    assert selected.shape[0] == 3
    assert selected.shape[1] >= 2
