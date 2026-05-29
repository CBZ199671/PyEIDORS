"""Branch-focused tests for reduced-order helper modules."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from pyeidors.inverse.reduced import lowrank_subspace as lowrank_module
from pyeidors.inverse.reduced import pod_basis as pod_module
import pyeidors.inverse.reduced.snapshot_bank as snapshot_module
from pyeidors.inverse.reduced.pod_basis import (
    compute_pod_basis,
    merge_orthonormal_bases,
)
from pyeidors.inverse.reduced.snapshot_bank import (
    SnapshotBank,
    _as_matrix,
    _stack_columns_direct,
    select_snapshot_matrix,
)


def test_pod_snapshot_normalization_and_rank_helpers_cover_edge_cases():
    np.testing.assert_allclose(
        pod_module._normalize_snapshot_matrix(np.array([1.0, 2.0], dtype=float)),
        np.array([[1.0], [2.0]], dtype=float),
    )
    assert pod_module._normalize_snapshot_matrix(
        np.zeros((3, 0), dtype=float)
    ).shape == (3, 0)
    with pytest.raises(ValueError, match="snapshots must be a 2D array"):
        pod_module._normalize_snapshot_matrix(np.zeros((2, 2, 2), dtype=float))

    assert pod_module._rank_from_energy(np.zeros(0, dtype=float), 0.9, 4) == 0
    assert pod_module._rank_from_energy(np.array([0.0, 0.0], dtype=float), 0.9, 4) == 0
    assert (
        pod_module._rank_from_energy(np.array([3.0, 2.0, 1.0], dtype=float), 0.5, 3)
        == 1
    )
    assert (
        pod_module._rank_from_energy(np.array([3.0, 2.0, 1.0], dtype=float), 1.0, 2)
        == 2
    )


def test_compute_pod_basis_handles_empty_zero_singular_values_and_energy_selection(
    monkeypatch: pytest.MonkeyPatch,
):
    assert compute_pod_basis(np.zeros((4, 0), dtype=float)).shape == (4, 0)

    monkeypatch.setattr(
        pod_module.np.linalg,
        "svd",
        lambda _mat, full_matrices=False: (
            np.zeros((3, 0), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros((0, 2), dtype=float),
        ),
    )
    assert compute_pod_basis(np.ones((3, 2), dtype=float)).shape == (3, 0)

    monkeypatch.setattr(
        pod_module.np.linalg,
        "svd",
        lambda mat, full_matrices=False: (
            np.eye(mat.shape[0], mat.shape[1], dtype=float),
            np.array([1e-16, 1e-18], dtype=float),
            np.eye(mat.shape[1], dtype=float),
        ),
    )
    assert compute_pod_basis(np.ones((3, 2), dtype=float), eps=1e-12).shape == (3, 0)

    monkeypatch.undo()
    rng = np.random.default_rng(0)
    snapshots = rng.standard_normal((5, 4))
    basis = compute_pod_basis(snapshots, rank=None, energy=0.6)
    assert basis.shape[0] == 5
    assert 1 <= basis.shape[1] <= 4


def test_merge_orthonormal_bases_covers_invalid_inputs_and_rank_cap():
    empty = merge_orthonormal_bases(None, np.zeros((0,), dtype=float))
    assert empty.shape == (0, 0)

    mixed = merge_orthonormal_bases(
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([[1.0], [0.0]], dtype=float),
    )
    assert mixed.shape == (3, 1)

    collapsed = merge_orthonormal_bases(
        np.array([[1.0], [0.0]], dtype=float),
        np.array([[1.0], [0.0]], dtype=float),
        eps=1e-9,
    )
    assert collapsed.shape == (2, 1)

    capped = merge_orthonormal_bases(np.eye(4), np.eye(4), rank_cap=2)
    assert capped.shape == (4, 2)


def test_v278_merge_orthonormal_bases_direct_fills_blocks(monkeypatch) -> None:
    block_a = np.eye(3, 2, dtype=np.float32)
    block_b = np.array([[0.0], [1.0], [1.0]], dtype=np.float64)
    expected = np.empty((3, 3), dtype=np.float64)
    expected[:, :2] = block_a
    expected[:, 2] = block_b[:, 0]

    direct = pod_module._stack_basis_blocks([block_a, block_b], n_param=3)
    np.testing.assert_allclose(direct, expected)

    def _fail_column_stack(*_args, **_kwargs):
        raise AssertionError("POD basis merge must direct-fill blocks")

    monkeypatch.setattr(pod_module.np, "column_stack", _fail_column_stack)

    merged = merge_orthonormal_bases(block_a, block_b, rank_cap=2)
    assert merged.shape == (3, 2)
    np.testing.assert_allclose(merged.T @ merged, np.eye(2), atol=1e-12)
    assert "np.column_stack" not in inspect.getsource(pod_module._stack_basis_blocks)
    assert "np.column_stack" not in inspect.getsource(
        pod_module.merge_orthonormal_bases
    )


def test_snapshot_bank_matrix_hash_and_selection_cover_edge_cases():
    bank = SnapshotBank(max_snapshots=2, normalize=True)
    bank.add(np.array([], dtype=float))
    bank.add(np.array([1.0, np.nan], dtype=float))
    assert bank.matrix().shape == (0, 0)
    assert bank.snapshot_hash() == "empty"

    bank.add(np.array([3.0, 4.0], dtype=float))
    bank.add(np.array([6.0, 8.0], dtype=float))
    bank.add(np.array([1.0, 0.0, 0.0], dtype=float))
    mat = bank.matrix()
    assert mat.shape == (3, 1)
    assert bank.snapshot_hash() != "empty"

    no_norm = SnapshotBank(max_snapshots=3, normalize=False)
    no_norm.add(np.array([2.0, 0.0], dtype=float))
    np.testing.assert_allclose(
        no_norm.matrix()[:, 0], np.array([2.0, 0.0], dtype=float)
    )

    assert _as_matrix(None, n_param=2).shape == (2, 0)
    assert _as_matrix(np.array([1.0, 2.0], dtype=float), n_param=2).shape == (2, 1)
    assert _as_matrix(np.zeros((2, 2, 1), dtype=float), n_param=2).shape == (2, 0)
    assert _as_matrix(np.ones((3, 1), dtype=float), n_param=2).shape == (2, 0)
    assert _as_matrix(np.zeros((2, 0), dtype=float), n_param=2).shape == (2, 0)

    bank_matrix = np.column_stack([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    synthetic = np.column_stack([np.array([1.0, 0.0]), np.array([1.0, 1.0])])
    cached = np.column_stack([np.array([0.0, 1.0]), np.array([2.0, 2.0])])

    selected_cache = select_snapshot_matrix(
        "cache",
        n_param=2,
        bank_matrix=bank_matrix,
        synthetic_matrix=synthetic,
        cached_matrix=cached,
    )
    assert selected_cache.shape == (2, 3)

    selected_synth = select_snapshot_matrix(
        "synthetic",
        n_param=2,
        bank_matrix=bank_matrix,
        synthetic_matrix=synthetic,
        cached_matrix=cached,
    )
    assert selected_synth.shape == (2, 3)

    selected_default = select_snapshot_matrix(
        "unknown",
        n_param=2,
        bank_matrix=None,
        synthetic_matrix=np.column_stack([np.array([1.0, 1.0]), np.array([1.0, 1.0])]),
        cached_matrix=None,
    )
    assert selected_default.shape == (2, 1)

    empty = select_snapshot_matrix(
        "cache",
        n_param=3,
        bank_matrix=np.ones((2, 1), dtype=float),
        synthetic_matrix=None,
        cached_matrix=None,
    )
    assert empty.shape == (3, 0)


def test_v277_snapshot_bank_direct_fills_column_matrices(monkeypatch) -> None:
    columns = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float64),
    ]
    expected = np.empty((2, 2), dtype=np.float64)
    expected[:, 0] = columns[0]
    expected[:, 1] = columns[1]

    def _fail_column_stack(*_args, **_kwargs):
        raise AssertionError("snapshot bank must not use np.column_stack")

    monkeypatch.setattr(np, "column_stack", _fail_column_stack)

    direct = _stack_columns_direct(columns)
    np.testing.assert_allclose(direct, expected)

    bank = SnapshotBank(max_snapshots=4, normalize=False)
    bank.add(columns[0])
    bank.add(columns[1])
    np.testing.assert_allclose(bank.matrix(), expected)

    selected = select_snapshot_matrix(
        "hybrid",
        n_param=2,
        bank_matrix=expected,
        synthetic_matrix=expected[:, :1],
        cached_matrix=expected[:, 1:],
    )
    np.testing.assert_allclose(selected, expected[:, ::-1])
    assert "np.column_stack" not in inspect.getsource(_stack_columns_direct)
    assert "np.column_stack" not in inspect.getsource(
        snapshot_module.select_snapshot_matrix
    )


def test_v491_snapshot_bank_add_uses_bounded_finite_scan() -> None:
    source = inspect.getsource(snapshot_module.SnapshotBank.add)

    assert "all_finite_values(vec)" in source
    assert "np.isfinite(vec).all()" not in source


def test_v595_unique_snapshot_blocks_hashes_column_views_without_local_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = np.arange(24, dtype=np.float64).reshape(8, 3)
    expected = block.copy()
    original_hash = snapshot_module.hash_array_payload
    captured: list[np.ndarray] = []

    def _capture_hash(arr: np.ndarray) -> str:
        captured.append(arr)
        return original_hash(arr)

    monkeypatch.setattr(snapshot_module, "hash_array_payload", _capture_hash)

    selected = snapshot_module._unique_snapshot_blocks([block], n_param=8)

    np.testing.assert_allclose(selected, expected)
    assert len(captured) == 3
    assert all(arr.dtype == np.float64 for arr in captured)
    assert all(not arr.flags.c_contiguous for arr in captured)
    assert "np.ascontiguousarray(block[:, col_idx]" not in inspect.getsource(
        snapshot_module._unique_snapshot_blocks
    )


def test_lowrank_subspace_helpers_cover_invalid_empty_and_randomized_paths():
    assert lowrank_module._rank_from_energy(np.zeros(0, dtype=float), 0.9, 4) == 0
    assert (
        lowrank_module._rank_from_energy(np.array([0.0, 0.0], dtype=float), 0.9, 4) == 0
    )

    singular_values, vt_mat = lowrank_module._randomized_right_svd(np.eye(4), 2)
    assert singular_values.shape[0] >= 2
    assert vt_mat.shape[1] == 4

    with pytest.raises(ValueError, match="jacobian must be 2D"):
        lowrank_module.build_lowrank_subspace(np.array([1.0, 2.0], dtype=float))

    empty_basis, empty_sv = lowrank_module.build_lowrank_subspace(
        np.zeros((3, 0), dtype=float)
    )
    assert empty_basis.shape == (0, 0)
    assert empty_sv.shape == (0,)

    basis_rand, sv_rand = lowrank_module.build_lowrank_subspace(
        np.arange(24, dtype=float).reshape(6, 4),
        rank=2,
        energy=0.9,
        method="randomized",
    )
    assert basis_rand.shape[0] == 4
    assert 1 <= basis_rand.shape[1] <= 2
    assert sv_rand.shape[0] == basis_rand.shape[1]


def test_reduced_helper_remaining_rank_and_single_column_edges(
    monkeypatch: pytest.MonkeyPatch,
):
    assert (
        lowrank_module._rank_from_energy(np.array([3.0, 2.0, 1.0], dtype=float), 3.0, 2)
        == 2
    )

    monkeypatch.setattr(
        lowrank_module,
        "_randomized_right_svd",
        lambda _jacobian, _rank: (
            np.array([1.0], dtype=float),
            np.zeros((0, 3), dtype=float),
        ),
    )
    empty_basis, empty_sv = lowrank_module.build_lowrank_subspace(
        np.ones((2, 3), dtype=float),
        rank=2,
        method="randomized",
    )
    assert empty_basis.shape == (3, 0)
    assert empty_sv.shape == (0,)

    assert (
        pod_module._rank_from_energy(np.array([3.0, 2.0, 1.0], dtype=float), 5.0, 2)
        == 2
    )

    monkeypatch.setattr(pod_module, "_rank_from_energy", lambda *_args, **_kwargs: 0)
    zero_rank_basis = compute_pod_basis(np.eye(2, dtype=float), rank=None, energy=0.8)
    assert zero_rank_basis.shape == (2, 0)

    merge_source = inspect.getsource(merge_orthonormal_bases)
    assert "np.diag(r_mat)" not in merge_source
    assert "r_mat.diagonal()" in merge_source

    all_zero_qr = merge_orthonormal_bases(np.zeros((2, 2), dtype=float))
    assert all_zero_qr.shape == (2, 0)

    single_snapshot = select_snapshot_matrix(
        "synthetic",
        n_param=2,
        bank_matrix=None,
        synthetic_matrix=np.array([1.0, 2.0], dtype=float),
        cached_matrix=None,
    )
    assert single_snapshot.shape == (2, 1)
