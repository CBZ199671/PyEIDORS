"""Tests for sparse solver modules to achieve 100% coverage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List
from unittest import mock

import numpy as np
import pytest

from pyeidors.utils.numeric_ops import safe_dot


# --- sparse_projection tests ---


class TestSparseProjection:
    """Cover lines in sparse_projection.py."""

    def test_resolve_coarse_sizes_from_levels(self):
        from pyeidors.inverse.solvers.sparse_projection import _resolve_coarse_sizes

        @dataclass
        class FakeConfig:
            coarse_levels: list = field(default_factory=lambda: [4, 2, 8])
            coarse_group_size: int = 0

        sizes = _resolve_coarse_sizes(FakeConfig())
        assert sorted(sizes) == [2, 4, 8]

    def test_resolve_coarse_sizes_from_group_size(self):
        from pyeidors.inverse.solvers.sparse_projection import _resolve_coarse_sizes

        @dataclass
        class FakeConfig:
            coarse_levels: list = field(default_factory=list)
            coarse_group_size: int = 5

        sizes = _resolve_coarse_sizes(FakeConfig())
        assert sizes == [5]

    def test_sum_group_columns(self):
        from pyeidors.inverse.solvers.sparse_projection import _sum_group_columns

        jac = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        groups = [np.array([0, 1]), np.array([2, 3])]
        result = _sum_group_columns(jac, groups)
        np.testing.assert_array_equal(result[0], [3, 11])
        np.testing.assert_array_equal(result[1], [7, 15])

    def test_init_power_vector_near_zero(self):
        from pyeidors.inverse.solvers.sparse_projection import _init_power_vector

        rng = mock.MagicMock()
        rng.standard_normal.return_value = np.zeros(3)
        vec = _init_power_vector(np.eye(3), rng)
        assert vec.shape == (3,)

    def test_estimate_lipschitz_empty(self):
        from pyeidors.inverse.solvers.sparse_projection import (
            estimate_lipschitz_constant,
        )

        result = estimate_lipschitz_constant(np.empty((0, 0)))
        assert result == pytest.approx(1e-12)

    def test_estimate_lipschitz_normal(self):
        from pyeidors.inverse.solvers.sparse_projection import (
            estimate_lipschitz_constant,
        )

        A = np.array([[1, 0], [0, 2]], dtype=float)
        L = estimate_lipschitz_constant(A, iters=5)
        assert L > 0

    def test_estimate_lipschitz_zero_norm(self):
        from pyeidors.inverse.solvers.sparse_projection import (
            estimate_lipschitz_constant,
        )

        A = np.zeros((3, 3))
        result = estimate_lipschitz_constant(A, iters=2)
        assert result == pytest.approx(1e-12)

    def test_build_coarse_hierarchy(self):
        from pyeidors.inverse.solvers.sparse_projection import build_coarse_hierarchy

        @dataclass
        class FakeConfig:
            coarse_levels: list = field(default_factory=lambda: [3])
            coarse_group_size: int = 0

        hierarchy = build_coarse_hierarchy(FakeConfig(), n_elements=9, cache={})
        assert len(hierarchy) == 1
        assert hierarchy[0][0] == 3

    def test_build_coarse_hierarchy_skips_large_size(self):
        from pyeidors.inverse.solvers.sparse_projection import build_coarse_hierarchy

        @dataclass
        class FakeConfig:
            coarse_levels: list = field(default_factory=lambda: [100])
            coarse_group_size: int = 0

        hierarchy = build_coarse_hierarchy(FakeConfig(), n_elements=10, cache={})
        assert len(hierarchy) == 0


# --- reduced module tests ---


class TestReducedGNStep:
    """Cover lines in reduced_gn_step.py."""

    def test_build_reduced_operator(self):
        from pyeidors.inverse.reduced.reduced_gn_step import build_reduced_operator

        J = np.random.randn(5, 10)
        U = np.random.randn(10, 3)
        R_apply = lambda x: 0.01 * x
        result = build_reduced_operator(
            jacobian=J, basis=U, regularization_apply=R_apply, lambda_eff=0.01
        )
        assert "H" in result
        assert "JU" in result
        assert result["H"].shape == (3, 3)

    def test_solve_reduced_step(self):
        from pyeidors.inverse.reduced.reduced_gn_step import (
            build_reduced_operator,
            solve_reduced_step,
        )

        J = np.random.randn(5, 10)
        U = np.random.randn(10, 3)
        R_apply = lambda x: 0.01 * x
        op_data = build_reduced_operator(
            jacobian=J, basis=U, regularization_apply=R_apply, lambda_eff=0.01
        )
        # rhs must be in full space (n_elements=10)
        rhs = np.random.randn(10)
        delta, info = solve_reduced_step(reduced_operator=op_data, rhs=rhs)
        assert delta.shape == (10,)

    def test_solve_reduced_step_cg(self):
        from pyeidors.inverse.reduced.reduced_gn_step import (
            build_reduced_operator,
            solve_reduced_step,
        )

        J = np.random.randn(5, 10)
        U = np.random.randn(10, 3)
        R_apply = lambda x: 0.01 * x
        op_data = build_reduced_operator(
            jacobian=J, basis=U, regularization_apply=R_apply, lambda_eff=0.01
        )
        rhs = np.random.randn(10)
        delta, info = solve_reduced_step(
            reduced_operator=op_data, rhs=rhs, inexact_tol=0.1, maxiter=50
        )
        assert delta.shape == (10,)


class TestInexactController:
    """Cover line 22 in inexact_controller.py."""

    def test_init_normalizes_mode(self):
        from pyeidors.inverse.reduced.inexact_controller import InexactController

        ctrl = InexactController(mode="FIXED", eta0=0.3)
        assert ctrl.mode == "fixed"


class TestLowrankSubspace:
    """Cover lines 20, 65 in lowrank_subspace.py."""

    def test_build_lowrank_subspace_tsvd(self):
        from pyeidors.inverse.reduced.lowrank_subspace import build_lowrank_subspace

        J = np.random.randn(20, 50)
        basis, svals = build_lowrank_subspace(J, energy=0.9, method="tsvd")
        assert basis.shape[0] == 50

    def test_build_lowrank_subspace_randomized(self):
        from pyeidors.inverse.reduced.lowrank_subspace import build_lowrank_subspace

        J = np.random.randn(20, 50)
        basis, svals = build_lowrank_subspace(J, energy=0.9, method="randomized")
        assert basis.shape[0] == 50


class TestPodBasis:
    """Cover lines 31, 63, 92, 103 in pod_basis.py."""

    def test_compute_pod_basis(self):
        from pyeidors.inverse.reduced.pod_basis import compute_pod_basis

        snapshots = np.random.randn(10, 5)
        basis = compute_pod_basis(snapshots, energy=0.95)
        assert basis.shape[0] == 10
        assert basis.shape[1] <= 5

    def test_compute_pod_basis_explicit_rank(self):
        from pyeidors.inverse.reduced.pod_basis import compute_pod_basis

        snapshots = np.random.randn(10, 5)
        basis = compute_pod_basis(snapshots, rank=2)
        assert basis.shape == (10, 2)

    def test_merge_orthonormal_bases(self):
        from pyeidors.inverse.reduced.pod_basis import merge_orthonormal_bases

        b1 = np.eye(5, 2)
        b2 = np.eye(5, 3)
        merged = merge_orthonormal_bases(b1, b2)
        assert merged.shape[0] == 5
        assert merged.shape[1] >= 1

    def test_merge_orthonormal_bases_with_none(self):
        from pyeidors.inverse.reduced.pod_basis import merge_orthonormal_bases

        b1 = np.eye(5, 2)
        merged = merge_orthonormal_bases(b1, None)
        assert merged.shape[0] == 5


class TestSnapshotBank:
    """Cover lines 37, 104, 116 in snapshot_bank.py."""

    def test_empty_bank_matrix(self):
        from pyeidors.inverse.reduced.snapshot_bank import SnapshotBank

        bank = SnapshotBank()
        mat = bank.matrix()
        assert mat.size == 0

    def test_bank_with_snapshots(self):
        from pyeidors.inverse.reduced.snapshot_bank import SnapshotBank

        bank = SnapshotBank()
        bank.add(np.ones(5))
        bank.add(np.zeros(5))
        mat = bank.matrix()
        assert mat.shape[1] == 2
