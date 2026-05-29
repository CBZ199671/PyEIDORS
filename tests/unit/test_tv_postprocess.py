"""Tests for ROI TV refinement after one-step RM reconstruction."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import pyeidors.inverse.postprocess.tv as tv_module
from pyeidors.inverse import (
    VoxelGrid,
    build_one_step_rm,
    graph_difference_operator,
    graph_laplacian,
    reconstruct_difference,
    refine_tv_pdhg,
    total_variation_norm,
)


def test_graph_difference_operator_matches_laplace_prior() -> None:
    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))

    D = graph_difference_operator(mesh)
    L = graph_laplacian(mesh)

    assert D.shape == (3, 4)
    np.testing.assert_allclose(2.0 * (D.T @ D).toarray(), L.toarray())


def test_refine_tv_pdhg_reduces_roi_tv_and_pins_outside_roi() -> None:
    mesh = VoxelGrid.from_bounds([0.0], [5.0], shape=(5,))
    seed = np.array([10.0, 0.0, 2.0, 0.0, -10.0], dtype=float)
    roi = np.array([False, True, True, True, False])
    D = graph_difference_operator(mesh)

    result = refine_tv_pdhg(
        seed,
        mesh,
        roi_mask=roi,
        tv_weight=0.5,
        max_iterations=80,
        tolerance=5.0e-4,
        return_metadata=True,
    )

    assert result.values.shape == seed.shape
    np.testing.assert_allclose(result.values[~roi], seed[~roi])
    assert total_variation_norm(result.values, D) < total_variation_norm(seed, D)
    assert result.metadata["method"] == "tv-pdhg"
    assert result.metadata["seed_source"] == "one_step_rm"
    assert result.metadata["roi_size"] == 3
    assert result.metadata["iterations"] <= 80
    assert result.metadata["roi_residual_norm_history"]
    assert result.metadata["stopped_reason"] in {
        "roi_residual_tolerance",
        "max_iterations",
    }


def test_refine_tv_pdhg_accepts_one_step_rm_output_seed() -> None:
    jacobian = np.eye(4, dtype=float)
    rm = build_one_step_rm(jacobian, lambda_=0.0, mode="tikhonov")
    seed = reconstruct_difference(
        rm,
        np.array([0.0, 1.0, 0.0, 1.0], dtype=float),
        normalize=False,
    )
    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))

    result = refine_tv_pdhg(
        seed,
        mesh,
        tv_weight=0.25,
        max_iterations=40,
        tolerance=1.0e-3,
        return_metadata=True,
        seed_source="one_step_rm",
    )

    assert result.values.shape == (4,)
    assert result.metadata["seed_source"] == "one_step_rm"
    assert result.metadata["difference_operator_shape"] == (3, 4)
    assert result.metadata["roi_residual_norm_history"][-1] >= 0.0


def test_v486_tv_seed_and_difference_guards_use_bounded_finite_scan() -> None:
    norm_source = inspect.getsource(tv_module.total_variation_norm)
    seed_source = inspect.getsource(tv_module._as_seed)

    assert "all_finite_values(diff)" in norm_source
    assert "np.isfinite(diff).all()" not in norm_source
    assert "all_finite_values(vector)" in seed_source
    assert "np.isfinite(vector).all()" not in seed_source


def test_v552_tv_roi_updates_avoid_subset_copies() -> None:
    source = inspect.getsource(tv_module.refine_tv_pdhg)
    helper_source = inspect.getsource(tv_module._roi_relative_residual)

    assert "x_new[roi]" not in source
    assert "previous[roi]" not in source
    assert "y[not_roi]" not in source
    assert "np.copyto(x_new, update, where=roi)" in source
    assert "_roi_relative_residual(x_new, previous, roi)" in source
    assert "np.subtract(current_arr[start:stop]" in helper_source
    assert "where=active" in helper_source

    current = np.array([1.0, 3.0, 7.0], dtype=float)
    previous = np.array([1.0, 1.0, 3.0], dtype=float)
    roi = np.array([False, True, True])
    expected = np.linalg.norm(current[roi] - previous[roi]) / max(
        float(np.linalg.norm(previous[roi])), 1.0
    )
    assert tv_module._roi_relative_residual(current, previous, roi, chunk_size=1) == (
        pytest.approx(expected)
    )


def test_v554_tv_pdhg_preserves_float32_seed_work_buffers() -> None:
    source = inspect.getsource(tv_module.refine_tv_pdhg)
    assert "dtype=np.float64" not in source
    assert "astype(y.dtype, copy=False)" in source
    assert "np.zeros(D.shape[0], dtype=y.dtype)" in source

    mesh = VoxelGrid.from_bounds([0.0], [4.0], shape=(4,))
    seed = np.array([0.0, 2.0, 0.0, 2.0], dtype=np.float32)

    result = refine_tv_pdhg(
        seed,
        mesh,
        tv_weight=0.1,
        max_iterations=5,
        tolerance=1.0e-3,
        return_metadata=True,
    )

    assert result.values.dtype == np.dtype(np.float32)
    assert result.values.shape == seed.shape
