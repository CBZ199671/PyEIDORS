"""Regression tests for warning-free numeric helpers."""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

from pyeidors.utils.numeric_ops import (
    _finite_summary,
    all_finite_values,
    any_abs_less_equal_values,
    any_equal_values,
    any_not_equal_values,
    has_nonzero_imaginary,
    min_alpha_for_value_floor,
    safe_dot,
    squared_distances_to_point,
)


def test_safe_dot_matrix_vector_has_no_runtime_warning() -> None:
    rng = np.random.default_rng(1234)
    matrix = rng.normal(scale=0.2, size=(10, 1089))
    matrix += np.eye(10, 1089) * 0.8
    vector = np.linspace(0.9, 1.1, 1089)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = safe_dot(matrix, vector, "test_numeric_ops.mv")

    assert result.shape == (10,)
    assert np.isfinite(result).all()
    assert np.allclose(result, np.dot(matrix, vector))


def test_safe_dot_matrix_matrix_has_no_runtime_warning() -> None:
    rng = np.random.default_rng(42)
    lhs = rng.normal(size=(120, 80))
    rhs = rng.normal(size=(80, 64))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = safe_dot(lhs, rhs, "test_numeric_ops.mm")

    assert result.shape == (120, 64)
    assert np.isfinite(result).all()
    assert np.allclose(result, np.dot(lhs, rhs))


def test_safe_dot_rejects_non_finite_input() -> None:
    lhs = np.array([1.0, np.inf], dtype=float)
    rhs = np.array([2.0, 3.0], dtype=float)
    with pytest.raises(FloatingPointError):
        safe_dot(lhs, rhs, "test_numeric_ops.non_finite")


def test_v413_numeric_finite_summary_scans_without_subset_copy() -> None:
    summary = _finite_summary(
        np.array([1.0 + 1.0j, np.nan + 0.0j, 3.0 + 4.0j], dtype=np.complex128)
    )

    assert "total=3" in summary
    assert "finite=2" in summary
    assert "max=5.000000e+00" in summary
    source = inspect.getsource(_finite_summary)
    assert "[np.isfinite" not in source
    assert "np.abs(finite)" not in source


def test_v475_nonzero_imaginary_scan_uses_bounded_work_buffers() -> None:
    values = np.zeros(10, dtype=np.complex64)
    values[8] = 1.0 + 0.25j

    assert has_nonzero_imaginary(values, tol=1.0e-3, chunk_size=3) is True
    assert has_nonzero_imaginary(values.real, tol=1.0e-3, chunk_size=3) is False
    assert has_nonzero_imaginary(values[:7], tol=1.0e-3, chunk_size=3) is False

    source = inspect.getsource(has_nonzero_imaginary)
    assert "np.abs(np.imag" not in source
    assert "np.any(np.abs" not in source
    assert "np.abs(chunk, out=abs_chunk)" in source
    assert "np.greater(abs_chunk, threshold, out=mask_chunk)" in source


def test_v476_all_finite_values_uses_bounded_work_buffer() -> None:
    assert all_finite_values(np.array([1.0, 2.0]), chunk_size=1) is True
    assert all_finite_values(np.array([1.0 + 0.0j, 2.0 + 0.5j]), chunk_size=1)
    assert not all_finite_values(np.array([1.0, np.inf]), chunk_size=1)

    source = inspect.getsource(all_finite_values)
    safe_dot_source = inspect.getsource(safe_dot)
    assert "np.isfinite(array).all()" not in source
    assert "np.isfinite(result_array).all()" not in safe_dot_source
    assert "np.isfinite(chunk, out=chunk_mask)" in source


def test_v548_comparison_scans_use_bounded_work_buffers() -> None:
    assert any_equal_values(np.array([1.0, 0.0, 2.0]), 0.0, chunk_size=1)
    assert not any_equal_values(np.array([1.0, 2.0]), 0.0, chunk_size=1)
    assert any_not_equal_values(
        np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0]), chunk_size=2
    )
    assert not any_not_equal_values(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), chunk_size=1
    )
    assert any_abs_less_equal_values(
        np.array([3.0 + 4.0j, 1.0e-12 + 0.0j]), 1.0e-9, chunk_size=1
    )
    assert not any_abs_less_equal_values(np.array([2.0, -3.0]), 1.0, chunk_size=1)

    for helper in (any_equal_values, any_not_equal_values, any_abs_less_equal_values):
        source = inspect.getsource(helper)
        assert "np.any(" not in source
        assert "out=" in source


def test_v550_floor_step_limit_scans_without_boolean_subsets() -> None:
    assert np.isinf(
        min_alpha_for_value_floor(
            np.array([1.0, 2.0]), np.array([0.1, 0.0]), 0.2, chunk_size=1
        )
    )
    assert min_alpha_for_value_floor(
        np.array([1.0, 2.0]), np.array([-0.5, -1.0]), 0.2, chunk_size=1
    ) == pytest.approx(0.8 / 0.5)
    assert (
        min_alpha_for_value_floor(
            np.array([0.2, 2.0]), np.array([-0.5, 0.0]), 0.2, chunk_size=1
        )
        == 0.0
    )

    source = inspect.getsource(min_alpha_for_value_floor)
    assert "[negative" not in source
    assert "values[mask]" not in source
    assert "np.divide(margin, ratios, out=ratios, where=negative)" in source


def test_squared_distances_to_point_reuses_one_work_vector() -> None:
    source = inspect.getsource(squared_distances_to_point)

    assert "center[None" not in source
    assert "points[:, :ndim] -" not in source
    assert "np.subtract" in source
    assert "out=work" in source
    points = np.array([[0.0, 0.0, 10.0], [3.0, 4.0, 20.0]], dtype=float)

    np.testing.assert_allclose(
        squared_distances_to_point(points, [0.0, 0.0], ndim=2),
        np.array([0.0, 25.0], dtype=float),
    )
    np.testing.assert_allclose(
        squared_distances_to_point(points[:, :2], [1.0], ndim=2),
        np.array([1.0, 20.0], dtype=float),
    )
