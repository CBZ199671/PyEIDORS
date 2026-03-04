"""Regression tests for warning-free numeric helpers."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyeidors.utils.numeric_ops import safe_dot


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
