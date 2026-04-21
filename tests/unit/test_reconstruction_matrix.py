"""Tests for online reconstruction-matrix helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from pyeidors.data.difference import normalize_time_difference
from pyeidors.inverse.reconstruction_matrix import reconstruct_difference


def test_reconstruct_difference_applies_rm_to_normalized_time_difference() -> None:
    rm = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 4.0]], dtype=float)
    reference = np.array([2.0, 4.0, -2.0], dtype=float)
    target = np.array([3.0, 8.0, -1.0], dtype=float)

    expected_dv = normalize_time_difference(target, reference)
    expected = rm @ expected_dv

    np.testing.assert_allclose(
        reconstruct_difference(rm, target, normalize=True, v_ref=reference),
        expected,
    )


def test_reconstruct_difference_accepts_preprojected_dv_and_sparse_rm() -> None:
    rm = sparse.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]], dtype=float))
    dv = np.array([0.25, -0.5, 2.0], dtype=float)

    np.testing.assert_allclose(
        reconstruct_difference(rm, dv, normalize=False),
        np.array([4.25, 2.5], dtype=float),
    )


def test_reconstruct_difference_validates_shapes_and_finite_values() -> None:
    rm = np.ones((2, 3), dtype=float)
    with pytest.raises(ValueError, match="RM column count"):
        reconstruct_difference(rm, np.ones(2), normalize=False)
    with pytest.raises(FloatingPointError, match="dv contains non-finite"):
        reconstruct_difference(rm, np.array([1.0, np.nan, 2.0]), normalize=False)
    with pytest.raises(ValueError, match="rm must be a 2D"):
        reconstruct_difference(np.ones(3), np.ones(3), normalize=False)
