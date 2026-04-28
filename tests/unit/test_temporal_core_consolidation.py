"""T86: shared temporal-array validation helper contracts."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.data import temporal_filtering
from pyeidors.data._temporal_core import (
    as_frame_batch,
    positive_int,
    unit_interval,
)
from pyeidors.inverse.postprocess import temporal as temporal_postprocess
from pyeidors.inverse.postprocess import tv as tv_postprocess


def test_temporal_modules_reuse_core_private_aliases() -> None:
    assert temporal_filtering._as_frame_batch is as_frame_batch
    assert temporal_filtering._positive_int is positive_int
    assert temporal_filtering._unit_interval is unit_interval

    assert temporal_postprocess._as_frame_batch is as_frame_batch
    assert temporal_postprocess._positive_int is positive_int
    assert temporal_postprocess._unit_interval is unit_interval

    assert tv_postprocess._positive_int is positive_int


def test_as_frame_batch_preserves_vector_and_batch_contracts() -> None:
    vector_batch, was_vector = as_frame_batch(np.array([1.0, 2.0], dtype=float))
    assert was_vector is True
    assert vector_batch.flags.c_contiguous
    np.testing.assert_allclose(vector_batch, np.array([[1.0, 2.0]], dtype=float))

    matrix_batch, was_vector = as_frame_batch(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float).T
    )
    assert was_vector is False
    assert matrix_batch.flags.c_contiguous
    np.testing.assert_allclose(matrix_batch, np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_temporal_core_preserves_validation_error_messages() -> None:
    with pytest.raises(ValueError, match="1D vector or 2D frame batch"):
        as_frame_batch(np.zeros((1, 1, 1), dtype=float))
    with pytest.raises(ValueError, match="non-empty"):
        as_frame_batch(np.empty((0, 2), dtype=float))
    with pytest.raises(FloatingPointError, match="non-finite"):
        as_frame_batch(np.array([1.0, np.nan], dtype=float))

    assert positive_int(3, "window") == 3
    with pytest.raises(ValueError, match="window must be positive"):
        positive_int(0, "window")

    assert unit_interval(0.5, "alpha") == 0.5
    with pytest.raises(ValueError, match=r"alpha must be finite and in \[0, 1\]"):
        unit_interval(1.5, "alpha")
