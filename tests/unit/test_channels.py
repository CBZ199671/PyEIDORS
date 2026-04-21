"""Tests for measurement-channel masks and weighting contracts."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.data.channels import (
    apply_measurement_contract_to_jacobian,
    apply_measurement_contract_to_vector,
    bad_channel_mask,
    normalize_bad_channel_mask,
    prepare_measurement_contract,
    zero_bad_channel_rows,
    zero_bad_channel_vector,
    zero_bad_channel_weights,
)


def test_bad_channel_mask_accepts_indices_and_bool_mask() -> None:
    np.testing.assert_array_equal(
        bad_channel_mask(5, [1, 3]),
        np.array([False, True, False, True, False]),
    )
    np.testing.assert_array_equal(
        normalize_bad_channel_mask([False, True, False], n_measurements=3),
        np.array([False, True, False]),
    )

    with pytest.raises(ValueError, match="out of range"):
        bad_channel_mask(3, [3])
    with pytest.raises(ValueError, match="length"):
        normalize_bad_channel_mask([True, False], n_measurements=3)


def test_zero_bad_channel_helpers_zero_rows_vector_and_weights() -> None:
    mask = np.array([False, True, False], dtype=bool)
    jacobian = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    residual = np.array([10.0, 20.0, 30.0])
    weights = np.array([[2.0, 0.5, 0.25], [0.5, 3.0, 0.1], [0.25, 0.1, 4.0]])

    np.testing.assert_allclose(
        zero_bad_channel_rows(jacobian, mask),
        np.array([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]),
    )
    np.testing.assert_allclose(
        zero_bad_channel_vector(residual, mask),
        np.array([10.0, 0.0, 30.0]),
    )
    masked_weights, kind = zero_bad_channel_weights(weights, mask, n_measurements=3)
    assert kind == "full"
    np.testing.assert_allclose(
        masked_weights,
        np.array([[2.0, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 4.0]]),
    )


def test_measurement_contract_applies_diagonal_sqrt_weights() -> None:
    mask = np.array([False, True, False], dtype=bool)
    weights = np.array([4.0, 9.0, 1.0])
    jacobian = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    residual = np.array([10.0, 20.0, 30.0])

    weighted_jac, jac_contract = apply_measurement_contract_to_jacobian(
        jacobian,
        channel_mask=mask,
        measurement_weights=weights,
    )
    weighted_residual, residual_contract = apply_measurement_contract_to_vector(
        residual,
        channel_mask=mask,
        measurement_weights=weights,
    )

    np.testing.assert_allclose(
        weighted_jac,
        np.array([[2.0, 4.0], [0.0, 0.0], [5.0, 6.0]]),
    )
    np.testing.assert_allclose(weighted_residual, np.array([20.0, 0.0, 30.0]))
    assert jac_contract.bad_channel_count == 1
    assert residual_contract.weight_kind == "diagonal"


def test_measurement_contract_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        prepare_measurement_contract(
            n_measurements=2,
            measurement_weights=np.array([1.0, -1.0]),
        )
    with pytest.raises(ValueError, match="symmetric"):
        prepare_measurement_contract(
            n_measurements=2,
            measurement_weights=np.array([[1.0, 2.0], [0.0, 1.0]]),
        )
