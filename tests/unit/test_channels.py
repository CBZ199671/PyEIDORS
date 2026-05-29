"""Tests for measurement-channel masks and weighting contracts."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import pyeidors.data.channels as channels_module
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


def test_v419_bad_channel_zeroing_scans_mask_without_boolean_lhs_indexing() -> None:
    for obj in (
        zero_bad_channel_rows,
        zero_bad_channel_vector,
        zero_bad_channel_weights,
    ):
        source = inspect.getsource(obj)
        assert "[mask" not in source
        assert "[:, mask]" not in source
    assert "_zero_masked_rows_in_place" in inspect.getsource(zero_bad_channel_rows)
    assert "_zero_masked_entries_in_place" in inspect.getsource(zero_bad_channel_vector)
    assert "_zero_masked_square_rows_cols_in_place" in inspect.getsource(
        zero_bad_channel_weights
    )

    mask = np.array([True, False, True], dtype=bool)
    matrix = np.arange(9.0, dtype=float).reshape(3, 3)
    channels_module._zero_masked_square_rows_cols_in_place(matrix, mask)
    np.testing.assert_allclose(
        matrix,
        np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_v483_measurement_channel_finite_guards_use_bounded_scanner() -> None:
    checked_functions = (
        zero_bad_channel_weights,
        channels_module._as_vector,
        channels_module._as_2d_array,
    )
    old_payload_scans = (
        "np.isfinite(weights).all()",
        "np.isfinite(array).all()",
        "np.any(weights < 0.0)",
    )

    for func in checked_functions:
        source = inspect.getsource(func)
        assert "all_finite_values(" in source
        for old_payload_scan in old_payload_scans:
            assert old_payload_scan not in source

    mask_source = inspect.getsource(bad_channel_mask)
    assert "np.min(indices" in mask_source
    assert "np.max(indices" in mask_source
    assert "np.any((indices < 0) | (indices >= n))" not in mask_source


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


def test_v557_measurement_contract_preserves_float32_inputs() -> None:
    jac_source = inspect.getsource(apply_measurement_contract_to_jacobian)
    vec_source = inspect.getsource(apply_measurement_contract_to_vector)
    diag_source = inspect.getsource(channels_module._DiagonalMatrix.__matmul__)

    assert "dtype=matrix.dtype" in jac_source
    assert "dtype=values.dtype" in vec_source
    assert "np.asarray(other, dtype=np.float64)" not in diag_source

    mask = np.array([False, True, False], dtype=bool)
    weights = np.array([4.0, 9.0, 0.25], dtype=np.float32)
    jacobian = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    residual = np.array([10.0, 20.0, 30.0], dtype=np.float32)

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
    identity_weighted, identity_contract = apply_measurement_contract_to_vector(
        residual,
        channel_mask=None,
        measurement_weights=None,
    )

    assert weighted_jac.dtype == np.dtype(np.float32)
    assert weighted_residual.dtype == np.dtype(np.float32)
    assert jac_contract.weight_transform.dtype == np.dtype(np.float32)
    assert residual_contract.weight_matrix.dtype == np.dtype(np.float32)
    assert identity_weighted.dtype == np.dtype(np.float32)
    assert identity_contract.weight_transform.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        weighted_jac, np.array([[2.0, 4.0], [0.0, 0.0], [2.5, 3.0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        weighted_residual, np.array([20.0, 0.0, 15.0], dtype=np.float32)
    )


def test_v423_diagonal_measurement_contract_stays_lightweight_until_array_conversion() -> (
    None
):
    source = inspect.getsource(prepare_measurement_contract)
    sqrt_source = inspect.getsource(channels_module._sqrt_weight_transform)
    assert "np.diag(weights)" not in source
    assert "np.diag(np.sqrt(weights))" not in sqrt_source

    contract = prepare_measurement_contract(
        n_measurements=4,
        channel_mask=[False, True, False, False],
        measurement_weights=np.array([4.0, 9.0, 0.25, 1.0], dtype=np.float64),
    )
    assert contract.weight_kind == "diagonal"
    assert contract.weight_matrix.shape == (4, 4)
    assert contract.weight_transform.shape == (4, 4)
    assert not isinstance(contract.weight_matrix, np.ndarray)
    assert not isinstance(contract.weight_transform, np.ndarray)

    matrix = np.arange(8.0, dtype=np.float64).reshape(4, 2)
    weighted = contract.weight_transform @ matrix
    np.testing.assert_allclose(
        weighted,
        np.array(
            [
                [0.0, 2.0],
                [0.0, 0.0],
                [2.0, 2.5],
                [6.0, 7.0],
            ],
            dtype=np.float64,
        ),
    )
    np.testing.assert_allclose(
        matrix.T @ contract.weight_transform,
        matrix.T @ np.diag([2.0, 0.0, 0.5, 1.0]),
    )
    np.testing.assert_allclose(
        np.asarray(contract.weight_matrix),
        np.diag([4.0, 0.0, 0.25, 1.0]),
    )


def test_v517_diagonal_matrix_array_conversion_direct_fills_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    array_source = inspect.getsource(channels_module._DiagonalMatrix.__array__)
    assert "np.diag" not in array_source

    contract = prepare_measurement_contract(
        n_measurements=3,
        measurement_weights=np.array([4.0, 0.25, 1.0], dtype=np.float64),
    )

    def _fail_diag(*_args, **_kwargs):
        raise AssertionError("diagonal matrix conversion must not call np.diag")

    monkeypatch.setattr(channels_module.np, "diag", _fail_diag)

    dense = np.asarray(contract.weight_matrix)
    expected = np.zeros((3, 3), dtype=np.float64)
    expected.reshape(-1)[::4] = np.array([4.0, 0.25, 1.0], dtype=np.float64)
    np.testing.assert_allclose(dense, expected)


def test_v424_full_weight_sqrt_transform_scales_eigenvectors_without_dense_diag() -> (
    None
):
    source = inspect.getsource(channels_module._sqrt_weight_transform)
    assert "np.diag(np.sqrt(clipped))" not in source
    assert "sqrt_values.reshape(-1, 1) * eigenvectors.T" in source

    weights = np.array([[2.0, 0.5], [0.5, 1.5]], dtype=np.float64)
    contract = prepare_measurement_contract(
        n_measurements=2,
        measurement_weights=weights,
    )

    assert isinstance(contract.weight_transform, np.ndarray)
    np.testing.assert_allclose(
        contract.weight_transform.T @ contract.weight_transform,
        weights,
        rtol=1e-12,
        atol=1e-12,
    )


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
