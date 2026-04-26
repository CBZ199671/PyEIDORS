"""Tests for EIDORS-compatible measurement noise injection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.data.noise import add_noise
from pyeidors.data.structures import EITData


def _snr(signal: np.ndarray, noise: np.ndarray) -> float:
    return float(np.linalg.norm(signal) / np.linalg.norm(noise))


def test_add_noise_is_exported_from_data_package() -> None:
    from pyeidors.data import add_noise as exported_add_noise

    assert exported_add_noise is add_noise


def test_add_noise_uses_v1_as_signal_when_reference_absent() -> None:
    v1 = np.array([2.0, 2.0, 2.0, 2.0], dtype=float)

    noisy = add_noise(2.0, v1, seed=123)
    noise = noisy - v1

    np.testing.assert_allclose(_snr(v1, noise), 2.0, rtol=1e-14, atol=1e-14)
    assert not np.shares_memory(noisy, v1)
    np.testing.assert_allclose(v1, np.full(4, 2.0))


def test_add_noise_uses_difference_signal_and_eidors_column_broadcast() -> None:
    v1 = np.array([[2.0, 2.2], [3.0, 2.7], [4.0, 4.4]], dtype=float)
    v2 = np.array([2.1, 2.9, 4.2], dtype=float)
    signal = v1 - v2[:, None]

    noisy = add_noise(0.5, v1, v2, seed=321)
    noise = noisy - v1

    assert noisy.shape == v1.shape
    np.testing.assert_allclose(_snr(signal, noise), 0.5, rtol=1e-14, atol=1e-14)


def test_add_noise_norm_option_uses_normalized_difference_signal() -> None:
    v1 = np.array([2.0, 3.0, 4.0], dtype=float)
    v2 = np.array([2.1, 2.9, 4.2], dtype=float)
    signal = (v1 - v2) / v2

    noisy = add_noise(0.9, v1, v2, "norm", seed=456)
    noise = noisy - v1

    np.testing.assert_allclose(_snr(signal, noise), 0.9, rtol=1e-14, atol=1e-14)


def test_add_noise_preserves_eitdata_metadata() -> None:
    data = EITData(
        meas=np.array([1.0, 2.0, 3.0], dtype=float),
        stim_pattern=np.eye(2, dtype=float),
        n_elec=2,
        n_stim=1,
        n_meas=3,
        type="difference",
        difference_mode="normalized",
    )

    noisy = add_noise(4.0, data, seed=7)

    assert isinstance(noisy, EITData)
    assert noisy is not data
    assert noisy.type == data.type
    assert noisy.difference_mode == data.difference_mode
    np.testing.assert_allclose(data.meas, np.array([1.0, 2.0, 3.0], dtype=float))
    np.testing.assert_allclose(
        _snr(data.meas, noisy.meas - data.meas),
        4.0,
        rtol=1e-14,
        atol=1e-14,
    )


def test_add_noise_preserves_generic_meas_objects() -> None:
    data = SimpleNamespace(
        meas=np.array([1.0, 1.5, 2.0], dtype=float),
        name="frame-a",
    )

    noisy = add_noise(3.0, data, seed=11)

    assert isinstance(noisy, SimpleNamespace)
    assert noisy is not data
    assert noisy.name == "frame-a"
    np.testing.assert_allclose(data.meas, np.array([1.0, 1.5, 2.0], dtype=float))
    np.testing.assert_allclose(_snr(data.meas, noisy.meas - data.meas), 3.0)


def test_add_noise_validates_rng_options_and_zero_signal() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="either rng or seed"):
        add_noise(1.0, [1.0], seed=1, rng=rng)
    with pytest.raises(ValueError, match="options"):
        add_noise(1.0, [1.0], options="norm")
    with pytest.raises(ValueError, match="positive and finite"):
        add_noise(0.0, [1.0])

    zero = np.zeros(3, dtype=float)
    np.testing.assert_allclose(add_noise(1.0, zero, seed=1), zero)
