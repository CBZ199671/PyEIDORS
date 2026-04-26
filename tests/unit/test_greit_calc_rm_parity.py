from __future__ import annotations

import numpy as np
import pytest

from pyeidors.inverse.greit import GREITRMComponents, calc_greit_rm


def test_calc_greit_rm_matches_exported_eidors_components() -> None:
    y = np.array(
        [
            [1.0, -0.5, 0.25],
            [0.2, 0.7, -0.4],
            [-0.3, 0.1, 0.9],
            [0.8, -0.6, 0.5],
        ],
        dtype=float,
    )
    d = np.array(
        [
            [1.0, 0.0, 0.2],
            [0.0, 1.0, -0.1],
            [0.5, 0.25, 0.75],
        ],
        dtype=float,
    )
    weight = 0.35
    noise_covar = 1.75

    result = calc_greit_rm(y, d, weight=weight, noise_covar=noise_covar)

    expected_pjt = d @ y.T
    expected_noiselev = weight * np.mean(np.abs(y))
    expected_sn = noise_covar * np.eye(y.shape[0])
    expected_m = y @ y.T + expected_noiselev**2 * expected_sn
    expected_rm = np.linalg.solve(expected_m.T, expected_pjt.T).T

    assert isinstance(result, GREITRMComponents)
    np.testing.assert_allclose(result.pjt, expected_pjt)
    assert result.noiselev == pytest.approx(expected_noiselev)
    np.testing.assert_allclose(result.sn, expected_sn)
    np.testing.assert_allclose(result.m, expected_m)
    np.testing.assert_allclose(result.rm, expected_rm)
    assert result.metadata["solver"] == "solve"
    assert result.metadata["singular_fallback"] is False
    assert result.metadata["eidors_component_parity"] is True
    assert result.metadata["transpose_semantics"] == (
        "matlab_nonconjugate_dot_transpose"
    )
    assert result.metadata["pjt_shape"] == expected_pjt.shape
    assert result.metadata["m_shape"] == expected_m.shape
    assert result.metadata["rm_shape"] == expected_rm.shape


def test_calc_greit_rm_uses_nonconjugate_transpose_for_complex_components() -> None:
    y = np.array(
        [
            [1.0 + 2.0j, 0.5 - 0.25j],
            [-0.2 + 0.4j, 0.9 + 0.1j],
        ],
        dtype=np.complex128,
    )
    d = np.array(
        [
            [0.8 - 0.3j, -0.1 + 0.2j],
            [0.5 + 0.7j, 1.2 - 0.4j],
        ],
        dtype=np.complex128,
    )

    result = calc_greit_rm(y, d, weight=0.1, noise_covar=0.5)

    expected_pjt = d @ y.T
    wrong_conjugate_pjt = d @ y.conj().T
    np.testing.assert_allclose(result.pjt, expected_pjt)
    assert not np.allclose(result.pjt, wrong_conjugate_pjt)


def test_calc_greit_rm_accepts_full_noise_covariance() -> None:
    y = np.array([[1.0, 0.2], [0.4, 0.9]], dtype=float)
    d = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=float)
    noise_covar = np.array([[2.0, 0.25], [0.25, 1.0]], dtype=float)

    result = calc_greit_rm(y, d, weight=0.2, noise_covar=noise_covar)

    np.testing.assert_allclose(result.sn, noise_covar)
    assert result.metadata["noise_covar_source"] == "matrix"


def test_calc_greit_rm_rejects_matrix_weight_until_nf_search_lands() -> None:
    y = np.eye(2, dtype=float)
    d = np.eye(2, dtype=float)

    with pytest.raises(NotImplementedError, match="scalar weight"):
        calc_greit_rm(y, d, weight=np.eye(2), noise_covar=1.0)


def test_calc_greit_rm_singular_fallback_reports_diagnostics() -> None:
    y = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=float)
    d = np.array([[1.0, 0.5], [0.25, 0.75]], dtype=float)

    result = calc_greit_rm(y, d, weight=0.0, noise_covar=0.0)

    expected_pjt = d @ y.T
    expected_m = y @ y.T
    expected_rm = (np.linalg.pinv(expected_m.T) @ expected_pjt.T).T
    np.testing.assert_allclose(result.rm, expected_rm)
    assert result.metadata["solver"] == "pinv"
    assert result.metadata["singular_fallback"] is True
    assert result.metadata["eidors_component_parity"] is False
    assert result.metadata["matrix_rank"] < expected_m.shape[0]
