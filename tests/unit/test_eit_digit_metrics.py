"""Tests for EIT end-to-end digit metrics."""

from __future__ import annotations

import csv
import inspect
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import pyeidors.data.eit_digit_metrics as digit_module
from pyeidors.data.eit_digit_metrics import (
    adjacent_measurement_count,
    build_surrogate_sensitivity,
    default_sigma_true,
    forward_surrogate,
    inverse_pyeidors_rm,
    inverse_surrogate,
    summarize_eit_digit_sweep,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v496_digit_metric_validators_use_bounded_finite_scan() -> None:
    vector_source = inspect.getsource(digit_module._as_float_vector)
    matrix_source = inspect.getsource(digit_module._as_float_matrix)

    assert "all_finite_values(arr)" in vector_source
    assert "all_finite_values(arr)" in matrix_source
    assert "np.all(np.isfinite(arr))" not in vector_source
    assert "np.all(np.isfinite(arr))" not in matrix_source


def test_adjacent_measurement_count_matches_ad_pattern_frame_sizes() -> None:
    assert adjacent_measurement_count(8) == 40
    assert adjacent_measurement_count(16) == 208


def test_surrogate_forward_inverse_round_trips_without_adc_error() -> None:
    sigma = default_sigma_true(5)
    sensitivity = build_surrogate_sensitivity(
        n_measurements=8,
        n_parameters=sigma.size,
        seed=123,
    )
    voltages = forward_surrogate(sigma, sensitivity)
    reconstructed = inverse_surrogate(voltages, sensitivity, ridge=0.0)

    np.testing.assert_allclose(reconstructed, sigma, rtol=1e-10, atol=1e-10)


def test_v516_digit_metric_ridge_terms_are_added_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    surrogate_source = inspect.getsource(digit_module.inverse_surrogate)
    measurement_source = inspect.getsource(digit_module.inverse_measurement_rm)
    assert "ridge_value * np.eye" not in surrogate_source
    assert "(lam * lam) * np.eye" not in measurement_source
    assert "add_scaled_diagonal_in_place(normal, identity_diag, ridge_value)" in (
        surrogate_source
    )
    assert "add_scaled_diagonal_in_place(lhs, identity_diag, lam * lam)" in (
        measurement_source
    )

    def _fail_eye(*_args, **_kwargs):
        raise AssertionError("digit metric ridge paths must not materialize np.eye")

    monkeypatch.setattr(digit_module.np, "eye", _fail_eye)

    sensitivity = np.array(
        [[1.0, 0.2], [0.5, 2.0], [0.1, 0.3]],
        dtype=float,
    )
    voltages = sensitivity @ np.array([1.2, 0.85], dtype=float)

    surrogate = inverse_surrogate(voltages, sensitivity, ridge=0.05)
    measurement = digit_module.inverse_measurement_rm(
        voltages,
        sensitivity,
        lambda_=0.1,
    )

    assert surrogate.shape == (2,)
    assert measurement.shape == (2,)
    assert np.isfinite(surrogate).all()
    assert np.isfinite(measurement).all()


def test_pyeidors_rm_inverse_round_trips_without_adc_error() -> None:
    sigma = default_sigma_true(5)
    sensitivity = build_surrogate_sensitivity(
        n_measurements=8,
        n_parameters=sigma.size,
        seed=123,
    )
    voltages = forward_surrogate(sigma, sensitivity)
    reconstructed = inverse_pyeidors_rm(
        voltages,
        sensitivity,
        lambda_=0.0,
        mode="tikhonov",
    )

    np.testing.assert_allclose(reconstructed, sigma, rtol=1e-10, atol=1e-10)


def test_pyeidors_rm_inverse_passes_noser_exponent() -> None:
    from pyeidors.inverse.reconstruction_matrix import build_one_step_rm

    sigma = np.array([1.2, 0.85], dtype=float)
    sensitivity = np.array([[1.0, 0.2], [0.5, 2.0], [0.1, 0.3]], dtype=float)
    voltages = sensitivity @ sigma

    reconstructed = inverse_pyeidors_rm(
        voltages,
        sensitivity,
        lambda_=0.2,
        mode="noser",
        noser_exponent=0.5,
    )
    expected_rm = build_one_step_rm(
        sensitivity,
        lambda_=0.2,
        mode="noser",
        noser_exponent=0.5,
    )
    default_reconstructed = inverse_pyeidors_rm(
        voltages,
        sensitivity,
        lambda_=0.2,
        mode="noser",
        noser_exponent=1.0,
    )

    np.testing.assert_allclose(reconstructed, expected_rm @ voltages)
    assert not np.allclose(reconstructed, default_reconstructed)


def test_eit_digit_sweep_reports_hypothesis_delta() -> None:
    rows = summarize_eit_digit_sweep(
        bits=[12, 16],
        full_scale_range=100.0,
        inverse_backend="pyeidors-rm",
        n_measurements=8,
        n_parameters=5,
        model_seed=123,
    )

    assert [row.bit for row in rows] == [12, 16]
    for row in rows:
        assert math.isfinite(row.voltage_rmse)
        assert math.isfinite(row.sigma_rmse)
        assert row.hypothesis_delta_digits == (
            row.sigma_effective_digits - row.voltage_effective_digits
        )
    assert rows[1].voltage_effective_digits >= rows[0].voltage_effective_digits


def test_eit_digit_sweep_supports_noise_and_enob() -> None:
    rows = summarize_eit_digit_sweep(
        bits=[12, 16],
        full_scale_range=100.0,
        enob=10.0,
        noise_std=0.001,
        noise_relative=0.001,
        seed=7,
        inverse_backend="pyeidors-rm",
        n_measurements=8,
        n_parameters=5,
        model_seed=123,
    )

    assert len(rows) == 2
    assert rows[0].voltage_rmse == rows[1].voltage_rmse
    assert rows[0].sigma_rmse == rows[1].sigma_rmse


def test_pyeidors_fem_digit_sweep_smoke() -> None:
    pytest.importorskip("dolfinx")

    rows = summarize_eit_digit_sweep(
        bits=[12],
        full_scale_range=10.0,
        forward_backend="pyeidors-fem",
        fem_n_elec=8,
        fem_grid=2,
        expected_fem_measurements=40,
        inverse_backend="pyeidors-rm",
        ridge=1e-2,
    )

    assert len(rows) == 1
    assert rows[0].bit == 12
    assert math.isfinite(rows[0].voltage_rmse)
    assert math.isfinite(rows[0].sigma_rmse)
    assert math.isfinite(rows[0].hypothesis_delta_digits)


def test_eit_end_to_end_cli_writes_expected_csv(tmp_path) -> None:
    output = tmp_path / "eit_digits.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_end_to_end_test.py",
            "--bits",
            "12",
            "16",
            "--full-scale",
            "100",
            "--noise",
            "0.001",
            "--enob",
            "11",
            "--n-measurements",
            "8",
            "--n-parameters",
            "5",
            "--forward-backend",
            "surrogate",
            "--inverse-backend",
            "pyeidors-rm",
            "--noser-exponent",
            "0.5",
            "--output",
            str(output),
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "model=surrogate+pyeidors-rm" in completed.stdout
    assert "rm_mode=tikhonov" in completed.stdout
    assert "noser_exponent=0.5" in completed.stdout
    assert "hypothesis_delta_digits" in completed.stdout
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["bit"] for row in rows] == ["12", "16"]
    assert set(rows[0]) == {
        "bit",
        "ideal_decimal_digits",
        "voltage_rmse",
        "voltage_effective_digits",
        "sigma_rmse",
        "sigma_effective_digits",
        "hypothesis_delta_digits",
    }
