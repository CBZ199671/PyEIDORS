"""Tests for pure ADC quantization precision metrics."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from pyeidors.data.adc_quantization import (
    adc_lsb,
    effective_digits_from_rmse,
    ideal_decimal_digits,
    pointwise_effective_digits,
    quantize_voltages,
    rmse,
    summarize_adc_sweep,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ideal_decimal_digits_and_lsb_follow_adc_formula() -> None:
    assert ideal_decimal_digits(12) == 12 * math.log10(2.0)
    assert adc_lsb(10.0, 12) == 10.0 / (2**12)


def test_quantize_voltages_uses_round_to_nearest_lsb() -> None:
    voltages = np.array([0.24, 0.26, -0.24, -0.26])

    np.testing.assert_allclose(
        quantize_voltages(voltages, bit=2, full_scale_range=1.0),
        np.array([0.25, 0.25, -0.25, -0.25]),
    )


def test_effective_digits_handle_rmse_and_zero_reference() -> None:
    reference = np.array([100.0, 10.0, 0.0])
    observed = np.array([99.0, 10.1, 0.5])

    assert rmse(reference, observed) > 0.0
    assert math.isfinite(effective_digits_from_rmse(reference, observed))

    digits = pointwise_effective_digits(reference, observed)
    np.testing.assert_allclose(digits[:2], [2.0, 2.0])
    assert math.isnan(float(digits[2]))

    exact_zero = pointwise_effective_digits([0.0], [0.0])
    assert math.isinf(float(exact_zero[0]))


def test_adc_sweep_summarizes_requested_bit_depths() -> None:
    rows = summarize_adc_sweep(
        [473.345698734, 42.3456987378],
        bits=[12, 16],
        full_scale_range=10000.0,
    )

    assert [row.bit for row in rows] == [12, 16]
    assert rows[0].lsb == 10000.0 / (2**12)
    assert rows[1].voltage_rmse <= rows[0].voltage_rmse
    assert rows[1].voltage_effective_digits >= rows[0].voltage_effective_digits


def test_adc_quant_cli_writes_expected_csv(tmp_path) -> None:
    output = tmp_path / "adc_quant.csv"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/adc_quant_test.py",
            "--bits",
            "12",
            "16",
            "--full-scale",
            "10000",
            "--voltages",
            "473.345698734",
            "42.3456987378",
            "--output",
            str(output),
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "voltage_effective_digits" in completed.stdout
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["bit"] for row in rows] == ["12", "16"]
    assert set(rows[0]) == {
        "bit",
        "ideal_decimal_digits",
        "full_scale",
        "lsb",
        "voltage_rmse",
        "voltage_effective_digits",
    }
