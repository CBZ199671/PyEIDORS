"""Tests for EIT digit report table generation."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import pytest

from pyeidors.data.adc_quantization import ideal_decimal_digits
from pyeidors.data.digit_report import (
    EIT_DIGIT_FIELDS,
    DigitReportCase,
    format_markdown_report,
    read_eit_digit_case,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_eit_digits_csv(path: Path, *, bad_delta: bool = False) -> None:
    row = {
        "bit": 12,
        "ideal_decimal_digits": ideal_decimal_digits(12),
        "voltage_rmse": 0.001,
        "voltage_effective_digits": 3.0,
        "sigma_rmse": 0.002,
        "sigma_effective_digits": 2.5,
        "hypothesis_delta_digits": -0.4 if bad_delta else -0.5,
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EIT_DIGIT_FIELDS)
        writer.writeheader()
        writer.writerow(row)


def test_read_eit_digit_case_attaches_settings_and_validates_delta(tmp_path) -> None:
    source = tmp_path / "eit_digits.csv"
    _write_eit_digits_csv(source)

    rows = read_eit_digit_case(
        DigitReportCase(
            label="surrogate",
            path=source,
            full_scale=10.0,
            enob=11.0,
            noise_std=0.0,
            noise_relative=0.001,
        )
    )

    assert len(rows) == 1
    assert rows[0].scenario == "surrogate"
    assert rows[0].full_scale == 10.0
    assert rows[0].enob == 11.0
    assert rows[0].hypothesis_delta_digits == -0.5

    report = format_markdown_report(rows)
    assert "full_scale" in report
    assert "hypothesis_delta_digits" in report
    assert "direct conductivity digit conclusion" in report


def test_read_eit_digit_case_rejects_bad_hypothesis_delta(tmp_path) -> None:
    source = tmp_path / "bad_eit_digits.csv"
    _write_eit_digits_csv(source, bad_delta=True)

    with pytest.raises(ValueError, match="hypothesis_delta_digits"):
        read_eit_digit_case(
            DigitReportCase(
                label="bad",
                path=source,
                full_scale=10.0,
            )
        )


def test_eit_digit_report_cli_writes_markdown_and_csv(tmp_path) -> None:
    source = tmp_path / "eit_digits.csv"
    output_md = tmp_path / "report.md"
    output_csv = tmp_path / "report.csv"
    _write_eit_digits_csv(source)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_digit_report.py",
            "--input",
            str(source),
            "--label",
            "surrogate",
            "--full-scale",
            "10",
            "--enob",
            "nominal",
            "--noise-std",
            "0",
            "--noise-relative",
            "0",
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "hypothesis_delta_digits" in completed.stdout
    assert output_md.exists()
    assert output_csv.exists()
    assert "surrogate" in output_md.read_text(encoding="utf-8")
    with output_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["scenario"] == "surrogate"
    assert rows[0]["enob"] == ""
