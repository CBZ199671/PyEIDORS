"""Tests for EIT digit plot generation."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import matplotlib

from pyeidors.data.digit_plot import (
    configure_times_new_roman,
    plot_digit_report_csv,
    read_digit_report_rows,
)
from pyeidors.data.digit_report import REPORT_FIELDS


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_report_csv(path: Path) -> None:
    rows = [
        {
            "scenario": "surrogate-rm",
            "bit": 12,
            "ideal_decimal_digits": 3.612359947967774,
            "full_scale": 10.0,
            "enob": "",
            "noise_std": 0.0,
            "noise_relative": 0.0,
            "voltage_rmse": 0.001,
            "voltage_effective_digits": 3.0,
            "sigma_rmse": 0.002,
            "sigma_effective_digits": 2.6,
            "hypothesis_delta_digits": -0.4,
        },
        {
            "scenario": "surrogate-rm",
            "bit": 16,
            "ideal_decimal_digits": 4.816479930623699,
            "full_scale": 10.0,
            "enob": "",
            "noise_std": 0.0,
            "noise_relative": 0.0,
            "voltage_rmse": 0.0001,
            "voltage_effective_digits": 4.0,
            "sigma_rmse": 0.0002,
            "sigma_effective_digits": 3.7,
            "hypothesis_delta_digits": -0.3,
        },
        {
            "scenario": "pyeidors-fem-rm",
            "bit": 12,
            "ideal_decimal_digits": 3.612359947967774,
            "full_scale": 10.0,
            "enob": 11.0,
            "noise_std": 0.0,
            "noise_relative": 0.001,
            "voltage_rmse": 0.003,
            "voltage_effective_digits": 2.0,
            "sigma_rmse": 0.08,
            "sigma_effective_digits": 1.1,
            "hypothesis_delta_digits": -0.9,
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_configure_times_new_roman_sets_global_matplotlib_family() -> None:
    configure_times_new_roman()

    assert matplotlib.rcParams["font.family"] == ["serif"]
    assert matplotlib.rcParams["font.serif"][0] == "Times New Roman"
    assert matplotlib.rcParams["axes.unicode_minus"] is False


def test_plot_digit_report_csv_writes_png(tmp_path) -> None:
    source = tmp_path / "report.csv"
    output = tmp_path / "plot.png"
    _write_report_csv(source)

    rows = read_digit_report_rows(source)
    assert [row.scenario for row in rows][:2] == ["surrogate-rm", "surrogate-rm"]
    assert rows[2].enob == 11.0

    written = plot_digit_report_csv(input_csv=source, output_path=output, dpi=120)

    assert written == output
    assert output.read_bytes().startswith(b"\x89PNG")
    assert output.stat().st_size > 1000


def test_eit_digit_plot_cli_writes_png(tmp_path) -> None:
    source = tmp_path / "report.csv"
    output = tmp_path / "plot.png"
    _write_report_csv(source)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_digit_plot.py",
            "--input-csv",
            str(source),
            "--output",
            str(output),
            "--dpi",
            "120",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "Wrote" in completed.stdout
    assert output.exists()
    assert output.read_bytes().startswith(b"\x89PNG")
