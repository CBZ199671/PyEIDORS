"""Tests for controlled voltage significant-digit EIT sweeps."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from pyeidors.data.eit_digit_metrics import build_surrogate_linearized_model
from pyeidors.data.voltage_digit_sweep import (
    keep_significant_digits,
    plot_voltage_digit_sweep,
    run_voltage_digit_sweep,
)
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_keep_significant_digits_truncates_like_word_table() -> None:
    values = np.array(
        [
            473.345698734,
            42.3456987378,
            4.32918985497,
            4273.34569873,
            0.0,
            -12.3456,
        ]
    )

    np.testing.assert_allclose(
        keep_significant_digits(values, 4),
        np.array([473.3, 42.34, 4.329, 4273.0, 0.0, -12.34]),
    )
    np.testing.assert_allclose(
        keep_significant_digits(values, 5),
        np.array([473.34, 42.345, 4.3291, 4273.3, 0.0, -12.345]),
    )


def test_keep_significant_digits_can_round_for_comparison() -> None:
    values = np.array([42.3456987378, -12.3456])

    np.testing.assert_allclose(
        keep_significant_digits(values, 5, method="round"),
        np.array([42.346, -12.346]),
    )


def test_voltage_digit_sweep_reports_distribution_errors() -> None:
    model = build_surrogate_linearized_model(
        n_measurements=10,
        n_parameters=5,
        seed=123,
    )
    summaries, field_rows = run_voltage_digit_sweep(
        model=model,
        target_digits=[4, 5, 6],
        inverse_backend="least-squares",
        ridge=1e-8,
    )

    assert [row.target_voltage_digits for row in summaries] == [4, 5, 6]
    assert len(field_rows) == model.sigma_true.size * len(summaries)
    for row in summaries:
        assert math.isfinite(row.achieved_voltage_effective_digits)
        assert math.isfinite(row.voltage_rmse)
        assert math.isfinite(row.sigma_rmse)
        assert math.isfinite(row.sigma_relative_rmse)
        assert math.isfinite(row.sigma_mae)
        assert math.isfinite(row.sigma_max_abs_error)
        assert math.isfinite(row.sigma_effective_digits)
    assert (
        summaries[2].voltage_rmse
        <= summaries[1].voltage_rmse
        <= summaries[0].voltage_rmse
    )
    assert summaries[2].sigma_rmse <= summaries[1].sigma_rmse <= summaries[0].sigma_rmse


def test_voltage_digit_sweep_plot_writes_png(tmp_path) -> None:
    model = build_surrogate_linearized_model(
        n_measurements=10,
        n_parameters=5,
        seed=123,
    )
    summaries, _ = run_voltage_digit_sweep(
        model=model,
        target_digits=[4, 5],
        inverse_backend="least-squares",
    )

    output = plot_voltage_digit_sweep(summaries, tmp_path / "sweep.png", dpi=80)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 1000


def test_eit_voltage_digit_sweep_cli_writes_expected_outputs(tmp_path) -> None:
    summary_output = tmp_path / "summary.csv"
    field_output = tmp_path / "fields.csv"
    plot_output = tmp_path / "plot.png"
    hdf5_output = tmp_path / "tables.h5"
    json_output = tmp_path / "tables.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_voltage_digit_sweep.py",
            "--target-digits",
            "4",
            "5",
            "--forward-backend",
            "surrogate",
            "--inverse-backend",
            "least-squares",
            "--n-measurements",
            "10",
            "--n-parameters",
            "5",
            "--noser-exponent",
            "0.5",
            "--output",
            str(summary_output),
            "--field-output",
            str(field_output),
            "--plot-output",
            str(plot_output),
            "--hdf5-output",
            str(hdf5_output),
            "--json-output",
            str(json_output),
            "--dpi",
            "80",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "model=surrogate+least-squares" in completed.stdout
    assert "noser_exponent=0.5" in completed.stdout
    assert "target_voltage_digits" in completed.stdout
    assert "Wrote hdf5:" in completed.stdout
    assert "Wrote json:" in completed.stdout

    with summary_output.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with field_output.open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))

    assert [row["target_voltage_digits"] for row in summary_rows] == ["4", "5"]
    assert set(summary_rows[0]) == {
        "target_voltage_digits",
        "achieved_voltage_effective_digits",
        "voltage_rmse",
        "sigma_rmse",
        "sigma_relative_rmse",
        "sigma_mae",
        "sigma_max_abs_error",
        "sigma_effective_digits",
    }
    assert len(field_rows) == 10
    assert set(field_rows[0]) == {
        "target_voltage_digits",
        "cell_index",
        "sigma_true",
        "sigma_recon",
        "sigma_error",
        "abs_sigma_error",
    }
    assert plot_output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    artifact = read_hdf5_artifact(hdf5_output, lazy=True, verify_checksums=False)
    payload = json.loads(json_output.read_text(encoding="utf-8"))
    assert artifact.metadata["table_names"] == [
        "voltage_digit_field",
        "voltage_digit_summary",
    ]
    assert payload["metadata"]["table_names"] == [
        "voltage_digit_field",
        "voltage_digit_summary",
    ]
    assert len(payload["tables"]["voltage_digit_summary"]["rows"]) == 2
