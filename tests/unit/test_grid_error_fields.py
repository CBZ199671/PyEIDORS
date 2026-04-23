"""Tests for T18 grid conductivity error-field plotting."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from pyeidors.data.grid_error_fields import (
    FIELD_FIELDS,
    SUMMARY_FIELDS,
    format_grid_error_report,
    plot_grid_error_fields,
    run_grid_error_fields,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _small_surrogate_cases():
    return run_grid_error_fields(
        fem_grid_levels=[4, 6],
        forward_backend="surrogate",
        inverse_backend="least-squares",
        n_elec=16,
        ridge=1e-8,
        target_voltage_digits=6,
        n_measurements=10,
        n_parameters=5,
        model_seed=123,
    )


def test_grid_error_cases_report_sigma_error_sign_and_locations() -> None:
    cases = _small_surrogate_cases()

    assert [case.summary.fem_grid for case in cases] == [4, 6]
    for case in cases:
        assert len(case.field_rows) == case.summary.n_parameters
        assert case.parameter_points.shape == (case.summary.n_parameters, 2)
        np.testing.assert_allclose(
            case.sigma_error,
            case.sigma_recon - case.sigma_true,
        )
        assert np.isfinite(case.summary.sigma_relative_rmse)
        assert np.isfinite(case.summary.sigma_effective_digits)
        assert 0 <= case.summary.max_abs_error_cell_index < case.summary.n_parameters


def test_grid_error_report_and_plot_outputs(tmp_path) -> None:
    cases = _small_surrogate_cases()
    report = format_grid_error_report(cases)
    output = plot_grid_error_fields(cases, tmp_path / "grid_fields.png", dpi=80)

    assert "误差定位" in report
    assert "不做跨网格逐单元相减" in report
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output.stat().st_size > 1000


def test_eit_grid_error_fields_cli_writes_expected_outputs(tmp_path) -> None:
    summary_output = tmp_path / "summary.csv"
    field_output = tmp_path / "fields.csv"
    report_output = tmp_path / "fields.md"
    plot_output = tmp_path / "fields.png"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_grid_error_fields.py",
            "--forward-backend",
            "surrogate",
            "--inverse-backend",
            "least-squares",
            "--fem-grid-levels",
            "4",
            "6",
            "--n-measurements",
            "10",
            "--n-parameters",
            "5",
            "--model-seed",
            "123",
            "--summary-output",
            str(summary_output),
            "--field-output",
            str(field_output),
            "--report-output",
            str(report_output),
            "--plot-output",
            str(plot_output),
            "--dpi",
            "80",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "model=surrogate+least-squares" in completed.stdout
    with summary_output.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with field_output.open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))

    assert len(summary_rows) == 2
    assert list(summary_rows[0].keys()) == SUMMARY_FIELDS
    assert len(field_rows) == 10
    assert list(field_rows[0].keys()) == FIELD_FIELDS
    assert report_output.read_text(encoding="utf-8").startswith(
        "# T18 网格电导率误差场报告"
    )
    assert plot_output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
