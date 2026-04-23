from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np

from pyeidors.data.eit_digit_metrics import build_surrogate_linearized_model
from pyeidors.data.holdout_fit_diff import (
    FIELD_FIELDS,
    STRUCTURE_FIELDS,
    SUMMARY_FIELDS,
    format_holdout_fit_report,
    plot_holdout_fit_curves,
    plot_holdout_fit_summary,
    plot_holdout_recon_compare,
    run_holdout_fit_diff,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _surrogate_model():
    return build_surrogate_linearized_model(
        n_measurements=208,
        n_parameters=6,
        seed=123,
    )


def test_holdout_fit_diff_compares_raw_160_and_three_fit_methods() -> None:
    case = run_holdout_fit_diff(
        model=_surrogate_model(),
        fit_methods=["poly2", "poly3", "spline"],
        raw_160_baseline=True,
        inverse_backend="least-squares",
        ridge=1e-4,
    )

    assert {row.recon_method for row in case.summaries} == {
        "raw_160",
        "poly2_208",
        "poly3_208",
        "spline_208",
    }
    inverse_points = {row.recon_method: row.n_inverse_points for row in case.summaries}
    assert inverse_points["raw_160"] == 160
    assert inverse_points["poly2_208"] == 208
    assert inverse_points["poly3_208"] == 208
    assert inverse_points["spline_208"] == 208
    assert len(case.field_rows) == 4 * case.model.sigma_true.size
    assert {row.recon_kind for row in case.structure_rows} == {
        "truth",
        "full_208",
        "raw_160",
        "poly2_208",
        "poly3_208",
        "spline_208",
    }
    for row in case.summaries:
        assert np.isfinite(row.recon_sigma_relative_rmse)
        assert np.isfinite(row.delta_sigma_effective_digits)


def test_holdout_fit_diff_plots_and_report(tmp_path) -> None:
    case = run_holdout_fit_diff(
        model=_surrogate_model(),
        fit_methods=["poly2"],
        raw_160_baseline=True,
        inverse_backend="least-squares",
        ridge=1e-4,
    )

    report = format_holdout_fit_report(case)
    curve = plot_holdout_fit_curves(case, tmp_path / "curves.png", dpi=80)
    recon = plot_holdout_recon_compare(case, tmp_path / "recon.png", dpi=80)
    summary = plot_holdout_fit_summary(case, tmp_path / "summary.png", dpi=80)

    assert "拟合" in report
    for path in [curve, recon, summary]:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 1000


def test_eit_holdout_fit_diff_cli_writes_expected_outputs(tmp_path) -> None:
    summary_output = tmp_path / "summary.csv"
    field_output = tmp_path / "fields.csv"
    point_output = tmp_path / "points.csv"
    structure_output = tmp_path / "structure.csv"
    report_output = tmp_path / "report.md"
    point_plot = tmp_path / "points.png"
    curve_plot = tmp_path / "curves.png"
    recon_plot = tmp_path / "recon.png"
    summary_plot = tmp_path / "summary.png"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_holdout_fit_diff_test.py",
            "--forward-backend",
            "surrogate",
            "--inverse-backend",
            "least-squares",
            "--fem-n-elec",
            "16",
            "--n-parameters",
            "6",
            "--model-seed",
            "123",
            "--ridge",
            "0.0001",
            "--raw-160-baseline",
            "--fit-methods",
            "poly2",
            "poly3",
            "spline",
            "--plot-voltage-points",
            "--plot-recon-compare",
            "--structure-metrics",
            "--output",
            str(summary_output),
            "--field-output",
            str(field_output),
            "--point-output",
            str(point_output),
            "--structure-output",
            str(structure_output),
            "--report-output",
            str(report_output),
            "--point-plot-output",
            str(point_plot),
            "--curve-plot-output",
            str(curve_plot),
            "--recon-plot-output",
            str(recon_plot),
            "--summary-plot-output",
            str(summary_plot),
            "--dpi",
            "80",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "raw_160=True" in completed.stdout
    with summary_output.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with field_output.open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))
    with point_output.open(newline="", encoding="utf-8") as handle:
        point_rows = list(csv.DictReader(handle))
    with structure_output.open(newline="", encoding="utf-8") as handle:
        structure_rows = list(csv.DictReader(handle))

    assert list(summary_rows[0].keys()) == SUMMARY_FIELDS
    assert {row["recon_method"] for row in summary_rows} == {
        "raw_160",
        "poly2_208",
        "poly3_208",
        "spline_208",
    }
    assert list(field_rows[0].keys()) == FIELD_FIELDS
    assert len(field_rows) == 24
    assert len(point_rows) == 256
    assert list(structure_rows[0].keys()) == STRUCTURE_FIELDS
    for path in [point_plot, curve_plot, recon_plot, summary_plot]:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_v36_fit_curves_keep_absolute_voltage_separate_from_diff() -> None:
    model = _surrogate_model()
    u_curve = np.array(
        [
            0.045,
            0.02,
            0.012,
            0.009,
            0.007,
            0.006,
            0.0058,
            0.006,
            0.007,
            0.009,
            0.012,
            0.02,
            0.045,
        ],
        dtype=float,
    )
    reference = np.tile(u_curve, 16)
    absolute_model = replace(
        model,
        voltage_reference=reference,
        voltage_true=reference + model.voltage_true,
    )

    case = run_holdout_fit_diff(
        model=absolute_model,
        fit_methods=["poly2"],
        raw_160_baseline=True,
        inverse_backend="least-squares",
        ridge=1e-4,
    )
    curve = case.frame_curves[0]

    np.testing.assert_allclose(curve.voltage_reference_full, reference[:13])
    np.testing.assert_allclose(
        curve.voltage_anomaly_full, reference[:13] + model.voltage_true[:13]
    )
    assert not np.allclose(curve.voltage_anomaly_full, curve.diff_full)
    assert "poly2" in curve.fitted_anomaly_by_method
