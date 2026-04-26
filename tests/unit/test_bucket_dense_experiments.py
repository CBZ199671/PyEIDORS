from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from pyeidors.data.bucket_dense_experiments import (
    BUCKET_DENSE_FIELD_FIELDS,
    BUCKET_DENSE_SUMMARY_FIELDS,
    BUCKET_FULL256_COMPARE_FIELD_FIELDS,
    BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
    build_circle_bucket_linearized_model,
    run_bucket_full256_compare_experiment,
    run_bucket_dense_experiments,
    write_bucket_full256_compare_outputs,
    write_bucket_dense_outputs,
)
from pyeidors.data.bucket_domain_audit import build_circle_bucket_domain
from pyeidors.data.eit_digit_metrics import reconstruct_linearized_sigma


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_circle_bucket_linearized_model_uses_dense_circle_geometry() -> None:
    bucket = build_circle_bucket_domain(mesh_h=0.1, n_elec=16)
    model = build_circle_bucket_linearized_model(bucket=bucket)

    assert bucket.n_cells >= 800
    assert model.label == "circle_bucket_dense"
    assert model.n_elec == 16
    assert model.n_measurements == 208
    assert model.sensitivity.shape == (208, bucket.n_cells)
    assert model.mesh_cells is not None
    assert model.mesh_cells.shape[0] == bucket.n_cells
    np.testing.assert_allclose(model.parameter_points, bucket.cell_centers)
    np.testing.assert_allclose(model.mesh_points, bucket.nodes)
    assert np.max(np.linalg.norm(model.parameter_points, axis=1)) <= 1.0 + 1e-10
    assert np.max(np.abs(model.voltage_true)) > 0.0
    assert np.max(np.abs(model.voltage_reference)) > 0.0


def test_v40_full256_circle_bucket_model_keeps_drive_related_points() -> None:
    bucket = build_circle_bucket_domain(
        mesh_h=0.16,
        n_elec=16,
        allow_coarse_smoke=True,
    )
    model_208 = build_circle_bucket_linearized_model(bucket=bucket)
    model_256 = build_circle_bucket_linearized_model(
        bucket=bucket,
        include_drive_related=True,
    )

    assert model_208.label == "circle_bucket_dense"
    assert model_208.n_measurements == 208
    assert model_208.sensitivity.shape == (208, bucket.n_cells)
    assert model_256.label == "circle_bucket_dense_full_256"
    assert model_256.n_measurements == 256
    assert model_256.sensitivity.shape == (256, bucket.n_cells)
    assert model_256.voltage_true.shape == (256,)
    assert np.max(np.abs(model_256.voltage_reference[:16])) > 0.0


def test_v36_circle_bucket_reference_voltage_is_rotated_local_u_curve() -> None:
    bucket = build_circle_bucket_domain(mesh_h=0.16, n_elec=16, allow_coarse_smoke=True)
    model = build_circle_bucket_linearized_model(bucket=bucket)

    reference_frames = model.voltage_reference.reshape(16, 13)
    np.testing.assert_allclose(reference_frames[0], reference_frames[1])
    np.testing.assert_allclose(reference_frames[0], reference_frames[8])
    assert reference_frames[0][0] > reference_frames[0][6]
    assert reference_frames[0][-1] > reference_frames[0][6]
    assert np.max(np.abs(model.voltage_true - model.voltage_reference)) < np.max(
        np.abs(model.voltage_reference)
    )


def test_measurement_rm_backend_reconstructs_dense_bucket_without_param_solve() -> None:
    bucket = build_circle_bucket_domain(
        mesh_h=0.16,
        n_elec=16,
        allow_coarse_smoke=True,
    )
    model = build_circle_bucket_linearized_model(bucket=bucket)
    sigma_recon = reconstruct_linearized_sigma(
        model=model,
        voltages=model.voltage_true,
        ridge=0.01,
        inverse_backend="measurement-rm",
    )

    assert sigma_recon.shape == bucket.sigma_true.shape
    assert np.all(np.isfinite(sigma_recon))
    assert np.std(sigma_recon) > 0.0


def test_v37_bucket_dense_experiment_outputs_and_recovers_visible_anomaly(
    tmp_path,
) -> None:
    case = run_bucket_dense_experiments(
        mesh_h=0.16,
        target_digits=[5],
        fit_methods=["poly2"],
        raw_160_baseline=True,
        allow_coarse_smoke=True,
    )
    written = write_bucket_dense_outputs(
        case,
        summary_output=tmp_path / "summary.csv",
        field_output=tmp_path / "fields.csv",
        report_output=tmp_path / "report.md",
        domain_plot_output=tmp_path / "domain.png",
        recon_plot_output=tmp_path / "recon.png",
        summary_plot_output=tmp_path / "summary.png",
        curve_plot_output=tmp_path / "curves.png",
        holdout_summary_plot_output=tmp_path / "holdout_summary.png",
        coarse_voltage_csv=None,
        coarse_holdout_csv=None,
        coarse_structure_csv=None,
        dpi=70,
    )
    reference = case.model.sigma_reference
    truth_contrast = float(np.max(case.bucket.sigma_true - reference))
    full_contrast = float(np.max(case.holdout_case.sigma_recon_full - reference))

    assert {row.experiment for row in case.summaries} == {
        "voltage_digit_sweep",
        "holdout_far3",
    }
    assert full_contrast >= 0.6 * truth_contrast
    assert {row.recon_method for row in case.summaries} == {
        "digits_5",
        "full_208",
        "raw_160",
        "poly2_208",
    }
    assert len(case.field_rows) == 4 * case.bucket.n_cells
    with written["summary"].open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with written["fields"].open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))

    assert list(summary_rows[0].keys()) == BUCKET_DENSE_SUMMARY_FIELDS
    assert list(field_rows[0].keys()) == BUCKET_DENSE_FIELD_FIELDS
    assert (
        written["report"]
        .read_text(encoding="utf-8")
        .startswith("# T23 密集圆形小水桶复测报告")
    )
    for key in [
        "domain_plot",
        "recon_plot",
        "summary_plot",
        "curve_plot",
        "holdout_summary_plot",
    ]:
        assert written[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_v41_full256_compare_outputs_numeric_and_visual_differences(
    tmp_path,
) -> None:
    case = run_bucket_full256_compare_experiment(
        mesh_h=0.16,
        fit_methods=["poly2"],
        raw_160_baseline=True,
        allow_coarse_smoke=True,
    )
    written = write_bucket_full256_compare_outputs(
        case,
        summary_output=tmp_path / "summary.csv",
        field_output=tmp_path / "fields.csv",
        report_output=tmp_path / "report.md",
        recon_plot_output=tmp_path / "recon.png",
        metrics_plot_output=tmp_path / "metrics.png",
        point_audit_plot_output=tmp_path / "point_audit.png",
        recon_delta_plot_output=tmp_path / "recon_delta.png",
        dpi=70,
    )

    assert case.model_full_256.n_measurements == 256
    assert case.model_full_208.n_measurements == 208
    assert {row.recon_method for row in case.summaries} == {
        "full_256",
        "full_208",
        "far3_drop_near3_keep_208",
        "raw_160",
        "poly2_208",
    }
    inverse_points = {row.recon_method: row.n_inverse_points for row in case.summaries}
    assert inverse_points["full_256"] == 256
    assert inverse_points["full_208"] == 208
    assert inverse_points["far3_drop_near3_keep_208"] == 208
    assert inverse_points["raw_160"] == 160
    assert inverse_points["poly2_208"] == 208
    baseline = next(row for row in case.summaries if row.recon_method == "full_208")
    assert baseline.delta_sigma_relative_rmse_vs_full_208 == 0.0
    assert baseline.delta_artifact_energy_vs_full_208 == 0.0
    assert baseline.delta_field_l2_vs_full_208 == 0.0
    assert len(case.field_rows) == 5 * case.bucket.n_cells

    with written["summary"].open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with written["fields"].open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))

    assert list(summary_rows[0].keys()) == BUCKET_FULL256_COMPARE_SUMMARY_FIELDS
    assert list(field_rows[0].keys()) == BUCKET_FULL256_COMPARE_FIELD_FIELDS
    assert "full_256" in written["report"].read_text(encoding="utf-8")
    for key in ["recon_plot", "metrics_plot", "point_audit_plot", "recon_delta_plot"]:
        assert written[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_eit_bucket_dense_experiments_cli_writes_expected_outputs(tmp_path) -> None:
    summary_output = tmp_path / "summary.csv"
    field_output = tmp_path / "fields.csv"
    report_output = tmp_path / "report.md"
    domain_plot = tmp_path / "domain.png"
    recon_plot = tmp_path / "recon.png"
    summary_plot = tmp_path / "summary.png"
    curve_plot = tmp_path / "curves.png"
    holdout_summary = tmp_path / "holdout_summary.png"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_bucket_dense_experiments.py",
            "--mesh-h",
            "0.16",
            "--allow-coarse-smoke",
            "--target-digits",
            "5",
            "--raw-160-baseline",
            "--fit-methods",
            "poly2",
            "--output",
            str(summary_output),
            "--field-output",
            str(field_output),
            "--report-output",
            str(report_output),
            "--domain-plot-output",
            str(domain_plot),
            "--recon-plot-output",
            str(recon_plot),
            "--summary-plot-output",
            str(summary_plot),
            "--curve-plot-output",
            str(curve_plot),
            "--holdout-summary-plot-output",
            str(holdout_summary),
            "--coarse-voltage-csv",
            "none",
            "--coarse-holdout-csv",
            "none",
            "--coarse-structure-csv",
            "none",
            "--dpi",
            "70",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "domain=circle_bucket" in completed.stdout
    assert "n_measurements=208" in completed.stdout
    with summary_output.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with field_output.open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))

    assert list(summary_rows[0].keys()) == BUCKET_DENSE_SUMMARY_FIELDS
    assert {row["recon_method"] for row in summary_rows} == {
        "digits_5",
        "full_208",
        "raw_160",
        "poly2_208",
    }
    assert list(field_rows[0].keys()) == BUCKET_DENSE_FIELD_FIELDS
    assert {row["inside_bucket"] for row in field_rows} == {"true"}
    for path in [domain_plot, recon_plot, summary_plot, curve_plot, holdout_summary]:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_eit_bucket_full256_compare_cli_writes_expected_outputs(tmp_path) -> None:
    summary_output = tmp_path / "summary.csv"
    field_output = tmp_path / "fields.csv"
    report_output = tmp_path / "report.md"
    recon_plot = tmp_path / "recon.png"
    recon_delta_plot = tmp_path / "recon_delta.png"
    metrics_plot = tmp_path / "metrics.png"
    point_audit_plot = tmp_path / "point_audit.png"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_bucket_full256_compare.py",
            "--mesh-h",
            "0.16",
            "--allow-coarse-smoke",
            "--raw-160-baseline",
            "--fit-methods",
            "poly2",
            "--output",
            str(summary_output),
            "--field-output",
            str(field_output),
            "--report-output",
            str(report_output),
            "--recon-plot-output",
            str(recon_plot),
            "--recon-delta-plot-output",
            str(recon_delta_plot),
            "--metrics-plot-output",
            str(metrics_plot),
            "--point-audit-plot-output",
            str(point_audit_plot),
            "--dpi",
            "70",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "full_256_measurements=256" in completed.stdout
    assert "full_208_measurements=208" in completed.stdout
    with summary_output.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with field_output.open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))

    assert list(summary_rows[0].keys()) == BUCKET_FULL256_COMPARE_SUMMARY_FIELDS
    assert {row["recon_method"] for row in summary_rows} == {
        "full_256",
        "full_208",
        "far3_drop_near3_keep_208",
        "raw_160",
        "poly2_208",
    }
    assert list(field_rows[0].keys()) == BUCKET_FULL256_COMPARE_FIELD_FIELDS
    assert {row["inside_bucket"] for row in field_rows} == {"true"}
    for path in [recon_plot, recon_delta_plot, metrics_plot, point_audit_plot]:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_eit_bucket_all_modes_noise_sweep_cli_writes_expected_outputs(
    tmp_path,
) -> None:
    output_dir = tmp_path / "noise_sweep"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_bucket_all_modes_noise_sweep.py",
            "--mesh-h",
            "0.16",
            "--allow-coarse-smoke",
            "--fit-methods",
            "poly2",
            "--snr-levels",
            "inf",
            "10",
            "--output-dir",
            str(output_dir),
            "--dpi",
            "70",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    summary = output_dir / "eit_bucket_all_modes_noise_sweep_summary_16e.csv"
    fields = output_dir / "eit_bucket_all_modes_noise_sweep_fields_16e.csv"
    report = output_dir / "eit_bucket_all_modes_noise_sweep_summary_16e.md"
    metrics = output_dir / "eit_bucket_all_modes_noise_sweep_metrics_16e.png"
    recon = output_dir / "eit_bucket_all_modes_noise_sweep_recon_grid_16e.png"
    errors = output_dir / "eit_bucket_all_modes_noise_sweep_error_grid_16e.png"

    assert "SNR=inf" in completed.stdout
    assert "SNR=10" in completed.stdout
    with summary.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    with fields.open(newline="", encoding="utf-8") as handle:
        field_rows = list(csv.DictReader(handle))

    assert summary_rows[0]["actual_snr"]
    assert {row["noise_snr_requested"] for row in summary_rows} == {"inf", "10"}
    assert {row["recon_method"] for row in summary_rows} == {
        "full_256",
        "full_208",
        "far3_drop_near3_keep_208",
        "raw_160",
        "poly2_208",
    }
    assert field_rows[0]["noise_seed"]
    assert "add_noise" in report.read_text(encoding="utf-8")
    for path in [metrics, recon, errors]:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
