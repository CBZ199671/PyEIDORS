"""T81 phase 2b: byte-stable CSV fixtures for migrated sweep rows."""

from __future__ import annotations

import csv
from dataclasses import fields
from io import StringIO
from pathlib import Path

import pytest

from pyeidors.data.bucket_dense_experiments import (
    BUCKET_DENSE_FIELD_FIELDS,
    BUCKET_DENSE_SUMMARY_FIELDS,
    BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
    BucketDenseFieldRow,
    BucketDenseSummaryRow,
    BucketFull256CompareSummaryRow,
)
from pyeidors.data.factor_sweep import CSV_FIELDS, FactorSweepRow
from pyeidors.data.holdout_fit_diff import (
    FIELD_FIELDS,
    STRUCTURE_FIELDS,
    SUMMARY_FIELDS,
    HoldoutFitDiffFieldRow,
    HoldoutFitDiffSummary,
    HoldoutStructureMetricRow,
)
from pyeidors.data.voltage_digit_sweep import (
    VoltageDigitFieldRow,
    VoltageDigitSweepSummary,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "sweep_csv_columns"


def _render_csv(fieldnames: list[str], row) -> bytes:
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row.as_csv_row())
    return handle.getvalue().encode("utf-8")


def _field_order(row_type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(row_type))


GOLDEN_CASES = [
    pytest.param(
        "voltage_digit_summary.csv",
        [
            "target_voltage_digits",
            "achieved_voltage_effective_digits",
            "voltage_rmse",
            "sigma_rmse",
            "sigma_relative_rmse",
            "sigma_mae",
            "sigma_max_abs_error",
            "sigma_effective_digits",
        ],
        VoltageDigitSweepSummary(
            target_voltage_digits=5,
            achieved_voltage_effective_digits=4.75,
            voltage_rmse=0.001,
            sigma_rmse=0.002,
            sigma_relative_rmse=0.003,
            sigma_mae=0.004,
            sigma_max_abs_error=0.005,
            sigma_effective_digits=6.5,
        ),
        id="voltage-summary",
    ),
    pytest.param(
        "voltage_digit_field.csv",
        [
            "target_voltage_digits",
            "cell_index",
            "sigma_true",
            "sigma_recon",
            "sigma_error",
            "abs_sigma_error",
        ],
        VoltageDigitFieldRow(
            target_voltage_digits=5,
            cell_index=7,
            sigma_true=1.0,
            sigma_recon=0.95,
            sigma_error=-0.05,
            abs_sigma_error=0.05,
        ),
        id="voltage-field",
    ),
    pytest.param(
        "factor_sweep_row.csv",
        CSV_FIELDS,
        FactorSweepRow(
            sweep="baseline",
            changed_factor="baseline",
            level="baseline",
            n_elec=16,
            fem_grid=4,
            ridge=0.01,
            target_voltage_digits=6,
            enob="nominal",
            noise_relative=0.0,
            noser_exponent=0.5,
            n_measurements=208,
            voltage_rmse=0.001,
            achieved_voltage_effective_digits=7.5,
            sigma_rmse=0.002,
            sigma_relative_rmse=0.003,
            sigma_mae=0.004,
            sigma_max_abs_error=0.005,
            sigma_effective_digits=6.5,
        ),
        id="factor-row",
    ),
    pytest.param(
        "bucket_dense_summary.csv",
        BUCKET_DENSE_SUMMARY_FIELDS,
        BucketDenseSummaryRow(
            experiment="voltage_digit_sweep",
            domain="circle",
            mesh_h=0.16,
            n_cells=12,
            n_dofs=34,
            n_elec=16,
            n_measurements=208,
            ridge=0.01,
            recon_method="digits_5",
            target_voltage_digits=None,
            holdout_voltage_rmse=None,
            diff_voltage_rmse=None,
            sigma_rmse=0.1,
            sigma_relative_rmse=0.2,
            sigma_mae=0.3,
            sigma_max_abs_error=0.4,
            sigma_effective_digits=5.0,
            centroid_error=0.6,
            eccentricity=0.7,
            artifact_area=0.8,
            artifact_energy=0.9,
            artifact_peak=1.0,
        ),
        id="bucket-summary",
    ),
    pytest.param(
        "bucket_dense_field.csv",
        BUCKET_DENSE_FIELD_FIELDS,
        BucketDenseFieldRow(
            experiment="holdout_far3",
            recon_method="raw_160",
            cell_index=3,
            cell_x=0.1,
            cell_y=-0.2,
            sigma_true=1.0,
            sigma_recon=0.9,
            sigma_error=-0.1,
            inside_bucket=True,
        ),
        id="bucket-field",
    ),
    pytest.param(
        "bucket_full256_summary.csv",
        BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
        BucketFull256CompareSummaryRow(
            experiment="full256_compare",
            domain="circle",
            mesh_h=0.16,
            n_cells=12,
            n_dofs=34,
            n_elec=16,
            n_measurements=256,
            n_inverse_points=208,
            ridge=0.01,
            recon_method="full_256",
            delta_sigma_relative_rmse_vs_full_208=0.001,
            delta_artifact_energy_vs_full_208=0.002,
            delta_field_rmse_vs_full_208=0.003,
            delta_field_l2_vs_full_208=0.004,
            delta_field_max_abs_vs_full_208=0.005,
            sigma_rmse=0.1,
            sigma_relative_rmse=0.2,
            sigma_mae=0.3,
            sigma_max_abs_error=0.4,
            sigma_effective_digits=5.0,
            centroid_error=0.6,
            eccentricity=0.7,
            artifact_area=0.8,
            artifact_energy=0.9,
            artifact_peak=1.0,
        ),
        id="bucket-full256",
    ),
    pytest.param(
        "holdout_fit_summary.csv",
        SUMMARY_FIELDS,
        HoldoutFitDiffSummary(
            recon_method="raw_160",
            n_inverse_points=160,
            frame_count=16,
            points_per_frame=13,
            holdout_per_frame=3,
            train_points_per_frame=10,
            holdout_voltage_rmse=0.001,
            diff_voltage_rmse=0.002,
            full_sigma_rmse=0.003,
            recon_sigma_rmse=0.004,
            delta_sigma_rmse=0.005,
            full_sigma_relative_rmse=0.006,
            recon_sigma_relative_rmse=0.007,
            delta_sigma_relative_rmse=0.008,
            full_sigma_effective_digits=5.1,
            recon_sigma_effective_digits=5.2,
            delta_sigma_effective_digits=5.3,
        ),
        id="holdout-summary",
    ),
    pytest.param(
        "holdout_fit_field.csv",
        FIELD_FIELDS,
        HoldoutFitDiffFieldRow(
            recon_method="raw_160",
            cell_index=4,
            sigma_true=1.0,
            sigma_recon_full=0.98,
            sigma_recon_candidate=0.97,
            sigma_error_full=-0.02,
            sigma_error_candidate=-0.03,
            delta_sigma_error=-0.01,
        ),
        id="holdout-field",
    ),
    pytest.param(
        "holdout_structure.csv",
        STRUCTURE_FIELDS,
        HoldoutStructureMetricRow(
            recon_kind="raw_160",
            threshold_rule="top20",
            centroid_x=0.1,
            centroid_y=0.2,
            centroid_error=0.3,
            equivalent_area=0.4,
            eccentricity=0.5,
            major_axis=0.6,
            minor_axis=0.7,
            artifact_area=0.8,
            artifact_energy=0.9,
            artifact_peak=1.0,
            sigma_rmse=0.01,
            sigma_relative_rmse=0.02,
            sigma_mae=0.03,
            sigma_max_abs_error=0.04,
            sigma_effective_digits=6.0,
        ),
        id="holdout-structure",
    ),
]


@pytest.mark.parametrize(("fixture_name", "fieldnames", "row"), GOLDEN_CASES)
def test_migrated_sweep_rows_match_csv_golden_fixture(
    fixture_name: str,
    fieldnames: list[str],
    row,
) -> None:
    assert tuple(fieldnames) == _field_order(type(row))
    assert _render_csv(fieldnames, row) == (FIXTURE_DIR / fixture_name).read_bytes()
