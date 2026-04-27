"""T81 phase 2d: shared sweep schema base contracts."""

from __future__ import annotations

from dataclasses import fields

from pyeidors.data._sweep_core import (
    RECON_METRIC_FIELDS,
    STRUCTURE_METRIC_FIELDS,
    STRUCTURE_SUMMARY_METRIC_FIELDS,
    ReconMetricRow,
    StructureMetricRow,
    StructureMetrics,
    SweepRow,
)
import pyeidors.data.bucket_dense_experiments as bucket_dense_module
from pyeidors.data.bucket_dense_experiments import (
    BUCKET_DENSE_FIELD_FIELDS,
    BUCKET_DENSE_SUMMARY_FIELDS,
    BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
    BucketDenseFieldRow,
    BucketDenseSummaryRow,
    BucketFull256CompareSummaryRow,
)
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


def _field_order(row_type) -> tuple[str, ...]:
    return tuple(field.name for field in fields(row_type))


def test_recon_metric_rows_share_zero_field_base_without_reordering() -> None:
    assert issubclass(VoltageDigitSweepSummary, ReconMetricRow)
    assert issubclass(HoldoutFitDiffSummary, ReconMetricRow)
    assert issubclass(BucketDenseSummaryRow, StructureMetricRow)
    assert issubclass(BucketFull256CompareSummaryRow, StructureMetricRow)

    assert _field_order(VoltageDigitSweepSummary) == (
        "target_voltage_digits",
        "achieved_voltage_effective_digits",
        "voltage_rmse",
        *RECON_METRIC_FIELDS,
    )
    assert _field_order(HoldoutFitDiffSummary) == tuple(SUMMARY_FIELDS)
    assert _field_order(BucketDenseSummaryRow) == tuple(BUCKET_DENSE_SUMMARY_FIELDS)
    assert _field_order(BucketFull256CompareSummaryRow) == tuple(
        BUCKET_FULL256_COMPARE_SUMMARY_FIELDS
    )


def test_structure_metrics_shared_value_object_replaces_local_duplicate() -> None:
    assert bucket_dense_module._StructureMetrics is StructureMetrics
    assert issubclass(HoldoutStructureMetricRow, StructureMetricRow)
    assert _field_order(StructureMetrics) == tuple(STRUCTURE_METRIC_FIELDS)
    assert _field_order(HoldoutStructureMetricRow) == tuple(STRUCTURE_FIELDS)


def test_field_rows_use_shared_sweep_row_serializer_without_schema_changes() -> None:
    assert issubclass(VoltageDigitFieldRow, SweepRow)
    assert issubclass(BucketDenseFieldRow, SweepRow)
    assert issubclass(HoldoutFitDiffFieldRow, SweepRow)

    assert _field_order(VoltageDigitFieldRow) == (
        "target_voltage_digits",
        "cell_index",
        "sigma_true",
        "sigma_recon",
        "sigma_error",
        "abs_sigma_error",
    )
    assert _field_order(BucketDenseFieldRow) == tuple(BUCKET_DENSE_FIELD_FIELDS)
    assert _field_order(HoldoutFitDiffFieldRow) == tuple(FIELD_FIELDS)


def test_shared_metric_views_are_explicit_per_row_family() -> None:
    voltage = VoltageDigitSweepSummary(
        target_voltage_digits=5,
        achieved_voltage_effective_digits=4.75,
        voltage_rmse=0.001,
        sigma_rmse=0.002,
        sigma_relative_rmse=0.003,
        sigma_mae=0.004,
        sigma_max_abs_error=0.005,
        sigma_effective_digits=6.5,
    )
    bucket = BucketDenseSummaryRow(
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
    )
    holdout = HoldoutFitDiffSummary(
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
    )

    assert tuple(voltage.recon_metric_values()) == RECON_METRIC_FIELDS
    assert tuple(bucket.structure_metric_values()) == STRUCTURE_SUMMARY_METRIC_FIELDS
    assert holdout.recon_metric_values() == {
        "recon_sigma_rmse": 0.004,
        "recon_sigma_relative_rmse": 0.007,
        "recon_sigma_effective_digits": 5.2,
    }
