"""Dense circular-bucket EIT voltage and holdout experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .adc_quantization import effective_digits_from_rmse, rmse
from .bucket_domain_audit import (
    CircleBucketDomain,
    build_circle_bucket_domain,
    plot_bucket_domain_audit,
)
from .eit_digit_metrics import ADJACENT_PATTERN, EITLinearizedModel
from .holdout_fit_diff import (
    HoldoutFitDiffCase,
    HoldoutStructureMetricRow,
    plot_holdout_fit_curves,
    plot_holdout_fit_summary,
    run_holdout_fit_diff,
)
from .holdout_point_audit import build_holdout_point_audit, plot_holdout_point_audit
from ._sweep_core import (
    StructureMetricRow,
    STRUCTURE_SUMMARY_METRIC_FIELDS,
    StructureMetrics as _StructureMetrics,
    SweepRow,
    write_csv_rows,
    write_sweep_table_artifacts,
)
from .voltage_digit_sweep import keep_significant_digits


BUCKET_DENSE_SUMMARY_FIELDS = [
    "experiment",
    "domain",
    "mesh_h",
    "n_cells",
    "n_dofs",
    "n_elec",
    "n_measurements",
    "ridge",
    "recon_method",
    "target_voltage_digits",
    "holdout_voltage_rmse",
    "diff_voltage_rmse",
    "sigma_rmse",
    "sigma_relative_rmse",
    "sigma_mae",
    "sigma_max_abs_error",
    "sigma_effective_digits",
    "centroid_error",
    "eccentricity",
    "artifact_area",
    "artifact_energy",
    "artifact_peak",
]

BUCKET_DENSE_FIELD_FIELDS = [
    "experiment",
    "recon_method",
    "cell_index",
    "cell_x",
    "cell_y",
    "sigma_true",
    "sigma_recon",
    "sigma_error",
    "inside_bucket",
]

BUCKET_FULL256_COMPARE_SUMMARY_FIELDS = [
    "experiment",
    "domain",
    "mesh_h",
    "n_cells",
    "n_dofs",
    "n_elec",
    "n_measurements",
    "n_inverse_points",
    "ridge",
    "recon_method",
    "delta_sigma_relative_rmse_vs_full_208",
    "delta_artifact_energy_vs_full_208",
    "delta_field_rmse_vs_full_208",
    "delta_field_l2_vs_full_208",
    "delta_field_max_abs_vs_full_208",
    "sigma_rmse",
    "sigma_relative_rmse",
    "sigma_mae",
    "sigma_max_abs_error",
    "sigma_effective_digits",
    "centroid_error",
    "eccentricity",
    "artifact_area",
    "artifact_energy",
    "artifact_peak",
]

BUCKET_FULL256_COMPARE_FIELD_FIELDS = BUCKET_DENSE_FIELD_FIELDS


@dataclass(frozen=True)
class BucketDenseSummaryRow(StructureMetricRow):
    """One dense-bucket summary row for voltage or holdout experiments."""

    structure_metric_fields = STRUCTURE_SUMMARY_METRIC_FIELDS

    experiment: str
    domain: str
    mesh_h: float
    n_cells: int
    n_dofs: int
    n_elec: int
    n_measurements: int
    ridge: float
    recon_method: str
    target_voltage_digits: int | None
    holdout_voltage_rmse: float | None
    diff_voltage_rmse: float | None
    sigma_rmse: float
    sigma_relative_rmse: float
    sigma_mae: float
    sigma_max_abs_error: float
    sigma_effective_digits: float
    centroid_error: float
    eccentricity: float
    artifact_area: float
    artifact_energy: float
    artifact_peak: float


@dataclass(frozen=True)
class BucketDenseFieldRow(SweepRow):
    """One per-cell dense-bucket reconstructed conductivity row."""

    experiment: str
    recon_method: str
    cell_index: int
    cell_x: float
    cell_y: float
    sigma_true: float
    sigma_recon: float
    sigma_error: float
    inside_bucket: bool


@dataclass(frozen=True)
class BucketFull256CompareSummaryRow(StructureMetricRow):
    """One full-256-vs-filtered reconstruction comparison row."""

    structure_metric_fields = STRUCTURE_SUMMARY_METRIC_FIELDS

    experiment: str
    domain: str
    mesh_h: float
    n_cells: int
    n_dofs: int
    n_elec: int
    n_measurements: int
    n_inverse_points: int
    ridge: float
    recon_method: str
    delta_sigma_relative_rmse_vs_full_208: float
    delta_artifact_energy_vs_full_208: float
    delta_field_rmse_vs_full_208: float
    delta_field_l2_vs_full_208: float
    delta_field_max_abs_vs_full_208: float
    sigma_rmse: float
    sigma_relative_rmse: float
    sigma_mae: float
    sigma_max_abs_error: float
    sigma_effective_digits: float
    centroid_error: float
    eccentricity: float
    artifact_area: float
    artifact_energy: float
    artifact_peak: float


@dataclass(frozen=True)
class BucketDenseExperimentCase:
    """Full dense bucket experiment bundle."""

    bucket: CircleBucketDomain
    model: EITLinearizedModel
    summaries: list[BucketDenseSummaryRow]
    field_rows: list[BucketDenseFieldRow]
    voltage_recon_by_method: dict[str, np.ndarray]
    holdout_case: HoldoutFitDiffCase


@dataclass(frozen=True)
class BucketFull256CompareCase:
    """Full 256-point comparison against native 208, raw 160, and fitted 208."""

    bucket: CircleBucketDomain
    model_full_256: EITLinearizedModel
    model_full_208: EITLinearizedModel
    holdout_case: HoldoutFitDiffCase
    summaries: list[BucketFull256CompareSummaryRow]
    field_rows: list[BucketDenseFieldRow]
    sigma_recon_by_method: dict[str, np.ndarray]


def _relative_rmse(reference: np.ndarray, observed: np.ndarray) -> float:
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    if ref_rms == 0.0:
        return math.nan
    return rmse(reference, observed) / ref_rms


def _electrode_center_points(bucket: CircleBucketDomain) -> np.ndarray:
    angles = np.radians([item.center_angle_deg for item in bucket.electrodes])
    return bucket.bucket_radius * np.column_stack([np.cos(angles), np.sin(angles)])


def _source_gradient(
    points: np.ndarray,
    electrode_point: np.ndarray,
    *,
    softening: float,
) -> np.ndarray:
    diff = np.asarray(points, dtype=float) - np.asarray(electrode_point, dtype=float)
    r2 = np.sum(diff * diff, axis=1) + float(softening) ** 2
    return diff / (2.0 * math.pi * r2[:, None])


def _source_potential(
    points: np.ndarray,
    electrode_point: np.ndarray,
    *,
    softening: float,
) -> np.ndarray:
    diff = np.asarray(points, dtype=float) - np.asarray(electrode_point, dtype=float)
    r2 = np.sum(diff * diff, axis=1) + float(softening) ** 2
    return 0.5 * np.log(r2) / (2.0 * math.pi)


def _pair_gradient(
    points: np.ndarray,
    electrode_points: np.ndarray,
    *,
    e1: int,
    e2: int,
    softening: float,
) -> np.ndarray:
    return _source_gradient(
        points,
        electrode_points[int(e1) % electrode_points.shape[0]],
        softening=softening,
    ) - _source_gradient(
        points,
        electrode_points[int(e2) % electrode_points.shape[0]],
        softening=softening,
    )


def _pair_potential(
    points: np.ndarray,
    electrode_points: np.ndarray,
    *,
    e1: int,
    e2: int,
    softening: float,
) -> np.ndarray:
    return _source_potential(
        points,
        electrode_points[int(e1) % electrode_points.shape[0]],
        softening=softening,
    ) - _source_potential(
        points,
        electrode_points[int(e2) % electrode_points.shape[0]],
        softening=softening,
    )


def _circle_bucket_measurement_rows(
    bucket: CircleBucketDomain,
    *,
    include_drive_related: bool,
):
    point_rows, summary = build_holdout_point_audit(n_elec=bucket.n_elec)
    if include_drive_related:
        rows = list(point_rows)
        expected = summary.full_candidate_count
    else:
        rows = [row for row in point_rows if row.point_status != "drive_removed"]
        expected = summary.kept_208_count
    if len(rows) != expected:
        raise RuntimeError("adjacent measurement count mismatch")
    return rows, expected


def _build_circle_bucket_reference_voltage(
    bucket: CircleBucketDomain,
    *,
    include_drive_related: bool = False,
) -> np.ndarray:
    measurement_rows, expected_count = _circle_bucket_measurement_rows(
        bucket,
        include_drive_related=include_drive_related,
    )
    electrodes = _electrode_center_points(bucket)
    softening = max(bucket.mesh_h * 0.2, bucket.bucket_radius * 1e-3)
    rows: list[float] = []
    for row in measurement_rows:
        electrode_voltage = _pair_potential(
            electrodes,
            electrodes,
            e1=row.stim_e1,
            e2=row.stim_e2,
            softening=softening,
        )
        rows.append(
            float(
                (
                    electrode_voltage[int(row.meas_e1) % bucket.n_elec]
                    - electrode_voltage[int(row.meas_e2) % bucket.n_elec]
                )
                / bucket.background_conductivity
            )
        )
    voltage = np.asarray(rows, dtype=float)
    if voltage.shape != (expected_count,):
        raise RuntimeError("circle bucket reference voltage shape mismatch")
    if not np.all(np.isfinite(voltage)):
        raise RuntimeError("circle bucket reference voltage contains non-finite values")
    return voltage


def _build_circle_bucket_sensitivity(
    bucket: CircleBucketDomain,
    *,
    normalize_rows: bool = False,
    include_drive_related: bool = False,
) -> np.ndarray:
    measurement_rows, expected_count = _circle_bucket_measurement_rows(
        bucket,
        include_drive_related=include_drive_related,
    )

    centers = bucket.cell_centers
    areas = bucket.cell_areas
    electrodes = _electrode_center_points(bucket)
    softening = max(bucket.mesh_h * 0.75, bucket.bucket_radius * 1e-3)
    rows: list[np.ndarray] = []
    for row in measurement_rows:
        stim_grad = _pair_gradient(
            centers,
            electrodes,
            e1=row.stim_e1,
            e2=row.stim_e2,
            softening=softening,
        )
        meas_grad = _pair_gradient(
            centers,
            electrodes,
            e1=row.meas_e1,
            e2=row.meas_e2,
            softening=softening,
        )
        sensitivity_row = -np.einsum("ij,ij->i", stim_grad, meas_grad) * areas
        rows.append(sensitivity_row)
    sensitivity = np.vstack(rows).astype(float)
    if normalize_rows:
        scales = np.linalg.norm(sensitivity, axis=1)
        good = scales > 0.0
        sensitivity[good, :] = sensitivity[good, :] / scales[good, None]
    if sensitivity.shape != (expected_count, bucket.n_dofs):
        raise RuntimeError("circle bucket sensitivity shape mismatch")
    if not np.all(np.isfinite(sensitivity)):
        raise RuntimeError("circle bucket sensitivity contains non-finite values")
    return sensitivity


def build_circle_bucket_linearized_model(
    *,
    bucket: CircleBucketDomain,
    normalize_rows: bool = False,
    include_drive_related: bool = False,
) -> EITLinearizedModel:
    """Build a dense circular-bucket linearized difference model."""

    sigma_reference = np.full(
        bucket.n_dofs,
        bucket.background_conductivity,
        dtype=float,
    )
    sensitivity = _build_circle_bucket_sensitivity(
        bucket,
        normalize_rows=normalize_rows,
        include_drive_related=include_drive_related,
    )
    contrast = bucket.sigma_true - sigma_reference
    voltage_reference = _build_circle_bucket_reference_voltage(
        bucket,
        include_drive_related=include_drive_related,
    )
    voltage_true = voltage_reference + sensitivity @ contrast

    def forward_solver(sigma: np.ndarray) -> np.ndarray:
        sigma_vec = np.asarray(sigma, dtype=float)
        if sigma_vec.shape != sigma_reference.shape:
            raise ValueError("sigma shape must match circle bucket dofs")
        return voltage_reference + sensitivity @ (sigma_vec - sigma_reference)

    return EITLinearizedModel(
        sigma_true=bucket.sigma_true.copy(),
        sigma_reference=sigma_reference,
        voltage_true=voltage_true,
        voltage_reference=voltage_reference,
        sensitivity=sensitivity,
        label="circle_bucket_dense_full_256"
        if include_drive_related
        else "circle_bucket_dense",
        n_elec=bucket.n_elec,
        stim_pattern=ADJACENT_PATTERN,
        meas_pattern=ADJACENT_PATTERN,
        n_measurements=int(voltage_true.size),
        parameter_points=bucket.cell_centers.copy(),
        mesh_points=bucket.nodes.copy(),
        mesh_cells=bucket.cells.copy(),
        forward_solver=forward_solver,
    )


def _measurement_submodel(
    model: EITLinearizedModel,
    indices: np.ndarray,
    *,
    label: str,
) -> EITLinearizedModel:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("indices must be a non-empty 1D vector")
    return replace(
        model,
        voltage_true=np.asarray(model.voltage_true, dtype=float)[idx],
        voltage_reference=np.asarray(model.voltage_reference, dtype=float)[idx],
        sensitivity=np.asarray(model.sensitivity, dtype=float)[idx, :],
        label=label,
        n_measurements=int(idx.size),
    )


def _far3_drop_near3_keep_indices(n_elec: int) -> np.ndarray:
    point_rows, point_summary = build_holdout_point_audit(n_elec=int(n_elec))
    indices = [
        row.global_index_256 for row in point_rows if row.point_status != "holdout_far3"
    ]
    expected = point_summary.full_candidate_count - point_summary.holdout_far3_count
    if len(indices) != expected:
        raise RuntimeError("far3-drop/near3-keep point count mismatch")
    return np.asarray(indices, dtype=np.int64)


def _weighted_structure(
    *,
    values: np.ndarray,
    points: np.ndarray,
    areas: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    weights_raw = np.abs(values)
    mask = weights_raw >= threshold
    if not np.any(mask):
        mask[int(np.argmax(weights_raw))] = True
    weights = weights_raw[mask] * areas[mask]
    if float(np.sum(weights)) <= 0.0:
        weights = areas[mask]
    coords = points[mask, :2]
    weight_sum = float(np.sum(weights))
    centroid = np.sum(coords * weights[:, None], axis=0) / weight_sum
    centered = coords - centroid
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    eigvals = np.sort(np.linalg.eigvalsh(covariance))
    minor_var = max(float(eigvals[0]), 0.0)
    major_var = max(float(eigvals[-1]), 0.0)
    eccentricity = (
        0.0 if major_var <= 0.0 else math.sqrt(max(0.0, 1.0 - minor_var / major_var))
    )
    major_axis = 4.0 * math.sqrt(major_var)
    minor_axis = 4.0 * math.sqrt(minor_var)
    equivalent_area = float(np.sum(areas[mask]))
    return (
        mask,
        float(centroid[0]),
        float(centroid[1]),
        equivalent_area,
        eccentricity,
        major_axis,
        minor_axis,
    )


def _structure_metrics(
    *,
    bucket: CircleBucketDomain,
    sigma_recon: np.ndarray,
) -> _StructureMetrics:
    sigma_true = bucket.sigma_true
    sigma_ref = np.full_like(sigma_true, bucket.background_conductivity)
    points = bucket.cell_centers
    areas = bucket.cell_areas
    truth_contrast = sigma_true - sigma_ref
    max_truth = float(np.max(np.abs(truth_contrast)))
    threshold = max(0.5 * max_truth, 1e-12)
    truth_mask, truth_x, truth_y, *_ = _weighted_structure(
        values=truth_contrast,
        points=points,
        areas=areas,
        threshold=threshold,
    )
    contrast = np.asarray(sigma_recon, dtype=float) - sigma_ref
    mask, cx, cy, area, ecc, major, minor = _weighted_structure(
        values=contrast,
        points=points,
        areas=areas,
        threshold=threshold,
    )
    outside = ~truth_mask
    artifact_values = np.abs(contrast[outside])
    artifact_active = mask & outside
    artifact_area = float(np.sum(areas[artifact_active]))
    artifact_energy = float(np.sum((contrast[outside] ** 2) * areas[outside]))
    artifact_peak = float(np.max(artifact_values)) if artifact_values.size else 0.0
    error = np.asarray(sigma_recon, dtype=float) - sigma_true
    abs_error = np.abs(error)
    return _StructureMetrics(
        centroid_error=math.hypot(cx - truth_x, cy - truth_y),
        equivalent_area=area,
        eccentricity=ecc,
        major_axis=major,
        minor_axis=minor,
        artifact_area=artifact_area,
        artifact_energy=artifact_energy,
        artifact_peak=artifact_peak,
        sigma_rmse=rmse(sigma_true, sigma_recon),
        sigma_relative_rmse=_relative_rmse(sigma_true, sigma_recon),
        sigma_mae=float(np.mean(abs_error)),
        sigma_max_abs_error=float(np.max(abs_error)),
        sigma_effective_digits=effective_digits_from_rmse(sigma_true, sigma_recon),
    )


def _summary_from_metrics(
    *,
    experiment: str,
    bucket: CircleBucketDomain,
    ridge: float,
    recon_method: str,
    target_voltage_digits: int | None,
    holdout_voltage_rmse: float | None,
    diff_voltage_rmse: float | None,
    metrics: _StructureMetrics,
) -> BucketDenseSummaryRow:
    return BucketDenseSummaryRow(
        experiment=experiment,
        domain=bucket.domain,
        mesh_h=bucket.mesh_h,
        n_cells=bucket.n_cells,
        n_dofs=bucket.n_dofs,
        n_elec=bucket.n_elec,
        n_measurements=bucket.n_measurements,
        ridge=float(ridge),
        recon_method=recon_method,
        target_voltage_digits=target_voltage_digits,
        holdout_voltage_rmse=holdout_voltage_rmse,
        diff_voltage_rmse=diff_voltage_rmse,
        sigma_rmse=metrics.sigma_rmse,
        sigma_relative_rmse=metrics.sigma_relative_rmse,
        sigma_mae=metrics.sigma_mae,
        sigma_max_abs_error=metrics.sigma_max_abs_error,
        sigma_effective_digits=metrics.sigma_effective_digits,
        centroid_error=metrics.centroid_error,
        eccentricity=metrics.eccentricity,
        artifact_area=metrics.artifact_area,
        artifact_energy=metrics.artifact_energy,
        artifact_peak=metrics.artifact_peak,
    )


def _full256_summary_from_metrics(
    *,
    bucket: CircleBucketDomain,
    ridge: float,
    recon_method: str,
    n_measurements: int,
    n_inverse_points: int,
    sigma_recon: np.ndarray,
    sigma_baseline: np.ndarray,
    metrics: _StructureMetrics,
    baseline_metrics: _StructureMetrics,
) -> BucketFull256CompareSummaryRow:
    delta_field = np.asarray(sigma_recon, dtype=float) - np.asarray(
        sigma_baseline,
        dtype=float,
    )
    return BucketFull256CompareSummaryRow(
        experiment="full256_compare",
        domain=bucket.domain,
        mesh_h=bucket.mesh_h,
        n_cells=bucket.n_cells,
        n_dofs=bucket.n_dofs,
        n_elec=bucket.n_elec,
        n_measurements=int(n_measurements),
        n_inverse_points=int(n_inverse_points),
        ridge=float(ridge),
        recon_method=recon_method,
        delta_sigma_relative_rmse_vs_full_208=(
            metrics.sigma_relative_rmse - baseline_metrics.sigma_relative_rmse
        ),
        delta_artifact_energy_vs_full_208=(
            metrics.artifact_energy - baseline_metrics.artifact_energy
        ),
        delta_field_rmse_vs_full_208=float(np.sqrt(np.mean(delta_field**2))),
        delta_field_l2_vs_full_208=float(np.linalg.norm(delta_field)),
        delta_field_max_abs_vs_full_208=float(np.max(np.abs(delta_field))),
        sigma_rmse=metrics.sigma_rmse,
        sigma_relative_rmse=metrics.sigma_relative_rmse,
        sigma_mae=metrics.sigma_mae,
        sigma_max_abs_error=metrics.sigma_max_abs_error,
        sigma_effective_digits=metrics.sigma_effective_digits,
        centroid_error=metrics.centroid_error,
        eccentricity=metrics.eccentricity,
        artifact_area=metrics.artifact_area,
        artifact_energy=metrics.artifact_energy,
        artifact_peak=metrics.artifact_peak,
    )


def _field_rows_for_sigma(
    *,
    bucket: CircleBucketDomain,
    experiment: str,
    recon_method: str,
    sigma_recon: np.ndarray,
) -> list[BucketDenseFieldRow]:
    error = np.asarray(sigma_recon, dtype=float) - bucket.sigma_true
    inside = np.linalg.norm(bucket.cell_centers, axis=1) <= bucket.bucket_radius + 1e-10
    return [
        BucketDenseFieldRow(
            experiment=experiment,
            recon_method=recon_method,
            cell_index=int(index),
            cell_x=float(point[0]),
            cell_y=float(point[1]),
            sigma_true=float(true_value),
            sigma_recon=float(recon_value),
            sigma_error=float(error_value),
            inside_bucket=bool(inside_value),
        )
        for index, (point, true_value, recon_value, error_value, inside_value) in (
            enumerate(
                zip(
                    bucket.cell_centers,
                    bucket.sigma_true,
                    sigma_recon,
                    error,
                    inside,
                    strict=True,
                )
            )
        )
    ]


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _structure_lookup(
    rows: Iterable[HoldoutStructureMetricRow],
) -> dict[str, HoldoutStructureMetricRow]:
    return {row.recon_kind: row for row in rows}


def _metrics_from_holdout_structure(
    row: HoldoutStructureMetricRow,
) -> _StructureMetrics:
    return _StructureMetrics(
        centroid_error=row.centroid_error,
        equivalent_area=row.equivalent_area,
        eccentricity=row.eccentricity,
        major_axis=row.major_axis,
        minor_axis=row.minor_axis,
        artifact_area=row.artifact_area,
        artifact_energy=row.artifact_energy,
        artifact_peak=row.artifact_peak,
        sigma_rmse=row.sigma_rmse,
        sigma_relative_rmse=row.sigma_relative_rmse,
        sigma_mae=row.sigma_mae,
        sigma_max_abs_error=row.sigma_max_abs_error,
        sigma_effective_digits=row.sigma_effective_digits,
    )


def run_bucket_dense_experiments(
    *,
    domain: str = "circle_bucket",
    bucket_radius: float = 1.0,
    n_elec: int = 16,
    mesh_h: float = 0.1,
    ridge: float = 1e-4,
    target_digits: Iterable[int] = (4, 5, 6, 7),
    holdout: str = "far3",
    raw_160_baseline: bool = True,
    fit_methods: Iterable[str] = ("poly2", "poly3", "spline"),
    inverse_backend: str = "measurement-rm",
    allow_coarse_smoke: bool = False,
    normalize_rows: bool = False,
) -> BucketDenseExperimentCase:
    """Run voltage-digit and holdout experiments on one dense circle bucket."""

    bucket = build_circle_bucket_domain(
        domain=domain,
        bucket_radius=bucket_radius,
        n_elec=n_elec,
        mesh_h=mesh_h,
        allow_coarse_smoke=allow_coarse_smoke,
    )
    model = build_circle_bucket_linearized_model(
        bucket=bucket,
        normalize_rows=normalize_rows,
    )
    summaries: list[BucketDenseSummaryRow] = []
    field_rows: list[BucketDenseFieldRow] = []
    voltage_recon_by_method: dict[str, np.ndarray] = {}

    for target in [int(value) for value in target_digits]:
        voltage_digit = keep_significant_digits(model.voltage_true, target)
        from .eit_digit_metrics import reconstruct_linearized_sigma

        sigma_recon = reconstruct_linearized_sigma(
            model=model,
            voltages=voltage_digit,
            ridge=ridge,
            inverse_backend=inverse_backend,
        )
        method = f"digits_{target}"
        voltage_recon_by_method[method] = sigma_recon
        metrics = _structure_metrics(bucket=bucket, sigma_recon=sigma_recon)
        summaries.append(
            _summary_from_metrics(
                experiment="voltage_digit_sweep",
                bucket=bucket,
                ridge=ridge,
                recon_method=method,
                target_voltage_digits=target,
                holdout_voltage_rmse=None,
                diff_voltage_rmse=rmse(model.voltage_true, voltage_digit),
                metrics=metrics,
            )
        )
        field_rows.extend(
            _field_rows_for_sigma(
                bucket=bucket,
                experiment="voltage_digit_sweep",
                recon_method=method,
                sigma_recon=sigma_recon,
            )
        )

    holdout_case = run_holdout_fit_diff(
        model=model,
        holdout=holdout,
        fit_methods=fit_methods,
        raw_160_baseline=raw_160_baseline,
        ridge=ridge,
        inverse_backend=inverse_backend,
    )
    structure = _structure_lookup(holdout_case.structure_rows)
    full_metrics = _metrics_from_holdout_structure(structure["full_208"])
    summaries.append(
        _summary_from_metrics(
            experiment="holdout_far3",
            bucket=bucket,
            ridge=ridge,
            recon_method="full_208",
            target_voltage_digits=None,
            holdout_voltage_rmse=0.0,
            diff_voltage_rmse=0.0,
            metrics=full_metrics,
        )
    )
    field_rows.extend(
        _field_rows_for_sigma(
            bucket=bucket,
            experiment="holdout_far3",
            recon_method="full_208",
            sigma_recon=holdout_case.sigma_recon_full,
        )
    )
    for row in holdout_case.summaries:
        metrics = _metrics_from_holdout_structure(structure[row.recon_method])
        summaries.append(
            _summary_from_metrics(
                experiment="holdout_far3",
                bucket=bucket,
                ridge=ridge,
                recon_method=row.recon_method,
                target_voltage_digits=None,
                holdout_voltage_rmse=_finite_or_none(row.holdout_voltage_rmse),
                diff_voltage_rmse=_finite_or_none(row.diff_voltage_rmse),
                metrics=metrics,
            )
        )
        field_rows.extend(
            _field_rows_for_sigma(
                bucket=bucket,
                experiment="holdout_far3",
                recon_method=row.recon_method,
                sigma_recon=holdout_case.sigma_recon_by_method[row.recon_method],
            )
        )

    return BucketDenseExperimentCase(
        bucket=bucket,
        model=model,
        summaries=summaries,
        field_rows=field_rows,
        voltage_recon_by_method=voltage_recon_by_method,
        holdout_case=holdout_case,
    )


def run_bucket_full256_compare_experiment(
    *,
    domain: str = "circle_bucket",
    bucket_radius: float = 1.0,
    n_elec: int = 16,
    mesh_h: float = 0.1,
    ridge: float = 1e-4,
    holdout: str = "far3",
    raw_160_baseline: bool = True,
    fit_methods: Iterable[str] = ("poly2", "poly3", "spline"),
    inverse_backend: str = "measurement-rm",
    allow_coarse_smoke: bool = False,
    normalize_rows: bool = False,
) -> BucketFull256CompareCase:
    """Compare a full 256-point model against 208/160/fitted reconstructions."""

    from .eit_digit_metrics import reconstruct_linearized_sigma

    bucket = build_circle_bucket_domain(
        domain=domain,
        bucket_radius=bucket_radius,
        n_elec=n_elec,
        mesh_h=mesh_h,
        allow_coarse_smoke=allow_coarse_smoke,
    )
    model_full_208 = build_circle_bucket_linearized_model(
        bucket=bucket,
        normalize_rows=normalize_rows,
    )
    model_full_256 = build_circle_bucket_linearized_model(
        bucket=bucket,
        normalize_rows=normalize_rows,
        include_drive_related=True,
    )
    expected_full = int(n_elec) * int(n_elec)
    if int(model_full_256.n_measurements) != expected_full:
        raise RuntimeError(
            "full 256 model measurement count mismatch: "
            f"{model_full_256.n_measurements} != {expected_full}"
        )
    if int(model_full_208.n_measurements) != bucket.n_measurements:
        raise RuntimeError("native 208 model measurement count mismatch")

    holdout_case = run_holdout_fit_diff(
        model=model_full_208,
        holdout=holdout,
        fit_methods=fit_methods,
        raw_160_baseline=raw_160_baseline,
        ridge=ridge,
        inverse_backend=inverse_backend,
    )
    sigma_full_256 = reconstruct_linearized_sigma(
        model=model_full_256,
        voltages=model_full_256.voltage_true,
        ridge=ridge,
        inverse_backend=inverse_backend,
    )
    far3_drop_near3_keep_indices = _far3_drop_near3_keep_indices(n_elec)
    model_far3_drop_near3_keep_208 = _measurement_submodel(
        model_full_256,
        far3_drop_near3_keep_indices,
        label="circle_bucket_dense_far3_drop_near3_keep_208",
    )
    sigma_far3_drop_near3_keep_208 = reconstruct_linearized_sigma(
        model=model_far3_drop_near3_keep_208,
        voltages=model_far3_drop_near3_keep_208.voltage_true,
        ridge=ridge,
        inverse_backend=inverse_backend,
    )

    sigma_recon_by_method: dict[str, np.ndarray] = {
        "full_256": sigma_full_256,
        "full_208": holdout_case.sigma_recon_full,
        "far3_drop_near3_keep_208": sigma_far3_drop_near3_keep_208,
    }
    sigma_recon_by_method.update(holdout_case.sigma_recon_by_method)

    metrics_by_method = {
        method: _structure_metrics(bucket=bucket, sigma_recon=sigma_recon)
        for method, sigma_recon in sigma_recon_by_method.items()
    }
    baseline_metrics = metrics_by_method["full_208"]
    inverse_points = {
        "full_256": int(model_full_256.n_measurements),
        "full_208": int(model_full_208.n_measurements),
        "far3_drop_near3_keep_208": int(model_far3_drop_near3_keep_208.n_measurements),
    }
    inverse_points.update(
        {row.recon_method: int(row.n_inverse_points) for row in holdout_case.summaries}
    )
    available_measurements = {
        "full_256": int(model_full_256.n_measurements),
        "full_208": int(model_full_208.n_measurements),
        "far3_drop_near3_keep_208": int(model_full_256.n_measurements),
    }
    available_measurements.update(
        {
            row.recon_method: int(model_full_208.n_measurements)
            for row in holdout_case.summaries
        }
    )

    summaries = [
        _full256_summary_from_metrics(
            bucket=bucket,
            ridge=ridge,
            recon_method=method,
            n_measurements=available_measurements[method],
            n_inverse_points=inverse_points[method],
            sigma_recon=sigma_recon_by_method[method],
            sigma_baseline=sigma_recon_by_method["full_208"],
            metrics=metrics,
            baseline_metrics=baseline_metrics,
        )
        for method, metrics in metrics_by_method.items()
    ]
    field_rows: list[BucketDenseFieldRow] = []
    for method, sigma_recon in sigma_recon_by_method.items():
        field_rows.extend(
            _field_rows_for_sigma(
                bucket=bucket,
                experiment="full256_compare",
                recon_method=method,
                sigma_recon=sigma_recon,
            )
        )

    return BucketFull256CompareCase(
        bucket=bucket,
        model_full_256=model_full_256,
        model_full_208=model_full_208,
        holdout_case=holdout_case,
        summaries=summaries,
        field_rows=field_rows,
        sigma_recon_by_method=sigma_recon_by_method,
    )


def _draw_circle(ax, radius: float) -> None:
    import matplotlib.patches as patches

    ax.add_patch(
        patches.Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            edgecolor="#111111",
            linewidth=0.9,
            zorder=4,
        )
    )


def _tripcolor_field(
    ax,
    case: BucketDenseExperimentCase,
    values: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
):
    import matplotlib.tri as mtri

    bucket = case.bucket
    triangulation = mtri.Triangulation(
        bucket.nodes[:, 0],
        bucket.nodes[:, 1],
        bucket.cells,
    )
    image = ax.tripcolor(
        triangulation,
        facecolors=values,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="#ffffff",
        linewidth=0.05,
    )
    _draw_circle(ax, bucket.bucket_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def plot_bucket_dense_recon_compare(
    case: BucketDenseExperimentCase,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot dense-bucket truth, full 208, raw 160, fitted 208, and errors."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fields: list[tuple[str, np.ndarray]] = [
        ("truth", case.bucket.sigma_true),
        ("full_208", case.holdout_case.sigma_recon_full),
    ]
    fields.extend(case.holdout_case.sigma_recon_by_method.items())
    sigma_values = np.concatenate([values for _, values in fields])
    sigma_min = float(np.min(sigma_values))
    sigma_max = float(np.max(sigma_values))
    errors = [values - case.bucket.sigma_true for _, values in fields[1:]]
    error_lim = max(float(max(np.max(np.abs(error)) for error in errors)), 1e-12)

    n_cols = len(fields)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(2.5 * n_cols, 5.6),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("T23 dense circle bucket recon compare", fontsize=14)
    for col_idx, (label, values) in enumerate(fields):
        image = _tripcolor_field(
            axes[0, col_idx],
            case,
            np.asarray(values, dtype=float),
            vmin=sigma_min,
            vmax=sigma_max,
            cmap="viridis",
        )
        axes[0, col_idx].set_title(label, fontsize=9)
        fig.colorbar(image, ax=axes[0, col_idx], fraction=0.046, pad=0.02)
        error_values = (
            np.zeros_like(case.bucket.sigma_true)
            if col_idx == 0
            else np.asarray(values) - case.bucket.sigma_true
        )
        err_image = _tripcolor_field(
            axes[1, col_idx],
            case,
            error_values,
            vmin=-error_lim,
            vmax=error_lim,
            cmap="coolwarm",
        )
        axes[1, col_idx].set_title("error", fontsize=9)
        fig.colorbar(err_image, ax=axes[1, col_idx], fraction=0.046, pad=0.02)

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_bucket_dense_summary(
    case: BucketDenseExperimentCase,
    output_path: Path,
    *,
    coarse_voltage_csv: Path | None = Path("outputs/eit_voltage_digit_sweep_16e.csv"),
    coarse_holdout_csv: Path | None = None,
    coarse_structure_csv: Path | None = None,
    dpi: int = 200,
) -> Path:
    """Plot dense-bucket metrics against available coarse-grid references."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    voltage_rows = [
        row for row in case.summaries if row.experiment == "voltage_digit_sweep"
    ]
    holdout_rows = [row for row in case.summaries if row.experiment == "holdout_far3"]
    coarse_voltage = _read_csv_rows(coarse_voltage_csv) if coarse_voltage_csv else []
    coarse_holdout = _read_csv_rows(coarse_holdout_csv) if coarse_holdout_csv else []
    coarse_structure = (
        _read_csv_rows(coarse_structure_csv) if coarse_structure_csv else []
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    fig.suptitle("T23 coarse vs dense circle bucket summary", fontsize=14)

    targets = [int(row.target_voltage_digits or 0) for row in voltage_rows]
    rel = [row.sigma_relative_rmse for row in voltage_rows]
    axes[0].plot(
        targets,
        rel,
        marker="o",
        linewidth=1.8,
        label="circle_bucket_dense",
        color="#1f77b4",
    )
    if coarse_voltage:
        axes[0].plot(
            [int(row["target_voltage_digits"]) for row in coarse_voltage],
            [float(row["sigma_relative_rmse"]) for row in coarse_voltage],
            marker="s",
            linestyle="--",
            linewidth=1.4,
            label="coarse previous",
            color="#ff7f0e",
        )
    axes[0].set_title("Voltage digits")
    axes[0].set_xlabel("Target digits")
    axes[0].set_ylabel("Sigma relative RMSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    labels = [row.recon_method for row in holdout_rows]
    x = np.arange(len(labels), dtype=float)
    axes[1].bar(
        x - 0.18,
        [row.sigma_relative_rmse for row in holdout_rows],
        width=0.36,
        label="circle_bucket_dense",
        color="#2ca02c",
    )
    coarse_map = {row["recon_method"]: row for row in coarse_holdout}
    if coarse_map:
        axes[1].bar(
            x + 0.18,
            [
                float(coarse_map[label]["recon_sigma_relative_rmse"])
                if label in coarse_map
                else math.nan
                for label in labels
            ],
            width=0.36,
            label="coarse previous",
            color="#d62728",
            alpha=0.75,
        )
    axes[1].set_title("Holdout recon")
    axes[1].set_ylabel("Sigma relative RMSE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    dense_artifact = {row.recon_method: row.artifact_energy for row in holdout_rows}
    coarse_artifact = {
        row["recon_kind"]: float(row["artifact_energy"]) for row in coarse_structure
    }
    art_labels = [label for label in labels if label != "full_208"]
    art_x = np.arange(len(art_labels), dtype=float)
    axes[2].bar(
        art_x - 0.18,
        [dense_artifact[label] for label in art_labels],
        width=0.36,
        label="circle_bucket_dense",
        color="#9467bd",
    )
    if coarse_artifact:
        axes[2].bar(
            art_x + 0.18,
            [coarse_artifact.get(label, math.nan) for label in art_labels],
            width=0.36,
            label="coarse previous",
            color="#8c564b",
            alpha=0.75,
        )
    axes[2].set_title("Artifact energy")
    axes[2].set_ylabel("Energy")
    axes[2].set_xticks(art_x)
    axes[2].set_xticklabels(art_labels, rotation=35, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(fontsize=8)

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_bucket_full256_compare_recon(
    case: BucketFull256CompareCase,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot full-256, native 208, raw 160, fitted 208, and error maps."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fields: list[tuple[str, np.ndarray]] = [("truth", case.bucket.sigma_true)]
    fields.extend(case.sigma_recon_by_method.items())
    sigma_values = np.concatenate([values for _, values in fields])
    sigma_min = float(np.min(sigma_values))
    sigma_max = float(np.max(sigma_values))
    errors = [values - case.bucket.sigma_true for _, values in fields[1:]]
    error_lim = max(float(max(np.max(np.abs(error)) for error in errors)), 1e-12)

    n_cols = len(fields)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(2.45 * n_cols, 5.6),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Full 256 vs 208/160/fitted recon compare", fontsize=14)
    for col_idx, (label, values) in enumerate(fields):
        image = _tripcolor_field(
            axes[0, col_idx],
            case,
            np.asarray(values, dtype=float),
            vmin=sigma_min,
            vmax=sigma_max,
            cmap="viridis",
        )
        axes[0, col_idx].set_title(label, fontsize=9)
        fig.colorbar(image, ax=axes[0, col_idx], fraction=0.046, pad=0.02)
        error_values = (
            np.zeros_like(case.bucket.sigma_true)
            if col_idx == 0
            else np.asarray(values) - case.bucket.sigma_true
        )
        err_image = _tripcolor_field(
            axes[1, col_idx],
            case,
            error_values,
            vmin=-error_lim,
            vmax=error_lim,
            cmap="coolwarm",
        )
        axes[1, col_idx].set_title("error", fontsize=9)
        fig.colorbar(err_image, ax=axes[1, col_idx], fraction=0.046, pad=0.02)

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_bucket_full256_compare_recon_with_full208_delta(
    case: BucketFull256CompareCase,
    output_path: Path,
    *,
    baseline_method: str = "full_208",
    dpi: int = 200,
) -> Path:
    """Plot recon, truth error, and direct field deltas against full_208."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    if baseline_method not in case.sigma_recon_by_method:
        raise ValueError(f"baseline_method not found: {baseline_method}")

    fields: list[tuple[str, np.ndarray]] = [("truth", case.bucket.sigma_true)]
    fields.extend(case.sigma_recon_by_method.items())
    baseline = np.asarray(case.sigma_recon_by_method[baseline_method], dtype=float)
    sigma_values = np.concatenate([values for _, values in fields])
    sigma_min = float(np.min(sigma_values))
    sigma_max = float(np.max(sigma_values))
    truth_errors = [np.asarray(values) - case.bucket.sigma_true for _, values in fields]
    error_lim = max(
        float(max(np.max(np.abs(error)) for error in truth_errors[1:])),
        1e-12,
    )
    full208_deltas = [np.asarray(values) - baseline for _, values in fields]
    delta_lim = max(
        float(max(np.max(np.abs(delta)) for delta in full208_deltas)),
        1e-12,
    )

    n_cols = len(fields)
    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(2.45 * n_cols, 8.3),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        "Full acquisition-mode recon compare: truth error and delta vs full_208",
        fontsize=14,
    )
    for col_idx, (label, values) in enumerate(fields):
        image = _tripcolor_field(
            axes[0, col_idx],
            case,
            np.asarray(values, dtype=float),
            vmin=sigma_min,
            vmax=sigma_max,
            cmap="viridis",
        )
        axes[0, col_idx].set_title(label, fontsize=8)
        fig.colorbar(image, ax=axes[0, col_idx], fraction=0.046, pad=0.02)

        err_image = _tripcolor_field(
            axes[1, col_idx],
            case,
            np.asarray(values, dtype=float) - case.bucket.sigma_true,
            vmin=-error_lim,
            vmax=error_lim,
            cmap="coolwarm",
        )
        axes[1, col_idx].set_title("error vs truth", fontsize=8)
        fig.colorbar(err_image, ax=axes[1, col_idx], fraction=0.046, pad=0.02)

        delta_image = _tripcolor_field(
            axes[2, col_idx],
            case,
            np.asarray(values, dtype=float) - baseline,
            vmin=-delta_lim,
            vmax=delta_lim,
            cmap="coolwarm",
        )
        axes[2, col_idx].set_title(f"delta vs {baseline_method}", fontsize=8)
        fig.colorbar(delta_image, ax=axes[2, col_idx], fraction=0.046, pad=0.02)

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_bucket_full256_compare_metrics(
    case: BucketFull256CompareCase,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot numeric full-256 comparison metrics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = case.summaries
    labels = [row.recon_method for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 7.0), constrained_layout=True)
    fig.suptitle("Full 256 numeric comparison", fontsize=14)
    panels = [
        (
            "Sigma relative RMSE",
            [row.sigma_relative_rmse for row in rows],
            "#1f77b4",
        ),
        (
            "Direct field L2 vs full_208",
            [row.delta_field_l2_vs_full_208 for row in rows],
            "#d62728",
        ),
        (
            "Direct field RMSE vs full_208",
            [row.delta_field_rmse_vs_full_208 for row in rows],
            "#8c564b",
        ),
        (
            "Sigma effective digits",
            [row.sigma_effective_digits for row in rows],
            "#2ca02c",
        ),
        ("Artifact energy", [row.artifact_energy for row in rows], "#9467bd"),
        ("Centroid error", [row.centroid_error for row in rows], "#ff7f0e"),
    ]
    for ax, (title, values, color) in zip(axes.ravel(), panels, strict=True):
        ax.bar(x, values, color=color, alpha=0.86)
        ax.axvline(1.5, color="#444444", linewidth=0.8, linestyle="--", alpha=0.55)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def format_bucket_full256_compare_report(case: BucketFull256CompareCase) -> str:
    """Format a Chinese report for full-256-vs-filtered comparison."""

    rows = list(case.summaries)
    by_method = {row.recon_method: row for row in rows}
    best_rmse = min(rows, key=lambda row: row.sigma_relative_rmse)
    best_artifact = min(rows, key=lambda row: row.artifact_energy)
    full256 = by_method["full_256"]
    full208 = by_method["full_208"]
    lines = [
        "# T27 full 256 不删点对比报告",
        "",
        f"- domain: `{case.bucket.domain}`，mesh_h `{case.bucket.mesh_h}`，"
        f"n_cells/n_dofs `{case.bucket.n_cells}`，n_elec `{case.bucket.n_elec}`。",
        f"- 点数：full_256 `{case.model_full_256.n_measurements}`；"
        f"原生 full_208 `{case.model_full_208.n_measurements}`；"
        "far3_drop_near3_keep_208 为从 256 候选点中仅删除远端 3 点/帧、"
        "保留激励相关近端 3 点/帧；"
        "raw_160 每帧再删 far3 后用 160 点；拟合 208 用 160 点训练补回 48 点。",
        f"- full_256 相对 full_208：delta_sigma_relative_rmse "
        f"`{full256.delta_sigma_relative_rmse_vs_full_208:.12g}`，"
        f"delta_artifact_energy "
        f"`{full256.delta_artifact_energy_vs_full_208:.12g}`。",
        f"- 最小 sigma_relative_rmse: `{best_rmse.recon_method}` = "
        f"`{best_rmse.sigma_relative_rmse:.12g}`；full_208 = "
        f"`{full208.sigma_relative_rmse:.12g}`。",
        f"- 最小 artifact_energy: `{best_artifact.recon_method}` = "
        f"`{best_artifact.artifact_energy:.12g}`；full_208 = "
        f"`{full208.artifact_energy:.12g}`。",
        "- 结论边界：该比较只改变测量点保留策略；圆形域、异常体、ridge、"
        "inverse backend 均固定。",
        "",
        "## 数值表",
        "",
        "| recon_method | n_inverse_points | sigma_relative_rmse | delta_rmse_vs_full_208 | artifact_energy | delta_artifact_vs_full_208 | direct_field_l2_vs_full_208 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.recon_method} | {row.n_inverse_points} | "
            f"{row.sigma_relative_rmse:.12g} | "
            f"{row.delta_sigma_relative_rmse_vs_full_208:.12g} | "
            f"{row.artifact_energy:.12g} | "
            f"{row.delta_artifact_energy_vs_full_208:.12g} | "
            f"{row.delta_field_l2_vs_full_208:.12g} |"
        )
    return "\n".join(lines) + "\n"


def format_bucket_dense_report(
    case: BucketDenseExperimentCase,
    *,
    coarse_voltage_csv: Path | None = Path("outputs/eit_voltage_digit_sweep_16e.csv"),
    coarse_holdout_csv: Path | None = None,
) -> str:
    """Format a Chinese dense-bucket comparison report."""

    voltage_rows = [
        row for row in case.summaries if row.experiment == "voltage_digit_sweep"
    ]
    holdout_rows = [row for row in case.summaries if row.experiment == "holdout_far3"]
    best_voltage = min(voltage_rows, key=lambda row: row.sigma_relative_rmse)
    best_holdout = min(holdout_rows, key=lambda row: row.sigma_relative_rmse)
    ridge = case.summaries[0].ridge if case.summaries else math.nan
    row_norms = np.linalg.norm(np.asarray(case.model.sensitivity, dtype=float), axis=1)
    positive_norms = row_norms[row_norms > 0.0]
    row_normalized = bool(
        positive_norms.size and np.allclose(positive_norms, 1.0, rtol=1e-6, atol=1e-9)
    )
    coarse_voltage = _read_csv_rows(coarse_voltage_csv) if coarse_voltage_csv else []
    coarse_holdout = _read_csv_rows(coarse_holdout_csv) if coarse_holdout_csv else []
    coarse_note = (
        f"已读取 coarse voltage `{coarse_voltage_csv}` {len(coarse_voltage)} rows；"
        f"coarse holdout `{coarse_holdout_csv}` {len(coarse_holdout)} rows。"
        if coarse_voltage or coarse_holdout
        else "未找到 coarse 参考 CSV；本报告仅列 dense bucket 结果。"
    )
    lines = [
        "# T23 密集圆形小水桶复测报告",
        "",
        f"- domain: `{case.bucket.domain}`，mesh_h `{case.bucket.mesh_h}`，"
        f"n_cells/n_dofs `{case.bucket.n_cells}`，n_measurements "
        f"`{case.bucket.n_measurements}`。",
        f"- voltage model: nonzero homogeneous reference="
        f"`{bool(np.max(np.abs(case.model.voltage_reference)) > 0.0)}`，"
        f"row_normalized=`{row_normalized}`。",
        f"- inverse: backend `measurement-rm`，ridge `{ridge:.12g}`。",
        f"- voltage sweep 最小相对误差：`{best_voltage.recon_method}`，"
        f"sigma_relative_rmse `{best_voltage.sigma_relative_rmse:.12g}`。",
        f"- holdout 最小相对误差：`{best_holdout.recon_method}`，"
        f"sigma_relative_rmse `{best_holdout.sigma_relative_rmse:.12g}`。",
        f"- 粗网格对比：{coarse_note}",
        "- 结论边界：本轮使用密集圆形域与 measurement-space RM；"
        "若与旧粗网格结论冲突，以 dense bucket 可视化/指标为主。",
        "",
    ]
    return "\n".join(lines)


def write_bucket_dense_outputs(
    case: BucketDenseExperimentCase,
    *,
    summary_output: Path,
    field_output: Path,
    report_output: Path,
    domain_plot_output: Path,
    recon_plot_output: Path,
    summary_plot_output: Path,
    curve_plot_output: Path,
    holdout_summary_plot_output: Path,
    coarse_voltage_csv: Path | None = Path("outputs/eit_voltage_digit_sweep_16e.csv"),
    coarse_holdout_csv: Path | None = None,
    coarse_structure_csv: Path | None = None,
    hdf5_output: Path | None = None,
    json_output: Path | None = None,
    dpi: int = 200,
) -> dict[str, Path]:
    """Write all T23 dense-bucket CSV, report, and visual outputs."""

    write_csv_rows(summary_output, case.summaries, BUCKET_DENSE_SUMMARY_FIELDS)
    write_csv_rows(field_output, case.field_rows, BUCKET_DENSE_FIELD_FIELDS)
    table_artifacts = write_sweep_table_artifacts(
        tables={
            "bucket_dense_summary": (
                BUCKET_DENSE_SUMMARY_FIELDS,
                case.summaries,
            ),
            "bucket_dense_field": (BUCKET_DENSE_FIELD_FIELDS, case.field_rows),
        },
        hdf5_output=hdf5_output,
        json_output=json_output,
        metadata={
            "report_kind": "bucket_dense_experiments",
            "domain": case.bucket.domain,
            "mesh_h": case.bucket.mesh_h,
            "n_cells": case.bucket.n_cells,
            "n_dofs": case.bucket.n_dofs,
            "n_elec": case.bucket.n_elec,
            "n_measurements": case.bucket.n_measurements,
            "ridge": case.summaries[0].ridge if case.summaries else "",
        },
    )

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        format_bucket_dense_report(
            case,
            coarse_voltage_csv=coarse_voltage_csv,
            coarse_holdout_csv=coarse_holdout_csv,
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary_output,
        "fields": field_output,
        "report": report_output,
        **table_artifacts,
        "domain_plot": plot_bucket_domain_audit(
            case.bucket,
            domain_plot_output,
            title="T23 dense circle bucket domain audit",
            dpi=dpi,
        ),
        "recon_plot": plot_bucket_dense_recon_compare(
            case,
            recon_plot_output,
            dpi=dpi,
        ),
        "summary_plot": plot_bucket_dense_summary(
            case,
            summary_plot_output,
            coarse_voltage_csv=coarse_voltage_csv,
            coarse_holdout_csv=coarse_holdout_csv,
            coarse_structure_csv=coarse_structure_csv,
            dpi=dpi,
        ),
        "curve_plot": plot_holdout_fit_curves(
            case.holdout_case,
            curve_plot_output,
            dpi=dpi,
        ),
        "holdout_summary_plot": plot_holdout_fit_summary(
            case.holdout_case,
            holdout_summary_plot_output,
            dpi=dpi,
        ),
    }


def write_bucket_full256_compare_outputs(
    case: BucketFull256CompareCase,
    *,
    summary_output: Path,
    field_output: Path,
    report_output: Path,
    recon_plot_output: Path,
    metrics_plot_output: Path,
    point_audit_plot_output: Path,
    recon_delta_plot_output: Path | None = None,
    hdf5_output: Path | None = None,
    json_output: Path | None = None,
    dpi: int = 200,
) -> dict[str, Path]:
    """Write all full-256 comparison CSV, report, and visual outputs."""

    write_csv_rows(
        summary_output,
        case.summaries,
        BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
    )
    write_csv_rows(field_output, case.field_rows, BUCKET_FULL256_COMPARE_FIELD_FIELDS)
    table_artifacts = write_sweep_table_artifacts(
        tables={
            "bucket_full256_summary": (
                BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
                case.summaries,
            ),
            "bucket_dense_field": (
                BUCKET_FULL256_COMPARE_FIELD_FIELDS,
                case.field_rows,
            ),
        },
        hdf5_output=hdf5_output,
        json_output=json_output,
        metadata={
            "report_kind": "bucket_full256_compare",
            "domain": case.bucket.domain,
            "mesh_h": case.bucket.mesh_h,
            "n_cells": case.bucket.n_cells,
            "n_dofs": case.bucket.n_dofs,
            "n_elec": case.bucket.n_elec,
            "full_256_measurements": case.model_full_256.n_measurements,
            "full_208_measurements": case.model_full_208.n_measurements,
            "ridge": case.summaries[0].ridge if case.summaries else "",
        },
    )

    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        format_bucket_full256_compare_report(case),
        encoding="utf-8",
    )
    written = {
        "summary": summary_output,
        "fields": field_output,
        "report": report_output,
        **table_artifacts,
        "recon_plot": plot_bucket_full256_compare_recon(
            case,
            recon_plot_output,
            dpi=dpi,
        ),
        "metrics_plot": plot_bucket_full256_compare_metrics(
            case,
            metrics_plot_output,
            dpi=dpi,
        ),
        "point_audit_plot": plot_holdout_point_audit(
            case.holdout_case.point_rows,
            point_audit_plot_output,
            n_elec=case.bucket.n_elec,
            dpi=dpi,
        ),
    }
    if recon_delta_plot_output is not None:
        written["recon_delta_plot"] = (
            plot_bucket_full256_compare_recon_with_full208_delta(
                case,
                recon_delta_plot_output,
                dpi=dpi,
            )
        )
    return written


__all__ = [
    "BUCKET_DENSE_FIELD_FIELDS",
    "BUCKET_DENSE_SUMMARY_FIELDS",
    "BUCKET_FULL256_COMPARE_FIELD_FIELDS",
    "BUCKET_FULL256_COMPARE_SUMMARY_FIELDS",
    "BucketDenseExperimentCase",
    "BucketDenseFieldRow",
    "BucketDenseSummaryRow",
    "BucketFull256CompareCase",
    "BucketFull256CompareSummaryRow",
    "build_circle_bucket_linearized_model",
    "format_bucket_dense_report",
    "format_bucket_full256_compare_report",
    "plot_bucket_dense_recon_compare",
    "plot_bucket_dense_summary",
    "plot_bucket_full256_compare_metrics",
    "plot_bucket_full256_compare_recon",
    "plot_bucket_full256_compare_recon_with_full208_delta",
    "run_bucket_dense_experiments",
    "run_bucket_full256_compare_experiment",
    "write_bucket_dense_outputs",
    "write_bucket_full256_compare_outputs",
]
