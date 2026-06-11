"""Holdout fitting and raw-160 baselines for adjacent EIT difference imaging."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from pyeidors.utils.numeric_ops import all_finite_values

from .adc_quantization import effective_digits_from_rmse, rmse
from .eit_digit_metrics import (
    EITLinearizedModel,
    _fallback_parameter_points,
    reconstruct_linearized_sigma,
)
from .holdout_point_audit import (
    HoldoutPointAuditRow,
    build_holdout_point_audit,
)
from ._sweep_core import ReconMetricRow, StructureMetricRow, SweepRow


FitMethod = Literal["poly2", "poly3", "spline"]
_INDEXED_RMSE_CHUNK_ITEMS = 1_048_576
_MASKED_STRUCTURE_CHUNK_ITEMS = 1_048_576

SUMMARY_FIELDS = [
    "recon_method",
    "n_inverse_points",
    "frame_count",
    "points_per_frame",
    "holdout_per_frame",
    "train_points_per_frame",
    "holdout_voltage_rmse",
    "diff_voltage_rmse",
    "full_sigma_rmse",
    "recon_sigma_rmse",
    "delta_sigma_rmse",
    "full_sigma_relative_rmse",
    "recon_sigma_relative_rmse",
    "delta_sigma_relative_rmse",
    "full_sigma_effective_digits",
    "recon_sigma_effective_digits",
    "delta_sigma_effective_digits",
]

FIELD_FIELDS = [
    "recon_method",
    "cell_index",
    "sigma_true",
    "sigma_recon_full",
    "sigma_recon_candidate",
    "sigma_error_full",
    "sigma_error_candidate",
    "delta_sigma_error",
]

STRUCTURE_FIELDS = [
    "recon_kind",
    "threshold_rule",
    "centroid_x",
    "centroid_y",
    "centroid_error",
    "equivalent_area",
    "eccentricity",
    "major_axis",
    "minor_axis",
    "artifact_area",
    "artifact_energy",
    "artifact_peak",
    "sigma_rmse",
    "sigma_relative_rmse",
    "sigma_mae",
    "sigma_max_abs_error",
    "sigma_effective_digits",
]

FIT_CURVE_LEGEND_LABELS = {
    "target_full": "target full: 13 original pts",
    "reference_full": "reference full: 13 baseline pts",
    "fit_input": "fit input: 10 original pts",
    "withheld_true": "withheld true: 3 original pts",
    "poly2": "poly2 pred: 3 pts",
    "poly3": "poly3 pred: 3 pts",
    "spline": "spline pred: 3 pts",
}


@dataclass(frozen=True)
class HoldoutFitDiffSummary(ReconMetricRow):
    """One reconstruction-method summary row for holdout comparison."""

    recon_metric_fields = (
        "recon_sigma_rmse",
        "recon_sigma_relative_rmse",
        "recon_sigma_effective_digits",
    )

    recon_method: str
    n_inverse_points: int
    frame_count: int
    points_per_frame: int
    holdout_per_frame: int
    train_points_per_frame: int
    holdout_voltage_rmse: float
    diff_voltage_rmse: float
    full_sigma_rmse: float
    recon_sigma_rmse: float
    delta_sigma_rmse: float
    full_sigma_relative_rmse: float
    recon_sigma_relative_rmse: float
    delta_sigma_relative_rmse: float
    full_sigma_effective_digits: float
    recon_sigma_effective_digits: float
    delta_sigma_effective_digits: float


@dataclass(frozen=True)
class HoldoutFitDiffFieldRow(SweepRow):
    """Per-cell full-vs-candidate reconstruction error row."""

    recon_method: str
    cell_index: int
    sigma_true: float
    sigma_recon_full: float
    sigma_recon_candidate: float
    sigma_error_full: float
    sigma_error_candidate: float
    delta_sigma_error: float


@dataclass(frozen=True)
class HoldoutStructureMetricRow(StructureMetricRow):
    """Structure metrics for truth/full/candidate conductivity fields."""

    recon_kind: str
    threshold_rule: str
    centroid_x: float
    centroid_y: float
    centroid_error: float
    equivalent_area: float
    eccentricity: float
    major_axis: float
    minor_axis: float
    artifact_area: float
    artifact_energy: float
    artifact_peak: float
    sigma_rmse: float
    sigma_relative_rmse: float
    sigma_mae: float
    sigma_max_abs_error: float
    sigma_effective_digits: float


@dataclass(frozen=True)
class HoldoutFitFrameCurve:
    """Per-frame voltage curve data for plotting fit behavior."""

    stim_index: int
    x_all: np.ndarray
    voltage_reference_full: np.ndarray
    voltage_anomaly_full: np.ndarray
    diff_full: np.ndarray
    train_mask: np.ndarray
    holdout_mask: np.ndarray
    fitted_reference_by_method: dict[str, np.ndarray]
    fitted_anomaly_by_method: dict[str, np.ndarray]
    fitted_diff_by_method: dict[str, np.ndarray]


@dataclass(frozen=True)
class HoldoutFitDiffCase:
    """Full holdout-comparison result bundle."""

    model: EITLinearizedModel
    point_rows: list[HoldoutPointAuditRow]
    summaries: list[HoldoutFitDiffSummary]
    field_rows: list[HoldoutFitDiffFieldRow]
    structure_rows: list[HoldoutStructureMetricRow]
    frame_curves: list[HoldoutFitFrameCurve]
    sigma_recon_by_method: dict[str, np.ndarray]
    sigma_recon_full: np.ndarray
    fit_voltage_by_method: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]


def _as_float_vector(values: Iterable[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not all_finite_values(arr):
        raise ValueError(f"{name} must be finite")
    return arr


def _as_float_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not all_finite_values(arr):
        raise ValueError(f"{name} must be finite")
    return arr


def _relative_rmse(reference: np.ndarray, observed: np.ndarray) -> float:
    ref_rms = float(np.sqrt(np.mean(reference**2)))
    if ref_rms == 0.0:
        return math.nan
    return rmse(reference, observed) / ref_rms


def _rmse_values_at_indices(
    values: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int = _INDEXED_RMSE_CHUNK_ITEMS,
) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    count = int(idx.size)
    if count == 0:
        return float("nan")
    block_size = max(1, min(int(chunk_size), count))
    work = np.empty(block_size, dtype=np.float64)
    total = 0.0
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        chunk = idx[start:stop]
        target = work[: chunk.size]
        np.take(arr, chunk, out=target)
        np.square(target, out=target)
        total += float(np.sum(target, dtype=np.float64))
    return float(math.sqrt(total / count))


def _rmse_pair_at_indices(
    reference: np.ndarray,
    observed: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_size: int = _INDEXED_RMSE_CHUNK_ITEMS,
) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    count = int(idx.size)
    if count == 0:
        return float("nan")
    block_size = max(1, min(int(chunk_size), count))
    work = np.empty(block_size, dtype=np.float64)
    ref_work = np.empty(block_size, dtype=np.float64)
    total = 0.0
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        chunk = idx[start:stop]
        target = work[: chunk.size]
        ref_target = ref_work[: chunk.size]
        np.take(obs, chunk, out=target)
        np.take(ref, chunk, out=ref_target)
        target -= ref_target
        np.square(target, out=target)
        total += float(np.sum(target, dtype=np.float64))
    return float(math.sqrt(total / count))


def _validate_fit_methods(fit_methods: Iterable[str]) -> list[FitMethod]:
    methods: list[FitMethod] = []
    for method in fit_methods:
        name = str(method).strip().lower()
        if name not in {"poly2", "poly3", "spline"}:
            raise ValueError("fit_methods must contain only poly2, poly3, or spline")
        if name not in methods:
            methods.append(name)  # type: ignore[arg-type]
    if not methods:
        raise ValueError("fit_methods must not be empty")
    return methods


def _rows_by_stim(
    point_rows: Iterable[HoldoutPointAuditRow],
    *,
    n_elec: int,
) -> list[list[HoldoutPointAuditRow]]:
    grouped: list[list[HoldoutPointAuditRow]] = [[] for _ in range(int(n_elec))]
    for row in point_rows:
        if row.point_status == "drive_removed":
            continue
        grouped[row.stim_index].append(row)
    for stim_rows in grouped:
        stim_rows.sort(key=lambda row: int(row.frame_index_13))
    return grouped


def _row_global_208_index(row: HoldoutPointAuditRow, points_per_frame: int) -> int:
    if row.frame_index_13 is None:
        raise ValueError("row has no frame_index_13")
    return int(row.stim_index) * int(points_per_frame) + int(row.frame_index_13)


def _train_and_holdout_indices(
    point_rows: Iterable[HoldoutPointAuditRow],
    *,
    points_per_frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    holdout: list[int] = []
    for row in point_rows:
        if row.point_status == "fit_train_160":
            train.append(_row_global_208_index(row, points_per_frame))
        elif row.point_status == "holdout_far3":
            holdout.append(_row_global_208_index(row, points_per_frame))
    return np.array(train, dtype=np.int64), np.array(holdout, dtype=np.int64)


def _submodel(model: EITLinearizedModel, indices: np.ndarray) -> EITLinearizedModel:
    idx = np.asarray(indices, dtype=np.int64)
    return replace(
        model,
        voltage_true=np.asarray(model.voltage_true, dtype=float)[idx],
        voltage_reference=np.asarray(model.voltage_reference, dtype=float)[idx],
        sensitivity=np.asarray(model.sensitivity, dtype=float)[idx, :],
        n_measurements=int(idx.size),
    )


def _fit_values(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    method: FitMethod,
) -> np.ndarray:
    if method in {"poly2", "poly3"}:
        degree = 2 if method == "poly2" else 3
        if x_train.size <= degree:
            raise ValueError(f"{method} needs at least {degree + 1} training points")
        coeffs = np.polyfit(x_train, y_train, deg=degree)
        return np.polyval(coeffs, x_holdout)

    from scipy.interpolate import CubicSpline

    if x_train.size < 4:
        raise ValueError("spline needs at least 4 training points")
    spline = CubicSpline(x_train, y_train, bc_type="natural", extrapolate=False)
    predicted = np.asarray(spline(x_holdout), dtype=float)
    if not all_finite_values(predicted):
        raise FloatingPointError("spline prediction produced non-finite values")
    return predicted


def _fit_frame_vector(
    values: np.ndarray,
    *,
    x_all: np.ndarray,
    train_mask: np.ndarray,
    holdout_mask: np.ndarray,
    method: FitMethod,
) -> np.ndarray:
    fitted = np.asarray(values, dtype=float).copy()
    fitted[holdout_mask] = _fit_values(
        x_train=x_all[train_mask],
        y_train=fitted[train_mask],
        x_holdout=x_all[holdout_mask],
        method=method,
    )
    return fitted


def _fit_full_model_data(
    *,
    model: EITLinearizedModel,
    point_rows: list[HoldoutPointAuditRow],
    n_elec: int,
    points_per_frame: int,
    method: FitMethod,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[HoldoutFitFrameCurve]]:
    rows_by_stim = _rows_by_stim(point_rows, n_elec=n_elec)
    v_ref = _as_float_vector(model.voltage_reference, name="model.voltage_reference")
    v_true = _as_float_vector(model.voltage_true, name="model.voltage_true")
    sensitivity = _as_float_matrix(model.sensitivity, name="model.sensitivity")
    fit_ref = v_ref.copy()
    fit_true = v_true.copy()
    fit_sens = sensitivity.copy()
    curves: list[HoldoutFitFrameCurve] = []

    for stim_index, stim_rows in enumerate(rows_by_stim):
        if len(stim_rows) != points_per_frame:
            raise RuntimeError(
                f"stim {stim_index} has {len(stim_rows)} kept points, "
                f"expected {points_per_frame}"
            )
        frame_indices = np.array(
            [_row_global_208_index(row, points_per_frame) for row in stim_rows],
            dtype=np.int64,
        )
        x_all = np.array([int(row.frame_index_13) for row in stim_rows], dtype=float)
        train_mask = np.array(
            [row.point_status == "fit_train_160" for row in stim_rows],
            dtype=bool,
        )
        holdout_mask = np.array(
            [row.point_status == "holdout_far3" for row in stim_rows],
            dtype=bool,
        )
        if int(train_mask.sum()) != points_per_frame - 3:
            raise RuntimeError("fit train point count per frame must be 10 for 16e")
        if int(holdout_mask.sum()) != 3:
            raise RuntimeError("holdout point count per frame must be 3")

        ref_frame = _fit_frame_vector(
            v_ref[frame_indices],
            x_all=x_all,
            train_mask=train_mask,
            holdout_mask=holdout_mask,
            method=method,
        )
        true_frame = _fit_frame_vector(
            v_true[frame_indices],
            x_all=x_all,
            train_mask=train_mask,
            holdout_mask=holdout_mask,
            method=method,
        )
        sens_frame = sensitivity[frame_indices, :].copy()
        for col_idx in range(sens_frame.shape[1]):
            sens_frame[:, col_idx] = _fit_frame_vector(
                sens_frame[:, col_idx],
                x_all=x_all,
                train_mask=train_mask,
                holdout_mask=holdout_mask,
                method=method,
            )
        fit_ref[frame_indices] = ref_frame
        fit_true[frame_indices] = true_frame
        fit_sens[frame_indices, :] = sens_frame
        curves.append(
            HoldoutFitFrameCurve(
                stim_index=stim_index,
                x_all=x_all,
                voltage_reference_full=v_ref[frame_indices],
                voltage_anomaly_full=v_true[frame_indices],
                diff_full=v_true[frame_indices] - v_ref[frame_indices],
                train_mask=train_mask,
                holdout_mask=holdout_mask,
                fitted_reference_by_method={method: ref_frame},
                fitted_anomaly_by_method={method: true_frame},
                fitted_diff_by_method={method: true_frame - ref_frame},
            )
        )

    return fit_ref, fit_true, fit_sens, curves


def _summarize_candidate(
    *,
    recon_method: str,
    n_inverse_points: int,
    frame_count: int,
    points_per_frame: int,
    holdout_per_frame: int,
    train_points_per_frame: int,
    holdout_voltage_rmse: float,
    diff_voltage_rmse: float,
    sigma_true: np.ndarray,
    sigma_recon_full: np.ndarray,
    sigma_recon_candidate: np.ndarray,
) -> HoldoutFitDiffSummary:
    full_rmse = rmse(sigma_true, sigma_recon_full)
    candidate_rmse = rmse(sigma_true, sigma_recon_candidate)
    full_relative = _relative_rmse(sigma_true, sigma_recon_full)
    candidate_relative = _relative_rmse(sigma_true, sigma_recon_candidate)
    full_digits = effective_digits_from_rmse(sigma_true, sigma_recon_full)
    candidate_digits = effective_digits_from_rmse(sigma_true, sigma_recon_candidate)
    return HoldoutFitDiffSummary(
        recon_method=recon_method,
        n_inverse_points=int(n_inverse_points),
        frame_count=int(frame_count),
        points_per_frame=int(points_per_frame),
        holdout_per_frame=int(holdout_per_frame),
        train_points_per_frame=int(train_points_per_frame),
        holdout_voltage_rmse=float(holdout_voltage_rmse),
        diff_voltage_rmse=float(diff_voltage_rmse),
        full_sigma_rmse=full_rmse,
        recon_sigma_rmse=candidate_rmse,
        delta_sigma_rmse=candidate_rmse - full_rmse,
        full_sigma_relative_rmse=full_relative,
        recon_sigma_relative_rmse=candidate_relative,
        delta_sigma_relative_rmse=candidate_relative - full_relative,
        full_sigma_effective_digits=full_digits,
        recon_sigma_effective_digits=candidate_digits,
        delta_sigma_effective_digits=candidate_digits - full_digits,
    )


def _field_rows(
    *,
    recon_method: str,
    sigma_true: np.ndarray,
    sigma_recon_full: np.ndarray,
    sigma_recon_candidate: np.ndarray,
) -> list[HoldoutFitDiffFieldRow]:
    full_error = sigma_recon_full - sigma_true
    candidate_error = sigma_recon_candidate - sigma_true
    return [
        HoldoutFitDiffFieldRow(
            recon_method=recon_method,
            cell_index=int(index),
            sigma_true=float(true_value),
            sigma_recon_full=float(full_value),
            sigma_recon_candidate=float(candidate_value),
            sigma_error_full=float(full_err),
            sigma_error_candidate=float(candidate_err),
            delta_sigma_error=float(candidate_err - full_err),
        )
        for index, (
            true_value,
            full_value,
            candidate_value,
            full_err,
            candidate_err,
        ) in enumerate(
            zip(
                sigma_true,
                sigma_recon_full,
                sigma_recon_candidate,
                full_error,
                candidate_error,
                strict=True,
            )
        )
    ]


def _cell_areas(model: EITLinearizedModel, n_parameters: int) -> np.ndarray:
    if model.mesh_points is None or model.mesh_cells is None:
        return np.ones(int(n_parameters), dtype=float)
    points = np.asarray(model.mesh_points, dtype=float)
    cells = np.asarray(model.mesh_cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[0] != n_parameters or cells.shape[1] < 3:
        return np.ones(int(n_parameters), dtype=float)
    areas = np.zeros(int(n_parameters), dtype=float)
    xy = points[:, :2]
    for idx, cell in enumerate(cells):
        polygon = xy[np.asarray(cell, dtype=np.int64)]
        x = polygon[:, 0]
        y = polygon[:, 1]
        areas[idx] = 0.5 * abs(
            float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        )
    if float(np.min(areas, initial=np.inf)) <= 0.0:
        return np.ones(int(n_parameters), dtype=float)
    return areas


def _parameter_points(model: EITLinearizedModel, n_parameters: int) -> np.ndarray:
    if model.parameter_points is not None:
        points = np.asarray(model.parameter_points, dtype=float)
        if (
            points.ndim == 2
            and points.shape[0] == n_parameters
            and points.shape[1] >= 2
        ):
            return points[:, :2].copy()
    return _fallback_parameter_points(n_parameters)


def _weighted_centroid_covariance_2d(
    coords: np.ndarray,
    weights: np.ndarray,
    *,
    weight_sum: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(coords, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("coords must be a 2D array with at least two columns.")
    if points.shape[0] != w.size:
        raise ValueError("coords and weights must have matching row counts.")
    total = float(np.sum(w)) if weight_sum is None else float(weight_sum)
    if total <= 0.0:
        raise ValueError("weights must have positive sum.")

    xy = points[:, :2]
    centroid = np.array(
        [
            float(np.dot(w, xy[:, 0])) / total,
            float(np.dot(w, xy[:, 1])) / total,
        ],
        dtype=float,
    )
    dx = np.empty(w.size, dtype=float)
    dy = np.empty(w.size, dtype=float)
    work = np.empty(w.size, dtype=float)
    np.subtract(xy[:, 0], centroid[0], out=dx)
    np.subtract(xy[:, 1], centroid[1], out=dy)

    np.square(dx, out=work)
    cxx = float(np.dot(w, work)) / total
    np.multiply(dx, dy, out=work)
    cxy = float(np.dot(w, work)) / total
    np.square(dy, out=work)
    cyy = float(np.dot(w, work)) / total
    covariance = np.array([[cxx, cxy], [cxy, cyy]], dtype=float)
    return centroid, covariance


def _combined_mask_chunk(
    *,
    include_mask: np.ndarray | None,
    exclude_mask: np.ndarray | None,
    start: int,
    stop: int,
    work: np.ndarray,
    exclude_work: np.ndarray,
) -> np.ndarray:
    active = work[: stop - start]
    if include_mask is None:
        active.fill(True)
    else:
        active[:] = include_mask[start:stop]
    if exclude_mask is not None:
        excluded = exclude_work[: stop - start]
        np.logical_not(exclude_mask[start:stop], out=excluded)
        np.logical_and(active, excluded, out=active)
    return active


def _masked_area_sum(
    areas: np.ndarray,
    include_mask: np.ndarray | None,
    *,
    exclude_mask: np.ndarray | None = None,
    chunk_size: int = _MASKED_STRUCTURE_CHUNK_ITEMS,
) -> float:
    area_arr = np.asarray(areas, dtype=float).reshape(-1)
    include = (
        None
        if include_mask is None
        else np.asarray(include_mask, dtype=bool).reshape(-1)
    )
    exclude = (
        None
        if exclude_mask is None
        else np.asarray(exclude_mask, dtype=bool).reshape(-1)
    )
    if include is not None and include.size != area_arr.size:
        raise ValueError("include_mask must match areas length.")
    if exclude is not None and exclude.size != area_arr.size:
        raise ValueError("exclude_mask must match areas length.")
    if area_arr.size == 0:
        return 0.0
    block_size = max(1, min(int(chunk_size), int(area_arr.size)))
    active_work = np.empty(block_size, dtype=bool)
    exclude_work = np.empty(block_size, dtype=bool)
    total = 0.0
    for start in range(0, int(area_arr.size), block_size):
        stop = min(start + block_size, int(area_arr.size))
        active = _combined_mask_chunk(
            include_mask=include,
            exclude_mask=exclude,
            start=start,
            stop=stop,
            work=active_work,
            exclude_work=exclude_work,
        )
        if bool(active.any()):
            total += float(np.sum(area_arr[start:stop], where=active, initial=0.0))
    return total


def _masked_abs_peak(
    values: np.ndarray,
    include_mask: np.ndarray | None = None,
    *,
    exclude_mask: np.ndarray | None = None,
    chunk_size: int = _MASKED_STRUCTURE_CHUNK_ITEMS,
) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    include = (
        None
        if include_mask is None
        else np.asarray(include_mask, dtype=bool).reshape(-1)
    )
    exclude = (
        None
        if exclude_mask is None
        else np.asarray(exclude_mask, dtype=bool).reshape(-1)
    )
    if include is not None and include.size != arr.size:
        raise ValueError("include_mask must match values length.")
    if exclude is not None and exclude.size != arr.size:
        raise ValueError("exclude_mask must match values length.")
    if arr.size == 0:
        return 0.0
    block_size = max(1, min(int(chunk_size), int(arr.size)))
    active_work = np.empty(block_size, dtype=bool)
    exclude_work = np.empty(block_size, dtype=bool)
    abs_work = np.empty(block_size, dtype=float)
    peak = 0.0
    for start in range(0, int(arr.size), block_size):
        stop = min(start + block_size, int(arr.size))
        active = _combined_mask_chunk(
            include_mask=include,
            exclude_mask=exclude,
            start=start,
            stop=stop,
            work=active_work,
            exclude_work=exclude_work,
        )
        if not bool(active.any()):
            continue
        abs_chunk = abs_work[: stop - start]
        np.abs(arr[start:stop], out=abs_chunk)
        peak = max(peak, float(np.max(abs_chunk, where=active, initial=0.0)))
    return peak


def _masked_square_area_sum(
    values: np.ndarray,
    areas: np.ndarray,
    include_mask: np.ndarray | None = None,
    *,
    exclude_mask: np.ndarray | None = None,
    chunk_size: int = _MASKED_STRUCTURE_CHUNK_ITEMS,
) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    area_arr = np.asarray(areas, dtype=float).reshape(-1)
    include = (
        None
        if include_mask is None
        else np.asarray(include_mask, dtype=bool).reshape(-1)
    )
    exclude = (
        None
        if exclude_mask is None
        else np.asarray(exclude_mask, dtype=bool).reshape(-1)
    )
    if area_arr.size != arr.size:
        raise ValueError("values and areas must have matching length.")
    if include is not None and include.size != arr.size:
        raise ValueError("include_mask must match values length.")
    if exclude is not None and exclude.size != arr.size:
        raise ValueError("exclude_mask must match values length.")
    if arr.size == 0:
        return 0.0
    block_size = max(1, min(int(chunk_size), int(arr.size)))
    active_work = np.empty(block_size, dtype=bool)
    exclude_work = np.empty(block_size, dtype=bool)
    value_work = np.empty(block_size, dtype=float)
    total = 0.0
    for start in range(0, int(arr.size), block_size):
        stop = min(start + block_size, int(arr.size))
        active = _combined_mask_chunk(
            include_mask=include,
            exclude_mask=exclude,
            start=start,
            stop=stop,
            work=active_work,
            exclude_work=exclude_work,
        )
        if not bool(active.any()):
            continue
        values_chunk = value_work[: stop - start]
        np.square(arr[start:stop], out=values_chunk)
        np.multiply(values_chunk, area_arr[start:stop], out=values_chunk)
        total += float(np.sum(values_chunk, where=active, initial=0.0))
    return total


def _masked_weighted_structure_stats_2d(
    *,
    points: np.ndarray,
    areas: np.ndarray,
    weights_raw: np.ndarray,
    mask: np.ndarray,
    chunk_size: int = _MASKED_STRUCTURE_CHUNK_ITEMS,
) -> tuple[np.ndarray, np.ndarray, float]:
    point_arr = np.asarray(points, dtype=float)
    area_arr = np.asarray(areas, dtype=float).reshape(-1)
    weight_raw_arr = np.asarray(weights_raw, dtype=float).reshape(-1)
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    if point_arr.ndim != 2 or point_arr.shape[1] < 2:
        raise ValueError("points must be a 2D array with at least two columns.")
    if (
        point_arr.shape[0] != area_arr.size
        or area_arr.size != weight_raw_arr.size
        or mask_arr.size != area_arr.size
    ):
        raise ValueError("points, areas, weights_raw, and mask must align.")
    if area_arr.size == 0:
        raise ValueError("masked structure inputs must be non-empty.")
    block_size = max(1, min(int(chunk_size), int(area_arr.size)))
    weight_work = np.empty(block_size, dtype=float)
    value_work = np.empty(block_size, dtype=float)

    def first_pass(*, use_raw_weights: bool) -> tuple[float, float, float, float]:
        weight_sum = 0.0
        x_sum = 0.0
        y_sum = 0.0
        equivalent_area = 0.0
        for start in range(0, int(area_arr.size), block_size):
            stop = min(start + block_size, int(area_arr.size))
            active = mask_arr[start:stop]
            if not bool(active.any()):
                continue
            weights = weight_work[: stop - start]
            if use_raw_weights:
                np.multiply(
                    weight_raw_arr[start:stop], area_arr[start:stop], out=weights
                )
            else:
                weights[:] = area_arr[start:stop]
            weight_sum += float(np.sum(weights, where=active, initial=0.0))
            equivalent_area += float(
                np.sum(area_arr[start:stop], where=active, initial=0.0)
            )
            values = value_work[: stop - start]
            np.multiply(point_arr[start:stop, 0], weights, out=values)
            x_sum += float(np.sum(values, where=active, initial=0.0))
            np.multiply(point_arr[start:stop, 1], weights, out=values)
            y_sum += float(np.sum(values, where=active, initial=0.0))
        return weight_sum, x_sum, y_sum, equivalent_area

    weight_sum, x_sum, y_sum, equivalent_area = first_pass(use_raw_weights=True)
    use_raw_weights = True
    if weight_sum <= 0.0:
        weight_sum, x_sum, y_sum, equivalent_area = first_pass(use_raw_weights=False)
        use_raw_weights = False
    if weight_sum <= 0.0:
        raise ValueError("masked structure weights must have positive sum.")

    centroid = np.array([x_sum / weight_sum, y_sum / weight_sum], dtype=float)
    cxx_sum = 0.0
    cxy_sum = 0.0
    cyy_sum = 0.0
    dx_work = np.empty(block_size, dtype=float)
    dy_work = np.empty(block_size, dtype=float)
    for start in range(0, int(area_arr.size), block_size):
        stop = min(start + block_size, int(area_arr.size))
        active = mask_arr[start:stop]
        if not bool(active.any()):
            continue
        weights = weight_work[: stop - start]
        if use_raw_weights:
            np.multiply(weight_raw_arr[start:stop], area_arr[start:stop], out=weights)
        else:
            weights[:] = area_arr[start:stop]
        dx = dx_work[: stop - start]
        dy = dy_work[: stop - start]
        values = value_work[: stop - start]
        np.subtract(point_arr[start:stop, 0], centroid[0], out=dx)
        np.subtract(point_arr[start:stop, 1], centroid[1], out=dy)
        np.square(dx, out=values)
        np.multiply(values, weights, out=values)
        cxx_sum += float(np.sum(values, where=active, initial=0.0))
        np.multiply(dx, dy, out=values)
        np.multiply(values, weights, out=values)
        cxy_sum += float(np.sum(values, where=active, initial=0.0))
        np.square(dy, out=values)
        np.multiply(values, weights, out=values)
        cyy_sum += float(np.sum(values, where=active, initial=0.0))
    covariance = np.array(
        [
            [cxx_sum / weight_sum, cxy_sum / weight_sum],
            [cxy_sum / weight_sum, cyy_sum / weight_sum],
        ],
        dtype=float,
    )
    return centroid, covariance, equivalent_area


def _weighted_structure(
    *,
    values: np.ndarray,
    points: np.ndarray,
    areas: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, float, float, float, float, float, float]:
    weights_raw = np.abs(values)
    mask = weights_raw >= threshold
    if not bool(mask.any()):
        index = int(np.argmax(weights_raw))
        mask[index] = True
    centroid, covariance, equivalent_area = _masked_weighted_structure_stats_2d(
        points=points,
        areas=areas,
        weights_raw=weights_raw,
        mask=mask,
    )
    eigvals = np.sort(np.linalg.eigvalsh(covariance))
    minor_var = max(float(eigvals[0]), 0.0)
    major_var = max(float(eigvals[-1]), 0.0)
    if major_var <= 0.0:
        eccentricity = 0.0
    else:
        eccentricity = math.sqrt(max(0.0, 1.0 - minor_var / major_var))
    major_axis = 4.0 * math.sqrt(major_var)
    minor_axis = 4.0 * math.sqrt(minor_var)
    return (
        mask,
        float(centroid[0]),
        float(centroid[1]),
        equivalent_area,
        eccentricity,
        major_axis,
        minor_axis,
    )


def _structure_metric_rows(
    *,
    model: EITLinearizedModel,
    sigma_recon_by_method: dict[str, np.ndarray],
    sigma_recon_full: np.ndarray,
) -> list[HoldoutStructureMetricRow]:
    sigma_true = _as_float_vector(model.sigma_true, name="model.sigma_true")
    sigma_ref = _as_float_vector(model.sigma_reference, name="model.sigma_reference")
    points = _parameter_points(model, sigma_true.size)
    areas = _cell_areas(model, sigma_true.size)
    truth_contrast = sigma_true - sigma_ref
    max_truth = float(np.max(np.abs(truth_contrast)))
    threshold = max(0.5 * max_truth, 1e-12)
    threshold_rule = "abs(contrast)>=0.5*max(abs(truth_contrast))"
    truth_mask, truth_x, truth_y, truth_area, truth_ecc, truth_major, truth_minor = (
        _weighted_structure(
            values=truth_contrast,
            points=points,
            areas=areas,
            threshold=threshold,
        )
    )

    rows: list[HoldoutStructureMetricRow] = []

    def append_row(recon_kind: str, sigma_values: np.ndarray) -> None:
        contrast = sigma_values - sigma_ref
        mask, cx, cy, area, ecc, major, minor = _weighted_structure(
            values=contrast,
            points=points,
            areas=areas,
            threshold=threshold,
        )
        artifact_area = _masked_area_sum(areas, mask, exclude_mask=truth_mask)
        artifact_energy = _masked_square_area_sum(
            contrast, areas, exclude_mask=truth_mask
        )
        artifact_peak = _masked_abs_peak(contrast, exclude_mask=truth_mask)
        if recon_kind == "truth":
            artifact_area = 0.0
            artifact_energy = 0.0
            artifact_peak = 0.0
            sigma_rmse = 0.0
            sigma_relative = 0.0
            sigma_mae = 0.0
            sigma_max = 0.0
            sigma_digits = math.inf
        else:
            error = sigma_values - sigma_true
            abs_error = np.abs(error)
            sigma_rmse = rmse(sigma_true, sigma_values)
            sigma_relative = _relative_rmse(sigma_true, sigma_values)
            sigma_mae = float(np.mean(abs_error))
            sigma_max = float(np.max(abs_error))
            sigma_digits = effective_digits_from_rmse(sigma_true, sigma_values)
        rows.append(
            HoldoutStructureMetricRow(
                recon_kind=recon_kind,
                threshold_rule=threshold_rule,
                centroid_x=cx,
                centroid_y=cy,
                centroid_error=math.hypot(cx - truth_x, cy - truth_y),
                equivalent_area=area if recon_kind != "truth" else truth_area,
                eccentricity=ecc if recon_kind != "truth" else truth_ecc,
                major_axis=major if recon_kind != "truth" else truth_major,
                minor_axis=minor if recon_kind != "truth" else truth_minor,
                artifact_area=artifact_area,
                artifact_energy=artifact_energy,
                artifact_peak=artifact_peak,
                sigma_rmse=sigma_rmse,
                sigma_relative_rmse=sigma_relative,
                sigma_mae=sigma_mae,
                sigma_max_abs_error=sigma_max,
                sigma_effective_digits=sigma_digits,
            )
        )

    append_row("truth", sigma_true)
    append_row("full_208", sigma_recon_full)
    for method_name, sigma_values in sigma_recon_by_method.items():
        append_row(method_name, sigma_values)
    return rows


def run_holdout_fit_diff(
    *,
    model: EITLinearizedModel,
    holdout: str = "far3",
    fit_methods: Iterable[str] = ("poly2", "poly3", "spline"),
    raw_160_baseline: bool = True,
    ridge: float = 1e-2,
    inverse_backend: str = "pyeidors-rm",
    rm_mode: str = "tikhonov",
    rm_form: str = "param",
    noser_exponent: float = 0.5,
) -> HoldoutFitDiffCase:
    """Run full 208, raw 160, and fitted 208 holdout reconstructions."""

    n_elec = int(model.n_elec or 16)
    point_rows, point_summary = build_holdout_point_audit(
        n_elec=n_elec, holdout=holdout
    )
    points_per_frame = point_summary.points_per_kept_frame
    train_points_per_frame = point_summary.points_per_train_frame
    if int(model.n_measurements) != point_summary.kept_208_count:
        raise ValueError(
            "model measurement count must match adjacent kept count: "
            f"{model.n_measurements} != {point_summary.kept_208_count}"
        )

    sigma_true = _as_float_vector(model.sigma_true, name="model.sigma_true")
    v_ref = _as_float_vector(model.voltage_reference, name="model.voltage_reference")
    v_true = _as_float_vector(model.voltage_true, name="model.voltage_true")
    sensitivity = _as_float_matrix(model.sensitivity, name="model.sensitivity")
    if sensitivity.shape != (v_true.size, sigma_true.size):
        raise ValueError("model sensitivity shape mismatch")

    sigma_recon_full = reconstruct_linearized_sigma(
        model=model,
        voltages=v_true,
        ridge=ridge,
        inverse_backend=inverse_backend,
        rm_mode=rm_mode,
        rm_form=rm_form,
        noser_exponent=noser_exponent,
    )
    train_indices, holdout_indices = _train_and_holdout_indices(
        point_rows,
        points_per_frame=points_per_frame,
    )
    full_diff = v_true - v_ref
    summaries: list[HoldoutFitDiffSummary] = []
    field_rows: list[HoldoutFitDiffFieldRow] = []
    sigma_recon_by_method: dict[str, np.ndarray] = {}
    fit_voltage_by_method: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    frame_curve_lookup: dict[int, HoldoutFitFrameCurve] = {}

    if raw_160_baseline:
        raw_model = _submodel(model, train_indices)
        sigma_raw = reconstruct_linearized_sigma(
            model=raw_model,
            voltages=raw_model.voltage_true,
            ridge=ridge,
            inverse_backend=inverse_backend,
            rm_mode=rm_mode,
            rm_form=rm_form,
            noser_exponent=noser_exponent,
        )
        sigma_recon_by_method["raw_160"] = sigma_raw
        summaries.append(
            _summarize_candidate(
                recon_method="raw_160",
                n_inverse_points=int(train_indices.size),
                frame_count=point_summary.frame_count,
                points_per_frame=points_per_frame,
                holdout_per_frame=3,
                train_points_per_frame=train_points_per_frame,
                holdout_voltage_rmse=math.nan,
                diff_voltage_rmse=_rmse_values_at_indices(
                    full_diff,
                    holdout_indices,
                ),
                sigma_true=sigma_true,
                sigma_recon_full=sigma_recon_full,
                sigma_recon_candidate=sigma_raw,
            )
        )
        field_rows.extend(
            _field_rows(
                recon_method="raw_160",
                sigma_true=sigma_true,
                sigma_recon_full=sigma_recon_full,
                sigma_recon_candidate=sigma_raw,
            )
        )

    for method in _validate_fit_methods(fit_methods):
        fit_ref, fit_true, fit_sens, method_curves = _fit_full_model_data(
            model=model,
            point_rows=point_rows,
            n_elec=n_elec,
            points_per_frame=points_per_frame,
            method=method,
        )
        fit_model = replace(
            model,
            voltage_reference=fit_ref,
            voltage_true=fit_true,
            sensitivity=fit_sens,
            n_measurements=int(fit_true.size),
        )
        sigma_fit = reconstruct_linearized_sigma(
            model=fit_model,
            voltages=fit_true,
            ridge=ridge,
            inverse_backend=inverse_backend,
            rm_mode=rm_mode,
            rm_form=rm_form,
            noser_exponent=noser_exponent,
        )
        recon_method = f"{method}_208"
        sigma_recon_by_method[recon_method] = sigma_fit
        fit_voltage_by_method[recon_method] = (fit_ref, fit_true, fit_sens)
        fit_diff = fit_true - fit_ref
        summaries.append(
            _summarize_candidate(
                recon_method=recon_method,
                n_inverse_points=int(fit_true.size),
                frame_count=point_summary.frame_count,
                points_per_frame=points_per_frame,
                holdout_per_frame=3,
                train_points_per_frame=train_points_per_frame,
                holdout_voltage_rmse=_rmse_pair_at_indices(
                    v_true,
                    fit_true,
                    holdout_indices,
                ),
                diff_voltage_rmse=rmse(full_diff, fit_diff),
                sigma_true=sigma_true,
                sigma_recon_full=sigma_recon_full,
                sigma_recon_candidate=sigma_fit,
            )
        )
        field_rows.extend(
            _field_rows(
                recon_method=recon_method,
                sigma_true=sigma_true,
                sigma_recon_full=sigma_recon_full,
                sigma_recon_candidate=sigma_fit,
            )
        )
        for curve in method_curves:
            existing = frame_curve_lookup.get(curve.stim_index)
            if existing is None:
                frame_curve_lookup[curve.stim_index] = curve
            else:
                merged_ref = dict(existing.fitted_reference_by_method)
                merged_ref.update(curve.fitted_reference_by_method)
                merged_true = dict(existing.fitted_anomaly_by_method)
                merged_true.update(curve.fitted_anomaly_by_method)
                merged = dict(existing.fitted_diff_by_method)
                merged.update(curve.fitted_diff_by_method)
                frame_curve_lookup[curve.stim_index] = replace(
                    existing,
                    fitted_reference_by_method=merged_ref,
                    fitted_anomaly_by_method=merged_true,
                    fitted_diff_by_method=merged,
                )

    structure_rows = _structure_metric_rows(
        model=model,
        sigma_recon_by_method=sigma_recon_by_method,
        sigma_recon_full=sigma_recon_full,
    )
    return HoldoutFitDiffCase(
        model=model,
        point_rows=point_rows,
        summaries=summaries,
        field_rows=field_rows,
        structure_rows=structure_rows,
        frame_curves=[frame_curve_lookup[idx] for idx in sorted(frame_curve_lookup)],
        sigma_recon_by_method=sigma_recon_by_method,
        sigma_recon_full=sigma_recon_full,
        fit_voltage_by_method=fit_voltage_by_method,
    )


def format_holdout_fit_report(case: HoldoutFitDiffCase) -> str:
    """Format a Chinese summary report for holdout comparison."""

    rows = sorted(
        case.summaries,
        key=lambda row: (
            row.delta_sigma_relative_rmse,
            row.diff_voltage_rmse if math.isfinite(row.diff_voltage_rmse) else math.inf,
        ),
    )
    raw = next((row for row in rows if row.recon_method == "raw_160"), None)
    harmful = []
    if raw is not None:
        struct_lookup = {row.recon_kind: row for row in case.structure_rows}
        raw_artifact = struct_lookup.get("raw_160")
        for row in rows:
            if not row.recon_method.endswith("_208"):
                continue
            artifact = struct_lookup.get(row.recon_method)
            worse_error = row.delta_sigma_relative_rmse > raw.delta_sigma_relative_rmse
            worse_artifact = (
                artifact is not None
                and raw_artifact is not None
                and artifact.artifact_energy > raw_artifact.artifact_energy
            )
            if worse_error or worse_artifact:
                harmful.append(row.recon_method)

    lines = [
        "# 远端 3 点扣除与拟合补点差分仿真报告",
        "",
        "## 方法排序",
        "",
        "| recon_method | n_inverse_points | diff_voltage_rmse | delta_sigma_relative_rmse | delta_sigma_effective_digits |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.recon_method} | {row.n_inverse_points} | "
            f"{row.diff_voltage_rmse:.12g} | {row.delta_sigma_relative_rmse:.12g} | "
            f"{row.delta_sigma_effective_digits:.12g} |"
        )
    lines.extend(["", "## 结论提示", ""])
    if harmful:
        lines.append(
            "- 下列拟合方法劣于 `raw_160` 的相对误差或伪影能量："
            + ", ".join(f"`{item}`" for item in harmful)
            + "；拟合引入误差可能超过被删点信息量收益。"
        )
    else:
        lines.append(
            "- 本次结果未发现拟合组相对 `raw_160` 同时表现出更差的结构误差风险。"
        )
    lines.append("- 该判断只针对当前 `sigma_true`、网格、正则化和拟合方法。")
    return "\n".join(lines) + "\n"


def _draw_field(
    ax,
    case: HoldoutFitDiffCase,
    values: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
):
    model = case.model
    if model.mesh_points is not None and model.mesh_cells is not None:
        cells = np.asarray(model.mesh_cells, dtype=np.int32)
        points = np.asarray(model.mesh_points, dtype=float)
        if cells.ndim == 2 and cells.shape[1] == 3:
            import matplotlib.tri as mtri

            triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
            return ax.tripcolor(
                triangulation,
                facecolors=values,
                shading="flat",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolors="#ffffff",
                linewidth=0.15,
            )
    points = _parameter_points(model, values.size)
    return ax.scatter(
        points[:, 0],
        points[:, 1],
        c=values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=42,
        marker="s",
        linewidths=0.0,
    )


def _field_value_range(fields: list[tuple[str, np.ndarray]]) -> tuple[float, float]:
    if not fields:
        raise ValueError("fields must not be empty.")
    first = np.asarray(fields[0][1], dtype=float)
    sigma_min = float(np.min(first))
    sigma_max = float(np.max(first))
    for _, values in fields[1:]:
        arr = np.asarray(values, dtype=float)
        sigma_min = float(np.minimum(sigma_min, float(np.min(arr))))
        sigma_max = float(np.maximum(sigma_max, float(np.max(arr))))
    return sigma_min, sigma_max


def _prediction_marker_offsets(methods: Iterable[str]) -> dict[str, float]:
    """Return small visual-only x offsets so overlapping prediction markers show."""

    names = sorted(dict.fromkeys(str(method) for method in methods))
    if len(names) <= 1:
        return {name: 0.0 for name in names}
    offsets = np.linspace(-0.18, 0.18, len(names), dtype=float)
    return {name: float(offsets[idx]) for idx, name in enumerate(names)}


def plot_holdout_fit_curves(
    case: HoldoutFitDiffCase,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot all 16 per-frame absolute U-shaped voltage curves."""

    if not case.frame_curves:
        raise ValueError("case must include fitted frame curves")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    n_frames = len(case.frame_curves)
    n_cols = 4
    n_rows = int(math.ceil(n_frames / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(13.5, 2.4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Holdout fit curves: absolute 13-point U curves")
    colors = {"poly2": "#9467bd", "poly3": "#8c564b", "spline": "#17becf"}
    markers = {"poly2": "D", "poly3": "s", "spline": "P"}
    for ax, curve in zip(axes.ravel(), case.frame_curves, strict=False):
        ax.plot(
            curve.x_all,
            curve.voltage_anomaly_full,
            color="#1f77b4",
            linewidth=1.25,
            label=FIT_CURVE_LEGEND_LABELS["target_full"],
            zorder=2,
        )
        if np.max(np.abs(curve.voltage_reference_full)) > 0.0:
            ax.plot(
                curve.x_all,
                curve.voltage_reference_full,
                color="#222222",
                linestyle=(0, (3.0, 2.0)),
                linewidth=0.95,
                alpha=0.78,
                label=FIT_CURVE_LEGEND_LABELS["reference_full"],
                zorder=3,
            )
        ax.scatter(
            curve.x_all[curve.train_mask],
            curve.voltage_anomaly_full[curve.train_mask],
            color="#2ca02c",
            s=20,
            label=FIT_CURVE_LEGEND_LABELS["fit_input"],
            zorder=4,
        )
        ax.scatter(
            curve.x_all[curve.holdout_mask],
            curve.voltage_anomaly_full[curve.holdout_mask],
            color="#ff7f0e",
            marker="x",
            s=34,
            label=FIT_CURVE_LEGEND_LABELS["withheld_true"],
            zorder=5,
        )
        marker_offsets = _prediction_marker_offsets(curve.fitted_anomaly_by_method)
        for method, fitted in sorted(curve.fitted_anomaly_by_method.items()):
            ax.scatter(
                curve.x_all[curve.holdout_mask] + marker_offsets[method],
                fitted[curve.holdout_mask],
                color=colors.get(method, "#444444"),
                marker=markers.get(method, "D"),
                s=30,
                label=FIT_CURVE_LEGEND_LABELS.get(method, f"{method} pred: 3 pts"),
                alpha=0.95,
                edgecolors="#ffffff",
                linewidths=0.45,
                zorder=6,
            )
        ax.set_title(f"stim {curve.stim_index}", fontsize=9)
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[n_frames:]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_holdout_recon_compare(
    case: HoldoutFitDiffCase,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot truth, full 208, raw 160 and fitted 208 reconstructions."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fields: list[tuple[str, np.ndarray]] = [
        ("truth", case.model.sigma_true),
        ("full_208", case.sigma_recon_full),
    ]
    fields.extend(case.sigma_recon_by_method.items())
    sigma_min, sigma_max = _field_value_range(fields)
    errors = [
        np.asarray(values, dtype=float) - case.model.sigma_true
        for _, values in fields[1:]
    ]
    error_lim = float(max(np.max(np.abs(error)) for error in errors))
    error_lim = max(error_lim, 1e-12)

    n_cols = len(fields)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(2.45 * n_cols, 5.4),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Holdout recon compare")
    for col_idx, (label, values) in enumerate(fields):
        ax = axes[0, col_idx]
        image = _draw_field(
            ax,
            case,
            np.asarray(values, dtype=float),
            vmin=sigma_min,
            vmax=sigma_max,
            cmap="viridis",
        )
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

        err_ax = axes[1, col_idx]
        error_values = (
            np.zeros_like(case.model.sigma_true)
            if col_idx == 0
            else np.asarray(values) - case.model.sigma_true
        )
        err_image = _draw_field(
            err_ax,
            case,
            error_values,
            vmin=-error_lim,
            vmax=error_lim,
            cmap="coolwarm",
        )
        err_ax.set_title("error", fontsize=9)
        err_ax.set_aspect("equal", adjustable="box")
        err_ax.set_xticks([])
        err_ax.set_yticks([])
        fig.colorbar(err_image, ax=err_ax, fraction=0.046, pad=0.02)

    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def plot_holdout_fit_summary(
    case: HoldoutFitDiffCase,
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Plot reconstruction-error and voltage-error summaries."""

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
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), constrained_layout=True)
    fig.suptitle("Holdout reconstruction summary")
    axes[0].bar(x, [row.delta_sigma_relative_rmse for row in rows], color="#1f77b4")
    axes[0].axhline(0.0, color="#444444", linewidth=0.8)
    axes[0].set_ylabel("Delta relative RMSE")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, [row.diff_voltage_rmse for row in rows], color="#ff7f0e")
    axes[1].set_ylabel("Diff voltage RMSE")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output
