"""Shared helpers for the real reconstruction gallery scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    denom = np.linalg.norm(np.asarray(left, dtype=np.float64)) + 1e-12
    return float(np.linalg.norm(diff) / denom)


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(diff**2)))


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float64).ravel()
    right_arr = np.asarray(right, dtype=np.float64).ravel()
    if left_arr.size == 0 or right_arr.size == 0:
        return float("nan")
    if np.allclose(left_arr, left_arr[0]) or np.allclose(right_arr, right_arr[0]):
        return 1.0 if np.allclose(left_arr, right_arr) else 0.0
    return float(np.corrcoef(left_arr, right_arr)[0, 1])


def truth_metrics(
    *,
    truth: np.ndarray,
    recon: np.ndarray,
    coords: np.ndarray,
    anomalies: list[dict[str, Any]],
    background_conductivity: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "relative_l2": relative_l2(truth, recon),
        "rmse": rmse(truth, recon),
        "pearson": safe_pearson(truth, recon),
    }
    background_mask = np.ones(truth.shape[0], dtype=bool)
    for item in anomalies:
        center = np.asarray(item["center"], dtype=np.float64)
        radius = float(item["radius"])
        roi = np.linalg.norm(coords - center[None, :], axis=1) <= radius
        background_mask &= ~roi
        truth_mean = float(np.mean(truth[roi]))
        recon_mean = float(np.mean(recon[roi]))
        bg_mean = (
            float(np.mean(recon[background_mask]))
            if np.any(background_mask)
            else float(background_conductivity)
        )
        denom = truth_mean - float(background_conductivity)
        metrics[f"contrast_recovery_{item['label']}"] = (
            0.0 if abs(denom) <= 1e-12 else float((recon_mean - bg_mean) / denom)
        )
        metrics[f"roi_mean_{item['label']}"] = recon_mean
    background_mean = (
        float(np.mean(recon[background_mask]))
        if np.any(background_mask)
        else float(background_conductivity)
    )
    metrics["background_bias"] = float(background_mean - float(background_conductivity))
    return metrics


def consistency_metrics(
    *,
    dim: int,
    baseline_cpu_meas: np.ndarray | None,
    baseline_gpu_meas: np.ndarray | None,
    target_cpu_meas: np.ndarray,
    target_gpu_meas: np.ndarray,
    cpu_recon: np.ndarray,
    gpu_recon: np.ndarray,
    measurement_rel_tol: float,
    image_rel_tol: float,
    image_rmse_tol_by_dim: dict[int, float],
) -> dict[str, Any]:
    baseline_rel = (
        relative_l2(baseline_cpu_meas, baseline_gpu_meas)
        if baseline_cpu_meas is not None and baseline_gpu_meas is not None
        else None
    )
    target_rel = relative_l2(target_cpu_meas, target_gpu_meas)
    image_rel = relative_l2(cpu_recon, gpu_recon)
    image_rmse = rmse(cpu_recon, gpu_recon)
    baseline_pass = bool(baseline_rel is None or float(baseline_rel) <= measurement_rel_tol)
    target_pass = bool(float(target_rel) <= measurement_rel_tol)
    measurement_pass = bool(baseline_pass and target_pass)
    image_pass = bool(image_rel <= image_rel_tol and image_rmse <= image_rmse_tol_by_dim[int(dim)])
    return {
        "baseline_measurement_relative_l2": baseline_rel,
        "target_measurement_relative_l2": target_rel,
        "image_relative_l2": image_rel,
        "image_rmse": image_rmse,
        "image_pearson": safe_pearson(cpu_recon, gpu_recon),
        "image_max_abs_diff": float(np.max(np.abs(np.asarray(cpu_recon) - np.asarray(gpu_recon)))),
        "measurement_threshold": measurement_rel_tol,
        "image_relative_l2_threshold": image_rel_tol,
        "image_rmse_threshold": image_rmse_tol_by_dim[int(dim)],
        "baseline_measurement_pass": baseline_pass,
        "target_measurement_pass": target_pass,
        "measurement_pass": measurement_pass,
        "image_pass": image_pass,
        "passed": bool(measurement_pass and image_pass),
    }


def save_case_data(path: Path, payload: dict[str, Any]) -> None:
    arrays = {k: v for k, v in payload.items() if isinstance(v, np.ndarray)}
    scalars = {k: v for k, v in payload.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrays, meta=np.array(json.dumps(jsonable(scalars)), dtype=object))
