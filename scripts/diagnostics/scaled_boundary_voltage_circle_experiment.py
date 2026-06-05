#!/usr/bin/env python3
"""Evaluate circular EIT reconstructions after fixed boundary-voltage scaling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from scripts.common.array_metrics import mean_where, safe_finite_pearson_correlation

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import EITData, MeshConfig, PatternConfig
from pyeidors.io._json import json_ready
from pyeidors.runtime_paths import pyeidors_cache_path, pyeidors_output_path


BACKGROUND_SIGMA = 1.0
TARGET_SIGMA = 1.5
DOMAIN_RADIUS = 1.0
DEFAULT_CASES = (
    ("center", 0.0, 0.0),
    ("off_center", 0.35, 0.18),
)


@dataclass(frozen=True)
class CircleCase:
    label: str
    center_x: float
    center_y: float

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_x, self.center_y], dtype=np.float64)

    @property
    def radial_offset(self) -> float:
        return float(np.hypot(self.center_x, self.center_y))


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str
    n_elec: int
    domain_radius: float
    mesh_size: float
    electrode_coverage: float
    drive_mode: str
    drive_value: float
    contact_impedance: float
    background_sigma: float
    target_sigma: float
    target_radius: float
    voltage_scale: float
    scale_mode: str
    difference_mode: str
    difference_orientation: str
    difference_preset: str
    hyperparameter: float
    threshold_fraction: float
    grid_resolution: int
    cases: tuple[CircleCase, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=pyeidors_output_path("diagnostics", "scaled_boundary_voltage_circle"),
    )
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--domain-radius", type=float, default=DOMAIN_RADIUS)
    parser.add_argument("--mesh-size", type=float, default=0.08)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--drive-mode", default="normalized")
    parser.add_argument("--drive-value", type=float, default=1.0)
    parser.add_argument("--contact-impedance", type=float, default=1.0e-5)
    parser.add_argument("--background-sigma", type=float, default=BACKGROUND_SIGMA)
    parser.add_argument("--target-sigma", type=float, default=TARGET_SIGMA)
    parser.add_argument("--target-radius", type=float, default=0.18)
    parser.add_argument("--voltage-scale", type=float, default=1.5)
    parser.add_argument(
        "--scale-mode",
        choices=("difference", "absolute"),
        default="difference",
        help=(
            "difference: U_scaled = U_ref + k*(U_target-U_ref); "
            "absolute: U_scaled = k*U_target"
        ),
    )
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    parser.add_argument("--threshold-fraction", type=float, default=0.5)
    parser.add_argument("--grid-resolution", type=int, default=220)
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        metavar="LABEL:X,Y",
        help=(
            "Circular target centre in model coordinates. Repeat to add cases. "
            "Default: center:0,0 and off_center:0.35,0.18."
        ),
    )
    return parser.parse_args()


def parse_cases(values: list[str] | None) -> tuple[CircleCase, ...]:
    if not values:
        return tuple(
            CircleCase(str(label), float(x), float(y)) for label, x, y in DEFAULT_CASES
        )
    cases: list[CircleCase] = []
    for raw in values:
        try:
            label, coords = str(raw).split(":", 1)
            x_text, y_text = coords.split(",", 1)
        except ValueError as exc:
            raise ValueError(
                "--case must use LABEL:X,Y, for example off:0.35,0.18"
            ) from exc
        clean_label = label.strip().replace(" ", "_")
        if not clean_label:
            raise ValueError("--case label must not be empty")
        cases.append(CircleCase(clean_label, float(x_text), float(y_text)))
    return tuple(cases)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    return safe_finite_pearson_correlation(left, right)


def center_of_mass(points: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(w) & (w > 0.0)
    if not np.any(valid):
        return None
    total = float(np.sum(w[valid]))
    if total <= 0.0 or not math.isfinite(total):
        return None
    return np.sum(pts[valid, :2] * w[valid, None], axis=0) / total


def weighted_eccentricity(points: np.ndarray, weights: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(w) & (w > 0.0)
    if int(np.count_nonzero(valid)) < 3:
        return float("nan")
    total = float(np.sum(w[valid]))
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    xy = pts[valid, :2]
    centroid = np.sum(xy * w[valid, None], axis=0) / total
    shifted = xy - centroid[None, :]
    cov = (shifted * w[valid, None]).T @ shifted / total
    eigvals = np.linalg.eigvalsh(cov)
    major = float(np.max(eigvals))
    minor = float(np.min(eigvals))
    if major <= 0.0:
        return float("nan")
    ratio = max(0.0, min(1.0, minor / major))
    return float(math.sqrt(1.0 - ratio))


def threshold_metrics(
    *,
    coords: np.ndarray,
    truth_delta: np.ndarray,
    recon_delta: np.ndarray,
    threshold_fraction: float,
    target_center: np.ndarray,
    target_radius: float,
) -> dict[str, float]:
    truth_abs = np.abs(np.asarray(truth_delta, dtype=np.float64))
    recon_abs = np.abs(np.asarray(recon_delta, dtype=np.float64))
    truth_peak = float(np.max(truth_abs)) if truth_abs.size else 0.0
    truth_mask = truth_abs > 0.5 * truth_peak
    recon_peak = float(np.max(recon_abs)) if recon_abs.size else 0.0
    if recon_peak <= 0.0 or not math.isfinite(recon_peak):
        recon_mask = np.zeros_like(truth_mask)
    else:
        recon_mask = recon_abs >= float(threshold_fraction) * recon_peak

    union = truth_mask | recon_mask
    intersection = truth_mask & recon_mask
    iou = (
        float(np.count_nonzero(intersection) / np.count_nonzero(union))
        if np.any(union)
        else float("nan")
    )
    area_ratio = (
        float(np.count_nonzero(recon_mask) / np.count_nonzero(truth_mask))
        if np.any(truth_mask)
        else float("nan")
    )
    com = center_of_mass(coords, recon_abs * recon_mask.astype(np.float64))
    localization_error = (
        float(np.linalg.norm(com - target_center[:2]))
        if com is not None
        else float("nan")
    )
    peak_idx = int(np.nanargmax(recon_abs)) if recon_abs.size else 0
    peak_error = float(np.linalg.norm(coords[peak_idx, :2] - target_center[:2]))
    eccentricity = weighted_eccentricity(
        coords,
        recon_abs * recon_mask.astype(np.float64),
    )
    truth_eccentricity = weighted_eccentricity(
        coords,
        truth_abs * truth_mask.astype(np.float64),
    )
    return {
        "iou_at_threshold": iou,
        "area_ratio_at_threshold": area_ratio,
        "localization_error": localization_error,
        "localization_error_over_radius": localization_error / target_radius,
        "peak_error": peak_error,
        "recon_eccentricity": eccentricity,
        "truth_eccentricity": truth_eccentricity,
        "eccentricity_error": abs(eccentricity - truth_eccentricity),
    }


def image_metrics(
    *,
    coords: np.ndarray,
    truth: np.ndarray,
    recon: np.ndarray,
    background_sigma: float,
    target_sigma: float,
    target_center: np.ndarray,
    target_radius: float,
    threshold_fraction: float,
) -> dict[str, float]:
    truth_delta = np.asarray(truth, dtype=np.float64) - float(background_sigma)
    recon_delta = np.asarray(recon, dtype=np.float64) - float(background_sigma)
    diff = recon_delta - truth_delta
    target_mask = (
        np.linalg.norm(coords[:, :2] - target_center[:2][None, :], axis=1)
        <= target_radius
    )
    background_mask = ~target_mask
    truth_contrast = float(target_sigma) - float(background_sigma)
    recon_target_mean = mean_where(recon_delta, target_mask)
    recon_background_mean = mean_where(recon_delta, background_mask)
    contrast_recovery = (
        float((recon_target_mean - recon_background_mean) / truth_contrast)
        if abs(truth_contrast) > 1e-12
        else float("nan")
    )
    metrics = {
        "pearson_delta": safe_pearson(truth_delta, recon_delta),
        "relative_l2_delta": float(
            np.linalg.norm(diff) / (np.linalg.norm(truth_delta) + 1e-12)
        ),
        "rmse_delta": float(np.sqrt(np.mean(diff * diff))),
        "recon_target_delta_mean": recon_target_mean,
        "recon_background_delta_mean": recon_background_mean,
        "contrast_recovery": contrast_recovery,
        "peak_delta": float(np.max(recon_delta)),
        "min_delta": float(np.min(recon_delta)),
    }
    metrics.update(
        threshold_metrics(
            coords=coords,
            truth_delta=truth_delta,
            recon_delta=recon_delta,
            threshold_fraction=threshold_fraction,
            target_center=target_center,
            target_radius=target_radius,
        )
    )
    return metrics


def compare_reconstructions(
    *,
    original: np.ndarray,
    scaled: np.ndarray,
    background_sigma: float,
    target_center: np.ndarray,
    target_radius: float,
    coords: np.ndarray,
    threshold_fraction: float,
) -> dict[str, float]:
    original_delta = np.asarray(original, dtype=np.float64) - float(background_sigma)
    scaled_delta = np.asarray(scaled, dtype=np.float64) - float(background_sigma)
    diff = scaled_delta - original_delta
    original_peak = (
        float(np.max(np.abs(original_delta))) if original_delta.size else 0.0
    )
    scaled_peak = float(np.max(np.abs(scaled_delta))) if scaled_delta.size else 0.0
    target_mask = (
        np.linalg.norm(coords[:, :2] - target_center[:2][None, :], axis=1)
        <= target_radius
    )
    original_target_mean = mean_where(original_delta, target_mask)
    scaled_target_mean = mean_where(scaled_delta, target_mask)
    original_shape = threshold_metrics(
        coords=coords,
        truth_delta=original_delta,
        recon_delta=original_delta,
        threshold_fraction=threshold_fraction,
        target_center=target_center,
        target_radius=target_radius,
    )
    scaled_shape = threshold_metrics(
        coords=coords,
        truth_delta=scaled_delta,
        recon_delta=scaled_delta,
        threshold_fraction=threshold_fraction,
        target_center=target_center,
        target_radius=target_radius,
    )
    return {
        "scaled_vs_original_pearson": safe_pearson(original_delta, scaled_delta),
        "scaled_vs_original_relative_l2": float(
            np.linalg.norm(diff) / (np.linalg.norm(original_delta) + 1e-12)
        ),
        "peak_abs_ratio_scaled_over_original": float(
            scaled_peak / (original_peak + 1e-12)
        ),
        "target_mean_ratio_scaled_over_original": float(
            scaled_target_mean / (original_target_mean + 1e-12)
        ),
        "localization_shift_scaled_minus_original": float(
            scaled_shape["localization_error"] - original_shape["localization_error"]
        ),
        "eccentricity_shift_scaled_minus_original": float(
            scaled_shape["recon_eccentricity"] - original_shape["recon_eccentricity"]
        ),
    }


def build_system(cfg: ExperimentConfig) -> EITSystem:
    pattern = PatternConfig(
        n_elec=cfg.n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode=cfg.drive_mode,
        drive_value=cfg.drive_value,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    mesh_config = MeshConfig(
        dimension=2,
        radius=cfg.domain_radius,
        mesh_size=cfg.mesh_size,
        electrode_coverage=cfg.electrode_coverage,
    )
    system = EITSystem(
        n_elec=cfg.n_elec,
        pattern_config=pattern,
        mesh_config=mesh_config,
        contact_impedance=np.full(cfg.n_elec, cfg.contact_impedance, dtype=float),
        base_conductivity=cfg.background_sigma,
        difference_mode=cfg.difference_mode,
        difference_orientation=cfg.difference_orientation,
        regularization_type="noser",
        regularization_alpha=1.0,
        hyperparameter=cfg.hyperparameter,
        jacobian_background_conductivity=cfg.background_sigma,
        difference_preset=cfg.difference_preset,
        linear_backend="scipy",
        performance_mode="safe",
        solver_mode="strict",
        cache_scope="both",
        cache_dir=str(pyeidors_cache_path("diagnostics", "scaled_boundary_voltage")),
    )
    system.setup(mesh_source="generated", dimension=2)
    if system.reconstructor is None:
        raise RuntimeError("EIT reconstructor did not initialize.")
    system.reconstructor.verbose = False
    return system


def scaled_measurement_data(
    *,
    target_data: EITData,
    reference_data: EITData,
    scale: float,
    scale_mode: str,
) -> EITData:
    target = np.asarray(target_data.meas, dtype=np.float64)
    reference = np.asarray(reference_data.meas, dtype=np.float64)
    if scale_mode == "difference":
        meas = reference + float(scale) * (target - reference)
    elif scale_mode == "absolute":
        meas = float(scale) * target
    else:
        raise ValueError(f"Unsupported scale_mode={scale_mode!r}")
    return EITData(
        meas=meas,
        stim_pattern=target_data.stim_pattern,
        n_elec=target_data.n_elec,
        n_stim=target_data.n_stim,
        n_meas=target_data.n_meas,
        type=f"scaled_{scale_mode}",
    )


def voltage_metrics(
    *,
    target: np.ndarray,
    reference: np.ndarray,
    scaled: np.ndarray,
    scale: float,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    scaled = np.asarray(scaled, dtype=np.float64)
    original_diff = target - reference
    scaled_diff = scaled - reference
    return {
        "voltage_scale": float(scale),
        "target_voltage_l2": float(np.linalg.norm(target)),
        "reference_voltage_l2": float(np.linalg.norm(reference)),
        "original_diff_voltage_l2": float(np.linalg.norm(original_diff)),
        "scaled_diff_voltage_l2": float(np.linalg.norm(scaled_diff)),
        "scaled_diff_over_original_diff_l2": float(
            np.linalg.norm(scaled_diff) / (np.linalg.norm(original_diff) + 1e-12)
        ),
        "voltage_diff_pearson": safe_pearson(original_diff, scaled_diff),
        "voltage_abs_pearson": safe_pearson(target, scaled),
    }


def run_case(
    *,
    system: EITSystem,
    case: CircleCase,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    if case.radial_offset + cfg.target_radius > cfg.domain_radius + 1e-12:
        raise ValueError(
            f"Case {case.label!r} is outside the circular domain: "
            f"offset={case.radial_offset:g}, radius={cfg.target_radius:g}, "
            f"domain={cfg.domain_radius:g}."
        )

    baseline_image = system.create_homogeneous_image(cfg.background_sigma)
    baseline_data = system.forward_solve(baseline_image)
    phantom = system.add_phantom(
        base_conductivity=cfg.background_sigma,
        phantom_conductivity=cfg.target_sigma,
        phantom_center=(case.center_x, case.center_y),
        phantom_radius=cfg.target_radius,
    )
    target_data = system.forward_solve(phantom)
    scaled_data = scaled_measurement_data(
        target_data=target_data,
        reference_data=baseline_data,
        scale=cfg.voltage_scale,
        scale_mode=cfg.scale_mode,
    )

    t0 = time.perf_counter()
    original_result = system.difference_reconstruct(
        measurement_data=target_data,
        reference_data=baseline_data,
        initial_image=baseline_image,
        metadata={"case": case.label, "variant": "original"},
    )
    original_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    scaled_result = system.difference_reconstruct(
        measurement_data=scaled_data,
        reference_data=baseline_data,
        initial_image=baseline_image,
        metadata={"case": case.label, "variant": "scaled"},
    )
    scaled_seconds = time.perf_counter() - t1

    coords = system.fwd_model.V_sigma.tabulate_dof_coordinates()
    truth = np.asarray(phantom.elem_data, dtype=np.float64)
    original_recon = np.asarray(original_result.conductivity, dtype=np.float64)
    scaled_recon = np.asarray(scaled_result.conductivity, dtype=np.float64)
    original_metrics = image_metrics(
        coords=coords,
        truth=truth,
        recon=original_recon,
        background_sigma=cfg.background_sigma,
        target_sigma=cfg.target_sigma,
        target_center=case.center,
        target_radius=cfg.target_radius,
        threshold_fraction=cfg.threshold_fraction,
    )
    scaled_metrics = image_metrics(
        coords=coords,
        truth=truth,
        recon=scaled_recon,
        background_sigma=cfg.background_sigma,
        target_sigma=cfg.target_sigma,
        target_center=case.center,
        target_radius=cfg.target_radius,
        threshold_fraction=cfg.threshold_fraction,
    )
    recon_comparison = compare_reconstructions(
        original=original_recon,
        scaled=scaled_recon,
        background_sigma=cfg.background_sigma,
        target_center=case.center,
        target_radius=cfg.target_radius,
        coords=coords,
        threshold_fraction=cfg.threshold_fraction,
    )
    volt_metrics = voltage_metrics(
        target=target_data.meas,
        reference=baseline_data.meas,
        scaled=scaled_data.meas,
        scale=cfg.voltage_scale,
    )
    info = system.get_system_info()
    row_base = {
        "case": case.label,
        "center_x": case.center_x,
        "center_y": case.center_y,
        "radial_offset": case.radial_offset,
        "target_radius": cfg.target_radius,
        "n_elec": cfg.n_elec,
        "n_measurements": int(info["n_measurements"]),
        "n_elements": int(info["n_elements"]),
        "voltage_scale": cfg.voltage_scale,
        "scale_mode": cfg.scale_mode,
    }
    rows = [
        {
            **row_base,
            "variant": "original",
            "solve_seconds": original_seconds,
            "measurement_relative_error": original_result.relative_error,
            "measurement_l2_error": original_result.l2_error,
            **original_metrics,
        },
        {
            **row_base,
            "variant": "scaled",
            "solve_seconds": scaled_seconds,
            "measurement_relative_error": scaled_result.relative_error,
            "measurement_l2_error": scaled_result.l2_error,
            **scaled_metrics,
            **recon_comparison,
        },
    ]
    return {
        "rows": rows,
        "coords": coords,
        "truth": truth,
        "original_recon": original_recon,
        "scaled_recon": scaled_recon,
        "reference_meas": np.asarray(baseline_data.meas, dtype=np.float64),
        "target_meas": np.asarray(target_data.meas, dtype=np.float64),
        "scaled_meas": np.asarray(scaled_data.meas, dtype=np.float64),
        "voltage_metrics": {**row_base, **volt_metrics},
    }


def sample_to_grid(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    radius: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.interpolate import griddata

    axis = np.linspace(-radius, radius, int(resolution))
    xg, yg = np.meshgrid(axis, axis)
    query = _query_points_2d(xg, yg)
    sampled = griddata(coords[:, :2], values, query, method="linear")
    missing = np.isnan(sampled)
    if np.any(missing):
        sampled[missing] = griddata(
            coords[:, :2],
            values,
            query[missing],
            method="nearest",
        )
    image = sampled.reshape(xg.shape)
    _apply_outside_radius_nan(image, xg, yg, radius)
    return xg, yg, image


def _query_points_2d(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    x_flat = np.asarray(x_grid).reshape(-1)
    y_flat = np.asarray(y_grid).reshape(-1)
    if x_flat.size != y_flat.size:
        raise ValueError("x/y grid sizes must match")
    query = np.empty((x_flat.size, 2), dtype=np.result_type(x_flat, y_flat, np.float64))
    query[:, 0] = x_flat
    query[:, 1] = y_flat
    return query


def _apply_outside_radius_nan(
    image: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    radius: float,
) -> None:
    if image.shape != x_grid.shape or image.shape != y_grid.shape:
        raise ValueError("image and grid shapes must match")
    radius_sq = float(radius) * float(radius)
    row_width = int(image.shape[1])
    squared = np.empty(row_width, dtype=np.float64)
    y_squared = np.empty(row_width, dtype=np.float64)
    mask = np.empty(row_width, dtype=bool)
    for row in range(int(image.shape[0])):
        np.multiply(x_grid[row, :], x_grid[row, :], out=squared)
        np.multiply(y_grid[row, :], y_grid[row, :], out=y_squared)
        np.add(squared, y_squared, out=squared)
        np.greater(squared, radius_sq, out=mask)
        row_values = image[row, :]
        row_values[mask] = np.nan


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    for font_path in (
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def save_case_plot(path: Path, payload: dict[str, Any], cfg: ExperimentConfig) -> None:
    configure_matplotlib()
    import matplotlib.pyplot as plt

    coords = payload["coords"]
    truth_delta = payload["truth"] - cfg.background_sigma
    original_delta = payload["original_recon"] - cfg.background_sigma
    scaled_delta = payload["scaled_recon"] - cfg.background_sigma
    delta_delta = scaled_delta - original_delta
    _, _, truth_grid = sample_to_grid(
        coords,
        truth_delta,
        radius=cfg.domain_radius,
        resolution=cfg.grid_resolution,
    )
    _, _, original_grid = sample_to_grid(
        coords,
        original_delta,
        radius=cfg.domain_radius,
        resolution=cfg.grid_resolution,
    )
    _, _, scaled_grid = sample_to_grid(
        coords,
        scaled_delta,
        radius=cfg.domain_radius,
        resolution=cfg.grid_resolution,
    )
    _, _, delta_grid = sample_to_grid(
        coords,
        delta_delta,
        radius=cfg.domain_radius,
        resolution=cfg.grid_resolution,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.5, 7.2),
        constrained_layout=True,
    )
    extent = [
        -cfg.domain_radius,
        cfg.domain_radius,
        -cfg.domain_radius,
        cfg.domain_radius,
    ]
    image_specs = [
        (truth_grid, "Truth", abs(cfg.target_sigma - cfg.background_sigma)),
        (
            original_grid,
            "Original U recon",
            abs(cfg.target_sigma - cfg.background_sigma),
        ),
        (
            scaled_grid,
            f"Scaled U recon, k={cfg.voltage_scale:g}",
            abs(cfg.target_sigma - cfg.background_sigma) * cfg.voltage_scale,
        ),
        (delta_grid, "Scaled - original", np.nanmax(np.abs(delta_grid))),
    ]
    for ax, (image, title, vlim) in zip(axes.flat[:4], image_specs):
        vlim = float(vlim)
        if not math.isfinite(vlim) or vlim <= 0.0:
            vlim = 1.0
        im = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                cfg.domain_radius,
                color="black",
                fill=False,
                linewidth=0.8,
                alpha=0.7,
            )
        )
        fig.colorbar(im, ax=ax, shrink=0.78)

    ref = payload["reference_meas"]
    target = payload["target_meas"]
    scaled = payload["scaled_meas"]
    idx = np.arange(target.size)
    axes[1, 1].plot(idx, target - ref, label="Original diff", linewidth=1.2)
    axes[1, 1].plot(idx, scaled - ref, label="Scaled diff", linewidth=1.2)
    axes[1, 1].set_title("Boundary voltage difference")
    axes[1, 1].set_xlabel("Measurement index")
    axes[1, 1].set_ylabel("Voltage difference")
    axes[1, 1].legend(frameon=False, fontsize=9)

    axes[1, 2].plot(idx, target, label="Target U", linewidth=1.1)
    axes[1, 2].plot(idx, scaled, label="Scaled U", linewidth=1.1)
    axes[1, 2].plot(idx, ref, label="Reference U", linewidth=0.9, alpha=0.75)
    axes[1, 2].set_title("Absolute boundary voltage")
    axes[1, 2].set_xlabel("Measurement index")
    axes[1, 2].set_ylabel("Voltage")
    axes[1, 2].legend(frameon=False, fontsize=9)

    fig.suptitle(
        f"{payload['rows'][0]['case']}: circular target and scaled boundary voltages",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "case",
        "variant",
        "scale_mode",
        "voltage_scale",
        "center_x",
        "center_y",
        "radial_offset",
        "target_radius",
        "n_elec",
        "n_measurements",
        "n_elements",
        "pearson_delta",
        "relative_l2_delta",
        "rmse_delta",
        "contrast_recovery",
        "iou_at_threshold",
        "area_ratio_at_threshold",
        "localization_error",
        "peak_error",
        "recon_eccentricity",
        "truth_eccentricity",
        "eccentricity_error",
        "scaled_vs_original_pearson",
        "scaled_vs_original_relative_l2",
        "peak_abs_ratio_scaled_over_original",
        "target_mean_ratio_scaled_over_original",
        "measurement_relative_error",
        "solve_seconds",
    ]
    ordered = [item for item in preferred if item in fields] + [
        item for item in fields if item not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in ordered})


def save_markdown(
    path: Path,
    *,
    cfg: ExperimentConfig,
    rows: list[dict[str, Any]],
    voltage_rows: list[dict[str, Any]],
) -> None:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), {})[str(row["variant"])] = row
    lines = [
        "# Scaled Boundary Voltage Circle Experiment",
        "",
        "## Setup",
        "",
        f"- Domain radius: {cfg.domain_radius:g}",
        f"- Mesh size target: {cfg.mesh_size:g}",
        f"- Electrodes: {cfg.n_elec}",
        f"- Circle radius: {cfg.target_radius:g}",
        f"- Conductivity: background {cfg.background_sigma:g}, target {cfg.target_sigma:g}",
        f"- Voltage scaling: mode `{cfg.scale_mode}`, k={cfg.voltage_scale:g}",
        f"- Difference inverse: {cfg.difference_mode}, preset {cfg.difference_preset}, lambda {cfg.hyperparameter:g}",
        "",
        "## Image Metrics",
        "",
        "| case | variant | Pearson | rel L2 | RMSE | contrast | IoU | loc err | peak err | ecc | meas rel err |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_label in sorted(by_case):
        for variant in ("original", "scaled"):
            row = by_case[case_label].get(variant)
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        case_label,
                        variant,
                        f"{safe_float(row.get('pearson_delta')):.4g}",
                        f"{safe_float(row.get('relative_l2_delta')):.4g}",
                        f"{safe_float(row.get('rmse_delta')):.4g}",
                        f"{safe_float(row.get('contrast_recovery')):.4g}",
                        f"{safe_float(row.get('iou_at_threshold')):.4g}",
                        f"{safe_float(row.get('localization_error')):.4g}",
                        f"{safe_float(row.get('peak_error')):.4g}",
                        f"{safe_float(row.get('recon_eccentricity')):.4g}",
                        f"{safe_float(row.get('measurement_relative_error')):.4g}",
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Scaled vs Original",
            "",
            "| case | recon Pearson | recon rel L2 | peak ratio | target-mean ratio | loc shift | ecc shift | voltage diff L2 ratio | voltage diff Pearson |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    voltage_by_case = {str(row["case"]): row for row in voltage_rows}
    for case_label in sorted(by_case):
        scaled = by_case[case_label].get("scaled", {})
        volt = voltage_by_case.get(case_label, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    case_label,
                    f"{safe_float(scaled.get('scaled_vs_original_pearson')):.4g}",
                    f"{safe_float(scaled.get('scaled_vs_original_relative_l2')):.4g}",
                    f"{safe_float(scaled.get('peak_abs_ratio_scaled_over_original')):.4g}",
                    f"{safe_float(scaled.get('target_mean_ratio_scaled_over_original')):.4g}",
                    f"{safe_float(scaled.get('localization_shift_scaled_minus_original')):.4g}",
                    f"{safe_float(scaled.get('eccentricity_shift_scaled_minus_original')):.4g}",
                    f"{safe_float(volt.get('scaled_diff_over_original_diff_l2')):.4g}",
                    f"{safe_float(volt.get('voltage_diff_pearson')):.4g}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig(
        output_dir=str(args.output_dir),
        n_elec=int(args.n_elec),
        domain_radius=float(args.domain_radius),
        mesh_size=float(args.mesh_size),
        electrode_coverage=float(args.electrode_coverage),
        drive_mode=str(args.drive_mode),
        drive_value=float(args.drive_value),
        contact_impedance=float(args.contact_impedance),
        background_sigma=float(args.background_sigma),
        target_sigma=float(args.target_sigma),
        target_radius=float(args.target_radius),
        voltage_scale=float(args.voltage_scale),
        scale_mode=str(args.scale_mode),
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
        difference_preset="eidors_one_step_noser",
        hyperparameter=float(args.lambda_),
        threshold_fraction=float(args.threshold_fraction),
        grid_resolution=int(args.grid_resolution),
        cases=parse_cases(args.case),
    )
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system = build_system(cfg)
    rows: list[dict[str, Any]] = []
    voltage_rows: list[dict[str, Any]] = []
    for case in cfg.cases:
        payload = run_case(system=system, case=case, cfg=cfg)
        rows.extend(payload["rows"])
        voltage_rows.append(payload["voltage_metrics"])
        save_case_plot(output_dir / f"overview_{case.label}.png", payload, cfg)
        original = payload["rows"][0]
        scaled = payload["rows"][1]
        print(
            "case",
            f"{case.label}",
            f"original_rel_l2={original['relative_l2_delta']:.4f}",
            f"scaled_rel_l2={scaled['relative_l2_delta']:.4f}",
            f"scaled_vs_original={scaled['scaled_vs_original_relative_l2']:.4f}",
            flush=True,
        )

    save_csv(output_dir / "metrics.csv", rows)
    save_csv(output_dir / "voltage_metrics.csv", voltage_rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "rows": rows,
                "voltage_rows": voltage_rows,
            },
            indent=2,
            sort_keys=True,
            default=json_ready,
        ),
        encoding="utf-8",
    )
    save_markdown(
        output_dir / "summary.md", cfg=cfg, rows=rows, voltage_rows=voltage_rows
    )
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
