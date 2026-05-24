#!/usr/bin/env python3
"""Compare 8-electrode and 16-electrode EIT reconstructions in a 2 cm disk."""

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
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import MeshConfig, PatternConfig
from pyeidors.io._json import json_ready


BACKGROUND_SIGMA = 1.0
TARGET_SIGMA = 1.5
DOMAIN_DIAMETER_M = 0.02
DOMAIN_RADIUS_M = DOMAIN_DIAMETER_M / 2.0
DEFAULT_TARGET_DIAMETERS_MM = (2.0, 4.0, 6.0, 8.0)
DEFAULT_TARGET_CENTERS_MM = (
    ("center", 0.0, 0.0),
    ("r03_x", 3.0, 0.0),
    ("r06_x", 6.0, 0.0),
    ("r045_diag", 3.182, 3.182),
)


@dataclass(frozen=True)
class CenterSpec:
    label: str
    x_m: float
    y_m: float

    @property
    def x_mm(self) -> float:
        return self.x_m * 1000.0

    @property
    def y_mm(self) -> float:
        return self.y_m * 1000.0

    @property
    def radial_offset_mm(self) -> float:
        return float(np.hypot(self.x_mm, self.y_mm))


@dataclass(frozen=True)
class ExperimentConfig:
    domain_diameter_m: float
    mesh_size_m: float
    electrode_counts: tuple[int, ...]
    target_diameters_mm: tuple[float, ...]
    target_centers: tuple[CenterSpec, ...]
    background_sigma: float
    target_sigma: float
    electrode_coverage: float
    drive_mode: str
    drive_value: float
    contact_impedance: float
    stim_pattern: str
    meas_pattern: str
    difference_mode: str
    difference_orientation: str
    difference_preset: str
    regularization: str
    hyperparameter: float
    threshold_fraction: float
    grid_resolution: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "diagnostics"
        / "electrode_count_small_domain",
    )
    parser.add_argument("--mesh-size-mm", type=float, default=0.5)
    parser.add_argument(
        "--target-diameters-mm",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_DIAMETERS_MM),
    )
    parser.add_argument(
        "--target-center-mm",
        action="append",
        default=None,
        metavar="LABEL:X,Y",
        help=(
            "Target centre in millimetres. Repeat to sweep positions. "
            "Default: center, r03_x, r06_x, r045_diag."
        ),
    )
    parser.add_argument("--electrode-counts", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--target-sigma", type=float, default=TARGET_SIGMA)
    parser.add_argument("--background-sigma", type=float, default=BACKGROUND_SIGMA)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--drive-current-a", type=float, default=1.0e-3)
    parser.add_argument("--contact-impedance", type=float, default=1.0e-6)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    parser.add_argument("--threshold-fraction", type=float, default=0.5)
    parser.add_argument("--grid-resolution", type=int, default=220)
    return parser.parse_args()


def parse_center_specs(values: list[str] | None) -> tuple[CenterSpec, ...]:
    if not values:
        return tuple(
            CenterSpec(str(label), float(x_mm) * 1e-3, float(y_mm) * 1e-3)
            for label, x_mm, y_mm in DEFAULT_TARGET_CENTERS_MM
        )
    centers: list[CenterSpec] = []
    for raw in values:
        try:
            label, xy = str(raw).split(":", 1)
            x_text, y_text = xy.split(",", 1)
        except ValueError as exc:
            raise ValueError(
                "--target-center-mm must use LABEL:X,Y, for example r03_x:3,0"
            ) from exc
        clean_label = label.strip().replace(" ", "_")
        if not clean_label:
            raise ValueError("--target-center-mm label must not be empty")
        centers.append(
            CenterSpec(
                clean_label,
                float(x_text.strip()) * 1e-3,
                float(y_text.strip()) * 1e-3,
            )
        )
    return tuple(centers)


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=np.float64).reshape(-1)
    rhs = np.asarray(right, dtype=np.float64).reshape(-1)
    mask = np.isfinite(lhs) & np.isfinite(rhs)
    lhs = lhs[mask]
    rhs = rhs[mask]
    if lhs.size < 2:
        return float("nan")
    if np.allclose(lhs, lhs[0]) or np.allclose(rhs, rhs[0]):
        return 1.0 if np.allclose(lhs, rhs) else 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


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
    target_radius_m: float,
) -> dict[str, float]:
    truth_abs = np.abs(np.asarray(truth_delta, dtype=np.float64))
    recon_abs = np.abs(np.asarray(recon_delta, dtype=np.float64))
    truth_mask = truth_abs > 0.5 * float(np.max(truth_abs))
    peak = float(np.max(recon_abs)) if recon_abs.size else 0.0
    if peak <= 0.0 or not math.isfinite(peak):
        recon_mask = np.zeros_like(truth_mask)
    else:
        recon_mask = recon_abs >= float(threshold_fraction) * peak

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
    localization_error_m = (
        float(np.linalg.norm(com - target_center[:2]))
        if com is not None
        else float("nan")
    )
    peak_idx = int(np.nanargmax(recon_abs)) if recon_abs.size else 0
    peak_error_m = float(np.linalg.norm(coords[peak_idx, :2] - target_center[:2]))
    eccentricity = weighted_eccentricity(
        coords,
        recon_abs * recon_mask.astype(np.float64),
    )
    true_eccentricity = weighted_eccentricity(
        coords,
        truth_abs * truth_mask.astype(np.float64),
    )
    return {
        "iou_at_threshold": iou,
        "area_ratio_at_threshold": area_ratio,
        "localization_error_mm": localization_error_m * 1000.0,
        "localization_error_over_radius": localization_error_m / target_radius_m,
        "peak_error_mm": peak_error_m * 1000.0,
        "recon_eccentricity": eccentricity,
        "truth_eccentricity": true_eccentricity,
        "eccentricity_error": abs(eccentricity - true_eccentricity),
    }


def image_metrics(
    *,
    coords: np.ndarray,
    truth: np.ndarray,
    recon: np.ndarray,
    background_sigma: float,
    target_sigma: float,
    target_center: np.ndarray,
    target_radius_m: float,
    threshold_fraction: float,
) -> dict[str, float]:
    truth_delta = np.asarray(truth, dtype=np.float64) - float(background_sigma)
    recon_delta = np.asarray(recon, dtype=np.float64) - float(background_sigma)
    diff = recon_delta - truth_delta
    relative_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(truth_delta) + 1e-12))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    target_mask = (
        np.linalg.norm(coords[:, :2] - target_center[:2][None, :], axis=1)
        <= target_radius_m
    )
    background_mask = ~target_mask
    truth_contrast = float(target_sigma) - float(background_sigma)
    recon_target_mean = (
        float(np.mean(recon_delta[target_mask]))
        if np.any(target_mask)
        else float("nan")
    )
    recon_background_mean = (
        float(np.mean(recon_delta[background_mask]))
        if np.any(background_mask)
        else float("nan")
    )
    contrast_recovery = (
        float((recon_target_mean - recon_background_mean) / truth_contrast)
        if abs(truth_contrast) > 1e-12
        else float("nan")
    )
    metrics = {
        "pearson_delta": safe_pearson(truth_delta, recon_delta),
        "relative_l2_delta": relative_l2,
        "rmse_delta": rmse,
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
            target_radius_m=target_radius_m,
        )
    )
    return metrics


def build_system(n_elec: int, cfg: ExperimentConfig) -> EITSystem:
    pattern = PatternConfig(
        n_elec=int(n_elec),
        stim_pattern=cfg.stim_pattern,
        meas_pattern=cfg.meas_pattern,
        drive_mode=cfg.drive_mode,
        drive_value=cfg.drive_value,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    mesh_config = MeshConfig(
        dimension=2,
        radius=cfg.domain_diameter_m / 2.0,
        mesh_size=cfg.mesh_size_m,
        electrode_coverage=cfg.electrode_coverage,
    )
    system = EITSystem(
        n_elec=int(n_elec),
        pattern_config=pattern,
        mesh_config=mesh_config,
        contact_impedance=np.full(int(n_elec), cfg.contact_impedance, dtype=float),
        base_conductivity=cfg.background_sigma,
        difference_mode=cfg.difference_mode,
        difference_orientation=cfg.difference_orientation,
        regularization_type=cfg.regularization,
        regularization_alpha=1.0,
        hyperparameter=cfg.hyperparameter,
        jacobian_background_conductivity=cfg.background_sigma,
        difference_preset=cfg.difference_preset,
        solver_mode="strict",
        linear_solver="auto",
        cache_scope="both",
        cache_dir=str(
            PROJECT_ROOT
            / ".pyeidors_cache"
            / "diagnostics"
            / "electrode_count_small_domain"
        ),
    )
    system.setup(mesh_source="generated", dimension=2)
    return system


def run_case(
    *,
    system: EITSystem,
    n_elec: int,
    target_diameter_mm: float,
    center: CenterSpec,
    cfg: ExperimentConfig,
) -> dict[str, Any]:
    target_radius_m = float(target_diameter_mm) * 0.5e-3
    target_center = np.array([center.x_m, center.y_m], dtype=np.float64)
    if float(np.linalg.norm(target_center)) + target_radius_m > DOMAIN_RADIUS_M + 1e-12:
        raise ValueError(
            f"Target {center.label!r}, d={target_diameter_mm:g} mm is outside "
            "the 2 cm circular domain."
        )
    baseline_image = system.create_homogeneous_image(cfg.background_sigma)
    baseline_data = system.forward_solve(baseline_image)
    phantom = system.add_phantom(
        base_conductivity=cfg.background_sigma,
        phantom_conductivity=cfg.target_sigma,
        phantom_center=tuple(target_center),
        phantom_radius=target_radius_m,
    )
    target_data = system.forward_solve(phantom)
    start = time.perf_counter()
    result = system.difference_reconstruct(
        measurement_data=target_data,
        reference_data=baseline_data,
        initial_image=baseline_image,
        metadata={
            "target_diameter_mm": float(target_diameter_mm),
            "n_elec": int(n_elec),
        },
    )
    elapsed = time.perf_counter() - start
    coords = system.fwd_model.V_sigma.tabulate_dof_coordinates()
    metrics = image_metrics(
        coords=coords,
        truth=phantom.elem_data,
        recon=result.conductivity,
        background_sigma=cfg.background_sigma,
        target_sigma=cfg.target_sigma,
        target_center=target_center,
        target_radius_m=target_radius_m,
        threshold_fraction=cfg.threshold_fraction,
    )
    info = system.get_system_info()
    metrics.update(
        {
            "target_center_label": center.label,
            "target_center_x_mm": center.x_mm,
            "target_center_y_mm": center.y_mm,
            "target_center_radial_offset_mm": center.radial_offset_mm,
            "n_elec": int(n_elec),
            "target_diameter_mm": float(target_diameter_mm),
            "target_radius_mm": float(target_diameter_mm) / 2.0,
            "domain_diameter_mm": cfg.domain_diameter_m * 1000.0,
            "target_diameter_over_domain": float(target_diameter_mm)
            / (cfg.domain_diameter_m * 1000.0),
            "n_measurements": int(info["n_measurements"]),
            "n_elements": int(info["n_elements"]),
            "solve_seconds": float(elapsed),
            "measurement_relative_error": result.relative_error,
            "measurement_l2_error": result.l2_error,
        }
    )
    return {
        "metrics": metrics,
        "coords": coords,
        "truth": np.asarray(phantom.elem_data, dtype=np.float64),
        "recon": np.asarray(result.conductivity, dtype=np.float64),
        "baseline_meas": np.asarray(baseline_data.meas, dtype=np.float64),
        "target_meas": np.asarray(target_data.meas, dtype=np.float64),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "target_center_label",
        "target_center_x_mm",
        "target_center_y_mm",
        "target_center_radial_offset_mm",
        "target_diameter_mm",
        "target_diameter_over_domain",
        "n_elec",
        "n_measurements",
        "n_elements",
        "pearson_delta",
        "relative_l2_delta",
        "rmse_delta",
        "contrast_recovery",
        "iou_at_threshold",
        "area_ratio_at_threshold",
        "localization_error_mm",
        "peak_error_mm",
        "recon_eccentricity",
        "truth_eccentricity",
        "eccentricity_error",
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
    query = np.column_stack([xg.ravel(), yg.ravel()])
    sampled = griddata(coords[:, :2], values, query, method="linear")
    if np.isnan(sampled).any():
        missing = np.isnan(sampled)
        sampled[missing] = griddata(
            coords[:, :2],
            values,
            query[missing],
            method="nearest",
        )
    image = sampled.reshape(xg.shape)
    image[(xg**2 + yg**2) > radius**2] = np.nan
    return xg, yg, image


def save_overview_plot(
    path: Path,
    cases: dict[tuple[float, int], dict[str, Any]],
    cfg: ExperimentConfig,
    center: CenterSpec,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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

    target_diameters = list(cfg.target_diameters_mm)
    electrode_counts = list(cfg.electrode_counts)
    n_rows = len(target_diameters)
    n_cols = 1 + len(electrode_counts)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    radius = cfg.domain_diameter_m / 2.0
    extent = [-radius * 1000.0, radius * 1000.0, -radius * 1000.0, radius * 1000.0]
    cmap = "RdBu_r" if cfg.target_sigma >= cfg.background_sigma else "RdBu"
    vlim = abs(cfg.target_sigma - cfg.background_sigma)
    for row, diameter in enumerate(target_diameters):
        truth_payload = cases[(center.label, float(diameter), electrode_counts[-1])]
        _, _, truth_grid = sample_to_grid(
            truth_payload["coords"],
            truth_payload["truth"] - cfg.background_sigma,
            radius=radius,
            resolution=cfg.grid_resolution,
        )
        images = [truth_grid]
        titles = [f"Truth, d={diameter:g} mm"]
        for n_elec in electrode_counts:
            payload = cases[(center.label, float(diameter), int(n_elec))]
            _, _, recon_grid = sample_to_grid(
                payload["coords"],
                payload["recon"] - cfg.background_sigma,
                radius=radius,
                resolution=cfg.grid_resolution,
            )
            images.append(recon_grid)
            titles.append(f"{n_elec} electrodes")
        for col, (image, title) in enumerate(zip(images, titles)):
            ax = axes[row, col]
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=-vlim,
                vmax=vlim,
                interpolation="nearest",
            )
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            circle = plt.Circle((0.0, 0.0), diameter / 2.0, color="black", fill=False)
            circle.center = (center.x_mm, center.y_mm)
            ax.add_patch(circle)
    colorbar = fig.colorbar(im, ax=axes, shrink=0.8)
    colorbar.set_label("Conductivity change (S/m)")
    fig.suptitle(
        "2 cm disk: truth vs difference reconstructions "
        f"({center.label}, x={center.x_mm:g} mm, y={center.y_mm:g} mm)",
        fontsize=14,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def metric_delta(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    by_size: dict[tuple[str, float], dict[int, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["target_center_label"]), float(row["target_diameter_mm"]))
        by_size.setdefault(key, {})[int(row["n_elec"])] = row
    deltas: list[dict[str, Any]] = []
    for (center_label, diameter), per_elec in sorted(by_size.items()):
        if 8 not in per_elec or 16 not in per_elec:
            continue
        value_8 = safe_float(per_elec[8].get(metric))
        value_16 = safe_float(per_elec[16].get(metric))
        deltas.append(
            {
                "target_center_label": center_label,
                "target_diameter_mm": diameter,
                f"{metric}_8e": value_8,
                f"{metric}_16e": value_16,
                f"{metric}_16e_minus_8e": value_16 - value_8,
            }
        )
    return deltas


def save_markdown(
    path: Path, rows: list[dict[str, Any]], cfg: ExperimentConfig
) -> None:
    by_size: dict[tuple[str, float], dict[int, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["target_center_label"]), float(row["target_diameter_mm"]))
        by_size.setdefault(key, {})[int(row["n_elec"])] = row
    lines = [
        "# 8 vs 16 electrode comparison in a 2 cm disk",
        "",
        "## Setup",
        "",
        f"- Domain diameter: {cfg.domain_diameter_m * 1000.0:.1f} mm",
        f"- Mesh size target: {cfg.mesh_size_m * 1000.0:.3f} mm",
        f"- Target diameters: {', '.join(f'{v:g} mm' for v in cfg.target_diameters_mm)}",
        "- Target centres: "
        + ", ".join(
            f"{item.label}=({item.x_mm:g}, {item.y_mm:g}) mm"
            for item in cfg.target_centers
        ),
        f"- Conductivity: background {cfg.background_sigma:g} S/m, target {cfg.target_sigma:g} S/m",
        f"- Pattern: {cfg.stim_pattern}/{cfg.meas_pattern}, {cfg.drive_mode}, {cfg.drive_value:g} A",
        f"- Difference mode: {cfg.difference_mode}, preset {cfg.difference_preset}, lambda {cfg.hyperparameter:g}",
        f"- Threshold metrics use {cfg.threshold_fraction:g} x reconstructed peak magnitude.",
        "",
        "## Metrics",
        "",
        "| centre | target d (mm) | metric | 8 electrodes | 16 electrodes | better |",
        "|---|---:|---|---:|---:|---|",
    ]
    metric_specs = [
        ("pearson_delta", "Pearson", "higher"),
        ("relative_l2_delta", "Relative L2", "lower"),
        ("contrast_recovery", "Contrast recovery", "closer1"),
        ("iou_at_threshold", "IoU", "higher"),
        ("area_ratio_at_threshold", "Area ratio", "closer1"),
        ("localization_error_mm", "Centroid error (mm)", "lower"),
        ("peak_error_mm", "Peak error (mm)", "lower"),
        ("recon_eccentricity", "Eccentricity", "lower"),
    ]
    for (center_label, diameter), per_elec in sorted(by_size.items()):
        if 8 not in per_elec or 16 not in per_elec:
            continue
        for key, label, preference in metric_specs:
            value_8 = safe_float(per_elec[8].get(key))
            value_16 = safe_float(per_elec[16].get(key))
            if preference == "higher":
                better = "16E" if value_16 > value_8 else "8E"
            elif preference == "lower":
                better = "16E" if value_16 < value_8 else "8E"
            else:
                better = "16E" if abs(value_16 - 1.0) < abs(value_8 - 1.0) else "8E"
            lines.append(
                f"| {center_label} | {diameter:g} | {label} | {value_8:.4g} | {value_16:.4g} | {better} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig(
        domain_diameter_m=DOMAIN_DIAMETER_M,
        mesh_size_m=float(args.mesh_size_mm) * 1e-3,
        electrode_counts=tuple(int(v) for v in args.electrode_counts),
        target_diameters_mm=tuple(float(v) for v in args.target_diameters_mm),
        target_centers=parse_center_specs(args.target_center_mm),
        background_sigma=float(args.background_sigma),
        target_sigma=float(args.target_sigma),
        electrode_coverage=float(args.electrode_coverage),
        drive_mode="total_current",
        drive_value=float(args.drive_current_a),
        contact_impedance=float(args.contact_impedance),
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        difference_mode="normalized",
        difference_orientation="target_minus_reference",
        difference_preset="eidors_one_step_noser",
        regularization="noser",
        hyperparameter=float(args.lambda_),
        threshold_fraction=float(args.threshold_fraction),
        grid_resolution=int(args.grid_resolution),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    systems = {
        int(n_elec): build_system(int(n_elec), cfg) for n_elec in cfg.electrode_counts
    }
    rows: list[dict[str, Any]] = []
    cases: dict[tuple[str, float, int], dict[str, Any]] = {}
    for center in cfg.target_centers:
        for target_diameter_mm in cfg.target_diameters_mm:
            for n_elec, system in systems.items():
                payload = run_case(
                    system=system,
                    n_elec=int(n_elec),
                    target_diameter_mm=float(target_diameter_mm),
                    center=center,
                    cfg=cfg,
                )
                row = payload["metrics"]
                rows.append(row)
                cases[(center.label, float(target_diameter_mm), int(n_elec))] = payload
                print(
                    "case",
                    f"center={center.label}",
                    f"d={target_diameter_mm:g}mm",
                    f"n_elec={n_elec}",
                    f"pearson={row['pearson_delta']:.4f}",
                    f"rel_l2={row['relative_l2_delta']:.4f}",
                    f"iou={row['iou_at_threshold']:.4f}",
                    flush=True,
                )

    save_csv(output_dir / "metrics.csv", rows)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "rows": rows,
                "deltas": {
                    "pearson": metric_delta(rows, "pearson_delta"),
                    "relative_l2": metric_delta(rows, "relative_l2_delta"),
                    "iou": metric_delta(rows, "iou_at_threshold"),
                    "eccentricity": metric_delta(rows, "recon_eccentricity"),
                },
            },
            indent=2,
            sort_keys=True,
            default=json_ready,
        ),
        encoding="utf-8",
    )
    for center in cfg.target_centers:
        save_overview_plot(
            output_dir / f"overview_{center.label}.png", cases, cfg, center
        )
    save_markdown(output_dir / "summary.md", rows, cfg)
    print(f"wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
