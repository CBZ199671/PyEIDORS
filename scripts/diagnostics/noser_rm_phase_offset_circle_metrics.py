#!/usr/bin/env python3
"""Evaluate NOSER-RM circular-object metrics under global voltage phase offsets."""

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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eit_app.controllers.forward_solver_controller import (  # noqa: E402
    execute_forward_request,
)
from eit_app.controllers.reconstruction_controller import (  # noqa: E402
    ReconstructionRequest,
    default_rm_inverse_mesh_size,
    run_reconstruction_request,
)
from pyeidors.runtime_paths import pyeidors_cache_path, pyeidors_output_path  # noqa: E402
from scripts.diagnostics.phase_offset_complex_reconstruction_sweep import (  # noqa: E402
    build_forward_request,
    frame_from_complex,
    json_ready,
    parse_complex_arg,
    register_times_new_roman,
    robust_limits,
    tripcolor_cell_data,
)


DEFAULT_ANGLES_DEG = (
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    *tuple(float(v) for v in range(10, 181, 10)),
)
DEFAULT_TOLERANCE_FRACTIONS = (0.05, 0.10, 0.15, 0.20)
DIMENSIONLESS_TOLERANCE_METRICS = (
    "centroid_error_over_radius",
    "equivalent_area_ratio",
    "eccentricity",
    "artifact_area_ratio",
    "artifact_energy_ratio",
    "artifact_peak_ratio",
    "shape_deformation",
    "amplitude_response",
    "iou",
)


@dataclass(frozen=True)
class NoserRmPhaseConfig:
    output_dir: str
    n_elec: int
    mesh_size: float
    radius: float
    electrode_coverage: float
    background_admittance: complex
    anomaly_admittance: complex
    anomaly_center_x: float
    anomaly_center_y: float
    anomaly_radius: float
    contact_impedance: complex
    drive_mode: str
    drive_value: float
    regularization_alpha: float
    difference_lambda: float
    difference_mode: str
    difference_orientation: str
    difference_preset: str
    compute_dtype: str
    angles_deg: tuple[float, ...]
    threshold_fraction: float
    tolerance_fractions: tuple[float, ...]
    rm_artifact_dir: str


def parse_angles(raw: str | None) -> tuple[float, ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_ANGLES_DEG
    out: list[float] = []
    for item in str(raw).replace(";", ",").split(","):
        text = item.strip()
        if text:
            out.append(float(text))
    if not out:
        raise ValueError("At least one phase angle is required.")
    return tuple(dict.fromkeys(out))


def parse_float_list(
    raw: str | None, *, default: tuple[float, ...]
) -> tuple[float, ...]:
    if raw is None or not str(raw).strip():
        return default
    out: list[float] = []
    for item in str(raw).replace(";", ",").split(","):
        text = item.strip()
        if text:
            out.append(float(text))
    if not out:
        raise ValueError("At least one value is required.")
    return tuple(dict.fromkeys(out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=pyeidors_output_path(
            "diagnostics",
            "noser_rm_phase_offset_circle_metrics",
        ),
    )
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--mesh-size", type=float, default=0.16)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument(
        "--background-admittance",
        type=parse_complex_arg,
        default=complex(1.0, 2.0),
    )
    parser.add_argument(
        "--anomaly-admittance",
        type=parse_complex_arg,
        default=complex(2.0, 3.0),
    )
    parser.add_argument("--anomaly-center-x", type=float, default=0.28)
    parser.add_argument("--anomaly-center-y", type=float, default=0.12)
    parser.add_argument("--anomaly-radius", type=float, default=0.22)
    parser.add_argument(
        "--contact-impedance",
        type=parse_complex_arg,
        default=complex(0.01, 0.0),
    )
    parser.add_argument("--drive-mode", default="line_current_density")
    parser.add_argument("--drive-value", type=float, default=1.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    parser.add_argument("--difference-lambda", type=float, default=0.01)
    parser.add_argument("--difference-mode", default="normalized")
    parser.add_argument("--difference-orientation", default="target_minus_reference")
    parser.add_argument("--difference-preset", default="noser_rm")
    parser.add_argument("--compute-dtype", default="complex64")
    parser.add_argument(
        "--angles",
        default=None,
        help="Comma-separated phase offsets in degrees.",
    )
    parser.add_argument("--threshold-fraction", type=float, default=0.5)
    parser.add_argument(
        "--tolerance-fractions",
        default="0.05,0.10,0.15,0.20",
        help="Comma-separated dimensionless metric-drift tolerances.",
    )
    parser.add_argument(
        "--rm-artifact-dir",
        type=Path,
        default=pyeidors_cache_path(
            "diagnostics",
            "noser_rm_phase_offset_circle_metrics",
            "gui_rm",
        ),
    )
    return parser.parse_args()


def polygon_cell_areas(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    node_coords = np.asarray(coords, dtype=np.float64)
    connectivity = np.asarray(cells, dtype=np.int64)
    areas = np.empty(int(connectivity.shape[0]), dtype=np.float64)
    for idx, cell in enumerate(connectivity):
        xy = node_coords[cell, :2]
        x = xy[:, 0]
        y = xy[:, 1]
        areas[idx] = 0.5 * abs(
            float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        )
    return areas


def cell_centers(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    node_coords = np.asarray(coords, dtype=np.float64)
    connectivity = np.asarray(cells, dtype=np.int64)
    return np.mean(node_coords[connectivity, :2], axis=1)


def weighted_structure(
    *,
    points: np.ndarray,
    areas: np.ndarray,
    weights_raw: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    pts = np.asarray(points, dtype=np.float64)
    area_arr = np.asarray(areas, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights_raw, dtype=np.float64).reshape(-1)
    active = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(active):
        active[int(np.nanargmax(np.abs(weights)))] = True
    weighted = np.abs(weights) * area_arr
    if float(np.sum(weighted, where=active, initial=0.0)) <= 0.0:
        weighted = area_arr
    total_weight = float(np.sum(weighted, where=active, initial=0.0))
    centroid = np.array(
        [
            float(np.sum(pts[:, 0] * weighted, where=active, initial=0.0)),
            float(np.sum(pts[:, 1] * weighted, where=active, initial=0.0)),
        ],
        dtype=np.float64,
    )
    centroid /= max(total_weight, 1.0e-30)
    shifted = pts - centroid[None, :]
    cxx = float(
        np.sum(shifted[:, 0] * shifted[:, 0] * weighted, where=active, initial=0.0)
    )
    cxy = float(
        np.sum(shifted[:, 0] * shifted[:, 1] * weighted, where=active, initial=0.0)
    )
    cyy = float(
        np.sum(shifted[:, 1] * shifted[:, 1] * weighted, where=active, initial=0.0)
    )
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64) / max(
        total_weight, 1.0e-30
    )
    eigvals = np.sort(np.linalg.eigvalsh(cov))
    minor_var = max(float(eigvals[0]), 0.0)
    major_var = max(float(eigvals[-1]), 0.0)
    eccentricity = (
        0.0 if major_var <= 0.0 else math.sqrt(max(0.0, 1.0 - minor_var / major_var))
    )
    return {
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "equivalent_area": float(np.sum(area_arr, where=active, initial=0.0)),
        "eccentricity": float(eccentricity),
        "major_axis": float(4.0 * math.sqrt(major_var)),
        "minor_axis": float(4.0 * math.sqrt(minor_var)),
    }


def circular_metrics(
    *,
    sigma: np.ndarray,
    background: complex,
    truth_strength: np.ndarray,
    truth_mask: np.ndarray,
    points: np.ndarray,
    areas: np.ndarray,
    anomaly_center: np.ndarray,
    anomaly_radius: float,
    threshold_fraction: float,
) -> dict[str, float]:
    contrast = np.asarray(sigma, dtype=np.complex128).reshape(-1) - complex(background)
    strength = np.abs(contrast)
    threshold = float(threshold_fraction) * max(float(np.max(truth_strength)), 1.0e-12)
    recon_mask = strength >= threshold
    structure = weighted_structure(
        points=points,
        areas=areas,
        weights_raw=strength,
        mask=recon_mask,
    )
    active_outside = recon_mask & ~truth_mask
    active_inside = recon_mask & truth_mask
    outside_energy = float(
        np.sum(strength * strength * areas, where=~truth_mask, initial=0.0)
    )
    inside_energy = float(
        np.sum(strength * strength * areas, where=truth_mask, initial=0.0)
    )
    outside_peak = float(np.max(strength, where=~truth_mask, initial=0.0))
    inside_peak = float(np.max(strength, where=truth_mask, initial=0.0))
    truth_area = float(np.sum(areas, where=truth_mask, initial=0.0))
    active_area = float(np.sum(areas, where=recon_mask, initial=0.0))
    artifact_area = float(np.sum(areas, where=active_outside, initial=0.0))
    covered_area = float(np.sum(areas, where=active_inside, initial=0.0))
    union_area = float(np.sum(areas, where=(recon_mask | truth_mask), initial=0.0))
    intersection_area = float(
        np.sum(areas, where=(recon_mask & truth_mask), initial=0.0)
    )
    target_mean = float(np.sum(strength * areas, where=truth_mask, initial=0.0)) / max(
        truth_area, 1.0e-30
    )
    truth_mean = float(
        np.sum(truth_strength * areas, where=truth_mask, initial=0.0)
    ) / max(truth_area, 1.0e-30)
    centroid_error = math.hypot(
        float(structure["centroid_x"]) - float(anomaly_center[0]),
        float(structure["centroid_y"]) - float(anomaly_center[1]),
    )
    return {
        **structure,
        "centroid_error": float(centroid_error),
        "centroid_error_over_radius": float(centroid_error / anomaly_radius),
        "truth_area": truth_area,
        "active_area": active_area,
        "equivalent_area_ratio": float(
            structure["equivalent_area"] / max(truth_area, 1.0e-30)
        ),
        "target_coverage_ratio": float(covered_area / max(truth_area, 1.0e-30)),
        "artifact_area": artifact_area,
        "artifact_area_ratio": float(artifact_area / max(truth_area, 1.0e-30)),
        "shape_deformation": float(artifact_area / max(active_area, 1.0e-30)),
        "artifact_energy": outside_energy,
        "artifact_energy_ratio": float(
            outside_energy / max(inside_energy + outside_energy, 1.0e-30)
        ),
        "artifact_peak": outside_peak,
        "artifact_peak_ratio": float(outside_peak / max(inside_peak, 1.0e-30)),
        "amplitude_response": float(target_mean / max(truth_mean, 1.0e-30)),
        "iou": float(intersection_area / max(union_area, 1.0e-30)),
    }


def tolerance_changes(
    row: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float | bool]:
    changes: dict[str, float | bool] = {}
    max_delta = 0.0
    for key in DIMENSIONLESS_TOLERANCE_METRICS:
        current = float(row.get(key, float("nan")))
        base = float(baseline.get(key, float("nan")))
        delta = (
            abs(current - base)
            if np.isfinite(current) and np.isfinite(base)
            else float("inf")
        )
        changes[f"delta_{key}"] = float(delta)
        max_delta = max(max_delta, float(delta))
    changes["max_dimensionless_metric_delta"] = float(max_delta)
    return changes


def tolerance_column(tolerance: float) -> str:
    percent = int(round(float(tolerance) * 100.0))
    return f"metric_delta_{percent}pct_pass"


def build_noser_rm_metadata(
    forward_meta: dict[str, Any],
    cfg: NoserRmPhaseConfig,
) -> dict[str, Any]:
    rm_inverse_mesh_size = default_rm_inverse_mesh_size(
        cfg.mesh_size,
        cfg.radius,
        mesh_dimension=2,
    )
    hp = math.sqrt(max(float(cfg.difference_lambda), 0.0))
    meta = dict(forward_meta)
    meta.update(
        {
            "mesh_dimension": 2,
            "mesh_refinement": cfg.mesh_size,
            "mesh_size": cfg.mesh_size,
            "n_elec": cfg.n_elec,
            "n_rings": 1,
            "background_sigma": cfg.background_admittance,
            "background_conductivity": cfg.background_admittance,
            "contact_impedance": cfg.contact_impedance,
            "difference_mode": cfg.difference_mode,
            "difference_orientation": cfg.difference_orientation,
            "difference_preset": cfg.difference_preset,
            "absolute_preset": "eidors_abs_gn",
            "request_source": "diagnostic_noser_rm_phase_offset",
            "compute_precision": cfg.compute_dtype,
            "compute_dtype": cfg.compute_dtype,
            "rm_dtype": cfg.compute_dtype,
            "rm_matmul_dtype": cfg.compute_dtype,
            "reconstruction_runtime": "single_step_cached",
            "simulation_inverse_route": "noser_rm",
            "simulation_inverse_route_kind": "rm",
            "simulation_inverse_debug_route": False,
            "rm_route_requires_artifact": True,
            "rm_auto_build": True,
            "rm_regularization": "noser",
            "rm_form": "measurement",
            "rm_output_display_mode": "absolute_sigma",
            "rm_artifact_dir": cfg.rm_artifact_dir,
            "rm_inverse_mesh_size": rm_inverse_mesh_size,
            "online_hot_path": "rm_matmul",
            "jacobian_representation": "auto",
            "linearized_solver_strategy": "auto",
            "linearized_maxiter": 0,
            "lazy_preconditioner_mode": "auto",
            "solver_mode": "auto",
            "line_search_mode": "auto",
            "linear_solver": "auto",
            "preconditioner": "auto",
            "fast_linear_path": "auto",
            "forward_solver_preset": "auto",
            "forward_mat_solve": "auto",
            "petsc_device": "cpu",
            "device": "cpu",
            "forward_backend": "dolfinx",
            "mesh_family": "tetra",
            "geometry_version": "geomv2",
            "difference_lambda": cfg.difference_lambda,
            "lambda_eff": cfg.difference_lambda,
            "lambda_eff_custom_enabled": False,
            "hp": hp,
            "hp_squared": hp * hp,
            "difference_lambda_semantics": "lambda_eff_equals_hp_squared",
            "regularization_alpha_input": cfg.regularization_alpha,
            "regularization_alpha_applied": False,
            "eit_value_mode": "complex_admittance",
            "complex_measurement_mode": "native_real_imag",
            "complex_reconstruction_dispatch": "one_step_rm",
        }
    )
    return meta


def run_noser_rm(
    *,
    reference_voltage: np.ndarray,
    target_voltage: np.ndarray,
    metadata: dict[str, Any],
    cfg: NoserRmPhaseConfig,
    frame_index: int,
) -> Any:
    request = ReconstructionRequest(
        reference_frame=frame_from_complex(reference_voltage, 0),
        target_frame=frame_from_complex(target_voltage, frame_index),
        use_part="complex",
        method="gn-difference",
        regularization_alpha=cfg.regularization_alpha,
        max_iterations=1,
        mesh_dimension=2,
        mesh_refinement=cfg.mesh_size,
        metadata=dict(metadata),
    )
    result = run_reconstruction_request(
        request,
        progress_cb=lambda message: print(f"    {message}", flush=True),
    )
    if result.error_msg:
        raise RuntimeError(result.error_msg)
    return result


def add_circle_overlays(ax: plt.Axes, cfg: NoserRmPhaseConfig) -> None:
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            cfg.radius,
            color="black",
            fill=False,
            linewidth=0.8,
            alpha=0.6,
        )
    )
    ax.add_patch(
        plt.Circle(
            (cfg.anomaly_center_x, cfg.anomaly_center_y),
            cfg.anomaly_radius,
            color="red",
            fill=False,
            linewidth=1.2,
            alpha=0.9,
        )
    )


def plot_summary_grid(
    *,
    path: Path,
    cfg: NoserRmPhaseConfig,
    coords: np.ndarray,
    cells: np.ndarray,
    truth_strength: np.ndarray,
    angles_deg: tuple[float, ...],
    strengths: list[np.ndarray],
    vmax: float,
) -> None:
    n_items = 1 + len(strengths)
    n_cols = 5
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.15 * n_cols, 3.05 * n_rows),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    artist = None
    for idx, ax in enumerate(axes_arr.flat):
        if idx >= n_items:
            ax.axis("off")
            continue
        if idx == 0:
            values = truth_strength
            title = "Truth |sigma-bg|"
        else:
            values = strengths[idx - 1]
            title = f"+{angles_deg[idx - 1]:g} deg"
        artist = tripcolor_cell_data(
            ax,
            coords=coords,
            cells=cells,
            values=values,
            title=title,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        add_circle_overlays(ax, cfg)
    if artist is not None:
        fig.colorbar(artist, ax=axes_arr.ravel().tolist(), shrink=0.72, label="S/m")
    fig.suptitle("NOSER-RM phase-offset circular-object strength", fontsize=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_selected_comparison(
    *,
    path: Path,
    cfg: NoserRmPhaseConfig,
    coords: np.ndarray,
    cells: np.ndarray,
    truth_strength: np.ndarray,
    baseline_strength: np.ndarray,
    selected: list[tuple[str, np.ndarray]],
    vmax: float,
) -> None:
    panels = [
        ("Truth", truth_strength),
        ("Original +0 deg", baseline_strength),
        *selected,
    ]
    n_cols = min(4, len(panels))
    n_rows = int(math.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.8 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    artist = None
    for idx, ax in enumerate(axes_arr.flat):
        if idx >= len(panels):
            ax.axis("off")
            continue
        title, values = panels[idx]
        artist = tripcolor_cell_data(
            ax,
            coords=coords,
            cells=cells,
            values=values,
            title=title,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        add_circle_overlays(ax, cfg)
    if artist is not None:
        fig.colorbar(artist, ax=axes_arr.ravel().tolist(), shrink=0.78, label="S/m")
    fig.suptitle("Truth and selected NOSER-RM reconstructions", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_delta(
    path: Path, rows: list[dict[str, float]], cfg: NoserRmPhaseConfig
) -> None:
    angles = np.asarray([float(row["angle_deg"]) for row in rows], dtype=np.float64)
    max_delta = np.asarray(
        [float(row["max_dimensionless_metric_delta"]) for row in rows],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(angles, max_delta, marker="o", linewidth=1.4, markersize=3.5)
    colors = ("red", "#cc7a00", "#7a52cc", "#27834f")
    for idx, tolerance in enumerate(cfg.tolerance_fractions):
        ax.axhline(
            float(tolerance),
            color=colors[idx % len(colors)],
            linestyle="--",
            linewidth=1.1,
            label=f"{float(tolerance) * 100:g}%",
        )
    ax.set_xlabel("phase offset (deg)")
    ax.set_ylabel("max dimensionless metric delta vs +0 deg")
    ax.set_title("NOSER-RM metric drift under global voltage phase offset")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Tolerance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metric_delta_zoom(
    path: Path,
    rows: list[dict[str, float]],
    cfg: NoserRmPhaseConfig,
    *,
    x_max: float = 5.0,
) -> None:
    selected = [row for row in rows if float(row["angle_deg"]) <= float(x_max)]
    angles = np.asarray([float(row["angle_deg"]) for row in selected], dtype=np.float64)
    max_delta = np.asarray(
        [float(row["max_dimensionless_metric_delta"]) for row in selected],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.plot(angles, max_delta, marker="o", linewidth=1.4, markersize=4.0)
    colors = ("red", "#cc7a00", "#7a52cc", "#27834f")
    for idx, tolerance in enumerate(cfg.tolerance_fractions):
        ax.axhline(
            float(tolerance),
            color=colors[idx % len(colors)],
            linestyle="--",
            linewidth=1.1,
            label=f"{float(tolerance) * 100:g}%",
        )
    y_max = max(
        float(np.max(max_delta)) if max_delta.size else 0.0,
        max(float(value) for value in cfg.tolerance_fractions),
    )
    ax.set_xlim(-0.05, float(x_max))
    ax.set_ylim(0.0, 1.1 * y_max)
    ax.set_xlabel("phase offset (deg)")
    ax.set_ylabel("max dimensionless metric delta vs +0 deg")
    ax.set_title(f"NOSER-RM metric drift, 0-{x_max:g} deg zoom")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Tolerance")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_per_angle(
    *,
    path: Path,
    cfg: NoserRmPhaseConfig,
    coords: np.ndarray,
    cells: np.ndarray,
    truth_strength: np.ndarray,
    baseline_strength: np.ndarray,
    shifted_strength: np.ndarray,
    artifact_mask_values: np.ndarray,
    angle_deg: float,
    row: dict[str, float],
    vmax: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.8), constrained_layout=True)
    panels = [
        ("Truth |sigma-bg|", truth_strength, "viridis", 0.0, vmax),
        ("Original +0 deg", baseline_strength, "viridis", 0.0, vmax),
        (f"NOSER-RM +{angle_deg:g} deg", shifted_strength, "viridis", 0.0, vmax),
        ("Artifact mask outside circle", artifact_mask_values, "magma", 0.0, 1.0),
    ]
    for ax, (title, values, cmap, vmin, vmax_i) in zip(axes.flat, panels, strict=True):
        tripcolor_cell_data(
            ax,
            coords=coords,
            cells=cells,
            values=values,
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_i,
        )
        add_circle_overlays(ax, cfg)
    fig.suptitle(
        f"+{angle_deg:g} deg: max delta={row['max_dimensionless_metric_delta']:.4f}, "
        f"pass 5%={bool(row.get('metric_delta_5pct_pass', False))}, "
        f"ecc={row['eccentricity']:.3f}, artifact area ratio={row['artifact_area_ratio']:.3f}",
        fontsize=12,
    )
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tolerance_summary_for_level(
    rows: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: float(row["angle_deg"]))
    contiguous_max = 0.0
    first_fail: float | None = None
    column = tolerance_column(tolerance)
    for row in sorted_rows:
        angle = float(row["angle_deg"])
        passed = bool(row[column])
        if passed:
            contiguous_max = angle
            continue
        first_fail = angle
        break
    pass_rows = [row for row in sorted_rows if bool(row[column])]
    return {
        "tolerance_fraction": float(tolerance),
        "tolerance_percent": float(tolerance * 100.0),
        "contiguous_max_passing_angle_deg": float(contiguous_max),
        "first_failing_angle_deg": first_fail,
        "any_passing_max_angle_deg": float(
            max(float(row["angle_deg"]) for row in pass_rows)
        )
        if pass_rows
        else None,
        "dimensionless_metrics": list(DIMENSIONLESS_TOLERANCE_METRICS),
    }


def tolerance_summary(
    rows: list[dict[str, Any]],
    tolerances: tuple[float, ...],
) -> dict[str, Any]:
    levels = [
        tolerance_summary_for_level(rows, tolerance=float(tolerance))
        for tolerance in tolerances
    ]
    return {
        "levels": levels,
        "dimensionless_metrics": list(DIMENSIONLESS_TOLERANCE_METRICS),
    }


def main() -> int:
    args = parse_args()
    cfg = NoserRmPhaseConfig(
        output_dir=str(args.output_dir),
        n_elec=int(args.n_elec),
        mesh_size=float(args.mesh_size),
        radius=float(args.radius),
        electrode_coverage=float(args.electrode_coverage),
        background_admittance=complex(args.background_admittance),
        anomaly_admittance=complex(args.anomaly_admittance),
        anomaly_center_x=float(args.anomaly_center_x),
        anomaly_center_y=float(args.anomaly_center_y),
        anomaly_radius=float(args.anomaly_radius),
        contact_impedance=complex(args.contact_impedance),
        drive_mode=str(args.drive_mode),
        drive_value=float(args.drive_value),
        regularization_alpha=float(args.lambda_),
        difference_lambda=float(args.difference_lambda),
        difference_mode=str(args.difference_mode),
        difference_orientation=str(args.difference_orientation),
        difference_preset=str(args.difference_preset),
        compute_dtype=str(args.compute_dtype),
        angles_deg=parse_angles(args.angles),
        threshold_fraction=float(args.threshold_fraction),
        tolerance_fractions=parse_float_list(
            args.tolerance_fractions,
            default=DEFAULT_TOLERANCE_FRACTIONS,
        ),
        rm_artifact_dir=str(args.rm_artifact_dir),
    )
    output_dir = Path(cfg.output_dir)
    per_angle_dir = output_dir / "per_angle"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_angle_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.rm_artifact_dir).mkdir(parents=True, exist_ok=True)

    register_times_new_roman()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    print(f"Output directory: {output_dir}", flush=True)
    print("Solving complex 2-D forward problem...", flush=True)
    started = time.perf_counter()
    forward = execute_forward_request(
        build_forward_request(cfg),
        progress_cb=lambda message: print(f"  {message}", flush=True),
    )
    if forward.error_msg:
        raise RuntimeError(forward.error_msg)
    print(f"Forward solve seconds: {time.perf_counter() - started:.2f}", flush=True)

    reference_voltage = np.asarray(forward.homogeneous_voltages, dtype=np.complex128)
    target_voltage = np.asarray(forward.boundary_voltages, dtype=np.complex128)
    metadata = build_noser_rm_metadata(forward.forward_model_config, cfg)

    print("Running original +0 deg NOSER-RM reconstruction...", flush=True)
    baseline_result = run_noser_rm(
        reference_voltage=reference_voltage,
        target_voltage=target_voltage,
        metadata=metadata,
        cfg=cfg,
        frame_index=1,
    )
    coords = np.asarray(baseline_result.node_coords, dtype=np.float64)
    cells = np.asarray(baseline_result.cell_connectivity, dtype=np.int64)
    areas = polygon_cell_areas(coords, cells)
    centers = cell_centers(coords, cells)
    anomaly_center = np.array(
        [cfg.anomaly_center_x, cfg.anomaly_center_y],
        dtype=np.float64,
    )
    truth_mask = (
        np.linalg.norm(centers[:, :2] - anomaly_center[None, :], axis=1)
        <= cfg.anomaly_radius
    )
    truth_sigma = np.full(
        cells.shape[0], cfg.background_admittance, dtype=np.complex128
    )
    truth_sigma[truth_mask] = cfg.anomaly_admittance
    truth_strength = np.abs(truth_sigma - cfg.background_admittance)
    baseline_sigma = np.asarray(baseline_result.conductivity, dtype=np.complex128)
    baseline_strength = np.abs(baseline_sigma - cfg.background_admittance)
    baseline_metric = circular_metrics(
        sigma=baseline_sigma,
        background=cfg.background_admittance,
        truth_strength=truth_strength,
        truth_mask=truth_mask,
        points=centers,
        areas=areas,
        anomaly_center=anomaly_center,
        anomaly_radius=cfg.anomaly_radius,
        threshold_fraction=cfg.threshold_fraction,
    )

    rows: list[dict[str, Any]] = []
    sigmas: list[np.ndarray] = []
    strengths: list[np.ndarray] = []
    shifted_voltages: list[np.ndarray] = []
    print("Sweeping phase offsets with NOSER-RM...", flush=True)
    for idx, angle_deg in enumerate(cfg.angles_deg):
        angle_rad = math.radians(float(angle_deg))
        shifted_voltage = np.abs(target_voltage) * np.exp(
            1j * (np.angle(target_voltage) + angle_rad)
        )
        shifted_voltages.append(shifted_voltage)
        if abs(float(angle_deg)) <= 1.0e-14:
            result = baseline_result
        else:
            print(f"  Phase +{angle_deg:g} deg", flush=True)
            result = run_noser_rm(
                reference_voltage=reference_voltage,
                target_voltage=shifted_voltage,
                metadata=metadata,
                cfg=cfg,
                frame_index=idx + 2,
            )
        sigma = np.asarray(result.conductivity, dtype=np.complex128)
        strength = np.abs(sigma - cfg.background_admittance)
        metrics = circular_metrics(
            sigma=sigma,
            background=cfg.background_admittance,
            truth_strength=truth_strength,
            truth_mask=truth_mask,
            points=centers,
            areas=areas,
            anomaly_center=anomaly_center,
            anomaly_radius=cfg.anomaly_radius,
            threshold_fraction=cfg.threshold_fraction,
        )
        row: dict[str, Any] = {
            "angle_deg": float(angle_deg),
            "method": "noser_rm",
            "use_part": "complex",
            "phase_transform": "target_voltage_abs_preserved_phase_plus_theta",
            **metrics,
            **tolerance_changes(metrics, baseline_metric),
        }
        for tolerance in cfg.tolerance_fractions:
            row[tolerance_column(tolerance)] = bool(
                float(row["max_dimensionless_metric_delta"]) <= float(tolerance)
            )
        rows.append(row)
        sigmas.append(sigma)
        strengths.append(strength)

    summary = tolerance_summary(rows, cfg.tolerance_fractions)
    vmax = robust_limits(
        [truth_strength, baseline_strength, *strengths],
        fallback=(0.0, abs(cfg.anomaly_admittance - cfg.background_admittance)),
    )[1]
    vmax = max(float(vmax), abs(cfg.anomaly_admittance - cfg.background_admittance))

    print("Rendering figures...", flush=True)
    plot_summary_grid(
        path=output_dir / "truth_and_all_noser_rm_phase_offsets.png",
        cfg=cfg,
        coords=coords,
        cells=cells,
        truth_strength=truth_strength,
        angles_deg=cfg.angles_deg,
        strengths=strengths,
        vmax=vmax,
    )
    selected: list[tuple[str, np.ndarray]] = []
    index_by_angle = {float(angle): i for i, angle in enumerate(cfg.angles_deg)}
    primary_level = summary["levels"][0]
    for label, angle in (
        ("Max pass 5%", primary_level["contiguous_max_passing_angle_deg"]),
        ("First fail 5%", primary_level["first_failing_angle_deg"]),
        ("+1 deg", 1.0),
        ("+10 deg", 10.0),
    ):
        if angle is None:
            continue
        idx = index_by_angle.get(float(angle))
        if idx is not None:
            selected.append((f"{label}: +{float(angle):g} deg", strengths[idx]))
    plot_selected_comparison(
        path=output_dir / "truth_original_selected_comparison.png",
        cfg=cfg,
        coords=coords,
        cells=cells,
        truth_strength=truth_strength,
        baseline_strength=baseline_strength,
        selected=selected,
        vmax=vmax,
    )
    plot_metric_delta(output_dir / "metric_delta_vs_angle.png", rows, cfg)
    plot_metric_delta_zoom(
        output_dir / "metric_delta_vs_angle_zoom_0_5deg.png", rows, cfg
    )
    threshold = cfg.threshold_fraction * max(float(np.max(truth_strength)), 1.0e-12)
    for angle_deg, strength, row in zip(cfg.angles_deg, strengths, rows, strict=True):
        active = strength >= threshold
        artifact_values = np.zeros_like(strength, dtype=np.float64)
        artifact_values[active & ~truth_mask] = 1.0
        safe_angle = str(angle_deg).replace(".", "p").replace("-", "m")
        plot_per_angle(
            path=per_angle_dir / f"noser_rm_phase_offset_{safe_angle}_deg.png",
            cfg=cfg,
            coords=coords,
            cells=cells,
            truth_strength=truth_strength,
            baseline_strength=baseline_strength,
            shifted_strength=strength,
            artifact_mask_values=artifact_values,
            angle_deg=float(angle_deg),
            row=row,
            vmax=vmax,
        )

    metrics_path = output_dir / "metrics.csv"
    settings = {
        "forward_problem": {
            "mesh_dimension": 2,
            "n_elec": cfg.n_elec,
            "mesh_size": cfg.mesh_size,
            "radius": cfg.radius,
            "electrode_coverage": cfg.electrode_coverage,
            "background_admittance": cfg.background_admittance,
            "anomaly": {
                "shape": "circle",
                "center_x": cfg.anomaly_center_x,
                "center_y": cfg.anomaly_center_y,
                "radius": cfg.anomaly_radius,
                "admittance": cfg.anomaly_admittance,
            },
            "contact_impedance": cfg.contact_impedance,
            "drive_mode": cfg.drive_mode,
            "drive_value": cfg.drive_value,
            "measurement_protocol": "eidors_full_3d",
            "stim_pattern": "{ad}",
            "meas_pattern": "{ad}",
            "rotate_meas": True,
            "noise_level": 0.0,
        },
        "phase_sweep": {
            "reference_voltage": "homogeneous forward voltage, unchanged",
            "target_voltage": "|V_target| * exp(1j*(angle(V_target)+theta))",
            "angles_deg": cfg.angles_deg,
        },
        "inverse_problem": {
            "method": "noser_rm",
            "use_part": "complex",
            "reconstruction_runtime": "single_step_cached",
            "difference_mode": cfg.difference_mode,
            "difference_orientation": cfg.difference_orientation,
            "difference_preset": cfg.difference_preset,
            "rm_regularization": "noser",
            "rm_form": "measurement",
            "rm_output_display_mode": "absolute_sigma",
            "difference_lambda": cfg.difference_lambda,
            "hp": math.sqrt(cfg.difference_lambda),
            "compute_dtype": cfg.compute_dtype,
            "rm_artifact_dir": cfg.rm_artifact_dir,
        },
        "metrics": {
            "strength_channel": "|sigma - background|",
            "threshold_rule": (
                "recon active if |sigma-background| >= "
                f"{cfg.threshold_fraction:g} * max(true |sigma-background|)"
            ),
            "tolerance_rule": (
                f"pass if max absolute change among {list(DIMENSIONLESS_TOLERANCE_METRICS)} "
                "relative to +0 deg baseline <= each configured tolerance"
            ),
            "tolerance_fractions": cfg.tolerance_fractions,
        },
    }
    write_csv(metrics_path, rows)
    (output_dir / "simulation_settings.json").write_text(
        json.dumps(json_ready(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    arrays_path = output_dir / "noser_rm_phase_metric_arrays.h5"
    summary_payload = {
        "summary": summary,
        "config": json_ready(asdict(cfg)),
        "settings": json_ready(settings),
        "baseline_metrics": json_ready(baseline_metric),
        "metrics": json_ready(rows),
        "forward_result": {
            "n_elements": int(forward.n_elements),
            "n_measurements": int(forward.n_measurements),
            "node_count": int(len(forward.node_coords)),
            "forward_model_config": json_ready(forward.forward_model_config),
        },
        "outputs": {
            "metrics_csv": str(metrics_path),
            "simulation_settings_json": str(output_dir / "simulation_settings.json"),
            "truth_and_all_grid": str(
                output_dir / "truth_and_all_noser_rm_phase_offsets.png"
            ),
            "selected_comparison": str(
                output_dir / "truth_original_selected_comparison.png"
            ),
            "metric_delta_plot": str(output_dir / "metric_delta_vs_angle.png"),
            "metric_delta_zoom_plot": str(
                output_dir / "metric_delta_vs_angle_zoom_0_5deg.png"
            ),
            "per_angle_dir": str(per_angle_dir),
            "arrays_h5": str(arrays_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with h5py.File(arrays_path, "w") as handle:
        for name, values in {
            "angles_deg": np.asarray(cfg.angles_deg, dtype=np.float64),
            "sigma": np.stack(sigmas, axis=0),
            "strength": np.stack(strengths, axis=0),
            "truth_sigma": truth_sigma,
            "truth_strength": truth_strength,
            "truth_mask": truth_mask,
            "node_coords": coords,
            "cell_connectivity": cells,
            "cell_areas": areas,
            "reference_voltage": reference_voltage,
            "target_voltage": target_voltage,
            "shifted_voltage": np.stack(shifted_voltages, axis=0),
        }.items():
            array = np.asarray(values)
            kwargs = {"compression": "gzip"} if array.ndim else {}
            handle.create_dataset(name, data=array, **kwargs)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
