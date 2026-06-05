#!/usr/bin/env python3
"""Sweep global phase offsets on 2-D complex boundary voltages.

The experiment solves one complex-admittance 2-D forward problem, keeps the
homogeneous reference frame fixed, then rotates only the target boundary
voltage phase by a set of angles.  Each rotated target frame is reconstructed
with the GUI reconstruction controller's native complex difference path.
"""

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
import matplotlib.tri as mtri
import numpy as np
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from eit_app.controllers.forward_solver_controller import (  # noqa: E402
    ForwardSolverRequest,
    execute_forward_request,
)
from eit_app.controllers.reconstruction_controller import (  # noqa: E402
    ReconstructionRequest,
    ReconstructionResult,
    run_reconstruction_request,
)
from eit_app.models.frame_model import FrameData  # noqa: E402
from eit_app.models.simulation_state import InhomogeneitySpec  # noqa: E402
from pyeidors.runtime_paths import pyeidors_output_path  # noqa: E402
from scripts.common.array_metrics import (  # noqa: E402
    mean_where,
    safe_finite_pearson_correlation,
)


DEFAULT_ANGLES_DEG = (0.0, 1.0, *tuple(float(v) for v in range(10, 181, 10)))


@dataclass(frozen=True)
class SweepConfig:
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
    difference_mode: str
    difference_orientation: str
    difference_preset: str
    compute_dtype: str
    angles_deg: tuple[float, ...]
    threshold_fraction: float


def parse_complex_arg(raw: str) -> complex:
    text = str(raw).strip().replace(" ", "")
    try:
        return complex(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Cannot parse complex scalar {raw!r}; examples: 1+2j, 0.01+0.05j"
        ) from exc


def parse_angles(raw: str | None) -> tuple[float, ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_ANGLES_DEG
    values: list[float] = []
    for item in str(raw).replace(";", ",").split(","):
        text = item.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("At least one phase angle is required.")
    return tuple(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=pyeidors_output_path(
            "diagnostics",
            "phase_offset_complex_reconstruction_sweep",
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
    parser.add_argument("--difference-mode", default="raw")
    parser.add_argument("--difference-orientation", default="target_minus_reference")
    parser.add_argument("--difference-preset", default="eidors_one_step_noser")
    parser.add_argument("--compute-dtype", default="complex128")
    parser.add_argument(
        "--angles",
        default=None,
        help="Comma-separated phase offsets in degrees. Default: 0,1,10,...,180.",
    )
    parser.add_argument("--threshold-fraction", type=float, default=0.5)
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def frame_from_complex(values: np.ndarray, frame_index: int) -> FrameData:
    arr = np.asarray(values, dtype=np.complex128).reshape(-1)
    return FrameData(
        real=np.asarray(arr.real, dtype=np.float64),
        imag=np.asarray(arr.imag, dtype=np.float64),
        timestamp=0.0,
        frame_index=frame_index,
    )


def cell_centers(coords: np.ndarray, cells: np.ndarray) -> np.ndarray:
    node_coords = np.asarray(coords, dtype=np.float64)
    connectivity = np.asarray(cells, dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.mean(node_coords[connectivity, :2], axis=1)


def triangles_for_cells(cells: np.ndarray) -> np.ndarray:
    connectivity = np.asarray(cells, dtype=np.int64)
    if connectivity.ndim != 2 or connectivity.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    verts = int(connectivity.shape[1])
    if verts == 3:
        return connectivity
    if verts == 4:
        return np.vstack(
            [
                connectivity[:, [0, 1, 2]],
                connectivity[:, [0, 2, 3]],
            ]
        )
    raise ValueError(f"Unsupported 2-D cell vertex count: {verts}")


def face_values_for_triangles(values: np.ndarray, cells: np.ndarray) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    connectivity = np.asarray(cells)
    if connectivity.ndim == 2 and connectivity.shape[1] == 4 and arr.size == len(cells):
        return np.repeat(arr, 2)
    return arr


def tripcolor_cell_data(
    ax: plt.Axes,
    *,
    coords: np.ndarray,
    cells: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> Any:
    node_coords = np.asarray(coords, dtype=np.float64)
    triangles = triangles_for_cells(cells)
    face_values = face_values_for_triangles(values, cells)
    triang = mtri.Triangulation(node_coords[:, 0], node_coords[:, 1], triangles)
    artist = ax.tripcolor(
        triang,
        facecolors=np.asarray(face_values, dtype=np.float64),
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return artist


def robust_limits(
    arrays: list[np.ndarray], fallback: tuple[float, float]
) -> tuple[float, float]:
    finite_values = [
        np.asarray(values, dtype=np.float64).reshape(-1)
        for values in arrays
        if np.asarray(values).size
    ]
    if not finite_values:
        return fallback
    merged = np.concatenate(finite_values)
    merged = merged[np.isfinite(merged)]
    if merged.size == 0:
        return fallback
    lo = float(np.percentile(merged, 1.0))
    hi = float(np.percentile(merged, 99.0))
    span = hi - lo
    if span <= 1.0e-12:
        center = 0.5 * (hi + lo)
        return center - 0.5, center + 0.5
    pad = 0.05 * span
    return lo - pad, hi + pad


def metric_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def reconstruction_metrics(
    *,
    angle_deg: float,
    truth_sigma: np.ndarray,
    baseline_sigma: np.ndarray,
    shifted_sigma: np.ndarray,
    background: complex,
    centers: np.ndarray,
    anomaly_center: np.ndarray,
    anomaly_radius: float,
    threshold_fraction: float,
) -> dict[str, float]:
    truth = np.asarray(truth_sigma, dtype=np.complex128).reshape(-1)
    baseline = np.asarray(baseline_sigma, dtype=np.complex128).reshape(-1)
    shifted = np.asarray(shifted_sigma, dtype=np.complex128).reshape(-1)
    if truth.size != shifted.size or baseline.size != shifted.size:
        return {
            "angle_deg": float(angle_deg),
            "metric_status": float("nan"),
        }

    truth_delta_abs = np.abs(truth - background)
    shifted_delta_abs = np.abs(shifted - background)
    baseline_delta_abs = np.abs(baseline - background)
    diff_truth = shifted - truth
    diff_baseline = shifted - baseline
    target_mask = np.linalg.norm(
        centers[:, :2] - anomaly_center[None, :2], axis=1
    ) <= float(anomaly_radius)
    background_mask = ~target_mask
    peak_index = int(np.nanargmax(shifted_delta_abs)) if shifted_delta_abs.size else 0
    peak_xy = centers[peak_index, :2] if centers.size else np.full(2, np.nan)
    peak_error = float(np.linalg.norm(peak_xy - anomaly_center[:2]))
    peak_value = (
        float(np.nanmax(shifted_delta_abs)) if shifted_delta_abs.size else float("nan")
    )
    peak_threshold = float(threshold_fraction) * peak_value
    active_mask = (
        shifted_delta_abs >= peak_threshold
        if np.isfinite(peak_threshold)
        else np.zeros_like(target_mask)
    )
    union = active_mask | target_mask
    intersection = active_mask & target_mask
    iou = (
        float(np.count_nonzero(intersection) / np.count_nonzero(union))
        if np.any(union)
        else float("nan")
    )
    return {
        "angle_deg": float(angle_deg),
        "corr_abs_delta_vs_truth": safe_finite_pearson_correlation(
            truth_delta_abs,
            shifted_delta_abs,
        ),
        "corr_abs_delta_vs_original_recon": safe_finite_pearson_correlation(
            baseline_delta_abs,
            shifted_delta_abs,
        ),
        "relative_l2_vs_truth_complex": float(
            np.linalg.norm(diff_truth) / (np.linalg.norm(truth - background) + 1.0e-12)
        ),
        "relative_l2_vs_original_recon_complex": float(
            np.linalg.norm(diff_baseline)
            / (np.linalg.norm(baseline - background) + 1.0e-12)
        ),
        "target_abs_delta_mean": mean_where(shifted_delta_abs, target_mask),
        "background_abs_delta_mean": mean_where(shifted_delta_abs, background_mask),
        "target_over_background_abs_delta": float(
            mean_where(shifted_delta_abs, target_mask)
            / (mean_where(shifted_delta_abs, background_mask, fallback=0.0) + 1.0e-12)
        ),
        "peak_abs_delta": peak_value,
        "peak_error": peak_error,
        "peak_error_over_radius": float(peak_error / float(anomaly_radius)),
        "iou_peak_fraction_vs_true_circle": iou,
    }


def build_forward_request(cfg: SweepConfig) -> ForwardSolverRequest:
    base_meta: dict[str, Any] = {
        "mesh_dimension": 2,
        "mesh_refinement": cfg.mesh_size,
        "mesh_size": cfg.mesh_size,
        "potential_order": 1,
        "background_conductivity": cfg.background_admittance,
        "noise_level": 0.0,
        "n_elec": cfg.n_elec,
        "n_rings": 1,
        "electrode_layout": "ring_major",
        "measurement_protocol": "eidors_full_3d",
        "stim_pattern": "{ad}",
        "meas_pattern": "{ad}",
        "rotate_meas": True,
        "use_meas_current": False,
        "stim_direction": "ccw",
        "meas_direction": "ccw",
        "stim_first_positive": False,
        "drive_mode": cfg.drive_mode,
        "drive_value": cfg.drive_value,
        "geometry_scale_to_m": 1.0,
        "electrode_coverage": cfg.electrode_coverage,
        "contact_impedance": cfg.contact_impedance,
        "radius": cfg.radius,
        "height": 1.0,
        "electrode_height_ratio": 0.2,
        "electrode_level_fractions": (0.25, 0.75),
        "z_center": 0.0,
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
        "acceleration_profile": "default",
    }
    return ForwardSolverRequest(
        mesh_dimension=2,
        mesh_refinement=cfg.mesh_size,
        n_electrodes=cfg.n_elec,
        background_conductivity=cfg.background_admittance,
        noise_level=0.0,
        inhomogeneities=[
            InhomogeneitySpec(
                shape="circle",
                center_x=cfg.anomaly_center_x,
                center_y=cfg.anomaly_center_y,
                size_x=cfg.anomaly_radius,
                size_y=cfg.anomaly_radius,
                conductivity=cfg.anomaly_admittance,
            )
        ],
        forward_model_config=base_meta,
    )


def build_reconstruction_metadata(
    forward_meta: dict[str, Any],
    cfg: SweepConfig,
) -> dict[str, Any]:
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
            "reconstruction_runtime": "full_gn",
            "eit_value_mode": "complex_admittance",
            "complex_measurement_mode": "native_real_imag",
            "complex_reconstruction_dispatch": "native_complex",
            "compute_dtype": cfg.compute_dtype,
            "petsc_device": "cpu",
            "device": "cpu",
            "forward_backend": "dolfinx",
            "solver_mode": "auto",
            "line_search_mode": "auto",
            "linear_solver": "auto",
            "preconditioner": "auto",
            "fast_linear_path": "auto",
            "forward_solver_preset": "auto",
            "forward_mat_solve": "auto",
            "linearized_solver_strategy": "auto",
            "linearized_maxiter": 0,
        }
    )
    return meta


def run_single_reconstruction(
    *,
    reference_voltage: np.ndarray,
    target_voltage: np.ndarray,
    metadata: dict[str, Any],
    cfg: SweepConfig,
    frame_index: int,
) -> ReconstructionResult:
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


def plot_angle_comparison(
    *,
    output_path: Path,
    angle_deg: float,
    truth_coords: np.ndarray,
    truth_cells: np.ndarray,
    truth_sigma: np.ndarray,
    recon_coords: np.ndarray,
    recon_cells: np.ndarray,
    baseline_sigma: np.ndarray,
    shifted_sigma: np.ndarray,
    re_limits: tuple[float, float],
    im_limits: tuple[float, float],
    metric: dict[str, float],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True)
    artists = []
    artists.append(
        tripcolor_cell_data(
            axes[0, 0],
            coords=truth_coords,
            cells=truth_cells,
            values=np.real(truth_sigma),
            title="Truth Re",
            cmap="viridis",
            vmin=re_limits[0],
            vmax=re_limits[1],
        )
    )
    artists.append(
        tripcolor_cell_data(
            axes[0, 1],
            coords=recon_coords,
            cells=recon_cells,
            values=np.real(baseline_sigma),
            title="Original Recon Re",
            cmap="viridis",
            vmin=re_limits[0],
            vmax=re_limits[1],
        )
    )
    artists.append(
        tripcolor_cell_data(
            axes[0, 2],
            coords=recon_coords,
            cells=recon_cells,
            values=np.real(shifted_sigma),
            title=f"Phase +{angle_deg:g} deg Recon Re",
            cmap="viridis",
            vmin=re_limits[0],
            vmax=re_limits[1],
        )
    )
    fig.colorbar(artists[-1], ax=axes[0, :], shrink=0.8, label="S/m")

    artists.append(
        tripcolor_cell_data(
            axes[1, 0],
            coords=truth_coords,
            cells=truth_cells,
            values=np.imag(truth_sigma),
            title="Truth Im",
            cmap="magma",
            vmin=im_limits[0],
            vmax=im_limits[1],
        )
    )
    artists.append(
        tripcolor_cell_data(
            axes[1, 1],
            coords=recon_coords,
            cells=recon_cells,
            values=np.imag(baseline_sigma),
            title="Original Recon Im",
            cmap="magma",
            vmin=im_limits[0],
            vmax=im_limits[1],
        )
    )
    artists.append(
        tripcolor_cell_data(
            axes[1, 2],
            coords=recon_coords,
            cells=recon_cells,
            values=np.imag(shifted_sigma),
            title=f"Phase +{angle_deg:g} deg Recon Im",
            cmap="magma",
            vmin=im_limits[0],
            vmax=im_limits[1],
        )
    )
    fig.colorbar(artists[-1], ax=axes[1, :], shrink=0.8, label="S/m")
    fig.suptitle(
        "2-D Complex EIT Global Boundary-Voltage Phase Offset\n"
        f"corr truth={metric_float(metric.get('corr_abs_delta_vs_truth')):.3f}, "
        f"corr original={metric_float(metric.get('corr_abs_delta_vs_original_recon')):.3f}, "
        f"relL2 original={metric_float(metric.get('relative_l2_vs_original_recon_complex')):.3f}, "
        f"peak error/r={metric_float(metric.get('peak_error_over_radius')):.2f}",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_grid(
    *,
    output_path: Path,
    title: str,
    coords: np.ndarray,
    cells: np.ndarray,
    angles_deg: tuple[float, ...],
    values_by_angle: list[np.ndarray],
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    n_items = len(values_by_angle)
    n_cols = 5
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    last_artist = None
    for idx, ax in enumerate(axes_arr.flat):
        if idx >= n_items:
            ax.axis("off")
            continue
        last_artist = tripcolor_cell_data(
            ax,
            coords=coords,
            cells=cells,
            values=values_by_angle[idx],
            title=f"+{angles_deg[idx]:g} deg",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    if last_artist is not None:
        fig.colorbar(
            last_artist, ax=axes_arr.ravel().tolist(), shrink=0.75, label="S/m"
        )
    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_voltage_phase_grid(
    *,
    output_path: Path,
    angles_deg: tuple[float, ...],
    original_voltage: np.ndarray,
    shifted_voltages: list[np.ndarray],
) -> None:
    base = np.asarray(original_voltage, dtype=np.complex128).reshape(-1)
    n_items = len(shifted_voltages)
    n_cols = 5
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.4 * n_rows),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    x = np.arange(1, base.size + 1)
    for idx, ax in enumerate(axes_arr.flat):
        if idx >= n_items:
            ax.axis("off")
            continue
        shifted = np.asarray(shifted_voltages[idx], dtype=np.complex128).reshape(-1)
        phase_delta = np.angle(shifted / base, deg=True)
        ax.plot(x, phase_delta, color="#2a6f9b", linewidth=1.1)
        ax.set_title(f"+{angles_deg[idx]:g} deg")
        ax.set_ylim(-190, 190)
        ax.grid(True, alpha=0.25)
        if idx % n_cols == 0:
            ax.set_ylabel("phase delta (deg)")
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("measurement index")
    fig.suptitle("Applied target-voltage phase offsets", fontsize=15)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def register_times_new_roman() -> None:
    """Register Times New Roman when the script runs from WSL."""

    candidate_paths = (
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    )
    for path in candidate_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def classify_failure_angle(rows: list[dict[str, float]]) -> dict[str, Any]:
    degraded: list[dict[str, float]] = []
    lost: list[dict[str, float]] = []
    for row in rows:
        angle = metric_float(row.get("angle_deg"))
        if angle <= 0:
            continue
        corr_original = metric_float(row.get("corr_abs_delta_vs_original_recon"))
        rel_l2_original = metric_float(row.get("relative_l2_vs_original_recon_complex"))
        peak_over_r = metric_float(row.get("peak_error_over_radius"))
        if corr_original < 0.9 or rel_l2_original > 0.25 or peak_over_r > 1.0:
            degraded.append(row)
        if corr_original < 0.6 or rel_l2_original > 0.75 or peak_over_r > 2.0:
            lost.append(row)
    return {
        "first_noticeably_degraded_angle_deg": (
            metric_float(degraded[0]["angle_deg"]) if degraded else None
        ),
        "first_lost_recognizability_angle_deg": (
            metric_float(lost[0]["angle_deg"]) if lost else None
        ),
        "heuristic_degraded_rule": (
            "corr_vs_original < 0.9 OR rel_l2_vs_original > 0.25 OR peak_error/r > 1.0"
        ),
        "heuristic_lost_rule": (
            "corr_vs_original < 0.6 OR rel_l2_vs_original > 0.75 OR peak_error/r > 2.0"
        ),
    }


def main() -> int:
    args = parse_args()
    cfg = SweepConfig(
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
        difference_mode=str(args.difference_mode),
        difference_orientation=str(args.difference_orientation),
        difference_preset=str(args.difference_preset),
        compute_dtype=str(args.compute_dtype),
        angles_deg=parse_angles(args.angles),
        threshold_fraction=float(args.threshold_fraction),
    )
    output_dir = Path(cfg.output_dir)
    per_angle_dir = output_dir / "per_angle"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_angle_dir.mkdir(parents=True, exist_ok=True)

    register_times_new_roman()
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )

    print(f"Output directory: {output_dir}", flush=True)
    print("Solving 2-D complex forward problem...", flush=True)
    forward_started = time.perf_counter()
    forward = execute_forward_request(
        build_forward_request(cfg),
        progress_cb=lambda message: print(f"  {message}", flush=True),
    )
    if forward.error_msg:
        raise RuntimeError(forward.error_msg)
    print(
        f"Forward solve seconds: {time.perf_counter() - forward_started:.2f}",
        flush=True,
    )

    reference_voltage = np.asarray(forward.homogeneous_voltages, dtype=np.complex128)
    target_voltage = np.asarray(forward.boundary_voltages, dtype=np.complex128)
    truth_sigma = np.asarray(forward.ground_truth_conductivity, dtype=np.complex128)
    truth_coords = np.asarray(forward.node_coords, dtype=np.float64)
    truth_cells = np.asarray(forward.cell_connectivity, dtype=np.int64)
    anomaly_center = np.array(
        [cfg.anomaly_center_x, cfg.anomaly_center_y], dtype=np.float64
    )
    metadata = build_reconstruction_metadata(forward.forward_model_config, cfg)

    print("Running original complex-voltage reconstruction...", flush=True)
    baseline_result = run_single_reconstruction(
        reference_voltage=reference_voltage,
        target_voltage=target_voltage,
        metadata=metadata,
        cfg=cfg,
        frame_index=1,
    )
    baseline_sigma = np.asarray(baseline_result.conductivity, dtype=np.complex128)
    recon_coords = np.asarray(baseline_result.node_coords, dtype=np.float64)
    recon_cells = np.asarray(baseline_result.cell_connectivity, dtype=np.int64)
    recon_centers = cell_centers(recon_coords, recon_cells)

    rows: list[dict[str, float]] = []
    shifted_sigmas: list[np.ndarray] = []
    shifted_voltages: list[np.ndarray] = []
    print("Sweeping phase offsets...", flush=True)
    for idx, angle_deg in enumerate(cfg.angles_deg):
        angle_rad = math.radians(float(angle_deg))
        shifted_voltage = np.abs(target_voltage) * np.exp(
            1j * (np.angle(target_voltage) + angle_rad)
        )
        shifted_voltages.append(np.asarray(shifted_voltage, dtype=np.complex128))
        if abs(angle_deg) <= 1.0e-12:
            result = baseline_result
        else:
            print(f"  Phase +{angle_deg:g} deg", flush=True)
            result = run_single_reconstruction(
                reference_voltage=reference_voltage,
                target_voltage=shifted_voltage,
                metadata=metadata,
                cfg=cfg,
                frame_index=idx + 2,
            )
        shifted_sigma = np.asarray(result.conductivity, dtype=np.complex128)
        shifted_sigmas.append(shifted_sigma)
        row = reconstruction_metrics(
            angle_deg=float(angle_deg),
            truth_sigma=truth_sigma,
            baseline_sigma=baseline_sigma,
            shifted_sigma=shifted_sigma,
            background=cfg.background_admittance,
            centers=recon_centers,
            anomaly_center=anomaly_center,
            anomaly_radius=cfg.anomaly_radius,
            threshold_fraction=cfg.threshold_fraction,
        )
        rows.append(row)

    re_limits = robust_limits(
        [
            np.real(truth_sigma),
            np.real(baseline_sigma),
            *[np.real(v) for v in shifted_sigmas],
        ],
        fallback=(cfg.background_admittance.real, cfg.anomaly_admittance.real),
    )
    im_limits = robust_limits(
        [
            np.imag(truth_sigma),
            np.imag(baseline_sigma),
            *[np.imag(v) for v in shifted_sigmas],
        ],
        fallback=(cfg.background_admittance.imag, cfg.anomaly_admittance.imag),
    )

    print("Rendering visualizations...", flush=True)
    for angle_deg, shifted_sigma, metric in zip(
        cfg.angles_deg, shifted_sigmas, rows, strict=True
    ):
        safe_angle = str(angle_deg).replace(".", "p").replace("-", "m")
        plot_angle_comparison(
            output_path=per_angle_dir / f"phase_offset_{safe_angle}_deg.png",
            angle_deg=float(angle_deg),
            truth_coords=truth_coords,
            truth_cells=truth_cells,
            truth_sigma=truth_sigma,
            recon_coords=recon_coords,
            recon_cells=recon_cells,
            baseline_sigma=baseline_sigma,
            shifted_sigma=shifted_sigma,
            re_limits=re_limits,
            im_limits=im_limits,
            metric=metric,
        )

    plot_reconstruction_grid(
        output_path=output_dir / "summary_re_sigma_grid.png",
        title="Re(sigma) reconstructions after global voltage phase offsets",
        coords=recon_coords,
        cells=recon_cells,
        angles_deg=cfg.angles_deg,
        values_by_angle=[np.real(v) for v in shifted_sigmas],
        cmap="viridis",
        vmin=re_limits[0],
        vmax=re_limits[1],
    )
    plot_reconstruction_grid(
        output_path=output_dir / "summary_im_sigma_grid.png",
        title="Im(sigma) reconstructions after global voltage phase offsets",
        coords=recon_coords,
        cells=recon_cells,
        angles_deg=cfg.angles_deg,
        values_by_angle=[np.imag(v) for v in shifted_sigmas],
        cmap="magma",
        vmin=im_limits[0],
        vmax=im_limits[1],
    )
    plot_reconstruction_grid(
        output_path=output_dir / "summary_abs_delta_grid.png",
        title="|sigma - background| reconstructions after global voltage phase offsets",
        coords=recon_coords,
        cells=recon_cells,
        angles_deg=cfg.angles_deg,
        values_by_angle=[np.abs(v - cfg.background_admittance) for v in shifted_sigmas],
        cmap="viridis",
        vmin=0.0,
        vmax=robust_limits(
            [np.abs(v - cfg.background_admittance) for v in shifted_sigmas],
            fallback=(0.0, abs(cfg.anomaly_admittance - cfg.background_admittance)),
        )[1],
    )
    plot_voltage_phase_grid(
        output_path=output_dir / "applied_voltage_phase_offsets.png",
        angles_deg=cfg.angles_deg,
        original_voltage=target_voltage,
        shifted_voltages=shifted_voltages,
    )

    metrics_path = output_dir / "metrics.csv"
    write_metrics_csv(metrics_path, rows)
    verdict = classify_failure_angle(rows)
    summary = {
        "config": json_ready(asdict(cfg)),
        "forward": {
            "n_elements": int(forward.n_elements),
            "n_measurements": int(forward.n_measurements),
            "node_count": int(len(forward.node_coords)),
            "forward_model_config": json_ready(forward.forward_model_config),
        },
        "result": verdict,
        "metrics": json_ready(rows),
        "outputs": {
            "metrics_csv": str(metrics_path),
            "per_angle_dir": str(per_angle_dir),
            "summary_re_sigma_grid": str(output_dir / "summary_re_sigma_grid.png"),
            "summary_im_sigma_grid": str(output_dir / "summary_im_sigma_grid.png"),
            "summary_abs_delta_grid": str(output_dir / "summary_abs_delta_grid.png"),
            "applied_voltage_phase_offsets": str(
                output_dir / "applied_voltage_phase_offsets.png"
            ),
            "arrays_npz": str(output_dir / "phase_sweep_arrays.npz"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "phase_sweep_arrays.npz",
        angles_deg=np.asarray(cfg.angles_deg, dtype=np.float64),
        truth_sigma=truth_sigma,
        baseline_sigma=baseline_sigma,
        shifted_sigma=np.stack(shifted_sigmas, axis=0),
        reference_voltage=reference_voltage,
        target_voltage=target_voltage,
        shifted_voltage=np.stack(shifted_voltages, axis=0),
        truth_node_coords=truth_coords,
        truth_cell_connectivity=truth_cells,
        recon_node_coords=recon_coords,
        recon_cell_connectivity=recon_cells,
    )
    print(json.dumps(summary["result"], ensure_ascii=False, indent=2), flush=True)
    print(f"Metrics: {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
