#!/usr/bin/env python3
"""Generate a real-valued 2D/3D CPU/GPU absolute-reconstruction gallery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

matplotlib = None  # type: ignore[assignment]
plt = None  # type: ignore[assignment]
TwoSlopeNorm = None  # type: ignore[assignment]
griddata = None  # type: ignore[assignment]

try:  # pragma: no cover - optional in lean environments
    import torch
except Exception:  # pragma: no cover
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyeidors.perf import ACCELERATION_PROFILE_GPU3D
from pyeidors.runtime_paths import pyeidors_output_path
from scripts.common.acceleration_profiles import (
    add_acceleration_profile_argument,
    resolve_3d_mesh_contract,
)
from scripts.common.hdf5_outputs import (
    GALLERY_ARRAYS_SCHEMA,
    read_output_bundle,
    write_output_bundle,
)
from scripts.diagnostics.gallery_shared import (
    consistency_metrics as _shared_consistency_metrics,
    jsonable as _jsonable,
    safe_pearson as _safe_pearson,  # noqa: F401 - module-level helper compatibility
)


BACKGROUND_CONDUCTIVITY = 1.0
REAL_PHANTOM_HIGH = 1.6
REAL_PHANTOM_LOW = 0.65
DEFAULT_OUTPUT_DIR = pyeidors_output_path("diagnostics", "reconstruction_gallery_real")
_MEASUREMENT_REL_TOL = 1e-6
_IMAGE_REL_TOL = 5e-5
_IMAGE_RMSE_TOL = {2: 1e-6, 3: 1.25e-6}


@dataclass(frozen=True)
class AnomalySpec:
    label: str
    center_norm: tuple[float, ...]
    radius_norm: float
    conductivity: float


ANOMALIES_2D = (
    AnomalySpec(
        label="high",
        center_norm=(0.35, 0.0),
        radius_norm=0.18,
        conductivity=REAL_PHANTOM_HIGH,
    ),
    AnomalySpec(
        label="low",
        center_norm=(-0.35, 0.0),
        radius_norm=0.18,
        conductivity=REAL_PHANTOM_LOW,
    ),
)
ANOMALIES_3D = (
    AnomalySpec(
        label="high",
        center_norm=(0.35, 0.0, 0.22),
        radius_norm=0.18,
        conductivity=REAL_PHANTOM_HIGH,
    ),
    AnomalySpec(
        label="low",
        center_norm=(-0.35, 0.0, -0.22),
        radius_norm=0.18,
        conductivity=REAL_PHANTOM_LOW,
    ),
)


def _consistency_metrics(
    *,
    dim: int,
    baseline_cpu_meas: np.ndarray | None,
    baseline_gpu_meas: np.ndarray | None,
    target_cpu_meas: np.ndarray,
    target_gpu_meas: np.ndarray,
    cpu_recon: np.ndarray,
    gpu_recon: np.ndarray,
) -> dict[str, Any]:
    return _shared_consistency_metrics(
        dim=dim,
        baseline_cpu_meas=baseline_cpu_meas,
        baseline_gpu_meas=baseline_gpu_meas,
        target_cpu_meas=target_cpu_meas,
        target_gpu_meas=target_gpu_meas,
        cpu_recon=cpu_recon,
        gpu_recon=gpu_recon,
        measurement_rel_tol=_MEASUREMENT_REL_TOL,
        image_rel_tol=_IMAGE_REL_TOL,
        image_rmse_tol_by_dim=_IMAGE_RMSE_TOL,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--mesh-size-2d", type=float, default=0.08)
    parser.add_argument("--radius-2d", type=float, default=1.0)
    parser.add_argument("--radius-3d", type=float, default=0.18)
    parser.add_argument("--height-3d", type=float, default=0.16)
    parser.add_argument("--refinement-3d", type=int, default=3)
    parser.add_argument("--electrode-height-ratio", type=float, default=0.2)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--contact-impedance", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--slice-resolution", type=int, default=220)
    parser.add_argument(
        "--report-title", type=str, default="Real-Valued Reconstruction Gallery"
    )
    add_acceleration_profile_argument(
        parser,
        flag="--gpu-acceleration-profile",
        default=ACCELERATION_PROFILE_GPU3D,
        help_suffix="Used for the 3D GPU gallery case and forwarded to the worker.",
    )
    return parser.parse_args()


def _cuda_available() -> bool:
    return bool(
        torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()
    )


def _ensure_plot_stack() -> None:
    global matplotlib, plt, TwoSlopeNorm, griddata
    if plt is not None and TwoSlopeNorm is not None and griddata is not None:
        return
    import matplotlib as _matplotlib

    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm
    from scipy.interpolate import griddata as _griddata

    matplotlib = _matplotlib  # type: ignore[assignment]
    plt = _plt  # type: ignore[assignment]
    TwoSlopeNorm = _TwoSlopeNorm  # type: ignore[assignment]
    griddata = _griddata  # type: ignore[assignment]


def _griddata_fill(
    points: np.ndarray, values: np.ndarray, query: np.ndarray
) -> np.ndarray:
    _ensure_plot_stack()
    sampled = griddata(points, values, query, method="linear")
    if np.isnan(sampled).any():
        missing = np.isnan(sampled)
        sampled[missing] = griddata(points, values, query[missing], method="nearest")
    return np.asarray(sampled, dtype=np.float64)


def _query_points(size: int, *columns: np.ndarray | float) -> np.ndarray:
    count = int(size)
    out = np.empty((count, len(columns)), dtype=np.float64)
    for idx, column in enumerate(columns):
        if np.ndim(column) == 0:
            out[:, idx] = float(column)
            continue
        arr = np.asarray(column, dtype=np.float64).reshape(-1)
        if arr.size != count:
            raise ValueError(
                f"query column length {arr.size} does not match {count} samples"
            )
        out[:, idx] = arr
    return out


def _sample_2d_field(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    radius: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_norm = np.linspace(-1.0, 1.0, int(resolution))
    y_norm = np.linspace(-1.0, 1.0, int(resolution))
    x_grid, y_grid = np.meshgrid(x_norm, y_norm)
    center = coords.mean(axis=0)
    query = _query_points(
        x_grid.size,
        center[0] + x_grid.ravel() * float(radius),
        center[1] + y_grid.ravel() * float(radius),
    )
    sampled = _griddata_fill(
        coords[:, :2], np.asarray(values, dtype=np.float64), query
    ).reshape(x_grid.shape)
    mask = (x_grid**2 + y_grid**2) > 1.0
    sampled[mask] = np.nan
    return x_grid, y_grid, sampled


def _sample_2d_profile(
    *, x_grid: np.ndarray, y_grid: np.ndarray, field: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mid_idx = int(np.argmin(np.abs(y_grid[:, 0])))
    return x_grid[mid_idx, :], field[mid_idx, :]


def _sample_3d_slice(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    radius: float,
    z_half_height: float,
    plane: str,
    plane_value_norm: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_norm = np.linspace(-1.0, 1.0, int(resolution))
    z_norm = np.linspace(-1.0, 1.0, int(resolution))
    center = coords.mean(axis=0)
    if plane == "axial":
        y_norm = np.linspace(-1.0, 1.0, int(resolution))
        x_grid, y_grid = np.meshgrid(x_norm, y_norm)
        z_value = center[2] + float(plane_value_norm) * float(z_half_height)
        query = _query_points(
            x_grid.size,
            center[0] + x_grid.ravel() * float(radius),
            center[1] + y_grid.ravel() * float(radius),
            z_value,
        )
        sampled = _griddata_fill(coords[:, :3], values, query).reshape(x_grid.shape)
        sampled[(x_grid**2 + y_grid**2) > 1.0] = np.nan
        return x_grid, y_grid, sampled

    if plane != "coronal":
        raise ValueError(f"unsupported plane {plane!r}")

    x_grid, z_grid = np.meshgrid(x_norm, z_norm)
    y_value = center[1] + float(plane_value_norm) * float(radius)
    query = _query_points(
        x_grid.size,
        center[0] + x_grid.ravel() * float(radius),
        y_value,
        center[2] + z_grid.ravel() * float(z_half_height),
    )
    sampled = _griddata_fill(coords[:, :3], values, query).reshape(x_grid.shape)
    sampled[np.abs(x_grid) > 1.0] = np.nan
    return x_grid, z_grid, sampled


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _apply_plot_style() -> None:
    _ensure_plot_stack()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _render_2d_overview(
    *,
    output_path: Path,
    truth_grid: np.ndarray,
    cpu_grid: np.ndarray,
    gpu_grid: np.ndarray,
    profile_x: np.ndarray,
    profile_truth: np.ndarray,
    profile_cpu: np.ndarray,
    profile_gpu: np.ndarray,
    consistency: dict[str, Any],
    cpu_metrics: dict[str, float],
    gpu_metrics: dict[str, float],
) -> None:
    _apply_plot_style()
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.1], hspace=0.18, wspace=0.25)

    recon_limit = float(
        max(
            np.nanmax(np.abs(truth_grid)),
            np.nanmax(np.abs(cpu_grid)),
            np.nanmax(np.abs(gpu_grid)),
            1e-9,
        )
    )
    diff_grid = gpu_grid - cpu_grid
    cpu_err = cpu_grid - truth_grid
    gpu_err = gpu_grid - truth_grid
    err_limit = float(
        max(
            np.nanmax(np.abs(diff_grid)),
            np.nanmax(np.abs(cpu_err)),
            np.nanmax(np.abs(gpu_err)),
            1e-9,
        )
    )

    recon_norm = TwoSlopeNorm(vcenter=0.0, vmin=-recon_limit, vmax=recon_limit)
    err_norm = TwoSlopeNorm(vcenter=0.0, vmin=-err_limit, vmax=err_limit)
    extent = (-1.0, 1.0, -1.0, 1.0)

    panels = [
        ("Truth Δσ", truth_grid, recon_norm, "PuOr_r"),
        ("CPU Recon Δσ", cpu_grid, recon_norm, "PuOr_r"),
        ("GPU Recon Δσ", gpu_grid, recon_norm, "PuOr_r"),
        ("GPU - CPU Δσ", diff_grid, err_norm, "RdBu_r"),
        ("CPU Error", cpu_err, err_norm, "RdBu_r"),
        ("GPU Error", gpu_err, err_norm, "RdBu_r"),
    ]

    image_handles = []
    for idx, (title, field, norm, cmap) in enumerate(panels):
        row = 0 if idx < 4 else 1
        col = idx if idx < 4 else idx - 4
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap, norm=norm)
        image_handles.append(im)
        ax.set_title(title)
        ax.set_xlabel("x / R")
        ax.set_ylabel("y / R")
        ax.set_aspect("equal")

    profile_ax = fig.add_subplot(gs[1, 2])
    profile_ax.plot(
        profile_x, profile_truth, label="Truth", color="#000000", linewidth=2.0
    )
    profile_ax.plot(profile_x, profile_cpu, label="CPU", color="#1f77b4", linewidth=1.8)
    profile_ax.plot(
        profile_x,
        profile_gpu,
        label="GPU",
        color="#d55e00",
        linewidth=1.8,
        linestyle="--",
    )
    profile_ax.set_title("Centerline Profile (y = 0)")
    profile_ax.set_xlabel("x / R")
    profile_ax.set_ylabel("Δσ")
    profile_ax.legend(frameon=False, fontsize=8)

    text_ax = fig.add_subplot(gs[1, 3])
    text_ax.axis("off")
    baseline_text = (
        "n/a"
        if consistency["baseline_measurement_relative_l2"] is None
        else f"{consistency['baseline_measurement_relative_l2']:.3e}"
    )
    summary_lines = [
        f"2D Consistency: {'PASS' if consistency['passed'] else 'FAIL'}",
        f"baseline meas rel-L2: {baseline_text}",
        f"target meas rel-L2:   {consistency['target_measurement_relative_l2']:.3e}",
        f"image rel-L2:         {consistency['image_relative_l2']:.3e}",
        f"image RMSE:           {consistency['image_rmse']:.3e}",
        "",
        "Truth vs Recon",
        f"CPU rel-L2: {cpu_metrics['relative_l2']:.3e}",
        f"GPU rel-L2: {gpu_metrics['relative_l2']:.3e}",
        f"CPU CRC(high/low): {cpu_metrics['contrast_recovery_high']:.3f} / {cpu_metrics['contrast_recovery_low']:.3f}",
        f"GPU CRC(high/low): {gpu_metrics['contrast_recovery_high']:.3f} / {gpu_metrics['contrast_recovery_low']:.3f}",
    ]
    text_ax.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9,
    )

    fig.colorbar(
        image_handles[0],
        ax=[fig.axes[0], fig.axes[1], fig.axes[2]],
        fraction=0.026,
        pad=0.02,
        label="Δσ",
    )
    fig.colorbar(
        image_handles[3],
        ax=[fig.axes[3], fig.axes[4], fig.axes[5]],
        fraction=0.026,
        pad=0.02,
        label="Δσ diff",
    )
    fig.suptitle("2D Absolute Reconstruction Overview", fontsize=14, fontweight="bold")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_3d_overview(
    *,
    output_path: Path,
    truth_slices: list[np.ndarray],
    cpu_slices: list[np.ndarray],
    gpu_slices: list[np.ndarray],
    slice_titles: list[str],
    consistency: dict[str, Any],
) -> None:
    _apply_plot_style()
    fig, axes = plt.subplots(3, 4, figsize=(14, 11), constrained_layout=True)
    recon_limit = float(
        max(
            max(np.nanmax(np.abs(field)) for field in truth_slices),
            max(np.nanmax(np.abs(field)) for field in cpu_slices),
            max(np.nanmax(np.abs(field)) for field in gpu_slices),
            1e-9,
        )
    )
    diff_slices = [gpu - cpu for cpu, gpu in zip(cpu_slices, gpu_slices)]
    diff_limit = float(
        max(max(np.nanmax(np.abs(field)) for field in diff_slices), 1e-9)
    )
    recon_norm = TwoSlopeNorm(vcenter=0.0, vmin=-recon_limit, vmax=recon_limit)
    diff_norm = TwoSlopeNorm(vcenter=0.0, vmin=-diff_limit, vmax=diff_limit)
    recon_handle = None
    diff_handle = None

    for row, title in enumerate(slice_titles):
        panels = [
            ("Truth Δσ", truth_slices[row], "PuOr_r", recon_norm),
            ("CPU Recon Δσ", cpu_slices[row], "PuOr_r", recon_norm),
            ("GPU Recon Δσ", gpu_slices[row], "PuOr_r", recon_norm),
            ("GPU - CPU Δσ", diff_slices[row], "RdBu_r", diff_norm),
        ]
        for col, (panel_title, field, cmap, norm) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                field,
                origin="lower",
                extent=(-1.0, 1.0, -1.0, 1.0),
                cmap=cmap,
                norm=norm,
            )
            if col == 0:
                ax.set_ylabel(title)
            if row == 0:
                ax.set_title(panel_title)
            ax.set_xlabel("normalized coord")
            ax.set_aspect("equal")
            if col < 3:
                recon_handle = im
            else:
                diff_handle = im

    if recon_handle is not None:
        fig.colorbar(
            recon_handle,
            ax=axes[:, :3].ravel().tolist(),
            fraction=0.028,
            pad=0.02,
            label="Δσ",
        )
    if diff_handle is not None:
        fig.colorbar(
            diff_handle,
            ax=axes[:, 3].ravel().tolist(),
            fraction=0.028,
            pad=0.02,
            label="Δσ diff",
        )
    fig.suptitle(
        "3D Absolute Reconstruction Overview\n"
        f"Consistency: {'PASS' if consistency['passed'] else 'FAIL'} | "
        f"baseline meas rel-L2={consistency['baseline_measurement_relative_l2']:.3e} | "
        f"target meas rel-L2={consistency['target_measurement_relative_l2']:.3e} | "
        f"image rel-L2={consistency['image_relative_l2']:.3e}",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return (
            f"{float(value):.3e}"
            if abs(float(value)) < 1e-2 or abs(float(value)) >= 1e3
            else f"{float(value):.4f}"
        )
    return str(value)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| "
        + " | ".join(_format_metric(row.get(column, "")) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _relative_drift(first: float, second: float) -> float:
    denom = max(abs(float(first)), abs(float(second)), 1e-12)
    return float(abs(float(first) - float(second)) / denom)


def _fairness_thresholds(*, dim_label: str) -> tuple[float, float, float]:
    if str(dim_label).upper() == "3D":
        return (0.10, 0.05, 0.10)
    return (0.10, 0.05, 0.10)


def _fairness_order_row(
    dim_label: str, order_label: str, run: dict[str, Any], *, report_only: bool
) -> dict[str, Any]:
    cpu = run["backend_runs"]["cpu"]
    gpu = run["backend_runs"]["gpu"]
    return {
        "dimension": dim_label,
        "order": order_label,
        "cold_forward_speedup_x": float(
            cpu["cold"]["forward_elapsed_sec"] / gpu["cold"]["forward_elapsed_sec"]
        ),
        "hot_forward_speedup_x": float(
            cpu["hot"]["forward_elapsed_sec"] / gpu["hot"]["forward_elapsed_sec"]
        ),
        "cold_inverse_speedup_x": float(
            cpu["cold"]["inverse_total_elapsed_sec"]
            / gpu["cold"]["inverse_total_elapsed_sec"]
        ),
        "hot_inverse_speedup_x": float(
            cpu["hot"]["inverse_total_elapsed_sec"]
            / gpu["hot"]["inverse_total_elapsed_sec"]
        ),
        "report_only": bool(report_only),
    }


def _fairness_backend_row(
    dim_label: str,
    backend_label: str,
    cpu_first: dict[str, Any],
    gpu_first: dict[str, Any],
    *,
    report_only: bool,
) -> dict[str, Any]:
    cold_forward_tol, hot_forward_tol, inverse_tol = _fairness_thresholds(
        dim_label=dim_label
    )
    first = cpu_first["backend_runs"][backend_label]
    second = gpu_first["backend_runs"][backend_label]
    cold_forward_drift = _relative_drift(
        first["cold"]["forward_elapsed_sec"], second["cold"]["forward_elapsed_sec"]
    )
    hot_forward_drift = _relative_drift(
        first["hot"]["forward_elapsed_sec"], second["hot"]["forward_elapsed_sec"]
    )
    inverse_drift = _relative_drift(
        first["cold"]["inverse_total_elapsed_sec"],
        second["cold"]["inverse_total_elapsed_sec"],
    )
    passed = bool(
        cold_forward_drift <= cold_forward_tol
        and hot_forward_drift <= hot_forward_tol
        and inverse_drift <= inverse_tol
    )
    return {
        "dimension": dim_label,
        "backend": first["label"].split()[-1],
        "cold_forward_drift_rel": cold_forward_drift,
        "hot_forward_drift_rel": hot_forward_drift,
        "inverse_total_drift_rel": inverse_drift,
        "cold_forward_tol": cold_forward_tol,
        "hot_forward_tol": hot_forward_tol,
        "inverse_tol": inverse_tol,
        "report_only": bool(report_only),
        "passed": bool(True if report_only else passed),
    }


def _write_report(
    *,
    output_path: Path,
    title: str,
    figures: dict[str, str],
    config: dict[str, Any],
    case_rows: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]],
    fairness_order_rows: list[dict[str, Any]],
    fairness_backend_rows: list[dict[str, Any]],
    all_passed: bool,
) -> None:
    def _speed_text(row: dict[str, Any], key: str) -> str:
        value = row.get(key)
        if value is None:
            return "n/a"
        return f"{float(value):.2f}x"

    fairness_3d_rows = [
        row
        for row in fairness_backend_rows
        if row["dimension"] == "3D" and not row["report_only"]
    ]
    fairness_3d_pass = bool(
        fairness_3d_rows and all(row["passed"] for row in fairness_3d_rows)
    )
    mesh_family_default, geometry_version_default, generator_revision_default = (
        resolve_3d_mesh_contract(
            acceleration_profile=config.get(
                "gpu_acceleration_profile", ACCELERATION_PROFILE_GPU3D
            ),
        )
    )
    mesh_family = str(config.get("mesh_family_3d", mesh_family_default))
    geometry_version = str(config.get("geometry_version_3d", geometry_version_default))
    generator_revision = str(
        config.get("generator_revision_3d", generator_revision_default)
    )
    quick_lines = [
        f"- 2D consistency: {'PASS' if consistency_rows[0]['passed'] else 'FAIL'}",
        f"- 3D consistency: {'PASS' if consistency_rows[1]['passed'] else 'FAIL'}",
        f"- 3D fairness: {'PASS' if fairness_3d_pass else 'FAIL'}",
        f"- Overall gallery gate: {'PASS' if all_passed else 'FAIL'}",
        "- Speed summary:",
        *[
            (
                f"  - {row['dimension']} {row['order']}: "
                f"cold forward {_speed_text(row, 'cold_forward_speedup_x')}, "
                f"hot forward {_speed_text(row, 'hot_forward_speedup_x')}, "
                f"cold inverse {_speed_text(row, 'cold_inverse_speedup_x')}, "
                f"hot inverse {_speed_text(row, 'hot_inverse_speedup_x')}"
            )
            for row in fairness_order_rows
        ],
        f"- 2D overview: ![2D overview]({figures['2d_overview']})",
        f"- 3D overview: ![3D overview]({figures['3d_overview']})",
    ]
    content = "\n".join(
        [
            f"# {title}",
            "",
            "## Setup",
            "",
            "- Mode: `absolute`, `real-valued`, `noise-free`",
            "- 2D backends: `dolfinx/cpu` vs `dolfinx/cuda`",
            "- 3D backends: `dolfinx/cpu` vs `cuda_structured`",
            f"- 3D mesh: `{mesh_family} + {geometry_version} + {generator_revision}`, refinement `{config['refinement_3d']}`",
            "",
            "## Quick Summary",
            "",
            *quick_lines,
            "",
            "## Figures",
            "",
            f"![2D overview]({figures['2d_overview']})",
            "",
            f"![3D overview]({figures['3d_overview']})",
            "",
            "## Truth vs Reconstruction",
            "",
            _markdown_table(case_rows),
            "",
            "## CPU vs GPU Consistency",
            "",
            _markdown_table(consistency_rows),
            "",
            "## Cross-Order Fairness",
            "",
            "### Order Speedups",
            "",
            _markdown_table(fairness_order_rows),
            "",
            "### Backend Drift",
            "",
            _markdown_table(fairness_backend_rows),
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")


def _case_table_row(dim_label: str, case: dict[str, Any]) -> dict[str, Any]:
    metrics = case["truth_metrics"]
    return {
        "case": f"{dim_label} {case['label'].split()[-1]}",
        "forward_backend": case["forward_backend"],
        "forward_sec": case["forward_elapsed_sec"],
        "inverse_sec": case["inverse_total_elapsed_sec"],
        "truth_rel_l2": metrics["relative_l2"],
        "truth_rmse": metrics["rmse"],
        "truth_pearson": metrics["pearson"],
        "crc_high": metrics["contrast_recovery_high"],
        "crc_low": metrics["contrast_recovery_low"],
        "background_bias": metrics["background_bias"],
        "data": case["data_path"],
    }


def _consistency_table_row(
    dim_label: str,
    consistency: dict[str, Any],
    *,
    cpu_case: dict[str, Any],
    gpu_case: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dimension": dim_label,
        "baseline_meas_rel_l2": consistency["baseline_measurement_relative_l2"],
        "baseline_meas_pass": consistency["baseline_measurement_pass"],
        "target_meas_rel_l2": consistency["target_measurement_relative_l2"],
        "target_meas_pass": consistency["target_measurement_pass"],
        "image_rel_l2": consistency["image_relative_l2"],
        "image_rmse": consistency["image_rmse"],
        "image_pearson": consistency["image_pearson"],
        "forward_speedup_x": float(
            cpu_case["forward_elapsed_sec"] / gpu_case["forward_elapsed_sec"]
        ),
        "inverse_speedup_x": float(
            cpu_case["inverse_total_elapsed_sec"]
            / gpu_case["inverse_total_elapsed_sec"]
        ),
        "passed": consistency["passed"],
    }


def _worker_command(
    args: argparse.Namespace,
    *,
    dim: int,
    output_dir: Path,
    worker_output_json: Path,
    run_kind: str = "correctness",
    backend_order: str = "cpu-first",
    backend_key: str = "both",
) -> list[str]:
    cmd = [
        sys.executable,
        str(
            (
                Path(__file__).resolve().parent
                / "run_real_reconstruction_gallery_worker.py"
            )
        ),
        "--dim",
        str(dim),
        "--output-dir",
        str(output_dir),
        "--n-elec",
        str(args.n_elec),
        "--mesh-size-2d",
        str(args.mesh_size_2d),
        "--radius-2d",
        str(args.radius_2d),
        "--radius-3d",
        str(args.radius_3d),
        "--height-3d",
        str(args.height_3d),
        "--refinement-3d",
        str(args.refinement_3d),
        "--electrode-height-ratio",
        str(args.electrode_height_ratio),
        "--electrode-coverage",
        str(args.electrode_coverage),
        "--contact-impedance",
        str(args.contact_impedance),
        "--max-iterations",
        str(args.max_iterations),
        "--gpu-acceleration-profile",
        str(args.gpu_acceleration_profile),
        "--run-kind",
        str(run_kind),
        "--backend-order",
        str(backend_order),
        "--backend-key",
        str(backend_key),
        "--worker-output-json",
        str(worker_output_json),
    ]
    return cmd


def _run_worker(
    args: argparse.Namespace,
    *,
    dim: int,
    output_dir: Path,
    run_kind: str = "correctness",
    backend_order: str = "cpu-first",
    backend_key: str = "both",
    clean_output_dir: bool = True,
) -> dict[str, Any]:
    if clean_output_dir:
        _ensure_clean_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    worker_json = output_dir / "data" / f"{dim}d_worker_summary.json"
    cmd = _worker_command(
        args,
        dim=dim,
        output_dir=output_dir,
        worker_output_json=worker_json,
        run_kind=run_kind,
        backend_order=backend_order,
        backend_key=backend_key,
    )
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)
    return json.loads(worker_json.read_text(encoding="utf-8"))


def _run_correctness_pair(
    args: argparse.Namespace, *, dim: int, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if int(dim) == 2:
        summary = _run_worker(
            args,
            dim=dim,
            output_dir=output_dir,
            run_kind="correctness",
            backend_key="both",
            backend_order="cpu-first",
            clean_output_dir=True,
        )
        bundle = read_output_bundle(output_dir / summary["bundle_path"])
        bundles = {
            "cpu": {
                "coords": np.asarray(bundle["coords"], dtype=np.float64),
                "truth_values": np.asarray(bundle["truth_values"], dtype=np.float64),
                "reconstruction": np.asarray(
                    bundle["cpu_reconstruction"], dtype=np.float64
                ),
            },
            "gpu": {
                "coords": np.asarray(bundle["coords"], dtype=np.float64),
                "truth_values": np.asarray(bundle["truth_values"], dtype=np.float64),
                "reconstruction": np.asarray(
                    bundle["gpu_reconstruction"], dtype=np.float64
                ),
            },
        }
        return (
            summary,
            {"case": summary["cpu_case"]},
            {"case": summary["gpu_case"]},
            bundles,
        )

    cpu_summary = _run_worker(
        args,
        dim=dim,
        output_dir=output_dir,
        run_kind="correctness",
        backend_key="cpu",
        clean_output_dir=True,
    )
    gpu_summary = _run_worker(
        args,
        dim=dim,
        output_dir=output_dir,
        run_kind="correctness",
        backend_key="gpu",
        clean_output_dir=False,
    )
    cpu_bundle = read_output_bundle(output_dir / cpu_summary["bundle_path"])
    gpu_bundle = read_output_bundle(output_dir / gpu_summary["bundle_path"])
    cpu_case = read_output_bundle(output_dir / cpu_summary["case"]["data_path"])
    gpu_case = read_output_bundle(output_dir / gpu_summary["case"]["data_path"])
    consistency = _consistency_metrics(
        dim=dim,
        baseline_cpu_meas=cpu_case["baseline_measured"],
        baseline_gpu_meas=gpu_case["baseline_measured"],
        target_cpu_meas=cpu_case["measured"],
        target_gpu_meas=gpu_case["measured"],
        cpu_recon=np.asarray(cpu_bundle["reconstruction"], dtype=np.float64),
        gpu_recon=np.asarray(gpu_bundle["reconstruction"], dtype=np.float64),
    )
    combined = {
        "mesh_radius": float(cpu_summary["mesh_radius"]),
        "anomalies": cpu_summary["anomalies"],
        "cpu_truth_metrics": cpu_summary["truth_metrics"],
        "gpu_truth_metrics": gpu_summary["truth_metrics"],
        "cpu_case": cpu_summary["case"],
        "gpu_case": gpu_summary["case"],
        "consistency": consistency,
    }
    return combined, cpu_summary, gpu_summary, {"cpu": cpu_bundle, "gpu": gpu_bundle}


def _run_fairness_order(
    args: argparse.Namespace, *, dim: int, output_dir: Path, order_label: str
) -> dict[str, Any]:
    if int(dim) == 2:
        cold = _run_worker(
            args,
            dim=dim,
            output_dir=output_dir,
            run_kind="correctness",
            backend_key="both",
            backend_order=order_label,
            clean_output_dir=True,
        )
        hot = _run_worker(
            args,
            dim=dim,
            output_dir=output_dir,
            run_kind="correctness",
            backend_key="both",
            backend_order=order_label,
            clean_output_dir=False,
        )
        return {
            "dim": int(dim),
            "backend_order": ["cpu", "gpu"]
            if order_label == "cpu-first"
            else ["gpu", "cpu"],
            "backend_runs": {
                "cpu": {
                    "label": cold["cpu_case"]["label"],
                    "forward_backend": cold["cpu_case"]["forward_backend"],
                    "petsc_device": cold["cpu_case"]["petsc_device"],
                    "cold": {
                        "forward_baseline_elapsed_sec": cold["cpu_case"][
                            "forward_baseline_elapsed_sec"
                        ],
                        "forward_elapsed_sec": cold["cpu_case"]["forward_elapsed_sec"],
                        "inverse_total_elapsed_sec": cold["cpu_case"][
                            "inverse_total_elapsed_sec"
                        ],
                    },
                    "hot": {
                        "forward_baseline_elapsed_sec": hot["cpu_case"][
                            "forward_baseline_elapsed_sec"
                        ],
                        "forward_elapsed_sec": hot["cpu_case"]["forward_elapsed_sec"],
                        "inverse_total_elapsed_sec": hot["cpu_case"][
                            "inverse_total_elapsed_sec"
                        ],
                    },
                },
                "gpu": {
                    "label": cold["gpu_case"]["label"],
                    "forward_backend": cold["gpu_case"]["forward_backend"],
                    "petsc_device": cold["gpu_case"]["petsc_device"],
                    "cold": {
                        "forward_baseline_elapsed_sec": cold["gpu_case"][
                            "forward_baseline_elapsed_sec"
                        ],
                        "forward_elapsed_sec": cold["gpu_case"]["forward_elapsed_sec"],
                        "inverse_total_elapsed_sec": cold["gpu_case"][
                            "inverse_total_elapsed_sec"
                        ],
                    },
                    "hot": {
                        "forward_baseline_elapsed_sec": hot["gpu_case"][
                            "forward_baseline_elapsed_sec"
                        ],
                        "forward_elapsed_sec": hot["gpu_case"]["forward_elapsed_sec"],
                        "inverse_total_elapsed_sec": hot["gpu_case"][
                            "inverse_total_elapsed_sec"
                        ],
                    },
                },
            },
        }

    order = ["cpu", "gpu"] if order_label == "cpu-first" else ["gpu", "cpu"]
    backend_runs: dict[str, Any] = {}
    first_backend = True
    for backend_key in order:
        cold = _run_worker(
            args,
            dim=dim,
            output_dir=output_dir,
            run_kind="correctness",
            backend_key=backend_key,
            clean_output_dir=first_backend,
        )
        hot = _run_worker(
            args,
            dim=dim,
            output_dir=output_dir,
            run_kind="correctness",
            backend_key=backend_key,
            clean_output_dir=False,
        )
        backend_runs[backend_key] = {
            "label": cold["case"]["label"],
            "forward_backend": cold["case"]["forward_backend"],
            "petsc_device": cold["case"]["petsc_device"],
            "cold": {
                "forward_baseline_elapsed_sec": cold["case"][
                    "forward_baseline_elapsed_sec"
                ],
                "forward_elapsed_sec": cold["case"]["forward_elapsed_sec"],
                "inverse_total_elapsed_sec": cold["case"]["inverse_total_elapsed_sec"],
            },
            "hot": {
                "forward_baseline_elapsed_sec": hot["case"][
                    "forward_baseline_elapsed_sec"
                ],
                "forward_elapsed_sec": hot["case"]["forward_elapsed_sec"],
                "inverse_total_elapsed_sec": hot["case"]["inverse_total_elapsed_sec"],
            },
        }
        first_backend = False
    return {
        "dim": int(dim),
        "backend_order": order,
        "backend_runs": backend_runs,
    }


def main() -> None:
    args = _parse_args()
    if not _cuda_available():
        raise RuntimeError(
            "This gallery requires the CUDA runtime (`nix develop .#cuda`)."
        )

    mesh_family_3d, geometry_version_3d, generator_revision_3d = (
        resolve_3d_mesh_contract(
            acceleration_profile=args.gpu_acceleration_profile,
        )
    )

    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    workers_dir = output_dir / "_workers"
    _ensure_clean_dir(figures_dir)
    _ensure_clean_dir(data_dir)
    _ensure_clean_dir(workers_dir)

    dim2, dim2_cpu_summary, dim2_gpu_summary, dim2_bundles = _run_correctness_pair(
        args,
        dim=2,
        output_dir=workers_dir / "correctness_2d",
    )
    dim3, dim3_cpu_summary, dim3_gpu_summary, dim3_bundles = _run_correctness_pair(
        args,
        dim=3,
        output_dir=workers_dir / "correctness_3d",
    )
    fairness_2d_cpu_first = _run_fairness_order(
        args,
        dim=2,
        output_dir=workers_dir / "fairness_2d_cpu_first",
        order_label="cpu-first",
    )
    fairness_2d_gpu_first = _run_fairness_order(
        args,
        dim=2,
        output_dir=workers_dir / "fairness_2d_gpu_first",
        order_label="gpu-first",
    )
    fairness_3d_cpu_first = _run_fairness_order(
        args,
        dim=3,
        output_dir=workers_dir / "fairness_3d_cpu_first",
        order_label="cpu-first",
    )
    fairness_3d_gpu_first = _run_fairness_order(
        args,
        dim=3,
        output_dir=workers_dir / "fairness_3d_gpu_first",
        order_label="gpu-first",
    )
    _ensure_plot_stack()

    x2, y2, truth2 = _sample_2d_field(
        coords=np.asarray(dim2_bundles["cpu"]["coords"], dtype=np.float64),
        values=np.asarray(dim2_bundles["cpu"]["truth_values"], dtype=np.float64)
        - BACKGROUND_CONDUCTIVITY,
        radius=float(dim2["mesh_radius"]),
        resolution=int(args.slice_resolution),
    )
    _, _, cpu2 = _sample_2d_field(
        coords=np.asarray(dim2_bundles["cpu"]["coords"], dtype=np.float64),
        values=np.asarray(dim2_bundles["cpu"]["reconstruction"], dtype=np.float64)
        - BACKGROUND_CONDUCTIVITY,
        radius=float(dim2["mesh_radius"]),
        resolution=int(args.slice_resolution),
    )
    _, _, gpu2 = _sample_2d_field(
        coords=np.asarray(dim2_bundles["gpu"]["coords"], dtype=np.float64),
        values=np.asarray(dim2_bundles["gpu"]["reconstruction"], dtype=np.float64)
        - BACKGROUND_CONDUCTIVITY,
        radius=float(dim2["mesh_radius"]),
        resolution=int(args.slice_resolution),
    )
    profile_x, profile_truth = _sample_2d_profile(x_grid=x2, y_grid=y2, field=truth2)
    _, profile_cpu = _sample_2d_profile(x_grid=x2, y_grid=y2, field=cpu2)
    _, profile_gpu = _sample_2d_profile(x_grid=x2, y_grid=y2, field=gpu2)
    write_output_bundle(
        data_dir / "2d_fields.h5",
        {
            "x_grid": x2,
            "y_grid": y2,
            "truth": truth2,
            "cpu": cpu2,
            "gpu": gpu2,
        },
        {"package_role": "gallery_2d_fields"},
        schema=GALLERY_ARRAYS_SCHEMA,
    )

    coords3 = np.asarray(dim3_bundles["cpu"]["coords"], dtype=np.float64)
    z_half_height = 0.5 * float(np.max(coords3[:, 2]) - np.min(coords3[:, 2]))
    slice_specs = [
        ("Axial z=+0.22", "axial", 0.22),
        ("Axial z=-0.22", "axial", -0.22),
        ("Coronal y=0", "coronal", 0.0),
    ]
    truth_slices: list[np.ndarray] = []
    cpu_slices: list[np.ndarray] = []
    gpu_slices: list[np.ndarray] = []
    slice_payload: dict[str, np.ndarray] = {}
    for name, plane, value_norm in slice_specs:
        _, _, truth_slice = _sample_3d_slice(
            coords=coords3,
            values=np.asarray(dim3_bundles["cpu"]["truth_values"], dtype=np.float64)
            - BACKGROUND_CONDUCTIVITY,
            radius=float(dim3["mesh_radius"]),
            z_half_height=z_half_height,
            plane=plane,
            plane_value_norm=value_norm,
            resolution=int(args.slice_resolution),
        )
        _, _, cpu_slice = _sample_3d_slice(
            coords=coords3,
            values=np.asarray(dim3_bundles["cpu"]["reconstruction"], dtype=np.float64)
            - BACKGROUND_CONDUCTIVITY,
            radius=float(dim3["mesh_radius"]),
            z_half_height=z_half_height,
            plane=plane,
            plane_value_norm=value_norm,
            resolution=int(args.slice_resolution),
        )
        _, _, gpu_slice = _sample_3d_slice(
            coords=coords3,
            values=np.asarray(dim3_bundles["gpu"]["reconstruction"], dtype=np.float64)
            - BACKGROUND_CONDUCTIVITY,
            radius=float(dim3["mesh_radius"]),
            z_half_height=z_half_height,
            plane=plane,
            plane_value_norm=value_norm,
            resolution=int(args.slice_resolution),
        )
        truth_slices.append(truth_slice)
        cpu_slices.append(cpu_slice)
        gpu_slices.append(gpu_slice)
        slice_key = name.lower().replace(" ", "_").replace("=", "")
        slice_payload[f"{slice_key}_truth"] = truth_slice
        slice_payload[f"{slice_key}_cpu"] = cpu_slice
        slice_payload[f"{slice_key}_gpu"] = gpu_slice
    write_output_bundle(
        data_dir / "3d_slices.h5",
        slice_payload,
        {"package_role": "gallery_3d_slices"},
        schema=GALLERY_ARRAYS_SCHEMA,
    )

    copied_case_paths = {
        "2d_cpu": data_dir / "2d_cpu_case.h5",
        "2d_gpu": data_dir / "2d_gpu_case.h5",
        "3d_cpu": data_dir / "3d_cpu_case.h5",
        "3d_gpu": data_dir / "3d_gpu_case.h5",
    }
    shutil.copy2(
        (workers_dir / "correctness_2d") / dim2_cpu_summary["case"]["data_path"],
        copied_case_paths["2d_cpu"],
    )
    shutil.copy2(
        (workers_dir / "correctness_2d") / dim2_gpu_summary["case"]["data_path"],
        copied_case_paths["2d_gpu"],
    )
    shutil.copy2(
        (workers_dir / "correctness_3d") / dim3_cpu_summary["case"]["data_path"],
        copied_case_paths["3d_cpu"],
    )
    shutil.copy2(
        (workers_dir / "correctness_3d") / dim3_gpu_summary["case"]["data_path"],
        copied_case_paths["3d_gpu"],
    )
    dim2["cpu_case"]["data_path"] = str(
        copied_case_paths["2d_cpu"].relative_to(output_dir)
    )
    dim2["gpu_case"]["data_path"] = str(
        copied_case_paths["2d_gpu"].relative_to(output_dir)
    )
    dim3["cpu_case"]["data_path"] = str(
        copied_case_paths["3d_cpu"].relative_to(output_dir)
    )
    dim3["gpu_case"]["data_path"] = str(
        copied_case_paths["3d_gpu"].relative_to(output_dir)
    )

    figure_2d = figures_dir / "2d_overview.png"
    figure_3d = figures_dir / "3d_overview.png"
    _render_2d_overview(
        output_path=figure_2d,
        truth_grid=truth2,
        cpu_grid=cpu2,
        gpu_grid=gpu2,
        profile_x=profile_x,
        profile_truth=profile_truth,
        profile_cpu=profile_cpu,
        profile_gpu=profile_gpu,
        consistency=dim2["consistency"],
        cpu_metrics=dim2["cpu_truth_metrics"],
        gpu_metrics=dim2["gpu_truth_metrics"],
    )
    _render_3d_overview(
        output_path=figure_3d,
        truth_slices=truth_slices,
        cpu_slices=cpu_slices,
        gpu_slices=gpu_slices,
        slice_titles=[spec[0] for spec in slice_specs],
        consistency=dim3["consistency"],
    )

    case_rows = [
        _case_table_row("2D", dim2["cpu_case"]),
        _case_table_row("2D", dim2["gpu_case"]),
        _case_table_row("3D", dim3["cpu_case"]),
        _case_table_row("3D", dim3["gpu_case"]),
    ]
    consistency_rows = [
        _consistency_table_row(
            "2D",
            dim2["consistency"],
            cpu_case=dim2["cpu_case"],
            gpu_case=dim2["gpu_case"],
        ),
        _consistency_table_row(
            "3D",
            dim3["consistency"],
            cpu_case=dim3["cpu_case"],
            gpu_case=dim3["gpu_case"],
        ),
    ]
    fairness_order_rows = [
        _fairness_order_row("2D", "CPU->GPU", fairness_2d_cpu_first, report_only=True),
        _fairness_order_row("2D", "GPU->CPU", fairness_2d_gpu_first, report_only=True),
        _fairness_order_row("3D", "CPU->GPU", fairness_3d_cpu_first, report_only=False),
        _fairness_order_row("3D", "GPU->CPU", fairness_3d_gpu_first, report_only=False),
    ]
    fairness_backend_rows = [
        _fairness_backend_row(
            "2D", "cpu", fairness_2d_cpu_first, fairness_2d_gpu_first, report_only=True
        ),
        _fairness_backend_row(
            "2D", "gpu", fairness_2d_cpu_first, fairness_2d_gpu_first, report_only=True
        ),
        _fairness_backend_row(
            "3D", "cpu", fairness_3d_cpu_first, fairness_3d_gpu_first, report_only=False
        ),
        _fairness_backend_row(
            "3D", "gpu", fairness_3d_cpu_first, fairness_3d_gpu_first, report_only=False
        ),
    ]
    correctness_pass = bool(all(row["passed"] for row in consistency_rows))
    fairness_pass = bool(
        all(row["passed"] for row in fairness_backend_rows if not row["report_only"])
    )
    all_passed = bool(correctness_pass and fairness_pass)

    payload = {
        "title": args.report_title,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_elec": int(args.n_elec),
            "mesh_size_2d": float(args.mesh_size_2d),
            "radius_2d": float(args.radius_2d),
            "radius_3d": float(args.radius_3d),
            "height_3d": float(args.height_3d),
            "refinement_3d": int(args.refinement_3d),
            "gpu_acceleration_profile": str(args.gpu_acceleration_profile),
            "mesh_family_3d": str(mesh_family_3d),
            "geometry_version_3d": str(geometry_version_3d),
            "generator_revision_3d": str(generator_revision_3d),
            "electrode_height_ratio": float(args.electrode_height_ratio),
            "electrode_coverage": float(args.electrode_coverage),
            "contact_impedance": float(args.contact_impedance),
            "max_iterations": int(args.max_iterations),
            "background_conductivity": BACKGROUND_CONDUCTIVITY,
            "anomalies_2d": [asdict(item) for item in ANOMALIES_2D],
            "anomalies_3d": [asdict(item) for item in ANOMALIES_3D],
        },
        "figures": {
            "2d_overview": str(figure_2d.relative_to(output_dir)),
            "3d_overview": str(figure_3d.relative_to(output_dir)),
        },
        "correctness": {
            "cases": case_rows,
            "consistency": consistency_rows,
            "passed": correctness_pass,
        },
        "fairness_cross_order": {
            "order_runs": fairness_order_rows,
            "backend_drift": fairness_backend_rows,
            "passed": fairness_pass,
        },
        "cases": case_rows,
        "consistency": consistency_rows,
        "all_passed": all_passed,
    }
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    metrics_path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    _write_report(
        output_path=report_path,
        title=args.report_title,
        figures=payload["figures"],
        config=payload["config"],
        case_rows=case_rows,
        consistency_rows=consistency_rows,
        fairness_order_rows=fairness_order_rows,
        fairness_backend_rows=fairness_backend_rows,
        all_passed=all_passed,
    )

    print(json.dumps(_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
