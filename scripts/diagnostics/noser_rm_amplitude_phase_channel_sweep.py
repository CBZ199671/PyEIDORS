#!/usr/bin/env python3
"""Sweep independent amplitude and phase errors with NOSER-RM channel views."""

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
from eit_app.ui.complex_channels import (  # noqa: E402
    COMPOSITE_CHANNEL,
    IMAG_CHANNEL,
    MAGNITUDE_CHANNEL,
    PHASE_CHANNEL,
    REAL_CHANNEL,
    channel_values,
)
from pyeidors.runtime_paths import pyeidors_cache_path, pyeidors_output_path  # noqa: E402
from scripts.diagnostics.noser_rm_phase_offset_circle_metrics import (  # noqa: E402
    DEFAULT_ANGLES_DEG,
    DEFAULT_TOLERANCE_FRACTIONS,
    DIMENSIONLESS_TOLERANCE_METRICS,
    add_circle_overlays,
    build_noser_rm_metadata,
    cell_centers,
    circular_metrics,
    parse_float_list,
    polygon_cell_areas,
    run_noser_rm,
    tolerance_changes,
    tolerance_column,
    tolerance_summary,
    write_csv,
)
from scripts.diagnostics.phase_offset_complex_reconstruction_sweep import (  # noqa: E402
    build_forward_request,
    json_ready,
    parse_complex_arg,
    register_times_new_roman,
    robust_limits,
    tripcolor_cell_data,
)


DEFAULT_AMPLITUDE_MIN = 0.5
DEFAULT_AMPLITUDE_MAX = 1.5
DEFAULT_AMPLITUDE_STEP = 0.1
CHANNEL_ORDER = (
    MAGNITUDE_CHANNEL,
    PHASE_CHANNEL,
    REAL_CHANNEL,
    IMAG_CHANNEL,
    COMPOSITE_CHANNEL,
)
CHANNEL_LABELS = {
    MAGNITUDE_CHANNEL: "Magnitude |sigma|",
    PHASE_CHANNEL: "Phase angle(sigma)",
    REAL_CHANNEL: "Real Re(sigma)",
    IMAG_CHANNEL: "Imag Im(sigma)",
    COMPOSITE_CHANNEL: "Composite GUI scalar",
}
CHANNEL_UNITS = {
    MAGNITUDE_CHANNEL: "S/m",
    PHASE_CHANNEL: "deg",
    REAL_CHANNEL: "S/m",
    IMAG_CHANNEL: "S/m",
    COMPOSITE_CHANNEL: "a.u.",
}
CHANNEL_CMAPS = {
    MAGNITUDE_CHANNEL: "viridis",
    PHASE_CHANNEL: "twilight_shifted",
    REAL_CHANNEL: "viridis",
    IMAG_CHANNEL: "magma",
    COMPOSITE_CHANNEL: "coolwarm",
}


@dataclass(frozen=True)
class NoserRmAmpPhaseConfig:
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
    amplitude_factors: tuple[float, ...]
    threshold_fraction: float
    tolerance_fractions: tuple[float, ...]
    rm_artifact_dir: str


def _unique_floats(values: list[float]) -> tuple[float, ...]:
    out: list[float] = []
    seen: set[float] = set()
    for value in values:
        key = round(float(value), 12)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(value))
    return tuple(out)


def parse_angles(raw: str | None) -> tuple[float, ...]:
    return parse_float_list(raw, default=DEFAULT_ANGLES_DEG)


def default_amplitude_factors(
    *,
    min_factor: float,
    max_factor: float,
    step: float,
) -> tuple[float, ...]:
    if step <= 0.0:
        raise ValueError("amplitude step must be positive.")
    if max_factor < min_factor:
        raise ValueError("amplitude max must be >= min.")
    count = int(round((max_factor - min_factor) / step))
    values = [min_factor + idx * step for idx in range(count + 1)]
    if not any(abs(value - 1.0) <= 1.0e-10 for value in values):
        values.append(1.0)
    return tuple(sorted(_unique_floats([round(value, 10) for value in values])))


def parse_amplitude_factors(args: argparse.Namespace) -> tuple[float, ...]:
    raw = getattr(args, "amplitude_factors", None)
    if raw is not None and str(raw).strip():
        factors = parse_float_list(raw, default=(1.0,))
    else:
        factors = default_amplitude_factors(
            min_factor=float(args.amplitude_min),
            max_factor=float(args.amplitude_max),
            step=float(args.amplitude_step),
        )
    if any(float(value) < 0.0 for value in factors):
        raise ValueError("amplitude factors must be non-negative.")
    return factors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=pyeidors_output_path(
            "diagnostics",
            "noser_rm_amplitude_phase_channel_sweep",
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
    parser.add_argument(
        "--amplitude-factors",
        default=None,
        help="Comma-separated amplitude scale factors. Overrides min/max/step.",
    )
    parser.add_argument("--amplitude-min", type=float, default=DEFAULT_AMPLITUDE_MIN)
    parser.add_argument("--amplitude-max", type=float, default=DEFAULT_AMPLITUDE_MAX)
    parser.add_argument("--amplitude-step", type=float, default=DEFAULT_AMPLITUDE_STEP)
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
            "noser_rm_amplitude_phase_channel_sweep",
            "gui_rm",
        ),
    )
    return parser.parse_args()


def channel_array(sigma: np.ndarray, channel: str) -> np.ndarray:
    return np.asarray(channel_values(sigma, channel), dtype=np.float64).reshape(-1)


def wrapped_phase_delta(current: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    current_rad = np.radians(current)
    baseline_rad = np.radians(baseline)
    return np.degrees(np.angle(np.exp(1j * (current_rad - baseline_rad))))


def channel_delta(
    current: np.ndarray,
    baseline: np.ndarray,
    *,
    channel: str,
) -> np.ndarray:
    if channel == PHASE_CHANNEL:
        return wrapped_phase_delta(current, baseline)
    return current - baseline


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    if int(np.count_nonzero(finite)) < 2:
        return float("nan")
    left_f = left[finite]
    right_f = right[finite]
    if float(np.std(left_f)) <= 1.0e-14 or float(np.std(right_f)) <= 1.0e-14:
        return float("nan")
    return float(np.corrcoef(left_f, right_f)[0, 1])


def channel_metrics(
    *,
    sweep_type: str,
    parameter: float,
    channel: str,
    sigma: np.ndarray,
    baseline_sigma: np.ndarray,
    truth_sigma: np.ndarray,
    truth_mask: np.ndarray,
) -> dict[str, Any]:
    values = channel_array(sigma, channel)
    baseline = channel_array(baseline_sigma, channel)
    truth = channel_array(truth_sigma, channel)
    delta = channel_delta(values, baseline, channel=channel)
    truth_delta = channel_delta(values, truth, channel=channel)
    target_mask = np.asarray(truth_mask, dtype=bool).reshape(-1)
    background_mask = ~target_mask
    norm_baseline = float(np.linalg.norm(baseline))
    norm_truth = float(np.linalg.norm(truth))
    return {
        "sweep_type": sweep_type,
        "parameter": float(parameter),
        "channel": channel,
        "channel_label": CHANNEL_LABELS[channel],
        "corr_vs_baseline": safe_corr(values, baseline),
        "corr_vs_truth": safe_corr(values, truth),
        "relative_l2_vs_baseline": float(
            np.linalg.norm(delta) / max(norm_baseline, 1.0e-30)
        ),
        "relative_l2_vs_truth": float(
            np.linalg.norm(truth_delta) / max(norm_truth, 1.0e-30)
        ),
        "mean_abs_delta_vs_baseline": float(np.mean(np.abs(delta))),
        "max_abs_delta_vs_baseline": float(np.max(np.abs(delta))),
        "target_mean": float(np.mean(values[target_mask])),
        "background_mean": float(np.mean(values[background_mask])),
        "target_minus_background_mean": float(
            np.mean(values[target_mask]) - np.mean(values[background_mask])
        ),
    }


def channel_limits(
    *,
    channel: str,
    truth_sigma: np.ndarray,
    baseline_sigma: np.ndarray,
    phase_sigmas: list[np.ndarray],
    amplitude_sigmas: list[np.ndarray],
) -> tuple[float, float]:
    arrays = [
        channel_array(truth_sigma, channel),
        channel_array(baseline_sigma, channel),
        *[channel_array(sigma, channel) for sigma in phase_sigmas],
        *[channel_array(sigma, channel) for sigma in amplitude_sigmas],
    ]
    if channel == PHASE_CHANNEL:
        return robust_limits(arrays, fallback=(-180.0, 180.0))
    if channel == COMPOSITE_CHANNEL:
        lo, hi = robust_limits(arrays, fallback=(-1.0, 1.0))
        span = max(abs(lo), abs(hi), 1.0e-12)
        return -span, span
    if channel == MAGNITUDE_CHANNEL:
        lo, hi = robust_limits(arrays, fallback=(0.0, 1.0))
        return max(0.0, lo), hi
    return robust_limits(arrays, fallback=(0.0, 1.0))


def plot_channel_grid(
    *,
    output_path: Path,
    title: str,
    cfg: NoserRmAmpPhaseConfig,
    coords: np.ndarray,
    cells: np.ndarray,
    parameters: tuple[float, ...],
    panel_prefix: str,
    channel: str,
    truth_values: np.ndarray,
    values_by_parameter: list[np.ndarray],
    vmin: float,
    vmax: float,
) -> None:
    n_items = 1 + len(values_by_parameter)
    n_cols = 5
    n_rows = int(math.ceil(n_items / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 3.0 * n_rows),
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes).reshape(n_rows, n_cols)
    artist = None
    for idx, ax in enumerate(axes_arr.flat):
        if idx >= n_items:
            ax.axis("off")
            continue
        if idx == 0:
            values = truth_values
            panel_title = "Truth"
        else:
            values = values_by_parameter[idx - 1]
            value = parameters[idx - 1]
            panel_title = f"{panel_prefix}{value:g}"
        artist = tripcolor_cell_data(
            ax,
            coords=coords,
            cells=cells,
            values=values,
            title=panel_title,
            cmap=CHANNEL_CMAPS[channel],
            vmin=vmin,
            vmax=vmax,
        )
        add_circle_overlays(ax, cfg)
    if artist is not None:
        fig.colorbar(
            artist,
            ax=axes_arr.ravel().tolist(),
            shrink=0.72,
            label=CHANNEL_UNITS[channel],
        )
    fig.suptitle(title, fontsize=15)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_channel_metric_curves(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    sweep_type: str,
    x_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    selected = [row for row in rows if row["sweep_type"] == sweep_type]
    for channel in CHANNEL_ORDER:
        channel_rows = [row for row in selected if row["channel"] == channel]
        channel_rows.sort(key=lambda row: float(row["parameter"]))
        x = [float(row["parameter"]) for row in channel_rows]
        y = [float(row["relative_l2_vs_baseline"]) for row in channel_rows]
        ax.plot(x, y, marker="o", linewidth=1.3, markersize=3.5, label=channel)
    ax.set_xlabel(x_label)
    ax.set_ylabel("relative L2 vs baseline")
    ax.set_title(f"{sweep_type} perturbation: channel drift vs baseline")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_voltage_perturbation_curves(
    *,
    output_path: Path,
    target_voltage: np.ndarray,
    phase_angles: tuple[float, ...],
    phase_voltages: list[np.ndarray],
    amplitude_factors: tuple[float, ...],
    amplitude_voltages: list[np.ndarray],
) -> None:
    base = np.asarray(target_voltage, dtype=np.complex128).reshape(-1)
    phase_rel_l2 = [
        float(np.linalg.norm(voltage - base) / max(np.linalg.norm(base), 1.0e-30))
        for voltage in phase_voltages
    ]
    amp_rel_l2 = [
        float(np.linalg.norm(voltage - base) / max(np.linalg.norm(base), 1.0e-30))
        for voltage in amplitude_voltages
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    axes[0].plot(phase_angles, phase_rel_l2, marker="o", linewidth=1.3, markersize=3.5)
    axes[0].set_xlabel("phase offset (deg)")
    axes[0].set_ylabel("relative L2 of voltage")
    axes[0].set_title("Phase-only voltage perturbation")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        amplitude_factors,
        amp_rel_l2,
        marker="o",
        linewidth=1.3,
        markersize=3.5,
    )
    axes[1].set_xlabel("amplitude factor")
    axes[1].set_ylabel("relative L2 of voltage")
    axes[1].set_title("Amplitude-only voltage perturbation")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Applied target-voltage perturbation size", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep() -> dict[str, Any]:
    args = parse_args()
    cfg = NoserRmAmpPhaseConfig(
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
        amplitude_factors=parse_amplitude_factors(args),
        threshold_fraction=float(args.threshold_fraction),
        tolerance_fractions=parse_float_list(
            args.tolerance_fractions,
            default=DEFAULT_TOLERANCE_FRACTIONS,
        ),
        rm_artifact_dir=str(args.rm_artifact_dir),
    )
    output_dir = Path(cfg.output_dir)
    phase_dir = output_dir / "phase"
    amplitude_dir = output_dir / "amplitude"
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir(parents=True, exist_ok=True)
    amplitude_dir.mkdir(parents=True, exist_ok=True)
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
    metadata["request_source"] = "diagnostic_noser_rm_amplitude_phase_channel_sweep"
    metadata["rm_artifact_dir"] = cfg.rm_artifact_dir

    print("Running baseline NOSER-RM reconstruction...", flush=True)
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

    phase_sigmas: list[np.ndarray] = []
    phase_voltages: list[np.ndarray] = []
    phase_circle_rows: list[dict[str, Any]] = []
    print("Sweeping phase-only perturbations...", flush=True)
    for idx, angle_deg in enumerate(cfg.angles_deg):
        angle_rad = math.radians(float(angle_deg))
        shifted_voltage = np.abs(target_voltage) * np.exp(
            1j * (np.angle(target_voltage) + angle_rad)
        )
        phase_voltages.append(np.asarray(shifted_voltage, dtype=np.complex128))
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
            "sweep_type": "phase",
            "angle_deg": float(angle_deg),
            **metrics,
            **tolerance_changes(metrics, baseline_metric),
        }
        for tolerance in cfg.tolerance_fractions:
            row[tolerance_column(tolerance)] = bool(
                float(row["max_dimensionless_metric_delta"]) <= float(tolerance)
            )
        phase_circle_rows.append(row)
        phase_sigmas.append(sigma)

    amplitude_sigmas: list[np.ndarray] = []
    amplitude_voltages: list[np.ndarray] = []
    amplitude_circle_rows: list[dict[str, Any]] = []
    print("Sweeping amplitude-only perturbations...", flush=True)
    for idx, factor in enumerate(cfg.amplitude_factors):
        shifted_voltage = (
            float(factor)
            * np.abs(target_voltage)
            * np.exp(1j * np.angle(target_voltage))
        )
        amplitude_voltages.append(np.asarray(shifted_voltage, dtype=np.complex128))
        if abs(float(factor) - 1.0) <= 1.0e-14:
            result = baseline_result
        else:
            print(f"  Amplitude x{factor:g}", flush=True)
            result = run_noser_rm(
                reference_voltage=reference_voltage,
                target_voltage=shifted_voltage,
                metadata=metadata,
                cfg=cfg,
                frame_index=10_000 + idx,
            )
        sigma = np.asarray(result.conductivity, dtype=np.complex128)
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
        row = {
            "sweep_type": "amplitude",
            "amplitude_factor": float(factor),
            **metrics,
            **tolerance_changes(metrics, baseline_metric),
        }
        for tolerance in cfg.tolerance_fractions:
            row[tolerance_column(tolerance)] = bool(
                float(row["max_dimensionless_metric_delta"]) <= float(tolerance)
            )
        amplitude_circle_rows.append(row)
        amplitude_sigmas.append(sigma)

    channel_rows: list[dict[str, Any]] = []
    for angle_deg, sigma in zip(cfg.angles_deg, phase_sigmas, strict=True):
        for channel in CHANNEL_ORDER:
            channel_rows.append(
                channel_metrics(
                    sweep_type="phase",
                    parameter=float(angle_deg),
                    channel=channel,
                    sigma=sigma,
                    baseline_sigma=baseline_sigma,
                    truth_sigma=truth_sigma,
                    truth_mask=truth_mask,
                )
            )
    for factor, sigma in zip(cfg.amplitude_factors, amplitude_sigmas, strict=True):
        for channel in CHANNEL_ORDER:
            channel_rows.append(
                channel_metrics(
                    sweep_type="amplitude",
                    parameter=float(factor),
                    channel=channel,
                    sigma=sigma,
                    baseline_sigma=baseline_sigma,
                    truth_sigma=truth_sigma,
                    truth_mask=truth_mask,
                )
            )

    limits = {
        channel: channel_limits(
            channel=channel,
            truth_sigma=truth_sigma,
            baseline_sigma=baseline_sigma,
            phase_sigmas=phase_sigmas,
            amplitude_sigmas=amplitude_sigmas,
        )
        for channel in CHANNEL_ORDER
    }

    print("Rendering channel grids...", flush=True)
    for channel in CHANNEL_ORDER:
        vmin, vmax = limits[channel]
        truth_values = channel_array(truth_sigma, channel)
        plot_channel_grid(
            output_path=phase_dir / f"phase_{channel}_grid.png",
            title=f"NOSER-RM phase-only sweep: {CHANNEL_LABELS[channel]}",
            cfg=cfg,
            coords=coords,
            cells=cells,
            parameters=cfg.angles_deg,
            panel_prefix="+",
            channel=channel,
            truth_values=truth_values,
            values_by_parameter=[
                channel_array(sigma, channel) for sigma in phase_sigmas
            ],
            vmin=vmin,
            vmax=vmax,
        )
        plot_channel_grid(
            output_path=amplitude_dir / f"amplitude_{channel}_grid.png",
            title=f"NOSER-RM amplitude-only sweep: {CHANNEL_LABELS[channel]}",
            cfg=cfg,
            coords=coords,
            cells=cells,
            parameters=cfg.amplitude_factors,
            panel_prefix="x",
            channel=channel,
            truth_values=truth_values,
            values_by_parameter=[
                channel_array(sigma, channel) for sigma in amplitude_sigmas
            ],
            vmin=vmin,
            vmax=vmax,
        )

    plot_channel_metric_curves(
        output_path=output_dir / "phase_channel_relative_l2.png",
        rows=channel_rows,
        sweep_type="phase",
        x_label="phase offset (deg)",
    )
    plot_channel_metric_curves(
        output_path=output_dir / "amplitude_channel_relative_l2.png",
        rows=channel_rows,
        sweep_type="amplitude",
        x_label="amplitude factor",
    )
    plot_voltage_perturbation_curves(
        output_path=output_dir / "applied_voltage_perturbation_size.png",
        target_voltage=target_voltage,
        phase_angles=cfg.angles_deg,
        phase_voltages=phase_voltages,
        amplitude_factors=cfg.amplitude_factors,
        amplitude_voltages=amplitude_voltages,
    )

    phase_circle_path = output_dir / "phase_circle_metrics.csv"
    amplitude_circle_path = output_dir / "amplitude_circle_metrics.csv"
    channel_metrics_path = output_dir / "channel_metrics.csv"
    write_csv(phase_circle_path, phase_circle_rows)
    write_csv(amplitude_circle_path, amplitude_circle_rows)
    write_metrics_csv(channel_metrics_path, channel_rows)
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
        "independent_variables": {
            "phase_only": {
                "reference_voltage": "homogeneous forward voltage, unchanged",
                "target_voltage": "|V_target| * exp(1j*(angle(V_target)+theta))",
                "angles_deg": cfg.angles_deg,
            },
            "amplitude_only": {
                "reference_voltage": "homogeneous forward voltage, unchanged",
                "target_voltage": "factor * |V_target| * exp(1j*angle(V_target))",
                "amplitude_factors": cfg.amplitude_factors,
            },
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
        "channels": {
            "order": list(CHANNEL_ORDER),
            "labels": CHANNEL_LABELS,
            "units": CHANNEL_UNITS,
            "composite_definition": "(angle(sigma)/pi) * abs(sigma), same scalar fallback as GUI complex channel helper",
        },
        "metrics": {
            "circle_strength_channel": "abs(sigma - background)",
            "threshold_rule": (
                "recon active if abs(sigma-background) >= "
                f"{cfg.threshold_fraction:g} * max(true abs(sigma-background))"
            ),
            "tolerance_rule": (
                f"pass if max absolute change among {list(DIMENSIONLESS_TOLERANCE_METRICS)} "
                "relative to baseline <= each configured tolerance"
            ),
            "tolerance_fractions": cfg.tolerance_fractions,
        },
    }
    arrays_path = output_dir / "noser_rm_amplitude_phase_channel_arrays.h5"
    outputs = {
        "phase_dir": str(phase_dir),
        "amplitude_dir": str(amplitude_dir),
        "phase_circle_metrics_csv": str(phase_circle_path),
        "amplitude_circle_metrics_csv": str(amplitude_circle_path),
        "channel_metrics_csv": str(channel_metrics_path),
        "phase_channel_relative_l2": str(output_dir / "phase_channel_relative_l2.png"),
        "amplitude_channel_relative_l2": str(
            output_dir / "amplitude_channel_relative_l2.png"
        ),
        "applied_voltage_perturbation_size": str(
            output_dir / "applied_voltage_perturbation_size.png"
        ),
        "arrays_h5": str(arrays_path),
    }
    summary = {
        "config": json_ready(asdict(cfg)),
        "settings": json_ready(settings),
        "phase_circle_summary": tolerance_summary(
            phase_circle_rows, cfg.tolerance_fractions
        ),
        "amplitude_circle_summary": {
            "levels": [
                {
                    "tolerance_fraction": float(tolerance),
                    "tolerance_percent": float(tolerance * 100.0),
                    "passing_factors": [
                        float(row["amplitude_factor"])
                        for row in amplitude_circle_rows
                        if bool(row[tolerance_column(tolerance)])
                    ],
                }
                for tolerance in cfg.tolerance_fractions
            ],
            "dimensionless_metrics": list(DIMENSIONLESS_TOLERANCE_METRICS),
        },
        "channel_metrics": json_ready(channel_rows),
        "baseline_circle_metrics": json_ready(baseline_metric),
        "forward_result": {
            "n_elements": int(forward.n_elements),
            "n_measurements": int(forward.n_measurements),
            "node_count": int(len(forward.node_coords)),
            "forward_model_config": json_ready(forward.forward_model_config),
        },
        "color_limits": {
            channel: {"vmin": float(values[0]), "vmax": float(values[1])}
            for channel, values in limits.items()
        },
        "outputs": outputs,
    }
    (output_dir / "simulation_settings.json").write_text(
        json.dumps(json_ready(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with h5py.File(arrays_path, "w") as handle:
        for name, values in {
            "phase_angles_deg": np.asarray(cfg.angles_deg, dtype=np.float64),
            "amplitude_factors": np.asarray(cfg.amplitude_factors, dtype=np.float64),
            "phase_sigma": np.stack(phase_sigmas, axis=0),
            "amplitude_sigma": np.stack(amplitude_sigmas, axis=0),
            "truth_sigma": truth_sigma,
            "baseline_sigma": baseline_sigma,
            "truth_mask": truth_mask,
            "node_coords": coords,
            "cell_connectivity": cells,
            "cell_areas": areas,
            "reference_voltage": reference_voltage,
            "target_voltage": target_voltage,
            "phase_voltage": np.stack(phase_voltages, axis=0),
            "amplitude_voltage": np.stack(amplitude_voltages, axis=0),
        }.items():
            array = np.asarray(values)
            kwargs = {"compression": "gzip"} if array.ndim else {}
            handle.create_dataset(name, data=array, **kwargs)
    print(json.dumps(json_ready(outputs), ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> int:
    run_sweep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
