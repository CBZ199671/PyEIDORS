#!/usr/bin/env python3
"""Render a full 3D reconstruction overview for cylindrical CEM test cases.

This script supports both difference and absolute inverse workflows using the
EIDORS-like multi-height zigzag electrode layout. The output emphasizes
volumetric structure and exports numerical diagnostics needed for parity
analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for candidate in (str(REPO_ROOT), str(SCRIPTS_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.font_manager import fontManager
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

from pyeidors import EITSystem
from pyeidors.core_system import DEFAULT_ABSOLUTE_PRESET, DEFAULT_DIFFERENCE_PRESET
from pyeidors.data.difference import build_difference_vector
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.femx import function_get_array
from pyeidors.geometry.mesh3d_generator import (
    DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
    normalize_electrode_level_fractions,
)
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.perf import DEFAULT_ACCELERATION_PROFILE
from pyeidors.utils.numeric_ops import squared_distances_to_point

from common.acceleration_profiles import (
    add_acceleration_profile_argument,
    resolve_3d_mesh_contract,
)
from common.hdf5_outputs import DIAGNOSTICS_ARRAYS_SCHEMA, write_output_bundle

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures_3d_inverse_demo"


def _configure_times_new_roman() -> None:
    candidate_fonts = [
        "/mnt/c/Windows/Fonts/times.ttf",
        "/mnt/c/Windows/Fonts/timesbd.ttf",
        "/mnt/c/Windows/Fonts/timesi.ttf",
        "/mnt/c/Windows/Fonts/timesbi.ttf",
    ]
    for font_path in candidate_fonts:
        path = Path(font_path)
        if path.exists():
            try:
                fontManager.addfont(str(path))
            except Exception:
                pass
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )


def _build_3d_phantom(
    eit_system: EITSystem,
    *,
    base_conductivity: float,
    phantom_conductivity: float,
    center: tuple[float, float, float],
    radius: float,
) -> EITImage:
    image = eit_system.create_homogeneous_image(conductivity=base_conductivity)
    sigma = np.asarray(image.elem_data, dtype=float).copy()
    coords = eit_system.fwd_model.V_sigma.tabulate_dof_coordinates()
    distances2 = squared_distances_to_point(coords, center, ndim=3)
    sigma[distances2 <= float(radius) ** 2] = float(phantom_conductivity)
    return EITImage(elem_data=sigma, fwd_model=eit_system.fwd_model)


def _build_cylinder_wireframe(
    *,
    radius: float,
    height: float,
    z_center: float,
    n_theta: int = 160,
    n_vertical: int = 12,
) -> list[np.ndarray]:
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta)
    z_top = z_center + 0.5 * height
    z_bottom = z_center - 0.5 * height
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    segments: list[np.ndarray] = []
    segments.append(_three_column_points(x, y, z_top))
    segments.append(_three_column_points(x, y, z_bottom))

    vertical_angles = np.linspace(0.0, 2.0 * math.pi, n_vertical, endpoint=False)
    for ang in vertical_angles:
        xv = radius * math.cos(float(ang))
        yv = radius * math.sin(float(ang))
        segments.append(np.array([[xv, yv, z_bottom], [xv, yv, z_top]], dtype=float))
    return segments


def _build_electrode_markers(
    *,
    n_elec: int,
    radius: float,
    height: float,
    z_center: float,
    electrode_level_fractions: tuple[float, ...],
) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, n_elec, endpoint=False)
    z_levels = (
        z_center
        - 0.5 * height
        + height
        * np.asarray(electrode_level_fractions, dtype=float)[
            np.arange(n_elec, dtype=np.int32) % len(electrode_level_fractions)
        ]
    )
    return _three_column_points(
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.asarray(z_levels, dtype=float),
    )


def _three_column_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray | float,
) -> np.ndarray:
    x_arr = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size:
        raise ValueError("x/y point columns must have the same length")
    out = np.empty((x_arr.size, 3), dtype=np.float64)
    out[:, 0] = x_arr
    out[:, 1] = y_arr
    if np.ndim(z_values) == 0:
        out[:, 2] = float(z_values)
    else:
        z_arr = np.asarray(z_values, dtype=np.float64).reshape(-1)
        if z_arr.size != x_arr.size:
            raise ValueError("z point column must match x/y length")
        out[:, 2] = z_arr
    return out


def _choose_threshold(
    values: np.ndarray, *, baseline: float, truth_mode: bool
) -> float:
    vmax = float(np.max(values))
    if truth_mode:
        return baseline + 0.55 * (vmax - baseline)
    return max(
        baseline + 0.60 * (vmax - baseline),
        float(np.percentile(values, 97.5)),
    )


def _parse_level_fractions(text: str | None) -> tuple[float, ...]:
    if text is None or not str(text).strip():
        return tuple(float(v) for v in DEFAULT_ZIGZAG_LEVEL_FRACTIONS)
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    return normalize_electrode_level_fractions(
        values,
        default=DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
    )


def _compute_shape_metrics(
    coords: np.ndarray,
    values: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    value_arr = np.asarray(values, dtype=float).reshape(-1)
    coord_arr = np.asarray(coords, dtype=float)
    mask = np.isfinite(value_arr)
    np.greater_equal(value_arr, float(threshold), out=mask, where=mask)
    selected_count = int(np.count_nonzero(mask))
    if selected_count == 0:
        return {
            "selected_count": 0,
            "threshold": float(threshold),
            "extent_x": 0.0,
            "extent_y": 0.0,
            "extent_z": 0.0,
            "z_to_xy_mean_ratio": float("nan"),
            "xy_aspect_ratio": float("nan"),
        }
    mins = np.empty(3, dtype=float)
    maxs = np.empty(3, dtype=float)
    for axis in range(3):
        column = coord_arr[:, axis]
        mins[axis] = np.min(column, where=mask, initial=np.inf)
        maxs[axis] = np.max(column, where=mask, initial=-np.inf)
    extents = maxs - mins
    xy_extent_min = max(float(min(extents[0], extents[1])), 1e-12)
    xy_extent_mean = max(float(0.5 * (extents[0] + extents[1])), 1e-12)
    return {
        "selected_count": selected_count,
        "threshold": float(threshold),
        "extent_x": float(extents[0]),
        "extent_y": float(extents[1]),
        "extent_z": float(extents[2]),
        "z_to_xy_mean_ratio": float(extents[2] / xy_extent_mean),
        "xy_aspect_ratio": float(max(extents[0], extents[1]) / xy_extent_min),
    }


def _apply_cylindrical_nan_mask_in_place(
    volume: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    *,
    radius: float,
) -> None:
    x2 = np.square(np.asarray(x_centers, dtype=float))
    y2 = np.square(np.asarray(y_centers, dtype=float))
    outside_xy = np.add.outer(x2, y2) > (float(radius) * 0.995) ** 2
    if np.any(outside_xy):
        volume[outside_xy, :] = np.nan


def _pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=float).reshape(-1)
    right_arr = np.asarray(right, dtype=float).reshape(-1)
    if left_arr.size <= 1 or left_arr.size != right_arr.size:
        return float("nan")
    left_centered = np.array(left_arr, dtype=np.float64, copy=True)
    right_centered = np.array(right_arr, dtype=np.float64, copy=True)
    left_centered -= float(np.mean(left_centered))
    right_centered -= float(np.mean(right_centered))
    numerator = float(np.dot(left_centered, right_centered))
    left_norm2 = float(np.dot(left_centered, left_centered))
    right_norm2 = float(np.dot(right_centered, right_centered))
    denominator = math.sqrt(left_norm2 * right_norm2)
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    work = np.array(left, dtype=np.float64, copy=True)
    work -= np.asarray(right, dtype=np.float64)
    np.square(work, out=work)
    return float(math.sqrt(float(np.mean(work))))


def _mean_where(values: np.ndarray, mask: np.ndarray) -> float:
    mask_arr = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(mask_arr))
    if count == 0:
        return float("nan")
    value_arr = np.asarray(values, dtype=float)
    total = np.sum(value_arr, where=mask_arr, initial=0.0, dtype=np.float64)
    return float(total / count)


def _compute_regular_volume_payload(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    radius: float,
    height: float,
    z_center: float,
    resolution: tuple[int, int, int],
    smooth_sigma: float,
) -> dict[str, np.ndarray]:
    nx, ny, nz = resolution
    x_centers = np.linspace(-radius, radius, nx)
    y_centers = np.linspace(-radius, radius, ny)
    z_centers = np.linspace(z_center - 0.5 * height, z_center + 0.5 * height, nz)
    Xc, Yc, Zc = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    volume = griddata(
        coords, values, (Xc, Yc, Zc), method="linear", fill_value=float(np.min(values))
    )
    if smooth_sigma > 0.0:
        volume = gaussian_filter(volume, sigma=smooth_sigma)
    _apply_cylindrical_nan_mask_in_place(
        volume,
        x_centers,
        y_centers,
        radius=float(radius),
    )

    x_edges = np.linspace(-radius, radius, nx + 1)
    y_edges = np.linspace(-radius, radius, ny + 1)
    z_edges = np.linspace(z_center - 0.5 * height, z_center + 0.5 * height, nz + 1)
    return {
        "x_edges": np.asarray(x_edges, dtype=float),
        "y_edges": np.asarray(y_edges, dtype=float),
        "z_edges": np.asarray(z_edges, dtype=float),
        "volume": np.asarray(volume, dtype=float),
    }


def _build_regular_volume(
    *,
    coords: np.ndarray,
    values: np.ndarray,
    radius: float,
    height: float,
    z_center: float,
    resolution: tuple[int, int, int] = (34, 34, 24),
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolution_tuple = tuple(int(v) for v in resolution)
    payload = _compute_regular_volume_payload(
        coords=np.asarray(coords, dtype=float),
        values=np.asarray(values, dtype=float),
        radius=float(radius),
        height=float(height),
        z_center=float(z_center),
        resolution=resolution_tuple,
        smooth_sigma=float(smooth_sigma),
    )
    x_edges = np.asarray(payload["x_edges"], dtype=float)
    y_edges = np.asarray(payload["y_edges"], dtype=float)
    z_edges = np.asarray(payload["z_edges"], dtype=float)
    volume = np.asarray(payload["volume"], dtype=float)
    Xe, Ye, Ze = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    return Xe, Ye, Ze, volume


def _add_voxel_volume(
    ax,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    values: np.ndarray,
    threshold: float,
    cmap,
    norm,
    alpha_surface: float,
) -> None:
    valid = np.isfinite(values)
    mask = valid & (values >= threshold)
    if not np.any(mask):
        return
    color_values = np.array(values, dtype=float, copy=True)
    color_values[~mask] = np.nan
    mapped = cmap(norm(color_values))
    alpha = mapped[..., 3]
    alpha.fill(0.0)
    alpha[mask] = float(alpha_surface)
    ax.voxels(
        X,
        Y,
        Z,
        mask,
        facecolors=mapped,
        edgecolor=None,
        shade=True,
    )


def _style_3d_axes(ax, *, radius: float, height: float, z_center: float) -> None:
    lim_xy = radius * 1.08
    z_half = 0.5 * height * 1.18
    ax.set_xlim(-lim_xy, lim_xy)
    ax.set_ylim(-lim_xy, lim_xy)
    ax.set_zlim(z_center - z_half, z_center + z_half)
    ax.set_box_aspect((1.0, 1.0, height / max(radius * 2.0, 1e-9)))
    ax.view_init(elev=21, azim=-50)
    ax.set_axis_off()


def run_case(
    *,
    output_dir: Path,
    refinement: int,
    max_iterations: int | None,
    radius: float,
    height: float,
    inverse_mode: str,
    difference_mode: str,
    difference_orientation: str,
    electrode_level_fractions: tuple[float, ...],
    difference_preset: str,
    absolute_preset: str,
    hyperparameter: float | None,
    difference_step_size_mode: str | None,
    best_homog_mode: str | None,
    acceleration_profile: str = DEFAULT_ACCELERATION_PROFILE,
    render_plot: bool = True,
    save_data: bool = True,
) -> dict[str, object]:
    n_elec = 16
    base_sigma = 1.0
    target_sigma = 2.0
    z_center = 0.0
    resolved_inverse_mode = str(inverse_mode).strip().lower()
    resolved_difference_preset = str(difference_preset).strip().lower()
    resolved_absolute_preset = str(absolute_preset).strip().lower()
    preset_name = (
        resolved_difference_preset
        if resolved_inverse_mode == "difference"
        else resolved_absolute_preset
    )
    resolved_level_fractions = normalize_electrode_level_fractions(
        electrode_level_fractions,
        default=DEFAULT_ZIGZAG_LEVEL_FRACTIONS,
    )

    wall_time_breakdown = {
        "setup_elapsed_sec": 0.0,
        "solve_elapsed_sec": 0.0,
        "postprocess_elapsed_sec": 0.0,
        "save_elapsed_sec": 0.0,
    }
    mesh_family, geometry_version, generator_revision = resolve_3d_mesh_contract(
        acceleration_profile=acceleration_profile,
    )

    setup_start = perf_counter()
    mesh = load_or_create_mesh(
        mesh_dir=str(REPO_ROOT / "eit_meshes"),
        n_elec=n_elec,
        dimension=3,
        radius=radius,
        height=height,
        refinement=refinement,
        electrode_coverage=0.5,
        electrode_height_ratio=0.2,
        electrode_level_fractions=resolved_level_fractions,
        z_center=z_center,
        mesh_family=mesh_family,
        geometry_version=geometry_version,
        generator_revision=generator_revision,
    )

    pattern_config = PatternConfig(
        n_elec=n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    system = EITSystem(
        n_elec=n_elec,
        pattern_config=pattern_config,
        contact_impedance=np.full(n_elec, 1e-5, dtype=float),
        base_conductivity=base_sigma,
        difference_mode=difference_mode,
        difference_orientation=difference_orientation,
        regularization_type="noser",
        regularization_alpha=1.0,
        hyperparameter=hyperparameter,
        difference_step_size_mode=difference_step_size_mode,
        difference_preset=resolved_difference_preset,
        absolute_preset=resolved_absolute_preset,
        best_homog_mode=best_homog_mode,
        acceleration_profile=str(acceleration_profile),
        linear_backend="scipy",
        performance_mode="safe",
        solver_mode="fast",
        linear_solver="auto",
        jacobian_update_every=2,
        jacobian_reuse_tol=1e-3,
        line_search_mode="fast",
    )
    system.setup(mesh=mesh)

    reference_img = system.create_homogeneous_image(conductivity=base_sigma)
    reference_data = system.forward_solve(reference_img)

    phantom_img = _build_3d_phantom(
        system,
        base_conductivity=base_sigma,
        phantom_conductivity=target_sigma,
        center=(radius * 0.36, -radius * 0.18, height * 0.20),
        radius=radius * 0.22,
    )
    phantom_data = system.forward_solve(phantom_img)

    if system.reconstructor is None:
        raise RuntimeError("Reconstructor was not initialized.")
    system.reconstructor.ensure_regularization_ready()
    system.reconstructor.clip_values = (1e-6, 3.0)
    if max_iterations is not None:
        system.reconstructor.max_iterations = max_iterations
    wall_time_breakdown["setup_elapsed_sec"] = perf_counter() - setup_start

    solve_start = perf_counter()
    if resolved_inverse_mode == "absolute":
        recon = system.inverse_solve(
            data=phantom_data,
            reference_data=None,
            initial_guess=None,
        )
        inverse_target = "eidors_abs_gn_prior"
    else:
        recon = system.inverse_solve(
            data=phantom_data,
            reference_data=reference_data,
            initial_guess=None,
        )
        if preset_name == "eidors_demo3d_tv":
            inverse_target = "eidors_demo_3d_simdata_tv"
        elif preset_name == "sphere_multistep_noser":
            inverse_target = "sphere_multistep_noser"
        else:
            inverse_target = "eidors_3d_difference_one_step_gn_noser"
    wall_time_breakdown["solve_elapsed_sec"] = perf_counter() - solve_start

    postprocess_start = perf_counter()
    truth_sigma = np.asarray(phantom_img.elem_data, dtype=float).copy()
    recon_sigma = function_get_array(recon.conductivity).copy()
    coords = system.fwd_model.V_sigma.tabulate_dof_coordinates()[:, :3]

    cond_rmse = _rmse(recon_sigma, truth_sigma)
    cond_corr = _pearson_correlation(truth_sigma, recon_sigma)
    pred_img = EITImage(elem_data=recon_sigma, fwd_model=system.fwd_model)
    pred_data = system.forward_solve(pred_img)
    vh = np.asarray(reference_data.meas, dtype=float).copy()
    vi = np.asarray(phantom_data.meas, dtype=float).copy()
    pred_vi = np.asarray(pred_data.meas, dtype=float).copy()
    dv_raw = build_difference_vector(
        vi,
        vh,
        mode="raw",
        orientation="target_minus_reference",
    )
    dv_norm = build_difference_vector(
        vi,
        vh,
        mode="normalized",
        orientation="target_minus_reference",
    )
    dv_data_space = build_difference_vector(
        vi,
        vh,
        mode=difference_mode,
        orientation=difference_orientation,
    )
    pred_dv_data_space = build_difference_vector(
        pred_vi,
        vh,
        mode=difference_mode,
        orientation=difference_orientation,
    )
    if resolved_inverse_mode == "absolute":
        measurement_vector = vi
        prediction_vector = pred_vi
    else:
        measurement_vector = dv_data_space
        prediction_vector = pred_dv_data_space
    residual_vector = prediction_vector - measurement_vector
    volt_rmse = _rmse(prediction_vector, measurement_vector)
    residual_l2 = float(np.linalg.norm(residual_vector))
    residual_max = float(np.max(np.abs(residual_vector)))
    truth_threshold = _choose_threshold(
        truth_sigma, baseline=base_sigma, truth_mode=True
    )
    recon_threshold = _choose_threshold(
        recon_sigma, baseline=base_sigma, truth_mode=False
    )
    truth_shape = _compute_shape_metrics(coords, truth_sigma, threshold=truth_threshold)
    recon_shape = _compute_shape_metrics(coords, recon_sigma, threshold=recon_threshold)
    target_mask = truth_sigma > (base_sigma + 0.5 * (target_sigma - base_sigma))
    background_mask = ~target_mask
    target_mean = _mean_where(recon_sigma, target_mask)
    background_mean = _mean_where(recon_sigma, background_mask)
    peak_conductivity = float(np.max(recon_sigma))
    contrast_recovery = float(
        (target_mean - background_mean) / (target_sigma - base_sigma)
    )
    fig = None
    png_path = output_dir / "inverse_3d_overview.png"
    svg_path = output_dir / "inverse_3d_overview.svg"
    if render_plot:
        cmap = plt.get_cmap("turbo")
        norm = mcolors.Normalize(
            vmin=base_sigma, vmax=max(np.max(truth_sigma), np.max(recon_sigma))
        )
        truth_volume_payload = _compute_regular_volume_payload(
            coords=coords,
            values=truth_sigma,
            radius=radius,
            height=height,
            z_center=z_center,
            resolution=(34, 34, 24),
            smooth_sigma=0.6,
        )
        recon_volume_payload = _compute_regular_volume_payload(
            coords=coords,
            values=recon_sigma,
            radius=radius,
            height=height,
            z_center=z_center,
            resolution=(34, 34, 24),
            smooth_sigma=1.0,
        )
        x_edges = np.asarray(truth_volume_payload["x_edges"], dtype=float)
        y_edges = np.asarray(truth_volume_payload["y_edges"], dtype=float)
        z_edges = np.asarray(truth_volume_payload["z_edges"], dtype=float)
        grid_X, grid_Y, grid_Z = np.meshgrid(
            x_edges,
            y_edges,
            z_edges,
            indexing="ij",
        )
        truth_volume = np.asarray(truth_volume_payload["volume"], dtype=float)
        recon_volume = np.asarray(recon_volume_payload["volume"], dtype=float)

        fig = plt.figure(figsize=(11.6, 5.6), facecolor="white")
        ax_truth = fig.add_subplot(1, 2, 1, projection="3d")
        ax_recon = fig.add_subplot(1, 2, 2, projection="3d")

        wire_segments = _build_cylinder_wireframe(
            radius=radius, height=height, z_center=z_center
        )
        wire_color = (0.45, 0.45, 0.45, 0.38)
        for ax in (ax_truth, ax_recon):
            for seg in wire_segments:
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    seg[:, 2],
                    color=wire_color,
                    linewidth=0.9,
                )
            electrodes = _build_electrode_markers(
                n_elec=n_elec,
                radius=radius * 1.002,
                height=height,
                z_center=z_center,
                electrode_level_fractions=resolved_level_fractions,
            )
            ax.scatter(
                electrodes[:, 0],
                electrodes[:, 1],
                electrodes[:, 2],
                s=16,
                c="#2f6f43",
                alpha=0.95,
                depthshade=False,
                linewidths=0.0,
            )

        _add_voxel_volume(
            ax_truth,
            X=grid_X,
            Y=grid_Y,
            Z=grid_Z,
            values=truth_volume,
            threshold=truth_threshold,
            cmap=cmap,
            norm=norm,
            alpha_surface=0.42,
        )
        _add_voxel_volume(
            ax_recon,
            X=grid_X,
            Y=grid_Y,
            Z=grid_Z,
            values=recon_volume,
            threshold=recon_threshold,
            cmap=cmap,
            norm=norm,
            alpha_surface=0.52,
        )

        for ax in (ax_truth, ax_recon):
            _style_3d_axes(ax, radius=radius, height=height, z_center=z_center)

        ax_truth.text2D(
            0.50,
            0.97,
            "Truth",
            transform=ax_truth.transAxes,
            ha="center",
            va="top",
            fontsize=15,
        )
        ax_recon.text2D(
            0.50,
            0.97,
            "Reconstruction",
            transform=ax_recon.transAxes,
            ha="center",
            va="top",
            fontsize=15,
        )
        fig.suptitle(
            f"{resolved_inverse_mode.capitalize()} / {preset_name}",
            fontsize=14,
            y=0.98,
        )
        ax_recon.text2D(
            0.50,
            0.03,
            f"Conductivity RMSE = {cond_rmse:.4f}\nVoltage RMSE = {volt_rmse:.2e}",
            transform=ax_recon.transAxes,
            ha="center",
            va="bottom",
            fontsize=11.5,
        )

        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax_truth, ax_recon], fraction=0.032, pad=0.02)
        cbar.set_label("Conductivity", fontsize=12)
        fig.subplots_adjust(left=0.02, right=0.90, top=0.95, bottom=0.06, wspace=0.02)
    wall_time_breakdown["postprocess_elapsed_sec"] = perf_counter() - postprocess_start

    metrics = {
        "conductivity_rmse": cond_rmse,
        "conductivity_correlation": cond_corr,
        "voltage_rmse": volt_rmse,
        "residual_l2": residual_l2,
        "residual_max": residual_max,
        "refinement": int(refinement),
        "max_iterations": int(recon.iterations),
        "radius": float(radius),
        "height": float(height),
        "inverse_mode": resolved_inverse_mode,
        "difference_mode": str(difference_mode),
        "difference_orientation": str(difference_orientation),
        "electrode_level_fractions": [float(v) for v in resolved_level_fractions],
        "inverse_target": inverse_target,
        "preset_name": preset_name,
        "acceleration_profile": str(acceleration_profile),
        "hyperparameter": recon.diagnostics.get("hyperparameter"),
        "lambda_eff": recon.diagnostics.get("lambda_eff"),
        "step_size": recon.diagnostics.get("difference_step_size", {}).get("value"),
        "difference_step_size": recon.diagnostics.get("difference_step_size", {}),
        "best_homog": recon.diagnostics.get("best_homog", {}),
        "jacobian_background_conductivity": recon.diagnostics.get(
            "jacobian_background_conductivity"
        ),
        "contrast_recovery": contrast_recovery,
        "target_mean": target_mean,
        "background_mean": background_mean,
        "peak_conductivity": peak_conductivity,
        "wall_time_breakdown": {
            key: float(value) for key, value in wall_time_breakdown.items()
        },
        "shape_metrics": {
            "truth": truth_shape,
            "reconstruction": recon_shape,
        },
        "measurement_space": recon.diagnostics.get("measurement_space", {}),
        "backend_info": recon.diagnostics.get("backend_info", {}),
    }

    save_start = perf_counter()
    should_create_output_dir = render_plot or save_data
    if should_create_output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    if render_plot and fig is not None:
        fig.savefig(png_path, dpi=320, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        plt.close(fig)
    if save_data:
        write_output_bundle(
            output_dir / "inverse_3d_overview_data.h5",
            {
                "coords": coords,
                "truth_sigma": truth_sigma,
                "recon_sigma": recon_sigma,
                "vh": vh,
                "vi": vi,
                "dv_raw": dv_raw,
                "dv_norm": dv_norm,
                "dv_measurement_space": dv_data_space,
                "pred_vi": pred_vi,
                "pred_dv_measurement_space": pred_dv_data_space,
                "measurement_vector": measurement_vector,
                "prediction_vector": prediction_vector,
                "residual_vector": residual_vector,
                "target_mask": target_mask,
                "background_mask": background_mask,
            },
            {"package_role": "inverse_3d_overview_data"},
            schema=DIAGNOSTICS_ARRAYS_SCHEMA,
        )
    if should_create_output_dir:
        wall_time_breakdown["save_elapsed_sec"] = perf_counter() - save_start
    metrics["wall_time_breakdown"] = {
        key: float(value) for key, value in wall_time_breakdown.items()
    }
    if save_data:
        (output_dir / "inverse_3d_overview_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a full 3D inverse reconstruction overview"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--radius", type=float, default=0.22)
    parser.add_argument("--height", type=float, default=0.16)
    add_acceleration_profile_argument(
        parser,
        default=DEFAULT_ACCELERATION_PROFILE,
        help_suffix="Only affects this 3D workflow.",
    )
    parser.add_argument(
        "--inverse-mode", choices=["difference", "absolute"], default="difference"
    )
    parser.add_argument(
        "--difference-mode", choices=["raw", "normalized"], default="normalized"
    )
    parser.add_argument(
        "--difference-orientation",
        choices=["target_minus_reference", "reference_minus_target"],
        default="target_minus_reference",
    )
    parser.add_argument(
        "--electrode-level-fractions",
        default="0.25,0.75",
        help="Comma-separated normalized electrode center heights in (0,1) for zigzag 3D electrodes.",
    )
    parser.add_argument(
        "--difference-preset",
        choices=["eidors_one_step_noser", "eidors_demo3d_tv", "sphere_multistep_noser"],
        default=DEFAULT_DIFFERENCE_PRESET,
    )
    parser.add_argument(
        "--absolute-preset",
        choices=["eidors_abs_gn"],
        default=DEFAULT_ABSOLUTE_PRESET,
    )
    parser.add_argument("--hyperparameter", type=float, default=None)
    parser.add_argument(
        "--difference-step-size-mode",
        choices=["off", "optimize", "fixed"],
        default=None,
    )
    parser.add_argument(
        "--best-homog-mode",
        choices=["off", "optimize"],
        default=None,
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip 3D plot construction and figure export; useful for pure runtime diagnostics.",
    )
    parser.add_argument(
        "--no-save-data",
        action="store_true",
        help="Skip JSON/HDF5 summary export; plot files are still written unless --no-plot is also set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_times_new_roman()
    metrics = run_case(
        output_dir=args.output_dir,
        refinement=args.refinement,
        max_iterations=args.max_iterations,
        radius=args.radius,
        height=args.height,
        inverse_mode=args.inverse_mode,
        difference_mode=args.difference_mode,
        difference_orientation=args.difference_orientation,
        electrode_level_fractions=_parse_level_fractions(
            args.electrode_level_fractions
        ),
        difference_preset=args.difference_preset,
        absolute_preset=args.absolute_preset,
        hyperparameter=args.hyperparameter,
        difference_step_size_mode=args.difference_step_size_mode,
        best_homog_mode=args.best_homog_mode,
        acceleration_profile=args.acceleration_profile,
        render_plot=not args.no_plot,
        save_data=not args.no_save_data,
    )
    print(json.dumps(metrics, indent=2))
    if not args.no_plot:
        print(f"Saved figure to: {args.output_dir / 'inverse_3d_overview.png'}")
        print(f"Saved figure to: {args.output_dir / 'inverse_3d_overview.svg'}")
    if not args.no_save_data:
        print(
            f"Saved metrics to: {args.output_dir / 'inverse_3d_overview_metrics.json'}"
        )
        print(f"Saved data to: {args.output_dir / 'inverse_3d_overview_data.h5'}")


if __name__ == "__main__":
    main()
