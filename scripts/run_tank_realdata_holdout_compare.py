#!/usr/bin/env python3
"""Compare native 208, raw 160, and fitted 208 on two-frame tank real data."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import matplotlib as mpl

mpl.rcParams.update(
    {
        "axes.unicode_minus": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
    }
)
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import diags

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
SCRIPTS_PATH = REPO_ROOT / "scripts"
for path in [SRC_PATH, SCRIPTS_PATH]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.hdf5_outputs import GALLERY_ARRAYS_SCHEMA, write_output_bundle
from common.io_utils import align_frames_polarity, load_csv_measurements
from common.array_metrics import pearson_correlation
from pyeidors.data.holdout_point_audit import build_holdout_point_audit
from pyeidors.data.structures import EITImage, PatternConfig
from pyeidors.forward.eit_forward_model import EITForwardModel
from pyeidors.geometry.optimized_mesh_generator import load_or_create_mesh
from pyeidors.inverse.jacobian.adjoint_jacobian import EidorsJacobianAdapter
from pyeidors.runtime_paths import pyeidors_cache_path
from pyeidors.utils.numeric_ops import all_finite_values

_CHUNK_ITEMS = 1_048_576


SUMMARY_FIELDS = [
    "recon_method",
    "n_inverse_points",
    "fit_holdout_rmse",
    "fit_diff_rmse",
    "pred_diff_rmse",
    "pred_diff_mae",
    "pred_diff_corr",
    "delta_sigma_min",
    "delta_sigma_max",
    "delta_sigma_l2",
    "delta_vs_full208_l2",
]


@dataclass(frozen=True)
class VariantResult:
    name: str
    n_inverse_points: int
    delta_sigma: np.ndarray
    sigma_est: np.ndarray
    pred_vi: np.ndarray
    pred_diff: np.ndarray
    fit_diff: np.ndarray
    fit_holdout_rmse: float
    fit_diff_rmse: float
    pred_diff_rmse: float
    pred_diff_mae: float
    pred_diff_corr: float


def _frame_indices_from_audit() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, summary = build_holdout_point_audit(n_elec=16, holdout="far3")
    if summary.kept_208_count != 208 or summary.fit_train_160_count != 160:
        raise RuntimeError("expected 208 kept points and 160 train points")
    kept_rows = [row for row in rows if row.point_status != "drive_removed"]
    kept_rows.sort(key=lambda row: (row.stim_index, int(row.frame_index_13)))
    train: list[int] = []
    holdout: list[int] = []
    for idx, row in enumerate(kept_rows):
        if row.point_status == "fit_train_160":
            train.append(idx)
        elif row.point_status == "holdout_far3":
            holdout.append(idx)
    all_indices = np.arange(summary.kept_208_count, dtype=np.int64)
    return (
        all_indices,
        np.array(train, dtype=np.int64),
        np.array(holdout, dtype=np.int64),
    )


def _fit_values(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_holdout: np.ndarray,
    method: str,
) -> np.ndarray:
    if method in {"poly2", "poly3"}:
        degree = 2 if method == "poly2" else 3
        coeffs = np.polyfit(x_train, y_train, deg=degree)
        return np.polyval(coeffs, x_holdout)
    spline = CubicSpline(x_train, y_train, bc_type="natural", extrapolate=False)
    predicted = np.asarray(spline(x_holdout), dtype=float)
    if not all_finite_values(predicted):
        raise FloatingPointError("spline prediction produced non-finite values")
    return predicted


def _fit_frame_vector(values: np.ndarray, method: str) -> np.ndarray:
    fitted = np.asarray(values, dtype=float).copy()
    for stim in range(16):
        start = stim * 13
        stop = start + 13
        frame = fitted[start:stop].copy()
        x_all = np.arange(13, dtype=float)
        holdout_mask = np.zeros(13, dtype=bool)
        holdout_mask[[5, 6, 7]] = True
        train_mask = ~holdout_mask
        frame[holdout_mask] = _fit_values(
            x_train=x_all[train_mask],
            y_train=frame[train_mask],
            x_holdout=x_all[holdout_mask],
            method=method,
        )
        fitted[start:stop] = frame
    return fitted


def _fit_jacobian_rows(jacobian: np.ndarray, method: str) -> np.ndarray:
    fitted = np.asarray(jacobian, dtype=float).copy()
    for col_idx in range(fitted.shape[1]):
        fitted[:, col_idx] = _fit_frame_vector(fitted[:, col_idx], method)
    return fitted


def _solve_delta(jacobian: np.ndarray, rhs: np.ndarray, lam: float) -> np.ndarray:
    jac = np.asarray(jacobian, dtype=float)
    diff = np.asarray(rhs, dtype=float)
    diag_entries = np.sum(jac * jac, axis=0)
    adaptive_floor = float(np.max(diag_entries)) * 1e-6
    diag_entries = np.maximum(diag_entries, max(adaptive_floor, 1e-100))
    reg = diags(diag_entries**0.5, offsets=0, format="csr")
    lhs = jac.T @ jac + float(lam) * reg
    rhs_vec = jac.T @ diff
    return np.linalg.solve(np.asarray(lhs, dtype=float), rhs_vec)


def _rmse_at_indices(values: np.ndarray, indices: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    idx = np.asarray(indices, dtype=np.intp).reshape(-1)
    if idx.size == 0:
        return math.nan
    chunk_items = min(int(idx.size), _CHUNK_ITEMS)
    work = np.empty(chunk_items, dtype=np.float64)
    total = 0.0
    for start in range(0, int(idx.size), chunk_items):
        stop = min(start + chunk_items, int(idx.size))
        count = stop - start
        target = work[:count]
        np.take(arr, idx[start:stop], out=target)
        np.square(target, out=target)
        total += float(np.sum(target))
    return float(math.sqrt(total / float(idx.size)))


def _max_abs_value(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return 0.0
    chunk_items = min(int(arr.size), _CHUNK_ITEMS)
    work = np.empty(chunk_items, dtype=np.float64)
    current = 0.0
    for start in range(0, int(arr.size), chunk_items):
        stop = min(start + chunk_items, int(arr.size))
        count = stop - start
        target = work[:count]
        np.abs(arr[start:stop], out=target)
        current = max(current, float(np.max(target)))
    return current


def _stack_two_frames(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_arr = np.asarray(first)
    second_arr = np.asarray(second)
    if first_arr.shape != second_arr.shape:
        raise ValueError("frame shapes must match")
    out = np.empty((2, first_arr.size), dtype=np.result_type(first_arr, second_arr))
    out[0, :] = first_arr.reshape(-1)
    out[1, :] = second_arr.reshape(-1)
    return out


def _stack_variant_vectors(variants: list[VariantResult], attr: str) -> np.ndarray:
    if not variants:
        return np.empty((0, 0), dtype=np.float64)
    first = np.asarray(getattr(variants[0], attr), dtype=np.float64).reshape(-1)
    out = np.empty((len(variants), first.size), dtype=np.float64)
    out[0, :] = first
    for row, variant in enumerate(variants[1:], start=1):
        values = np.asarray(getattr(variant, attr), dtype=np.float64).reshape(-1)
        if values.size != first.size:
            raise ValueError(f"variant {variant.name} field {attr} size mismatch")
        out[row, :] = values
    return out


def _forward_measurement(
    fwd_model: EITForwardModel, sigma_values: np.ndarray
) -> np.ndarray:
    image = EITImage(
        elem_data=np.asarray(sigma_values, dtype=float), fwd_model=fwd_model
    )
    solution, _ = fwd_model.fwd_solve(image)
    return np.asarray(solution.meas, dtype=float)


def _metric_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_correlation(a, b)


def _run_variant(
    *,
    name: str,
    jacobian: np.ndarray,
    rhs: np.ndarray,
    fit_diff: np.ndarray,
    vh: np.ndarray,
    vi: np.ndarray,
    base_meas: np.ndarray,
    sigma_bg: np.ndarray,
    fwd_model: EITForwardModel,
    lam: float,
    n_inverse_points: int,
    holdout_indices: np.ndarray,
    full_diff: np.ndarray,
    fit_metrics_applicable: bool = True,
) -> VariantResult:
    delta_sigma = _solve_delta(jacobian, rhs, lam)
    sigma_est = sigma_bg + delta_sigma
    pred_vi = _forward_measurement(fwd_model, sigma_est)
    pred_diff = pred_vi - base_meas
    residual = pred_diff - full_diff
    fit_residual = fit_diff - full_diff
    fit_holdout_rmse = _rmse_at_indices(fit_residual, holdout_indices)
    fit_diff_rmse = float(np.sqrt(np.mean(fit_residual**2)))
    if not fit_metrics_applicable:
        fit_holdout_rmse = math.nan
        fit_diff_rmse = math.nan

    return VariantResult(
        name=name,
        n_inverse_points=n_inverse_points,
        delta_sigma=delta_sigma,
        sigma_est=sigma_est,
        pred_vi=pred_vi,
        pred_diff=pred_diff,
        fit_diff=fit_diff,
        fit_holdout_rmse=fit_holdout_rmse,
        fit_diff_rmse=fit_diff_rmse,
        pred_diff_rmse=float(np.sqrt(np.mean(residual**2))),
        pred_diff_mae=float(np.mean(np.abs(residual))),
        pred_diff_corr=_metric_corr(full_diff, pred_diff),
    )


def _mesh_cells(mesh) -> tuple[np.ndarray, np.ndarray]:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    if c2v is None:
        raise RuntimeError("failed to build cell-to-vertex connectivity")
    cells = np.array([c2v.links(i) for i in range(mesh.num_cells())], dtype=np.int32)
    points = np.asarray(mesh.geometry.x[:, :2], dtype=float)
    return points, cells


def _plot_reconstruction_comparison(
    *,
    output: Path,
    mesh,
    variants: list[VariantResult],
    dpi: int,
) -> Path:
    points, cells = _mesh_cells(mesh)
    tri = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    fields = [(item.name, item.delta_sigma) for item in variants]
    vmax = max(max(_max_abs_value(field) for _, field in fields), 1e-12)
    full = variants[0].delta_sigma
    diff_lim = 1e-12
    for item in variants[1:]:
        diff_lim = max(diff_lim, _max_abs_value(item.delta_sigma - full))
    n_cols = len(variants)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(2.55 * n_cols, 5.5),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Real tank holdout reconstruction comparison")
    for col, (label, field) in enumerate(fields):
        ax = axes[0, col]
        image = ax.tripcolor(
            tri,
            facecolors=field,
            shading="flat",
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            edgecolors="#ffffff",
            linewidth=0.04,
        )
        ax.set_title(label, fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        ax2 = axes[1, col]
        if col == 0:
            diff = np.zeros_like(field)
        else:
            diff = field - full
        diff_image = ax2.tripcolor(
            tri,
            facecolors=diff,
            shading="flat",
            cmap="coolwarm",
            vmin=-diff_lim,
            vmax=diff_lim,
            edgecolors="#ffffff",
            linewidth=0.04,
        )
        ax2.set_title("delta vs full_208", fontsize=9)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.colorbar(diff_image, ax=ax2, fraction=0.046, pad=0.02)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_fit_voltage_comparison(
    *,
    output: Path,
    reference_voltage: np.ndarray,
    target_voltage: np.ndarray,
    fitted_target_by_method: dict[str, np.ndarray],
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(
        4,
        4,
        figsize=(13.0, 9.0),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Real tank far3 fit on absolute target voltage U curves")
    colors = {"poly2_208": "#9467bd", "poly3_208": "#8c564b", "spline_208": "#17becf"}
    markers = {"poly2_208": "D", "poly3_208": "s", "spline_208": "P"}
    offsets = {"poly2_208": -0.18, "poly3_208": 0.0, "spline_208": 0.18}
    for stim, ax in enumerate(axes.ravel()):
        start = stim * 13
        stop = start + 13
        x = np.arange(13, dtype=float)
        target_frame = target_voltage[start:stop]
        reference_frame = reference_voltage[start:stop]
        holdout = np.zeros(13, dtype=bool)
        holdout[[5, 6, 7]] = True
        train = ~holdout
        ax.plot(x, target_frame, color="#1f77b4", lw=1.15, label="target full 13")
        ax.plot(
            x,
            reference_frame,
            color="#222222",
            lw=0.85,
            ls=(0, (3.0, 2.0)),
            alpha=0.78,
            label="reference full 13",
        )
        ax.scatter(
            x[train],
            target_frame[train],
            color="#2ca02c",
            s=16,
            label="fit input 10",
        )
        ax.scatter(
            x[holdout],
            target_frame[holdout],
            color="#ff7f0e",
            marker="x",
            s=34,
            label="withheld true 3",
        )
        for method, fitted in fitted_target_by_method.items():
            ax.scatter(
                x[holdout] + offsets[method],
                fitted[start:stop][holdout],
                color=colors[method],
                marker=markers[method],
                s=26,
                edgecolors="#ffffff",
                linewidths=0.45,
                label=f"{method} pred",
            )
        ax.set_title(f"stim {stim}", fontsize=9)
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def _plot_diff_prediction_comparison(
    *,
    output: Path,
    full_diff: np.ndarray,
    variants: list[VariantResult],
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(
        len(variants),
        2,
        figsize=(11.5, 2.35 * len(variants)),
        squeeze=False,
        constrained_layout=True,
    )
    idx = np.arange(full_diff.size)
    for row, variant in enumerate(variants):
        ax = axes[row, 0]
        ax.plot(idx, full_diff, color="#1f77b4", lw=0.9, label="Measured diff")
        ax.plot(
            idx,
            variant.pred_diff,
            color="#d62728",
            lw=0.9,
            ls="--",
            label="Predicted diff",
        )
        ax.set_title(
            f"{variant.name}: RMSE={variant.pred_diff_rmse:.4g}, r={variant.pred_diff_corr:.4f}",
            fontsize=9,
        )
        ax.set_xlabel("Measurement index")
        ax.set_ylabel("Voltage")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        scatter = axes[row, 1]
        scatter.scatter(full_diff, variant.pred_diff, s=12, alpha=0.75, color="#4682b4")
        vmin = min(float(np.min(full_diff)), float(np.min(variant.pred_diff)))
        vmax = max(float(np.max(full_diff)), float(np.max(variant.pred_diff)))
        scatter.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.0)
        scatter.set_xlabel("Measured diff")
        scatter.set_ylabel("Predicted diff")
        scatter.set_aspect("equal", adjustable="box")
        scatter.grid(alpha=0.25)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def _write_summary(
    output: Path,
    variants: list[VariantResult],
) -> Path:
    full = variants[0].delta_sigma
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for variant in variants:
            writer.writerow(
                {
                    "recon_method": variant.name,
                    "n_inverse_points": variant.n_inverse_points,
                    "fit_holdout_rmse": variant.fit_holdout_rmse,
                    "fit_diff_rmse": variant.fit_diff_rmse,
                    "pred_diff_rmse": variant.pred_diff_rmse,
                    "pred_diff_mae": variant.pred_diff_mae,
                    "pred_diff_corr": variant.pred_diff_corr,
                    "delta_sigma_min": float(np.min(variant.delta_sigma)),
                    "delta_sigma_max": float(np.max(variant.delta_sigma)),
                    "delta_sigma_l2": float(np.linalg.norm(variant.delta_sigma)),
                    "delta_vs_full208_l2": float(
                        np.linalg.norm(variant.delta_sigma - full)
                    ),
                }
            )
    return output


def _write_report(
    *,
    output: Path,
    args: argparse.Namespace,
    variants: list[VariantResult],
) -> Path:
    best_pred = min(variants, key=lambda item: item.pred_diff_rmse)
    best_fit = min(
        [
            item
            for item in variants
            if item.name.endswith("_208") and item.name != "full_208"
        ],
        key=lambda item: item.fit_holdout_rmse,
    )
    lines = [
        "# 实测 tank 208/160/拟合 208 对比实验",
        "",
        "## 参数",
        "",
        f"- CSV: `{args.csv}`",
        f"- mesh_name: `{args.mesh_name}`",
        f"- background_sigma: `{args.background_sigma}`",
        f"- lambda: `{args.lam}`",
        f"- pattern_amplitude: `{args.pattern_amplitude}`",
        f"- measurement_gain: `{args.measurement_gain}`",
        "- 对比：full_208、raw_160、poly2_208、poly3_208、spline_208。",
        "",
        "## 结论",
        "",
        f"- 预测差分电压 RMSE 最低：`{best_pred.name}` = `{best_pred.pred_diff_rmse:.12g}`。",
        f"- 远端 3 点拟合 holdout RMSE 最低：`{best_fit.name}` = `{best_fit.fit_holdout_rmse:.12g}`。",
        "- 本实验为实测数据，无电导率真值；重构图只能做方法间相对视觉比较。",
        "",
        "## 指标表",
        "",
        "| recon_method | n_inverse_points | fit_holdout_rmse | pred_diff_rmse | pred_diff_corr | delta_vs_full208_l2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    full = variants[0].delta_sigma
    for item in variants:
        lines.append(
            f"| {item.name} | {item.n_inverse_points} | "
            f"{item.fit_holdout_rmse:.12g} | {item.pred_diff_rmse:.12g} | "
            f"{item.pred_diff_corr:.12g} | "
            f"{np.linalg.norm(item.delta_sigma - full):.12g} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lambda", dest="lam", type=float, default=1.5)
    parser.add_argument("--background-sigma", type=float, default=0.008)
    parser.add_argument("--pattern-amplitude", type=float, default=5e-5)
    parser.add_argument("--measurement-gain", type=float, default=1.0)
    parser.add_argument("--mesh-name", type=str, default="mesh_16e_r0p025_ref10_cov0p5")
    parser.add_argument(
        "--mesh-dir", type=str, default=str(pyeidors_cache_path("eit_meshes"))
    )
    parser.add_argument("--radius", type=float, default=0.025)
    parser.add_argument("--electrode-coverage", type=float, default=0.5)
    parser.add_argument("--contact-impedance", type=float, default=1e-6)
    parser.add_argument("--use-part", choices=["real", "imag", "mag"], default="real")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    vh, vi = load_csv_measurements(
        args.csv,
        use_part=args.use_part,
        measurement_gain=args.measurement_gain,
    )
    if vh.shape != (208,) or vi.shape != (208,):
        raise RuntimeError(f"expected 208-point frames, got {vh.shape} and {vi.shape}")

    mesh = load_or_create_mesh(
        mesh_dir=args.mesh_dir,
        mesh_name=args.mesh_name,
        n_elec=16,
        radius=args.radius,
        electrode_coverage=args.electrode_coverage,
    )
    pattern_cfg = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="total_current",
        drive_value=args.pattern_amplitude,
        use_meas_current=False,
        rotate_meas=True,
    )
    fwd_model = EITForwardModel(
        n_elec=16,
        pattern_config=pattern_cfg,
        z=np.full(16, args.contact_impedance, dtype=float),
        mesh=mesh,
    )
    n_elem = int(
        fwd_model.V_sigma.dofmap.index_map.size_local
        * fwd_model.V_sigma.dofmap.index_map_bs
    )
    sigma_bg = np.full(n_elem, args.background_sigma, dtype=float)
    img_bg = EITImage(elem_data=sigma_bg, fwd_model=fwd_model)

    base_forward, _ = fwd_model.fwd_solve(img_bg)
    aligned, flipped = align_frames_polarity(
        _stack_two_frames(vh, vi), base_forward.meas
    )
    vh, vi = aligned
    if flipped:
        print(f"[INFO] Polarity correction: flipped frame indices {flipped}")

    jac_calc = EidorsJacobianAdapter(fwd_model, use_torch=False)
    jacobian = jac_calc.calculate_from_image(img_bg)
    if jacobian.shape[0] != 208:
        raise RuntimeError(f"expected 208 Jacobian rows, got {jacobian.shape}")

    _, train_indices, holdout_indices = _frame_indices_from_audit()
    full_diff = vi - vh

    variants: list[VariantResult] = []
    variants.append(
        _run_variant(
            name="full_208",
            jacobian=jacobian,
            rhs=full_diff,
            fit_diff=full_diff,
            vh=vh,
            vi=vi,
            base_meas=base_forward.meas,
            sigma_bg=sigma_bg,
            fwd_model=fwd_model,
            lam=args.lam,
            n_inverse_points=208,
            holdout_indices=holdout_indices,
            full_diff=full_diff,
            fit_metrics_applicable=False,
        )
    )
    variants.append(
        _run_variant(
            name="raw_160",
            jacobian=jacobian[train_indices, :],
            rhs=full_diff[train_indices],
            fit_diff=full_diff,
            vh=vh,
            vi=vi,
            base_meas=base_forward.meas,
            sigma_bg=sigma_bg,
            fwd_model=fwd_model,
            lam=args.lam,
            n_inverse_points=160,
            holdout_indices=holdout_indices,
            full_diff=full_diff,
            fit_metrics_applicable=False,
        )
    )

    fitted_diff_by_method: dict[str, np.ndarray] = {}
    fitted_target_by_method: dict[str, np.ndarray] = {}
    for method in ["poly2", "poly3", "spline"]:
        fit_vh = _fit_frame_vector(vh, method)
        fit_vi = _fit_frame_vector(vi, method)
        fit_j = _fit_jacobian_rows(jacobian, method)
        fit_diff = fit_vi - fit_vh
        name = f"{method}_208"
        fitted_diff_by_method[name] = fit_diff
        fitted_target_by_method[name] = fit_vi
        variants.append(
            _run_variant(
                name=name,
                jacobian=fit_j,
                rhs=fit_diff,
                fit_diff=fit_diff,
                vh=fit_vh,
                vi=fit_vi,
                base_meas=base_forward.meas,
                sigma_bg=sigma_bg,
                fwd_model=fwd_model,
                lam=args.lam,
                n_inverse_points=208,
                holdout_indices=holdout_indices,
                full_diff=full_diff,
            )
        )

    _write_summary(args.output / "summary.csv", variants)
    _write_report(output=args.output / "README.md", args=args, variants=variants)
    _plot_fit_voltage_comparison(
        output=args.output / "fit_voltage_comparison.png",
        reference_voltage=vh,
        target_voltage=vi,
        fitted_target_by_method=fitted_target_by_method,
        dpi=args.dpi,
    )
    _plot_diff_prediction_comparison(
        output=args.output / "diff_comparison.png",
        full_diff=full_diff,
        variants=variants,
        dpi=args.dpi,
    )
    _plot_reconstruction_comparison(
        output=args.output / "reconstruction_comparison.png",
        mesh=mesh,
        variants=variants,
        dpi=args.dpi,
    )
    write_output_bundle(
        args.output / "outputs.h5",
        {
            "vh": vh,
            "vi": vi,
            "full_diff": full_diff,
            "train_indices": train_indices,
            "holdout_indices": holdout_indices,
            "jacobian_shape": np.array(jacobian.shape, dtype=np.int64),
            "method_names": np.array([item.name for item in variants]),
            "delta_sigma": _stack_variant_vectors(variants, "delta_sigma"),
            "pred_diff": _stack_variant_vectors(variants, "pred_diff"),
            "fit_diff": _stack_variant_vectors(variants, "fit_diff"),
            "lambda_": np.array(args.lam),
            "background_sigma": np.array(args.background_sigma),
            "pattern_amplitude": np.array(args.pattern_amplitude),
            "measurement_gain": np.array(args.measurement_gain),
        },
        {
            "package_role": "tank_realdata_holdout_compare_outputs",
            "artifact_source": "scripts/run_tank_realdata_holdout_compare.py",
        },
        schema=GALLERY_ARRAYS_SCHEMA,
    )
    print("Saved to", args.output)
    for item in variants:
        print(
            f"{item.name}: n={item.n_inverse_points}, "
            f"fit_holdout_rmse={item.fit_holdout_rmse:.6g}, "
            f"pred_diff_rmse={item.pred_diff_rmse:.6g}, "
            f"corr={item.pred_diff_corr:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
