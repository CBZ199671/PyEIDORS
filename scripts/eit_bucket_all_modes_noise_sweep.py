#!/usr/bin/env python3
"""Run all acquisition-mode dense-bucket comparisons across noise SNR levels."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from pyeidors.data import add_noise
from pyeidors.data.bucket_dense_experiments import (
    BUCKET_FULL256_COMPARE_FIELD_FIELDS,
    BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
    _field_rows_for_sigma,
    _full256_summary_from_metrics,
    _measurement_submodel,
    _structure_metrics,
    build_circle_bucket_linearized_model,
)
from pyeidors.data.bucket_domain_audit import build_circle_bucket_domain
from pyeidors.data.eit_digit_metrics import (
    EITLinearizedModel,
    reconstruct_linearized_sigma,
)
from pyeidors.data.holdout_fit_diff import run_holdout_fit_diff
from pyeidors.data.holdout_point_audit import build_holdout_point_audit
from pyeidors.data._sweep_core import write_sweep_table_artifacts


NOISE_SUMMARY_FIELDS = [
    "noise_snr_requested",
    "noise_seed",
    "actual_snr",
    *BUCKET_FULL256_COMPARE_SUMMARY_FIELDS,
]
NOISE_FIELD_FIELDS = [
    "noise_snr_requested",
    "noise_seed",
    *BUCKET_FULL256_COMPARE_FIELD_FIELDS,
]
METHOD_ORDER = [
    "full_256",
    "full_208",
    "far3_drop_near3_keep_208",
    "raw_160",
    "poly2_208",
    "poly3_208",
    "spline_208",
]


def _format_float(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _parse_snr(value: str) -> float:
    name = str(value).strip().lower()
    if name in {"inf", "infinity", "none", "noiseless", "0"}:
        return math.inf
    parsed = float(name)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("SNR must be positive or inf")
    return parsed


def _actual_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    signal_norm = float(np.linalg.norm(signal))
    noise_norm = float(np.linalg.norm(noise))
    if noise_norm == 0.0:
        return math.inf
    return signal_norm / noise_norm


def _global_208_indices(n_elec: int) -> tuple[np.ndarray, np.ndarray]:
    point_rows, point_summary = build_holdout_point_audit(n_elec=n_elec)
    full208_indices = [
        row.global_index_256
        for row in point_rows
        if row.point_status != "drive_removed"
    ]
    train208_indices = []
    for row in point_rows:
        if row.point_status != "fit_train_160":
            continue
        if row.frame_index_13 is None:
            raise RuntimeError("train row missing frame_index_13")
        train208_indices.append(
            int(row.stim_index) * int(point_summary.points_per_kept_frame)
            + int(row.frame_index_13)
        )
    if len(full208_indices) != point_summary.kept_208_count:
        raise RuntimeError("full208 index count mismatch")
    if len(train208_indices) != point_summary.fit_train_160_count:
        raise RuntimeError("train160 index count mismatch")
    return (
        np.asarray(full208_indices, dtype=np.int64),
        np.asarray(train208_indices, dtype=np.int64),
    )


def _far3_drop_near3_keep_256_indices(n_elec: int) -> np.ndarray:
    point_rows, point_summary = build_holdout_point_audit(n_elec=n_elec)
    indices = [
        row.global_index_256 for row in point_rows if row.point_status != "holdout_far3"
    ]
    expected = point_summary.full_candidate_count - point_summary.holdout_far3_count
    if len(indices) != expected:
        raise RuntimeError("far3-drop/near3-keep count mismatch")
    return np.asarray(indices, dtype=np.int64)


def _replace_voltage_true(
    model: EITLinearizedModel,
    voltage_true: np.ndarray,
    *,
    label: str,
) -> EITLinearizedModel:
    from dataclasses import replace

    values = np.asarray(voltage_true, dtype=float)
    if values.shape != np.asarray(model.voltage_true).shape:
        raise ValueError("voltage_true shape mismatch")
    return replace(model, voltage_true=values.copy(), label=label)


def _add_noise_to_full256(
    model_full_256: EITLinearizedModel,
    *,
    snr: float,
    seed: int,
) -> np.ndarray:
    clean = np.asarray(model_full_256.voltage_true, dtype=float)
    if math.isinf(float(snr)):
        return clean.copy()
    return np.asarray(
        add_noise(
            float(snr),
            clean,
            np.asarray(model_full_256.voltage_reference, dtype=float),
            seed=int(seed),
        ),
        dtype=float,
    )


def _actual_snr_by_method(
    *,
    method: str,
    clean_models: dict[str, EITLinearizedModel],
    noisy_models: dict[str, EITLinearizedModel],
    clean_holdout,
    noisy_holdout,
    train208_indices: np.ndarray,
) -> float:
    if method in {"full_256", "full_208", "far3_drop_near3_keep_208"}:
        clean = clean_models[method]
        noisy = noisy_models[method]
        signal = np.asarray(clean.voltage_true) - np.asarray(clean.voltage_reference)
        noise = np.asarray(noisy.voltage_true) - np.asarray(clean.voltage_true)
        return _actual_snr(signal, noise)
    if method == "raw_160":
        signal = (
            np.asarray(clean_models["full_208"].voltage_true)[train208_indices]
            - np.asarray(clean_models["full_208"].voltage_reference)[train208_indices]
        )
        noise = (
            np.asarray(noisy_models["full_208"].voltage_true)[train208_indices]
            - np.asarray(clean_models["full_208"].voltage_true)[train208_indices]
        )
        return _actual_snr(signal, noise)

    clean_ref, clean_true, _ = clean_holdout.fit_voltage_by_method[method]
    _, noisy_true, _ = noisy_holdout.fit_voltage_by_method[method]
    return _actual_snr(clean_true - clean_ref, noisy_true - clean_true)


def _run_one_snr(
    *,
    bucket,
    model_full_256: EITLinearizedModel,
    model_full_208: EITLinearizedModel,
    clean_holdout,
    snr: float,
    seed: int,
    ridge: float,
    fit_methods: list[str],
    inverse_backend: str,
    full208_indices: np.ndarray,
    train208_indices: np.ndarray,
    far3_drop_indices: np.ndarray,
) -> tuple[
    list[dict[str, str | float | int]],
    list[dict[str, str | float | int]],
    dict[str, np.ndarray],
]:
    noisy_full_256_values = _add_noise_to_full256(
        model_full_256,
        snr=snr,
        seed=seed,
    )
    noisy_full_256 = _replace_voltage_true(
        model_full_256,
        noisy_full_256_values,
        label="circle_bucket_dense_full_256_noisy",
    )
    noisy_full_208 = _replace_voltage_true(
        model_full_208,
        noisy_full_256_values[full208_indices],
        label="circle_bucket_dense_noisy",
    )
    noisy_far3_drop = _measurement_submodel(
        noisy_full_256,
        far3_drop_indices,
        label="circle_bucket_dense_far3_drop_near3_keep_208_noisy",
    )
    clean_far3_drop = _measurement_submodel(
        model_full_256,
        far3_drop_indices,
        label="circle_bucket_dense_far3_drop_near3_keep_208",
    )

    noisy_holdout = run_holdout_fit_diff(
        model=noisy_full_208,
        holdout="far3",
        fit_methods=fit_methods,
        raw_160_baseline=True,
        ridge=ridge,
        inverse_backend=inverse_backend,
    )
    sigma_full_256 = reconstruct_linearized_sigma(
        model=noisy_full_256,
        voltages=noisy_full_256.voltage_true,
        ridge=ridge,
        inverse_backend=inverse_backend,
    )
    sigma_far3_drop = reconstruct_linearized_sigma(
        model=noisy_far3_drop,
        voltages=noisy_far3_drop.voltage_true,
        ridge=ridge,
        inverse_backend=inverse_backend,
    )
    sigma_by_method: dict[str, np.ndarray] = {
        "full_256": sigma_full_256,
        "full_208": noisy_holdout.sigma_recon_full,
        "far3_drop_near3_keep_208": sigma_far3_drop,
    }
    sigma_by_method.update(noisy_holdout.sigma_recon_by_method)

    metrics_by_method = {
        method: _structure_metrics(bucket=bucket, sigma_recon=sigma)
        for method, sigma in sigma_by_method.items()
    }
    baseline_metrics = metrics_by_method["full_208"]
    clean_models = {
        "full_256": model_full_256,
        "full_208": model_full_208,
        "far3_drop_near3_keep_208": clean_far3_drop,
    }
    noisy_models = {
        "full_256": noisy_full_256,
        "full_208": noisy_full_208,
        "far3_drop_near3_keep_208": noisy_far3_drop,
    }
    n_measurements = {
        "full_256": int(model_full_256.n_measurements),
        "full_208": int(model_full_208.n_measurements),
        "far3_drop_near3_keep_208": int(model_full_256.n_measurements),
        "raw_160": int(model_full_208.n_measurements),
        "poly2_208": int(model_full_208.n_measurements),
        "poly3_208": int(model_full_208.n_measurements),
        "spline_208": int(model_full_208.n_measurements),
    }
    n_inverse_points = {
        "full_256": int(model_full_256.n_measurements),
        "full_208": int(model_full_208.n_measurements),
        "far3_drop_near3_keep_208": int(noisy_far3_drop.n_measurements),
        "raw_160": int(train208_indices.size),
        "poly2_208": int(model_full_208.n_measurements),
        "poly3_208": int(model_full_208.n_measurements),
        "spline_208": int(model_full_208.n_measurements),
    }
    snr_label = _format_float(snr)
    summary_rows: list[dict[str, str | float | int]] = []
    field_rows: list[dict[str, str | float | int]] = []
    for method in METHOD_ORDER:
        if method not in sigma_by_method:
            continue
        summary = _full256_summary_from_metrics(
            bucket=bucket,
            ridge=ridge,
            recon_method=method,
            n_measurements=n_measurements[method],
            n_inverse_points=n_inverse_points[method],
            sigma_recon=sigma_by_method[method],
            sigma_baseline=sigma_by_method["full_208"],
            metrics=metrics_by_method[method],
            baseline_metrics=baseline_metrics,
        )
        summary_dict = summary.as_csv_row()
        summary_dict.update(
            {
                "noise_snr_requested": snr_label,
                "noise_seed": int(seed),
                "actual_snr": _actual_snr_by_method(
                    method=method,
                    clean_models=clean_models,
                    noisy_models=noisy_models,
                    clean_holdout=clean_holdout,
                    noisy_holdout=noisy_holdout,
                    train208_indices=train208_indices,
                ),
            }
        )
        summary_rows.append(summary_dict)
        for field_row in _field_rows_for_sigma(
            bucket=bucket,
            experiment="full256_noise_sweep",
            recon_method=method,
            sigma_recon=sigma_by_method[method],
        ):
            row_dict = field_row.as_csv_row()
            row_dict.update(
                {
                    "noise_snr_requested": snr_label,
                    "noise_seed": int(seed),
                }
            )
            field_rows.append(row_dict)
    return summary_rows, field_rows, sigma_by_method


def _read_metric(row: dict[str, str | float | int], name: str) -> float:
    return float(row[name])


def plot_noise_metric_summary(
    rows: list[dict[str, str | float | int]],
    output_path: Path,
    *,
    dpi: int = 200,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyeidors.data.digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    x_by_label: dict[str, float] = {}
    for row in rows:
        snr = row["noise_snr_requested"]
        snr_value = math.inf if str(snr) == "inf" else float(snr)
        x_by_label[str(snr)] = 0.0 if math.isinf(snr_value) else 1.0 / snr_value
    labels = sorted(x_by_label, key=lambda key: x_by_label[key])

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.0), constrained_layout=True)
    fig.suptitle("All acquisition modes across EIDORS add_noise SNR ladder")
    panels = [
        ("Sigma relative RMSE", "sigma_relative_rmse"),
        ("Artifact energy", "artifact_energy"),
        ("Direct field L2 vs full_208", "delta_field_l2_vs_full_208"),
        ("Actual inverse-input SNR", "actual_snr"),
    ]
    for ax, (title, field) in zip(axes.ravel(), panels, strict=True):
        for method in METHOD_ORDER:
            method_rows = [row for row in rows if str(row["recon_method"]) == method]
            by_snr = {str(row["noise_snr_requested"]): row for row in method_rows}
            values = [
                _read_metric(by_snr[label], field) if label in by_snr else math.nan
                for label in labels
            ]
            if field == "actual_snr":
                values = [math.nan if math.isinf(value) else value for value in values]
            ax.plot(
                [x_by_label[label] for label in labels],
                values,
                marker="o",
                linewidth=1.35,
                label=method,
            )
        ax.set_title(title)
        ax.set_xlabel("Noise strength 1/SNR (0=noiseless)")
        ax.grid(True, alpha=0.25)
        if field == "actual_snr":
            ax.set_ylabel("actual SNR")
        else:
            ax.set_ylabel(field)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def _draw_bucket_field(
    ax, bucket, values: np.ndarray, *, vmin: float, vmax: float, cmap: str
):
    import matplotlib.tri as mtri
    from matplotlib.patches import Circle

    triangulation = mtri.Triangulation(
        bucket.nodes[:, 0],
        bucket.nodes[:, 1],
        bucket.cells,
    )
    image = ax.tripcolor(
        triangulation,
        facecolors=np.asarray(values, dtype=float),
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="#ffffff",
        linewidth=0.03,
    )
    ax.add_patch(
        Circle((0.0, 0.0), bucket.bucket_radius, fill=False, color="#111111", lw=0.8)
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def _value_range(values: list[np.ndarray]) -> tuple[float, float]:
    if not values:
        raise ValueError("values must not be empty")
    vmin = math.inf
    vmax = -math.inf
    for value in values:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size == 0:
            continue
        vmin = min(vmin, float(np.min(arr)))
        vmax = max(vmax, float(np.max(arr)))
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        raise ValueError("values contain no finite entries")
    return vmin, vmax


def _max_abs_value(values: list[np.ndarray]) -> float:
    limit = 0.0
    for value in values:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size:
            limit = max(limit, float(np.max(np.abs(arr))))
    return limit


def plot_noise_recon_grid(
    bucket,
    sigma_by_snr: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    *,
    selected_snr: list[str],
    error: bool = False,
    dpi: int = 200,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pyeidors.data.digit_plot import configure_times_new_roman

    configure_times_new_roman()
    output = Path(output_path).with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)

    available = [snr for snr in selected_snr if snr in sigma_by_snr]
    if not available:
        raise ValueError("selected_snr has no overlap with produced data")
    values = []
    for snr in available:
        for method in METHOD_ORDER:
            if method not in sigma_by_snr[snr]:
                continue
            field = sigma_by_snr[snr][method]
            values.append(field - bucket.sigma_true if error else field)
    if error:
        limit = _max_abs_value(values)
        vmin, vmax, cmap = -limit, limit, "coolwarm"
        title = "Noise ladder recon error vs truth"
    else:
        vmin, vmax = _value_range(values)
        cmap = "viridis"
        title = "Noise ladder reconstructed conductivity"

    fig, axes = plt.subplots(
        len(available),
        len(METHOD_ORDER),
        figsize=(2.15 * len(METHOD_ORDER), 2.0 * len(available)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=13)
    for row_idx, snr in enumerate(available):
        for col_idx, method in enumerate(METHOD_ORDER):
            ax = axes[row_idx, col_idx]
            if method not in sigma_by_snr[snr]:
                ax.axis("off")
                continue
            field = sigma_by_snr[snr][method]
            shown = field - bucket.sigma_true if error else field
            image = _draw_bucket_field(
                ax, bucket, shown, vmin=vmin, vmax=vmax, cmap=cmap
            )
            ax.set_title(f"{method}\nSNR={snr}", fontsize=7)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def format_noise_report(rows: list[dict[str, str | float | int]]) -> str:
    by_snr: dict[str, list[dict[str, str | float | int]]] = {}
    for row in rows:
        by_snr.setdefault(str(row["noise_snr_requested"]), []).append(row)
    labels = sorted(
        by_snr,
        key=lambda label: 0.0 if label == "inf" else 1.0 / float(label),
    )
    lines = [
        "# 仿真各情况加噪声梯度测试",
        "",
        "- 噪声接口：`pyeidors.data.add_noise`，EIDORS SNR-exact 口径。",
        "- 噪声添加：先对 full_256 target 电压加噪，再按采集模式取子集或拟合。",
        "- 模式：full_256、full_208、far3_drop_near3_keep_208、raw_160、poly2_208、poly3_208、spline_208。",
        "",
        "| requested SNR | best sigma_relative_rmse | best artifact_energy | best direct L2 vs full_208 |",
        "|---:|---|---|---|",
    ]
    for label in labels:
        snr_rows = by_snr[label]
        best_rmse = min(snr_rows, key=lambda row: float(row["sigma_relative_rmse"]))
        best_artifact = min(snr_rows, key=lambda row: float(row["artifact_energy"]))
        candidates = [row for row in snr_rows if row["recon_method"] != "full_208"]
        best_l2 = min(
            candidates, key=lambda row: float(row["delta_field_l2_vs_full_208"])
        )
        lines.append(
            f"| {label} | {best_rmse['recon_method']}="
            f"{float(best_rmse['sigma_relative_rmse']):.6g} | "
            f"{best_artifact['recon_method']}="
            f"{float(best_artifact['artifact_energy']):.6g} | "
            f"{best_l2['recon_method']}="
            f"{float(best_l2['delta_field_l2_vs_full_208']):.6g} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all-mode dense bucket reconstruction under add_noise SNR ladder.",
    )
    parser.add_argument("--mesh-h", type=float, default=0.1)
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--bucket-radius", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument(
        "--snr-levels",
        nargs="+",
        type=_parse_snr,
        default=[math.inf, 100.0, 50.0, 20.0, 10.0, 5.0, 2.0],
    )
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument(
        "--fit-methods",
        nargs="+",
        choices=["poly2", "poly3", "spline"],
        default=["poly2", "poly3", "spline"],
    )
    parser.add_argument(
        "--inverse-backend",
        choices=["measurement-rm", "pyeidors-rm", "least-squares"],
        default="measurement-rm",
    )
    parser.add_argument("--allow-coarse-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hdf5-output",
        type=Path,
        default=None,
        help="Optional shared HDF5 report-table artifact output path.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional shared JSON report-table artifact output path.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_methods = list(dict.fromkeys(str(item) for item in args.fit_methods))

    bucket = build_circle_bucket_domain(
        domain="circle_bucket",
        bucket_radius=args.bucket_radius,
        n_elec=args.n_elec,
        mesh_h=args.mesh_h,
        allow_coarse_smoke=bool(args.allow_coarse_smoke),
    )
    model_full_208 = build_circle_bucket_linearized_model(bucket=bucket)
    model_full_256 = build_circle_bucket_linearized_model(
        bucket=bucket,
        include_drive_related=True,
    )
    full208_indices, train208_indices = _global_208_indices(args.n_elec)
    far3_drop_indices = _far3_drop_near3_keep_256_indices(args.n_elec)
    clean_holdout = run_holdout_fit_diff(
        model=model_full_208,
        holdout="far3",
        fit_methods=fit_methods,
        raw_160_baseline=True,
        ridge=args.ridge,
        inverse_backend=args.inverse_backend,
    )

    all_summary_rows: list[dict[str, str | float | int]] = []
    all_field_rows: list[dict[str, str | float | int]] = []
    sigma_by_snr: dict[str, dict[str, np.ndarray]] = {}
    for level_index, snr in enumerate(args.snr_levels):
        seed = int(args.seed) + int(level_index)
        summary_rows, field_rows, sigma_by_method = _run_one_snr(
            bucket=bucket,
            model_full_256=model_full_256,
            model_full_208=model_full_208,
            clean_holdout=clean_holdout,
            snr=float(snr),
            seed=seed,
            ridge=float(args.ridge),
            fit_methods=fit_methods,
            inverse_backend=str(args.inverse_backend),
            full208_indices=full208_indices,
            train208_indices=train208_indices,
            far3_drop_indices=far3_drop_indices,
        )
        all_summary_rows.extend(summary_rows)
        all_field_rows.extend(field_rows)
        sigma_by_snr[_format_float(float(snr))] = sigma_by_method
        print(
            f"SNR={_format_float(float(snr))}: "
            f"{len(summary_rows)} methods, seed={seed}",
        )

    summary_csv = output_dir / "eit_bucket_all_modes_noise_sweep_summary_16e.csv"
    fields_csv = output_dir / "eit_bucket_all_modes_noise_sweep_fields_16e.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NOISE_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(all_summary_rows)
    with fields_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NOISE_FIELD_FIELDS)
        writer.writeheader()
        writer.writerows(all_field_rows)
    table_artifacts = write_sweep_table_artifacts(
        tables={
            "bucket_all_modes_noise_summary": (
                NOISE_SUMMARY_FIELDS,
                all_summary_rows,
            ),
            "bucket_all_modes_noise_field": (NOISE_FIELD_FIELDS, all_field_rows),
        },
        hdf5_output=args.hdf5_output,
        json_output=args.json_output,
        metadata={
            "report_kind": "bucket_all_modes_noise_sweep",
            "domain": bucket.domain,
            "mesh_h": bucket.mesh_h,
            "n_cells": bucket.n_cells,
            "n_dofs": bucket.n_dofs,
            "n_elec": bucket.n_elec,
            "ridge": float(args.ridge),
            "inverse_backend": str(args.inverse_backend),
            "fit_methods": fit_methods,
        },
    )

    report = output_dir / "eit_bucket_all_modes_noise_sweep_summary_16e.md"
    report.write_text(format_noise_report(all_summary_rows), encoding="utf-8")
    plot_noise_metric_summary(
        all_summary_rows,
        output_dir / "eit_bucket_all_modes_noise_sweep_metrics_16e.png",
        dpi=args.dpi,
    )
    selected = [label for label in ["inf", "20", "5", "2"] if label in sigma_by_snr]
    if not selected:
        selected = list(sigma_by_snr)[: min(4, len(sigma_by_snr))]
    plot_noise_recon_grid(
        bucket,
        sigma_by_snr,
        output_dir / "eit_bucket_all_modes_noise_sweep_recon_grid_16e.png",
        selected_snr=selected,
        error=False,
        dpi=args.dpi,
    )
    plot_noise_recon_grid(
        bucket,
        sigma_by_snr,
        output_dir / "eit_bucket_all_modes_noise_sweep_error_grid_16e.png",
        selected_snr=selected,
        error=True,
        dpi=args.dpi,
    )
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote fields: {fields_csv}")
    for label, path in table_artifacts.items():
        print(f"Wrote {label}: {path}")
    print(f"Wrote report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
