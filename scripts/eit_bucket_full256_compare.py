#!/usr/bin/env python3
"""Run T27 full-256-vs-filtered dense circular-bucket comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from pyeidors.data.bucket_dense_experiments import (
    BucketFull256CompareCase,
    run_bucket_full256_compare_experiment,
    write_bucket_full256_compare_outputs,
)


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _format_table(case: BucketFull256CompareCase) -> str:
    fields = [
        "recon_method",
        "n_inverse_points",
        "sigma_relative_rmse",
        "delta_rmse_vs_208",
        "artifact_energy",
        "delta_artifact_vs_208",
        "direct_l2_vs_208",
    ]
    rendered = [
        [
            row.recon_method,
            str(row.n_inverse_points),
            _format_float(row.sigma_relative_rmse),
            _format_float(row.delta_sigma_relative_rmse_vs_full_208),
            _format_float(row.artifact_energy),
            _format_float(row.delta_artifact_energy_vs_full_208),
            _format_float(row.delta_field_l2_vs_full_208),
        ]
        for row in case.summaries
    ]
    widths = [
        max(len(fields[idx]), *(len(row[idx]) for row in rendered))
        for idx in range(len(fields))
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(fields)),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in rendered
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dense circle-bucket full 256-point comparison against native "
            "208, raw 160, and fitted 208 reconstructions."
        ),
    )
    parser.add_argument("--domain", choices=["circle_bucket"], default="circle_bucket")
    parser.add_argument("--bucket-radius", type=float, default=1.0)
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument(
        "--mesh-h",
        type=float,
        default=0.1,
        help="Dense circle-bucket mesh spacing. 0.1 gives about 800 cells.",
    )
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--holdout", choices=["far3"], default="far3")
    parser.add_argument("--raw-160-baseline", action="store_true")
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
    parser.add_argument(
        "--allow-coarse-smoke",
        action="store_true",
        help="Allow mesh below dense threshold for tests or explicit smokes.",
    )
    parser.add_argument(
        "--row-normalize",
        action="store_true",
        help="Enable row normalization in the analytic circle-bucket sensitivity.",
    )
    parser.add_argument(
        "--no-row-normalize",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eit_bucket_full256_compare_summary_16e.csv"),
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=Path("outputs/eit_bucket_full256_compare_fields_16e.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/eit_bucket_full256_compare_summary_16e.md"),
    )
    parser.add_argument(
        "--recon-plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_full256_compare_recon_16e.png"),
    )
    parser.add_argument(
        "--metrics-plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_full256_compare_metrics_16e.png"),
    )
    parser.add_argument(
        "--recon-delta-plot-output",
        type=Path,
        default=None,
        help=(
            "Optional 3-row reconstruction plot with sigma, error vs truth, "
            "and direct delta vs full_208."
        ),
    )
    parser.add_argument(
        "--point-audit-plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_full256_point_audit_16e.png"),
    )
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
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    case = run_bucket_full256_compare_experiment(
        domain=args.domain,
        bucket_radius=args.bucket_radius,
        n_elec=args.n_elec,
        mesh_h=args.mesh_h,
        ridge=args.ridge,
        holdout=args.holdout,
        raw_160_baseline=bool(args.raw_160_baseline),
        fit_methods=args.fit_methods,
        inverse_backend=args.inverse_backend,
        allow_coarse_smoke=args.allow_coarse_smoke,
        normalize_rows=bool(args.row_normalize and not args.no_row_normalize),
    )
    written = write_bucket_full256_compare_outputs(
        case,
        summary_output=args.output,
        field_output=args.field_output,
        report_output=args.report_output,
        recon_plot_output=args.recon_plot_output,
        metrics_plot_output=args.metrics_plot_output,
        point_audit_plot_output=args.point_audit_plot_output,
        recon_delta_plot_output=args.recon_delta_plot_output,
        hdf5_output=args.hdf5_output,
        json_output=args.json_output,
        dpi=args.dpi,
    )
    print(
        "settings: "
        f"domain={case.bucket.domain}, n_elec={case.bucket.n_elec}, "
        f"mesh_h={case.bucket.mesh_h}, n_cells={case.bucket.n_cells}, "
        f"n_dofs={case.bucket.n_dofs}, "
        f"full_256_measurements={case.model_full_256.n_measurements}, "
        f"full_208_measurements={case.model_full_208.n_measurements}, "
        f"ridge={_format_float(args.ridge)}, inverse_backend={args.inverse_backend}, "
        f"row_normalize={bool(args.row_normalize and not args.no_row_normalize)}, "
        f"raw_160={bool(args.raw_160_baseline)}, "
        f"fit_methods={','.join(args.fit_methods)}"
    )
    print(_format_table(case))
    for label, path in written.items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
