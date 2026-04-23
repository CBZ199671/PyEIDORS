#!/usr/bin/env python3
"""Run T23 dense circular-bucket voltage and holdout experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from pyeidors.data.bucket_dense_experiments import (
    BucketDenseExperimentCase,
    run_bucket_dense_experiments,
    write_bucket_dense_outputs,
)


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _format_table(case: BucketDenseExperimentCase) -> str:
    fields = [
        "experiment",
        "recon_method",
        "target_digits",
        "sigma_relative_rmse",
        "sigma_effective_digits",
        "artifact_energy",
    ]
    rendered = [
        [
            row.experiment,
            row.recon_method,
            "" if row.target_voltage_digits is None else str(row.target_voltage_digits),
            _format_float(row.sigma_relative_rmse),
            _format_float(row.sigma_effective_digits),
            _format_float(row.artifact_energy),
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


def _parse_optional_path(value: str) -> Path | None:
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "-"}:
        return None
    return Path(text)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run dense circle-bucket voltage-digit and far3 holdout experiments, "
            "then compare against available coarse-grid reference CSV files."
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
    parser.add_argument(
        "--target-digits",
        nargs="+",
        type=int,
        default=[4, 5, 6, 7],
    )
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
        default=Path("outputs/eit_bucket_dense_summary_16e.csv"),
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=Path("outputs/eit_bucket_dense_fields_16e.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/eit_bucket_dense_summary_16e.md"),
    )
    parser.add_argument(
        "--domain-plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_dense_domain_audit_16e.png"),
    )
    parser.add_argument(
        "--recon-plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_dense_recon_compare_16e.png"),
    )
    parser.add_argument(
        "--summary-plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_dense_summary_16e.png"),
    )
    parser.add_argument(
        "--curve-plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_dense_fit_curves_16e.png"),
    )
    parser.add_argument(
        "--holdout-summary-plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_dense_holdout_summary_16e.png"),
    )
    parser.add_argument(
        "--coarse-voltage-csv",
        type=_parse_optional_path,
        default=Path("outputs/eit_voltage_digit_sweep_16e.csv"),
    )
    parser.add_argument(
        "--coarse-holdout-csv",
        type=_parse_optional_path,
        default=None,
        help="Optional coarse holdout CSV. Defaults to none because square-grid holdout experiments were removed.",
    )
    parser.add_argument(
        "--coarse-structure-csv",
        type=_parse_optional_path,
        default=None,
        help="Optional coarse structure CSV. Defaults to none because square-grid holdout experiments were removed.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    include_raw = bool(args.raw_160_baseline)
    case = run_bucket_dense_experiments(
        domain=args.domain,
        bucket_radius=args.bucket_radius,
        n_elec=args.n_elec,
        mesh_h=args.mesh_h,
        ridge=args.ridge,
        target_digits=args.target_digits,
        holdout=args.holdout,
        raw_160_baseline=include_raw,
        fit_methods=args.fit_methods,
        inverse_backend=args.inverse_backend,
        allow_coarse_smoke=args.allow_coarse_smoke,
        normalize_rows=bool(args.row_normalize and not args.no_row_normalize),
    )
    written = write_bucket_dense_outputs(
        case,
        summary_output=args.output,
        field_output=args.field_output,
        report_output=args.report_output,
        domain_plot_output=args.domain_plot_output,
        recon_plot_output=args.recon_plot_output,
        summary_plot_output=args.summary_plot_output,
        curve_plot_output=args.curve_plot_output,
        holdout_summary_plot_output=args.holdout_summary_plot_output,
        coarse_voltage_csv=args.coarse_voltage_csv,
        coarse_holdout_csv=args.coarse_holdout_csv,
        coarse_structure_csv=args.coarse_structure_csv,
        dpi=args.dpi,
    )
    print(
        "settings: "
        f"domain={case.bucket.domain}, n_elec={case.bucket.n_elec}, "
        f"mesh_h={case.bucket.mesh_h}, n_cells={case.bucket.n_cells}, "
        f"n_dofs={case.bucket.n_dofs}, n_measurements={case.bucket.n_measurements}, "
        f"ridge={_format_float(args.ridge)}, inverse_backend={args.inverse_backend}, "
        f"row_normalize={bool(args.row_normalize and not args.no_row_normalize)}, "
        f"raw_160={include_raw}, fit_methods={','.join(args.fit_methods)}"
    )
    print(_format_table(case))
    for label, path in written.items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
