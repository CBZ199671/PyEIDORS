#!/usr/bin/env python3
"""Run T21 holdout/raw-160/fitted-208 difference-imaging experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from pyeidors.data.eit_digit_metrics import (
    adjacent_measurement_count,
    build_pyeidors_fem_linearized_model,
    build_surrogate_linearized_model,
)
from pyeidors.data.holdout_fit_diff import (
    FIELD_FIELDS,
    STRUCTURE_FIELDS,
    SUMMARY_FIELDS,
    HoldoutFitDiffCase,
    format_holdout_fit_report,
    plot_holdout_fit_curves,
    plot_holdout_fit_summary,
    plot_holdout_recon_compare,
    populate_point_rows_with_voltages,
    run_holdout_fit_diff,
    write_holdout_point_audit_plot,
)
from pyeidors.data.holdout_point_audit import POINT_AUDIT_FIELDS, HoldoutPointAuditRow


def _format_float(value: float | None) -> str:
    if value is None:
        return "none"
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _write_csv(path: Path, fields: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _write_point_csv(path: Path, rows: list[HoldoutPointAuditRow]) -> None:
    _write_csv(path, POINT_AUDIT_FIELDS, rows)


def _build_model(args: argparse.Namespace):
    backend = str(args.forward_backend).strip().lower()
    expected = adjacent_measurement_count(args.fem_n_elec)
    if backend in {"surrogate", "linear-surrogate"}:
        n_measurements = (
            expected if args.n_measurements is None else args.n_measurements
        )
        if int(n_measurements) != expected:
            raise ValueError(
                "--n-measurements must match adjacent 208-style count for holdout: "
                f"{n_measurements} != {expected}"
            )
        return build_surrogate_linearized_model(
            n_measurements=n_measurements,
            n_parameters=args.n_parameters,
            seed=args.model_seed,
        )
    if backend in {"pyeidors-fem", "fem"}:
        return build_pyeidors_fem_linearized_model(
            n_elec=args.fem_n_elec,
            grid=args.fem_grid,
            expected_measurements=expected,
            sigma_rule=args.anomaly_rule,
        )
    raise ValueError("--forward-backend must be surrogate or pyeidors-fem")


def _format_table(case: HoldoutFitDiffCase) -> str:
    columns = [
        "recon_method",
        "n_inverse_points",
        "diff_voltage_rmse",
        "delta_sigma_relative_rmse",
        "delta_sigma_effective_digits",
    ]
    rendered = [
        [
            row.recon_method,
            str(row.n_inverse_points),
            _format_float(row.diff_voltage_rmse),
            _format_float(row.delta_sigma_relative_rmse),
            _format_float(row.delta_sigma_effective_digits),
        ]
        for row in case.summaries
    ]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rendered))
        for idx in range(len(columns))
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(columns)),
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
            "Compare full 208-point reconstruction, raw 160-point reconstruction, "
            "and fitted 208-point reconstructions after far3 holdout."
        ),
    )
    parser.add_argument(
        "--forward-backend",
        choices=["surrogate", "pyeidors-fem"],
        default="pyeidors-fem",
        help="Forward model backend.",
    )
    parser.add_argument("--fem-n-elec", type=int, default=16, help="Electrode count.")
    parser.add_argument("--fem-grid", type=int, default=4, help="FEM grid size.")
    parser.add_argument(
        "--n-measurements",
        type=int,
        help="Surrogate measurement count; defaults to adjacent kept count.",
    )
    parser.add_argument(
        "--n-parameters",
        type=int,
        default=8,
        help="Surrogate conductivity parameter count.",
    )
    parser.add_argument("--model-seed", type=int, default=20260422)
    parser.add_argument(
        "--anomaly-rule",
        default="default",
        help="FEM sigma_true anomaly rule.",
    )
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument(
        "--inverse-backend",
        choices=["pyeidors-rm", "least-squares"],
        default="pyeidors-rm",
    )
    parser.add_argument(
        "--rm-mode",
        choices=["tikhonov", "noser"],
        default="tikhonov",
    )
    parser.add_argument(
        "--rm-form",
        choices=["param", "measurement"],
        default="param",
    )
    parser.add_argument("--noser-exponent", type=float, default=0.5)
    parser.add_argument(
        "--holdout",
        choices=["far3"],
        default="far3",
        help="Holdout rule.",
    )
    parser.add_argument(
        "--raw-160-baseline",
        action="store_true",
        help="Include direct 160-point reconstruction baseline.",
    )
    parser.add_argument(
        "--fit-methods",
        nargs="+",
        choices=["poly2", "poly3", "spline"],
        default=["poly2", "poly3", "spline"],
    )
    parser.add_argument("--plot-voltage-points", action="store_true")
    parser.add_argument("--plot-recon-compare", action="store_true")
    parser.add_argument("--structure-metrics", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eit_holdout_fit_diff_16e.csv"),
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=Path("outputs/eit_holdout_fit_diff_fields_16e.csv"),
    )
    parser.add_argument(
        "--point-output",
        type=Path,
        default=Path("outputs/eit_holdout_voltage_points_16e.csv"),
    )
    parser.add_argument(
        "--structure-output",
        type=Path,
        default=Path("outputs/eit_holdout_structure_metrics_16e.csv"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/eit_holdout_fit_diff_16e.md"),
    )
    parser.add_argument(
        "--point-plot-output",
        type=Path,
        default=Path("outputs/eit_holdout_voltage_points_16e.png"),
    )
    parser.add_argument(
        "--curve-plot-output",
        type=Path,
        default=Path("outputs/eit_holdout_fit_curves_16e.png"),
    )
    parser.add_argument(
        "--recon-plot-output",
        type=Path,
        default=Path("outputs/eit_holdout_recon_compare_16e.png"),
    )
    parser.add_argument(
        "--summary-plot-output",
        type=Path,
        default=Path("outputs/eit_holdout_fit_diff_16e.png"),
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    model = _build_model(args)
    include_raw = bool(args.raw_160_baseline)
    case = run_holdout_fit_diff(
        model=model,
        holdout=args.holdout,
        fit_methods=args.fit_methods,
        raw_160_baseline=include_raw,
        ridge=args.ridge,
        inverse_backend=args.inverse_backend,
        rm_mode=args.rm_mode,
        rm_form=args.rm_form,
        noser_exponent=args.noser_exponent,
    )
    _write_csv(args.output, SUMMARY_FIELDS, case.summaries)
    _write_csv(args.field_output, FIELD_FIELDS, case.field_rows)
    representative_method = next(
        (
            f"{method}_208"
            for method in args.fit_methods
            if f"{method}_208" in case.fit_voltage_by_method
        ),
        None,
    )
    representative_fit = (
        None
        if representative_method is None
        else case.fit_voltage_by_method[representative_method]
    )
    point_rows = populate_point_rows_with_voltages(
        point_rows=case.point_rows,
        model=model,
        fit_voltage=representative_fit,
    )
    _write_point_csv(args.point_output, point_rows)
    if args.structure_metrics:
        _write_csv(args.structure_output, STRUCTURE_FIELDS, case.structure_rows)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(format_holdout_fit_report(case), encoding="utf-8")
    point_plot = None
    if args.plot_voltage_points:
        point_plot = write_holdout_point_audit_plot(
            case,
            args.point_plot_output,
            dpi=args.dpi,
        )
    curve_plot = plot_holdout_fit_curves(case, args.curve_plot_output, dpi=args.dpi)
    summary_plot = plot_holdout_fit_summary(
        case, args.summary_plot_output, dpi=args.dpi
    )
    recon_plot = None
    if args.plot_recon_compare:
        recon_plot = plot_holdout_recon_compare(
            case, args.recon_plot_output, dpi=args.dpi
        )

    print(
        "settings: "
        f"model={args.forward_backend}+{args.inverse_backend}, "
        f"fem_n_elec={args.fem_n_elec}, fem_grid={args.fem_grid}, "
        f"ridge={_format_float(args.ridge)}, holdout={args.holdout}, "
        f"raw_160={include_raw}, fit_methods={','.join(args.fit_methods)}, "
        f"n_measurements={model.n_measurements}, n_parameters={model.sigma_true.size}"
    )
    print(_format_table(case))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.field_output}")
    print(f"Wrote {args.point_output}")
    if args.structure_metrics:
        print(f"Wrote {args.structure_output}")
    print(f"Wrote {args.report_output}")
    if point_plot is not None:
        print(f"Wrote {point_plot}")
    print(f"Wrote {curve_plot}")
    print(f"Wrote {summary_plot}")
    if recon_plot is not None:
        print(f"Wrote {recon_plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
