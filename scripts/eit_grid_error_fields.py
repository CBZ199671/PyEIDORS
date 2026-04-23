#!/usr/bin/env python3
"""Plot conductivity true/recon/error fields for FEM grid comparisons."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from pyeidors.data.grid_error_fields import (
    FIELD_FIELDS,
    SUMMARY_FIELDS,
    GridErrorCase,
    format_grid_error_report,
    plot_grid_error_fields,
    run_grid_error_fields,
)


def _format_float(value: float | None) -> str:
    if value is None:
        return "none"
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _write_summary_csv(path: Path, cases: list[GridErrorCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for case in cases:
            writer.writerow(case.summary.as_csv_row())


def _write_field_csv(path: Path, cases: list[GridErrorCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_FIELDS)
        writer.writeheader()
        for case in cases:
            for row in case.field_rows:
                writer.writerow(row.as_csv_row())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_table(cases: list[GridErrorCase]) -> str:
    fields = [
        "fem_grid",
        "n_parameters",
        "sigma_relative_rmse",
        "sigma_effective_digits",
        "sigma_mae",
        "sigma_max_abs_error",
        "max_abs_error_cell",
    ]
    rendered = [
        [
            str(case.summary.fem_grid),
            str(case.summary.n_parameters),
            _format_float(case.summary.sigma_relative_rmse),
            _format_float(case.summary.sigma_effective_digits),
            _format_float(case.summary.sigma_mae),
            _format_float(case.summary.sigma_max_abs_error),
            str(case.summary.max_abs_error_cell_index),
        ]
        for case in cases
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
            "Build each FEM grid independently, reconstruct conductivity under "
            "the same voltage-digit and inverse settings, then plot true, "
            "reconstructed, and error fields."
        ),
    )
    parser.add_argument(
        "--fem-grid-levels",
        nargs="+",
        type=int,
        default=[4, 6, 8],
        help="FEM grid levels to plot.",
    )
    parser.add_argument(
        "--forward-backend",
        choices=["surrogate", "pyeidors-fem"],
        default="pyeidors-fem",
        help="Forward model backend.",
    )
    parser.add_argument(
        "--fem-n-elec",
        type=int,
        default=16,
        help="PyEIDORS FEM electrode count.",
    )
    parser.add_argument(
        "--expected-fem-measurements",
        type=int,
        help="Expected PyEIDORS FEM measurement count.",
    )
    parser.add_argument(
        "--target-digits",
        type=int,
        default=6,
        help="Target voltage significant digits.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-2,
        help="Inverse regularization parameter.",
    )
    parser.add_argument(
        "--inverse-backend",
        choices=["pyeidors-rm", "least-squares"],
        default="pyeidors-rm",
        help="Inverse backend for sigma reconstruction.",
    )
    parser.add_argument(
        "--rm-mode",
        choices=["tikhonov", "noser"],
        default="tikhonov",
        help="PyEIDORS RM regularization mode.",
    )
    parser.add_argument(
        "--rm-form",
        choices=["param", "measurement"],
        default="param",
        help="PyEIDORS RM construction form.",
    )
    parser.add_argument(
        "--n-measurements",
        type=int,
        default=16,
        help="Surrogate measurement count.",
    )
    parser.add_argument(
        "--n-parameters",
        type=int,
        default=8,
        help="Surrogate conductivity parameter count.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=20260422,
        help="Surrogate model seed.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/eit_grid_error_summary_16e.csv"),
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=Path("outputs/eit_grid_error_fields_16e.csv"),
        help="Per-cell field CSV output path.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/eit_grid_error_fields_16e.md"),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("outputs/eit_grid_error_fields_16e.png"),
        help="PNG plot output path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG plot DPI.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = run_grid_error_fields(
        fem_grid_levels=args.fem_grid_levels,
        forward_backend=args.forward_backend,
        n_elec=args.fem_n_elec,
        expected_measurements=args.expected_fem_measurements,
        ridge=args.ridge,
        target_voltage_digits=args.target_digits,
        inverse_backend=args.inverse_backend,
        rm_mode=args.rm_mode,
        rm_form=args.rm_form,
        n_measurements=args.n_measurements,
        n_parameters=args.n_parameters,
        model_seed=args.model_seed,
    )
    _write_summary_csv(args.summary_output, cases)
    _write_field_csv(args.field_output, cases)
    report = format_grid_error_report(cases)
    _write_text(args.report_output, report)
    plot_path = plot_grid_error_fields(
        cases,
        args.plot_output,
        title="16e grid conductivity error fields"
        if args.forward_backend == "pyeidors-fem"
        else "Surrogate grid conductivity error fields",
        dpi=args.dpi,
    )

    print(
        "settings: "
        f"model={args.forward_backend}+{args.inverse_backend}, "
        f"fem_n_elec={args.fem_n_elec}, "
        f"fem_grid_levels={','.join(str(value) for value in args.fem_grid_levels)}, "
        f"target_digits={args.target_digits}, "
        f"ridge={_format_float(args.ridge)}, "
        "stim_pattern={ad}, meas_pattern={ad}"
    )
    print(_format_table(cases))
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.field_output}")
    print(f"Wrote {args.report_output}")
    print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
