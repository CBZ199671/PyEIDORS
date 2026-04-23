#!/usr/bin/env python3
"""Run controlled voltage significant-digit sweeps for EIT reconstruction."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from pyeidors.data.voltage_digit_sweep import (
    VoltageDigitFieldRow,
    VoltageDigitSweepSummary,
    plot_voltage_digit_sweep,
    run_voltage_digit_sweep_from_backend,
)


SUMMARY_FIELDS = [
    "target_voltage_digits",
    "achieved_voltage_effective_digits",
    "voltage_rmse",
    "sigma_rmse",
    "sigma_relative_rmse",
    "sigma_mae",
    "sigma_max_abs_error",
    "sigma_effective_digits",
]

FIELD_FIELDS = [
    "target_voltage_digits",
    "cell_index",
    "sigma_true",
    "sigma_recon",
    "sigma_error",
    "abs_sigma_error",
]


def _format_float(value: float | None) -> str:
    if value is None:
        return "none"
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _format_table(rows: list[VoltageDigitSweepSummary]) -> str:
    rendered_rows = [
        [
            str(row.target_voltage_digits),
            _format_float(row.achieved_voltage_effective_digits),
            _format_float(row.voltage_rmse),
            _format_float(row.sigma_rmse),
            _format_float(row.sigma_relative_rmse),
            _format_float(row.sigma_mae),
            _format_float(row.sigma_max_abs_error),
            _format_float(row.sigma_effective_digits),
        ]
        for row in rows
    ]
    widths = [
        max(len(SUMMARY_FIELDS[idx]), *(len(row[idx]) for row in rendered_rows))
        for idx in range(len(SUMMARY_FIELDS))
    ]
    lines = [
        " | ".join(
            header.ljust(widths[idx]) for idx, header in enumerate(SUMMARY_FIELDS)
        ),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in rendered_rows
    )
    return "\n".join(lines)


def _write_summary_csv(path: Path, rows: list[VoltageDigitSweepSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _write_field_csv(path: Path, rows: list[VoltageDigitFieldRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one EIT linearized model, keep boundary voltages at target "
            "decimal significant digits, reconstruct conductivity, and report "
            "distribution errors."
        ),
    )
    parser.add_argument(
        "--target-digits",
        nargs="+",
        type=int,
        default=[4, 5, 6, 7],
        help="Target decimal significant digits for boundary voltages.",
    )
    parser.add_argument(
        "--digit-method",
        choices=["truncate", "round"],
        default="truncate",
        help="How to keep significant digits. Default matches the Word table.",
    )
    parser.add_argument(
        "--forward-backend",
        choices=["surrogate", "pyeidors-fem"],
        default="pyeidors-fem",
        help="Forward model backend.",
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
        help="Random seed for deterministic surrogate sensitivity.",
    )
    parser.add_argument(
        "--fem-n-elec",
        type=int,
        default=16,
        help="PyEIDORS FEM electrode count.",
    )
    parser.add_argument(
        "--fem-grid",
        type=int,
        default=4,
        help="PyEIDORS FEM unit-square grid.",
    )
    parser.add_argument(
        "--expected-fem-measurements",
        type=int,
        help="Expected PyEIDORS FEM measurement count; checked before reporting.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-2,
        help="Inverse regularization parameter. Used as RM lambda for pyeidors-rm.",
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
        "--output",
        type=Path,
        default=Path("outputs/eit_voltage_digit_sweep_16e.csv"),
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--field-output",
        type=Path,
        default=Path("outputs/eit_voltage_digit_fields_16e.csv"),
        help="Per-cell conductivity error CSV output path.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("outputs/eit_voltage_digit_sweep_16e.png"),
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
    model, summaries, field_rows = run_voltage_digit_sweep_from_backend(
        target_digits=args.target_digits,
        forward_backend=args.forward_backend,
        n_measurements=args.n_measurements,
        n_parameters=args.n_parameters,
        model_seed=args.model_seed,
        fem_n_elec=args.fem_n_elec,
        fem_grid=args.fem_grid,
        expected_fem_measurements=args.expected_fem_measurements,
        ridge=args.ridge,
        inverse_backend=args.inverse_backend,
        rm_mode=args.rm_mode,
        rm_form=args.rm_form,
        digit_method=args.digit_method,
    )

    _write_summary_csv(args.output, summaries)
    _write_field_csv(args.field_output, field_rows)
    plot_path = plot_voltage_digit_sweep(
        summaries,
        args.plot_output,
        title="16e adjacent voltage digit sweep"
        if args.forward_backend == "pyeidors-fem"
        else "Surrogate voltage digit sweep",
        dpi=args.dpi,
    )

    print(
        "settings: "
        f"model={args.forward_backend}+{args.inverse_backend}, "
        f"target_digits={','.join(str(value) for value in args.target_digits)}, "
        f"digit_method={args.digit_method}, "
        f"ridge={_format_float(args.ridge)}, "
        f"fem_n_elec={args.fem_n_elec}, fem_grid={args.fem_grid}, "
        f"stim_pattern={model.stim_pattern or '{ad}'}, "
        f"meas_pattern={model.meas_pattern or '{ad}'}, "
        f"n_measurements={model.n_measurements}, "
        f"n_parameters={model.sigma_true.size}, "
        f"rm_mode={args.rm_mode}, rm_form={args.rm_form}"
    )
    print(_format_table(summaries))
    print(f"Wrote {args.output}")
    print(f"Wrote {args.field_output}")
    print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
