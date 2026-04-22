#!/usr/bin/env python3
"""Run EIT end-to-end digit tests with ADC/noise injection."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from pyeidors.data.eit_digit_metrics import EITDigitSummary, summarize_eit_digit_sweep


CSV_FIELDS = [
    "bit",
    "ideal_decimal_digits",
    "voltage_rmse",
    "voltage_effective_digits",
    "sigma_rmse",
    "sigma_effective_digits",
    "hypothesis_delta_digits",
]


def _format_float(value: float | None) -> str:
    if value is None:
        return "none"
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _format_table(rows: list[EITDigitSummary]) -> str:
    rendered_rows = [
        [
            str(row.bit),
            _format_float(row.ideal_decimal_digits),
            _format_float(row.voltage_rmse),
            _format_float(row.voltage_effective_digits),
            _format_float(row.sigma_rmse),
            _format_float(row.sigma_effective_digits),
            _format_float(row.hypothesis_delta_digits),
        ]
        for row in rows
    ]
    widths = [
        max(len(CSV_FIELDS[idx]), *(len(row[idx]) for row in rendered_rows))
        for idx in range(len(CSV_FIELDS))
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(CSV_FIELDS)),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in rendered_rows
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[EITDigitSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run sigma_true -> forward -> ADC/noise -> inverse -> metrics. "
            "Default model is a deterministic linear EIT surrogate; real PyEIDORS "
            "solver hookup is reserved for the next integration step."
        ),
    )
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=[12, 16, 20, 24],
        help="ADC bit depths to test.",
    )
    parser.add_argument(
        "--full-scale",
        type=float,
        default=10.0,
        help="ADC full-scale range width in volts.",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="RMS-relative Gaussian noise level. Alias for --noise-relative.",
    )
    parser.add_argument(
        "--noise-relative",
        type=float,
        help="RMS-relative Gaussian noise level. Overrides --noise when set.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Absolute Gaussian noise standard deviation in volts.",
    )
    parser.add_argument(
        "--enob",
        type=float,
        help="Optional effective number of bits. Must be <= every requested bit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for voltage noise.",
    )
    parser.add_argument(
        "--model-seed",
        type=int,
        default=20260422,
        help="Random seed for deterministic surrogate sensitivity.",
    )
    parser.add_argument(
        "--n-measurements",
        type=int,
        default=16,
        help="Surrogate boundary-voltage measurement count.",
    )
    parser.add_argument(
        "--n-parameters",
        type=int,
        default=8,
        help="Surrogate conductivity parameter count.",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=1e-8,
        help="Ridge parameter for surrogate inverse solve.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eit_digits.csv"),
        help="CSV output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    noise_relative = args.noise if args.noise_relative is None else args.noise_relative
    rows = summarize_eit_digit_sweep(
        bits=args.bits,
        full_scale_range=args.full_scale,
        enob=args.enob,
        noise_std=args.noise_std,
        noise_relative=noise_relative,
        seed=args.seed,
        ridge=args.ridge,
        n_measurements=args.n_measurements,
        n_parameters=args.n_parameters,
        model_seed=args.model_seed,
    )
    _write_csv(args.output, rows)
    print(
        "settings: "
        "model=linear-surrogate, "
        f"full_scale={_format_float(args.full_scale)}, "
        f"enob={_format_float(args.enob)}, "
        f"noise_std={_format_float(args.noise_std)}, "
        f"noise_relative={_format_float(noise_relative)}, "
        f"seed={args.seed}, model_seed={args.model_seed}, ridge={_format_float(args.ridge)}"
    )
    print(_format_table(rows))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
