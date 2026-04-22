#!/usr/bin/env python3
"""Run pure ADC quantization tests for boundary-voltage samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

from pyeidors.data.adc_quantization import (
    DEFAULT_BOUNDARY_VOLTAGES,
    ADCQuantizationSummary,
    summarize_adc_sweep,
)


CSV_FIELDS = [
    "bit",
    "ideal_decimal_digits",
    "full_scale",
    "lsb",
    "voltage_rmse",
    "voltage_effective_digits",
]


def _load_csv_column(path: Path, column: int) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if column < 0 or column >= data.shape[1]:
        raise ValueError(f"column {column} out of range for CSV shape {data.shape}")
    return np.asarray(data[:, column], dtype=float)


def _format_float(value: float) -> str:
    if np.isinf(value):
        return "inf"
    if np.isnan(value):
        return "nan"
    return f"{value:.12g}"


def _format_table(rows: list[ADCQuantizationSummary]) -> str:
    headers = CSV_FIELDS
    rendered_rows = [
        [
            str(row.bit),
            _format_float(row.ideal_decimal_digits),
            _format_float(row.full_scale),
            _format_float(row.lsb),
            _format_float(row.voltage_rmse),
            _format_float(row.voltage_effective_digits),
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rendered_rows))
        for idx in range(len(headers))
    ]
    lines = [
        " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    lines.extend(
        " | ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in rendered_rows
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[ADCQuantizationSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate ideal ADC quantization for boundary voltages.",
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
        default=10000.0,
        help="ADC full-scale range width in volts, e.g. 10 for +/-5 V.",
    )
    parser.add_argument(
        "--enob",
        type=float,
        help="Optional effective number of bits. Must be <= every requested bit.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Absolute Gaussian noise standard deviation in volts.",
    )
    parser.add_argument(
        "--noise-relative",
        type=float,
        default=0.0,
        help="RMS-relative Gaussian noise level, e.g. 0.001 for 0.1%.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for noise injection.",
    )
    parser.add_argument(
        "--voltages",
        nargs="+",
        type=float,
        help="Boundary-voltage samples. Defaults to the Word-table examples.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="Optional numeric CSV input. When set, voltages come from one column.",
    )
    parser.add_argument(
        "--column",
        type=int,
        default=0,
        help="Zero-based CSV column for --input-csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/adc_quant.csv"),
        help="CSV output path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_csv is not None:
        voltages = _load_csv_column(args.input_csv, args.column)
    elif args.voltages is not None:
        voltages = np.asarray(args.voltages, dtype=float)
    else:
        voltages = DEFAULT_BOUNDARY_VOLTAGES

    rows = summarize_adc_sweep(
        voltages,
        bits=args.bits,
        full_scale_range=args.full_scale,
        enob=args.enob,
        noise_std=args.noise_std,
        noise_relative=args.noise_relative,
        seed=args.seed,
    )
    _write_csv(args.output, rows)
    print(
        "settings: "
        f"full_scale={_format_float(args.full_scale)}, "
        f"enob={_format_float(args.enob) if args.enob is not None else 'nominal'}, "
        f"noise_std={_format_float(args.noise_std)}, "
        f"noise_relative={_format_float(args.noise_relative)}, "
        f"seed={args.seed}"
    )
    print(_format_table(rows))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
