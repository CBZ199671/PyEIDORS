#!/usr/bin/env python3
"""Generate Markdown/CSV report tables from EIT digit CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pyeidors.data.digit_report import (
    DigitReportCase,
    format_markdown_report,
    read_eit_digit_cases,
    write_report_files,
)


def _expand(values: list | None, *, count: int, default, name: str) -> list:
    if not values:
        return [default for _ in range(count)]
    if len(values) != count:
        raise ValueError(f"--{name} count must match --input count")
    return values


def _parse_enob(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "nominal", "null"}:
        return None
    parsed = float(text)
    if parsed < 0.0:
        raise ValueError("--enob must be non-negative, none, or nominal")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a report table from one or more EIT digit CSV files.",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        required=True,
        help="Input CSV from scripts/eit_end_to_end_test.py. Repeat per scenario.",
    )
    parser.add_argument(
        "--label",
        action="append",
        help="Scenario label. Defaults to each input file stem.",
    )
    parser.add_argument(
        "--full-scale",
        action="append",
        type=float,
        required=True,
        help="ADC full-scale range for each input. Repeat per scenario.",
    )
    parser.add_argument(
        "--enob",
        action="append",
        help="ENOB for each input, or 'nominal'. Repeat per scenario if used.",
    )
    parser.add_argument(
        "--noise-std",
        action="append",
        type=float,
        help="Absolute voltage noise std for each input. Defaults to 0.",
    )
    parser.add_argument(
        "--noise-relative",
        action="append",
        type=float,
        help="Relative RMS voltage noise for each input. Defaults to 0.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/eit_digit_report.md"),
        help="Markdown report path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/eit_digit_report.csv"),
        help="Combined CSV report path.",
    )
    parser.add_argument(
        "--title",
        default="EIT digit report table",
        help="Markdown report title.",
    )
    return parser.parse_args(argv)


def _build_cases(args: argparse.Namespace) -> list[DigitReportCase]:
    count = len(args.input)
    labels = _expand(
        args.label,
        count=count,
        default=None,
        name="label",
    )
    full_scales = _expand(
        args.full_scale,
        count=count,
        default=None,
        name="full-scale",
    )
    enobs = _expand(args.enob, count=count, default=None, name="enob")
    noise_stds = _expand(
        args.noise_std,
        count=count,
        default=0.0,
        name="noise-std",
    )
    noise_relatives = _expand(
        args.noise_relative,
        count=count,
        default=0.0,
        name="noise-relative",
    )

    return [
        DigitReportCase(
            label=str(labels[idx] or args.input[idx].stem),
            path=args.input[idx],
            full_scale=float(full_scales[idx]),
            enob=_parse_enob(enobs[idx]),
            noise_std=float(noise_stds[idx]),
            noise_relative=float(noise_relatives[idx]),
        )
        for idx in range(count)
    ]


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        rows = read_eit_digit_cases(_build_cases(args))
        write_report_files(
            rows=rows,
            markdown_path=args.output_md,
            csv_path=args.output_csv,
            title=args.title,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_markdown_report(rows, title=args.title))
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
