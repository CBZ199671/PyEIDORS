#!/usr/bin/env python3
"""Render T10 EIT digit plots from the combined T9 report CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pyeidors.data.digit_plot import plot_digit_report_csv
from pyeidors.runtime_paths import pyeidors_output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render voltage/sigma digit and hypothesis-delta PNG plots.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=pyeidors_output_path("eit_digit_report.csv"),
        help="Combined CSV from scripts/eit_digit_report.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=pyeidors_output_path("eit_digit_plot.png"),
        help="PNG output path.",
    )
    parser.add_argument(
        "--title",
        default="EIT digit hypothesis check",
        help="Figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = plot_digit_report_csv(
            input_csv=args.input_csv,
            output_path=args.output,
            title=args.title,
            dpi=args.dpi,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
