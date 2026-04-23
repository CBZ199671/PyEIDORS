#!/usr/bin/env python3
"""Generate T24 visual audit index plots and confidence reports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pyeidors.data.visual_audit import run_visual_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build visual audit index panels for historical EIT precision "
            "experiments and mark unaudited legacy results as smoke-only."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing existing experiment outputs.",
    )
    parser.add_argument(
        "--audit-output-dir",
        type=Path,
        default=None,
        help="Directory for visual audit outputs; defaults to --output-dir.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional audit slugs to generate. Defaults to all T24 experiments.",
    )
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run = run_visual_audit(
        output_dir=args.output_dir,
        audit_output_dir=args.audit_output_dir,
        slugs=args.experiments,
        dpi=args.dpi,
    )
    print(f"Wrote CSV: {run.csv_path}")
    print(f"Wrote Markdown: {run.md_path}")
    print(f"Wrote index plot: {run.index_plot_path}")
    for path in run.experiment_plot_paths:
        print(f"Wrote experiment audit: {path}")
    print("task | slug | status | missing")
    print("-----+------+--------+--------")
    for row in run.rows:
        missing = ",".join(row.missing_required_visuals) or "-"
        print(f"{row.task_id} | {row.slug} | {row.audit_status} | {missing}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
