#!/usr/bin/env python3
"""Render the 256 -> 208 -> 160 adjacent EIT point audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from pyeidors.data.holdout_point_audit import (
    POINT_AUDIT_FIELDS,
    HoldoutPointAuditRow,
    HoldoutPointAuditSummary,
    build_holdout_point_audit,
    plot_holdout_point_audit,
)


def _write_csv(path: Path, rows: list[HoldoutPointAuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _write_report(path: Path, summary: HoldoutPointAuditSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 16e holdout point audit",
        "",
        f"- full candidates: {summary.full_candidate_count}",
        f"- drive-related removed: {summary.drive_removed_count}",
        f"- kept before far3 holdout: {summary.kept_208_count}",
        f"- far3 holdout removed: {summary.holdout_far3_count}",
        f"- fit-train points after far3 holdout: {summary.fit_train_160_count}",
        f"- per frame: {summary.points_per_full_frame} -> "
        f"{summary.points_per_kept_frame} -> {summary.points_per_train_frame}",
        "",
        "For stim pair `(0,1)`: drive-related removed measurement pairs are "
        "`(15,0)`, `(0,1)`, `(1,2)`; far3 holdout pairs are "
        "`(7,8)`, `(8,9)`, `(9,10)`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_summary(summary: HoldoutPointAuditSummary) -> str:
    return "\n".join(
        [
            "point audit counts:",
            f"  full candidates          = {summary.full_candidate_count}",
            f"  drive-related removed    = {summary.drive_removed_count}",
            f"  kept before far3         = {summary.kept_208_count}",
            f"  far3 holdout             = {summary.holdout_far3_count}",
            f"  fit-train after far3     = {summary.fit_train_160_count}",
            "  per frame                = "
            f"{summary.points_per_full_frame} -> "
            f"{summary.points_per_kept_frame} -> "
            f"{summary.points_per_train_frame}",
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit 16-electrode adjacent measurement points: full 256 candidates, "
            "drive-related removal to 208, then far3 removal to 160."
        ),
    )
    parser.add_argument("--fem-n-elec", type=int, default=16, help="Electrode count.")
    parser.add_argument(
        "--holdout",
        choices=["far3"],
        default="far3",
        help="Holdout rule after drive-related measurements are removed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eit_holdout_voltage_points_16e.csv"),
        help="Point-audit CSV path.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("outputs/eit_holdout_voltage_points_16e.png"),
        help="Point-audit PNG path.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/eit_holdout_voltage_points_16e.md"),
        help="Short markdown summary path.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG plot DPI.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, summary = build_holdout_point_audit(
        n_elec=args.fem_n_elec,
        holdout=args.holdout,
    )
    _write_csv(args.output, rows)
    plot_path = plot_holdout_point_audit(
        rows,
        args.plot_output,
        n_elec=args.fem_n_elec,
        dpi=args.dpi,
    )
    _write_report(args.report_output, summary)
    print(_format_summary(summary))
    print(f"Wrote {args.output}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
