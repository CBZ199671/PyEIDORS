#!/usr/bin/env python3
"""Render the T22 dense circular bucket domain audit."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from pyeidors.data.bucket_domain_audit import (
    BUCKET_DOMAIN_AUDIT_FIELDS,
    BucketDomainAuditRow,
    build_bucket_domain_audit_rows,
    build_circle_bucket_domain,
    format_bucket_domain_report,
    plot_bucket_domain_audit,
)


def _write_csv(path: Path, rows: list[BucketDomainAuditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BUCKET_DOMAIN_AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _format_summary(rows: list[BucketDomainAuditRow]) -> str:
    first = rows[0]
    return "\n".join(
        [
            "bucket domain audit:",
            f"  domain          = {first.domain}",
            f"  radius          = {first.bucket_radius}",
            f"  n_elec          = {first.n_elec}",
            f"  mesh_h          = {first.mesh_h}",
            f"  n_nodes         = {first.n_nodes}",
            f"  n_cells/n_dofs  = {first.n_cells}",
            f"  n_measurements  = {first.n_measurements}",
            f"  electrode arc   = {first.electrode_arc_length}",
            f"  electrode width = {first.electrode_width}",
        ]
    )


def _parse_xy(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected 'x,y'")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected finite numeric 'x,y'") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a dense circular bucket domain: disk mesh, equally spaced "
            "boundary electrode arcs, and the conductivity truth inclusion."
        ),
    )
    parser.add_argument(
        "--domain",
        choices=["circle_bucket"],
        default="circle_bucket",
        help="Domain geometry to audit.",
    )
    parser.add_argument(
        "--bucket-radius",
        type=float,
        default=1.0,
        help="Circular bucket radius in physical coordinates.",
    )
    parser.add_argument(
        "--n-elec",
        type=int,
        default=16,
        help="Number of equally spaced boundary electrodes.",
    )
    parser.add_argument(
        "--mesh-h",
        type=float,
        default=0.05,
        help="Target triangular-lattice spacing for the dense audit mesh.",
    )
    parser.add_argument(
        "--electrode-coverage",
        type=float,
        default=0.5,
        help="Fraction of each angular electrode slot covered by the arc.",
    )
    parser.add_argument(
        "--anomaly-center",
        type=_parse_xy,
        default=(0.35, 0.2),
        help="Conductivity anomaly center as 'x,y'.",
    )
    parser.add_argument(
        "--anomaly-radius",
        type=float,
        default=0.22,
        help="Circular anomaly radius.",
    )
    parser.add_argument(
        "--background-conductivity",
        type=float,
        default=1.0,
        help="Background conductivity value.",
    )
    parser.add_argument(
        "--anomaly-conductivity",
        type=float,
        default=1.15,
        help="Anomaly conductivity value.",
    )
    parser.add_argument(
        "--allow-coarse-smoke",
        action="store_true",
        help="Allow meshes below the T22 dense threshold for explicit smokes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/eit_bucket_domain_audit_16e.csv"),
        help="Domain/electrode audit CSV path.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("outputs/eit_bucket_domain_audit_16e.png"),
        help="Domain/electrode audit PNG path.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/eit_bucket_domain_audit_16e.md"),
        help="Short markdown audit report path.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Write the visual domain audit PNG.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="PNG plot DPI.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bucket = build_circle_bucket_domain(
        domain=args.domain,
        bucket_radius=args.bucket_radius,
        n_elec=args.n_elec,
        mesh_h=args.mesh_h,
        electrode_coverage=args.electrode_coverage,
        anomaly_center=args.anomaly_center,
        anomaly_radius=args.anomaly_radius,
        background_conductivity=args.background_conductivity,
        anomaly_conductivity=args.anomaly_conductivity,
        allow_coarse_smoke=args.allow_coarse_smoke,
    )
    rows = build_bucket_domain_audit_rows(bucket)
    _write_csv(args.output, rows)
    _write_text(args.report_output, format_bucket_domain_report(bucket))
    print(_format_summary(rows))
    print(f"Wrote {args.output}")
    if args.plot:
        plot_path = plot_bucket_domain_audit(bucket, args.plot_output, dpi=args.dpi)
        print(f"Wrote {plot_path}")
    print(f"Wrote {args.report_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
