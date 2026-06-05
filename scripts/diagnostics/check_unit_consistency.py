#!/usr/bin/env python3
"""Run unit consistency prechecks for PyEIDORS experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import sys

if sys.platform == "darwin":
    # Keep diagnostics stable on macOS when gmsh/OpenMP runtimes overlap.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.core_system import EITSystem
from pyeidors.data.structures import PatternConfig
from pyeidors.perf import DEFAULT_ACCELERATION_PROFILE
from pyeidors.runtime_paths import pyeidors_cache_path
from scripts.common.acceleration_profiles import add_acceleration_profile_argument


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh-source", choices=["cache", "generated"], default="cache"
    )
    parser.add_argument(
        "--mesh-dir", type=str, default=str(pyeidors_cache_path("eit_meshes"))
    )
    parser.add_argument("--mesh-name", type=str, default=None)
    parser.add_argument("--n-elec", type=int, default=16)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--mesh-size", type=float, default=0.1)
    parser.add_argument(
        "--drive-mode",
        choices=["line_current_density", "total_current", "normalized"],
        default="line_current_density",
    )
    parser.add_argument("--drive-value", type=float, default=5e-5)
    parser.add_argument("--geometry-scale-to-m", type=float, default=1.0)
    parser.add_argument("--contact-impedance", type=float, default=1e-5)
    parser.add_argument("--expected-domain-size-m", type=float, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero on ERROR checks.",
    )
    add_acceleration_profile_argument(
        parser,
        default=DEFAULT_ACCELERATION_PROFILE,
        help_suffix="For this diagnostics script the profile is accepted mainly for CLI consistency.",
    )
    return parser.parse_args()


def _format_table(rows: list[tuple[str, str, str]]) -> str:
    headers = ("CHECK", "LEVEL", "MESSAGE")
    all_rows = [headers, *rows]
    widths = [max(len(r[i]) for r in all_rows) for i in range(3)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep]
    for idx, row in enumerate(all_rows):
        lines.append(
            "| " + " | ".join(f"{row[i]:<{widths[i]}}" for i in range(3)) + " |"
        )
        if idx == 0:
            lines.append(sep)
    lines.append(sep)
    return "\n".join(lines)


def _report_to_json(report) -> dict[str, Any]:
    return {
        "has_errors": report.has_errors,
        "items": [
            {
                "name": item.name,
                "level": item.level.value,
                "passed": item.passed,
                "message": item.message,
                "details": item.details,
            }
            for item in report.items
        ],
    }


def main() -> int:
    args = parse_args()

    pattern = PatternConfig(
        n_elec=args.n_elec,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode=args.drive_mode,
        drive_value=float(args.drive_value),
        geometry_scale_to_m=float(args.geometry_scale_to_m),
        use_meas_current=False,
        rotate_meas=True,
    )
    try:
        system = EITSystem(
            n_elec=args.n_elec,
            pattern_config=pattern,
            contact_impedance=np.full(
                args.n_elec, float(args.contact_impedance), dtype=float
            ),
            acceleration_profile=str(args.acceleration_profile),
        )
        if args.mesh_source == "cache":
            system.setup(
                mesh_source="cache", mesh_dir=args.mesh_dir, mesh_name=args.mesh_name
            )
        else:
            system.setup(
                mesh_source="generated", radius=args.radius, mesh_size=args.mesh_size
            )

        report = system.run_unit_precheck(
            expected_domain_size_m=args.expected_domain_size_m,
            strict=False,
        )
    except Exception as exc:
        rows = [("drive_config_validity", "ERROR", str(exc))]
        print(_format_table(rows))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(
                    {
                        "has_errors": True,
                        "items": [
                            {
                                "name": "drive_config_validity",
                                "level": "ERROR",
                                "passed": False,
                                "message": str(exc),
                                "details": {},
                            }
                        ],
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"\nWrote JSON report: {args.json_out}")
        return 1

    rows = [(item.name, item.level.value, item.message) for item in report.items]
    print(_format_table(rows))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(_report_to_json(report), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote JSON report: {args.json_out}")

    if args.strict and report.has_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
