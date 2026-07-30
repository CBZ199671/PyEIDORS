#!/usr/bin/env python3
"""导出 X01 夹具和冻结 CSV 证据。 / Export the X01 fixture and CSV evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
for source_path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))


def _matrix_fraction_strings(matrix: Any) -> list[list[str]]:
    return [
        [str(matrix[row, column]) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def export_assets(suite_output: Path) -> dict[str, Path]:
    from scripts.benchmarks.cem_exact_extension_suite import (
        EXTENSION_CASES,
        prepare_extension_case_fixture,
        solve_exact_extension_case,
    )

    case = next(item for item in EXTENSION_CASES if item.case_id == "X01")
    manifest_path = suite_output / "suite_manifest.json"
    fixture: dict[str, Any] | None = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fixture = next(
            item for item in manifest["cases"] if item["case_id"] == case.case_id
        )
    if fixture is None:
        fixture = prepare_extension_case_fixture(suite_output, case)

    fixture_dir = PACKAGE_DIR / "fixtures" / case.case_id
    common_output = fixture_dir / "common_mesh"
    common_output.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for key, filename in (
        ("mat_path", "cem_exact_extension_p1.mat"),
        ("msh_path", "cem_exact_extension_p1.msh"),
        ("metadata_path", "cem_exact_extension_p1.json"),
    ):
        source = Path(fixture[key])
        destination = common_output / filename
        shutil.copy2(source, destination)
        copied[key] = destination

    reference = solve_exact_extension_case(case)
    reference_payload = {
        "schema": "cem-professor-portable-exact-reference-v1",
        "case_id": case.case_id,
        "exact_domain": reference["exact_domain"],
        "exact_linear_solver": reference["exact_linear_solver"],
        "truth_sha256": reference["truth_sha256"],
        "voltage": reference["truth_fraction_strings"],
        "reduced_map": _matrix_fraction_strings(reference["reduced_map"]),
        "reduced_rhs": _matrix_fraction_strings(reference["reduced_rhs"]),
        "certification": {
            "exact_classic_residual_zero": reference["exact_classic_residual_zero"],
            "exact_robin_residual_zero": reference["exact_robin_residual_zero"],
            "exact_classic_robin_identical": reference["exact_classic_robin_identical"],
            "exact_voltage_gauge_zero": True,
        },
        "note": (
            "All strings are exact fractions over QQ. This portable X01 "
            "reference is for teaching and cross-runtime metric reproduction."
        ),
    }
    reference_path = fixture_dir / "exact_reference.json"
    reference_path.write_text(
        json.dumps(reference_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    copied["exact_reference"] = reference_path

    expected_dir = PACKAGE_DIR / "expected"
    expected_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "cem_exact_extension_metrics.csv",
        "cem_exact_extension_timing.csv",
    ):
        source = suite_output / filename
        if not source.exists():
            raise FileNotFoundError(f"report evidence not found: {source}")
        destination = expected_dir / filename
        shutil.copy2(source, destination)
        copied[filename] = destination
    return copied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-output",
        type=Path,
        default=REPO_ROOT / "output" / "cem_exact_extension",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for label, path in export_assets(args.suite_output.resolve()).items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
