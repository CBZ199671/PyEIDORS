#!/usr/bin/env python3
"""Build a reproducible EIDORS/PyEIDORS Bridge v2 acceptance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from eit_app.interop import validate_bridge_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eidors-to-pyeidors-2d", type=Path, required=True)
    parser.add_argument("--eidors-to-pyeidors-3d", type=Path, required=True)
    parser.add_argument("--pyeidors-to-eidors-3d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _validate_package(path: Path) -> dict[str, Any]:
    report = validate_bridge_package(path)
    if not report["valid"]:
        raise RuntimeError(
            f"Bridge Package validation failed for {path}: {report['errors']}"
        )
    return report


def _run_import_smoke(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyeidors.interop",
            "import-geometry",
            str(path),
            "--forward-smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"PyEIDORS import smoke failed for {path}: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    bridge_schema = _read_json(
        repo_root / "schemas/interop/eidors_pyeidors_bridge_v2.schema.json"
    )
    geometry_schema = _read_json(
        repo_root / "schemas/interop/eidors_pyeidors_geometry_v2.schema.json"
    )

    eidors_to_pyeidors_2d = args.eidors_to_pyeidors_2d.resolve()
    eidors_to_pyeidors_3d = args.eidors_to_pyeidors_3d.resolve()
    pyeidors_to_eidors_3d = args.pyeidors_to_eidors_3d.resolve()
    package_reports = {
        "eidors_to_pyeidors_2d": _validate_package(eidors_to_pyeidors_2d),
        "eidors_to_pyeidors_3d": _validate_package(eidors_to_pyeidors_3d),
        "pyeidors_to_eidors_3d": _validate_package(pyeidors_to_eidors_3d),
    }
    import_reports = {
        "eidors_to_pyeidors_2d": _run_import_smoke(eidors_to_pyeidors_2d),
        "eidors_to_pyeidors_3d": _run_import_smoke(eidors_to_pyeidors_3d),
    }
    eidors_report = _read_json(pyeidors_to_eidors_3d / "eidors_import_report.json")

    checks = {
        "schemas_parse": (
            bridge_schema.get("type") == "object"
            and geometry_schema.get("type") == "object"
        ),
        "all_packages_valid": all(
            report["valid"] for report in package_reports.values()
        ),
        "real_2d_triangle_import": (
            import_reports["eidors_to_pyeidors_2d"].get("dimension") == 2
            and import_reports["eidors_to_pyeidors_2d"].get("mesh_family") == "triangle"
            and import_reports["eidors_to_pyeidors_2d"].get("forward_smoke") == "passed"
            and import_reports["eidors_to_pyeidors_2d"].get(
                "forward_measurements_finite"
            )
            is True
        ),
        "real_3d_tetrahedron_import": (
            import_reports["eidors_to_pyeidors_3d"].get("dimension") == 3
            and import_reports["eidors_to_pyeidors_3d"].get("mesh_family")
            == "tetrahedron"
            and import_reports["eidors_to_pyeidors_3d"].get("electrode_projection")
            == "exact_surface_nodes"
            and import_reports["eidors_to_pyeidors_3d"].get("forward_smoke") == "passed"
            and import_reports["eidors_to_pyeidors_3d"].get(
                "forward_measurements_finite"
            )
            is True
        ),
        "real_eidors_3d_reverse_import": (
            eidors_report.get("status") == "passed"
            and eidors_report.get("dimension") == 3
            and eidors_report.get("boundary_exact") is True
            and eidors_report.get("electrodes_exact") is True
            and eidors_report.get("protocol_exact") is True
            and eidors_report.get("forward_finite") is True
            and eidors_report.get("forward_count_exact") is True
        ),
    }
    report = {
        "schema": "eidors_pyeidors_bidirectional_acceptance_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "package_reports": package_reports,
        "pyeidors_import_reports": import_reports,
        "eidors_import_report": eidors_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
