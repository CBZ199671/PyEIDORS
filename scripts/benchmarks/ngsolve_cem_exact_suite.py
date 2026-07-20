#!/usr/bin/env python3
"""Run every prepared rational exact-suite case with NGSolve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.ngsolve_cem_exact_case import run_case


def run_suite(suite_output: Path, *, timing_repeats: int) -> int:
    manifest_path = suite_output / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    for fixture in cases:
        case_dir = Path(fixture["case_dir"])
        run_case(
            Path(fixture["msh"]),
            Path(fixture["metadata_path"]),
            case_dir / "ngsolve_report.json",
            timing_repeats=timing_repeats,
        )
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-output", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=11)
    args = parser.parse_args()
    count = run_suite(
        args.suite_output.resolve(),
        timing_repeats=int(args.timing_repeats),
    )
    print(f"NGSolve exact CEM reports: {count} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
