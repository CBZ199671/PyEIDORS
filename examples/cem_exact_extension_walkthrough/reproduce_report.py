#!/usr/bin/env python3
"""从冻结 CSV 复现报告数字。 / Reproduce report numbers from frozen CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment_common import (
    load_csv_records,
    summarize_accuracy_records,
    summarize_timing_records,
)


PACKAGE_DIR = Path(__file__).resolve().parent
EXPECTED_DIR = PACKAGE_DIR / "expected"


def build_report_summary(
    metrics_path: Path,
    timing_path: Path,
) -> dict[str, Any]:
    accuracy = summarize_accuracy_records(load_csv_records(metrics_path))
    timing = summarize_timing_records(load_csv_records(timing_path))
    return {"accuracy": accuracy, "timing": timing}


def _print_summary(summary: dict[str, Any]) -> None:
    accuracy = summary["accuracy"]
    print("38 案例精度汇总 / 38-case accuracy summary")
    print(
        f"records={accuracy['record_count']} cases={accuracy['case_count']} "
        f"solvers={len(accuracy['solvers'])}"
    )
    for formulation in accuracy["formulations"]:
        print(f"\n{formulation}")
        for solver in accuracy["solvers"]:
            geometric_mean = accuracy["geometric_means"][formulation][solver]
            wins = accuracy["win_counts"][formulation][solver]
            print(f"  {solver:20s} GM={geometric_mean:.6e} wins={wins}/38")
        q4 = accuracy["q4_summary"][formulation]
        print(
            "  Q4 same order="
            f"{q4['same_order_all_cases']} order={' < '.join(q4['ordering'])}"
        )

    print("\n计时：Robin/Classic 几何平均比 / Timing: GM Robin/Classic ratio")
    for solver, phases in summary["timing"].items():
        values = " ".join(
            f"{phase}={item['geometric_mean_robin_over_classic']:.4f}"
            f" ({item['robin_faster_case_count']}/{item['case_count']} faster)"
            for phase, item in phases.items()
        )
        print(f"  {solver}: {values}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=EXPECTED_DIR / "cem_exact_extension_metrics.csv",
    )
    parser.add_argument(
        "--timing",
        type=Path,
        default=EXPECTED_DIR / "cem_exact_extension_timing.csv",
    )
    parser.add_argument("--json-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_report_summary(args.metrics.resolve(), args.timing.resolve())
    _print_summary(summary)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
