#!/usr/bin/env python3
"""Compare benchmark snapshots and enforce performance gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, Tuple


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline metrics JSON.")
    parser.add_argument("--optimized", type=Path, required=True, help="Optimized metrics JSON.")
    parser.add_argument("--report", type=Path, required=True, help="Markdown report output path.")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.50,
        help="Required median improvement ratio ((baseline-optimized)/baseline). 0.50 == 2x speedup.",
    )
    parser.add_argument("--max-regression", type=float, default=0.05, help="Allowed worst-case regression ratio.")
    return parser.parse_args()


def _load_cases(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "cases" in data:
        return {str(k): float(v) for k, v in data["cases"].items()}
    return {str(k): float(v) for k, v in data.items()}


def _shared_cases(
    baseline: Dict[str, float],
    optimized: Dict[str, float],
) -> Iterable[Tuple[str, float, float]]:
    for name in sorted(set(baseline) & set(optimized)):
        yield name, baseline[name], optimized[name]


def _improvement_ratio(base: float, opt: float) -> float:
    if base <= 0:
        return 0.0
    return (base - opt) / base


def _render_report(
    rows: Iterable[Tuple[str, float, float, float]],
    median_improvement: float,
    worst_regression: float,
    out: Path,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Performance Guard Report",
        "",
        f"- Median improvement: `{median_improvement:.2%}`",
        f"- Median speedup: `{(1.0 / max(1e-12, 1.0 - median_improvement)):.2f}x`",
        f"- Worst regression: `{worst_regression:.2%}`",
        "",
        "| Case | Baseline (s) | Optimized (s) | Improvement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, base, opt, ratio in rows:
        lines.append(f"| {name} | {base:.6f} | {opt:.6f} | {ratio:.2%} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    baseline = _load_cases(args.baseline)
    optimized = _load_cases(args.optimized)

    rows = []
    for name, base, opt in _shared_cases(baseline, optimized):
        rows.append((name, base, opt, _improvement_ratio(base, opt)))

    if not rows:
        raise SystemExit("No overlapping benchmark cases found between baseline and optimized snapshots.")

    improvements = [row[3] for row in rows]
    median_improvement = float(median(improvements))
    worst_regression = min(improvements)
    _render_report(rows, median_improvement, worst_regression, args.report)

    if median_improvement < args.min_improvement:
        print(
            "FAIL: median improvement below threshold "
            f"({median_improvement:.2%} < {args.min_improvement:.2%})"
        )
        return 1

    if worst_regression < -args.max_regression:
        print(
            "FAIL: worst-case regression exceeds threshold "
            f"({-worst_regression:.2%} > {args.max_regression:.2%})"
        )
        return 1

    print(
        "PASS: performance guard satisfied "
        f"(median improvement {median_improvement:.2%}, worst regression {worst_regression:.2%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
