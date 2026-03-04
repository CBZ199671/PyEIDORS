#!/usr/bin/env python3
"""Collect deterministic performance snapshots for CI perf guard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from statistics import median
from typing import Dict, List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--work-dir", type=Path, default=Path("test_results/perf"))
    return parser.parse_args()


def _run_once(cmd: List[str], env: Dict[str, str]) -> float:
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=env)
    return time.perf_counter() - t0


def _timed_case(
    name: str,
    cmd: List[str],
    repeat: int,
    env: Dict[str, str],
) -> float:
    samples = [_run_once(cmd, env) for _ in range(max(repeat, 1))]
    value = float(median(samples))
    print(f"{name}: median {value:.4f}s from {len(samples)} run(s)")
    return value


def _read_single_warm_metric(csv_path: Path) -> float:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one warm row in {csv_path}, got {len(rows)}")
    warm_raw = rows[0].get("warm_sec")
    if warm_raw is None or warm_raw == "":
        raise RuntimeError(f"Missing warm_sec in {csv_path}")
    warm = float(warm_raw)
    if warm <= 0:
        raise RuntimeError(f"Non-positive warm_sec in {csv_path}: {warm}")
    return warm


def main() -> int:
    args = _parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    mode_dir = root / args.mode
    if mode_dir.exists():
        shutil.rmtree(mode_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)

    python = str(Path(".venv/bin/python"))
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    diff_csv = mode_dir / "difference_runtime.csv"
    parity_dir = mode_dir / "synthetic_parity"

    if args.mode == "baseline":
        diff_solver = "parameter"
        parity_solver = "gauss-newton"
    else:
        diff_solver = "measurement"
        parity_solver = "single-step"

    diff_repeat = 20
    diff_cmd = [
        python,
        "scripts/benchmarks/benchmark_difference_runtime.py",
        "--refinements",
        "9",
        "--repeat",
        str(diff_repeat),
        "--measure-warm",
        "--single-step-space",
        diff_solver,
        "--csv-out",
        str(diff_csv),
    ]

    parity_cmd = [
        python,
        "scripts/run_synthetic_parity.py",
        "--output-root",
        str(parity_dir),
        "--mode",
        "difference",
        "--difference-solver",
        parity_solver,
        "--refinement",
        "8",
    ]
    if parity_solver == "gauss-newton":
        parity_cmd.extend(["--difference-max-iterations", "2"])

    _timed_case(
        "benchmark_difference_runtime (command)",
        diff_cmd,
        args.repeat,
        env,
    )
    diff_metric = _read_single_warm_metric(diff_csv)
    print(f"benchmark_difference_runtime (warm solve): {diff_metric:.4f}s")

    cases = {
        "benchmark_difference_runtime": diff_metric,
        "run_synthetic_parity": _timed_case(
            "run_synthetic_parity",
            parity_cmd,
            args.repeat,
            env,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cases": cases}, indent=2), encoding="utf-8")
    print(f"Wrote snapshot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
