#!/usr/bin/env python3
"""Compare CPU and CUDA benchmark/parity reports and summarize speedup/parity fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_stage_timings(report: dict[str, Any]) -> float:
    timings = report.get("stage_timings")
    if isinstance(timings, dict) and timings:
        return float(sum(float(v) for v in timings.values()))
    stages = report.get("stages")
    if isinstance(stages, list):
        return float(sum(float(item.get("elapsed_sec", 0.0)) for item in stages if isinstance(item, dict)))
    return float("nan")


def _legacy_forward_heavy_seconds(report: dict[str, Any], *, kind: str) -> float:
    if kind == "2d":
        timings = report.get("stage_timings", {})
        if isinstance(timings, dict):
            return float(timings.get("forward_homogeneous", 0.0)) + float(timings.get("forward_phantom", 0.0))
        return float("nan")
    breakdown = report.get("stage_breakdown", {})
    total = 0.0
    if isinstance(breakdown, dict):
        absolute = breakdown.get("absolute", {})
        difference = breakdown.get("difference", {})
        if isinstance(absolute, dict):
            total += float(absolute.get("forward", 0.0))
        if isinstance(difference, dict):
            total += float(difference.get("forward_validate", 0.0))
    return total


def _backend(report: dict[str, Any]) -> dict[str, Any]:
    for key in ("perf_summary", "absolute_solver"):
        payload = report.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def _forward_probe_seconds(report: dict[str, Any]) -> float:
    backend = _backend(report)
    value = backend.get("forward_probe_elapsed_sec")
    if value is None:
        return float("nan")
    return float(value)


def _pick_forward_heavy_metric(report: dict[str, Any], *, kind: str) -> tuple[str, float]:
    if kind == "3d":
        probe = _forward_probe_seconds(report)
        if math.isfinite(probe) and probe > 0.0:
            return "absolute_forward_probe", probe
    return "legacy_aggregate", _legacy_forward_heavy_seconds(report, kind=kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True, help="CPU report JSON")
    parser.add_argument("--gpu", type=Path, required=True, help="GPU report JSON")
    parser.add_argument("--kind", choices=["2d", "3d"], required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cpu = _load(args.cpu)
    gpu = _load(args.gpu)

    cpu_total = _sum_stage_timings(cpu)
    gpu_total = _sum_stage_timings(gpu)
    forward_metric_label, cpu_forward = _pick_forward_heavy_metric(cpu, kind=str(args.kind))
    _, gpu_forward = _pick_forward_heavy_metric(gpu, kind=str(args.kind))
    cpu_forward_legacy = _legacy_forward_heavy_seconds(cpu, kind=str(args.kind))
    gpu_forward_legacy = _legacy_forward_heavy_seconds(gpu, kind=str(args.kind))

    total_speedup = float(cpu_total / gpu_total) if gpu_total > 0 else 0.0
    forward_speedup = float(cpu_forward / gpu_forward) if gpu_forward > 0 else 0.0
    legacy_forward_speedup = float(cpu_forward_legacy / gpu_forward_legacy) if gpu_forward_legacy > 0 else 0.0

    payload = {
        "kind": str(args.kind),
        "cpu_report": str(args.cpu),
        "gpu_report": str(args.gpu),
        "cpu_total_elapsed_sec": cpu_total,
        "gpu_total_elapsed_sec": gpu_total,
        "forward_heavy_metric": forward_metric_label,
        "cpu_forward_heavy_sec": cpu_forward,
        "gpu_forward_heavy_sec": gpu_forward,
        "cpu_forward_legacy_sec": cpu_forward_legacy,
        "gpu_forward_legacy_sec": gpu_forward_legacy,
        "total_speedup_x": total_speedup,
        "forward_heavy_speedup_x": forward_speedup,
        "forward_legacy_speedup_x": legacy_forward_speedup,
        "gpu_faster_total": bool(gpu_total < cpu_total),
        "gpu_faster_forward_heavy": bool(gpu_forward < cpu_forward),
        "gpu_faster_forward_legacy": bool(gpu_forward_legacy < cpu_forward_legacy),
        "cpu_backend": _backend(cpu),
        "gpu_backend": _backend(gpu),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
