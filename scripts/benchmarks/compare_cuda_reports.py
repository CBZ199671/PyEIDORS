#!/usr/bin/env python3
"""Compare CPU and CUDA benchmark/parity reports and summarize speedup/parity fields."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_stage_timings(report: dict[str, Any]) -> float:
    timings = report.get("stage_timings")
    if isinstance(timings, dict) and timings:
        return float(sum(float(v) for v in timings.values()))
    stages = report.get("stages")
    if isinstance(stages, list):
        return float(
            sum(
                float(item.get("elapsed_sec", 0.0))
                for item in stages
                if isinstance(item, dict)
            )
        )
    return float("nan")


def _legacy_forward_heavy_seconds(report: dict[str, Any], *, kind: str) -> float:
    if kind == "2d":
        timings = report.get("stage_timings", {})
        if isinstance(timings, dict):
            return float(timings.get("forward_homogeneous", 0.0)) + float(
                timings.get("forward_phantom", 0.0)
            )
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


def _forward_solver_benchmark(report: dict[str, Any]) -> dict[str, Any]:
    payload = report.get("forward_solver_benchmark")
    return payload if isinstance(payload, dict) else {}


def _decode_array_payload(payload: Any) -> np.ndarray | None:
    if not isinstance(payload, dict):
        return None
    shape = payload.get("shape")
    if not isinstance(shape, list):
        return None
    if "real" in payload or "imag" in payload:
        real = np.asarray(payload.get("real", []), dtype=np.float64)
        imag = np.asarray(payload.get("imag", []), dtype=np.float64)
        if real.shape != imag.shape:
            return None
        return (real + 1j * imag).reshape(tuple(int(v) for v in shape))
    if "values" in payload:
        return np.asarray(payload.get("values", []), dtype=np.float64).reshape(
            tuple(int(v) for v in shape)
        )
    return None


def _relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = float(np.linalg.norm(reference.reshape(-1)))
    if denom == 0.0:
        denom = np.finfo(np.float64).eps
    return float(np.linalg.norm((candidate - reference).reshape(-1)) / denom)


def _forward_output_parity(
    cpu_report: dict[str, Any],
    gpu_report: dict[str, Any],
) -> dict[str, Any]:
    cpu_forward = _forward_solver_benchmark(cpu_report)
    gpu_forward = _forward_solver_benchmark(gpu_report)
    cpu_values = _decode_array_payload(cpu_forward.get("electrode_voltages"))
    gpu_values = _decode_array_payload(gpu_forward.get("electrode_voltages"))
    if cpu_values is None or gpu_values is None:
        return {"available": False, "reason": "missing_forward_electrode_voltages"}
    if cpu_values.shape != gpu_values.shape:
        return {
            "available": False,
            "reason": "shape_mismatch",
            "cpu_shape": list(cpu_values.shape),
            "gpu_shape": list(gpu_values.shape),
        }
    diff = gpu_values - cpu_values
    abs_diff = np.abs(diff.reshape(-1))
    return {
        "available": True,
        "relative_l2": _relative_l2(cpu_values, gpu_values),
        "max_abs": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "rms_abs": float(np.sqrt(np.mean(abs_diff**2))) if abs_diff.size else 0.0,
        "cpu_dtype": str(cpu_values.dtype),
        "gpu_dtype": str(gpu_values.dtype),
        "shape": list(cpu_values.shape),
    }


def _forward_probe_seconds(report: dict[str, Any]) -> float:
    backend = _backend(report)
    value = backend.get("warm_forward_total_sec")
    if value is None:
        value = backend.get("forward_probe_elapsed_sec")
    if value is None:
        return float("nan")
    return float(value)


def _warm_forward_avg_seconds(report: dict[str, Any]) -> float:
    backend = _backend(report)
    value = backend.get("warm_forward_avg_sec")
    if value is None:
        repeats = backend.get("forward_probe_repeats")
        probe = backend.get("forward_probe_elapsed_sec")
        if (
            isinstance(repeats, (int, float))
            and float(repeats) > 0
            and isinstance(probe, (int, float))
        ):
            return float(probe) / float(repeats)
        return float("nan")
    return float(value)


def _first_forward_seconds(report: dict[str, Any]) -> float:
    backend = _backend(report)
    value = backend.get("first_forward_elapsed_sec")
    if value is None:
        return float("nan")
    return float(value)


def _absolute_reconstruct_seconds(report: dict[str, Any]) -> float:
    backend = _backend(report)
    value = backend.get("absolute_reconstruct_elapsed_sec")
    if value is None:
        return float("nan")
    return float(value)


def _difference_total_seconds(report: dict[str, Any]) -> float:
    solver = report.get("difference_solver", {})
    if not isinstance(solver, dict):
        return float("nan")
    cold = solver.get("difference_context_cold_elapsed_sec")
    warm = solver.get("difference_context_warm_elapsed_sec")
    reconstruct = solver.get("difference_reconstruct_elapsed_sec")
    if not all(isinstance(value, (int, float)) for value in (cold, warm, reconstruct)):
        return float("nan")
    return float(cold) + float(warm) + float(reconstruct)


def _pick_forward_heavy_metric(
    report: dict[str, Any], *, kind: str
) -> tuple[str, float]:
    if kind == "3d":
        warm_avg = _warm_forward_avg_seconds(report)
        if math.isfinite(warm_avg) and warm_avg > 0.0:
            return "absolute_warm_forward_avg", warm_avg
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
    forward_metric_label, cpu_forward = _pick_forward_heavy_metric(
        cpu, kind=str(args.kind)
    )
    _, gpu_forward = _pick_forward_heavy_metric(gpu, kind=str(args.kind))
    cpu_first_forward = _first_forward_seconds(cpu)
    gpu_first_forward = _first_forward_seconds(gpu)
    cpu_warm_forward_avg = _warm_forward_avg_seconds(cpu)
    gpu_warm_forward_avg = _warm_forward_avg_seconds(gpu)
    cpu_forward_legacy = _legacy_forward_heavy_seconds(cpu, kind=str(args.kind))
    gpu_forward_legacy = _legacy_forward_heavy_seconds(gpu, kind=str(args.kind))
    cpu_absolute_reconstruct = _absolute_reconstruct_seconds(cpu)
    gpu_absolute_reconstruct = _absolute_reconstruct_seconds(gpu)
    cpu_difference_total = _difference_total_seconds(cpu)
    gpu_difference_total = _difference_total_seconds(gpu)

    total_speedup = float(cpu_total / gpu_total) if gpu_total > 0 else 0.0
    forward_speedup = float(cpu_forward / gpu_forward) if gpu_forward > 0 else 0.0
    first_forward_speedup = (
        float(cpu_first_forward / gpu_first_forward) if gpu_first_forward > 0 else 0.0
    )
    warm_forward_avg_speedup = (
        float(cpu_warm_forward_avg / gpu_warm_forward_avg)
        if gpu_warm_forward_avg > 0
        else 0.0
    )
    legacy_forward_speedup = (
        float(cpu_forward_legacy / gpu_forward_legacy)
        if gpu_forward_legacy > 0
        else 0.0
    )
    absolute_reconstruct_speedup = (
        float(cpu_absolute_reconstruct / gpu_absolute_reconstruct)
        if gpu_absolute_reconstruct > 0
        else 0.0
    )
    difference_total_speedup = (
        float(cpu_difference_total / gpu_difference_total)
        if gpu_difference_total > 0
        else 0.0
    )

    payload = {
        "kind": str(args.kind),
        "cpu_report": str(args.cpu),
        "gpu_report": str(args.gpu),
        "cpu_total_elapsed_sec": cpu_total,
        "gpu_total_elapsed_sec": gpu_total,
        "forward_heavy_metric": forward_metric_label,
        "cpu_forward_heavy_sec": cpu_forward,
        "gpu_forward_heavy_sec": gpu_forward,
        "cpu_first_forward_sec": cpu_first_forward,
        "gpu_first_forward_sec": gpu_first_forward,
        "cpu_warm_forward_avg_sec": cpu_warm_forward_avg,
        "gpu_warm_forward_avg_sec": gpu_warm_forward_avg,
        "cpu_forward_legacy_sec": cpu_forward_legacy,
        "gpu_forward_legacy_sec": gpu_forward_legacy,
        "cpu_absolute_reconstruct_sec": cpu_absolute_reconstruct,
        "gpu_absolute_reconstruct_sec": gpu_absolute_reconstruct,
        "cpu_difference_total_sec": cpu_difference_total,
        "gpu_difference_total_sec": gpu_difference_total,
        "total_speedup_x": total_speedup,
        "forward_heavy_speedup_x": forward_speedup,
        "first_forward_speedup_x": first_forward_speedup,
        "warm_forward_avg_speedup_x": warm_forward_avg_speedup,
        "forward_legacy_speedup_x": legacy_forward_speedup,
        "absolute_reconstruct_speedup_x": absolute_reconstruct_speedup,
        "difference_total_speedup_x": difference_total_speedup,
        "gpu_faster_total": bool(gpu_total < cpu_total),
        "gpu_faster_forward_heavy": bool(gpu_forward < cpu_forward),
        "gpu_faster_first_forward": bool(gpu_first_forward < cpu_first_forward)
        if math.isfinite(cpu_first_forward) and math.isfinite(gpu_first_forward)
        else False,
        "gpu_faster_warm_forward_avg": bool(gpu_warm_forward_avg < cpu_warm_forward_avg)
        if math.isfinite(cpu_warm_forward_avg) and math.isfinite(gpu_warm_forward_avg)
        else False,
        "gpu_faster_forward_legacy": bool(gpu_forward_legacy < cpu_forward_legacy),
        "gpu_faster_absolute_reconstruct": bool(
            gpu_absolute_reconstruct < cpu_absolute_reconstruct
        )
        if math.isfinite(cpu_absolute_reconstruct)
        and math.isfinite(gpu_absolute_reconstruct)
        else False,
        "gpu_faster_difference_total": bool(gpu_difference_total < cpu_difference_total)
        if math.isfinite(cpu_difference_total) and math.isfinite(gpu_difference_total)
        else False,
        "cpu_mesh_info": cpu.get("mesh_info"),
        "gpu_mesh_info": gpu.get("mesh_info"),
        "cpu_backend": _backend(cpu),
        "gpu_backend": _backend(gpu),
        "cpu_forward_solver_benchmark": _forward_solver_benchmark(cpu),
        "gpu_forward_solver_benchmark": _forward_solver_benchmark(gpu),
        "forward_output_parity": _forward_output_parity(cpu, gpu),
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
