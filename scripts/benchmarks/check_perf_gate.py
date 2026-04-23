#!/usr/bin/env python3
"""Validate 3D benchmark report against performance gates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pyeidors.perf.policy import (
    PERF_GATE_AUTOTUNE_JACOBIAN_SPEEDUP_REF2,
    PERF_GATE_COMBINED_TOTAL_TARGETS,
    PERF_GATE_PEAK_MEMORY_LIMIT_RATIO,
    PROFILE_A_BASELINE,
    PROFILE_B_CHOLMOD_ONLY,
    PROFILE_C_AUTOTUNE_ONLY,
    PROFILE_D_COMBINED,
    PROFILE_E_FUSED,
    QUICK_BENCHMARK_PEAK_OVERHEAD_LIMIT,
)


BASELINE_DIFF_COLD_SEC = 38.74433858299972
BASELINE_ABSOLUTE_TOTAL_SEC = 74.58313283300004
BASELINE_ABSOLUTE_ITERS = 2
BASELINE_ABSOLUTE_PEAK_MIB = 218.19056129455566


@dataclass(slots=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def _stage_map(payload: dict) -> dict[str, dict]:
    stages = payload.get("stages", [])
    out: dict[str, dict] = {}
    if not isinstance(stages, list):
        return out
    for item in stages:
        if not isinstance(item, dict):
            continue
        name = item.get("stage")
        if isinstance(name, str):
            out[name] = item
    return out


def _safe_float(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _cache_build_elapsed(payload: dict, section: str) -> float | None:
    cache = payload.get("cache")
    if not isinstance(cache, dict):
        return None
    block = cache.get(section)
    if not isinstance(block, dict) or not block:
        return None
    total = 0.0
    saw_numeric = False
    for value in block.values():
        current = _safe_float(value)
        if current is None:
            continue
        total += current
        saw_numeric = True
    if not saw_numeric:
        return None
    return float(total)


def _evaluate(payload: dict) -> list[GateResult]:
    if isinstance(payload.get("results"), dict):
        return _evaluate_fair_compare(payload)

    stage = _stage_map(payload)
    config = payload.get("config", {})
    refinement = _safe_float(config.get("refinement")) or 1.0
    problem_scale = max(1.0, refinement)
    diff_scale = max(1.0, refinement**1.5)
    diff_cold = _cache_build_elapsed(payload, "cold_build")
    if diff_cold is None:
        diff_cold = _safe_float(stage.get("diff_context_cold", {}).get("elapsed_sec"))
    diff_warm = _cache_build_elapsed(payload, "warm_build")
    if diff_warm is None:
        diff_warm = _safe_float(stage.get("diff_context_warm", {}).get("elapsed_sec"))
    absolute_total = _safe_float(
        stage.get("absolute_reconstruct", {}).get("elapsed_sec")
    )
    absolute_peak = _safe_float(stage.get("absolute_reconstruct", {}).get("peak_mib"))

    results: list[GateResult] = []

    if diff_cold is None:
        results.append(
            GateResult(
                "diff_cold_speedup", False, "missing diff_context_cold.elapsed_sec"
            )
        )
    else:
        threshold = (BASELINE_DIFF_COLD_SEC / 2.5) * diff_scale
        results.append(
            GateResult(
                "diff_cold_speedup",
                diff_cold <= threshold,
                f"cold={diff_cold:.4f}s threshold<={threshold:.4f}s",
            )
        )

    if diff_cold is None or diff_warm is None:
        results.append(
            GateResult(
                "diff_warm_cold_ratio", False, "missing diff warm/cold elapsed_sec"
            )
        )
    else:
        ratio = diff_cold / max(diff_warm, 1e-12)
        results.append(
            GateResult(
                "diff_warm_cold_ratio",
                ratio >= 80.0,
                f"ratio={ratio:.2f} threshold>=80.00",
            )
        )

    if absolute_total is None:
        results.append(
            GateResult(
                "absolute_iter_speedup",
                False,
                "missing absolute_reconstruct.elapsed_sec",
            )
        )
    else:
        current_per_iter = absolute_total / max(
            1,
            int(config.get("absolute_iters", BASELINE_ABSOLUTE_ITERS)),
        )
        baseline_per_iter = BASELINE_ABSOLUTE_TOTAL_SEC / BASELINE_ABSOLUTE_ITERS
        threshold = (baseline_per_iter / 3.5) * problem_scale
        results.append(
            GateResult(
                "absolute_iter_speedup",
                current_per_iter <= threshold,
                f"per_iter={current_per_iter:.4f}s threshold<={threshold:.4f}s",
            )
        )

    if absolute_peak is None:
        results.append(
            GateResult(
                "absolute_peak_memory", False, "missing absolute_reconstruct.peak_mib"
            )
        )
    else:
        threshold = (BASELINE_ABSOLUTE_PEAK_MIB * 0.70) * problem_scale
        results.append(
            GateResult(
                "absolute_peak_memory",
                absolute_peak <= threshold,
                f"peak={absolute_peak:.3f}MiB threshold<={threshold:.3f}MiB",
            )
        )

    capabilities = payload.get("capabilities")
    absolute_solver = payload.get("absolute_solver")
    cholmod_available = isinstance(capabilities, dict) and bool(
        capabilities.get("cholmod", False)
    )
    if cholmod_available:
        fast_solver_path = None
        fallback_reason = None
        if isinstance(absolute_solver, dict):
            fast_solver_path = absolute_solver.get("fast_solver_path")
            fallback_reason = absolute_solver.get("fallback_reason")
        has_cholmod_path = (
            isinstance(fast_solver_path, str) and "cholmod" in fast_solver_path.lower()
        )
        has_fallback = (
            isinstance(fallback_reason, str) and len(fallback_reason.strip()) > 0
        )
        results.append(
            GateResult(
                "absolute_solver_path_cholmod",
                has_cholmod_path or has_fallback,
                f"path={fast_solver_path!r} fallback_reason={fallback_reason!r}",
            )
        )
    return results


def _evaluate_fair_compare(payload: dict) -> list[GateResult]:
    results: list[GateResult] = []
    compare_results = payload.get("results", {})
    if not isinstance(compare_results, dict) or not compare_results:
        return [GateResult("fair_compare_payload", False, "missing results payload")]
    benchmark_phase = str(payload.get("benchmark_phase", "full"))
    if benchmark_phase == "quick":
        quick_eval = payload.get("quick_eval", {})
        thresholds = payload.get("quick_thresholds", {})
        total_improve = _safe_float(quick_eval.get("total_improvement_ratio"))
        linear_improve = _safe_float(quick_eval.get("linear_improvement_ratio"))
        peak_delta = _safe_float(quick_eval.get("peak_memory_delta_ratio"))
        total_th = _safe_float(thresholds.get("total")) or 0.05
        linear_th = _safe_float(thresholds.get("linear")) or 0.10
        peak_th = _safe_float(thresholds.get("peak_overhead_limit")) or 0.10

        if total_improve is None or linear_improve is None or peak_delta is None:
            return [GateResult("quick.metrics", False, "missing quick_eval metrics")]

        peak_limit = min(float(peak_th), QUICK_BENCHMARK_PEAK_OVERHEAD_LIMIT)
        passed = (total_improve >= total_th) or (
            linear_improve >= linear_th and peak_delta <= peak_limit
        )
        declared_pass = bool(payload.get("quick_pass", False))
        results.append(
            GateResult(
                "quick.improvement_gate",
                passed,
                (
                    f"total={total_improve:.4f} threshold>={total_th:.4f}; "
                    f"linear={linear_improve:.4f} threshold>={linear_th:.4f}; "
                    f"peak_delta={peak_delta:.4f} limit<={peak_limit:.4f}"
                ),
            )
        )
        results.append(
            GateResult(
                "quick.pass_consistency",
                declared_pass == passed,
                f"reported_quick_pass={declared_pass} computed={passed}",
            )
        )
        return results

    for ref_key, block in sorted(compare_results.items()):
        if not isinstance(block, dict):
            results.append(GateResult(f"{ref_key}.format", False, "invalid block"))
            continue
        profiles = block.get("profiles", {})
        speedup = block.get("speedup_vs_A", {})
        if not isinstance(profiles, dict) or not isinstance(speedup, dict):
            results.append(
                GateResult(
                    f"{ref_key}.profiles", False, "missing profiles or speedup_vs_A"
                )
            )
            continue
        baseline = profiles.get(PROFILE_A_BASELINE, {}).get("median", {})
        baseline_peak = (
            _safe_float(baseline.get("absolute_peak_mib"))
            if isinstance(baseline, dict)
            else None
        )
        if baseline_peak is None:
            results.append(
                GateResult(
                    f"{ref_key}.baseline_peak",
                    False,
                    f"missing {PROFILE_A_BASELINE}.absolute_peak_mib",
                )
            )
            continue

        ref_num = int(ref_key.split("_")[-1]) if "_" in ref_key else 0
        b_profile = profiles.get(PROFILE_B_CHOLMOD_ONLY, {}).get("median", {})
        b_path = str(b_profile.get("fast_solver_path", "") or "")
        b_fallback = str(b_profile.get("fallback_reason", "") or "")
        used_cholmod_precond = (
            "cholmod-precond" in b_path.lower() or "cholmod" in b_path.lower()
        )
        results.append(
            GateResult(
                f"{ref_key}.cholmod_path_or_fallback",
                used_cholmod_precond or len(b_fallback.strip()) > 0,
                f"path={b_path!r} fallback_reason={b_fallback!r}",
            )
        )

        if ref_num == 2:
            jacobian_speed = _safe_float(
                speedup.get(PROFILE_C_AUTOTUNE_ONLY, {}).get(
                    "absolute_jacobian_assembly_speedup_x"
                )
            )
            if jacobian_speed is None:
                results.append(
                    GateResult(
                        f"{ref_key}.autotune_jacobian_assembly_speedup",
                        False,
                        f"missing {PROFILE_C_AUTOTUNE_ONLY} absolute_jacobian_assembly_speedup_x",
                    )
                )
            else:
                results.append(
                    GateResult(
                        f"{ref_key}.autotune_jacobian_assembly_speedup",
                        jacobian_speed >= PERF_GATE_AUTOTUNE_JACOBIAN_SPEEDUP_REF2,
                        f"speedup={jacobian_speed:.3f} threshold>={PERF_GATE_AUTOTUNE_JACOBIAN_SPEEDUP_REF2:.3f}",
                    )
                )

        combined_total = _safe_float(
            speedup.get(PROFILE_D_COMBINED, {}).get("absolute_total_speedup_x")
        )
        combined_target = PERF_GATE_COMBINED_TOTAL_TARGETS.get(ref_num, 1.0)
        if combined_total is None:
            results.append(
                GateResult(
                    f"{ref_key}.combined_total_speedup",
                    False,
                    f"missing {PROFILE_D_COMBINED} speedup",
                )
            )
        else:
            results.append(
                GateResult(
                    f"{ref_key}.combined_total_speedup",
                    combined_total >= combined_target,
                    f"speedup={combined_total:.3f} threshold>={combined_target:.3f}",
                )
            )

        combined_peak = _safe_float(
            profiles.get(PROFILE_D_COMBINED, {})
            .get("median", {})
            .get("absolute_peak_mib")
        )
        if combined_peak is None:
            results.append(
                GateResult(
                    f"{ref_key}.combined_peak_memory",
                    False,
                    f"missing {PROFILE_D_COMBINED} peak",
                )
            )
        else:
            peak_limit = baseline_peak * PERF_GATE_PEAK_MEMORY_LIMIT_RATIO
            results.append(
                GateResult(
                    f"{ref_key}.combined_peak_memory",
                    combined_peak <= peak_limit,
                    f"peak={combined_peak:.3f}MiB limit<={peak_limit:.3f}MiB",
                )
            )

        fused_peak = _safe_float(
            profiles.get(PROFILE_E_FUSED, {}).get("median", {}).get("absolute_peak_mib")
        )
        if fused_peak is None:
            results.append(
                GateResult(
                    f"{ref_key}.fused_peak_memory",
                    False,
                    f"missing {PROFILE_E_FUSED} peak",
                )
            )
        else:
            peak_limit = baseline_peak * PERF_GATE_PEAK_MEMORY_LIMIT_RATIO
            results.append(
                GateResult(
                    f"{ref_key}.fused_peak_memory",
                    fused_peak <= peak_limit,
                    f"peak={fused_peak:.3f}MiB limit<={peak_limit:.3f}MiB",
                )
            )

        fused_profile = profiles.get(PROFILE_E_FUSED, {}).get("median", {})
        fused_path = str(fused_profile.get("fast_solver_path", "") or "")
        fused_rom = bool(fused_profile.get("rom_enabled_effective", False))
        fused_fallback = str(fused_profile.get("fallback_reason", "") or "")
        results.append(
            GateResult(
                f"{ref_key}.fused_path_or_rom_enabled",
                fused_rom
                or fused_path.startswith("fused-")
                or len(fused_fallback.strip()) > 0,
                f"rom_enabled={fused_rom} path={fused_path!r} fallback_reason={fused_fallback!r}",
            )
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Benchmark JSON file")
    parser.add_argument(
        "--mode",
        choices=["off", "warn", "strict"],
        default="warn",
        help="Gate severity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    results = _evaluate(payload)
    failed = [item for item in results if not item.passed]

    for item in results:
        prefix = "PASS" if item.passed else "FAIL"
        print(f"[{prefix}] {item.name}: {item.detail}")

    if args.mode == "off":
        return
    if failed and args.mode == "warn":
        print(f"[WARN] performance gate failed for {len(failed)} checks")
        return
    if failed:
        print(
            f"[ERROR] performance gate failed for {len(failed)} checks", file=sys.stderr
        )
        raise SystemExit(1)
    print("[OK] performance gate passed")


if __name__ == "__main__":
    main()
