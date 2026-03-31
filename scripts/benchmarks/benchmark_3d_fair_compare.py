#!/usr/bin/env python3
"""Run fair A/B profile comparisons for 3D EIT acceleration paths."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyeidors.perf.policy import (
    DEFAULT_CHOLMOD_MAX_MEMORY_GIB,
    PROFILE_A_BASELINE,
    PROFILE_B_CHOLMOD_ONLY,
    PROFILE_C_AUTOTUNE_ONLY,
    PROFILE_D_COMBINED,
    PROFILE_E_FUSED,
    EXPERIMENTAL_PERF_PROFILES,
    PRIMARY_PERF_PROFILE,
    QUICK_BENCHMARK_PEAK_OVERHEAD_LIMIT,
    QUICK_PERF_PROFILES,
    is_experimental_profile,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-phase",
        choices=["quick", "full"],
        default="quick",
        help="quick: short validation gate, full: exhaustive A/B/C/D median compare",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports") / "perf" / "fair_compare_latest.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports") / "perf" / "fair_compare_latest.md",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repeated runs per (profile, refinement) for full mode.",
    )
    parser.add_argument(
        "--refinements",
        type=str,
        default="1,2",
        help="Comma-separated refinement levels for full mode.",
    )
    parser.add_argument("--mesh-dir", type=Path, default=Path("eit_meshes"))
    parser.add_argument("--cache-root", type=Path, default=Path(".pyeidors_cache") / "fair_compare")
    parser.add_argument("--cholmod-max-memory-gib", type=float, default=DEFAULT_CHOLMOD_MAX_MEMORY_GIB)
    parser.add_argument(
        "--benchmark-quick-threshold-total",
        type=float,
        default=0.05,
        help="Quick phase minimum total-time improvement ratio for D vs A.",
    )
    parser.add_argument(
        "--benchmark-quick-threshold-linear",
        type=float,
        default=0.10,
        help="Quick phase minimum linear-solve improvement ratio for D vs A.",
    )
    return parser.parse_args()


def _stage_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for stage in payload.get("stages", []):
        if isinstance(stage, dict) and isinstance(stage.get("stage"), str):
            out[stage["stage"]] = stage
    return out


def _safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _sum_numeric(block: Any) -> float:
    if not isinstance(block, dict):
        return 0.0
    total = 0.0
    for value in block.values():
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _run_runtime_report(
    benchmark_script: Path,
    *,
    output_json: Path,
    cache_dir: Path,
    mesh_dir: Path,
    refinement: int,
    profile_label: str,
    preconditioner: str,
    jacobian_block_tune: str,
    jacobian_block_size: int,
    fast_linear_path: str,
    rom_mode: str,
    rom_rank_global: int,
    rom_rank_adaptive: int,
    rom_refresh_every: int,
    rom_snapshot_source: str,
    inexact_mode: str,
    inexact_forcing: str,
    inexact_eta0: float,
    inexact_eta_min: float,
    inexact_eta_max: float,
    lowrank_mode: str,
    lowrank_rank: int,
    lowrank_method: str,
    lowrank_energy: float,
    cholmod_max_n: int,
    cholmod_max_memory_gib: float,
    absolute_startup_cache: str,
    run_diff: str,
    run_absolute: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(benchmark_script),
        "--mesh-dir",
        str(mesh_dir),
        "--cache-dir",
        str(cache_dir),
        "--perf-report",
        str(output_json),
        "--profile-label",
        profile_label,
        "--refinement",
        str(refinement),
        "--preconditioner",
        preconditioner,
        "--fast-linear-path",
        fast_linear_path,
        "--rom-mode",
        rom_mode,
        "--rom-rank-global",
        str(rom_rank_global),
        "--rom-rank-adaptive",
        str(rom_rank_adaptive),
        "--rom-refresh-every",
        str(rom_refresh_every),
        "--rom-snapshot-source",
        rom_snapshot_source,
        "--inexact-mode",
        inexact_mode,
        "--inexact-forcing",
        inexact_forcing,
        "--inexact-eta0",
        str(inexact_eta0),
        "--inexact-eta-min",
        str(inexact_eta_min),
        "--inexact-eta-max",
        str(inexact_eta_max),
        "--lowrank-mode",
        lowrank_mode,
        "--lowrank-rank",
        str(lowrank_rank),
        "--lowrank-method",
        lowrank_method,
        "--lowrank-energy",
        str(lowrank_energy),
        "--jacobian-block-tune",
        jacobian_block_tune,
        "--jacobian-block-size",
        str(jacobian_block_size),
        "--cholmod-max-n",
        str(cholmod_max_n),
        "--cholmod-max-memory-gib",
        str(cholmod_max_memory_gib),
        "--absolute-startup-cache",
        absolute_startup_cache,
        "--run-diff",
        run_diff,
        "--run-absolute",
        run_absolute,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"benchmark failed for profile={profile_label}, refinement={refinement}, "
            f"run_diff={run_diff}, run_absolute={run_absolute}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    return json.loads(output_json.read_text(encoding="utf-8"))


def _extract_metrics(diff_payload: dict[str, Any] | None, absolute_payload: dict[str, Any] | None) -> dict[str, Any]:
    diff_payload = diff_payload or {}
    absolute_payload = absolute_payload or {}

    diff_stage = _stage_map(diff_payload)
    diff_cache = diff_payload.get("cache", {})
    diff_cold = _sum_numeric(diff_cache.get("cold_build"))
    diff_warm = _sum_numeric(diff_cache.get("warm_build"))
    if diff_cold <= 0:
        diff_cold = _safe_float(diff_stage.get("diff_context_cold", {}).get("elapsed_sec"), 0.0)
    if diff_warm <= 0:
        diff_warm = _safe_float(diff_stage.get("diff_context_warm", {}).get("elapsed_sec"), 0.0)

    abs_stage = _stage_map(absolute_payload)
    abs_breakdown = absolute_payload.get("stage_breakdown", {}).get("absolute", {})
    abs_solver = absolute_payload.get("absolute_solver", {})
    mesh_info = absolute_payload.get("mesh_info", {})

    absolute_total = _safe_float(abs_stage.get("absolute_reconstruct", {}).get("elapsed_sec"), 0.0)
    absolute_peak = _safe_float(abs_stage.get("absolute_reconstruct", {}).get("peak_mib"), 0.0)
    absolute_linear = _safe_float(abs_breakdown.get("linear_solve"), 0.0)
    absolute_jacobian = _safe_float(abs_breakdown.get("jacobian"), 0.0)
    jacobian_assembly_only = _safe_float(
        abs_solver.get("jacobian_assembly_elapsed_only"),
        absolute_jacobian,
    )

    fast_solver_path = str(abs_solver.get("fast_solver_path", "") or "")
    fallback_reason = str(abs_solver.get("fallback_reason", "") or "")
    rom_rank_effective = int(abs_solver.get("rom_rank_effective", 0) or 0)
    lowrank_rank_effective = int(abs_solver.get("lowrank_rank_effective", 0) or 0)
    first_forward = _safe_float(abs_solver.get("first_forward_elapsed_sec"), 0.0)
    warm_forward_avg = _safe_float(abs_solver.get("warm_forward_avg_sec"), 0.0)
    target_forward = _safe_float(abs_solver.get("target_forward_elapsed_sec"), 0.0)

    return {
        "diff_context_cold_sec": float(diff_cold),
        "diff_context_warm_sec": float(diff_warm),
        "diff_warm_cold_ratio": float(diff_cold / max(diff_warm, 1e-12)) if diff_cold > 0 else 0.0,
        "absolute_total_sec": float(absolute_total),
        "absolute_linear_sec": float(absolute_linear),
        "absolute_jacobian_sec": float(absolute_jacobian),
        "absolute_jacobian_assembly_only_sec": float(jacobian_assembly_only),
        "absolute_peak_mib": float(absolute_peak),
        "absolute_first_forward_sec": float(first_forward),
        "absolute_warm_forward_avg_sec": float(warm_forward_avg),
        "absolute_target_forward_sec": float(target_forward),
        "mesh_nodes": int(mesh_info.get("nodes", 0) or 0),
        "mesh_elements": int(mesh_info.get("elements", 0) or 0),
        "mesh_potential_dofs": int(mesh_info.get("potential_dofs", 0) or 0),
        "mesh_sigma_dofs": int(mesh_info.get("sigma_dofs", 0) or 0),
        "fast_solver_path": fast_solver_path,
        "fast_linear_path_selected": str(abs_solver.get("fast_linear_path_selected", "") or ""),
        "fast_linear_path_reason": str(abs_solver.get("fast_linear_path_reason", "") or ""),
        "fallback_reason": fallback_reason,
        "cholmod_path_used": "cholmod" in fast_solver_path.lower(),
        "rom_enabled_effective": bool(abs_solver.get("rom_enabled_effective", False)),
        "rom_rank_effective": rom_rank_effective,
        "lowrank_rank_effective": lowrank_rank_effective,
        "degrade_stage_counts": abs_solver.get("degrade_stage_counts", {}),
        "effective_solver_path_counts": abs_solver.get("effective_solver_path_counts", {}),
    }


def _base_profiles(cholmod_max_memory_gib: float) -> dict[str, dict[str, Any]]:
    return {
        PROFILE_A_BASELINE: {
            "preconditioner": "diag",
            "jacobian_block_tune": "off",
            "jacobian_block_size": 256,
            "fast_linear_path": "pcg",
            "rom_mode": "off",
            "inexact_mode": "off",
            "lowrank_mode": "off",
            "cholmod_max_n": 50000,
            "cholmod_max_memory_gib": cholmod_max_memory_gib,
            "absolute_startup_cache": "off",
        },
        PROFILE_B_CHOLMOD_ONLY: {
            "preconditioner": "cholmod",
            "jacobian_block_tune": "off",
            "jacobian_block_size": 256,
            "fast_linear_path": "pcg",
            "rom_mode": "off",
            "inexact_mode": "off",
            "lowrank_mode": "off",
            "cholmod_max_n": 50000,
            "cholmod_max_memory_gib": cholmod_max_memory_gib,
            "absolute_startup_cache": "off",
        },
        PROFILE_C_AUTOTUNE_ONLY: {
            "preconditioner": "diag",
            "jacobian_block_tune": "auto",
            "jacobian_block_size": 0,
            "fast_linear_path": "pcg",
            "rom_mode": "off",
            "inexact_mode": "off",
            "lowrank_mode": "off",
            "cholmod_max_n": 50000,
            "cholmod_max_memory_gib": cholmod_max_memory_gib,
            "absolute_startup_cache": "off",
        },
        PROFILE_D_COMBINED: {
            "preconditioner": "cholmod",
            "jacobian_block_tune": "auto",
            "jacobian_block_size": 0,
            "fast_linear_path": "auto",
            "rom_mode": "off",
            "inexact_mode": "off",
            "lowrank_mode": "off",
            "cholmod_max_n": 50000,
            "cholmod_max_memory_gib": cholmod_max_memory_gib,
            "absolute_startup_cache": "off",
        },
        PROFILE_E_FUSED: {
            "experimental": True,
            "preconditioner": "cholmod",
            "jacobian_block_tune": "auto",
            "jacobian_block_size": 0,
            "fast_linear_path": "auto",
            "rom_mode": "on",
            "rom_rank_global": 16,
            "rom_rank_adaptive": 8,
            "rom_refresh_every": 3,
            "rom_snapshot_source": "hybrid",
            "inexact_mode": "auto",
            "inexact_forcing": "eisenstat-walker",
            "inexact_eta0": 0.2,
            "inexact_eta_min": 1e-3,
            "inexact_eta_max": 0.5,
            "lowrank_mode": "auto",
            "lowrank_rank": 8,
            "lowrank_method": "tsvd",
            "lowrank_energy": 0.995,
            "cholmod_max_n": 50000,
            "cholmod_max_memory_gib": cholmod_max_memory_gib,
            "absolute_startup_cache": "off",
        },
    }


def _compute_speedup_block(ref_profiles: dict[str, Any]) -> dict[str, dict[str, float]]:
    baseline = ref_profiles[PROFILE_A_BASELINE]["median"]
    speedup: dict[str, dict[str, float]] = {}
    for profile_name, profile_data in ref_profiles.items():
        median_metrics = profile_data["median"]
        speedup[profile_name] = {
            "absolute_total_speedup_x": float(baseline["absolute_total_sec"])
            / max(float(median_metrics["absolute_total_sec"]), 1e-12),
            "absolute_linear_speedup_x": float(baseline["absolute_linear_sec"])
            / max(float(median_metrics["absolute_linear_sec"]), 1e-12),
            "absolute_jacobian_speedup_x": float(baseline["absolute_jacobian_sec"])
            / max(float(median_metrics["absolute_jacobian_sec"]), 1e-12),
            "absolute_jacobian_assembly_speedup_x": float(
                baseline["absolute_jacobian_assembly_only_sec"]
            )
            / max(float(median_metrics["absolute_jacobian_assembly_only_sec"]), 1e-12),
            "diff_cold_speedup_x": float(baseline["diff_context_cold_sec"])
            / max(float(median_metrics["diff_context_cold_sec"]), 1e-12),
        }
    return speedup


def _render_markdown(payload: dict[str, Any]) -> str:
    phase = str(payload.get("benchmark_phase", "full"))
    lines: list[str] = []
    lines.append(f"# PyEIDORS 3D Fair Compare ({phase})")
    lines.append("")
    lines.append(f"- Generated (UTC): {payload['generated_at_utc']}")
    lines.append(f"- Repeat per profile: {payload['repeat']}")
    lines.append(f"- Refinements: {', '.join(str(v) for v in payload['refinements'])}")
    if phase == "quick":
        eval_block = payload.get("quick_eval", {})
        lines.append(f"- Quick pass: {bool(payload.get('quick_pass', False))}")
        lines.append(f"- Primary profile: {payload.get('primary_profile', PRIMARY_PERF_PROFILE)}")
        lines.append(
            f"- Quick deltas: total={float(eval_block.get('total_improvement_ratio', 0.0)):.4f}, "
            f"linear={float(eval_block.get('linear_improvement_ratio', 0.0)):.4f}, "
            f"peak_delta={float(eval_block.get('peak_memory_delta_ratio', 0.0)):.4f}"
        )
    lines.append("")

    experimental_profiles = {str(name) for name in payload.get("experimental_profiles", EXPERIMENTAL_PERF_PROFILES)}
    for ref_key in sorted(payload["results"].keys()):
        block = payload["results"][ref_key]
        lines.append(f"## {ref_key}")
        lines.append("")
        lines.append(
            "| Profile | diff cold(s) | diff warm(s) | warm/cold | abs total(s) | abs linear(s) | "
            "abs jacobian(s) | jacobian assembly(s) | peak(MiB) | solver path | fallback |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for profile_name, profile_data in block["profiles"].items():
            median = profile_data["median"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        profile_name + (" (experimental)" if profile_name in experimental_profiles or is_experimental_profile(profile_name) else ""),
                        f"{median['diff_context_cold_sec']:.4f}",
                        f"{median['diff_context_warm_sec']:.4f}",
                        f"{median['diff_warm_cold_ratio']:.2f}",
                        f"{median['absolute_total_sec']:.4f}",
                        f"{median['absolute_linear_sec']:.4f}",
                        f"{median['absolute_jacobian_sec']:.4f}",
                        f"{median['absolute_jacobian_assembly_only_sec']:.4f}",
                        f"{median['absolute_peak_mib']:.2f}",
                        str(median["fast_solver_path"]),
                        str(median["fallback_reason"]),
                    ]
                )
                + " |"
            )

        speedup = block.get("speedup_vs_A")
        if isinstance(speedup, dict):
            lines.append("")
            lines.append("### Speedup vs A_baseline")
            lines.append("")
            lines.append(
                "| Profile | abs total x | abs linear x | jacobian x | jacobian assembly x | diff cold x |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|")
            for profile_name, speed in speedup.items():
                lines.append(
                    f"| {profile_name} | {speed['absolute_total_speedup_x']:.3f} | "
                    f"{speed['absolute_linear_speedup_x']:.3f} | {speed['absolute_jacobian_speedup_x']:.3f} | "
                    f"{speed['absolute_jacobian_assembly_speedup_x']:.3f} | {speed['diff_cold_speedup_x']:.3f} |"
                )
        lines.append("")

    return "\n".join(lines) + "\n"


def _run_profiles(
    *,
    benchmark_script: Path,
    cache_root: Path,
    mesh_dir: Path,
    profiles: dict[str, dict[str, Any]],
    refinement: int,
    repeat: int,
    run_diff: bool,
    run_absolute: bool,
) -> dict[str, Any]:
    ref_profiles: dict[str, Any] = {}
    for profile_name, profile_cfg in profiles.items():
        run_records: list[dict[str, Any]] = []
        for run_index in range(int(repeat)):
            diff_payload = None
            absolute_payload = None
            source_reports: dict[str, str] = {}
            if run_diff:
                diff_json = (
                    cache_root
                    / "reports"
                    / f"ref_{refinement}"
                    / profile_name
                    / "diff"
                    / f"run_{run_index:02d}.json"
                )
                diff_json.parent.mkdir(parents=True, exist_ok=True)
                diff_payload = _run_runtime_report(
                    benchmark_script,
                    output_json=diff_json,
                    cache_dir=cache_root / "cache" / f"ref_{refinement}" / profile_name / "diff" / f"run_{run_index:02d}",
                    mesh_dir=mesh_dir,
                    refinement=refinement,
                    profile_label=f"{profile_name}-diff",
                    preconditioner=str(profile_cfg["preconditioner"]),
                    jacobian_block_tune=str(profile_cfg["jacobian_block_tune"]),
                    jacobian_block_size=int(profile_cfg["jacobian_block_size"]),
                    fast_linear_path=str(profile_cfg.get("fast_linear_path", "auto")),
                    rom_mode=str(profile_cfg.get("rom_mode", "off")),
                    rom_rank_global=int(profile_cfg.get("rom_rank_global", 32)),
                    rom_rank_adaptive=int(profile_cfg.get("rom_rank_adaptive", 16)),
                    rom_refresh_every=int(profile_cfg.get("rom_refresh_every", 2)),
                    rom_snapshot_source=str(profile_cfg.get("rom_snapshot_source", "hybrid")),
                    inexact_mode=str(profile_cfg.get("inexact_mode", "off")),
                    inexact_forcing=str(profile_cfg.get("inexact_forcing", "eisenstat-walker")),
                    inexact_eta0=float(profile_cfg.get("inexact_eta0", 0.2)),
                    inexact_eta_min=float(profile_cfg.get("inexact_eta_min", 1e-3)),
                    inexact_eta_max=float(profile_cfg.get("inexact_eta_max", 0.5)),
                    lowrank_mode=str(profile_cfg.get("lowrank_mode", "off")),
                    lowrank_rank=int(profile_cfg.get("lowrank_rank", 16)),
                    lowrank_method=str(profile_cfg.get("lowrank_method", "tsvd")),
                    lowrank_energy=float(profile_cfg.get("lowrank_energy", 0.995)),
                    cholmod_max_n=int(profile_cfg["cholmod_max_n"]),
                    cholmod_max_memory_gib=float(profile_cfg["cholmod_max_memory_gib"]),
                    absolute_startup_cache=str(profile_cfg.get("absolute_startup_cache", "off")),
                    run_diff="on",
                    run_absolute="off",
                )
                source_reports["diff"] = str(diff_json)

            if run_absolute:
                absolute_json = (
                    cache_root
                    / "reports"
                    / f"ref_{refinement}"
                    / profile_name
                    / "absolute"
                    / f"run_{run_index:02d}.json"
                )
                absolute_json.parent.mkdir(parents=True, exist_ok=True)
                absolute_payload = _run_runtime_report(
                    benchmark_script,
                    output_json=absolute_json,
                    cache_dir=cache_root
                    / "cache"
                    / f"ref_{refinement}"
                    / profile_name
                    / "absolute"
                    / f"run_{run_index:02d}",
                    mesh_dir=mesh_dir,
                    refinement=refinement,
                    profile_label=f"{profile_name}-absolute",
                    preconditioner=str(profile_cfg["preconditioner"]),
                    jacobian_block_tune=str(profile_cfg["jacobian_block_tune"]),
                    jacobian_block_size=int(profile_cfg["jacobian_block_size"]),
                    fast_linear_path=str(profile_cfg.get("fast_linear_path", "auto")),
                    rom_mode=str(profile_cfg.get("rom_mode", "off")),
                    rom_rank_global=int(profile_cfg.get("rom_rank_global", 32)),
                    rom_rank_adaptive=int(profile_cfg.get("rom_rank_adaptive", 16)),
                    rom_refresh_every=int(profile_cfg.get("rom_refresh_every", 2)),
                    rom_snapshot_source=str(profile_cfg.get("rom_snapshot_source", "hybrid")),
                    inexact_mode=str(profile_cfg.get("inexact_mode", "off")),
                    inexact_forcing=str(profile_cfg.get("inexact_forcing", "eisenstat-walker")),
                    inexact_eta0=float(profile_cfg.get("inexact_eta0", 0.2)),
                    inexact_eta_min=float(profile_cfg.get("inexact_eta_min", 1e-3)),
                    inexact_eta_max=float(profile_cfg.get("inexact_eta_max", 0.5)),
                    lowrank_mode=str(profile_cfg.get("lowrank_mode", "off")),
                    lowrank_rank=int(profile_cfg.get("lowrank_rank", 16)),
                    lowrank_method=str(profile_cfg.get("lowrank_method", "tsvd")),
                    lowrank_energy=float(profile_cfg.get("lowrank_energy", 0.995)),
                    cholmod_max_n=int(profile_cfg["cholmod_max_n"]),
                    cholmod_max_memory_gib=float(profile_cfg["cholmod_max_memory_gib"]),
                    absolute_startup_cache=str(profile_cfg.get("absolute_startup_cache", "off")),
                    run_diff="off",
                    run_absolute="on",
                )
                source_reports["absolute"] = str(absolute_json)

            metrics = _extract_metrics(diff_payload, absolute_payload)
            run_records.append(
                {
                    "run_index": run_index,
                    "metrics": metrics,
                    "source_reports": source_reports,
                }
            )

        def _metric_list(name: str) -> list[float]:
            return [float(run["metrics"][name]) for run in run_records]

        median_metrics = {
            "diff_context_cold_sec": _median(_metric_list("diff_context_cold_sec")),
            "diff_context_warm_sec": _median(_metric_list("diff_context_warm_sec")),
            "diff_warm_cold_ratio": _median(_metric_list("diff_warm_cold_ratio")),
            "absolute_total_sec": _median(_metric_list("absolute_total_sec")),
            "absolute_linear_sec": _median(_metric_list("absolute_linear_sec")),
            "absolute_jacobian_sec": _median(_metric_list("absolute_jacobian_sec")),
            "absolute_jacobian_assembly_only_sec": _median(
                _metric_list("absolute_jacobian_assembly_only_sec")
            ),
            "absolute_peak_mib": _median(_metric_list("absolute_peak_mib")),
            "fast_solver_path": sorted(
                [str(run["metrics"]["fast_solver_path"]) for run in run_records]
            )[len(run_records) // 2],
            "fast_linear_path_selected": sorted(
                [str(run["metrics"]["fast_linear_path_selected"]) for run in run_records]
            )[len(run_records) // 2],
            "fast_linear_path_reason": sorted(
                [str(run["metrics"]["fast_linear_path_reason"]) for run in run_records]
            )[len(run_records) // 2],
            "fallback_reason": sorted(
                [str(run["metrics"]["fallback_reason"]) for run in run_records]
            )[len(run_records) // 2],
            "cholmod_path_used": any(
                bool(run["metrics"]["cholmod_path_used"]) for run in run_records
            ),
            "rom_enabled_effective": any(
                bool(run["metrics"].get("rom_enabled_effective", False)) for run in run_records
            ),
            "rom_rank_effective": int(
                max(float(run["metrics"].get("rom_rank_effective", 0)) for run in run_records)
            ),
            "lowrank_rank_effective": int(
                max(float(run["metrics"].get("lowrank_rank_effective", 0)) for run in run_records)
            ),
            "degrade_stage_counts": {
                k: sum(
                    int((run["metrics"].get("degrade_stage_counts") or {}).get(k, 0))
                    for run in run_records
                )
                for k in {
                    key
                    for run in run_records
                    for key in (run["metrics"].get("degrade_stage_counts") or {}).keys()
                }
            },
        }

        ref_profiles[profile_name] = {
            "config": profile_cfg,
            "runs": run_records,
            "median": median_metrics,
        }
    return ref_profiles


def main() -> None:
    args = _parse_args()
    if int(args.repeat) <= 0:
        raise ValueError("--repeat must be a positive integer.")

    root = Path(__file__).resolve().parents[2]
    benchmark_script = root / "scripts" / "benchmarks" / "benchmark_3d_runtime.py"
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    profiles_all = _base_profiles(float(args.cholmod_max_memory_gib))

    if args.benchmark_phase == "quick":
        refinements = [1]
        repeat = 1
        profiles = {name: profiles_all[name] for name in QUICK_PERF_PROFILES}
        ref_profiles = _run_profiles(
            benchmark_script=benchmark_script,
            cache_root=cache_root,
            mesh_dir=args.mesh_dir.resolve(),
            profiles=profiles,
            refinement=1,
            repeat=repeat,
            run_diff=False,
            run_absolute=True,
        )
        speedup = _compute_speedup_block(ref_profiles)

        a = ref_profiles[PROFILE_A_BASELINE]["median"]
        d = ref_profiles[PROFILE_D_COMBINED]["median"]
        total_improvement = (float(a["absolute_total_sec"]) - float(d["absolute_total_sec"])) / max(
            float(a["absolute_total_sec"]), 1e-12
        )
        linear_improvement = (float(a["absolute_linear_sec"]) - float(d["absolute_linear_sec"])) / max(
            float(a["absolute_linear_sec"]), 1e-12
        )
        peak_delta = (float(d["absolute_peak_mib"]) - float(a["absolute_peak_mib"])) / max(
            float(a["absolute_peak_mib"]),
            1e-12,
        )

        quick_pass = bool(
            (total_improvement >= float(args.benchmark_quick_threshold_total))
            or (
                linear_improvement >= float(args.benchmark_quick_threshold_linear)
                and peak_delta <= QUICK_BENCHMARK_PEAK_OVERHEAD_LIMIT
            )
        )

        payload = {
            "benchmark_phase": "quick",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repeat": repeat,
            "refinements": refinements,
            "profiles": profiles,
            "primary_profile": PROFILE_D_COMBINED,
            "experimental_profiles": list(EXPERIMENTAL_PERF_PROFILES),
            "results": {"ref_1": {"profiles": ref_profiles, "speedup_vs_A": speedup}},
            "quick_thresholds": {
                "total": float(args.benchmark_quick_threshold_total),
                "linear": float(args.benchmark_quick_threshold_linear),
                "peak_overhead_limit": QUICK_BENCHMARK_PEAK_OVERHEAD_LIMIT,
            },
            "quick_eval": {
                "total_improvement_ratio": float(total_improvement),
                "linear_improvement_ratio": float(linear_improvement),
                "peak_memory_delta_ratio": float(peak_delta),
            },
            "quick_pass": quick_pass,
        }
    else:
        refinements = sorted(
            {
                int(token.strip())
                for token in str(args.refinements).split(",")
                if token.strip()
            }
        )
        if not refinements:
            raise ValueError("At least one refinement must be provided.")

        results: dict[str, Any] = {}
        for refinement in refinements:
            ref_profiles = _run_profiles(
                benchmark_script=benchmark_script,
                cache_root=cache_root,
                mesh_dir=args.mesh_dir.resolve(),
                profiles=profiles_all,
                refinement=refinement,
                repeat=int(args.repeat),
                run_diff=True,
                run_absolute=True,
            )
            results[f"ref_{refinement}"] = {
                "profiles": ref_profiles,
                "speedup_vs_A": _compute_speedup_block(ref_profiles),
            }

        payload = {
            "benchmark_phase": "full",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "repeat": int(args.repeat),
            "refinements": refinements,
            "profiles": profiles_all,
            "primary_profile": PROFILE_D_COMBINED,
            "experimental_profiles": list(EXPERIMENTAL_PERF_PROFILES),
            "results": results,
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(payload), encoding="utf-8")

    print(f"[OK] Fair compare JSON: {args.output_json}")
    print(f"[OK] Fair compare Markdown: {args.output_md}")
    if payload.get("benchmark_phase") == "quick":
        print(f"[OK] Quick pass: {bool(payload.get('quick_pass', False))}")


if __name__ == "__main__":
    main()
