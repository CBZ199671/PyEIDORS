"""Output writers for unified reconstruction execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .recon_cli_models import CaseResult, ReconstructionCase, ReconstructionMethod


def _bump(counter: Dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _gather_cache_layers(metrics: Dict[str, Any], counter: Dict[str, int]) -> None:
    lookups = metrics.get("cache_lookups")
    if not isinstance(lookups, dict):
        return
    context = lookups.get("context")
    if isinstance(context, dict):
        for value in context.values():
            if isinstance(value, dict):
                layer = value.get("layer")
                if isinstance(layer, str):
                    _bump(counter, layer)
            elif isinstance(value, str):
                _bump(counter, value)
    forward_factor = lookups.get("forward_factor")
    if isinstance(forward_factor, dict):
        layer = forward_factor.get("layer")
        if isinstance(layer, str):
            _bump(counter, layer)


def _gather_miss_reasons(metrics: Dict[str, Any], counter: Dict[str, int]) -> None:
    reasons = metrics.get("cache_miss_reasons")
    if not isinstance(reasons, dict):
        return
    for reason in reasons.values():
        if isinstance(reason, str):
            _bump(counter, reason)


def _gather_build_seconds(metrics: Dict[str, Any], total: Dict[str, float]) -> None:
    build = metrics.get("cache_build_seconds")
    if not isinstance(build, dict):
        return
    for key, value in build.items():
        if isinstance(key, str) and isinstance(value, (int, float)):
            total[key] = total.get(key, 0.0) + float(value)


def _aggregate_cache_summary(results: List[CaseResult]) -> Dict[str, Any]:
    layer_counts: Dict[str, int] = {}
    miss_reasons: Dict[str, int] = {}
    build_seconds: Dict[str, float] = {}
    stage_timings: Dict[str, float] = {}
    latest_stats: Dict[str, Any] = {}
    jacobian_block_size_selected: Dict[str, int] = {}
    jacobian_tune_source: Dict[str, int] = {}
    jacobian_assembly_elapsed_only_total = 0.0
    jacobian_assembly_elapsed_only_count = 0
    rom_enabled_effective_count = 0
    rom_rank_effective_max = 0
    lowrank_rank_effective_max = 0
    inexact_eta_history: list[float] = []
    degrade_stage_counts: Dict[str, int] = {}
    effective_solver_path_counts: Dict[str, int] = {}
    for result in results:
        if not isinstance(result.metrics, dict):
            continue
        _gather_cache_layers(result.metrics, layer_counts)
        _gather_miss_reasons(result.metrics, miss_reasons)
        _gather_build_seconds(result.metrics, build_seconds)
        stage = result.metrics.get("stage_timings")
        if isinstance(stage, dict):
            for key, value in stage.items():
                if isinstance(key, str) and isinstance(value, (int, float)):
                    stage_timings[key] = stage_timings.get(key, 0.0) + float(value)
        diagnostics = result.metrics.get("diagnostics")
        if isinstance(diagnostics, dict):
            backend = diagnostics.get("backend_info")
            if isinstance(backend, dict):
                tuning = backend.get("jacobian_block_tune")
                if isinstance(tuning, dict):
                    selected = tuning.get("selected_block_size")
                    if isinstance(selected, int):
                        selected_key = str(selected)
                        jacobian_block_size_selected[selected_key] = (
                            jacobian_block_size_selected.get(selected_key, 0) + 1
                        )
                    source = tuning.get("tune_source")
                    if isinstance(source, str):
                        jacobian_tune_source[source] = (
                            jacobian_tune_source.get(source, 0) + 1
                        )
                    assembly_elapsed = tuning.get("assembly_elapsed_only")
                    if isinstance(assembly_elapsed, (int, float)):
                        jacobian_assembly_elapsed_only_total += float(assembly_elapsed)
                        jacobian_assembly_elapsed_only_count += 1
                backend_assembly = backend.get("jacobian_assembly_elapsed_only")
                if isinstance(backend_assembly, (int, float)):
                    jacobian_assembly_elapsed_only_total += float(backend_assembly)
                    jacobian_assembly_elapsed_only_count += 1
                if bool(backend.get("rom_enabled_effective", False)):
                    rom_enabled_effective_count += 1
                rom_rank_val = backend.get("rom_rank_effective")
                if isinstance(rom_rank_val, int):
                    rom_rank_effective_max = max(
                        rom_rank_effective_max, int(rom_rank_val)
                    )
                lowrank_rank_val = backend.get("lowrank_rank_effective")
                if isinstance(lowrank_rank_val, int):
                    lowrank_rank_effective_max = max(
                        lowrank_rank_effective_max, int(lowrank_rank_val)
                    )
                eta_hist = backend.get("inexact_eta_history")
                if isinstance(eta_hist, list):
                    for eta in eta_hist:
                        if isinstance(eta, (int, float)):
                            inexact_eta_history.append(float(eta))
                degrade_counts = backend.get("degrade_stage_counts")
                if isinstance(degrade_counts, dict):
                    for key, value in degrade_counts.items():
                        if isinstance(key, str) and isinstance(value, (int, float)):
                            degrade_stage_counts[key] = degrade_stage_counts.get(
                                key, 0
                            ) + int(value)
                path_counts = backend.get("effective_solver_path_counts")
                if isinstance(path_counts, dict):
                    for key, value in path_counts.items():
                        if isinstance(key, str) and isinstance(value, (int, float)):
                            effective_solver_path_counts[key] = (
                                effective_solver_path_counts.get(key, 0) + int(value)
                            )
        cache_stats = result.metrics.get("cache_stats")
        if isinstance(cache_stats, dict):
            latest_stats = cache_stats
    return {
        "layer_hits": layer_counts,
        "miss_reasons": miss_reasons,
        "build_seconds_total": build_seconds,
        "stage_timings_total": stage_timings,
        "jacobian_block_size_selected": jacobian_block_size_selected,
        "jacobian_tune_source": jacobian_tune_source,
        "jacobian_assembly_elapsed_only_total": jacobian_assembly_elapsed_only_total,
        "jacobian_assembly_elapsed_only_avg": (
            jacobian_assembly_elapsed_only_total / jacobian_assembly_elapsed_only_count
            if jacobian_assembly_elapsed_only_count
            else 0.0
        ),
        "rom_enabled_effective_count": int(rom_enabled_effective_count),
        "rom_rank_effective_max": int(rom_rank_effective_max),
        "lowrank_rank_effective_max": int(lowrank_rank_effective_max),
        "inexact_eta_history": inexact_eta_history[-32:],
        "degrade_stage_counts": degrade_stage_counts,
        "effective_solver_path_counts": effective_solver_path_counts,
        "latest_cache_stats": latest_stats,
    }


def write_batch_summary(
    *,
    method: ReconstructionMethod,
    output_root: Path,
    cases: Iterable[ReconstructionCase],
    results: List[CaseResult],
    config: Dict[str, Any],
) -> Path:
    """Persist batch summary JSON and return its path."""
    output_root.mkdir(parents=True, exist_ok=True)

    case_list = list(cases)
    total = len(case_list)
    processed = sum(1 for r in results if r.status == "success")
    skipped = sum(1 for r in results if r.status == "skipped")
    failed = sum(1 for r in results if r.status == "failed")

    payload = {
        "method": method.value,
        "total": total,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "config": config,
        "results": [result.to_dict() for result in results],
        "cache_summary": _aggregate_cache_summary(results),
    }

    summary_path = output_root / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return summary_path


def format_dry_run(cases: Iterable[ReconstructionCase]) -> str:
    """Return a human-readable dry-run summary string."""
    lines = []
    for case in cases:
        lines.append(json.dumps(case.to_dict(), ensure_ascii=False))
    return "\n".join(lines)
