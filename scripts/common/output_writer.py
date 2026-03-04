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


def _aggregate_cache_summary(results: List[CaseResult]) -> Dict[str, Any]:
    layer_counts: Dict[str, int] = {}
    latest_stats: Dict[str, Any] = {}
    for result in results:
        if not isinstance(result.metrics, dict):
            continue
        _gather_cache_layers(result.metrics, layer_counts)
        cache_stats = result.metrics.get("cache_stats")
        if isinstance(cache_stats, dict):
            latest_stats = cache_stats
    return {
        "layer_hits": layer_counts,
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
