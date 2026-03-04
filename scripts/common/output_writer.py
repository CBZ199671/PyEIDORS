"""Output writers for unified reconstruction execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .recon_cli_models import CaseResult, ReconstructionCase, ReconstructionMethod


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
