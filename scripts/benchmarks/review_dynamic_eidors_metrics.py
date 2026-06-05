#!/usr/bin/env python3
"""Review dynamic sweep conclusions with EIDORS-aligned metrics."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyeidors.io._json import json_ready as _json_ready  # noqa: E402
from pyeidors.runtime_paths import pyeidors_output_path  # noqa: E402


SCHEMA = "pyeidors-dynamic-eidors-metric-review-v1"
OFFICIAL_METRIC_ORDER = (
    "AR_error",
    "PE",
    "RES",
    "SD",
    "RNG",
    "NF",
    "solution_error",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        action="append",
        type=Path,
        required=True,
        help="Dynamic T65/T66/T67 sweep JSON with EIDORS-aligned metric fields.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=pyeidors_output_path(
            "runtime_benchmarks",
            "dynamic_eidors_metric_review_20260426.json",
        ),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=pyeidors_output_path(
            "runtime_benchmarks",
            "dynamic_eidors_metric_review_20260426.md",
        ),
    )
    return parser.parse_args(argv)


def review_reports(paths: Sequence[Path]) -> dict[str, Any]:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    scenario_reviews = [
        _review_one_report(path=Path(path), payload=payload)
        for path, payload in zip(paths, reports, strict=True)
    ]
    method_counter: Counter[str] = Counter()
    metric_counter: Counter[str] = Counter()
    legacy_counter: Counter[str] = Counter()
    for scenario in scenario_reviews:
        legacy_counter[scenario["legacy_best_method"]] += 1
        for metric, winner in scenario["official_metric_winners"].items():
            method_counter[winner["method_family"]] += 1
            metric_counter[metric] += 1
    propagation_gate_counter = Counter(
        "passed" if scenario["propagation_aware_A_gate"]["passed"] else "failed"
        for scenario in scenario_reviews
        if scenario["propagation_aware_A_gate"]["enabled"]
    )
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_reports": [str(path) for path in paths],
        "scenario_count": len(scenario_reviews),
        "scenarios": scenario_reviews,
        "summary": {
            "legacy_best_method_counts": dict(legacy_counter),
            "official_metric_winner_counts": dict(method_counter),
            "official_metric_counts": dict(metric_counter),
            "propagation_aware_A_gate_counts": dict(propagation_gate_counter),
            "review_statement": _review_statement(scenario_reviews),
        },
    }


def write_payload(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target


def write_markdown(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_markdown(payload), encoding="utf-8")
    return target


def _review_one_report(*, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = _candidate_rows(payload)
    if not rows:
        raise ValueError(f"no method rows found in {path}")
    official_metric_winners = {
        metric: _best_by_metric(rows, metric) for metric in OFFICIAL_METRIC_ORDER
    }
    legacy_best = payload["summary"]["best_overall_by_fast_conduction_score"]
    legacy_family = _method_family(str(legacy_best["method"]))
    official_counts = Counter(
        winner["method_family"] for winner in official_metric_winners.values()
    )
    return {
        "source": str(path),
        "config": {
            "fixture": payload["config"]["fixture"],
            "domain": payload["config"]["domain"],
            "noise_std": payload["config"]["noise_std"],
            "seed": payload["config"]["seed"],
        },
        "candidate_count": len(rows),
        "legacy_best_method": legacy_family,
        "legacy_best": legacy_best,
        "official_metric_winners": official_metric_winners,
        "official_winner_counts": dict(official_counts),
        "old_conclusion_supported_by_official_majority": official_counts[legacy_family]
        >= 4,
        "propagation_aware_A_gate": _propagation_aware_A_gate(rows),
    }


def _candidate_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("t65_l2_by_lambda_t", {}).values():
        rows.append(
            _candidate(
                "t65_spatiotemporal_l2", item, params={"lambda_t": item["lambda_t"]}
            )
        )
    for item in payload.get("t66_rows", []):
        rows.append(
            _candidate(
                "t66_spatiotemporal_tv_huber",
                item,
                params={
                    "lambda_t": item["lambda_t"],
                    "huber_delta": item["huber_delta"],
                },
            )
        )
    for item in payload.get("t67_kalman_rows", []):
        rows.append(
            _candidate(
                "t67_kalman_fixed_lag",
                item,
                params={
                    "transition_kind": item.get("transition_kind", "identity"),
                    "transition_velocity": item.get("transition_velocity"),
                    "fixed_lag": item["fixed_lag"],
                    "process_noise": item["process_noise"],
                    "measurement_noise": item["measurement_noise"],
                },
            )
        )
    return rows


def _candidate(
    method: str, item: Mapping[str, Any], *, params: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = item.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{method} row is missing nested metrics.")
    fom = metrics.get("eidors_greit_figures_of_merit") or metrics.get("spatial_metrics")
    if not isinstance(fom, Mapping):
        raise ValueError(f"{method} row is missing EIDORS GREIT figures of merit.")
    required = ("AR", "PE", "RES", "SD", "RNG")
    missing = [key for key in required if key not in fom]
    if missing:
        raise ValueError(f"{method} row missing GREIT metric(s): {missing}")
    if "eidors_noise_figure" not in metrics:
        raise ValueError(f"{method} row missing EIDORS noise figure.")
    if "eidors_solution_error" not in metrics:
        raise ValueError(f"{method} row missing EIDORS solution error.")
    return {
        "method": method,
        "method_family": _method_family(method),
        "params": dict(params),
        "legacy_fast_conduction_score": float(item["fast_conduction_score"]),
        "rmse": float(metrics["rmse"]),
        "propagation_speed_abs_error": float(metrics["propagation_speed_abs_error"]),
        "peak_time_mean_abs_error": float(metrics["peak_time_mean_abs_error"]),
        "onset_time_mean_abs_error": float(metrics["onset_time_mean_abs_error"]),
        "official_metrics": {
            "AR": float(fom["AR"]),
            "AR_error": abs(float(fom["AR"]) - 1.0),
            "PE": float(fom["PE"]),
            "RES": float(fom["RES"]),
            "SD": float(fom["SD"]),
            "RNG": float(fom["RNG"]),
            "NF": float(metrics["eidors_noise_figure"]),
            "solution_error": float(metrics["eidors_solution_error"]),
            "clean_solution_error": float(
                metrics.get("eidors_clean_solution_error", np.nan)
            ),
        },
    }


def _best_by_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    best = min(rows, key=lambda row: float(row["official_metrics"][metric]))
    return {
        "metric": metric,
        "method": best["method"],
        "method_family": best["method_family"],
        "params": best["params"],
        "value": float(best["official_metrics"][metric]),
        "legacy_fast_conduction_score": float(best["legacy_fast_conduction_score"]),
        "rmse": float(best["rmse"]),
        "speed_error": float(best["propagation_speed_abs_error"]),
    }


def _propagation_aware_A_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    t67_rows = [row for row in rows if str(row["method"]).startswith("t67")]
    propagation_rows = [
        row for row in t67_rows if row["params"].get("transition_kind") == "propagation"
    ]
    identity_rows = [
        row for row in t67_rows if row["params"].get("transition_kind") == "identity"
    ]
    threshold = len(OFFICIAL_METRIC_ORDER) // 2 + 1
    if not propagation_rows or not identity_rows:
        return {
            "enabled": False,
            "passed": False,
            "threshold": int(threshold),
            "metric_count": int(len(OFFICIAL_METRIC_ORDER)),
            "reason": "requires both identity and propagation T67 rows",
            "identity_row_count": int(len(identity_rows)),
            "propagation_row_count": int(len(propagation_rows)),
        }
    contenders = identity_rows + propagation_rows
    winners = {
        metric: _best_by_metric(contenders, metric) for metric in OFFICIAL_METRIC_ORDER
    }
    transition_counts = Counter(
        str(winner["params"].get("transition_kind", "identity"))
        for winner in winners.values()
    )
    deltas = {}
    for metric in OFFICIAL_METRIC_ORDER:
        best_identity = _best_by_metric(identity_rows, metric)
        best_propagation = _best_by_metric(propagation_rows, metric)
        deltas[metric] = {
            "identity_minus_propagation": float(best_identity["value"])
            - float(best_propagation["value"]),
            "best_identity": best_identity,
            "best_propagation": best_propagation,
        }
    passed = transition_counts["propagation"] >= threshold
    return {
        "enabled": True,
        "passed": bool(passed),
        "threshold": int(threshold),
        "metric_count": int(len(OFFICIAL_METRIC_ORDER)),
        "admission_policy": "pass iff propagation wins >=4/7 lower-is-better EIDORS-aligned metrics among T67 identity/propagation rows",
        "identity_row_count": int(len(identity_rows)),
        "propagation_row_count": int(len(propagation_rows)),
        "transition_winner_counts": dict(transition_counts),
        "official_metric_winners": winners,
        "metric_deltas_identity_minus_propagation": deltas,
        "review_statement": "propagation-aware A admitted for this scenario"
        if passed
        else "propagation-aware A kept experimental for this scenario",
    }


def _method_family(method: str) -> str:
    if method.startswith("t65"):
        return "T65"
    if method.startswith("t66"):
        return "T66"
    if method.startswith("t67"):
        return "T67"
    return method


def _review_statement(scenarios: Sequence[Mapping[str, Any]]) -> str:
    unsupported = [
        scenario
        for scenario in scenarios
        if not bool(scenario["old_conclusion_supported_by_official_majority"])
    ]
    if not unsupported:
        return "Previous dynamic-score conclusions are broadly supported by the EIDORS-aligned metric majority, but individual metric trade-offs remain visible."
    return "At least one previous dynamic-score conclusion is not supported by the EIDORS-aligned metric majority; inspect per-metric winners before promoting a method."


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Dynamic EIDORS Metric Review",
        "",
        f"- schema: `{payload['schema']}`",
        f"- created_utc: `{payload['created_utc']}`",
        f"- scenarios: `{payload['scenario_count']}`",
        f"- statement: {payload['summary']['review_statement']}",
        "",
        "## Scenario Summary",
        "",
        "| source | noise | seed | legacy best | official winner counts | official majority supports legacy | propagation-A gate |",
        "|---|---:|---:|---|---|:---:|---|",
    ]
    for scenario in payload["scenarios"]:
        lines.append(
            "| {source} | {noise} | {seed} | {legacy} | {counts} | {supported} | {propagation_gate} |".format(
                source=Path(scenario["source"]).name,
                noise=_fmt(scenario["config"]["noise_std"]),
                seed=scenario["config"]["seed"],
                legacy=scenario["legacy_best_method"],
                counts=", ".join(
                    f"{key}:{value}"
                    for key, value in sorted(scenario["official_winner_counts"].items())
                ),
                supported="yes"
                if scenario["old_conclusion_supported_by_official_majority"]
                else "no",
                propagation_gate=_propagation_gate_cell(
                    scenario["propagation_aware_A_gate"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Metric Winners",
            "",
            "| source | AR err | PE | RES | SD | RNG | NF | solution error |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for scenario in payload["scenarios"]:
        winners = scenario["official_metric_winners"]
        lines.append(
            "| {source} | {ar} | {pe} | {res} | {sd} | {rng} | {nf} | {se} |".format(
                source=Path(scenario["source"]).name,
                ar=_winner_cell(winners["AR_error"]),
                pe=_winner_cell(winners["PE"]),
                res=_winner_cell(winners["RES"]),
                sd=_winner_cell(winners["SD"]),
                rng=_winner_cell(winners["RNG"]),
                nf=_winner_cell(winners["NF"]),
                se=_winner_cell(winners["solution_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- EIDORS does not define a single scalar score for AR/PE/RES/SD/RNG/NF/solution-error, so this report reviews per-metric winners instead of replacing them with RMSE.",
            "- `AR err` is `abs(AR - 1)`. Lower is better for every column in this review.",
            "- The previous dynamic score is still useful for propagation timing, but it is no longer treated as the whole quality story.",
            "",
        ]
    )
    propagation_scenarios = [
        scenario
        for scenario in payload["scenarios"]
        if scenario["propagation_aware_A_gate"]["enabled"]
    ]
    if propagation_scenarios:
        lines.extend(
            [
                "## Propagation-A Gate",
                "",
                "| source | winner counts | pass | AR err | PE | RES | SD | RNG | NF | solution error |",
                "|---|---|:---:|---|---|---|---|---|---|---|",
            ]
        )
        for scenario in propagation_scenarios:
            gate = scenario["propagation_aware_A_gate"]
            winners = gate["official_metric_winners"]
            lines.append(
                "| {source} | {counts} | {passed} | {ar} | {pe} | {res} | {sd} | {rng} | {nf} | {se} |".format(
                    source=Path(scenario["source"]).name,
                    counts=", ".join(
                        f"{key}:{value}"
                        for key, value in sorted(
                            gate["transition_winner_counts"].items()
                        )
                    ),
                    passed="yes" if gate["passed"] else "no",
                    ar=_winner_cell(winners["AR_error"]),
                    pe=_winner_cell(winners["PE"]),
                    res=_winner_cell(winners["RES"]),
                    sd=_winner_cell(winners["SD"]),
                    rng=_winner_cell(winners["RNG"]),
                    nf=_winner_cell(winners["NF"]),
                    se=_winner_cell(winners["solution_error"]),
                )
            )
        lines.extend(
            [
                "",
                "- Propagation-A gate compares only T67 identity vs T67 propagation rows; promotion requires a 4/7 majority across the EIDORS-aligned metrics above.",
                "",
            ]
        )
    return "\n".join(lines)


def _winner_cell(winner: Mapping[str, Any]) -> str:
    label = str(winner["method_family"])
    params = winner.get("params", {})
    if label == "T67" and isinstance(params, Mapping):
        transition = params.get("transition_kind")
        if transition:
            label = f"{label}/{transition}"
            velocity = params.get("transition_velocity")
            if velocity is not None:
                label = f"{label}@v={_fmt(velocity)}"
    return f"{label} {_fmt(winner['value'])}"


def _propagation_gate_cell(gate: Mapping[str, Any]) -> str:
    if not gate["enabled"]:
        return "n/a"
    counts = ", ".join(
        f"{key}:{value}"
        for key, value in sorted(gate["transition_winner_counts"].items())
    )
    status = "pass" if gate["passed"] else "hold"
    return f"{status} ({counts})"


def _fmt(value: Any) -> str:
    return f"{float(value):.6g}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = review_reports(args.input_json)
    json_path = write_payload(args.output_json, payload)
    md_path = write_markdown(args.output_md, payload)
    print(f"[OK] EIDORS metric review saved: {json_path}")
    print(f"[OK] EIDORS metric review report saved: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
