"""Deterministic sequence acceptance for the persistent realtime Kalman path."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from pyeidors.inverse.dynamic_session import (
    DiagonalKalmanConfig,
    PersistentDiagonalKalmanRegistry,
    PersistentMeasurementDiagonalKalmanSession,
)


SCHEMA_VERSION = "pyeidors-dynamic-acceptance-v1"
STATE_SIZE = 8
TEMPORAL_BAD_WEIGHT = 0.02
MINIMUM_ISOLATED_SUPPRESSION = 0.90
MAXIMUM_STEP_BIAS = 0.05
MAXIMUM_PEAK_TIME_ERROR_BLOCKS = 2
REQUIRED_TOTAL_LATENCY_FRAMES = 2


def build_dynamic_acceptance_report() -> dict[str, Any]:
    config = DiagonalKalmanConfig(upstream_latency_frames=REQUIRED_TOTAL_LATENCY_FRAMES)
    scenarios = [
        _run_sequence(
            "positive_isolated_spike",
            [0.0] * 6 + [10.0] + [0.0] * 6,
            config,
            weights=[1.0] * 6 + [TEMPORAL_BAD_WEIGHT] + [1.0] * 6,
            candidates=[False] * 6 + [True] + [False] * 6,
        ),
        _run_sequence(
            "negative_isolated_spike",
            [0.0] * 6 + [-10.0] + [0.0] * 6,
            config,
            weights=[1.0] * 6 + [TEMPORAL_BAD_WEIGHT] + [1.0] * 6,
            candidates=[False] * 6 + [True] + [False] * 6,
        ),
        _run_sequence(
            "sustained_step",
            [0.0] * 5 + [1.0] * 15,
            config,
        ),
        _run_sequence(
            "three_frame_pulse",
            [0.0] * 5 + [0.4, 1.0, 0.4] + [0.0] * 6,
            config,
        ),
        _run_sequence(
            "continuous_ramp",
            [*np.linspace(0.0, 1.0, 11), *np.linspace(0.9, 0.0, 10)],
            config,
        ),
        _run_sequence(
            "biphasic_response",
            [0.0, 0.0, 0.2, 0.5, 1.0, 0.6, 0.1, -0.2, -0.8, -1.2, -0.7, -0.2, 0.0, 0.0],
            config,
        ),
        _run_sequence(
            "dropout_gap",
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            config,
            blocks=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11],
        ),
    ]
    by_name = {str(item["name"]): item for item in scenarios}
    positive_suppression = _isolated_suppression(by_name["positive_isolated_spike"])
    negative_suppression = _isolated_suppression(by_name["negative_isolated_spike"])
    isolated_suppression = min(positive_suppression, negative_suppression)
    step = by_name["sustained_step"]
    step_bias = abs(float(np.mean(step["filtered_values"][-5:])) - 1.0)
    ramp_peak_error = _peak_time_error(by_name["continuous_ramp"])
    biphasic_peak_error = _peak_time_error(by_name["biphasic_response"])
    peak_time_error = max(ramp_peak_error, biphasic_peak_error)
    latency_values = {
        int(value)
        for scenario in scenarios
        for value in scenario["total_latency_frames"]
    }
    candidate_actions = {
        action
        for name in ("positive_isolated_spike", "negative_isolated_spike")
        for action, candidate in zip(
            by_name[name]["actions"],
            by_name[name]["innovation_candidates"],
            strict=True,
        )
        if candidate
    }
    noncandidate_step_actions = set(step["actions"][5:])
    dropout = by_name["dropout_gap"]
    session_reset = _session_reset_acceptance(config)
    checks = {
        "isolated_suppression": isolated_suppression >= MINIMUM_ISOLATED_SUPPRESSION,
        "step_bias": step_bias < MAXIMUM_STEP_BIAS,
        "peak_time": peak_time_error <= MAXIMUM_PEAK_TIME_ERROR_BLOCKS,
        "candidate_gate": bool(candidate_actions & {"inflate", "reject"}),
        "noncandidate_step_preserved": noncandidate_step_actions == {"update"},
        "multi_frame_pulse_preserved": set(by_name["three_frame_pulse"]["actions"])
        <= {"initialize", "update"},
        "dropout_gap": max(dropout["block_steps"]) == 2,
        "total_latency": latency_values == {REQUIRED_TOTAL_LATENCY_FRAMES},
        "session_reset": session_reset["passed"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "algorithm_schema": "pyeidors-dynamic-measurement-diagonal-session-v1",
        "mode": "measurement",
        "thresholds": {
            "minimum_isolated_suppression": MINIMUM_ISOLATED_SUPPRESSION,
            "maximum_step_bias": MAXIMUM_STEP_BIAS,
            "maximum_peak_time_error_blocks": MAXIMUM_PEAK_TIME_ERROR_BLOCKS,
            "required_total_latency_frames": REQUIRED_TOTAL_LATENCY_FRAMES,
            "temporal_bad_weight": TEMPORAL_BAD_WEIGHT,
        },
        "isolated_suppression": isolated_suppression,
        "positive_isolated_suppression": positive_suppression,
        "negative_isolated_suppression": negative_suppression,
        "step_steady_state_bias": step_bias,
        "ramp_peak_time_error_blocks": ramp_peak_error,
        "biphasic_peak_time_error_blocks": biphasic_peak_error,
        "maximum_peak_time_error_blocks": peak_time_error,
        "total_latency_frames": sorted(latency_values),
        "candidate_gate_actions": sorted(candidate_actions),
        "noncandidate_step_actions": sorted(noncandidate_step_actions),
        "dropout_max_block_step": max(dropout["block_steps"]),
        "session_reset": session_reset,
        "checks": checks,
        "scenarios": scenarios,
        "passed": all(checks.values()),
    }


def _run_sequence(
    name: str,
    values: list[float] | np.ndarray,
    config: DiagonalKalmanConfig,
    *,
    weights: list[float] | None = None,
    candidates: list[bool] | None = None,
    blocks: list[int] | None = None,
) -> dict[str, Any]:
    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    sample_count = int(observations.size)
    weight_values = np.ones(sample_count, dtype=np.float64)
    if weights is not None:
        weight_values = np.asarray(weights, dtype=np.float64).reshape(-1)
    candidate_values = [False] * sample_count if candidates is None else candidates
    block_values = list(range(1, sample_count + 1)) if blocks is None else blocks
    if not (
        weight_values.size == len(candidate_values) == len(block_values) == sample_count
    ):
        raise ValueError("Dynamic acceptance sequence arrays must have equal lengths.")

    session = PersistentMeasurementDiagonalKalmanSession(
        fingerprint=f"dynamic-acceptance:{name}",
        config=config,
    )
    model = np.eye(STATE_SIZE, dtype=np.float64)
    measurement_scale = np.ones(STATE_SIZE, dtype=np.float64)
    filtered: list[float] = []
    actions: list[str] = []
    nis_per_dof: list[float] = []
    variance_inflations: list[float] = []
    total_latency: list[int] = []
    block_steps: list[int] = []
    for value, weight, candidate, block in zip(
        observations,
        weight_values,
        candidate_values,
        block_values,
        strict=True,
    ):
        measurement = np.full(STATE_SIZE, value, dtype=np.float64)
        # The static inverse already consumes the same confidence weight before
        # the dynamic measurement update. This models the real EitHost pipeline.
        static_observation = measurement * float(weight)
        update = session.update(
            static_observation,
            measurement,
            model,
            measurement_scale=measurement_scale,
            measurement_weights=np.full(STATE_SIZE, weight, dtype=np.float64),
            block_number=int(block),
            innovation_candidate=bool(candidate),
        )
        metadata = update.metadata
        filtered.append(float(np.mean(update.state)))
        actions.append(str(metadata["action"]))
        nis_per_dof.append(float(metadata["innovation_nis_per_dof"]))
        variance_inflations.append(float(metadata["variance_inflation"]))
        total_latency.append(int(metadata["total_latency_frames"]))
        block_steps.append(int(metadata["block_step"]))

    return {
        "name": name,
        "blocks": [int(value) for value in block_values],
        "raw_values": [float(value) for value in observations],
        "static_weighted_values": [
            float(value * weight)
            for value, weight in zip(observations, weight_values, strict=True)
        ],
        "filtered_values": filtered,
        "measurement_weights": [float(value) for value in weight_values],
        "innovation_candidates": [bool(value) for value in candidate_values],
        "actions": actions,
        "nis_per_dof": nis_per_dof,
        "variance_inflations": variance_inflations,
        "total_latency_frames": total_latency,
        "block_steps": block_steps,
    }


def _isolated_suppression(scenario: dict[str, Any]) -> float:
    candidates = scenario["innovation_candidates"]
    candidate_index = next(index for index, value in enumerate(candidates) if value)
    raw = abs(float(scenario["raw_values"][candidate_index]))
    filtered = abs(float(scenario["filtered_values"][candidate_index]))
    return 1.0 if raw <= np.finfo(float).eps else 1.0 - filtered / raw


def _peak_time_error(scenario: dict[str, Any]) -> int:
    blocks = scenario["blocks"]
    raw_index = int(np.argmax(np.abs(scenario["raw_values"])))
    filtered_index = int(np.argmax(np.abs(scenario["filtered_values"])))
    return abs(int(blocks[raw_index]) - int(blocks[filtered_index]))


def _session_reset_acceptance(config: DiagonalKalmanConfig) -> dict[str, Any]:
    registry = PersistentDiagonalKalmanRegistry(max_sessions=4)
    first = registry.update(
        "set-a:ref0",
        np.zeros(STATE_SIZE),
        fingerprint="set-a;ref=0",
        config=config,
        block_number=1,
    )
    other = registry.update(
        "set-b:ref0",
        np.full(STATE_SIZE, 5.0),
        fingerprint="set-b;ref=0",
        config=config,
        block_number=1,
    )
    reset = registry.update(
        "set-a:ref0",
        np.full(STATE_SIZE, 2.0),
        fingerprint="set-a;ref=1",
        config=config,
        block_number=10,
        reset=True,
    )
    passed = (
        first.metadata["registry_action"] == "created"
        and other.metadata["registry_action"] == "created"
        and reset.metadata["registry_action"] == "reset"
        and np.array_equal(reset.state, np.full(STATE_SIZE, 2.0))
        and registry.session_count == 2
    )
    return {
        "passed": bool(passed),
        "session_count": int(registry.session_count),
        "reset_action": str(reset.metadata["registry_action"]),
        "reset_update_action": str(reset.metadata["action"]),
        "session_ids": list(registry.session_ids),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_dynamic_acceptance_report()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(args.output.resolve())
    return 0 if bool(report["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
