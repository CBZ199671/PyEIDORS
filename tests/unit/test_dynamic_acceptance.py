from __future__ import annotations

import json

from eit_app.dynamic_acceptance import (
    MAXIMUM_PEAK_TIME_ERROR_BLOCKS,
    MAXIMUM_STEP_BIAS,
    MINIMUM_ISOLATED_SUPPRESSION,
    REQUIRED_TOTAL_LATENCY_FRAMES,
    build_dynamic_acceptance_report,
    main,
)


def test_v681_dynamic_acceptance_covers_required_sequences_and_thresholds() -> None:
    report = build_dynamic_acceptance_report()

    assert report["passed"] is True
    assert report["isolated_suppression"] >= MINIMUM_ISOLATED_SUPPRESSION
    assert report["step_steady_state_bias"] < MAXIMUM_STEP_BIAS
    assert report["maximum_peak_time_error_blocks"] <= MAXIMUM_PEAK_TIME_ERROR_BLOCKS
    assert report["total_latency_frames"] == [REQUIRED_TOTAL_LATENCY_FRAMES]
    assert set(report["candidate_gate_actions"]) & {"inflate", "reject"}
    assert report["noncandidate_step_actions"] == ["update"]
    assert report["dropout_max_block_step"] == 2
    assert report["session_reset"]["passed"] is True
    assert {item["name"] for item in report["scenarios"]} == {
        "positive_isolated_spike",
        "negative_isolated_spike",
        "sustained_step",
        "three_frame_pulse",
        "continuous_ramp",
        "biphasic_response",
        "dropout_gap",
    }


def test_v681_dynamic_acceptance_cli_writes_traceable_json(tmp_path) -> None:
    output = tmp_path / "dynamic-acceptance.json"

    assert main(["--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == "pyeidors-dynamic-acceptance-v1"
    assert report["algorithm_schema"].startswith(
        "pyeidors-dynamic-measurement-diagonal-session"
    )
    assert report["passed"] is True
