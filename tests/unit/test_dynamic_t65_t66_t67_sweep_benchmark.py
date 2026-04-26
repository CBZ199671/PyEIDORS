"""Contract tests for the T65/T66/T67 dynamic sweep benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "benchmark_dynamic_tv_huber_sweep.py"
)
REVIEW_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "review_dynamic_eidors_metrics.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_dynamic_tv_huber_sweep_t67", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_review_module():
    spec = importlib.util.spec_from_file_location(
        "review_dynamic_eidors_metrics_t67", REVIEW_SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {REVIEW_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dynamic_sweep_reports_t67_kalman_lag_qr_comparison(
    tmp_path: Path,
) -> None:
    module = _load_module()

    payload = module.run_sweep(
        n_cells=8,
        n_frames=7,
        n_measurements=5,
        lambda_s=0.05,
        lambda_t_values=(0.04,),
        huber_delta_values=(0.03,),
        temporal_order=2,
        noise_std=1.0e-4,
        seed=20260426,
        max_outer_iterations=3,
        peak_delay_limit=0.50,
        kalman_lag_values=(0, 2),
        kalman_process_noise_values=(0.02,),
        kalman_measurement_noise_values=(0.04, 0.08),
    )
    output = module.write_payload(tmp_path / "dynamic_t65_t66_t67.json", payload)
    report = module.write_markdown_report(
        tmp_path / "dynamic_t65_t66_t67.md",
        payload,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    report_text = report.read_text(encoding="utf-8")

    assert saved["schema"] == module.SCHEMA
    assert len(saved["t66_rows"]) == 1
    assert len(saved["t67_kalman_rows"]) == 4
    assert saved["summary"]["t65_row_count"] == 1
    assert saved["summary"]["t66_row_count"] == 1
    assert saved["summary"]["t67_row_count"] == 4
    assert "best_t67_kalman" in saved["summary"]
    assert "recommended_kalman_process_noise_range" in saved["summary"]
    assert "T67 Kalman" in report_text
    assert "Q" in report_text
    assert "R" in report_text

    best_t67 = saved["summary"]["best_t67_kalman"]
    assert best_t67["fixed_lag"] in {0, 2}
    assert best_t67["process_noise"] == 0.02
    assert best_t67["measurement_noise"] in {0.04, 0.08}
    assert np.isfinite(best_t67["speed_error_t67"])
    assert np.isfinite(best_t67["fast_conduction_score"])

    comparison = saved["summary"]["comparison"]["best_t67_vs_best_t66"]
    assert np.isfinite(comparison["speed_error_delta_reference_minus_method"])
    for row in saved["t67_kalman_rows"]:
        metadata = row["metadata"]
        assert metadata["schema"].endswith("kalman-fixed-lag-v1")
        assert metadata["online_hot_path"] == "rm_observation_plus_kalman"
        assert metadata["default_enabled"] is False
        assert metadata["forward_solve_count"] == 0
        assert metadata["adjoint_solve_count"] == 0
        assert metadata["ksp_solve_count"] == 0


def test_eidors_metric_review_rechecks_dynamic_sweep_payload(
    tmp_path: Path,
) -> None:
    sweep = _load_module()
    review = _load_review_module()
    payload = sweep.run_sweep(
        n_cells=8,
        n_frames=7,
        n_measurements=5,
        lambda_s=0.05,
        lambda_t_values=(0.04,),
        huber_delta_values=(0.03,),
        temporal_order=2,
        noise_std=1.0e-4,
        seed=20260426,
        max_outer_iterations=3,
        peak_delay_limit=0.50,
        kalman_lag_values=(0,),
        kalman_process_noise_values=(0.02,),
        kalman_measurement_noise_values=(0.04,),
    )
    source = sweep.write_payload(tmp_path / "dynamic_t65_t66_t67.json", payload)

    reviewed = review.review_reports([source])
    report = review.write_markdown(tmp_path / "eidors_review.md", reviewed)

    assert reviewed["schema"] == review.SCHEMA
    assert reviewed["scenario_count"] == 1
    scenario = reviewed["scenarios"][0]
    assert set(scenario["official_metric_winners"]) == set(review.OFFICIAL_METRIC_ORDER)
    for winner in scenario["official_metric_winners"].values():
        assert winner["method_family"] in {"T65", "T66", "T67"}
        assert np.isfinite(winner["value"])
    assert "Per-Metric Winners" in report.read_text(encoding="utf-8")
