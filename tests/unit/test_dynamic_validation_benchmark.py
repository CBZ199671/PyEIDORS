"""Contract tests for the dynamic validation benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from pyeidors.inverse import GREIT_METRIC_KEYS


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmarks"
    / "benchmark_dynamic_validation.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_dynamic_validation", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise AssertionError(f"failed to load script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dynamic_validation_benchmark_reports_required_dynamic_metrics(
    tmp_path: Path,
) -> None:
    module = _load_module()

    payload = module.run_benchmark(
        n_cells=12,
        n_frames=10,
        n_measurements=8,
        lambda_=0.10,
        ridge=1.0e-8,
        noise_std=1.0e-4,
        temporal_alpha=0.80,
        lambda_t=0.12,
        temporal_order=2,
        peak_delay_tolerance=0.30,
        seed=321,
    )
    output = module.write_payload(tmp_path / "dynamic_validation.json", payload)
    report = module.write_markdown_report(
        tmp_path / "dynamic_validation_4d_gn_vs_rowwise_rm.md",
        payload,
    )
    saved = json.loads(output.read_text(encoding="utf-8"))
    report_text = report.read_text(encoding="utf-8")

    assert saved["schema"] == module.SCHEMA
    assert set(saved["fixtures"]) == {"travelling_wave", "plant_slow_pulse"}
    assert saved["gate"]["passed"] is True
    assert saved["summary"]["fixture_count"] == 2
    assert saved["summary"]["method_count"] == 6
    assert saved["summary"]["spatiotemporal_4d_vs_rowwise_rm"]["enabled"] is True
    assert "4D GN vs Rowwise RM" in report_text
    assert "travelling_wave" in report_text

    for fixture in saved["fixtures"].values():
        assert (
            fixture["sequence"]["schema"] == "pyeidors-dynamic-measurement-sequence-v1"
        )
        assert fixture["truth_shape"] == [10, 12]
        assert set(fixture["methods"]) == {
            "rm_raw",
            "measurement_ema",
            "spatiotemporal_gn_4d",
        }
        for method_name, method in fixture["methods"].items():
            fidelity = method["fidelity"]
            for key in (
                "onset_time_mean_abs_error",
                "peak_time_mean_abs_error",
                "propagation_speed_abs_error",
                "amplitude_attenuation",
                "snr_gain_db",
            ):
                assert np.isfinite(float(fidelity[key]))
            assert set(fidelity["spatial_metrics"]) == set(GREIT_METRIC_KEYS)
            assert method["cold_metadata"]["offline_rm_build_seconds"] >= 0.0
            online = method["online_metadata"]
            assert online["online_rm_apply_seconds"] >= 0.0
            assert online["forward_solve_count"] == 0
            assert online["adjoint_solve_count"] == 0
            assert online["ksp_solve_count"] == 0
            assert online["jacobian_rebuild_count"] == 0
            if method_name == "spatiotemporal_gn_4d":
                assert method["rowwise_rm_comparison"]["enabled"] is True
                assert method["rowwise_rm_baseline"]["enabled"] is True
                assert method["cold_metadata"]["lambda_t"] == 0.12
                assert method["cold_metadata"]["temporal_order"] == 2
                assert "rowwise_rm_fidelity" in method


def test_peak_delay_gate_reports_violations() -> None:
    module = _load_module()
    fixtures = {
        "wave": {
            "methods": {
                "raw": {"fidelity": {"peak_time_max_positive_delay": 0.02}},
                "slow": {"fidelity": {"peak_time_max_positive_delay": 0.25}},
            }
        }
    }

    gate = module.evaluate_peak_delay_gate(fixtures, peak_delay_tolerance=0.10)

    assert gate["passed"] is False
    assert gate["max_peak_time_positive_delay"] == 0.25
    assert gate["violations"] == [
        {
            "fixture": "wave",
            "method": "slow",
            "peak_time_max_positive_delay": 0.25,
        }
    ]


def test_dynamic_validation_fixture_input_validation() -> None:
    module = _load_module()
    for kwargs, message in (
        ({"n_cells": 3, "n_frames": 4, "n_measurements": 4}, "n_cells"),
        ({"n_cells": 4, "n_frames": 3, "n_measurements": 4}, "n_frames"),
        ({"n_cells": 4, "n_frames": 4, "n_measurements": 1}, "n_measurements"),
    ):
        try:
            module.build_travelling_wave_fixture(
                **kwargs,
                noise_std=0.0,
                seed=1,
            )
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover - assertion clarity
            raise AssertionError(f"expected ValueError containing {message!r}")
