"""Tests for T15 controlled multi-factor sweeps."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys

import numpy as np

from pyeidors.data.factor_sweep import (
    CSV_FIELDS,
    format_factor_sweep_report,
    normalize_enob_level,
    plot_factor_sweep,
    run_factor_sweep,
)
from pyeidors.data.eit_digit_metrics import sigma_true_from_anomaly_rule


REPO_ROOT = Path(__file__).resolve().parents[2]


def _small_surrogate_rows():
    return run_factor_sweep(
        fem_grid_levels=[4, 6],
        ridge_levels=[1e-2, 1e-1],
        target_digits=[5, 6],
        noise_relative_levels=[0.0, 0.001],
        enob_levels=["nominal", "5"],
        forward_backend="surrogate",
        inverse_backend="least-squares",
        n_elec=16,
        baseline_fem_grid=4,
        baseline_ridge=1e-2,
        baseline_target_digits=6,
        full_scale_range=10.0,
        adc_bit=8,
        n_measurements=10,
        n_parameters=5,
        model_seed=123,
    )


def test_normalize_enob_level_accepts_nominal_and_numeric() -> None:
    assert normalize_enob_level("nominal") == ("nominal", None)
    assert normalize_enob_level(None) == ("nominal", None)
    assert normalize_enob_level("11") == ("11", 11.0)


def test_factor_sweep_rows_keep_control_variables_fixed() -> None:
    rows = _small_surrogate_rows()

    assert len(rows) == 15
    assert [row.changed_factor for row in rows].count("baseline") == 1
    assert {row.sweep for row in rows} == {
        "baseline",
        "single_factor",
        "grid_ridge_interaction",
    }
    for row in rows:
        assert row.n_elec == 16
        assert row.n_measurements == 10
        assert np.isfinite(row.voltage_rmse)
        assert np.isfinite(row.sigma_relative_rmse)
        assert np.isfinite(row.sigma_effective_digits)

    ridge_rows = [row for row in rows if row.changed_factor == "ridge"]
    assert {row.fem_grid for row in ridge_rows} == {4}
    assert {row.target_voltage_digits for row in ridge_rows} == {6}
    assert {row.enob for row in ridge_rows} == {"nominal"}
    assert {row.noise_relative for row in ridge_rows} == {0.0}

    target_rows = [row for row in rows if row.changed_factor == "target_voltage_digits"]
    assert {row.fem_grid for row in target_rows} == {4}
    assert {row.ridge for row in target_rows} == {1e-2}


def test_anomaly_rules_build_distinct_sigma_fields() -> None:
    points = np.array(
        [
            [0.2, 0.2],
            [0.5, 0.5],
            [0.75, 0.75],
            [0.95, 0.5],
        ],
        dtype=float,
    )

    default = sigma_true_from_anomaly_rule(4, parameter_points=points, rule="default")
    center = sigma_true_from_anomaly_rule(
        4,
        parameter_points=points,
        rule="center_high",
    )
    dual = sigma_true_from_anomaly_rule(
        4,
        parameter_points=points,
        rule="dual_contrast",
    )

    assert not np.allclose(center, default)
    assert not np.allclose(dual, default)


def test_factor_sweep_t17_extended_factors_are_optional() -> None:
    rows = run_factor_sweep(
        fem_grid_levels=[4, 6],
        ridge_levels=[1e-2, 1e-1],
        target_digits=[5, 6],
        noise_relative_levels=[0.0, 0.001],
        enob_levels=["nominal", "5"],
        full_scale_levels=[5.0, 10.0],
        rm_mode_levels=["tikhonov", "noser"],
        anomaly_rule_levels=["default", "center_high"],
        forward_backend="surrogate",
        inverse_backend="pyeidors-rm",
        n_elec=16,
        baseline_fem_grid=4,
        baseline_ridge=1e-2,
        baseline_target_digits=6,
        baseline_enob="5",
        full_scale_range=10.0,
        adc_bit=8,
        n_measurements=10,
        n_parameters=5,
        model_seed=123,
    )

    assert len(rows) == 21
    assert {row.changed_factor for row in rows} == {
        "baseline",
        "fem_grid",
        "ridge",
        "target_voltage_digits",
        "noise_relative",
        "enob",
        "full_scale",
        "rm_mode",
        "anomaly_rule",
        "grid_x_ridge",
    }
    assert [row.changed_factor for row in rows].count("full_scale") == 2
    assert [row.changed_factor for row in rows].count("rm_mode") == 2
    assert [row.changed_factor for row in rows].count("anomaly_rule") == 2
    assert all(np.isfinite(row.sigma_relative_rmse) for row in rows)


def test_v20_anomaly_rule_sweep_reuses_grid_model(monkeypatch) -> None:
    from pyeidors.data import factor_sweep as factor_sweep_module

    calls: list[int] = []
    original = factor_sweep_module._build_model_for_grid

    def counted_build_model_for_grid(**kwargs):
        calls.append(int(kwargs["fem_grid"]))
        return original(**kwargs)

    monkeypatch.setattr(
        factor_sweep_module,
        "_build_model_for_grid",
        counted_build_model_for_grid,
    )

    run_factor_sweep(
        fem_grid_levels=[4, 6],
        ridge_levels=[1e-2],
        target_digits=[6],
        noise_relative_levels=[0.0],
        enob_levels=["nominal"],
        anomaly_rule_levels=["default", "center_high", "dual_contrast"],
        forward_backend="surrogate",
        inverse_backend="least-squares",
        n_elec=16,
        baseline_fem_grid=4,
        baseline_ridge=1e-2,
        baseline_target_digits=6,
        n_measurements=10,
        n_parameters=5,
        model_seed=123,
    )

    assert calls.count(4) == 1
    assert calls.count(6) == 1


def test_factor_sweep_report_and_plot_outputs(tmp_path) -> None:
    rows = _small_surrogate_rows()
    report = format_factor_sweep_report(
        rows,
        full_scale_range=10.0,
        adc_bit=8,
        rm_mode="tikhonov",
        baseline_anomaly_rule="default",
    )
    plot_path = plot_factor_sweep(rows, tmp_path / "factor.png", dpi=80)

    assert "主效应排序" in report
    assert "grid × ridge" in report
    assert "delta_sigma_relative_rmse" in report
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert plot_path.stat().st_size > 1000


def test_eit_factor_sweep_cli_writes_expected_outputs(tmp_path) -> None:
    csv_output = tmp_path / "factor.csv"
    report_output = tmp_path / "factor.md"
    plot_output = tmp_path / "factor.png"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/eit_factor_sweep.py",
            "--forward-backend",
            "surrogate",
            "--inverse-backend",
            "least-squares",
            "--fem-grid-levels",
            "4",
            "6",
            "--ridge-levels",
            "0.01",
            "0.1",
            "--target-digits",
            "5",
            "6",
            "--noise-relative-levels",
            "0",
            "0.001",
            "--enob-levels",
            "nominal",
            "5",
            "--adc-bit",
            "8",
            "--n-measurements",
            "10",
            "--n-parameters",
            "5",
            "--model-seed",
            "123",
            "--output",
            str(csv_output),
            "--report-output",
            str(report_output),
            "--plot-output",
            str(plot_output),
            "--dpi",
            "80",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "model=surrogate+least-squares" in completed.stdout
    with csv_output.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))

    assert len(csv_rows) == 15
    assert list(csv_rows[0].keys()) == CSV_FIELDS
    assert report_output.read_text(encoding="utf-8").startswith(
        "# T15 多因素控制变量实验报告"
    )
    assert plot_output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
