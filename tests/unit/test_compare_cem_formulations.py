"""Unit tests for the reproducible CEM comparison helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmarks.compare_cem_formulations import (
    characteristic_rows,
    comparison_metrics,
    relative_l2,
    trigonometric_current_patterns,
)


def test_trigonometric_patterns_are_balanced_and_labeled() -> None:
    patterns, labels = trigonometric_current_patterns(16, 0.7)

    assert patterns.shape == (16, 16)
    assert labels[0] == ("cosine", 1)
    assert labels[-1] == ("sine", 8)
    np.testing.assert_allclose(np.sum(patterns, axis=0), 0.0, atol=1e-14)
    assert np.all(np.linalg.norm(patterns, axis=0) > 1.0)


def test_characteristic_rows_preserve_raw_si_norm_ratio() -> None:
    currents = np.asarray([[1.0, 0.0], [-1.0, 2.0]])
    voltages = 3.0 * currents
    labels = [("cosine", 1), ("sine", 1)]

    rows = characteristic_rows("fixture", "classic", currents, voltages, labels)

    assert [row["characteristic_resistance_ohm"] for row in rows] == pytest.approx(
        [3.0, 3.0]
    )
    assert relative_l2(voltages, voltages) == 0.0
    with pytest.raises(ValueError, match="matching shapes"):
        characteristic_rows("fixture", "classic", currents, voltages[:, :1], labels)


def test_comparison_metrics_separate_formula_and_solver_differences() -> None:
    rows = []
    for solver, scale in (("solver-a", 1.0), ("solver-b", 1.1)):
        for formulation, formula_scale in (
            ("classic", 1.0),
            ("robin_transconductance", 1.0),
        ):
            for frequency in (1, 2):
                rows.append(
                    {
                        "solver": solver,
                        "formulation": formulation,
                        "mode": "cosine",
                        "spatial_frequency": frequency,
                        "current_norm_a": 1.0,
                        "voltage_norm_v": scale * formula_scale / frequency,
                        "characteristic_resistance_ohm": (
                            scale * formula_scale / frequency
                        ),
                    }
                )

    metrics = comparison_metrics(rows)

    assert len(metrics["within_solver_formulation"]) == 2
    assert all(
        item["curve_relative_l2_robin_vs_classic"] == 0.0
        for item in metrics["within_solver_formulation"]
    )
    assert len(metrics["cross_solver_discretization"]) == 2
    assert all(
        item["raw_curve_relative_l2"] > 0.0
        for item in metrics["cross_solver_discretization"]
    )
