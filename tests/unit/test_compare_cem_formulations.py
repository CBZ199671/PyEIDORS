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
from scripts.benchmarks.cem_fair_common import (
    TIMING_SCHEMA,
    TIMING_SCOPE,
    benchmark_preassembled_blocks,
    canonical_mesh_fingerprint,
    timing_summary,
    validate_solver_reports,
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
    assert len(metrics["cross_solver_implementation"]) == 2
    assert all(
        item["raw_curve_relative_l2"] > 0.0
        for item in metrics["cross_solver_implementation"]
    )


def test_mesh_fingerprint_is_orientation_invariant_but_tag_sensitive() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cells = np.asarray([[0, 1, 2], [1, 3, 2]])
    edges = np.asarray([[0, 1, 1], [1, 3, 0], [3, 2, 2], [2, 0, 0]])

    reference = canonical_mesh_fingerprint(nodes, cells, edges)
    permuted = canonical_mesh_fingerprint(
        nodes,
        cells[::-1, ::-1],
        edges[::-1][:, [1, 0, 2]],
    )
    retagged = edges.copy()
    retagged[0, 2] = 2

    assert permuted == reference
    assert canonical_mesh_fingerprint(nodes, cells, retagged) != reference
    moved_nodes = nodes.copy()
    moved_nodes[0, 0] = 1e-5
    assert canonical_mesh_fingerprint(moved_nodes, cells, edges) != reference


def test_v699_mesh_fingerprint_is_vertex_numbering_invariant() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cells = np.asarray([[0, 1, 2], [1, 3, 2]])
    edges = np.asarray([[0, 1, 1], [1, 3, 0], [3, 2, 2], [2, 0, 0]])
    vertex_order = np.asarray([2, 0, 3, 1], dtype=np.int64)
    source_to_permuted = np.empty(vertex_order.size, dtype=np.int64)
    source_to_permuted[vertex_order] = np.arange(vertex_order.size, dtype=np.int64)

    reference = canonical_mesh_fingerprint(nodes, cells, edges)
    permuted = canonical_mesh_fingerprint(
        nodes[vertex_order],
        source_to_permuted[cells],
        np.column_stack((source_to_permuted[edges[:, :2]], edges[:, 2])),
    )

    assert permuted == reference


def test_v699_mesh_fingerprint_rejects_ambiguous_or_invalid_nodes() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    cells = np.asarray([[0, 1, 2]])
    edges = np.asarray([[0, 1, 1], [1, 2, 0], [2, 0, 0]])

    duplicate_nodes = nodes.copy()
    duplicate_nodes[2] = duplicate_nodes[1]
    with pytest.raises(ValueError, match="remain unique"):
        canonical_mesh_fingerprint(duplicate_nodes, cells, edges)

    invalid_cells = np.asarray([[0, 1, 3]])
    with pytest.raises(ValueError, match="out-of-range"):
        canonical_mesh_fingerprint(nodes, invalid_cells, edges)


def _fair_report(solver: str, fingerprint: str = "a" * 64) -> dict[str, object]:
    return {
        "solver": solver,
        "discretization": {
            "potential_order": 1,
            "mesh_fingerprint": fingerprint,
            "mesh_import_verified": True,
        },
        "linear_solver": {"scalar_dtype": "float64"},
        "timing": {
            "schema": TIMING_SCHEMA,
            "scope": TIMING_SCOPE,
            "operations_per_sample": 16,
            "paired_cold_decomposition": True,
            "cross_formulation_cache_reuse": False,
            "classic": {
                "cold_seconds": {"median": 2.0},
                "warm_reuse_seconds": {"median": 1.0},
            },
            "robin_transconductance": {
                "cold_seconds": {"median": 2.0},
                "warm_reuse_seconds": {"median": 1.0},
            },
        },
    }


def test_report_validation_rejects_precision_mesh_and_cache_confounds() -> None:
    first = _fair_report("first")
    second = _fair_report("second")
    assert validate_solver_reports([first, second]) == "a" * 64

    second["linear_solver"]["scalar_dtype"] = "complex64"  # type: ignore[index]
    with pytest.raises(ValueError, match="float64"):
        validate_solver_reports([first, second])
    second = _fair_report("second", fingerprint="b" * 64)
    with pytest.raises(ValueError, match="fingerprints differ"):
        validate_solver_reports([first, second])
    second = _fair_report("second")
    second["timing"]["cross_formulation_cache_reuse"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="reused cache"):
        validate_solver_reports([first, second])


def test_timing_summary_reports_median_and_linear_iqr() -> None:
    summary = timing_summary([1.0, 2.0, 3.0, 4.0, 5.0])

    assert summary["median"] == 3.0
    assert summary["iqr"] == 2.0
    assert summary["samples"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_v709_cold_setup_and_warm_reuse_are_paired_and_unambiguous() -> None:
    robin_matrix = np.asarray([[3.0, 0.0], [0.0, 2.0]])
    coupling = np.asarray([[-0.5, 0.0], [0.0, -0.5]])
    electrode_matrix = np.eye(2)
    currents = np.asarray([[1.0, -1.0], [-1.0, 1.0]])

    timing, _, _ = benchmark_preassembled_blocks(
        robin_matrix,
        coupling,
        electrode_matrix,
        currents,
        repeats=3,
        operations_per_sample=4,
    )

    assert timing["schema"] == TIMING_SCHEMA
    assert timing["paired_cold_decomposition"] is True
    assert timing["operations_per_sample"] == 4
    for formulation in ("classic", "robin_transconductance"):
        phase = timing[formulation]
        cold = phase["cold_seconds"]
        setup = phase["setup_seconds"]
        cold_solve = phase["cold_solve_seconds"]
        warm = phase["warm_reuse_seconds"]
        assert len(cold["samples"]) == 3
        assert len(setup["samples"]) == 3
        assert len(cold_solve["samples"]) == 3
        assert len(warm["samples"]) == 3
        assert all(
            total >= component
            for total, component in zip(cold["samples"], setup["samples"])
        )
        assert cold["median"] > warm["median"]
        assert phase["cold_over_warm_reuse_speedup"] == pytest.approx(
            cold["median"] / warm["median"]
        )
