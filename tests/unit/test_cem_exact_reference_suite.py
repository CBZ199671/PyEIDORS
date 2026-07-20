from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sympy import Matrix, zeros

from pyeidors.interop.geometry_exchange import build_mesh_from_exchange_mat

from scripts.benchmarks.cem_exact_reference_suite import (
    BASELINE_SETTING_ID,
    BOUNDARY_COUNT,
    CASES,
    FORMULATIONS,
    N_ELECTRODES,
    REFINEMENT_CASE_IDS,
    SETTINGS,
    _plot_factorial_suite,
    aggregate_metrics,
    aggregate_timing_metrics,
    assemble_exact_cem,
    exact_case_mesh,
    exact_accuracy_metrics,
    exact_circular_mesh,
    exact_current_patterns,
    exact_basis_cache_clear,
    exact_basis_cache_info,
    exact_refined_circular_mesh,
    float_nodes,
    prepare_case_fixture,
    solve_exact_case,
    timing_records_from_report,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v701_exact_circular_mesh_preserves_electrode_topology() -> None:
    for ring_count, expected_nodes, expected_cells in (
        (0, 33, 32),
        (1, 65, 96),
        (2, 97, 160),
    ):
        nodes, cells, edges, electrode_nodes, electrode_counts = exact_circular_mesh(
            ring_count
        )
        outer_offset = 1 + ring_count * BOUNDARY_COUNT
        expected_electrode_nodes = np.asarray(
            [
                (outer_offset + 2 * electrode, outer_offset + 2 * electrode + 1)
                for electrode in range(N_ELECTRODES)
            ],
            dtype=np.int64,
        )

        assert len(nodes) == expected_nodes
        assert cells.shape == (expected_cells, 3)
        assert edges.shape == (BOUNDARY_COUNT, 3)
        assert np.array_equal(electrode_nodes, expected_electrode_nodes)
        assert np.array_equal(
            electrode_counts,
            np.full(N_ELECTRODES, 2, dtype=np.int64),
        )
        assert int(np.max(electrode_nodes)) < len(nodes)
        assert np.array_equal(edges[::2, :2], expected_electrode_nodes)

        radii_squared = {
            x * x + y * y
            for x, y in nodes[outer_offset : outer_offset + BOUNDARY_COUNT]
        }
        assert len(radii_squared) == 1
        assert all(
            coordinate.denominator & (coordinate.denominator - 1) == 0
            for point in nodes
            for coordinate in point
        )
        assert all(
            isinstance(coordinate, Fraction) for point in nodes for coordinate in point
        )
        assert float_nodes(nodes).dtype == np.float64


def test_v702_exact_cem_blocks_and_solves_are_rational_identities() -> None:
    case = CASES[0]
    nodes, cells, edges, _, _ = exact_circular_mesh(case.ring_count)
    a_r, coupling, electrode_matrix = assemble_exact_cem(
        nodes,
        cells,
        edges,
        conductivity=case.conductivity,
        contact_impedance=case.contact_impedance,
    )
    node_ones = Matrix.ones(len(nodes), 1)
    electrode_ones = Matrix.ones(N_ELECTRODES, 1)
    assert a_r * node_ones + coupling * electrode_ones == zeros(len(nodes), 1)
    assert coupling.T * node_ones + electrode_matrix * electrode_ones == zeros(
        N_ELECTRODES,
        1,
    )

    reference = solve_exact_case(case)
    assert reference["exact_classic_residual_zero"] is True
    assert reference["exact_robin_residual_zero"] is True
    assert reference["exact_classic_robin_identical"] is True
    assert reference["exact_linear_solver"] == "DomainMatrix.lu_solve"
    assert reference["exact_domain"] == "QQ"
    assert all(
        sum(reference["voltage"][row, column] for row in range(N_ELECTRODES)) == 0
        for column in range(N_ELECTRODES)
    )


def test_v707_saved_mat_uses_one_based_electrode_connectivity(tmp_path: Path) -> None:
    fixture = prepare_case_fixture(tmp_path, CASES[0])
    payload = loadmat(fixture["mat_path"], squeeze_me=True, struct_as_record=False)
    boundary_edges = np.asarray(payload["boundary_edges"], dtype=np.int64).reshape(
        -1,
        2,
    )
    electrode_nodes = np.asarray(payload["electrode_nodes"], dtype=np.int64).reshape(
        N_ELECTRODES,
        2,
    )

    assert int(np.min(boundary_edges)) >= 1
    assert int(np.min(electrode_nodes)) >= 1
    assert np.array_equal(electrode_nodes, boundary_edges[::2])

    mesh, _ = build_mesh_from_exchange_mat(fixture["mat_path"])
    assert mesh.facet_tags is not None
    for electrode_tag in range(2, N_ELECTRODES + 2):
        assert mesh.facet_tags.find(electrode_tag).size == 1


def test_v703_exact_suite_case_matrix_and_integer_currents_are_complete() -> None:
    assert [case.case_id for case in CASES] == [
        f"G{index:02d}" for index in range(1, 49)
    ]
    assert len(SETTINGS) == 12
    assert REFINEMENT_CASE_IDS == ("G01", "G13", "G25", "G37")
    refinement_cases = [case for case in CASES if case.case_id in REFINEMENT_CASE_IDS]
    assert [case.refinement_level_id for case in refinement_cases] == [
        "Q0",
        "Q1",
        "Q2",
        "Q3",
    ]
    assert {
        (case.edge_subdivisions, case.radial_layers) for case in refinement_cases
    } == {(1, 1), (1, 2), (2, 2), (2, 4)}
    assert {case.contact_impedance for case in CASES} == {
        Fraction(1, 8),
        Fraction(1),
        Fraction(8),
    }
    assert {case.conductivity for case in CASES} == {
        Fraction(1, 4),
        Fraction(1),
    }
    assert {case.drive_skip for case in CASES} == {1, 4}
    expected_factorial = {
        (level, sigma, impedance, drive)
        for level in ("Q0", "Q1", "Q2", "Q3")
        for sigma in (Fraction(1, 4), Fraction(1))
        for impedance in (Fraction(1, 8), Fraction(1), Fraction(8))
        for drive in (1, 4)
    }
    observed_factorial = {
        (
            case.refinement_level_id,
            case.conductivity,
            case.contact_impedance,
            case.drive_skip,
        )
        for case in CASES
    }
    assert observed_factorial == expected_factorial
    assert len(observed_factorial) == len(CASES)
    assert all(case.setting_id == BASELINE_SETTING_ID for case in refinement_cases)
    for skip in (1, 4):
        currents = exact_current_patterns(skip)
        assert currents.dtype == np.float64
        assert np.array_equal(np.sum(currents, axis=0), np.zeros(N_ELECTRODES))
        assert set(np.unique(currents)) == {-1.0, 0.0, 1.0}


def test_v724_exact_basis_cache_reuses_only_drive_independent_system() -> None:
    adjacent = next(
        case
        for case in CASES
        if case.refinement_level_id == "Q0"
        and case.conductivity == Fraction(1, 4)
        and case.contact_impedance == Fraction(1)
        and case.drive_skip == 1
    )
    skip_four = next(
        case
        for case in CASES
        if case.refinement_level_id == "Q0"
        and case.conductivity == adjacent.conductivity
        and case.contact_impedance == adjacent.contact_impedance
        and case.drive_skip == 4
    )
    exact_basis_cache_clear()
    try:
        adjacent_reference = solve_exact_case(adjacent)
        first = exact_basis_cache_info()
        skip_reference = solve_exact_case(skip_four)
        second = exact_basis_cache_info()

        assert first.misses == 1
        assert first.hits == 0
        assert second.misses == 1
        assert second.hits == 1
        for reference in (adjacent_reference, skip_reference):
            assert reference["exact_classic_residual_zero"] is True
            assert reference["exact_robin_residual_zero"] is True
            assert reference["exact_classic_robin_identical"] is True
    finally:
        exact_basis_cache_clear()


def test_v717_nested_rational_refinement_preserves_exact_fixed_domain() -> None:
    expected = {
        "Q0": (1, 1, 33, 32),
        "Q1": (1, 2, 65, 96),
        "Q2": (2, 2, 129, 192),
        "Q3": (2, 4, 257, 448),
    }
    node_sets: list[set[tuple[Fraction, Fraction]]] = []
    for level_id, (
        edge_subdivisions,
        radial_layers,
        nodes_count,
        cells_count,
    ) in expected.items():
        nodes, cells, edges, electrode_nodes, electrode_counts = (
            exact_refined_circular_mesh(
                edge_subdivisions=edge_subdivisions,
                radial_layers=radial_layers,
            )
        )
        assert len(nodes) == nodes_count
        assert cells.shape == (cells_count, 3)
        assert edges.shape == (BOUNDARY_COUNT * edge_subdivisions, 3)
        assert electrode_nodes.shape == (
            N_ELECTRODES,
            edge_subdivisions + 1,
        )
        assert np.array_equal(
            electrode_counts,
            np.full(N_ELECTRODES, edge_subdivisions + 1, dtype=np.int64),
        )
        assert np.count_nonzero(edges[:, 2]) == N_ELECTRODES * edge_subdivisions
        assert all(
            coordinate.denominator & (coordinate.denominator - 1) == 0
            for point in nodes
            for coordinate in point
        )
        assert float_nodes(nodes).dtype == np.float64
        node_sets.append(set(nodes))
        case = next(item for item in CASES if item.refinement_level_id == level_id)
        assert exact_case_mesh(case)[0] == nodes

    assert all(
        coarse <= fine
        for coarse, fine in zip(node_sets[:-1], node_sets[1:], strict=True)
    )


def test_v720_refinement_aggregation_keeps_grid_sequence_separate() -> None:
    metrics = []
    solvers = ("PyEIDORS/DOLFINx", "EIDORS", "NGSolve")
    case_lookup = {case.case_id: case for case in CASES}
    for case_id in REFINEMENT_CASE_IDS:
        case = case_lookup[case_id]
        for formulation in FORMULATIONS:
            for rank, solver in enumerate(solvers, start=1):
                metrics.append(
                    {
                        "case_id": case_id,
                        "refinement_level_id": case.refinement_level_id,
                        "nodes": 10 * case.radial_layers * case.edge_subdivisions,
                        "cells": 20 * case.radial_layers * case.edge_subdivisions,
                        "solver": solver,
                        "formulation": formulation,
                        "truth_relative_l2": rank * 1.0e-15,
                    }
                )

    result = aggregate_metrics(metrics)
    for formulation in FORMULATIONS:
        summary = result["refinement_summary"][formulation]
        assert summary["case_ids"] == list(REFINEMENT_CASE_IDS)
        assert summary["level_ids"] == ["Q0", "Q1", "Q2", "Q3"]
        assert summary["win_counts"]["PyEIDORS/DOLFINx"] == 4


def test_v725_factorial_aggregation_is_balanced_by_mesh_and_setting() -> None:
    metrics = []
    solvers = ("PyEIDORS/DOLFINx", "EIDORS", "NGSolve")
    for case in CASES:
        for formulation in FORMULATIONS:
            for rank, solver in enumerate(solvers, start=1):
                metrics.append(
                    {
                        "case_id": case.case_id,
                        "solver": solver,
                        "formulation": formulation,
                        "truth_relative_l2": rank * 1.0e-15,
                    }
                )

    result = aggregate_metrics(metrics)["factorial_summary"]

    assert result["full_factorial_complete"] is True
    assert result["case_count"] == 48
    assert result["mesh_level_ids"] == ["Q0", "Q1", "Q2", "Q3"]
    assert result["setting_ids"] == [setting.setting_id for setting in SETTINGS]
    for formulation in FORMULATIONS:
        for level in result["mesh_level_ids"]:
            group = result["by_mesh"][formulation][level]
            assert group["case_count"] == 12
            assert group["win_counts"]["PyEIDORS/DOLFINx"] == 12
            assert group["per_solver"]["NGSolve"]["record_count"] == 12
        for setting_id in result["setting_ids"]:
            group = result["by_setting"][formulation][setting_id]
            assert group["case_count"] == 4
            assert group["win_counts"]["PyEIDORS/DOLFINx"] == 4
            assert group["per_solver"]["EIDORS"]["record_count"] == 4


def test_v725_factorial_plot_and_report_cover_complete_evidence(
    tmp_path: Path,
) -> None:
    metrics = []
    solvers = ("PyEIDORS/DOLFINx", "EIDORS", "NGSolve")
    for case in CASES:
        for formulation in FORMULATIONS:
            for rank, solver in enumerate(solvers, start=1):
                metrics.append(
                    {
                        "case_id": case.case_id,
                        "solver": solver,
                        "formulation": formulation,
                        "truth_relative_l2": rank * 1.0e-15,
                    }
                )

    plot_path = tmp_path / "factorial.png"
    _plot_factorial_suite(metrics, plot_path)
    report = (ROOT / "docs" / "benchmarks" / "cem_exact_accuracy_report.md").read_text(
        encoding="utf-8"
    )

    assert plot_path.stat().st_size > 0
    for evidence in (
        "48 个 case",
        "288 条精度记录",
        "`misses=24`",
        "`hits=24`",
        "S02/S08",
        "288/288 条计时记录",
        "cem_exact_factorial_heatmap.png",
    ):
        assert evidence in report


def test_v704_truth_error_and_backward_residual_detect_a_perturbation() -> None:
    reference = solve_exact_case(CASES[0])
    rounded_truth = np.asarray(reference["voltage"], dtype=np.float64)
    baseline = exact_accuracy_metrics(rounded_truth, reference)
    perturbed = rounded_truth.copy()
    perturbed[0, 0] += 1.0e-8
    changed = exact_accuracy_metrics(perturbed, reference)

    assert baseline["truth_relative_l2"] < 1.0e-15
    assert baseline["exact_reduced_scaled_backward_residual"] < 1.0e-15
    assert changed["truth_relative_l2"] > baseline["truth_relative_l2"]
    assert (
        changed["exact_reduced_scaled_backward_residual"]
        > baseline["exact_reduced_scaled_backward_residual"]
    )


def test_v705_aggregation_exposes_case_order_reversals() -> None:
    solvers = ("EIDORS", "NGSolve", "PyEIDORS/DOLFINx")
    metrics = []
    for case_index, case in enumerate(CASES):
        for formulation in FORMULATIONS:
            ordering = solvers if case_index != 1 else tuple(reversed(solvers))
            for rank, solver in enumerate(ordering, start=1):
                metrics.append(
                    {
                        "case_id": case.case_id,
                        "solver": solver,
                        "formulation": formulation,
                        "truth_relative_l2": float(rank) * 1.0e-15,
                    }
                )

    aggregate = aggregate_metrics(metrics)
    for formulation in FORMULATIONS:
        assert aggregate["universal_ordering"][formulation]["supported"] is False
        assert aggregate["universal_ordering"][formulation]["ordering"] is None


def test_v703_external_runners_export_exact_suite_identity_fields() -> None:
    ngsolve_source = (
        ROOT / "scripts" / "benchmarks" / "ngsolve_cem_exact_case.py"
    ).read_text(encoding="utf-8")
    matlab_source = (
        ROOT / "compare_with_Eidors" / "compare_cem_formulations.m"
    ).read_text(encoding="utf-8")
    for source in (ngsolve_source, matlab_source):
        assert "suite_schema" in source
        assert "case_id" in source
        assert "drive_skip" in source

    ngsolve_suite_source = (
        ROOT / "scripts" / "benchmarks" / "ngsolve_cem_exact_suite.py"
    ).read_text(encoding="utf-8")
    matlab_suite_source = (
        ROOT / "compare_with_Eidors" / "run_cem_exact_suite.m"
    ).read_text(encoding="utf-8")
    assert 'cases = manifest["cases"]' in ngsolve_suite_source
    assert "for fixture in cases" in ngsolve_suite_source
    assert "run_case(" in ngsolve_suite_source
    assert "fixtures = manifest.cases" in matlab_suite_source
    assert "for index = 1:numel(fixtures)" in matlab_suite_source
    assert "compare_cem_formulations.m" in matlab_suite_source


def test_v688_timing_aggregation_keeps_phases_and_ratio_direction_explicit() -> None:
    records = []
    for case in CASES:
        records.extend(
            (
                {
                    "case_id": case.case_id,
                    "solver": "solver-a",
                    "formulation": "classic",
                    "cold_median_seconds": 4.0,
                    "setup_median_seconds": 2.0,
                    "warm_reuse_median_seconds": 1.0,
                    "cold_over_warm_reuse_speedup": 4.0,
                },
                {
                    "case_id": case.case_id,
                    "solver": "solver-a",
                    "formulation": "robin_transconductance",
                    "cold_median_seconds": 2.0,
                    "setup_median_seconds": 1.0,
                    "warm_reuse_median_seconds": 0.5,
                    "cold_over_warm_reuse_speedup": 4.0,
                },
            )
        )

    result = aggregate_timing_metrics(records)
    assert len(result["per_case_robin_over_classic_ratios"]) == len(CASES)
    for phase in ("cold", "setup", "warm_reuse"):
        summary = result["solver_phase_summary"]["solver-a"][phase]
        assert summary["geometric_mean_robin_over_classic_ratio"] == 0.5
        assert summary["robin_faster_case_count"] == len(CASES)
    absolute = result["solver_formulation_absolute_summary"]["solver-a"]
    assert absolute["classic"]["geometric_mean_cold_over_warm_reuse_speedup"] == 4.0
    assert (
        absolute["robin_transconductance"][
            "geometric_mean_cold_over_warm_reuse_speedup"
        ]
        == 4.0
    )


def test_v708_missing_optional_timing_is_strict_json_null() -> None:
    phase = {
        "cold_seconds": {"median": 1.0, "iqr": 0.1},
        "setup_seconds": {"median": 0.5, "iqr": 0.05},
        "cold_solve_seconds": {"median": 0.4, "iqr": 0.04},
        "warm_reuse_seconds": {"median": 0.2, "iqr": 0.02},
        "cold_over_warm_reuse_speedup": 5.0,
    }
    report = {
        "solver": "test-solver",
        "timing": {
            "classic": phase,
            "robin_transconductance": phase,
        },
    }

    records = timing_records_from_report(report, case_id="G1")

    assert records[0]["assembly_seconds"] is None
    assert records[0]["mesh_import_seconds"] is None
    assert records[0]["setup_median_seconds"] == 0.5
    assert records[0]["warm_reuse_median_seconds"] == 0.2
    json.dumps(records, allow_nan=False)
