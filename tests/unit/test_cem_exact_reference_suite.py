from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sympy import Matrix, zeros

from pyeidors.interop.geometry_exchange import build_mesh_from_exchange_mat

from scripts.benchmarks.cem_exact_reference_suite import (
    BOUNDARY_COUNT,
    CASES,
    FORMULATIONS,
    N_ELECTRODES,
    aggregate_metrics,
    aggregate_timing_metrics,
    assemble_exact_cem,
    exact_accuracy_metrics,
    exact_circular_mesh,
    exact_current_patterns,
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
    assert [case.case_id for case in CASES] == [f"G{index}" for index in range(1, 8)]
    assert {case.ring_count for case in CASES} == {0, 1, 2}
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
    for skip in (1, 4):
        currents = exact_current_patterns(skip)
        assert currents.dtype == np.float64
        assert np.array_equal(np.sum(currents, axis=0), np.zeros(N_ELECTRODES))
        assert set(np.unique(currents)) == {-1.0, 0.0, 1.0}


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
                    "warm_population_seconds": 2.0,
                    "warm_median_seconds": 1.0,
                },
                {
                    "case_id": case.case_id,
                    "solver": "solver-a",
                    "formulation": "robin_transconductance",
                    "cold_median_seconds": 2.0,
                    "warm_population_seconds": 1.0,
                    "warm_median_seconds": 0.5,
                },
            )
        )

    result = aggregate_timing_metrics(records)
    assert len(result["per_case_robin_over_classic_ratios"]) == len(CASES)
    for phase in ("cold", "warm_population", "warm_reuse"):
        summary = result["solver_phase_summary"]["solver-a"][phase]
        assert summary["geometric_mean_robin_over_classic_ratio"] == 0.5
        assert summary["robin_faster_case_count"] == len(CASES)


def test_v708_missing_optional_timing_is_strict_json_null() -> None:
    phase = {
        "cold_seconds": {"median": 1.0, "iqr": 0.1},
        "warm_population_seconds": 0.5,
        "warm_seconds": {"median": 0.2, "iqr": 0.02},
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
    json.dumps(records, allow_nan=False)
