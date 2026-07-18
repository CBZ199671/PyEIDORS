"""Tests for the independent multiprecision CEM accuracy reference."""

from __future__ import annotations

from mpmath import mp
import numpy as np
import pytest

from scripts.benchmarks.cem_fair_common import canonical_mesh_fingerprint
from scripts.benchmarks.cem_high_precision_reference import (
    FIXTURE_SCHEMA,
    _solver_accuracy_metrics,
    _validate_report,
    assemble_multiprecision_cem,
    canonical_fan_mesh,
    prepare_common_fixture,
    solve_reference_at_dps,
)
from scripts.benchmarks.compare_cem_formulations import (
    trigonometric_current_patterns,
)


def test_v696_canonical_fan_mesh_contract_and_determinism(tmp_path) -> None:
    nodes, cells, edges, electrode_nodes, electrode_counts = canonical_fan_mesh()

    assert nodes.shape == (33, 2)
    assert cells.shape == (32, 3)
    assert edges.shape == (32, 3)
    assert electrode_nodes.shape == (16, 2)
    np.testing.assert_array_equal(electrode_counts, 2)
    assert np.count_nonzero(edges[:, 2] > 0) == 16
    assert set(edges[:, 2]) == set(range(17))
    signed_double_areas = []
    for triangle in cells:
        a, b, c = nodes[triangle]
        ab = b - a
        ac = c - a
        signed_double_areas.append(ab[0] * ac[1] - ab[1] * ac[0])
    assert np.all(np.asarray(signed_double_areas) > 0.0)

    repeated = canonical_fan_mesh()
    assert canonical_mesh_fingerprint(nodes, cells, edges) == (
        canonical_mesh_fingerprint(repeated[0], repeated[1], repeated[2])
    )
    fixture = prepare_common_fixture(tmp_path)
    assert fixture["fixture_schema"] == FIXTURE_SCHEMA
    assert fixture["nodes"] == 33
    assert fixture["cells"] == 32
    assert fixture["boundary_edges"] == 32
    assert fixture["mat_path"].exists()
    assert fixture["msh_path"].exists()


def test_v693_analytic_cem_blocks_preserve_constant_voltage_null_mode() -> None:
    nodes, cells, edges, _, _ = canonical_fan_mesh(n_electrodes=4)

    with mp.workdps(80):
        a_r, coupling, electrode_matrix = assemble_multiprecision_cem(
            nodes,
            cells,
            edges,
            n_electrodes=4,
            conductivity=0.25,
            contact_impedance=1.0,
        )
        node_ones = mp.ones(nodes.shape[0], 1)
        electrode_ones = mp.ones(4, 1)
        body_null = a_r * node_ones + coupling * electrode_ones
        electrode_null = coupling.T * node_ones + electrode_matrix * electrode_ones

        assert max(abs(value) for value in body_null) < mp.mpf("1e-75")
        assert max(abs(value) for value in electrode_null) < mp.mpf("1e-75")


def test_v694_multiprecision_reference_converges_across_precisions() -> None:
    nodes, cells, edges, _, _ = canonical_fan_mesh(n_electrodes=4)
    currents, _ = trigonometric_current_patterns(4, 0.7)
    lower = solve_reference_at_dps(
        nodes,
        cells,
        edges,
        currents,
        n_electrodes=4,
        conductivity=0.25,
        contact_impedance=1.0,
        dps=50,
    )
    higher = solve_reference_at_dps(
        nodes,
        cells,
        edges,
        currents,
        n_electrodes=4,
        conductivity=0.25,
        contact_impedance=1.0,
        dps=80,
    )

    with mp.workdps(80):
        delta = higher["voltage"] - lower["voltage"]
        delta_norm = mp.sqrt(mp.fsum(abs(value) ** 2 for value in delta))
        reference_norm = mp.sqrt(
            mp.fsum(abs(value) ** 2 for value in higher["voltage"])
        )
        assert delta_norm / reference_norm < mp.mpf("1e-45")
        assert higher["scaled_full_residual"] < mp.mpf("1e-70")


def test_v695_accuracy_metrics_use_high_precision_truth_and_reduced_residual() -> None:
    nodes, cells, edges, _, _ = canonical_fan_mesh(n_electrodes=4)
    currents, _ = trigonometric_current_patterns(4, 0.7)
    reference = solve_reference_at_dps(
        nodes,
        cells,
        edges,
        currents,
        n_electrodes=4,
        conductivity=0.25,
        contact_impedance=1.0,
        dps=80,
    )
    candidate = np.asarray(
        [
            [float(reference["voltage"][row, column]) for column in range(4)]
            for row in range(4)
        ],
        dtype=np.float64,
    )

    metrics = _solver_accuracy_metrics(candidate, reference)

    assert metrics["electrode_voltage_relative_l2"] < 1e-15
    assert metrics["reduced_scaled_backward_residual"] < 1e-15
    assert metrics["voltage_gauge_relative_residual"] < 1e-15
    assert len(metrics["per_rhs_relative_l2"]) == 4


def test_v695_report_validation_fails_closed_on_mesh_and_raw_shape() -> None:
    currents = np.ones((4, 4), dtype=np.float64)
    fixture = {
        "mesh_fingerprint": "a" * 64,
        "n_electrodes": 4,
        "currents": currents,
    }
    report = {
        "solver": "fixture",
        "discretization": {
            "mesh_fingerprint": "a" * 64,
            "mesh_import_verified": True,
            "potential_order": 1,
        },
        "linear_solver": {"scalar_dtype": "float64"},
        "raw_electrode_voltages": {
            "classic": np.ones((4, 4)).tolist(),
            "robin_transconductance": np.ones((4, 4)).tolist(),
        },
    }
    _validate_report(report, fixture)

    report["discretization"]["mesh_fingerprint"] = "b" * 64
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _validate_report(report, fixture)
    report["discretization"]["mesh_fingerprint"] = "a" * 64
    report["raw_electrode_voltages"]["classic"] = [[1.0]]
    with pytest.raises(ValueError, match="shape/finite mismatch"):
        _validate_report(report, fixture)
