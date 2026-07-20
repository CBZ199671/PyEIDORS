from __future__ import annotations

import json
import inspect

import numpy as np
import pytest

from scripts.benchmarks.cem_continuum_reference import (
    ContinuumGeometry,
    certify_continuum_reference,
    continuum_current_patterns,
    disk_ntd_apply,
    solve_continuum_level,
)
from scripts.benchmarks.cem_continuum_reference_suite import (
    CASES,
    MESH_LEVELS,
    generalized_richardson_triplet,
    generate_true_circle_mesh,
    shared_reference_sensitivity,
    uncertainty_aware_ranking,
)
from scripts.benchmarks import cem_continuum_reference_suite as continuum_suite


def test_v711_disk_ntd_matches_one_fourier_mode() -> None:
    count = 640
    radius = 1.7
    conductivity = 0.4
    mode = 5
    theta = (np.arange(count, dtype=np.float64) + 0.5) * 2.0 * np.pi / count
    flux = np.cos(mode * theta)

    potential = disk_ntd_apply(flux, conductivity=conductivity, radius=radius)

    expected = radius / (conductivity * mode) * flux
    np.testing.assert_allclose(potential, expected, rtol=2e-13, atol=2e-13)


def test_v711_current_patterns_are_integer_zero_sum() -> None:
    currents = continuum_current_patterns(n_electrodes=16, drive_skip=4)

    assert currents.shape == (16, 16)
    assert np.array_equal(currents, currents.astype(np.int64))
    assert np.array_equal(np.sum(currents, axis=0), np.zeros(16))


def test_v712_continuum_level_satisfies_cem_constraints() -> None:
    geometry = ContinuumGeometry()
    result = solve_continuum_level(
        conductivity=0.25,
        contact_impedance=1.0,
        drive_skip=1,
        n_theta=640,
        geometry=geometry,
    )

    assert result.voltages.shape == (16, 16)
    assert result.linear_relative_residual < 1e-10
    assert result.current_relative_residual < 1e-10
    assert result.robin_relative_residual < 1e-10
    assert result.gauge_relative_residual < 1e-12
    assert result.gmres_info == 0


def test_v712_reference_certification_is_strict_json_and_convergent() -> None:
    certificate = certify_continuum_reference(
        conductivity=0.25,
        contact_impedance=1.0,
        drive_skip=1,
        n_theta_levels=(640, 1280, 2560, 5120),
        max_extrapolation_disagreement=5e-3,
    )

    assert certificate["certified"] is True
    assert certificate["observed_order_last"] > 0.0
    assert certificate["relative_extrapolation_disagreement"] < 5e-3
    assert np.asarray(certificate["reference_voltages"]).shape == (16, 16)
    json.dumps(certificate, allow_nan=False)


def test_v710_true_circle_mesh_refines_boundary_and_interior(tmp_path) -> None:
    coarse = generate_true_circle_mesh(
        tmp_path / "coarse",
        target_h=0.25,
        level_id="H0",
    )
    fine = generate_true_circle_mesh(
        tmp_path / "fine",
        target_h=0.125,
        level_id="H1",
    )

    assert coarse["circle_radius_max_abs_error"] < 2e-12
    assert fine["circle_radius_max_abs_error"] < 2e-12
    assert fine["h_max"] < coarse["h_max"]
    assert fine["boundary_chord_max"] < coarse["boundary_chord_max"]
    assert fine["boundary_sagitta_max"] < coarse["boundary_sagitta_max"]
    assert fine["mesh_fingerprint"] != coarse["mesh_fingerprint"]
    assert set(np.unique(fine["tagged_edges"][:, 2])) == set(range(17))


def test_v713_declares_all_physical_cases_and_mesh_levels() -> None:
    assert [case.case_id for case in CASES] == ["C1", "C2", "C3", "C4", "C5"]
    assert {case.drive_skip for case in CASES} == {1, 4}
    assert {case.contact_impedance for case in CASES} == {0.125, 1.0, 8.0}
    assert [level.target_h for level in MESH_LEVELS] == [0.25, 0.125, 0.0625, 0.03125]


def test_v714_uncertainty_aware_ranking_does_not_rank_overlapping_errors() -> None:
    tied = uncertainty_aware_ranking(
        {
            "PyEIDORS/DOLFINx": 1.00e-3,
            "NGSolve": 1.01e-3,
            "EIDORS": 0.99e-3,
        },
        reference_relative_uncertainty=2e-5,
    )
    separated = uncertainty_aware_ranking(
        {
            "PyEIDORS/DOLFINx": 1.0e-3,
            "NGSolve": 2.0e-3,
            "EIDORS": 3.0e-3,
        },
        reference_relative_uncertainty=1e-5,
    )

    assert tied["strict_order_supported"] is False
    assert set(tied["best_tie"]) == {"PyEIDORS/DOLFINx", "NGSolve", "EIDORS"}
    assert separated["strict_order_supported"] is True
    assert separated["ordering"] == ["PyEIDORS/DOLFINx", "NGSolve", "EIDORS"]


def test_v722_shared_reference_sensitivity_detects_order_reversal() -> None:
    candidates = {
        "PyEIDORS/DOLFINx": np.asarray([[0.99, 1.0]], dtype=np.float64),
        "NGSolve": np.asarray([[1.01, 1.0]], dtype=np.float64),
        "EIDORS": np.asarray([[1.02, 1.0]], dtype=np.float64),
    }
    references = {
        "previous_extrapolated": np.asarray([[0.98, 1.0]], dtype=np.float64),
        "final_extrapolated": np.asarray([[1.0, 1.0]], dtype=np.float64),
        "finest_raw": np.asarray([[1.015, 1.0]], dtype=np.float64),
    }

    result = shared_reference_sensitivity(candidates, references)

    assert result["primary_reference"] == "final_extrapolated"
    assert result["ordering_stable_across_references"] is False
    assert result["best_solver_stable_across_references"] is False
    assert set(result["reference_rankings"]) == set(references)
    assert len(result["pairwise_solver_comparisons"]) == 3
    json.dumps(result, allow_nan=False)
    for comparison in result["pairwise_solver_comparisons"]:
        assert comparison["symmetric_relative_voltage_separation"] > 0.0
        assert comparison["squared_error_identity_closure_abs"] < 1.0e-15


def test_v716_generalized_richardson_uses_measured_nonuniform_h() -> None:
    truth = np.asarray([[2.0, -1.0], [0.5, -1.5]], dtype=np.float64)
    coefficient = np.asarray([[0.3, -0.2], [0.1, 0.4]], dtype=np.float64)
    h_values = (0.16, 0.11, 0.067)
    approximations = tuple(truth + coefficient * h**2 for h in h_values)

    result = generalized_richardson_triplet(
        approximations[0],
        approximations[1],
        approximations[2],
        h_coarse=h_values[0],
        h_middle=h_values[1],
        h_fine=h_values[2],
    )

    assert result is not None
    assert result["observed_order"] == pytest.approx(2.0, rel=1e-10)
    np.testing.assert_allclose(result["extrapolated"], truth, rtol=1e-12, atol=1e-12)


def test_v715_report_contract_explains_reference_and_uncertainty() -> None:
    source = inspect.getsource(continuum_suite._write_markdown_report)
    compare_source = inspect.getsource(continuum_suite.compare_suite)

    assert "Neumann-to-Dirichlet" in source
    assert "Richardson" in source
    assert "严格顺序成立" in source
    assert "共享参考敏感性" in source
    assert "离散误差" in source
    assert "代数误差" in source
    assert "finest_shared_reference_sensitivity" in compare_source
    assert "accuracy_evidence_hierarchy" in compare_source
    assert "continuum_relative_l2" in continuum_suite.METRIC_FIELDS
    assert continuum_suite.METRIC_SCHEMA == "cem-continuum-accuracy-metrics-v1"
