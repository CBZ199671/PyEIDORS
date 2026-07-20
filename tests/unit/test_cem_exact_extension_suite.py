from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pyeidors.interop.geometry_exchange import build_mesh_from_exchange_mat

from scripts.benchmarks.cem_exact_extension_suite import (
    ATTRIBUTION_CASE_IDS,
    EXTENSION_CASES,
    extension_basis_cache_clear,
    extension_basis_cache_info,
    extension_case_cell_conductivities,
    extension_case_system_key,
    extension_current_patterns,
    extension_refined_circular_mesh,
    pyeidors_cell_conductivity_values,
    prepare_extension_case_fixture,
    solve_exact_extension_case,
    _validated_flint_result,
)


def test_v726_preregistered_extension_matrix_is_frozen_and_complete() -> None:
    assert [case.case_id for case in EXTENSION_CASES] == [
        f"X{index:02d}" for index in range(1, 39)
    ]
    assert len({extension_case_system_key(case) for case in EXTENSION_CASES}) == 19
    family_counts = {
        family: sum(case.family == family for case in EXTENSION_CASES)
        for family in {case.family for case in EXTENSION_CASES}
    }
    assert family_counts == {
        "range": 16,
        "heterogeneous": 8,
        "electrode_count": 8,
        "large_q4": 6,
    }
    assert sum(case.refinement_level_id == "Q4" for case in EXTENSION_CASES) == 6
    assert ATTRIBUTION_CASE_IDS == ("X05", "X13", "X33", "X21")


def test_v727_dynamic_electrode_mesh_and_q4_counts_are_exact() -> None:
    nodes, cells, edges, electrode_nodes, electrode_counts = (
        extension_refined_circular_mesh(
            edge_subdivisions=1,
            radial_layers=1,
            n_electrodes=8,
        )
    )
    assert len(nodes) == 33
    assert cells.shape == (32, 3)
    assert electrode_nodes.shape == (8, 3)
    assert np.array_equal(electrode_counts, np.full(8, 3, dtype=np.int64))
    assert np.count_nonzero(edges[:, 2]) == 16
    for label in range(1, 9):
        assert np.count_nonzero(edges[:, 2] == label) == 2

    q4_nodes, q4_cells, q4_edges, _, _ = extension_refined_circular_mesh(
        edge_subdivisions=4,
        radial_layers=4,
        n_electrodes=16,
    )
    assert len(q4_nodes) == 513
    assert q4_cells.shape == (896, 3)
    assert q4_edges.shape == (128, 3)
    assert all(
        coordinate.denominator & (coordinate.denominator - 1) == 0
        for point in q4_nodes
        for coordinate in point
    )


def test_v727_heterogeneous_sigma_is_rational_and_reordered_explicitly(
    tmp_path: Path,
) -> None:
    case = next(case for case in EXTENSION_CASES if case.family == "heterogeneous")
    nodes, cells, *_ = extension_refined_circular_mesh(
        edge_subdivisions=case.edge_subdivisions,
        radial_layers=case.radial_layers,
        n_electrodes=case.n_electrodes,
    )
    exact_values = extension_case_cell_conductivities(case, nodes, cells)
    assert set(exact_values) == {Fraction(1, 4), Fraction(1)}
    assert all(Fraction.from_float(float(value)) == value for value in exact_values)

    fixture = prepare_extension_case_fixture(tmp_path, case)
    payload = loadmat(fixture["mat_path"], squeeze_me=True, struct_as_record=False)
    assert np.array_equal(
        np.asarray(payload["truth_elem_data"], dtype=np.float64).reshape(-1),
        np.asarray(exact_values, dtype=np.float64),
    )
    assert str(payload["conductivity_pattern"]) == "left_right"
    mesh, _ = build_mesh_from_exchange_mat(fixture["mat_path"])
    remapped = pyeidors_cell_conductivity_values(
        mesh,
        np.asarray(exact_values, dtype=np.float64),
    )
    original = np.asarray(mesh.mesh.topology.original_cell_index, dtype=np.int64)
    assert np.array_equal(
        remapped, np.asarray(exact_values, dtype=np.float64)[original]
    )
    assert not np.array_equal(original, np.arange(original.size, dtype=np.int64))


def test_v728_dynamic_qq_truth_and_drive_cache_are_certified() -> None:
    adjacent = next(
        case
        for case in EXTENSION_CASES
        if case.family == "electrode_count"
        and case.refinement_level_id == "Q0"
        and case.contact_impedance == Fraction(1)
        and case.drive_skip == 1
    )
    skip_two = next(
        case
        for case in EXTENSION_CASES
        if case.family == "electrode_count"
        and case.refinement_level_id == "Q0"
        and case.contact_impedance == Fraction(1)
        and case.drive_skip == 2
    )
    for case in (adjacent, skip_two):
        currents = extension_current_patterns(case.n_electrodes, case.drive_skip)
        assert currents.shape == (8, 8)
        assert np.array_equal(np.sum(currents, axis=0), np.zeros(8))

    extension_basis_cache_clear()
    try:
        first = solve_exact_extension_case(adjacent)
        after_first = extension_basis_cache_info()
        second = solve_exact_extension_case(skip_two)
        after_second = extension_basis_cache_info()
        assert after_first.misses == 1 and after_first.hits == 0
        assert after_second.misses == 1 and after_second.hits == 1
        for reference in (first, second):
            assert reference["exact_classic_residual_zero"] is True
            assert reference["exact_robin_residual_zero"] is True
            assert reference["exact_classic_robin_identical"] is True
            assert reference["voltage"].shape == (8, 8)
    finally:
        extension_basis_cache_clear()


def test_v727_external_extension_runners_preserve_sigma_and_blocks() -> None:
    root = Path(__file__).resolve().parents[2]
    ngsolve = (
        root / "scripts/benchmarks/ngsolve_cem_exact_extension_case.py"
    ).read_text(encoding="utf-8")
    matlab = (root / "compare_with_Eidors/compare_cem_exact_extension.m").read_text(
        encoding="utf-8"
    )
    for source in (ngsolve, matlab):
        assert "conductivity_digest" in source
        assert "assembled_blocks" in source
        assert "current_patterns" in source


def test_v731_flint_cache_requires_key_digest_and_exact_certification(
    tmp_path: Path,
) -> None:
    key = (4, 4, 16, "uniform", 1, 4, 1, 4, 1, 32)
    solution = [["0"], ["1"], ["-1"], ["0"]]
    voltage_rows = solution[1:3]
    digest = hashlib.sha256(
        json.dumps(voltage_rows, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    payload = {
        "schema": "cem-exact-flint-basis-v1",
        "system_key": list(key),
        "node_count": 1,
        "electrode_count": 2,
        "python_flint_version": "0.6.0",
        "classic_solution_basis": solution,
        "basis_voltage_sha256": digest,
        "classic_residual_zero": True,
        "robin_residual_zero": True,
        "classic_robin_identical": True,
    }
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert _validated_flint_result(cache, key)["basis_voltage_sha256"] == digest

    cache.write_text(
        json.dumps({**payload, "basis_voltage_sha256": "bad"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="truth digest mismatch"):
        _validated_flint_result(cache, key)

    helper = (
        Path(__file__).resolve().parents[2]
        / "scripts/benchmarks/cem_exact_flint_backend.py"
    ).read_text(encoding="utf-8")
    assert "fmpq_mat.solve" in helper
    assert "temporary.replace(path)" in helper
