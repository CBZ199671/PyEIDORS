#!/usr/bin/env python3
"""Compare CEM solvers against an independently assembled multiprecision truth."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpmath import mp
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_path in (ROOT, SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from dolfinx import fem

from pyeidors.forward import EITForwardModel
from pyeidors.interop.geometry_exchange import (
    STANDARD_INTEROP_FORMAT,
    build_mesh_from_exchange_mat,
    save_exchange_mat,
)

from scripts.benchmarks.cem_fair_common import (
    MESH_FINGERPRINT_SCHEMA,
    benchmark_preassembled_blocks,
    canonical_mesh_fingerprint,
    write_gmsh22,
)
from scripts.benchmarks.compare_cem_formulations import (
    BenchmarkConfig,
    _assemble_pyeidors_blocks,
    _extract_tagged_boundary_edges,
    _pattern_config,
    configure_fonts,
    trigonometric_current_patterns,
    write_json,
)


REFERENCE_SCHEMA = "cem-multiprecision-reference-v1"
FIXTURE_SCHEMA = "cem-absolute-accuracy-fan-p1-v1"
CSV_FIELDS = (
    "solver",
    "formulation",
    "electrode_voltage_relative_l2",
    "electrode_voltage_max_abs",
    "reduced_scaled_backward_residual",
    "voltage_gauge_relative_residual",
)


def canonical_fan_mesh(
    *,
    n_electrodes: int = 16,
    radius_m: float = 4.0,
    electrode_coverage: float = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic center-fan P1 mesh with one edge per electrode."""

    count = int(n_electrodes)
    nodes = np.zeros((1 + 2 * count, 2), dtype=np.float64)
    sector = 2.0 * np.pi / count
    for electrode in range(count):
        theta_start = electrode * sector
        theta_stop = theta_start + electrode_coverage * sector
        nodes[1 + 2 * electrode] = radius_m * np.asarray(
            [np.cos(theta_start), np.sin(theta_start)]
        )
        nodes[2 + 2 * electrode] = radius_m * np.asarray(
            [np.cos(theta_stop), np.sin(theta_stop)]
        )

    boundary_count = 2 * count
    cells = np.asarray(
        [
            [0, 1 + boundary, 1 + ((boundary + 1) % boundary_count)]
            for boundary in range(boundary_count)
        ],
        dtype=np.int64,
    )
    edge_rows: list[tuple[int, int, int]] = []
    electrode_nodes = np.zeros((count, 2), dtype=np.int64)
    for electrode in range(count):
        start = 1 + 2 * electrode
        stop = start + 1
        next_start = 1 + ((2 * electrode + 2) % boundary_count)
        edge_rows.append((start, stop, electrode + 1))
        edge_rows.append((stop, next_start, 0))
        electrode_nodes[electrode] = (start + 1, stop + 1)
    tagged_edges = np.asarray(edge_rows, dtype=np.int64)
    electrode_counts = np.full(count, 2, dtype=np.int64)
    return nodes, cells, tagged_edges, electrode_nodes, electrode_counts


def prepare_common_fixture(output_dir: Path) -> dict[str, Any]:
    """Write the neutral coarse mesh in MAT/MSH/JSON forms."""

    config = BenchmarkConfig(mesh_refinement=0, timing_repeats=3)
    currents, _ = trigonometric_current_patterns(
        config.n_electrodes, config.electrode_coverage
    )
    nodes, cells, tagged_edges, electrode_nodes, electrode_counts = canonical_fan_mesh(
        n_electrodes=config.n_electrodes,
        radius_m=config.radius_m,
        electrode_coverage=config.electrode_coverage,
    )
    fingerprint = canonical_mesh_fingerprint(nodes, cells, tagged_edges)
    common_dir = output_dir / "common_mesh"
    mat_path = common_dir / "cem_absolute_common_p1.mat"
    msh_path = common_dir / "cem_absolute_common_p1.msh"
    json_path = common_dir / "cem_absolute_common_p1.json"
    payload = {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "source_framework": "neutral_analytic_fixture",
        "nodes": nodes,
        "elems": cells + 1,
        "boundary_edges": tagged_edges[:, :2] + 1,
        "tagged_boundary_edges": np.column_stack(
            (tagged_edges[:, :2] + 1, tagged_edges[:, 2])
        ),
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "n_elec": config.n_electrodes,
        "background": config.conductivity_s_per_m,
        "truth_elem_data": np.full(cells.shape[0], config.conductivity_s_per_m),
        "contact_impedance": config.contact_impedance,
        "mesh_name": "cem_absolute_common_p1",
        "mesh_level": "analytic_fan_32",
        "scenario_name": "homogeneous_cem_absolute_accuracy",
        "electrode_coverage": config.electrode_coverage,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "fixture_schema": FIXTURE_SCHEMA,
        "current_patterns": currents,
    }
    save_exchange_mat(mat_path, payload)
    write_gmsh22(
        msh_path,
        nodes,
        cells,
        tagged_edges,
        config.n_electrodes,
    )
    metadata = {
        "fixture_schema": FIXTURE_SCHEMA,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "nodes": int(nodes.shape[0]),
        "cells": int(cells.shape[0]),
        "boundary_edges": int(tagged_edges.shape[0]),
        "electrode_edges": int(np.count_nonzero(tagged_edges[:, 2] > 0)),
        "potential_order": 1,
        "scalar_dtype": "float64",
        "mat": str(mat_path),
        "msh": str(msh_path),
    }
    write_json(json_path, metadata)
    return {
        **metadata,
        "mat_path": mat_path,
        "msh_path": msh_path,
        "json_path": json_path,
    }


def run_pyeidors_fixture(output_dir: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    """Solve the neutral fixture with the PyEIDORS float64 assembly."""

    config = BenchmarkConfig(mesh_refinement=0, timing_repeats=3)
    eit_mesh, _ = build_mesh_from_exchange_mat(Path(fixture["mat_path"]))
    impedance = np.full(config.n_electrodes, config.contact_impedance, dtype=np.float64)
    model = EITForwardModel(
        n_elec=config.n_electrodes,
        pattern_config=_pattern_config(config),
        z=impedance,
        mesh=eit_mesh,
        potential_order=1,
        linear_backend="scipy",
    )
    if np.dtype(model.scalar_dtype) != np.dtype(np.float64):
        raise RuntimeError("absolute CEM fixture requires real float64 PyEIDORS")
    loaded_edges = _extract_tagged_boundary_edges(eit_mesh, list(model.electrode_tags))
    loaded_fingerprint = canonical_mesh_fingerprint(
        np.asarray(eit_mesh.coordinates(), dtype=np.float64)[:, :2],
        np.asarray(eit_mesh.cells(), dtype=np.int64),
        loaded_edges,
    )
    if loaded_fingerprint != fixture["mesh_fingerprint"]:
        raise RuntimeError(
            "PyEIDORS absolute fixture fingerprint mismatch: "
            f"{loaded_fingerprint} != {fixture['mesh_fingerprint']}"
        )

    sigma = fem.Function(model.V_sigma)
    sigma.x.array[:] = config.conductivity_s_per_m
    currents, _ = trigonometric_current_patterns(
        config.n_electrodes, config.electrode_coverage
    )
    robin_matrix, coupling, electrode_matrix = _assemble_pyeidors_blocks(model, sigma)
    timing, voltages, parity = benchmark_preassembled_blocks(
        robin_matrix,
        coupling,
        electrode_matrix,
        currents,
        repeats=3,
    )
    report = {
        "solver": "PyEIDORS/DOLFINx",
        "fixture_schema": FIXTURE_SCHEMA,
        "physical_config": asdict(config),
        "discretization": {
            "vertices": int(eit_mesh.num_vertices()),
            "cells": int(eit_mesh.num_cells()),
            "boundary_edges": int(loaded_edges.shape[0]),
            "degrees_of_freedom": int(model.dofs),
            "element_family": "DOLFINx P1 Lagrange triangle",
            "potential_order": 1,
            "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
            "mesh_fingerprint": loaded_fingerprint,
            "mesh_import_verified": True,
        },
        "linear_solver": {
            "classic": "SciPy SuperLU augmented CEM",
            "robin": "SciPy SuperLU A_R plus dense reduced LU",
            "scalar_dtype": "float64",
        },
        "timing": timing,
        "within_solver": parity,
        "raw_electrode_voltages": {
            formulation: np.asarray(voltage, dtype=np.float64).tolist()
            for formulation, voltage in voltages.items()
        },
    }
    write_json(output_dir / "pyeidors_report.json", report)
    return report


def _mp_exact_float(value: float):
    numerator, denominator = float(value).as_integer_ratio()
    return mp.mpf(numerator) / mp.mpf(denominator)


def _mp_frobenius(matrix) -> Any:
    return mp.sqrt(
        mp.fsum(
            abs(matrix[row, column]) ** 2
            for row in range(matrix.rows)
            for column in range(matrix.cols)
        )
    )


def _mp_max_abs(matrix) -> Any:
    return max(
        (
            abs(matrix[row, column])
            for row in range(matrix.rows)
            for column in range(matrix.cols)
        ),
        default=mp.mpf("0"),
    )


def _mp_matrix_from_float64(values: np.ndarray):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("multiprecision matrix input must be two-dimensional")
    matrix = mp.matrix(array.shape[0], array.shape[1])
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            matrix[row, column] = _mp_exact_float(array[row, column])
    return matrix


def assemble_multiprecision_cem(
    nodes: np.ndarray,
    cells: np.ndarray,
    tagged_edges: np.ndarray,
    *,
    n_electrodes: int,
    conductivity: float,
    contact_impedance: float,
) -> tuple[Any, Any, Any]:
    """Assemble P1 CEM blocks analytically in the active mpmath precision."""

    node_array = np.asarray(nodes, dtype=np.float64)
    cell_array = np.asarray(cells, dtype=np.int64)
    edge_array = np.asarray(tagged_edges, dtype=np.int64)
    coordinates = [
        (_mp_exact_float(x), _mp_exact_float(y)) for x, y in node_array[:, :2]
    ]
    sigma = _mp_exact_float(conductivity)
    impedance = _mp_exact_float(contact_impedance)
    node_count = node_array.shape[0]
    a_r = mp.zeros(node_count, node_count)
    coupling = mp.zeros(node_count, n_electrodes)
    electrode_matrix = mp.zeros(n_electrodes, n_electrodes)

    for triangle in cell_array:
        indices = [int(value) for value in triangle]
        x1, y1 = coordinates[indices[0]]
        x2, y2 = coordinates[indices[1]]
        x3, y3 = coordinates[indices[2]]
        determinant = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = abs(determinant) / 2
        if area == 0:
            raise ValueError("canonical CEM mesh contains a zero-area triangle")
        b = (y2 - y3, y3 - y1, y1 - y2)
        c = (x3 - x2, x1 - x3, x2 - x1)
        for local_row, global_row in enumerate(indices):
            for local_column, global_column in enumerate(indices):
                a_r[global_row, global_column] += (
                    sigma
                    * (b[local_row] * b[local_column] + c[local_row] * c[local_column])
                    / (4 * area)
                )

    for vertex_a, vertex_b, label in edge_array:
        electrode_label = int(label)
        if electrode_label <= 0:
            continue
        electrode = electrode_label - 1
        a = int(vertex_a)
        b_index = int(vertex_b)
        dx = coordinates[a][0] - coordinates[b_index][0]
        dy = coordinates[a][1] - coordinates[b_index][1]
        edge_length_over_z = mp.sqrt(dx * dx + dy * dy) / impedance
        a_r[a, a] += edge_length_over_z / 3
        a_r[b_index, b_index] += edge_length_over_z / 3
        a_r[a, b_index] += edge_length_over_z / 6
        a_r[b_index, a] += edge_length_over_z / 6
        coupling[a, electrode] -= edge_length_over_z / 2
        coupling[b_index, electrode] -= edge_length_over_z / 2
        electrode_matrix[electrode, electrode] += edge_length_over_z
    return a_r, coupling, electrode_matrix


def _zero_sum_basis_mp(count: int):
    basis = mp.zeros(count, count - 1)
    for column in range(1, count):
        scale = mp.sqrt(column * (column + 1))
        for row in range(column):
            basis[row, column - 1] = 1 / scale
        basis[column, column - 1] = -column / scale
    return basis


def _solve_columns(matrix, right_hand_side):
    solution = mp.zeros(matrix.rows, right_hand_side.cols)
    for column in range(right_hand_side.cols):
        rhs_column = mp.matrix(
            [right_hand_side[row, column] for row in range(right_hand_side.rows)]
        )
        solved = mp.lu_solve(matrix, rhs_column)
        for row in range(matrix.rows):
            solution[row, column] = solved[row]
    return solution


def solve_reference_at_dps(
    nodes: np.ndarray,
    cells: np.ndarray,
    tagged_edges: np.ndarray,
    currents: np.ndarray,
    *,
    n_electrodes: int,
    conductivity: float,
    contact_impedance: float,
    dps: int,
) -> dict[str, Any]:
    """Solve the independently assembled classic CEM system at ``dps`` digits."""

    with mp.workdps(int(dps)):
        a_r, coupling, electrode_matrix = assemble_multiprecision_cem(
            nodes,
            cells,
            tagged_edges,
            n_electrodes=n_electrodes,
            conductivity=conductivity,
            contact_impedance=contact_impedance,
        )
        node_count = a_r.rows
        full_size = node_count + n_electrodes + 1
        full_matrix = mp.zeros(full_size, full_size)
        for row in range(node_count):
            for column in range(node_count):
                full_matrix[row, column] = a_r[row, column]
        for row in range(node_count):
            for electrode in range(n_electrodes):
                value = coupling[row, electrode]
                full_matrix[row, node_count + electrode] = value
                full_matrix[node_count + electrode, row] = value
        for row in range(n_electrodes):
            for column in range(n_electrodes):
                full_matrix[node_count + row, node_count + column] = electrode_matrix[
                    row, column
                ]
            full_matrix[node_count + row, full_size - 1] = 1
            full_matrix[full_size - 1, node_count + row] = 1

        current_matrix = _mp_matrix_from_float64(currents)
        rhs = mp.zeros(full_size, current_matrix.cols)
        for electrode in range(n_electrodes):
            for column in range(current_matrix.cols):
                rhs[node_count + electrode, column] = current_matrix[electrode, column]
        solution = _solve_columns(full_matrix, rhs)
        voltage = mp.zeros(n_electrodes, current_matrix.cols)
        for electrode in range(n_electrodes):
            for column in range(current_matrix.cols):
                voltage[electrode, column] = solution[node_count + electrode, column]

        residual = full_matrix * solution - rhs
        scaled_residual = _mp_frobenius(residual) / (
            _mp_frobenius(full_matrix) * _mp_frobenius(solution) + _mp_frobenius(rhs)
        )
        basis = _zero_sum_basis_mp(n_electrodes)
        response_basis = _solve_columns(a_r, coupling * basis)
        reduced_map = basis.T * (electrode_matrix * basis - coupling.T * response_basis)
        return {
            "dps": int(dps),
            "voltage": voltage,
            "basis": basis,
            "reduced_map": reduced_map,
            "currents": current_matrix,
            "scaled_full_residual": scaled_residual,
        }


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    cells = np.asarray(payload["elems"], dtype=np.int64).reshape(-1, 3) - 1
    tagged_edges = np.asarray(payload["tagged_boundary_edges"], dtype=np.int64).reshape(
        -1, 3
    )
    tagged_edges[:, :2] -= 1
    return {
        "nodes": nodes,
        "cells": cells,
        "tagged_edges": tagged_edges,
        "currents": np.asarray(payload["current_patterns"], dtype=np.float64),
        "n_electrodes": int(np.asarray(payload["n_elec"]).reshape(-1)[0]),
        "conductivity": float(np.asarray(payload["background"]).reshape(-1)[0]),
        "contact_impedance": float(
            np.asarray(payload["contact_impedance"]).reshape(-1)[0]
        ),
        "mesh_fingerprint": str(np.asarray(payload["mesh_fingerprint"]).reshape(-1)[0]),
    }


def _solver_accuracy_metrics(
    candidate: np.ndarray,
    reference: dict[str, Any],
) -> dict[str, Any]:
    with mp.workdps(int(reference["dps"])):
        candidate_mp = _mp_matrix_from_float64(candidate)
        voltage_reference = reference["voltage"]
        delta = candidate_mp - voltage_reference
        relative_error = _mp_frobenius(delta) / _mp_frobenius(voltage_reference)
        basis = reference["basis"]
        reduced_map = reference["reduced_map"]
        reduced_solution = basis.T * candidate_mp
        reduced_rhs = basis.T * reference["currents"]
        reduced_residual = reduced_map * reduced_solution - reduced_rhs
        backward_residual = _mp_frobenius(reduced_residual) / (
            _mp_frobenius(reduced_map) * _mp_frobenius(reduced_solution)
            + _mp_frobenius(reduced_rhs)
        )
        gauge = mp.matrix(1, candidate_mp.cols)
        for column in range(candidate_mp.cols):
            gauge[0, column] = mp.fsum(
                candidate_mp[row, column] for row in range(candidate_mp.rows)
            )
        gauge_residual = _mp_frobenius(gauge) / _mp_frobenius(candidate_mp)
        per_rhs = []
        for column in range(candidate_mp.cols):
            delta_norm = mp.sqrt(
                mp.fsum(
                    abs(delta[row, column]) ** 2 for row in range(candidate_mp.rows)
                )
            )
            reference_norm = mp.sqrt(
                mp.fsum(
                    abs(voltage_reference[row, column]) ** 2
                    for row in range(candidate_mp.rows)
                )
            )
            per_rhs.append(float(delta_norm / reference_norm))
        return {
            "electrode_voltage_relative_l2": float(relative_error),
            "electrode_voltage_relative_l2_decimal": mp.nstr(relative_error, 40),
            "electrode_voltage_max_abs": float(_mp_max_abs(delta)),
            "electrode_voltage_max_abs_decimal": mp.nstr(_mp_max_abs(delta), 40),
            "reduced_scaled_backward_residual": float(backward_residual),
            "reduced_scaled_backward_residual_decimal": mp.nstr(backward_residual, 40),
            "voltage_gauge_relative_residual": float(gauge_residual),
            "per_rhs_relative_l2": per_rhs,
        }


def _validate_report(
    report: dict[str, Any],
    fixture: dict[str, Any],
) -> None:
    solver = str(report.get("solver", "unknown"))
    discretization = report.get("discretization", {})
    if discretization.get("mesh_fingerprint") != fixture["mesh_fingerprint"]:
        raise ValueError(f"{solver} absolute fixture mesh fingerprint mismatch")
    if not bool(discretization.get("mesh_import_verified", False)):
        raise ValueError(f"{solver} did not verify the absolute fixture import")
    if int(discretization.get("potential_order", -1)) != 1:
        raise ValueError(f"{solver} absolute fixture must use P1")
    if report.get("linear_solver", {}).get("scalar_dtype") != "float64":
        raise ValueError(f"{solver} absolute fixture must use float64")
    raw = report.get("raw_electrode_voltages", {})
    expected_shape = (
        int(fixture["n_electrodes"]),
        int(fixture["currents"].shape[1]),
    )
    for formulation in ("classic", "robin_transconductance"):
        values = np.asarray(raw.get(formulation), dtype=np.float64)
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(
                f"{solver} {formulation} raw voltage shape/finite mismatch: "
                f"{values.shape} != {expected_shape}"
            )


def _write_accuracy_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in metrics:
            writer.writerow({field: item[field] for field in CSV_FIELDS})


def _plot_accuracy(metrics: list[dict[str, Any]], path: Path) -> None:
    configure_fonts()
    solvers = sorted({str(item["solver"]) for item in metrics})
    formulations = ("classic", "robin_transconductance")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    width = 0.36
    positions = np.arange(len(solvers), dtype=float)
    fields = (
        ("electrode_voltage_relative_l2", "Relative error vs 128-dps truth"),
        ("reduced_scaled_backward_residual", "Scaled backward residual"),
    )
    for axis, (field, ylabel) in zip(axes, fields, strict=True):
        for formulation_index, formulation in enumerate(formulations):
            by_solver = {
                str(item["solver"]): item
                for item in metrics
                if item["formulation"] == formulation
            }
            values = [float(by_solver[solver][field]) for solver in solvers]
            offset = (formulation_index - 0.5) * width
            axis.bar(
                positions + offset,
                values,
                width,
                label=formulation,
            )
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, solvers, rotation=12, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(fontsize=8)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontname("Times New Roman")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def compare_against_reference(
    mesh_mat: Path,
    report_paths: list[Path],
    output_dir: Path,
    *,
    check_dps: int,
    primary_dps: int,
) -> dict[str, Any]:
    """Certify the reference and rank every solver/formulation against it."""

    if check_dps >= primary_dps:
        raise ValueError("check_dps must be lower than primary_dps")
    fixture = _load_fixture(mesh_mat)
    fingerprint = canonical_mesh_fingerprint(
        fixture["nodes"], fixture["cells"], fixture["tagged_edges"]
    )
    if fingerprint != fixture["mesh_fingerprint"]:
        raise ValueError("multiprecision fixture fingerprint mismatch")

    reference_check = solve_reference_at_dps(
        fixture["nodes"],
        fixture["cells"],
        fixture["tagged_edges"],
        fixture["currents"],
        n_electrodes=fixture["n_electrodes"],
        conductivity=fixture["conductivity"],
        contact_impedance=fixture["contact_impedance"],
        dps=check_dps,
    )
    reference = solve_reference_at_dps(
        fixture["nodes"],
        fixture["cells"],
        fixture["tagged_edges"],
        fixture["currents"],
        n_electrodes=fixture["n_electrodes"],
        conductivity=fixture["conductivity"],
        contact_impedance=fixture["contact_impedance"],
        dps=primary_dps,
    )
    with mp.workdps(primary_dps):
        reference_delta = reference["voltage"] - reference_check["voltage"]
        convergence = _mp_frobenius(reference_delta) / _mp_frobenius(
            reference["voltage"]
        )
    convergence_limit = mp.mpf("1e-60")
    residual_limit = mp.mpf("1e-70")
    if convergence > convergence_limit:
        raise RuntimeError(
            f"multiprecision reference did not converge: {mp.nstr(convergence, 20)}"
        )
    if reference["scaled_full_residual"] > residual_limit:
        raise RuntimeError(
            "multiprecision reference scaled residual too large: "
            f"{mp.nstr(reference['scaled_full_residual'], 20)}"
        )

    reports = [json_load(path) for path in report_paths]
    solver_names = [str(report.get("solver", "unknown")) for report in reports]
    if len(set(solver_names)) != 3:
        raise ValueError(f"expected three distinct solver reports, got {solver_names}")
    metrics: list[dict[str, Any]] = []
    for report in reports:
        _validate_report(report, fixture)
        for formulation in ("classic", "robin_transconductance"):
            candidate = np.asarray(
                report["raw_electrode_voltages"][formulation],
                dtype=np.float64,
            )
            metrics.append(
                {
                    "solver": report["solver"],
                    "formulation": formulation,
                    **_solver_accuracy_metrics(candidate, reference),
                }
            )

    rankings: dict[str, list[dict[str, Any]]] = {}
    for formulation in ("classic", "robin_transconductance"):
        selected = sorted(
            (item for item in metrics if item["formulation"] == formulation),
            key=lambda item: item["electrode_voltage_relative_l2"],
        )
        rankings[formulation] = [
            {
                "rank": rank,
                "solver": item["solver"],
                "electrode_voltage_relative_l2": item["electrode_voltage_relative_l2"],
                "ratio_to_best": item["electrode_voltage_relative_l2"]
                / selected[0]["electrode_voltage_relative_l2"],
            }
            for rank, item in enumerate(selected, start=1)
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cem_absolute_accuracy.csv"
    plot_path = output_dir / "cem_absolute_accuracy.png"
    _write_accuracy_csv(csv_path, metrics)
    _plot_accuracy(metrics, plot_path)
    report = {
        "schema": REFERENCE_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "absolute rounding/assembly/solve accuracy for one canonical discrete "
            "P1 CEM fixture; not continuum PDE discretization accuracy"
        ),
        "fixture": {
            "schema": FIXTURE_SCHEMA,
            "mesh_fingerprint": fingerprint,
            "vertices": int(fixture["nodes"].shape[0]),
            "cells": int(fixture["cells"].shape[0]),
            "boundary_edges": int(fixture["tagged_edges"].shape[0]),
            "n_electrodes": int(fixture["n_electrodes"]),
        },
        "reference": {
            "assembly": (
                "independent analytic P1 triangle stiffness and exact line-electrode "
                "mass/coupling terms; float64 inputs promoted by exact rational ratio"
            ),
            "matrix_source_from_any_solver": False,
            "check_dps": int(check_dps),
            "primary_dps": int(primary_dps),
            "check_vs_primary_voltage_relative_l2": float(convergence),
            "check_vs_primary_voltage_relative_l2_decimal": mp.nstr(convergence, 40),
            "primary_scaled_full_residual": float(reference["scaled_full_residual"]),
            "primary_scaled_full_residual_decimal": mp.nstr(
                reference["scaled_full_residual"], 40
            ),
            "electrode_voltage_decimal_50": [
                [
                    mp.nstr(reference["voltage"][row, column], 50)
                    for column in range(reference["voltage"].cols)
                ]
                for row in range(reference["voltage"].rows)
            ],
        },
        "solver_reports": [str(path) for path in report_paths],
        "metrics": metrics,
        "rankings": rankings,
        "artifacts": {
            "csv": csv_path.name,
            "plot": plot_path.name,
        },
    }
    write_json(output_dir / "cem_absolute_accuracy.json", report)
    return report


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="write fixture and run PyEIDORS")
    prepare.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "cem_absolute_accuracy",
    )
    compare = subparsers.add_parser("compare", help="certify truth and rank reports")
    compare.add_argument("--mesh-mat", type=Path, required=True)
    compare.add_argument("--solver-report", type=Path, action="append", required=True)
    compare.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "cem_absolute_accuracy",
    )
    compare.add_argument("--check-dps", type=int, default=80)
    compare.add_argument("--primary-dps", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        fixture = prepare_common_fixture(output_dir)
        run_pyeidors_fixture(output_dir, fixture)
        print(f"CEM absolute fixture and PyEIDORS report: {output_dir}")
        return 0
    report = compare_against_reference(
        args.mesh_mat.resolve(),
        [path.resolve() for path in args.solver_report],
        output_dir,
        check_dps=int(args.check_dps),
        primary_dps=int(args.primary_dps),
    )
    print(
        "CEM absolute accuracy winners: "
        + ", ".join(
            f"{formulation}={ranking[0]['solver']}"
            for formulation, ranking in report["rankings"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
