#!/usr/bin/env python3
"""Run a multi-case CEM benchmark against exact rational discrete solutions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpmath import mp
import numpy as np
from sympy import Matrix, QQ, Rational, SparseMatrix

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_path in (ROOT, SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from pyeidors.forward import EITForwardModel
from pyeidors.interop.geometry_exchange import (
    STANDARD_INTEROP_FORMAT,
    build_mesh_from_exchange_mat,
    save_exchange_mat,
)

from scripts.benchmarks.cem_fair_common import (
    DEFAULT_OPERATIONS_PER_SAMPLE,
    MESH_FINGERPRINT_SCHEMA,
    TIMING_SCHEMA,
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
    write_json,
)


SUITE_SCHEMA = "cem-exact-rational-suite-v2"
GEOMETRY_SCHEMA = "cem-rational-fixed-32gon-refinement-v2"
METRIC_SCHEMA = "cem-exact-accuracy-metrics-v2"
TIMING_REPEATS = 11
TIMING_OPERATIONS_PER_SAMPLE = DEFAULT_OPERATIONS_PER_SAMPLE
N_ELECTRODES = 16
BOUNDARY_COUNT = 2 * N_ELECTRODES
COORDINATE_SCALE = 8192
CIRCLE_INTEGER_RADIUS = 5525
ELECTRODE_TANGENT_DIVISOR = 4
FORMULATIONS = ("classic", "robin_transconductance")
CSV_FIELDS = (
    "case_id",
    "refinement_level_id",
    "edge_subdivisions",
    "radial_layers",
    "nodes",
    "cells",
    "solver",
    "formulation",
    "truth_relative_l2",
    "truth_max_abs",
    "exact_reduced_scaled_backward_residual",
    "voltage_gauge_relative_residual",
    "reduced_condition_number_2_estimate",
    "classic_robin_relative_l2",
)
TIMING_CSV_FIELDS = (
    "case_id",
    "solver",
    "formulation",
    "cold_median_seconds",
    "cold_iqr_seconds",
    "setup_median_seconds",
    "setup_iqr_seconds",
    "cold_solve_median_seconds",
    "cold_solve_iqr_seconds",
    "warm_reuse_median_seconds",
    "warm_reuse_iqr_seconds",
    "cold_over_warm_reuse_speedup",
    "assembly_seconds",
    "mesh_import_seconds",
)


@dataclass(frozen=True)
class ExactCase:
    case_id: str
    label: str
    refinement_level_id: str
    edge_subdivisions: int
    radial_layers: int
    conductivity_numerator: int
    conductivity_denominator: int
    impedance_numerator: int
    impedance_denominator: int
    drive_skip: int
    drive_label: str

    @property
    def conductivity(self) -> Fraction:
        return Fraction(
            self.conductivity_numerator,
            self.conductivity_denominator,
        )

    @property
    def contact_impedance(self) -> Fraction:
        return Fraction(
            self.impedance_numerator,
            self.impedance_denominator,
        )

    @property
    def ring_count(self) -> int:
        """Return internal ring count for backward-compatible metadata."""

        return self.radial_layers - 1


CASES = (
    ExactCase("G1", "q0_baseline", "Q0", 1, 1, 1, 4, 1, 1, 1, "adjacent"),
    ExactCase("G2", "q1_baseline", "Q1", 1, 2, 1, 4, 1, 1, 1, "adjacent"),
    ExactCase("G3", "q2_baseline", "Q2", 2, 2, 1, 4, 1, 1, 1, "adjacent"),
    ExactCase("G4", "q1_low_z", "Q1", 1, 2, 1, 4, 1, 8, 1, "adjacent"),
    ExactCase("G5", "q1_high_z", "Q1", 1, 2, 1, 4, 8, 1, 1, "adjacent"),
    ExactCase("G6", "q1_high_sigma", "Q1", 1, 2, 1, 1, 1, 1, 1, "adjacent"),
    ExactCase("G7", "q1_skip4_drive", "Q1", 1, 2, 1, 4, 1, 1, 4, "skip-4"),
    ExactCase("G8", "q3_baseline", "Q3", 2, 4, 1, 4, 1, 1, 1, "adjacent"),
)
REFINEMENT_CASE_IDS = ("G1", "G2", "G3", "G8")


def _first_quadrant_centers() -> tuple[tuple[int, int], ...]:
    return (
        (5427, 1036),
        (4557, 3124),
        (3124, 4557),
        (1036, 5427),
    )


def integer_electrode_centers() -> tuple[tuple[int, int], ...]:
    """Return 16 counter-clockwise lattice points on one integer circle."""

    q1 = _first_quadrant_centers()
    q2 = tuple((-x, y) for x, y in reversed(q1))
    q3 = tuple((-x, -y) for x, y in q1)
    q4 = tuple((x, -y) for x, y in reversed(q1))
    centers = (*q1, *q2, *q3, *q4)
    expected_radius_squared = CIRCLE_INTEGER_RADIUS**2
    if any(x * x + y * y != expected_radius_squared for x, y in centers):
        raise RuntimeError("rational circular centers do not share one radius")
    return centers


def exact_boundary_nodes() -> tuple[tuple[Fraction, Fraction], ...]:
    """Return 32 exactly co-circular dyadic electrode endpoints."""

    nodes: list[tuple[Fraction, Fraction]] = []
    scale = Fraction(1, COORDINATE_SCALE)
    for center_x, center_y in integer_electrode_centers():
        half_tangent_x = Fraction(-center_y, 2 * ELECTRODE_TANGENT_DIVISOR)
        half_tangent_y = Fraction(center_x, 2 * ELECTRODE_TANGENT_DIVISOR)
        start = (
            (Fraction(center_x) - half_tangent_x) * scale,
            (Fraction(center_y) - half_tangent_y) * scale,
        )
        stop = (
            (Fraction(center_x) + half_tangent_x) * scale,
            (Fraction(center_y) + half_tangent_y) * scale,
        )
        nodes.extend((start, stop))

    radii_squared = {x * x + y * y for x, y in nodes}
    if len(radii_squared) != 1:
        raise RuntimeError("rational electrode endpoints are not exactly co-circular")
    for value in (coordinate for point in nodes for coordinate in point):
        if value.denominator & (value.denominator - 1):
            raise RuntimeError("circular fixture coordinate is not dyadic")
    return tuple(nodes)


def _ring_radii(ring_count: int) -> tuple[Fraction, ...]:
    if ring_count == 0:
        return (Fraction(1),)
    if ring_count == 1:
        return (Fraction(1, 2), Fraction(1))
    if ring_count == 2:
        return (Fraction(1, 2), Fraction(3, 4), Fraction(1))
    raise ValueError("ring_count must be 0, 1, or 2")


def exact_circular_mesh(
    ring_count: int,
) -> tuple[
    tuple[tuple[Fraction, Fraction], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build a concentric exact-rational circular P1 mesh."""

    boundary = exact_boundary_nodes()
    nodes: list[tuple[Fraction, Fraction]] = [(Fraction(0), Fraction(0))]
    radii = _ring_radii(int(ring_count))
    for radius in radii:
        nodes.extend((radius * x, radius * y) for x, y in boundary)

    cells: list[tuple[int, int, int]] = []
    first_ring = 1
    for index in range(BOUNDARY_COUNT):
        next_index = (index + 1) % BOUNDARY_COUNT
        cells.append((0, first_ring + index, first_ring + next_index))
    for inner_ring in range(len(radii) - 1):
        inner_offset = 1 + inner_ring * BOUNDARY_COUNT
        outer_offset = inner_offset + BOUNDARY_COUNT
        for index in range(BOUNDARY_COUNT):
            next_index = (index + 1) % BOUNDARY_COUNT
            cells.append(
                (inner_offset + index, outer_offset + index, outer_offset + next_index)
            )
            cells.append(
                (
                    inner_offset + index,
                    outer_offset + next_index,
                    inner_offset + next_index,
                )
            )

    outer_offset = 1 + (len(radii) - 1) * BOUNDARY_COUNT
    edges = np.asarray(
        [
            (
                outer_offset + index,
                outer_offset + ((index + 1) % BOUNDARY_COUNT),
                index // 2 + 1 if index % 2 == 0 else 0,
            )
            for index in range(BOUNDARY_COUNT)
        ],
        dtype=np.int64,
    )
    electrode_nodes = np.asarray(
        [
            (outer_offset + 2 * electrode, outer_offset + 2 * electrode + 1)
            for electrode in range(N_ELECTRODES)
        ],
        dtype=np.int64,
    )
    electrode_counts = np.full(N_ELECTRODES, 2, dtype=np.int64)
    cell_array = np.asarray(cells, dtype=np.int64)

    for triangle in cell_array:
        a, b, c = (nodes[int(index)] for index in triangle)
        determinant = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if determinant <= 0:
            raise RuntimeError("exact circular mesh contains a non-positive triangle")
    return tuple(nodes), cell_array, edges, electrode_nodes, electrode_counts


def _positive_power_of_two(value: int, *, name: str) -> int:
    count = int(value)
    if count <= 0 or count & (count - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return count


def exact_refined_circular_mesh(
    *,
    edge_subdivisions: int,
    radial_layers: int,
) -> tuple[
    tuple[tuple[Fraction, Fraction], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Refine one fixed rational 32-gon with nested dyadic P1 triangles."""

    subdivisions = _positive_power_of_two(
        edge_subdivisions,
        name="edge_subdivisions",
    )
    layers = _positive_power_of_two(radial_layers, name="radial_layers")
    base_boundary = exact_boundary_nodes()
    boundary: list[tuple[Fraction, Fraction]] = []
    boundary_labels: list[int] = []
    for base_index, start in enumerate(base_boundary):
        stop = base_boundary[(base_index + 1) % BOUNDARY_COUNT]
        for sub_index in range(subdivisions):
            fraction = Fraction(sub_index, subdivisions)
            boundary.append(
                (
                    (1 - fraction) * start[0] + fraction * stop[0],
                    (1 - fraction) * start[1] + fraction * stop[1],
                )
            )
            boundary_labels.append(base_index // 2 + 1 if base_index % 2 == 0 else 0)

    refined_boundary_count = len(boundary)
    nodes: list[tuple[Fraction, Fraction]] = [(Fraction(0), Fraction(0))]
    for layer in range(1, layers + 1):
        radius = Fraction(layer, layers)
        nodes.extend((radius * x, radius * y) for x, y in boundary)

    cells: list[tuple[int, int, int]] = []
    for index in range(refined_boundary_count):
        next_index = (index + 1) % refined_boundary_count
        cells.append((0, 1 + index, 1 + next_index))
    for inner_ring in range(layers - 1):
        inner_offset = 1 + inner_ring * refined_boundary_count
        outer_offset = inner_offset + refined_boundary_count
        for index in range(refined_boundary_count):
            next_index = (index + 1) % refined_boundary_count
            cells.append(
                (
                    inner_offset + index,
                    outer_offset + index,
                    outer_offset + next_index,
                )
            )
            cells.append(
                (
                    inner_offset + index,
                    outer_offset + next_index,
                    inner_offset + next_index,
                )
            )

    outer_offset = 1 + (layers - 1) * refined_boundary_count
    edges = np.asarray(
        [
            (
                outer_offset + index,
                outer_offset + ((index + 1) % refined_boundary_count),
                boundary_labels[index],
            )
            for index in range(refined_boundary_count)
        ],
        dtype=np.int64,
    )
    electrode_nodes = np.asarray(
        [
            [
                outer_offset + 2 * electrode * subdivisions + sub_index
                for sub_index in range(subdivisions + 1)
            ]
            for electrode in range(N_ELECTRODES)
        ],
        dtype=np.int64,
    )
    electrode_counts = np.full(
        N_ELECTRODES,
        subdivisions + 1,
        dtype=np.int64,
    )
    cell_array = np.asarray(cells, dtype=np.int64)
    for triangle in cell_array:
        a, b, c = (nodes[int(index)] for index in triangle)
        determinant = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if determinant <= 0:
            raise RuntimeError("refined exact mesh contains a non-positive triangle")
    for value in (coordinate for point in nodes for coordinate in point):
        if value.denominator & (value.denominator - 1):
            raise RuntimeError("refined exact mesh coordinate is not dyadic")
    return tuple(nodes), cell_array, edges, electrode_nodes, electrode_counts


def exact_case_mesh(
    case: ExactCase,
) -> tuple[
    tuple[tuple[Fraction, Fraction], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build the declared fixed-domain mesh for one exact-suite case."""

    return exact_refined_circular_mesh(
        edge_subdivisions=case.edge_subdivisions,
        radial_layers=case.radial_layers,
    )


def float_nodes(
    exact_nodes: tuple[tuple[Fraction, Fraction], ...],
) -> np.ndarray:
    nodes = np.asarray(
        [[float(x), float(y)] for x, y in exact_nodes],
        dtype=np.float64,
    )
    for exact_point, float_point in zip(exact_nodes, nodes, strict=True):
        for exact_value, float_value in zip(exact_point, float_point, strict=True):
            if Fraction.from_float(float(float_value)) != exact_value:
                raise RuntimeError("exact circular coordinate changed in float64")
    return nodes


def exact_current_patterns(drive_skip: int) -> np.ndarray:
    skip = int(drive_skip)
    if skip <= 0 or skip >= N_ELECTRODES:
        raise ValueError("drive_skip must be between 1 and 15")
    currents = np.zeros((N_ELECTRODES, N_ELECTRODES), dtype=np.float64)
    for column in range(N_ELECTRODES):
        currents[column, column] = 1.0
        currents[(column + skip) % N_ELECTRODES, column] = -1.0
    if not np.array_equal(np.sum(currents, axis=0), np.zeros(N_ELECTRODES)):
        raise RuntimeError("exact current patterns are not zero-sum")
    return currents


def _fraction_string(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _case_directory(output_dir: Path, case: ExactCase) -> Path:
    return output_dir / "cases" / f"{case.case_id}_{case.label}"


def prepare_case_fixture(output_dir: Path, case: ExactCase) -> dict[str, Any]:
    exact_nodes, cells, edges, electrode_nodes, electrode_counts = exact_case_mesh(case)
    nodes = float_nodes(exact_nodes)
    currents = exact_current_patterns(case.drive_skip)
    fingerprint = canonical_mesh_fingerprint(nodes, cells, edges)
    case_dir = _case_directory(output_dir, case)
    common_dir = case_dir / "common_mesh"
    mat_path = common_dir / "cem_exact_common_p1.mat"
    msh_path = common_dir / "cem_exact_common_p1.msh"
    metadata_path = common_dir / "cem_exact_common_p1.json"
    payload = {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "source_framework": "exact_rational_circular_fixture",
        "nodes": nodes,
        "elems": cells + 1,
        "boundary_edges": edges[:, :2] + 1,
        "tagged_boundary_edges": np.column_stack((edges[:, :2] + 1, edges[:, 2])),
        "electrode_nodes": electrode_nodes + 1,
        "electrode_node_counts": electrode_counts,
        "n_elec": N_ELECTRODES,
        "background": float(case.conductivity),
        "truth_elem_data": np.full(cells.shape[0], float(case.conductivity)),
        "contact_impedance": float(case.contact_impedance),
        "mesh_name": f"cem_exact_{case.case_id.lower()}",
        "mesh_level": case.refinement_level_id,
        "scenario_name": case.label,
        "electrode_coverage": 0.64,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "suite_schema": SUITE_SCHEMA,
        "geometry_schema": GEOMETRY_SCHEMA,
        "case_id": case.case_id,
        "ring_count": case.ring_count,
        "refinement_level_id": case.refinement_level_id,
        "edge_subdivisions": case.edge_subdivisions,
        "radial_layers": case.radial_layers,
        "current_patterns": currents,
        "drive_skip": case.drive_skip,
    }
    save_exchange_mat(mat_path, payload)
    write_gmsh22(msh_path, nodes, cells, edges, N_ELECTRODES)
    metadata = {
        "suite_schema": SUITE_SCHEMA,
        "geometry_schema": GEOMETRY_SCHEMA,
        "case": asdict(case),
        "case_id": case.case_id,
        "label": case.label,
        "ring_count": case.ring_count,
        "refinement_level_id": case.refinement_level_id,
        "edge_subdivisions": case.edge_subdivisions,
        "radial_layers": case.radial_layers,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "nodes": int(nodes.shape[0]),
        "cells": int(cells.shape[0]),
        "boundary_edges": int(edges.shape[0]),
        "electrode_edges": int(np.count_nonzero(edges[:, 2] > 0)),
        "n_electrodes": N_ELECTRODES,
        "potential_order": 1,
        "scalar_dtype": "float64",
        "conductivity": float(case.conductivity),
        "conductivity_exact": _fraction_string(case.conductivity),
        "contact_impedance": float(case.contact_impedance),
        "contact_impedance_exact": _fraction_string(case.contact_impedance),
        "drive_skip": case.drive_skip,
        "drive_label": case.drive_label,
        "current_patterns": currents.tolist(),
        "coordinate_scale": COORDINATE_SCALE,
        "circle_integer_radius": CIRCLE_INTEGER_RADIUS,
        "electrode_length_exact": _fraction_string(
            Fraction(
                CIRCLE_INTEGER_RADIUS,
                ELECTRODE_TANGENT_DIVISOR * COORDINATE_SCALE,
            )
        ),
        "mat": str(mat_path.resolve()),
        "msh": str(msh_path.resolve()),
    }
    write_json(metadata_path, metadata)
    return {
        **metadata,
        "case_dir": case_dir,
        "mat_path": mat_path,
        "msh_path": msh_path,
        "metadata_path": metadata_path,
    }


def run_pyeidors_case(fixture: dict[str, Any]) -> dict[str, Any]:
    from dolfinx import fem

    case_dir = Path(fixture["case_dir"])
    mesh, _ = build_mesh_from_exchange_mat(Path(fixture["mat_path"]))
    config = BenchmarkConfig(mesh_refinement=0, timing_repeats=TIMING_REPEATS)
    impedance = np.full(
        N_ELECTRODES,
        float(fixture["contact_impedance"]),
        dtype=np.float64,
    )
    model = EITForwardModel(
        n_elec=N_ELECTRODES,
        pattern_config=_pattern_config(config),
        z=impedance,
        mesh=mesh,
        potential_order=1,
        linear_backend="scipy",
    )
    if np.dtype(model.scalar_dtype) != np.dtype(np.float64):
        raise RuntimeError("exact CEM suite requires real float64 PyEIDORS")
    loaded_edges = _extract_tagged_boundary_edges(mesh, list(model.electrode_tags))
    loaded_fingerprint = canonical_mesh_fingerprint(
        np.asarray(mesh.coordinates(), dtype=np.float64)[:, :2],
        np.asarray(mesh.cells(), dtype=np.int64),
        loaded_edges,
    )
    if loaded_fingerprint != fixture["mesh_fingerprint"]:
        raise RuntimeError(f"{fixture['case_id']} PyEIDORS mesh fingerprint mismatch")

    sigma = fem.Function(model.V_sigma)
    sigma.x.array[:] = float(fixture["conductivity"])
    currents = np.asarray(fixture["current_patterns"], dtype=np.float64)
    robin_matrix, coupling, electrode_matrix = _assemble_pyeidors_blocks(model, sigma)
    timing, voltages, parity = benchmark_preassembled_blocks(
        robin_matrix,
        coupling,
        electrode_matrix,
        currents,
        repeats=TIMING_REPEATS,
    )
    report = {
        "solver": "PyEIDORS/DOLFINx",
        "suite_schema": SUITE_SCHEMA,
        "case_id": fixture["case_id"],
        "physical_config": {
            "n_electrodes": N_ELECTRODES,
            "conductivity": fixture["conductivity"],
            "contact_impedance": fixture["contact_impedance"],
            "drive_skip": fixture["drive_skip"],
        },
        "discretization": {
            "vertices": int(mesh.num_vertices()),
            "cells": int(mesh.num_cells()),
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
    write_json(case_dir / "pyeidors_report.json", report)
    return report


def prepare_suite(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for case in CASES:
        fixture = prepare_case_fixture(output_dir, case)
        run_pyeidors_case(fixture)
        fixtures.append(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in fixture.items()
                if key not in {"current_patterns"}
            }
        )
    manifest = {
        "suite_schema": SUITE_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_count": len(CASES),
        "cases": fixtures,
    }
    write_json(output_dir / "suite_manifest.json", manifest)
    return manifest


def _sympy_rational(value: Fraction) -> Rational:
    return Rational(value.numerator, value.denominator)


def _sqrt_fraction_exact(value: Fraction) -> Fraction:
    if value < 0:
        raise ValueError("cannot take exact square root of a negative fraction")
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if (
        numerator * numerator != value.numerator
        or denominator * denominator != value.denominator
    ):
        raise ValueError(f"fraction has no rational square root: {value}")
    return Fraction(numerator, denominator)


def assemble_exact_cem(
    nodes: tuple[tuple[Fraction, Fraction], ...],
    cells: np.ndarray,
    tagged_edges: np.ndarray,
    *,
    conductivity: Fraction,
    contact_impedance: Fraction,
) -> tuple[Matrix, Matrix, Matrix]:
    """Assemble exact P1 CEM blocks over the rational field."""

    node_count = len(nodes)
    a_r = SparseMatrix.zeros(node_count, node_count)
    coupling = SparseMatrix.zeros(node_count, N_ELECTRODES)
    electrode_matrix = SparseMatrix.zeros(N_ELECTRODES, N_ELECTRODES)

    for triangle in np.asarray(cells, dtype=np.int64):
        indices = [int(value) for value in triangle]
        x1, y1 = nodes[indices[0]]
        x2, y2 = nodes[indices[1]]
        x3, y3 = nodes[indices[2]]
        determinant = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = abs(determinant) / 2
        if area == 0:
            raise ValueError("exact CEM mesh contains a zero-area triangle")
        b = (y2 - y3, y3 - y1, y1 - y2)
        c = (x3 - x2, x1 - x3, x2 - x1)
        for local_row, global_row in enumerate(indices):
            for local_column, global_column in enumerate(indices):
                value = (
                    conductivity
                    * (b[local_row] * b[local_column] + c[local_row] * c[local_column])
                    / (4 * area)
                )
                a_r[global_row, global_column] += _sympy_rational(value)

    for vertex_a, vertex_b, label in np.asarray(tagged_edges, dtype=np.int64):
        electrode_label = int(label)
        if electrode_label <= 0:
            continue
        a = int(vertex_a)
        b_index = int(vertex_b)
        dx = nodes[a][0] - nodes[b_index][0]
        dy = nodes[a][1] - nodes[b_index][1]
        length = _sqrt_fraction_exact(dx * dx + dy * dy)
        length_over_z = length / contact_impedance
        diagonal = _sympy_rational(length_over_z / 3)
        off_diagonal = _sympy_rational(length_over_z / 6)
        half = _sympy_rational(length_over_z / 2)
        total = _sympy_rational(length_over_z)
        electrode = electrode_label - 1
        a_r[a, a] += diagonal
        a_r[b_index, b_index] += diagonal
        a_r[a, b_index] += off_diagonal
        a_r[b_index, a] += off_diagonal
        coupling[a, electrode] -= half
        coupling[b_index, electrode] -= half
        electrode_matrix[electrode, electrode] += total
    return Matrix(a_r), Matrix(coupling), Matrix(electrode_matrix)


def _exact_currents_matrix(currents: np.ndarray) -> Matrix:
    array = np.asarray(currents, dtype=np.float64)
    values = []
    for row in array:
        values.append(
            [_sympy_rational(Fraction.from_float(float(value))) for value in row]
        )
    return Matrix(values)


def _rational_zero_sum_basis() -> Matrix:
    basis = SparseMatrix.zeros(N_ELECTRODES, N_ELECTRODES - 1)
    for column in range(N_ELECTRODES - 1):
        basis[column, column] = 1
        basis[N_ELECTRODES - 1, column] = -1
    return Matrix(basis)


def solve_exact_case(case: ExactCase) -> dict[str, Any]:
    """Solve both exact classic and exact Robin systems over QQ."""

    nodes, cells, edges, _, _ = exact_case_mesh(case)
    a_r, coupling, electrode_matrix = assemble_exact_cem(
        nodes,
        cells,
        edges,
        conductivity=case.conductivity,
        contact_impedance=case.contact_impedance,
    )
    currents = _exact_currents_matrix(exact_current_patterns(case.drive_skip))
    node_count = len(nodes)
    full_size = node_count + N_ELECTRODES + 1
    full_matrix = SparseMatrix.zeros(full_size, full_size)
    full_matrix[:node_count, :node_count] = a_r
    full_matrix[:node_count, node_count : node_count + N_ELECTRODES] = coupling
    full_matrix[node_count : node_count + N_ELECTRODES, :node_count] = coupling.T
    full_matrix[
        node_count : node_count + N_ELECTRODES,
        node_count : node_count + N_ELECTRODES,
    ] = electrode_matrix
    for electrode in range(N_ELECTRODES):
        full_matrix[node_count + electrode, full_size - 1] = 1
        full_matrix[full_size - 1, node_count + electrode] = 1
    rhs = SparseMatrix.zeros(full_size, currents.cols)
    rhs[node_count : node_count + N_ELECTRODES, :] = currents

    full_domain_matrix = full_matrix.to_DM().convert_to(QQ)
    rhs_domain_matrix = rhs.to_DM().convert_to(QQ)
    classic_solution = full_domain_matrix.lu_solve(rhs_domain_matrix).to_Matrix()
    classic_residual = Matrix(full_matrix) * classic_solution - Matrix(rhs)
    if any(value != 0 for value in classic_residual):
        raise RuntimeError(f"{case.case_id} exact classic residual is not zero")
    classic_voltage = classic_solution[
        node_count : node_count + N_ELECTRODES,
        :,
    ]

    a_r_domain_matrix = a_r.to_DM().convert_to(QQ)
    coupling_domain_matrix = coupling.to_DM().convert_to(QQ)
    response = a_r_domain_matrix.lu_solve(coupling_domain_matrix).to_Matrix()
    schur = electrode_matrix - coupling.T * response
    basis = _rational_zero_sum_basis()
    reduced_map = basis.T * schur * basis
    reduced_rhs = basis.T * currents
    reduced_domain_matrix = reduced_map.to_DM().convert_to(QQ)
    reduced_rhs_domain_matrix = reduced_rhs.to_DM().convert_to(QQ)
    coefficients = reduced_domain_matrix.lu_solve(reduced_rhs_domain_matrix).to_Matrix()
    robin_voltage = basis * coefficients
    robin_residual = reduced_map * coefficients - reduced_rhs
    if any(value != 0 for value in robin_residual):
        raise RuntimeError(f"{case.case_id} exact Robin residual is not zero")
    if classic_voltage != robin_voltage:
        raise RuntimeError(f"{case.case_id} exact classic and Robin voltages differ")
    if any(
        sum(classic_voltage[row, column] for row in range(N_ELECTRODES)) != 0
        for column in range(classic_voltage.cols)
    ):
        raise RuntimeError(f"{case.case_id} exact voltage gauge is not zero")

    exact_strings = [
        [str(classic_voltage[row, column]) for column in range(classic_voltage.cols)]
        for row in range(classic_voltage.rows)
    ]
    truth_digest = hashlib.sha256(
        json.dumps(exact_strings, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "case": case,
        "voltage": classic_voltage,
        "reduced_map": reduced_map,
        "reduced_rhs": reduced_rhs,
        "basis": basis,
        "exact_classic_residual_zero": True,
        "exact_robin_residual_zero": True,
        "exact_classic_robin_identical": True,
        "exact_linear_solver": "DomainMatrix.lu_solve",
        "exact_domain": "QQ",
        "truth_sha256": truth_digest,
        "truth_fraction_strings": exact_strings,
    }


def _mp_from_sympy(value: Any):
    numerator, denominator = value.as_numer_denom()
    return mp.mpf(int(numerator)) / mp.mpf(int(denominator))


def _mp_matrix_from_sympy(matrix: Matrix):
    result = mp.matrix(matrix.rows, matrix.cols)
    for row in range(matrix.rows):
        for column in range(matrix.cols):
            result[row, column] = _mp_from_sympy(matrix[row, column])
    return result


def _mp_matrix_from_float64(values: np.ndarray):
    array = np.asarray(values, dtype=np.float64)
    result = mp.matrix(array.shape[0], array.shape[1])
    for row in range(array.shape[0]):
        for column in range(array.shape[1]):
            fraction = Fraction.from_float(float(array[row, column]))
            result[row, column] = mp.mpf(fraction.numerator) / fraction.denominator
    return result


def _mp_frobenius(matrix) -> Any:
    return mp.sqrt(mp.fsum(abs(value) ** 2 for value in matrix))


def _mp_max_abs(matrix) -> Any:
    return max((abs(value) for value in matrix), default=mp.mpf("0"))


def exact_accuracy_metrics(
    candidate: np.ndarray,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare one float64 voltage matrix with the exact rational solution."""

    with mp.workdps(100):
        candidate_mp = _mp_matrix_from_float64(candidate)
        truth_mp = _mp_matrix_from_sympy(reference["voltage"])
        delta = candidate_mp - truth_mp
        relative_error = _mp_frobenius(delta) / _mp_frobenius(truth_mp)

        gauge = mp.matrix(1, candidate_mp.cols)
        centered = mp.matrix(candidate_mp.rows, candidate_mp.cols)
        for column in range(candidate_mp.cols):
            mean = (
                mp.fsum(candidate_mp[row, column] for row in range(candidate_mp.rows))
                / candidate_mp.rows
            )
            gauge[0, column] = mean * candidate_mp.rows
            for row in range(candidate_mp.rows):
                centered[row, column] = candidate_mp[row, column] - mean
        coefficients = mp.matrix(N_ELECTRODES - 1, candidate_mp.cols)
        for row in range(N_ELECTRODES - 1):
            for column in range(candidate_mp.cols):
                coefficients[row, column] = centered[row, column]
        reduced_map = _mp_matrix_from_sympy(reference["reduced_map"])
        reduced_rhs = _mp_matrix_from_sympy(reference["reduced_rhs"])
        residual = reduced_map * coefficients - reduced_rhs
        backward = _mp_frobenius(residual) / (
            _mp_frobenius(reduced_map) * _mp_frobenius(coefficients)
            + _mp_frobenius(reduced_rhs)
        )
        gauge_residual = _mp_frobenius(gauge) / _mp_frobenius(candidate_mp)
        per_rhs = []
        for column in range(candidate_mp.cols):
            delta_norm = mp.sqrt(
                mp.fsum(abs(delta[row, column]) ** 2 for row in range(delta.rows))
            )
            truth_norm = mp.sqrt(
                mp.fsum(abs(truth_mp[row, column]) ** 2 for row in range(truth_mp.rows))
            )
            per_rhs.append(float(delta_norm / truth_norm))
        reduced_float = np.asarray(reference["reduced_map"], dtype=np.float64)
        return {
            "truth_relative_l2": float(relative_error),
            "truth_relative_l2_decimal": mp.nstr(relative_error, 40),
            "truth_max_abs": float(_mp_max_abs(delta)),
            "truth_max_abs_decimal": mp.nstr(_mp_max_abs(delta), 40),
            "exact_reduced_scaled_backward_residual": float(backward),
            "exact_reduced_scaled_backward_residual_decimal": mp.nstr(backward, 40),
            "voltage_gauge_relative_residual": float(gauge_residual),
            "per_rhs_truth_relative_l2": per_rhs,
            "reduced_condition_number_2_estimate": float(
                np.linalg.cond(reduced_float, p=2)
            ),
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_solver_report(
    report: dict[str, Any],
    fixture: dict[str, Any],
) -> None:
    solver = str(report.get("solver", "unknown"))
    if report.get("case_id") != fixture["case_id"]:
        raise ValueError(f"{solver} case id mismatch")
    discretization = report.get("discretization", {})
    if discretization.get("mesh_fingerprint") != fixture["mesh_fingerprint"]:
        raise ValueError(f"{solver} mesh fingerprint mismatch")
    if not bool(discretization.get("mesh_import_verified", False)):
        raise ValueError(f"{solver} did not verify mesh import")
    if int(discretization.get("potential_order", -1)) != 1:
        raise ValueError(f"{solver} exact suite must use P1")
    if report.get("linear_solver", {}).get("scalar_dtype") != "float64":
        raise ValueError(f"{solver} exact suite must use float64")
    physical = report.get("physical_config", {})
    for key in ("conductivity", "contact_impedance", "drive_skip"):
        if float(physical.get(key, math.nan)) != float(fixture[key]):
            raise ValueError(f"{solver} physical setting mismatch for {key}")
    timing = report.get("timing", {})
    if timing.get("schema") != TIMING_SCHEMA:
        raise ValueError(f"{solver} timing schema mismatch")
    if timing.get("scope") != "preassembled_A_R_C_D_blocks":
        raise ValueError(f"{solver} timing scope mismatch")
    if int(timing.get("repeats", -1)) != TIMING_REPEATS:
        raise ValueError(f"{solver} timing repeats mismatch")
    if int(timing.get("operations_per_sample", -1)) != TIMING_OPERATIONS_PER_SAMPLE:
        raise ValueError(f"{solver} timing operation batch mismatch")
    if int(timing.get("rhs_count", -1)) != N_ELECTRODES:
        raise ValueError(f"{solver} timing RHS count mismatch")
    for flag in (
        "alternating_order",
        "untimed_runtime_priming",
        "paired_cold_decomposition",
    ):
        if not bool(timing.get(flag, False)):
            raise ValueError(f"{solver} timing fairness flag {flag} is false")
    if bool(timing.get("cross_formulation_cache_reuse", True)):
        raise ValueError(f"{solver} reused state across formulations")
    expected_shape = (N_ELECTRODES, N_ELECTRODES)
    for formulation in FORMULATIONS:
        phase = timing.get(formulation, {})
        cold = float(phase.get("cold_seconds", {}).get("median", math.nan))
        warm = float(phase.get("warm_reuse_seconds", {}).get("median", math.nan))
        if not (math.isfinite(cold) and math.isfinite(warm) and cold > warm > 0.0):
            raise ValueError(f"{solver} {formulation} cold/warm timing is invalid")
        values = np.asarray(
            report.get("raw_electrode_voltages", {}).get(formulation),
            dtype=np.float64,
        )
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{solver} {formulation} raw voltage mismatch")


def aggregate_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case metrics without hiding ordering reversals."""

    observed_case_ids = {str(item["case_id"]) for item in metrics}
    observed_cases = [case for case in CASES if case.case_id in observed_case_ids]
    solvers = sorted({str(item["solver"]) for item in metrics})
    rankings: dict[str, list[dict[str, Any]]] = {}
    strict_orders: dict[str, list[tuple[str, ...]]] = {
        formulation: [] for formulation in FORMULATIONS
    }
    win_counts = {
        formulation: {solver: 0 for solver in solvers} for formulation in FORMULATIONS
    }
    for case in observed_cases:
        for formulation in FORMULATIONS:
            selected = sorted(
                (
                    item
                    for item in metrics
                    if item["case_id"] == case.case_id
                    and item["formulation"] == formulation
                ),
                key=lambda item: item["truth_relative_l2"],
            )
            key = f"{case.case_id}:{formulation}"
            rankings[key] = [
                {
                    "rank": rank,
                    "solver": item["solver"],
                    "truth_relative_l2": item["truth_relative_l2"],
                    "ratio_to_best": item["truth_relative_l2"]
                    / selected[0]["truth_relative_l2"],
                }
                for rank, item in enumerate(selected, start=1)
            ]
            order = tuple(str(item["solver"]) for item in selected)
            strict_orders[formulation].append(order)
            win_counts[formulation][order[0]] += 1

    solver_summary: dict[str, dict[str, Any]] = {}
    for solver in solvers:
        solver_summary[solver] = {}
        for formulation in FORMULATIONS:
            errors = np.asarray(
                [
                    item["truth_relative_l2"]
                    for item in metrics
                    if item["solver"] == solver and item["formulation"] == formulation
                ],
                dtype=np.float64,
            )
            solver_summary[solver][formulation] = {
                "geometric_mean_truth_relative_l2": float(
                    np.exp(np.mean(np.log(errors)))
                ),
                "median_truth_relative_l2": float(np.median(errors)),
                "worst_truth_relative_l2": float(np.max(errors)),
                "win_count": int(win_counts[formulation][solver]),
            }

    universal_ordering = {}
    for formulation, orders in strict_orders.items():
        same = bool(orders) and all(order == orders[0] for order in orders[1:])
        universal_ordering[formulation] = {
            "supported": same,
            "ordering": list(orders[0]) if same else None,
            "observed_orders": [list(order) for order in orders],
        }

    case_lookup = {case.case_id: case for case in CASES}
    refinement_case_ids = [
        case_id for case_id in REFINEMENT_CASE_IDS if case_id in observed_case_ids
    ]
    refinement_mesh_sizes = {}
    for case_id in refinement_case_ids:
        exact_nodes, exact_cells, *_ = exact_case_mesh(case_lookup[case_id])
        refinement_mesh_sizes[case_id] = (len(exact_nodes), int(exact_cells.shape[0]))
    refinement_summary: dict[str, Any] = {}
    for formulation in FORMULATIONS:
        level_orders = [
            tuple(item["solver"] for item in rankings[f"{case_id}:{formulation}"])
            for case_id in refinement_case_ids
        ]
        level_win_counts = {solver: 0 for solver in solvers}
        for order in level_orders:
            level_win_counts[str(order[0])] += 1
        per_solver: dict[str, Any] = {}
        for solver in solvers:
            selected = [
                next(
                    item
                    for item in metrics
                    if item["case_id"] == case_id
                    and item["solver"] == solver
                    and item["formulation"] == formulation
                )
                for case_id in refinement_case_ids
            ]
            errors = np.asarray(
                [item["truth_relative_l2"] for item in selected],
                dtype=np.float64,
            )
            per_solver[solver] = {
                "nodes": [
                    int(
                        item.get(
                            "nodes",
                            refinement_mesh_sizes[str(item["case_id"])][0],
                        )
                    )
                    for item in selected
                ],
                "cells": [
                    int(
                        item.get(
                            "cells",
                            refinement_mesh_sizes[str(item["case_id"])][1],
                        )
                    )
                    for item in selected
                ],
                "truth_relative_l2": [float(value) for value in errors],
                "geometric_mean_truth_relative_l2": float(
                    np.exp(np.mean(np.log(errors)))
                ),
                "median_truth_relative_l2": float(np.median(errors)),
                "worst_truth_relative_l2": float(np.max(errors)),
            }
        refinement_summary[formulation] = {
            "case_ids": list(refinement_case_ids),
            "level_ids": [
                case_lookup[case_id].refinement_level_id
                for case_id in refinement_case_ids
            ],
            "edge_subdivisions": [
                case_lookup[case_id].edge_subdivisions
                for case_id in refinement_case_ids
            ],
            "radial_layers": [
                case_lookup[case_id].radial_layers for case_id in refinement_case_ids
            ],
            "win_counts": level_win_counts,
            "observed_orders": [list(order) for order in level_orders],
            "universal_ordering_supported": bool(level_orders)
            and all(order == level_orders[0] for order in level_orders[1:]),
            "per_solver": per_solver,
        }
    return {
        "rankings": rankings,
        "solver_summary": solver_summary,
        "universal_ordering": universal_ordering,
        "refinement_summary": refinement_summary,
    }


def timing_records_from_report(
    report: dict[str, Any],
    *,
    case_id: str,
) -> list[dict[str, Any]]:
    timing = report["timing"]
    records = []
    for formulation in FORMULATIONS:
        formulation_timing = timing[formulation]
        cold_median = float(formulation_timing["cold_seconds"]["median"])
        warm_median = float(formulation_timing["warm_reuse_seconds"]["median"])
        if not cold_median > warm_median > 0.0:
            raise ValueError(
                f"{report['solver']} {formulation} requires cold > warm reuse"
            )
        records.append(
            {
                "case_id": case_id,
                "solver": report["solver"],
                "formulation": formulation,
                "cold_median_seconds": cold_median,
                "cold_iqr_seconds": float(formulation_timing["cold_seconds"]["iqr"]),
                "setup_median_seconds": float(
                    formulation_timing["setup_seconds"]["median"]
                ),
                "setup_iqr_seconds": float(formulation_timing["setup_seconds"]["iqr"]),
                "cold_solve_median_seconds": float(
                    formulation_timing["cold_solve_seconds"]["median"]
                ),
                "cold_solve_iqr_seconds": float(
                    formulation_timing["cold_solve_seconds"]["iqr"]
                ),
                "warm_reuse_median_seconds": warm_median,
                "warm_reuse_iqr_seconds": float(
                    formulation_timing["warm_reuse_seconds"]["iqr"]
                ),
                "cold_over_warm_reuse_speedup": float(cold_median / warm_median),
                "assembly_seconds": (
                    float(timing["assembly_seconds"])
                    if "assembly_seconds" in timing
                    else None
                ),
                "mesh_import_seconds": (
                    float(timing["mesh_import_seconds"])
                    if "mesh_import_seconds" in timing
                    else None
                ),
            }
        )
    return records


def aggregate_timing_metrics(
    timing_records: list[dict[str, Any]],
) -> dict[str, Any]:
    ratio_fields = {
        "cold": "cold_median_seconds",
        "setup": "setup_median_seconds",
        "warm_reuse": "warm_reuse_median_seconds",
    }
    per_case_ratios = []
    for case in CASES:
        for solver in sorted({str(item["solver"]) for item in timing_records}):
            selected = {
                item["formulation"]: item
                for item in timing_records
                if item["case_id"] == case.case_id and item["solver"] == solver
            }
            ratio = {"case_id": case.case_id, "solver": solver}
            for phase, field in ratio_fields.items():
                ratio[f"robin_over_classic_{phase}_ratio"] = float(
                    selected["robin_transconductance"][field]
                    / selected["classic"][field]
                )
            per_case_ratios.append(ratio)

    summary = {}
    absolute_summary = {}
    for solver in sorted({str(item["solver"]) for item in timing_records}):
        summary[solver] = {}
        absolute_summary[solver] = {}
        solver_ratios = [item for item in per_case_ratios if item["solver"] == solver]
        for phase in ratio_fields:
            field = f"robin_over_classic_{phase}_ratio"
            values = np.asarray(
                [item[field] for item in solver_ratios],
                dtype=np.float64,
            )
            summary[solver][phase] = {
                "geometric_mean_robin_over_classic_ratio": float(
                    np.exp(np.mean(np.log(values)))
                ),
                "median_robin_over_classic_ratio": float(np.median(values)),
                "minimum_robin_over_classic_ratio": float(np.min(values)),
                "maximum_robin_over_classic_ratio": float(np.max(values)),
                "robin_faster_case_count": int(np.count_nonzero(values < 1.0)),
                "case_count": int(values.size),
            }
        for formulation in FORMULATIONS:
            selected = [
                item
                for item in timing_records
                if item["solver"] == solver and item["formulation"] == formulation
            ]
            cold = np.asarray(
                [item["cold_median_seconds"] for item in selected], dtype=np.float64
            )
            setup = np.asarray(
                [item["setup_median_seconds"] for item in selected], dtype=np.float64
            )
            warm = np.asarray(
                [item["warm_reuse_median_seconds"] for item in selected],
                dtype=np.float64,
            )
            speedup = cold / warm
            if np.any(speedup <= 1.0):
                raise ValueError(f"{solver} {formulation} has cold <= warm reuse")
            absolute_summary[solver][formulation] = {
                "geometric_mean_cold_seconds": float(np.exp(np.mean(np.log(cold)))),
                "geometric_mean_setup_seconds": float(np.exp(np.mean(np.log(setup)))),
                "geometric_mean_warm_reuse_seconds": float(
                    np.exp(np.mean(np.log(warm)))
                ),
                "geometric_mean_cold_over_warm_reuse_speedup": float(
                    np.exp(np.mean(np.log(speedup)))
                ),
                "minimum_cold_over_warm_reuse_speedup": float(np.min(speedup)),
                "case_count": int(speedup.size),
            }
    return {
        "per_case_robin_over_classic_ratios": per_case_ratios,
        "solver_phase_summary": summary,
        "solver_formulation_absolute_summary": absolute_summary,
    }


def _write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in metrics:
            writer.writerow({field: item[field] for field in CSV_FIELDS})


def _write_timing_csv(path: Path, timing_records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMING_CSV_FIELDS)
        writer.writeheader()
        for item in timing_records:
            writer.writerow({field: item[field] for field in TIMING_CSV_FIELDS})


def _plot_suite(metrics: list[dict[str, Any]], path: Path) -> None:
    configure_fonts()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
        }
    )
    solvers = sorted({str(item["solver"]) for item in metrics})
    colors = dict(zip(solvers, ("#1f5a94", "#c56a1a", "#687a3c"), strict=True))
    markers = dict(zip(solvers, ("o", "s", "^"), strict=True))
    line_styles = dict(zip(solvers, ("-", "--", ":"), strict=True))
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.2), constrained_layout=True)
    case_lookup = {case.case_id: case for case in CASES}
    refinement_cases = [case_lookup[case_id] for case_id in REFINEMENT_CASE_IDS]
    for column, formulation in enumerate(FORMULATIONS):
        for solver in solvers:
            selected = [
                next(
                    item
                    for item in metrics
                    if item["case_id"] == case.case_id
                    and item["solver"] == solver
                    and item["formulation"] == formulation
                )
                for case in refinement_cases
            ]
            x = np.asarray([item["nodes"] for item in selected], dtype=np.float64)
            axes[0, column].plot(
                x,
                [item["truth_relative_l2"] for item in selected],
                marker=markers[solver],
                linestyle=line_styles[solver],
                label=solver,
                color=colors[solver],
                markerfacecolor="white",
                markeredgewidth=1.3,
                linewidth=1.6,
                markersize=6.0,
            )
            axes[1, column].plot(
                x,
                [item["exact_reduced_scaled_backward_residual"] for item in selected],
                marker=markers[solver],
                linestyle=line_styles[solver],
                label=solver,
                color=colors[solver],
                markerfacecolor="white",
                markeredgewidth=1.3,
                linewidth=1.6,
                markersize=6.0,
            )
        display_name = "Classic CEM" if formulation == "classic" else "Robin CEM"
        axes[0, column].set_title(f"{display_name}: exact forward error")
        axes[1, column].set_title(f"{display_name}: exact scaled residual")
        for row in range(2):
            axis = axes[row, column]
            axis.set_yscale("log")
            axis.set_xscale("log", base=2)
            axis.set_xticks(
                x,
                [
                    f"{case.refinement_level_id}\n{int(nodes)}"
                    for case, nodes in zip(refinement_cases, x, strict=True)
                ],
            )
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
            for label in axis.get_xticklabels() + axis.get_yticklabels():
                label.set_fontname("Times New Roman")
    axes[0, 0].set_ylabel("Relative L2 vs exact QQ solution")
    axes[1, 0].set_ylabel("Exact-system scaled residual")
    figure.suptitle(
        "Exact rational P1 CEM accuracy across nested mesh refinement",
        fontsize=15,
    )
    figure.supxlabel(
        "Refinement level and node count; fixed rational polygon; 16 RHS per level",
        fontsize=9,
    )
    figure.savefig(path, dpi=220)
    plt.close(figure)


def compare_suite(output_dir: Path) -> dict[str, Any]:
    manifest = _load_json(output_dir / "suite_manifest.json")
    fixtures = {item["case_id"]: item for item in manifest["cases"]}
    metrics: list[dict[str, Any]] = []
    timing_records: list[dict[str, Any]] = []
    truth_records: dict[str, Any] = {}
    for case in CASES:
        fixture = fixtures[case.case_id]
        reference = solve_exact_case(case)
        case_dir = Path(fixture["case_dir"])
        report_paths = (
            case_dir / "pyeidors_report.json",
            case_dir / "ngsolve_report.json",
            case_dir / "eidors_report.json",
        )
        reports = [_load_json(path) for path in report_paths]
        if len({report.get("solver") for report in reports}) != 3:
            raise ValueError(f"{case.case_id} requires three distinct solvers")
        truth_records[case.case_id] = {
            "truth_sha256": reference["truth_sha256"],
            "exact_classic_residual_zero": True,
            "exact_robin_residual_zero": True,
            "exact_classic_robin_identical": True,
            "exact_linear_solver": reference["exact_linear_solver"],
            "exact_domain": reference["exact_domain"],
            "electrode_voltage_fractions": reference["truth_fraction_strings"],
        }
        for report in reports:
            _validate_solver_report(report, fixture)
            timing_records.extend(
                timing_records_from_report(report, case_id=case.case_id)
            )
            raw = report["raw_electrode_voltages"]
            classic = np.asarray(raw["classic"], dtype=np.float64)
            robin = np.asarray(raw["robin_transconductance"], dtype=np.float64)
            internal_delta = float(
                np.linalg.norm(robin - classic) / np.linalg.norm(classic)
            )
            for formulation in FORMULATIONS:
                candidate = np.asarray(raw[formulation], dtype=np.float64)
                metrics.append(
                    {
                        "case_id": case.case_id,
                        "case_label": case.label,
                        "refinement_level_id": case.refinement_level_id,
                        "edge_subdivisions": case.edge_subdivisions,
                        "radial_layers": case.radial_layers,
                        "nodes": int(fixture["nodes"]),
                        "cells": int(fixture["cells"]),
                        "solver": report["solver"],
                        "formulation": formulation,
                        "classic_robin_relative_l2": internal_delta,
                        **exact_accuracy_metrics(candidate, reference),
                    }
                )

    aggregate = aggregate_metrics(metrics)
    timing_aggregate = aggregate_timing_metrics(timing_records)
    csv_path = output_dir / "cem_exact_accuracy_metrics.csv"
    timing_csv_path = output_dir / "cem_exact_timing_metrics.csv"
    plot_path = output_dir / "cem_exact_accuracy.png"
    _write_metrics_csv(csv_path, metrics)
    _write_timing_csv(timing_csv_path, timing_records)
    _plot_suite(metrics, plot_path)
    report = {
        "suite_schema": SUITE_SCHEMA,
        "metric_schema": METRIC_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "exact solution of each finite-dimensional rational P1 CEM system; "
            "not the analytic solution of the continuum PDE"
        ),
        "truth_method": {
            "domain": "QQ exact rational arithmetic",
            "solver": "SymPy DomainMatrix QQ multi-RHS lu_solve",
            "uses_any_fem_solver_matrix": False,
            "ranking_requires_zero_exact_residual": True,
            "ranking_requires_exact_classic_robin_identity": True,
        },
        "cases": [asdict(case) for case in CASES],
        "refinement_case_ids": list(REFINEMENT_CASE_IDS),
        "truth": truth_records,
        "metrics": metrics,
        "timing_methodology": {
            "schema": TIMING_SCHEMA,
            "repeats": TIMING_REPEATS,
            "operations_per_sample": TIMING_OPERATIONS_PER_SAMPLE,
            "scope": "preassembled_A_R_C_D_blocks",
            "rhs_per_sample": N_ELECTRODES,
            "alternating_order": True,
            "untimed_runtime_priming": True,
            "cross_formulation_cache_reuse": False,
            "paired_cold_decomposition": True,
            "cold_definition": "fresh state setup plus first 16-RHS solve",
            "setup_definition": "state construction component paired within each cold operation; not a warm phase",
            "warm_reuse_definition": "16-RHS solve using one retained per-formulation state",
            "ratio_definition": "Robin seconds divided by classic seconds; below 1 means Robin is faster",
        },
        "timing_records": timing_records,
        "timing_summary": timing_aggregate,
        **aggregate,
        "artifacts": {
            "csv": csv_path.name,
            "timing_csv": timing_csv_path.name,
            "plot": plot_path.name,
        },
    }
    json.dumps(report, allow_nan=False)
    write_json(output_dir / "cem_exact_accuracy.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "compare"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--output-dir",
            type=Path,
            default=ROOT / "output" / "cem_exact_accuracy",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    if args.command == "prepare":
        manifest = prepare_suite(output_dir)
        print(f"Prepared {manifest['case_count']} exact CEM cases: {output_dir}")
        return 0
    report = compare_suite(output_dir)
    for formulation, conclusion in report["universal_ordering"].items():
        print(
            f"{formulation}: universal_ordering={conclusion['supported']} "
            f"order={conclusion['ordering']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
