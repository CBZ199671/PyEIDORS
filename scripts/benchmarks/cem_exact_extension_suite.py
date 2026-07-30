#!/usr/bin/env python3
"""Run the preregistered rational CEM extension against exact QQ truth."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpmath import mp
import numpy as np
from scipy.io import savemat
from sympy import Matrix, QQ, SparseMatrix

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

from scripts.benchmarks.cem_exact_reference_suite import (
    FORMULATIONS,
    _mp_frobenius,
    _mp_matrix_from_float64,
    _mp_matrix_from_sympy,
    _sqrt_fraction_exact,
    _sympy_rational,
    exact_boundary_nodes,
    float_nodes,
)
from scripts.benchmarks.cem_fair_common import (
    DEFAULT_OPERATIONS_PER_SAMPLE,
    MESH_FINGERPRINT_SCHEMA,
    TIMING_SCHEMA,
    benchmark_preassembled_blocks,
    canonical_mesh_fingerprint,
)
from scripts.benchmarks.compare_cem_formulations import (
    BenchmarkConfig,
    _assemble_pyeidors_blocks,
    _extract_tagged_boundary_edges,
    _pattern_config,
    configure_fonts,
    write_json,
)


SUITE_SCHEMA = "cem-exact-rational-extension-v1"
GEOMETRY_SCHEMA = "cem-rational-fixed-32gon-extension-v1"
METRIC_SCHEMA = "cem-exact-extension-metrics-v1"
TIMING_REPEATS = 11
TIMING_OPERATIONS_PER_SAMPLE = DEFAULT_OPERATIONS_PER_SAMPLE
BOUNDARY_COUNT = 32
ATTRIBUTION_CASE_IDS = ("X05", "X13", "X33", "X21")
FLINT_SCHEMA = "cem-exact-flint-basis-v1"
FLINT_VERSION = "0.6.0"
DEFAULT_QQ_CACHE_DIR = ROOT / "output" / "cem_exact_extension" / "qq_cache"


@dataclass(frozen=True)
class ExtensionCase:
    case_id: str
    family: str
    label: str
    refinement_level_id: str
    edge_subdivisions: int
    radial_layers: int
    n_electrodes: int
    conductivity_pattern: str
    conductivity_low_numerator: int
    conductivity_low_denominator: int
    conductivity_high_numerator: int
    conductivity_high_denominator: int
    impedance_numerator: int
    impedance_denominator: int
    drive_skip: int
    drive_label: str

    @property
    def conductivity_low(self) -> Fraction:
        return Fraction(
            self.conductivity_low_numerator,
            self.conductivity_low_denominator,
        )

    @property
    def conductivity_high(self) -> Fraction:
        return Fraction(
            self.conductivity_high_numerator,
            self.conductivity_high_denominator,
        )

    @property
    def contact_impedance(self) -> Fraction:
        return Fraction(self.impedance_numerator, self.impedance_denominator)


def _extension_cases() -> tuple[ExtensionCase, ...]:
    rows: list[dict[str, Any]] = []

    def add(
        *,
        family: str,
        mesh: tuple[str, int, int],
        n_electrodes: int,
        pattern: str,
        sigma_low: Fraction,
        sigma_high: Fraction,
        impedance: Fraction,
        drive_skip: int,
    ) -> None:
        level, edge_subdivisions, radial_layers = mesh
        drive_label = "adjacent" if drive_skip == 1 else f"skip-{drive_skip}"
        rows.append(
            {
                "family": family,
                "label": (
                    f"{family}_{level.lower()}_{pattern}_sigma_"
                    f"{sigma_low}_to_{sigma_high}_z_{impedance}_{drive_label}"
                ).replace("/", "_"),
                "refinement_level_id": level,
                "edge_subdivisions": edge_subdivisions,
                "radial_layers": radial_layers,
                "n_electrodes": n_electrodes,
                "conductivity_pattern": pattern,
                "conductivity_low_numerator": sigma_low.numerator,
                "conductivity_low_denominator": sigma_low.denominator,
                "conductivity_high_numerator": sigma_high.numerator,
                "conductivity_high_denominator": sigma_high.denominator,
                "impedance_numerator": impedance.numerator,
                "impedance_denominator": impedance.denominator,
                "drive_skip": drive_skip,
                "drive_label": drive_label,
            }
        )

    q0 = ("Q0", 1, 1)
    q2 = ("Q2", 2, 2)
    q4 = ("Q4", 4, 4)
    for mesh in (q0, q2):
        for sigma, impedance in (
            (Fraction(1, 8), Fraction(1)),
            (Fraction(4), Fraction(1)),
            (Fraction(1, 4), Fraction(1, 32)),
            (Fraction(1, 4), Fraction(32)),
        ):
            for drive_skip in (1, 4):
                add(
                    family="range",
                    mesh=mesh,
                    n_electrodes=16,
                    pattern="uniform",
                    sigma_low=sigma,
                    sigma_high=sigma,
                    impedance=impedance,
                    drive_skip=drive_skip,
                )
    for mesh in (q0, q2):
        for impedance in (Fraction(1, 8), Fraction(1)):
            for drive_skip in (1, 4):
                add(
                    family="heterogeneous",
                    mesh=mesh,
                    n_electrodes=16,
                    pattern="left_right",
                    sigma_low=Fraction(1, 4),
                    sigma_high=Fraction(1),
                    impedance=impedance,
                    drive_skip=drive_skip,
                )
    for mesh in (q0, q2):
        for impedance in (Fraction(1, 8), Fraction(1)):
            for drive_skip in (1, 2):
                add(
                    family="electrode_count",
                    mesh=mesh,
                    n_electrodes=8,
                    pattern="uniform",
                    sigma_low=Fraction(1, 4),
                    sigma_high=Fraction(1, 4),
                    impedance=impedance,
                    drive_skip=drive_skip,
                )
    for impedance in (Fraction(1, 32), Fraction(1, 8), Fraction(1)):
        for drive_skip in (1, 4):
            add(
                family="large_q4",
                mesh=q4,
                n_electrodes=16,
                pattern="uniform",
                sigma_low=Fraction(1, 4),
                sigma_high=Fraction(1, 4),
                impedance=impedance,
                drive_skip=drive_skip,
            )
    return tuple(
        ExtensionCase(case_id=f"X{index:02d}", **row)
        for index, row in enumerate(rows, start=1)
    )


EXTENSION_CASES = _extension_cases()


def _positive_power_of_two(value: int, *, name: str) -> int:
    result = int(value)
    if result <= 0 or result & (result - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return result


def extension_refined_circular_mesh(
    *,
    edge_subdivisions: int,
    radial_layers: int,
    n_electrodes: int,
) -> tuple[
    tuple[tuple[Fraction, Fraction], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build the fixed rational 32-gon for 8 or 16 equal-coverage electrodes."""

    subdivisions = _positive_power_of_two(
        edge_subdivisions,
        name="edge_subdivisions",
    )
    layers = _positive_power_of_two(radial_layers, name="radial_layers")
    electrode_count = int(n_electrodes)
    if electrode_count not in (8, 16):
        raise ValueError("extension mesh supports exactly 8 or 16 electrodes")
    block_width = BOUNDARY_COUNT // electrode_count
    active_width = block_width // 2
    base_boundary = exact_boundary_nodes()
    boundary: list[tuple[Fraction, Fraction]] = []
    boundary_labels: list[int] = []
    for base_index, start in enumerate(base_boundary):
        stop = base_boundary[(base_index + 1) % BOUNDARY_COUNT]
        within_block = base_index % block_width
        label = base_index // block_width + 1 if within_block < active_width else 0
        for sub_index in range(subdivisions):
            fraction = Fraction(sub_index, subdivisions)
            boundary.append(
                (
                    (1 - fraction) * start[0] + fraction * stop[0],
                    (1 - fraction) * start[1] + fraction * stop[1],
                )
            )
            boundary_labels.append(label)

    refined_boundary_count = len(boundary)
    nodes: list[tuple[Fraction, Fraction]] = [(Fraction(0), Fraction(0))]
    for layer in range(1, layers + 1):
        radius = Fraction(layer, layers)
        nodes.extend((radius * x, radius * y) for x, y in boundary)

    cells: list[tuple[int, int, int]] = []
    for index in range(refined_boundary_count):
        cells.append((0, 1 + index, 1 + (index + 1) % refined_boundary_count))
    for inner_ring in range(layers - 1):
        inner_offset = 1 + inner_ring * refined_boundary_count
        outer_offset = inner_offset + refined_boundary_count
        for index in range(refined_boundary_count):
            next_index = (index + 1) % refined_boundary_count
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
    active_edge_count = active_width * subdivisions
    electrode_nodes = np.asarray(
        [
            [
                outer_offset + electrode * block_width * subdivisions + sub_index
                for sub_index in range(active_edge_count + 1)
            ]
            for electrode in range(electrode_count)
        ],
        dtype=np.int64,
    )
    electrode_counts = np.full(
        electrode_count,
        active_edge_count + 1,
        dtype=np.int64,
    )
    cell_array = np.asarray(cells, dtype=np.int64)
    for triangle in cell_array:
        a, b, c = (nodes[int(index)] for index in triangle)
        determinant = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if determinant <= 0:
            raise RuntimeError("extension mesh contains a non-positive triangle")
    return tuple(nodes), cell_array, edges, electrode_nodes, electrode_counts


def extension_case_mesh(case: ExtensionCase):
    return extension_refined_circular_mesh(
        edge_subdivisions=case.edge_subdivisions,
        radial_layers=case.radial_layers,
        n_electrodes=case.n_electrodes,
    )


def extension_case_cell_conductivities(
    case: ExtensionCase,
    nodes: tuple[tuple[Fraction, Fraction], ...],
    cells: np.ndarray,
) -> tuple[Fraction, ...]:
    """Return the canonical source-order rational DG0 conductivity field."""

    if case.conductivity_pattern == "uniform":
        return (case.conductivity_low,) * int(cells.shape[0])
    if case.conductivity_pattern != "left_right":
        raise ValueError(
            f"unsupported conductivity pattern: {case.conductivity_pattern}"
        )
    values = []
    for triangle in np.asarray(cells, dtype=np.int64):
        centroid_x = sum(nodes[int(index)][0] for index in triangle) / 3
        values.append(
            case.conductivity_low if centroid_x < 0 else case.conductivity_high
        )
    return tuple(values)


def conductivity_digest(values: tuple[Fraction, ...] | np.ndarray) -> str:
    exact_strings = []
    for value in values:
        fraction = (
            value if isinstance(value, Fraction) else Fraction.from_float(float(value))
        )
        exact_strings.append(f"{fraction.numerator}/{fraction.denominator}")
    return hashlib.sha256(
        json.dumps(exact_strings, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def extension_current_patterns(n_electrodes: int, drive_skip: int) -> np.ndarray:
    count = int(n_electrodes)
    skip = int(drive_skip)
    if count < 2 or skip <= 0 or skip >= count:
        raise ValueError("drive_skip must lie in [1, n_electrodes)")
    currents = np.zeros((count, count), dtype=np.float64)
    for column in range(count):
        currents[column, column] = 1.0
        currents[(column + skip) % count, column] = -1.0
    return currents


def extension_case_system_key(case: ExtensionCase) -> tuple[Any, ...]:
    return (
        case.edge_subdivisions,
        case.radial_layers,
        case.n_electrodes,
        case.conductivity_pattern,
        case.conductivity_low_numerator,
        case.conductivity_low_denominator,
        case.conductivity_high_numerator,
        case.conductivity_high_denominator,
        case.impedance_numerator,
        case.impedance_denominator,
    )


def assemble_exact_extension_cem(
    nodes: tuple[tuple[Fraction, Fraction], ...],
    cells: np.ndarray,
    tagged_edges: np.ndarray,
    *,
    cell_conductivities: tuple[Fraction, ...],
    contact_impedance: Fraction,
    n_electrodes: int,
) -> tuple[Matrix, Matrix, Matrix]:
    node_count = len(nodes)
    electrode_count = int(n_electrodes)
    a_r = SparseMatrix.zeros(node_count, node_count)
    coupling = SparseMatrix.zeros(node_count, electrode_count)
    electrode_matrix = SparseMatrix.zeros(electrode_count, electrode_count)
    if len(cell_conductivities) != int(cells.shape[0]):
        raise ValueError("one exact conductivity is required per cell")
    for triangle, conductivity in zip(cells, cell_conductivities, strict=True):
        indices = [int(value) for value in triangle]
        x1, y1 = nodes[indices[0]]
        x2, y2 = nodes[indices[1]]
        x3, y3 = nodes[indices[2]]
        determinant = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = abs(determinant) / 2
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
        if int(label) <= 0:
            continue
        a = int(vertex_a)
        b_index = int(vertex_b)
        dx = nodes[a][0] - nodes[b_index][0]
        dy = nodes[a][1] - nodes[b_index][1]
        length_over_z = _sqrt_fraction_exact(dx * dx + dy * dy) / contact_impedance
        diagonal = _sympy_rational(length_over_z / 3)
        off_diagonal = _sympy_rational(length_over_z / 6)
        half = _sympy_rational(length_over_z / 2)
        total = _sympy_rational(length_over_z)
        electrode = int(label) - 1
        a_r[a, a] += diagonal
        a_r[b_index, b_index] += diagonal
        a_r[a, b_index] += off_diagonal
        a_r[b_index, a] += off_diagonal
        coupling[a, electrode] -= half
        coupling[b_index, electrode] -= half
        electrode_matrix[electrode, electrode] += total
    return Matrix(a_r), Matrix(coupling), Matrix(electrode_matrix)


def _zero_sum_basis(count: int) -> Matrix:
    basis = SparseMatrix.zeros(count, count - 1)
    for column in range(count - 1):
        basis[column, column] = 1
        basis[count - 1, column] = -1
    return Matrix(basis)


def _sparse_matrix_payload(matrix: Any) -> dict[str, Any]:
    entries = [
        [int(row), int(column), str(value)]
        for (row, column), value in matrix.todok().items()
        if value != 0
    ]
    return {
        "rows": int(matrix.rows),
        "columns": int(matrix.cols),
        "entries": entries,
    }


def _matrix_from_fraction_strings(values: list[list[str]]) -> Matrix:
    return Matrix(
        [[_sympy_rational(Fraction(value)) for value in row] for row in values]
    )


def _flint_cache_path(system_key: tuple[Any, ...]) -> Path:
    digest = hashlib.sha256(
        json.dumps(list(system_key), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    override = os.environ.get("CEM_EXTENSION_QQ_CACHE_DIR")
    cache_dir = Path(override).resolve() if override else DEFAULT_QQ_CACHE_DIR
    return cache_dir / f"{digest}.json"


def _validated_flint_result(
    path: Path,
    system_key: tuple[Any, ...],
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != FLINT_SCHEMA:
        raise RuntimeError(f"invalid FLINT cache schema: {path}")
    if result.get("system_key") != list(system_key):
        raise RuntimeError(f"FLINT cache system key mismatch: {path}")
    if result.get("python_flint_version") != FLINT_VERSION:
        raise RuntimeError(f"FLINT cache backend version mismatch: {path}")
    solution_strings = result["classic_solution_basis"]
    node_count = int(result["node_count"])
    electrode_count = int(result["electrode_count"])
    voltage_rows = solution_strings[node_count : node_count + electrode_count]
    digest = hashlib.sha256(
        json.dumps(voltage_rows, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if digest != result.get("basis_voltage_sha256"):
        raise RuntimeError(f"FLINT cache truth digest mismatch: {path}")
    for flag in (
        "classic_residual_zero",
        "robin_residual_zero",
        "classic_robin_identical",
    ):
        if result.get(flag) is not True:
            raise RuntimeError(f"FLINT cache certification flag {flag} is false")
    return result


def _solve_large_system_with_flint(
    *,
    system_key: tuple[Any, ...],
    full_matrix: Any,
    basis_rhs: Any,
    a_r: Any,
    coupling: Any,
    electrode_matrix: Any,
    basis: Any,
    node_count: int,
    electrode_count: int,
) -> dict[str, Any]:
    cache_path = _flint_cache_path(system_key)
    if cache_path.is_file():
        return _validated_flint_result(cache_path, system_key)
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("Q4 exact solve requires isolated uv/python-flint backend")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": FLINT_SCHEMA,
        "system_key": list(system_key),
        "node_count": node_count,
        "electrode_count": electrode_count,
        "full_matrix": _sparse_matrix_payload(full_matrix),
        "basis_rhs": _sparse_matrix_payload(basis_rhs),
        "A_R": _sparse_matrix_payload(a_r),
        "C": _sparse_matrix_payload(coupling),
        "D": _sparse_matrix_payload(electrode_matrix),
        "basis": _sparse_matrix_payload(basis),
    }
    helper = ROOT / "scripts" / "benchmarks" / "cem_exact_flint_backend.py"
    helper_environment = os.environ.copy()
    for variable in ("LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH"):
        helper_environment.pop(variable, None)
    with tempfile.TemporaryDirectory(
        prefix="cem_exact_flint_",
        dir=cache_path.parent,
    ) as temporary_dir:
        input_path = Path(temporary_dir) / "input.json"
        input_path.write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                uv,
                "run",
                "--no-project",
                "--python",
                "/usr/bin/python3",
                "--with",
                f"python-flint=={FLINT_VERSION}",
                "python",
                str(helper),
                "--input",
                str(input_path),
                "--output",
                str(cache_path),
            ],
            cwd=ROOT,
            check=True,
            env=helper_environment,
        )
    return _validated_flint_result(cache_path, system_key)


@lru_cache(maxsize=None)
def _solve_extension_basis_system(*key: Any) -> dict[str, Any]:
    (
        edge_subdivisions,
        radial_layers,
        n_electrodes,
        pattern,
        low_num,
        low_den,
        high_num,
        high_den,
        z_num,
        z_den,
    ) = key
    synthetic_case = ExtensionCase(
        "CACHE",
        "cache",
        "cache",
        "cache",
        int(edge_subdivisions),
        int(radial_layers),
        int(n_electrodes),
        str(pattern),
        int(low_num),
        int(low_den),
        int(high_num),
        int(high_den),
        int(z_num),
        int(z_den),
        1,
        "adjacent",
    )
    nodes, cells, edges, _, _ = extension_case_mesh(synthetic_case)
    cell_sigma = extension_case_cell_conductivities(synthetic_case, nodes, cells)
    a_r, coupling, electrode_matrix = assemble_exact_extension_cem(
        nodes,
        cells,
        edges,
        cell_conductivities=cell_sigma,
        contact_impedance=synthetic_case.contact_impedance,
        n_electrodes=synthetic_case.n_electrodes,
    )
    count = synthetic_case.n_electrodes
    basis = _zero_sum_basis(count)
    node_count = len(nodes)
    full_size = node_count + count + 1
    full_matrix = SparseMatrix.zeros(full_size, full_size)
    full_matrix[:node_count, :node_count] = a_r
    full_matrix[:node_count, node_count : node_count + count] = coupling
    full_matrix[node_count : node_count + count, :node_count] = coupling.T
    full_matrix[
        node_count : node_count + count,
        node_count : node_count + count,
    ] = electrode_matrix
    for electrode in range(count):
        full_matrix[node_count + electrode, full_size - 1] = 1
        full_matrix[full_size - 1, node_count + electrode] = 1
    basis_rhs = SparseMatrix.zeros(full_size, count - 1)
    basis_rhs[node_count : node_count + count, :] = basis
    if node_count >= 500:
        compiled = _solve_large_system_with_flint(
            system_key=tuple(key),
            full_matrix=full_matrix,
            basis_rhs=basis_rhs,
            a_r=a_r,
            coupling=coupling,
            electrode_matrix=electrode_matrix,
            basis=basis,
            node_count=node_count,
            electrode_count=count,
        )
        classic_solution_basis = _matrix_from_fraction_strings(
            compiled["classic_solution_basis"]
        )
        reduced_map = _matrix_from_fraction_strings(compiled["reduced_map"])
        coefficient_basis = _matrix_from_fraction_strings(
            compiled["robin_coefficient_basis"]
        )
        exact_backend = str(compiled["backend"])
        persistent_cache = str(_flint_cache_path(tuple(key)).resolve())
    else:
        classic_solution_basis = (
            full_matrix.to_DM()
            .convert_to(QQ)
            .lu_solve(basis_rhs.to_DM().convert_to(QQ))
            .to_Matrix()
        )
        response = (
            a_r.to_DM()
            .convert_to(QQ)
            .lu_solve(coupling.to_DM().convert_to(QQ))
            .to_Matrix()
        )
        schur = electrode_matrix - coupling.T * response
        reduced_map = basis.T * schur * basis
        reduced_rhs_basis = basis.T * basis
        coefficient_basis = (
            reduced_map.to_DM()
            .convert_to(QQ)
            .lu_solve(reduced_rhs_basis.to_DM().convert_to(QQ))
            .to_Matrix()
        )
        exact_backend = "SymPy DomainMatrix QQ lu_solve"
        persistent_cache = None
    if any(value != 0 for value in full_matrix * classic_solution_basis - basis_rhs):
        raise RuntimeError("extension exact classic basis residual is not zero")
    reduced_rhs_basis = basis.T * basis
    if any(value != 0 for value in reduced_map * coefficient_basis - reduced_rhs_basis):
        raise RuntimeError("extension exact Robin basis residual is not zero")
    classic_voltage_basis = classic_solution_basis[node_count : node_count + count, :]
    robin_voltage_basis = basis * coefficient_basis
    if classic_voltage_basis != robin_voltage_basis:
        raise RuntimeError("extension exact Classic and Robin basis voltages differ")
    return {
        "basis": basis,
        "full_matrix": full_matrix,
        "node_count": node_count,
        "classic_solution_basis": classic_solution_basis,
        "robin_coefficient_basis": coefficient_basis,
        "reduced_map": reduced_map,
        "exact_backend": exact_backend,
        "persistent_cache": persistent_cache,
    }


def extension_basis_cache_clear() -> None:
    _solve_extension_basis_system.cache_clear()


def extension_basis_cache_info():
    return _solve_extension_basis_system.cache_info()


def _exact_currents(currents: np.ndarray) -> Matrix:
    return Matrix(
        [
            [_sympy_rational(Fraction.from_float(float(value))) for value in row]
            for row in np.asarray(currents, dtype=np.float64)
        ]
    )


def solve_exact_extension_case(case: ExtensionCase) -> dict[str, Any]:
    system = _solve_extension_basis_system(*extension_case_system_key(case))
    count = case.n_electrodes
    currents = _exact_currents(extension_current_patterns(count, case.drive_skip))
    current_coordinates = currents[: count - 1, :]
    node_count = int(system["node_count"])
    full_size = node_count + count + 1
    rhs = SparseMatrix.zeros(full_size, currents.cols)
    rhs[node_count : node_count + count, :] = currents
    classic_solution = system["classic_solution_basis"] * current_coordinates
    if any(
        value != 0 for value in system["full_matrix"] * classic_solution - Matrix(rhs)
    ):
        raise RuntimeError(f"{case.case_id} exact classic residual is not zero")
    classic_voltage = classic_solution[node_count : node_count + count, :]
    basis = system["basis"]
    reduced_rhs = basis.T * currents
    coefficients = system["robin_coefficient_basis"] * current_coordinates
    robin_voltage = basis * coefficients
    if any(value != 0 for value in system["reduced_map"] * coefficients - reduced_rhs):
        raise RuntimeError(f"{case.case_id} exact Robin residual is not zero")
    if classic_voltage != robin_voltage:
        raise RuntimeError(f"{case.case_id} exact Classic and Robin voltages differ")
    if any(
        sum(classic_voltage[row, column] for row in range(count)) != 0
        for column in range(classic_voltage.cols)
    ):
        raise RuntimeError(f"{case.case_id} exact voltage gauge is not zero")
    exact_strings = [
        [str(classic_voltage[row, column]) for column in range(classic_voltage.cols)]
        for row in range(classic_voltage.rows)
    ]
    return {
        "case": case,
        "voltage": classic_voltage,
        "reduced_map": system["reduced_map"],
        "reduced_rhs": reduced_rhs,
        "basis": basis,
        "exact_classic_residual_zero": True,
        "exact_robin_residual_zero": True,
        "exact_classic_robin_identical": True,
        "exact_linear_solver": system["exact_backend"],
        "exact_domain": "QQ",
        "exact_basis_cache_key": list(extension_case_system_key(case)),
        "exact_basis_rhs_count": count - 1,
        "truth_sha256": hashlib.sha256(
            json.dumps(exact_strings, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "truth_fraction_strings": exact_strings,
        "persistent_cache": system["persistent_cache"],
    }


def exact_extension_accuracy_metrics(
    candidate: np.ndarray,
    reference: dict[str, Any],
) -> dict[str, Any]:
    with mp.workdps(100):
        candidate_mp = _mp_matrix_from_float64(candidate)
        truth_mp = _mp_matrix_from_sympy(reference["voltage"])
        delta = candidate_mp - truth_mp
        relative_error = _mp_frobenius(delta) / _mp_frobenius(truth_mp)
        count = candidate_mp.rows
        gauge = mp.matrix(1, candidate_mp.cols)
        centered = mp.matrix(candidate_mp.rows, candidate_mp.cols)
        for column in range(candidate_mp.cols):
            mean = mp.fsum(candidate_mp[row, column] for row in range(count)) / count
            gauge[0, column] = mean * count
            for row in range(count):
                centered[row, column] = candidate_mp[row, column] - mean
        coefficients = mp.matrix(count - 1, candidate_mp.cols)
        for row in range(count - 1):
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
        maximum = max((abs(value) for value in delta), default=mp.mpf("0"))
        return {
            "truth_relative_l2": float(relative_error),
            "truth_relative_l2_decimal": mp.nstr(relative_error, 40),
            "truth_max_abs": float(maximum),
            "truth_max_abs_decimal": mp.nstr(maximum, 40),
            "exact_reduced_scaled_backward_residual": float(backward),
            "exact_reduced_scaled_backward_residual_decimal": mp.nstr(backward, 40),
            "voltage_gauge_relative_residual": float(gauge_residual),
            "reduced_condition_number_2_estimate": float(
                np.linalg.cond(np.asarray(reference["reduced_map"], dtype=np.float64))
            ),
        }


def _write_extension_gmsh22(
    path: Path,
    nodes: np.ndarray,
    cells: np.ndarray,
    edges: np.ndarray,
    *,
    n_electrodes: int,
    material_ids: np.ndarray,
    material_count: int,
) -> None:
    insulating_tag = n_electrodes + 1
    material_tags = [n_electrodes + 1 + index for index in range(1, material_count + 1)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$PhysicalNames\n{n_electrodes + 1 + material_count}\n")
        for electrode in range(1, n_electrodes + 1):
            handle.write(f'1 {electrode} "electrode_{electrode}"\n')
        handle.write(f'1 {insulating_tag} "insulating"\n')
        for index, tag in enumerate(material_tags, start=1):
            handle.write(f'2 {tag} "sigma_{index}"\n')
        handle.write("$EndPhysicalNames\n")
        handle.write(f"$Nodes\n{nodes.shape[0]}\n")
        for index, coordinate in enumerate(nodes, start=1):
            handle.write(f"{index} {coordinate[0]:.17g} {coordinate[1]:.17g} 0\n")
        handle.write("$EndNodes\n")
        handle.write(f"$Elements\n{edges.shape[0] + cells.shape[0]}\n")
        element_id = 1
        for vertex_a, vertex_b, label in edges:
            tag = int(label) if int(label) > 0 else insulating_tag
            handle.write(
                f"{element_id} 1 2 {tag} {tag} {int(vertex_a) + 1} {int(vertex_b) + 1}\n"
            )
            element_id += 1
        for triangle, material_id in zip(cells, material_ids, strict=True):
            tag = material_tags[int(material_id) - 1]
            handle.write(
                f"{element_id} 2 2 {tag} {tag} {int(triangle[0]) + 1} "
                f"{int(triangle[1]) + 1} {int(triangle[2]) + 1}\n"
            )
            element_id += 1
        handle.write("$EndElements\n")


def prepare_extension_case_fixture(
    output_dir: Path,
    case: ExtensionCase,
) -> dict[str, Any]:
    exact_nodes, cells, edges, electrode_nodes, electrode_counts = extension_case_mesh(
        case
    )
    nodes = float_nodes(exact_nodes)
    exact_sigma = extension_case_cell_conductivities(case, exact_nodes, cells)
    sigma = np.asarray(exact_sigma, dtype=np.float64)
    unique_sigma = tuple(sorted(set(exact_sigma)))
    material_ids = np.asarray(
        [unique_sigma.index(value) + 1 for value in exact_sigma],
        dtype=np.int64,
    )
    currents = extension_current_patterns(case.n_electrodes, case.drive_skip)
    fingerprint = canonical_mesh_fingerprint(nodes, cells, edges)
    sigma_digest = conductivity_digest(exact_sigma)
    case_dir = output_dir / "cases" / f"{case.case_id}_{case.label}"
    common_dir = case_dir / "common_mesh"
    mat_path = common_dir / "cem_exact_extension_p1.mat"
    msh_path = common_dir / "cem_exact_extension_p1.msh"
    metadata_path = common_dir / "cem_exact_extension_p1.json"
    payload = {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "schema_version": 3,
        "index_base": 1,
        "dimension": 2,
        "cell_type": "triangle",
        "boundary_entity_type": "edge",
        "source_framework": "exact_rational_extension_fixture",
        "nodes": nodes,
        "elems": cells + 1,
        "boundary_facets": edges[:, :2] + 1,
        "boundary_edges": edges[:, :2] + 1,
        "tagged_boundary_edges": np.column_stack((edges[:, :2] + 1, edges[:, 2])),
        "electrode_nodes": electrode_nodes + 1,
        "electrode_node_counts": electrode_counts,
        "n_elec": case.n_electrodes,
        "background": float(case.conductivity_low),
        "truth_elem_data": sigma,
        "cell_material_ids": material_ids,
        "material_conductivities": np.asarray(unique_sigma, dtype=np.float64),
        "contact_impedance": float(case.contact_impedance),
        "mesh_name": f"cem_exact_extension_{case.case_id.lower()}",
        "mesh_level": case.refinement_level_id,
        "scenario_name": case.label,
        "electrode_coverage": 0.5,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "suite_schema": SUITE_SCHEMA,
        "case_id": case.case_id,
        "current_patterns": currents,
        "drive_skip": case.drive_skip,
        "conductivity_pattern": case.conductivity_pattern,
        "conductivity_digest": sigma_digest,
    }
    save_exchange_mat(mat_path, payload)
    _write_extension_gmsh22(
        msh_path,
        nodes,
        cells,
        edges,
        n_electrodes=case.n_electrodes,
        material_ids=material_ids,
        material_count=len(unique_sigma),
    )
    metadata = {
        "suite_schema": SUITE_SCHEMA,
        "geometry_schema": GEOMETRY_SCHEMA,
        "case": asdict(case),
        "case_id": case.case_id,
        "family": case.family,
        "label": case.label,
        "refinement_level_id": case.refinement_level_id,
        "edge_subdivisions": case.edge_subdivisions,
        "radial_layers": case.radial_layers,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "nodes": int(nodes.shape[0]),
        "cells": int(cells.shape[0]),
        "boundary_edges": int(edges.shape[0]),
        "n_electrodes": case.n_electrodes,
        "potential_order": 1,
        "scalar_dtype": "float64",
        "conductivity_pattern": case.conductivity_pattern,
        "conductivity_low_exact": str(case.conductivity_low),
        "conductivity_high_exact": str(case.conductivity_high),
        "conductivity_digest": sigma_digest,
        "material_conductivities": [float(value) for value in unique_sigma],
        "contact_impedance": float(case.contact_impedance),
        "contact_impedance_exact": str(case.contact_impedance),
        "drive_skip": case.drive_skip,
        "drive_label": case.drive_label,
        "current_patterns": currents.tolist(),
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


def pyeidors_cell_conductivity_values(
    mesh: Any, source_values: np.ndarray
) -> np.ndarray:
    values = np.asarray(source_values, dtype=np.float64).reshape(-1)
    original = np.asarray(
        mesh.mesh.topology.original_cell_index, dtype=np.int64
    ).reshape(-1)
    if (
        original.size != values.size
        or np.any(original < 0)
        or np.any(original >= values.size)
    ):
        raise ValueError(
            "DOLFINx original_cell_index does not match source conductivity"
        )
    return np.ascontiguousarray(values[original], dtype=np.float64)


def run_pyeidors_extension_case(fixture: dict[str, Any]) -> dict[str, Any]:
    from dolfinx import fem
    from scipy.io import loadmat

    case_dir = Path(fixture["case_dir"])
    mesh, _ = build_mesh_from_exchange_mat(Path(fixture["mat_path"]))
    payload = loadmat(fixture["mat_path"], squeeze_me=True, struct_as_record=False)
    source_sigma = np.asarray(payload["truth_elem_data"], dtype=np.float64).reshape(-1)
    count = int(fixture["n_electrodes"])
    config = BenchmarkConfig(
        n_electrodes=count,
        mesh_refinement=0,
        timing_repeats=TIMING_REPEATS,
    )
    model = EITForwardModel(
        n_elec=count,
        pattern_config=_pattern_config(config),
        z=np.full(count, float(fixture["contact_impedance"]), dtype=np.float64),
        mesh=mesh,
        potential_order=1,
        linear_backend="scipy",
    )
    if np.dtype(model.scalar_dtype) != np.dtype(np.float64):
        raise RuntimeError("extension suite requires real float64 PyEIDORS")
    loaded_edges = _extract_tagged_boundary_edges(mesh, list(model.electrode_tags))
    loaded_fingerprint = canonical_mesh_fingerprint(
        np.asarray(mesh.coordinates(), dtype=np.float64)[:, :2],
        np.asarray(mesh.cells(), dtype=np.int64),
        loaded_edges,
    )
    if loaded_fingerprint != fixture["mesh_fingerprint"]:
        raise RuntimeError(f"{fixture['case_id']} PyEIDORS mesh fingerprint mismatch")
    sigma_values = pyeidors_cell_conductivity_values(mesh, source_sigma)
    sigma = fem.Function(model.V_sigma)
    sigma.x.array[:] = sigma_values
    currents = np.asarray(fixture["current_patterns"], dtype=np.float64)
    robin_matrix, coupling, electrode_matrix = _assemble_pyeidors_blocks(model, sigma)
    timing, voltages, parity = benchmark_preassembled_blocks(
        robin_matrix,
        coupling,
        electrode_matrix,
        currents,
        repeats=TIMING_REPEATS,
    )
    block_path = case_dir / "pyeidors_assembled_blocks.mat"
    from scripts.benchmarks.cem_low_z_attribution import block_payload_sha256

    block_digest = block_payload_sha256(
        robin_matrix,
        coupling,
        electrode_matrix,
        currents,
    )
    savemat(
        block_path,
        {
            "A_R": robin_matrix,
            "C": coupling,
            "D": electrode_matrix,
            "currents": currents,
            "case_id": fixture["case_id"],
            "assembly": "PyEIDORS/DOLFINx",
            "assembled_blocks_sha256": block_digest,
        },
    )
    report = {
        "solver": "PyEIDORS/DOLFINx",
        "suite_schema": SUITE_SCHEMA,
        "case_id": fixture["case_id"],
        "physical_config": {
            "n_electrodes": count,
            "conductivity_pattern": fixture["conductivity_pattern"],
            "conductivity_digest": fixture["conductivity_digest"],
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
            "cell_conductivity_order": "source[original_cell_index]",
        },
        "linear_solver": {
            "classic": "SciPy SuperLU augmented CEM",
            "robin": "SciPy SuperLU A_R plus dense reduced LU",
            "scalar_dtype": "float64",
        },
        "timing": timing,
        "within_solver": parity,
        "assembled_blocks": str(block_path.resolve()),
        "assembled_blocks_sha256": block_digest,
        "raw_electrode_voltages": {
            formulation: np.asarray(voltage, dtype=np.float64).tolist()
            for formulation, voltage in voltages.items()
        },
    }
    write_json(case_dir / "pyeidors_report.json", report)
    return report


def prepare_extension_suite(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for case in EXTENSION_CASES:
        fixture = prepare_extension_case_fixture(output_dir, case)
        run_pyeidors_extension_case(fixture)
        fixtures.append(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in fixture.items()
            }
        )
    manifest = {
        "suite_schema": SUITE_SCHEMA,
        "preregistered_case_ids": [case.case_id for case in EXTENSION_CASES],
        "attribution_case_ids": list(ATTRIBUTION_CASE_IDS),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "case_count": len(fixtures),
        "cases": fixtures,
    }
    write_json(output_dir / "suite_manifest.json", manifest)
    return manifest


def aggregate_extension_timing(output_dir: Path) -> dict[str, Any]:
    manifest = _load_json(output_dir / "suite_manifest.json")
    records = []
    report_names = (
        "pyeidors_report.json",
        "ngsolve_report.json",
        "eidors_report.json",
    )
    for fixture in manifest["cases"]:
        case_dir = Path(fixture["case_dir"])
        for report_name in report_names:
            report = _load_json(case_dir / report_name)
            for formulation in FORMULATIONS:
                phase = report["timing"][formulation]
                cold = float(phase["cold_seconds"]["median"])
                setup = float(phase["setup_seconds"]["median"])
                warm = float(phase["warm_reuse_seconds"]["median"])
                if not cold > warm > 0.0:
                    raise RuntimeError(
                        f"{fixture['case_id']} {report['solver']} {formulation} "
                        "does not satisfy cold > warm reuse"
                    )
                records.append(
                    {
                        "case_id": fixture["case_id"],
                        "solver": report["solver"],
                        "formulation": formulation,
                        "cold_median_seconds": cold,
                        "setup_median_seconds": setup,
                        "warm_reuse_median_seconds": warm,
                        "cold_over_warm_reuse_speedup": cold / warm,
                    }
                )
    solvers = sorted({str(record["solver"]) for record in records})
    formulation_summary = {}
    ratio_summary = {}
    for solver in solvers:
        formulation_summary[solver] = {}
        ratio_summary[solver] = {}
        for formulation in FORMULATIONS:
            selected = [
                record
                for record in records
                if record["solver"] == solver and record["formulation"] == formulation
            ]
            formulation_summary[solver][formulation] = {
                "case_count": len(selected),
                "geometric_mean_cold_seconds": float(
                    np.exp(
                        np.mean(
                            np.log(
                                [record["cold_median_seconds"] for record in selected]
                            )
                        )
                    )
                ),
                "geometric_mean_setup_seconds": float(
                    np.exp(
                        np.mean(
                            np.log(
                                [record["setup_median_seconds"] for record in selected]
                            )
                        )
                    )
                ),
                "geometric_mean_warm_reuse_seconds": float(
                    np.exp(
                        np.mean(
                            np.log(
                                [
                                    record["warm_reuse_median_seconds"]
                                    for record in selected
                                ]
                            )
                        )
                    )
                ),
                "geometric_mean_cold_over_warm_reuse_speedup": float(
                    np.exp(
                        np.mean(
                            np.log(
                                [
                                    record["cold_over_warm_reuse_speedup"]
                                    for record in selected
                                ]
                            )
                        )
                    )
                ),
            }
        for phase in (
            "cold_median_seconds",
            "setup_median_seconds",
            "warm_reuse_median_seconds",
        ):
            ratios = []
            for case in EXTENSION_CASES:
                selected = {
                    record["formulation"]: record
                    for record in records
                    if record["case_id"] == case.case_id and record["solver"] == solver
                }
                ratios.append(
                    selected["robin_transconductance"][phase]
                    / selected["classic"][phase]
                )
            ratio_summary[solver][phase] = {
                "geometric_mean_robin_over_classic_ratio": float(
                    np.exp(np.mean(np.log(ratios)))
                ),
                "robin_faster_case_count": int(
                    np.count_nonzero(np.asarray(ratios) < 1)
                ),
                "case_count": len(ratios),
            }
    fields = tuple(records[0])
    csv_path = output_dir / "cem_exact_extension_timing.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    report = {
        "schema": "cem-exact-extension-timing-v1",
        "scope": "preassembled_A_R_C_D_blocks",
        "repeats": TIMING_REPEATS,
        "operations_per_sample": TIMING_OPERATIONS_PER_SAMPLE,
        "records": records,
        "formulation_summary": formulation_summary,
        "robin_over_classic_summary": ratio_summary,
        "artifact": csv_path.name,
    }
    json.dumps(report, allow_nan=False)
    write_json(output_dir / "cem_exact_extension_timing.json", report)
    return report


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_extension_report(
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
        raise ValueError(f"{solver} extension suite must use P1")
    if report.get("linear_solver", {}).get("scalar_dtype") != "float64":
        raise ValueError(f"{solver} extension suite must use float64")
    physical = report.get("physical_config", {})
    if int(physical.get("n_electrodes", -1)) != int(fixture["n_electrodes"]):
        raise ValueError(f"{solver} electrode count mismatch")
    if str(physical.get("conductivity_pattern")) != str(
        fixture["conductivity_pattern"]
    ):
        raise ValueError(f"{solver} conductivity pattern mismatch")
    if str(physical.get("conductivity_digest")) != str(fixture["conductivity_digest"]):
        raise ValueError(f"{solver} conductivity digest mismatch")
    if float(physical.get("contact_impedance", math.nan)) != float(
        fixture["contact_impedance"]
    ):
        raise ValueError(f"{solver} contact impedance mismatch")
    if int(physical.get("drive_skip", -1)) != int(fixture["drive_skip"]):
        raise ValueError(f"{solver} drive skip mismatch")
    timing = report.get("timing", {})
    if timing.get("schema") != TIMING_SCHEMA:
        raise ValueError(f"{solver} timing schema mismatch")
    if int(timing.get("repeats", -1)) != TIMING_REPEATS:
        raise ValueError(f"{solver} timing repeats mismatch")
    expected_shape = (int(fixture["n_electrodes"]),) * 2
    for formulation in FORMULATIONS:
        values = np.asarray(
            report.get("raw_electrode_voltages", {}).get(formulation),
            dtype=np.float64,
        )
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{solver} {formulation} raw voltage mismatch")


def _extension_aggregate(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    solvers = sorted({str(item["solver"]) for item in metrics})
    case_lookup = {case.case_id: case for case in EXTENSION_CASES}
    rankings: dict[str, list[dict[str, Any]]] = {}
    win_counts = {
        formulation: {solver: 0 for solver in solvers} for formulation in FORMULATIONS
    }
    for case in EXTENSION_CASES:
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
            rankings[f"{case.case_id}:{formulation}"] = [
                {
                    "rank": rank,
                    "solver": item["solver"],
                    "truth_relative_l2": item["truth_relative_l2"],
                    "ratio_to_best": item["truth_relative_l2"]
                    / selected[0]["truth_relative_l2"],
                }
                for rank, item in enumerate(selected, start=1)
            ]
            win_counts[formulation][str(selected[0]["solver"])] += 1

    family_summary: dict[str, Any] = {}
    for family in ("range", "heterogeneous", "electrode_count", "large_q4"):
        family_ids = {case.case_id for case in EXTENSION_CASES if case.family == family}
        family_summary[family] = {}
        for formulation in FORMULATIONS:
            per_solver = {}
            for solver in solvers:
                errors = np.asarray(
                    [
                        item["truth_relative_l2"]
                        for item in metrics
                        if item["case_id"] in family_ids
                        and item["formulation"] == formulation
                        and item["solver"] == solver
                    ],
                    dtype=np.float64,
                )
                wins = sum(
                    rankings[f"{case_id}:{formulation}"][0]["solver"] == solver
                    for case_id in family_ids
                )
                per_solver[solver] = {
                    "record_count": int(errors.size),
                    "win_count": int(wins),
                    "geometric_mean_truth_relative_l2": float(
                        np.exp(np.mean(np.log(errors)))
                    ),
                    "median_truth_relative_l2": float(np.median(errors)),
                    "worst_truth_relative_l2": float(np.max(errors)),
                }
            family_summary[family][formulation] = per_solver

    q4_ids = [
        case.case_id for case in EXTENSION_CASES if case.refinement_level_id == "Q4"
    ]
    q4_summary = {}
    for formulation in FORMULATIONS:
        orders = [
            tuple(item["solver"] for item in rankings[f"{case_id}:{formulation}"])
            for case_id in q4_ids
        ]
        q4_summary[formulation] = {
            "case_ids": q4_ids,
            "observed_orders": [list(order) for order in orders],
            "same_order_all_six": all(order == orders[0] for order in orders[1:]),
            "ordering": list(orders[0])
            if all(order == orders[0] for order in orders[1:])
            else None,
        }
    universal = {}
    for formulation in FORMULATIONS:
        orders = [
            tuple(item["solver"] for item in rankings[f"{case.case_id}:{formulation}"])
            for case in EXTENSION_CASES
        ]
        same = all(order == orders[0] for order in orders[1:])
        universal[formulation] = {
            "supported": same,
            "ordering": list(orders[0]) if same else None,
        }
    return {
        "rankings": rankings,
        "win_counts": win_counts,
        "family_summary": family_summary,
        "q4_summary": q4_summary,
        "universal_ordering": universal,
        "case_metadata": {
            case_id: asdict(case) for case_id, case in case_lookup.items()
        },
    }


def _write_extension_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "family",
        "refinement_level_id",
        "n_electrodes",
        "conductivity_pattern",
        "contact_impedance_exact",
        "drive_label",
        "solver",
        "formulation",
        "truth_relative_l2",
        "truth_max_abs",
        "exact_reduced_scaled_backward_residual",
        "voltage_gauge_relative_residual",
        "reduced_condition_number_2_estimate",
        "classic_robin_relative_l2",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in metrics)


def _plot_extension_metrics(metrics: list[dict[str, Any]], path: Path) -> None:
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
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    for axis, formulation in zip(axes, FORMULATIONS, strict=True):
        for solver in solvers:
            selected = [
                next(
                    item
                    for item in metrics
                    if item["case_id"] == case.case_id
                    and item["solver"] == solver
                    and item["formulation"] == formulation
                )
                for case in EXTENSION_CASES
            ]
            axis.plot(
                np.arange(1, len(selected) + 1),
                [item["truth_relative_l2"] for item in selected],
                marker="o",
                markersize=3,
                linewidth=1.1,
                label=solver,
                color=colors[solver],
            )
        axis.set_yscale("log")
        axis.set_ylabel("Relative L2 vs exact QQ")
        axis.set_title("Classic CEM" if formulation == "classic" else "Robin CEM")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("Preregistered extension case index X01...X38")
    figure.suptitle("Exact rational CEM extension: absolute float64 accuracy")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def compare_extension_suite(output_dir: Path) -> dict[str, Any]:
    extension_basis_cache_clear()
    manifest = _load_json(output_dir / "suite_manifest.json")
    if manifest.get("preregistered_case_ids") != [
        case.case_id for case in EXTENSION_CASES
    ]:
        raise ValueError("extension manifest differs from preregistered case order")
    fixtures = {item["case_id"]: item for item in manifest["cases"]}
    metrics: list[dict[str, Any]] = []
    truth_records: dict[str, Any] = {}
    for case in EXTENSION_CASES:
        fixture = fixtures[case.case_id]
        reference = solve_exact_extension_case(case)
        truth_records[case.case_id] = {
            "truth_sha256": reference["truth_sha256"],
            "exact_linear_solver": reference["exact_linear_solver"],
            "exact_classic_residual_zero": True,
            "exact_robin_residual_zero": True,
            "exact_classic_robin_identical": True,
            "exact_basis_cache_key": reference["exact_basis_cache_key"],
            "exact_basis_rhs_count": reference["exact_basis_rhs_count"],
            "electrode_voltage_fractions": reference["truth_fraction_strings"],
        }
        case_dir = Path(fixture["case_dir"])
        reports = [
            _load_json(case_dir / "pyeidors_report.json"),
            _load_json(case_dir / "ngsolve_report.json"),
            _load_json(case_dir / "eidors_report.json"),
        ]
        if len({str(report["solver"]) for report in reports}) != 3:
            raise ValueError(f"{case.case_id} requires three distinct solvers")
        for report in reports:
            _validate_extension_report(report, fixture)
            raw = report["raw_electrode_voltages"]
            classic = np.asarray(raw["classic"], dtype=np.float64)
            robin = np.asarray(raw["robin_transconductance"], dtype=np.float64)
            internal_delta = float(
                np.linalg.norm(robin - classic) / np.linalg.norm(classic)
            )
            for formulation in FORMULATIONS:
                metrics.append(
                    {
                        "case_id": case.case_id,
                        "family": case.family,
                        "refinement_level_id": case.refinement_level_id,
                        "n_electrodes": case.n_electrodes,
                        "conductivity_pattern": case.conductivity_pattern,
                        "contact_impedance_exact": str(case.contact_impedance),
                        "drive_label": case.drive_label,
                        "solver": report["solver"],
                        "formulation": formulation,
                        "classic_robin_relative_l2": internal_delta,
                        **exact_extension_accuracy_metrics(
                            np.asarray(raw[formulation], dtype=np.float64),
                            reference,
                        ),
                    }
                )
    cache = extension_basis_cache_info()
    if cache.misses != 19 or cache.hits != 19:
        raise RuntimeError(
            f"extension QQ cache expected 19 misses/19 hits, got {cache}"
        )
    aggregate = _extension_aggregate(metrics)
    csv_path = output_dir / "cem_exact_extension_metrics.csv"
    plot_path = output_dir / "cem_exact_extension_accuracy.png"
    _write_extension_metrics_csv(csv_path, metrics)
    _plot_extension_metrics(metrics, plot_path)
    report = {
        "suite_schema": SUITE_SCHEMA,
        "metric_schema": METRIC_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "exact finite-dimensional rational P1 CEM systems; not continuum PDE truth",
        "preregistered_case_ids": [case.case_id for case in EXTENSION_CASES],
        "attribution_case_ids": list(ATTRIBUTION_CASE_IDS),
        "truth_method": {
            "domain": "QQ",
            "solver": (
                "SymPy DomainMatrix QQ multi-RHS lu_solve for Q0/Q2; "
                "isolated python-flint fmpq_mat.solve for Q4 with main-process "
                "SymPy exact residual and Classic/Robin identity revalidation"
            ),
            "uses_any_fem_solver_matrix": False,
            "persistent_q4_truth_cache": True,
            "basis_cache_hits": cache.hits,
            "basis_cache_misses": cache.misses,
        },
        "truth": truth_records,
        "metrics": metrics,
        **aggregate,
        "artifacts": {"csv": csv_path.name, "plot": plot_path.name},
    }
    json.dumps(report, allow_nan=False)
    write_json(output_dir / "cem_exact_extension_accuracy.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "compare", "timing"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--output-dir",
            type=Path,
            default=ROOT / "output" / "cem_exact_extension",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        manifest = prepare_extension_suite(args.output_dir.resolve())
        print(
            f"Prepared {manifest['case_count']} extension cases: {args.output_dir.resolve()}"
        )
        return 0
    if args.command == "timing":
        report = aggregate_extension_timing(args.output_dir.resolve())
        print(f"Aggregated {len(report['records'])} extension timing records")
        return 0
    report = compare_extension_suite(args.output_dir.resolve())
    for formulation, conclusion in report["q4_summary"].items():
        print(
            f"{formulation}: q4_same_order={conclusion['same_order_all_six']} "
            f"order={conclusion['ordering']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
