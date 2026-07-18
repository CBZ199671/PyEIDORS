"""Shared mesh and timing primitives for the fair cross-FEM CEM benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import bmat, csc_matrix, issparse
from scipy.sparse.linalg import splu


MESH_FINGERPRINT_SCHEMA = "cem-p1-mesh-sha256-v1"
TIMING_SCOPE = "preassembled_A_R_C_D_blocks"


def zero_sum_helmert_basis(count: int) -> np.ndarray:
    """Return an orthonormal float64 basis for the zero-sum electrode space."""

    basis = np.zeros((count, count - 1), dtype=np.float64)
    for column in range(1, count):
        scale = np.sqrt(float(column * (column + 1)))
        basis[:column, column - 1] = 1.0 / scale
        basis[column, column - 1] = -float(column) / scale
    return basis


def _canonical_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("canonical connectivity must be a two-dimensional array")
    if array.shape[0] == 0:
        return np.ascontiguousarray(array, dtype="<i8")
    keys = tuple(array[:, column] for column in reversed(range(array.shape[1])))
    return np.ascontiguousarray(array[np.lexsort(keys)], dtype="<i8")


def canonical_mesh_fingerprint(
    nodes: np.ndarray,
    cells: np.ndarray,
    tagged_boundary_edges: np.ndarray,
) -> str:
    """Hash node coordinates, P1 triangles, and electrode-labelled boundary edges."""

    node_array = np.asarray(nodes, dtype=np.float64)
    cell_array = np.asarray(cells, dtype=np.int64)
    edge_array = np.asarray(tagged_boundary_edges, dtype=np.int64)
    if node_array.ndim != 2 or node_array.shape[1] < 2:
        raise ValueError("nodes must have at least two coordinate columns")
    if cell_array.ndim != 2 or cell_array.shape[1] != 3:
        raise ValueError("cells must contain P1 triangle connectivity")
    if edge_array.ndim != 2 or edge_array.shape[1] != 3:
        raise ValueError("tagged boundary edges must have [v0, v1, electrode_label]")

    canonical_nodes = np.ascontiguousarray(
        np.round(node_array[:, :2], decimals=12), dtype="<f8"
    )
    canonical_cells = _canonical_rows(np.sort(cell_array, axis=1))
    endpoints = np.sort(edge_array[:, :2], axis=1)
    canonical_edges = _canonical_rows(np.column_stack((endpoints, edge_array[:, 2])))

    digest = hashlib.sha256()
    digest.update(MESH_FINGERPRINT_SCHEMA.encode("ascii"))
    for array in (canonical_nodes, canonical_cells, canonical_edges):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_gmsh22(
    path: Path,
    nodes: np.ndarray,
    cells: np.ndarray,
    tagged_boundary_edges: np.ndarray,
    n_electrodes: int,
) -> None:
    """Write a canonical ASCII Gmsh 2.2 P1 mesh understood by NGSolve."""

    node_array = np.asarray(nodes, dtype=np.float64)
    cell_array = np.asarray(cells, dtype=np.int64)
    edge_array = np.asarray(tagged_boundary_edges, dtype=np.int64)
    insulating_tag = int(n_electrodes) + 1
    domain_tag = int(n_electrodes) + 2
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        handle.write(f"$PhysicalNames\n{n_electrodes + 2}\n")
        for electrode in range(1, n_electrodes + 1):
            handle.write(f'1 {electrode} "electrode_{electrode}"\n')
        handle.write(f'1 {insulating_tag} "insulating"\n')
        handle.write(f'2 {domain_tag} "domain"\n')
        handle.write("$EndPhysicalNames\n")
        handle.write(f"$Nodes\n{node_array.shape[0]}\n")
        for index, coordinate in enumerate(node_array, start=1):
            handle.write(f"{index} {coordinate[0]:.17g} {coordinate[1]:.17g} 0\n")
        handle.write("$EndNodes\n")
        handle.write(f"$Elements\n{edge_array.shape[0] + cell_array.shape[0]}\n")
        element_id = 1
        for vertex_a, vertex_b, label in edge_array:
            physical_tag = int(label) if int(label) > 0 else insulating_tag
            handle.write(
                f"{element_id} 1 2 {physical_tag} {physical_tag} "
                f"{int(vertex_a) + 1} {int(vertex_b) + 1}\n"
            )
            element_id += 1
        for triangle in cell_array:
            handle.write(
                f"{element_id} 2 2 {domain_tag} {domain_tag} "
                f"{int(triangle[0]) + 1} {int(triangle[1]) + 1} "
                f"{int(triangle[2]) + 1}\n"
            )
            element_id += 1
        handle.write("$EndElements\n")


def timing_summary(samples: list[float]) -> dict[str, Any]:
    """Return raw samples plus robust timing statistics."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("timing samples must be a non-empty vector")
    q1, q3 = np.percentile(values, [25.0, 75.0])
    return {
        "samples": values.tolist(),
        "median": float(np.median(values)),
        "iqr": float(q3 - q1),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def _as_csc(matrix: Any) -> csc_matrix:
    if issparse(matrix):
        return matrix.astype(np.float64, copy=False).tocsc()
    return csc_matrix(np.asarray(matrix, dtype=np.float64))


@dataclass(frozen=True)
class _ClassicState:
    factor: Any
    node_count: int
    electrode_count: int


@dataclass(frozen=True)
class _RobinState:
    body_factor: Any
    response_basis: np.ndarray
    electrode_basis: np.ndarray
    reduced_factor: tuple[np.ndarray, np.ndarray]


def _classic_state(
    robin_matrix: csc_matrix,
    coupling: csc_matrix,
    electrode_matrix: csc_matrix,
) -> _ClassicState:
    electrode_count = int(electrode_matrix.shape[0])
    constraint = csc_matrix(np.ones((electrode_count, 1), dtype=np.float64))
    augmented = bmat(
        [
            [robin_matrix, coupling, None],
            [coupling.T, electrode_matrix, constraint],
            [None, constraint.T, csc_matrix((1, 1), dtype=np.float64)],
        ],
        format="csc",
    )
    return _ClassicState(
        factor=splu(augmented),
        node_count=int(robin_matrix.shape[0]),
        electrode_count=electrode_count,
    )


def _solve_classic(state: _ClassicState, currents: np.ndarray):
    rhs = np.zeros(
        (state.node_count + state.electrode_count + 1, currents.shape[1]),
        dtype=np.float64,
    )
    rhs[state.node_count : state.node_count + state.electrode_count, :] = currents
    solution = state.factor.solve(rhs)
    return (
        solution[: state.node_count, :],
        solution[state.node_count : state.node_count + state.electrode_count, :],
    )


def _robin_state(
    robin_matrix: csc_matrix,
    coupling: csc_matrix,
    electrode_matrix: csc_matrix,
) -> _RobinState:
    basis = zero_sum_helmert_basis(int(electrode_matrix.shape[0]))
    body_factor = splu(robin_matrix)
    response_basis = body_factor.solve(coupling @ basis)
    reduced_map = basis.T @ (electrode_matrix @ basis - coupling.T @ response_basis)
    return _RobinState(
        body_factor=body_factor,
        response_basis=np.asarray(response_basis, dtype=np.float64),
        electrode_basis=basis,
        reduced_factor=lu_factor(np.asarray(reduced_map, dtype=np.float64)),
    )


def _solve_robin(state: _RobinState, currents: np.ndarray):
    coefficients = lu_solve(
        state.reduced_factor,
        state.electrode_basis.T @ currents,
    )
    return (
        -(state.response_basis @ coefficients),
        state.electrode_basis @ coefficients,
    )


def _elapsed(call: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = call()
    return float(time.perf_counter() - started), result


def benchmark_preassembled_blocks(
    robin_matrix: Any,
    coupling: Any,
    electrode_matrix: Any,
    currents: np.ndarray,
    *,
    repeats: int = 11,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float]]:
    """Benchmark independent classic/Robin cold and warm algebraic states."""

    if repeats < 3:
        raise ValueError("fair timing requires at least three repetitions")
    a_r = _as_csc(robin_matrix)
    c = _as_csc(coupling)
    d = _as_csc(electrode_matrix)
    current_matrix = np.asarray(currents, dtype=np.float64)
    if current_matrix.ndim != 2 or current_matrix.shape[0] != d.shape[0]:
        raise ValueError("currents must have shape (n_electrodes, n_rhs)")

    cold_samples = {"classic": [], "robin_transconductance": []}
    last_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def cold_classic():
        state = _classic_state(a_r, c, d)
        return _solve_classic(state, current_matrix)

    def cold_robin():
        state = _robin_state(a_r, c, d)
        return _solve_robin(state, current_matrix)

    cold_calls = {
        "classic": cold_classic,
        "robin_transconductance": cold_robin,
    }
    # Prime Python/SciPy dispatch and allocator paths without retaining either
    # factor state. Timed cold samples still rebuild every formulation state.
    cold_classic()
    cold_robin()
    for repetition in range(repeats):
        order = (
            ("classic", "robin_transconductance")
            if repetition % 2 == 0
            else ("robin_transconductance", "classic")
        )
        for name in order:
            elapsed, result = _elapsed(cold_calls[name])
            cold_samples[name].append(elapsed)
            last_results[name] = result

    classic_population, classic_state = _elapsed(lambda: _classic_state(a_r, c, d))
    robin_population, robin_state = _elapsed(lambda: _robin_state(a_r, c, d))
    warm_calls = {
        "classic": lambda: _solve_classic(classic_state, current_matrix),
        "robin_transconductance": lambda: _solve_robin(robin_state, current_matrix),
    }
    warm_samples = {"classic": [], "robin_transconductance": []}
    for repetition in range(repeats):
        order = (
            ("robin_transconductance", "classic")
            if repetition % 2 == 0
            else ("classic", "robin_transconductance")
        )
        for name in order:
            elapsed, result = _elapsed(warm_calls[name])
            warm_samples[name].append(elapsed)
            last_results[name] = result

    classic_potential, classic_voltage = last_results["classic"]
    robin_potential, robin_voltage = last_results["robin_transconductance"]
    denominator_u = max(float(np.linalg.norm(classic_potential)), np.finfo(float).eps)
    denominator_v = max(float(np.linalg.norm(classic_voltage)), np.finfo(float).eps)
    parity = {
        "body_potential_relative_l2": float(
            np.linalg.norm(robin_potential - classic_potential) / denominator_u
        ),
        "electrode_voltage_relative_l2": float(
            np.linalg.norm(robin_voltage - classic_voltage) / denominator_v
        ),
    }
    timing = {
        "schema": "cem-fair-timing-v1",
        "scope": TIMING_SCOPE,
        "repeats": int(repeats),
        "rhs_count": int(current_matrix.shape[1]),
        "alternating_order": True,
        "untimed_runtime_priming": True,
        "cross_formulation_cache_reuse": False,
        "classic": {
            "cold_seconds": timing_summary(cold_samples["classic"]),
            "warm_population_seconds": classic_population,
            "warm_seconds": timing_summary(warm_samples["classic"]),
            "cold_sparse_factorizations": int(repeats),
            "cold_dense_factorizations": 0,
            "warm_sparse_factorizations": 1,
            "warm_dense_factorizations": 0,
            "warm_cache_hits": int(repeats),
            "rhs_solves_per_sample": int(current_matrix.shape[1]),
        },
        "robin_transconductance": {
            "cold_seconds": timing_summary(cold_samples["robin_transconductance"]),
            "warm_population_seconds": robin_population,
            "warm_seconds": timing_summary(warm_samples["robin_transconductance"]),
            "cold_sparse_factorizations": int(repeats),
            "cold_dense_factorizations": int(repeats),
            "warm_sparse_factorizations": 1,
            "warm_dense_factorizations": 1,
            "warm_cache_hits": int(repeats),
            "rhs_solves_per_sample": int(current_matrix.shape[1]),
            "response_basis_rhs_count": int(d.shape[0] - 1),
        },
    }
    voltages = {
        "classic": classic_voltage,
        "robin_transconductance": robin_voltage,
    }
    return timing, voltages, parity


def validate_solver_reports(reports: list[dict[str, Any]]) -> str:
    """Fail closed unless all reports use one P1 float64 mesh and fair timing."""

    if not reports:
        raise ValueError("at least one solver report is required")
    fingerprints: set[str] = set()
    for report in reports:
        solver = str(report.get("solver", "unknown"))
        dtype = str(report.get("linear_solver", {}).get("scalar_dtype", ""))
        if dtype != "float64":
            raise ValueError(f"{solver} scalar dtype must be float64, got {dtype!r}")
        discretization = report.get("discretization", {})
        if int(discretization.get("potential_order", -1)) != 1:
            raise ValueError(f"{solver} must use P1 potential elements")
        fingerprint = str(discretization.get("mesh_fingerprint", ""))
        if len(fingerprint) != 64:
            raise ValueError(f"{solver} is missing a valid mesh fingerprint")
        if not bool(discretization.get("mesh_import_verified", False)):
            raise ValueError(f"{solver} did not verify the imported common mesh")
        timing = report.get("timing", {})
        if timing.get("scope") != TIMING_SCOPE:
            raise ValueError(f"{solver} timing scope is not fair preassembled blocks")
        if bool(timing.get("cross_formulation_cache_reuse", True)):
            raise ValueError(f"{solver} reused cache artifacts across formulations")
        fingerprints.add(fingerprint)
    if len(fingerprints) != 1:
        raise ValueError(f"solver mesh fingerprints differ: {sorted(fingerprints)}")
    return fingerprints.pop()


__all__ = [
    "MESH_FINGERPRINT_SCHEMA",
    "TIMING_SCOPE",
    "benchmark_preassembled_blocks",
    "canonical_mesh_fingerprint",
    "timing_summary",
    "validate_solver_reports",
    "write_gmsh22",
    "zero_sum_helmert_basis",
]
