#!/usr/bin/env python3
"""Neutral contracts and independent checks for the six-method CEM study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_fair_common import (
    canonical_mesh_fingerprint,
    zero_sum_helmert_basis,
)


REPORT_SCHEMA = "cem-multifem-accuracy-v1"
FORMULATION_CLASSIC = "classic_augmented"
FORMULATION_ROBIN = "robin_transconductance"
BLOCK_KEYS = ("K", "B", "C_plus", "D", "A_R")
NATIVE_IDENTITY_TOLERANCE = 5.0e-11


@dataclass(frozen=True)
class PrimaryMethod:
    solver: str
    formulation: str
    role: str


PRIMARY_METHODS = (
    PrimaryMethod("EIDORS", FORMULATION_CLASSIC, "standard_implementation"),
    PrimaryMethod("PyEIDORS-DOLFINx", FORMULATION_ROBIN, "general_fem"),
    PrimaryMethod("NGSolve", FORMULATION_ROBIN, "general_fem"),
    PrimaryMethod("MFEM", FORMULATION_ROBIN, "general_fem"),
    PrimaryMethod("FreeFEM", FORMULATION_ROBIN, "general_fem"),
    PrimaryMethod("GetFEM", FORMULATION_ROBIN, "general_fem"),
)
PRIMARY_METHOD_BY_SOLVER = {method.solver: method for method in PRIMARY_METHODS}
NEW_GENERAL_SOLVERS = frozenset({"MFEM", "FreeFEM", "GetFEM"})


@dataclass(frozen=True)
class RobinSolution:
    A_R: np.ndarray
    T: np.ndarray
    Q: np.ndarray
    reduced_map: np.ndarray
    body_potential: np.ndarray
    electrode_voltage: np.ndarray


def _array(value: Any, *, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _relative_frobenius(actual: np.ndarray, expected: np.ndarray) -> float:
    numerator = float(np.linalg.norm(actual - expected, ord="fro"))
    denominator = max(float(np.linalg.norm(expected, ord="fro")), np.finfo(float).tiny)
    return numerator / denominator


def _scaled_inf(residual: np.ndarray, *scales: np.ndarray) -> float:
    denominator = max(
        *(float(np.linalg.norm(value, ord=np.inf)) for value in scales),
        1.0,
    )
    return float(np.linalg.norm(residual, ord=np.inf) / denominator)


def solve_robin_from_blocks(
    *,
    K: np.ndarray,
    B: np.ndarray,
    C_plus: np.ndarray,
    D: np.ndarray,
    currents: np.ndarray,
) -> RobinSolution:
    """Independently rebuild the Robin Schur solution from exported blocks.

    Candidate adapters must perform this solve natively. This routine verifies
    their exported state; it is not a candidate FEM implementation.
    """

    K = _array(K, ndim=2, name="K")
    B = _array(B, ndim=2, name="B")
    C_plus = _array(C_plus, ndim=2, name="C_plus")
    D = _array(D, ndim=2, name="D")
    currents = _array(currents, ndim=2, name="currents")
    n_nodes = K.shape[0]
    n_electrodes = D.shape[0]
    if K.shape != (n_nodes, n_nodes) or B.shape != K.shape:
        raise ValueError("K and B must be square with identical shape")
    if C_plus.shape != (n_nodes, n_electrodes):
        raise ValueError("C_plus shape mismatch")
    if D.shape != (n_electrodes, n_electrodes):
        raise ValueError("D must be square")
    if currents.shape[0] != n_electrodes:
        raise ValueError("current row count must equal electrode count")
    if not np.allclose(np.sum(currents, axis=0), 0.0, atol=1.0e-13, rtol=0.0):
        raise ValueError("currents must be exactly balanced within float64 tolerance")

    A_R = K + B
    Q = zero_sum_helmert_basis(n_electrodes)
    T = D - C_plus.T @ np.linalg.solve(A_R, C_plus)
    reduced_map = Q.T @ T @ Q
    coordinates = np.linalg.solve(reduced_map, Q.T @ currents)
    electrode_voltage = Q @ coordinates
    body_potential = np.linalg.solve(A_R, C_plus @ electrode_voltage)
    return RobinSolution(
        A_R=A_R,
        T=T,
        Q=Q,
        reduced_map=reduced_map,
        body_potential=body_potential,
        electrode_voltage=electrode_voltage,
    )


def _contains_forbidden_timing(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).lower() in {"timing", "elapsed_seconds", "runtime_seconds"}
            or _contains_forbidden_timing(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_timing(item) for item in value)
    return False


def validate_native_report(
    report: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    expected_solver: str,
    tolerance: float = NATIVE_IDENTITY_TOLERANCE,
) -> dict[str, float]:
    """Fail closed on the shared-mesh, P1, native-assembly contract."""

    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("multi-FEM report schema mismatch")
    if report.get("solver") != expected_solver:
        raise ValueError("solver identity mismatch")
    method = PRIMARY_METHOD_BY_SOLVER.get(expected_solver)
    if method is None:
        raise ValueError(f"unknown primary solver {expected_solver}")
    if report.get("formulation") != method.formulation:
        raise ValueError(f"{expected_solver} primary formulation mismatch")
    if _contains_forbidden_timing(report):
        raise ValueError("accuracy-only report must not contain timing fields")
    if expected_solver in NEW_GENERAL_SOLVERS and not bool(
        report.get("implementation", {}).get("native_assembly", False)
    ):
        raise ValueError(f"{expected_solver} must declare native assembly")

    discretization = report.get("discretization", {})
    if int(discretization.get("potential_order", -1)) != 1:
        raise ValueError("controlled experiment requires P1 potential order")
    if int(discretization.get("geometry_order", -1)) != 1:
        raise ValueError("controlled experiment requires straight P1 geometry")
    if discretization.get("scalar_dtype") != "float64":
        raise ValueError("controlled experiment requires real float64")
    if not bool(discretization.get("mesh_import_verified", False)):
        raise ValueError("solver did not verify its imported mesh")

    nodes = _array(discretization.get("imported_nodes"), ndim=2, name="nodes")
    cells = np.asarray(discretization.get("imported_cells_zero_based"), dtype=np.int64)
    edges = np.asarray(
        discretization.get("imported_tagged_boundary_edges_zero_based"),
        dtype=np.int64,
    )
    imported_fingerprint = canonical_mesh_fingerprint(nodes, cells, edges)
    declared_fingerprint = discretization.get("mesh_fingerprint")
    if imported_fingerprint != declared_fingerprint:
        raise ValueError("declared fingerprint does not match actual imported topology")
    if imported_fingerprint != fixture["mesh_fingerprint"]:
        raise ValueError("solver imported a different common mesh")

    blocks = report.get("blocks", {})
    native_blocks = {
        key: _array(blocks.get(key), ndim=2, name=key) for key in BLOCK_KEYS
    }
    currents = _array(
        report.get("physical_config", {}).get("currents"),
        ndim=2,
        name="currents",
    )
    expected_currents = _array(fixture["currents"], ndim=2, name="fixture currents")
    if currents.shape != expected_currents.shape or not np.array_equal(
        currents, expected_currents
    ):
        raise ValueError("current patterns differ from fixture")

    reference = solve_robin_from_blocks(
        K=native_blocks["K"],
        B=native_blocks["B"],
        C_plus=native_blocks["C_plus"],
        D=native_blocks["D"],
        currents=currents,
    )
    solution = report.get("solution", {})
    native_T = _array(solution.get("T"), ndim=2, name="T")
    native_reduced = _array(solution.get("reduced_map"), ndim=2, name="reduced_map")
    native_body = _array(solution.get("body_potential"), ndim=2, name="body_potential")
    native_voltage = _array(
        solution.get("electrode_voltage"), ndim=2, name="electrode_voltage"
    )
    metrics = {
        "A_R_identity_relative_frobenius": _relative_frobenius(
            native_blocks["A_R"], reference.A_R
        ),
        "T_identity_relative_frobenius": _relative_frobenius(native_T, reference.T),
        "reduced_map_identity_relative_frobenius": _relative_frobenius(
            native_reduced, reference.reduced_map
        ),
        "body_solution_relative_frobenius": _relative_frobenius(
            native_body, reference.body_potential
        ),
        "voltage_solution_relative_frobenius": _relative_frobenius(
            native_voltage, reference.electrode_voltage
        ),
        "robin_residual_scaled_inf": _scaled_inf(
            native_blocks["A_R"] @ native_body
            - native_blocks["C_plus"] @ native_voltage,
            native_blocks["A_R"] @ native_body,
            native_blocks["C_plus"] @ native_voltage,
        ),
        "current_recovery_residual_scaled_inf": _scaled_inf(
            native_blocks["D"] @ native_voltage
            - native_blocks["C_plus"].T @ native_body
            - currents,
            currents,
        ),
        "voltage_gauge_inf": float(np.max(np.abs(np.sum(native_voltage, axis=0)))),
    }
    failures = {key: value for key, value in metrics.items() if value > tolerance}
    if failures:
        raise ValueError(f"native Robin identities failed: {failures}")
    return metrics
