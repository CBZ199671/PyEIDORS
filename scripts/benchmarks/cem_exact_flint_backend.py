#!/usr/bin/env python3
"""Pure-python-flint helper for compiled exact-rational CEM basis solves."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from flint import fmpq, fmpq_mat


SCHEMA = "cem-exact-flint-basis-v1"


def _load_matrix(payload: dict[str, Any]) -> fmpq_mat:
    matrix = fmpq_mat(int(payload["rows"]), int(payload["columns"]))
    for row, column, value in payload["entries"]:
        matrix[int(row), int(column)] = fmpq(str(value))
    return matrix


def _matrix_strings(matrix: fmpq_mat) -> list[list[str]]:
    return [
        [str(matrix[row, column]) for column in range(matrix.ncols())]
        for row in range(matrix.nrows())
    ]


def solve_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported FLINT exact CEM schema")
    full_matrix = _load_matrix(payload["full_matrix"])
    basis_rhs = _load_matrix(payload["basis_rhs"])
    a_r = _load_matrix(payload["A_R"])
    coupling = _load_matrix(payload["C"])
    electrode = _load_matrix(payload["D"])
    basis = _load_matrix(payload["basis"])

    classic_solution_basis = full_matrix.solve(basis_rhs)
    if full_matrix * classic_solution_basis != basis_rhs:
        raise RuntimeError("FLINT Classic exact residual is not zero")
    response = a_r.solve(coupling)
    reduced_map = (
        basis.transpose() * (electrode - coupling.transpose() * response) * basis
    )
    reduced_rhs_basis = basis.transpose() * basis
    coefficient_basis = reduced_map.solve(reduced_rhs_basis)
    if reduced_map * coefficient_basis != reduced_rhs_basis:
        raise RuntimeError("FLINT Robin exact residual is not zero")
    node_count = int(payload["node_count"])
    electrode_count = int(payload["electrode_count"])
    classic_voltage_basis = fmpq_mat(electrode_count, electrode_count - 1)
    for row in range(electrode_count):
        for column in range(electrode_count - 1):
            classic_voltage_basis[row, column] = classic_solution_basis[
                node_count + row,
                column,
            ]
    robin_voltage_basis = basis * coefficient_basis
    if classic_voltage_basis != robin_voltage_basis:
        raise RuntimeError("FLINT Classic and Robin exact voltages differ")
    voltage_strings = _matrix_strings(classic_voltage_basis)
    truth_sha256 = hashlib.sha256(
        json.dumps(voltage_strings, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "schema": SCHEMA,
        "system_key": payload["system_key"],
        "backend": "python-flint fmpq_mat.solve",
        "python_flint_version": "0.6.0",
        "node_count": node_count,
        "electrode_count": electrode_count,
        "classic_solution_basis": _matrix_strings(classic_solution_basis),
        "reduced_map": _matrix_strings(reduced_map),
        "robin_coefficient_basis": _matrix_strings(coefficient_basis),
        "basis_voltage_sha256": truth_sha256,
        "classic_residual_zero": True,
        "robin_residual_zero": True,
        "classic_robin_identical": True,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, separators=(",", ":"), ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = solve_payload(payload)
    _atomic_json(args.output, result)
    print(
        f"FLINT exact CEM basis solved: {result['node_count']} nodes, "
        f"{result['electrode_count'] - 1} RHS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
