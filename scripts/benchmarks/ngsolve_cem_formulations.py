#!/usr/bin/env python3
"""NGSolve assembly benchmark for classic and Robin CEM formulations.

This is a headless, script form of the author's NGSolve notebook.  NGSolve
assembles the P2 volume/Robin forms; SciPy SuperLU solves both the augmented
classic CEM matrix and its Robin Schur complement on exactly those blocks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import ngsolve as ngs
from netgen.geom2d import SplineGeometry
import numpy as np
from scipy.sparse import bmat, csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu


CSV_FIELDS = (
    "solver",
    "formulation",
    "mode",
    "spatial_frequency",
    "current_norm_a",
    "voltage_norm_v",
    "characteristic_resistance_ohm",
)


@dataclass(frozen=True)
class Config:
    n_electrodes: int = 16
    radius_m: float = 4.0
    conductivity_s_per_m: float = 0.25
    contact_impedance: float = 1.0
    electrode_coverage: float = 0.7
    maxh_m: float = 0.2
    arc_subdivisions: int = 12
    potential_order: int = 2


def helmert_basis(count: int) -> np.ndarray:
    basis = np.zeros((count, count - 1), dtype=float)
    for column in range(1, count):
        scale = np.sqrt(float(column * (column + 1)))
        basis[:column, column - 1] = 1.0 / scale
        basis[column, column - 1] = -float(column) / scale
    return basis


def current_patterns(config: Config) -> tuple[np.ndarray, list[tuple[str, int]]]:
    frequencies = np.arange(1, config.n_electrodes // 2 + 1, dtype=float)
    index = np.arange(config.n_electrodes, dtype=float)
    angles = (
        2.0 * np.pi * (index + config.electrode_coverage / 2.0) / config.n_electrodes
    )
    cosine = np.cos(angles[:, None] * frequencies[None, :])
    sine = np.sin(angles[:, None] * frequencies[None, :])
    patterns = np.column_stack((cosine, sine))
    patterns -= np.mean(patterns, axis=0, keepdims=True)
    labels = [
        *(("cosine", int(k)) for k in frequencies),
        *(("sine", int(k)) for k in frequencies),
    ]
    return patterns, labels


def add_arc(
    geometry: SplineGeometry,
    radius: float,
    theta_a: float,
    theta_b: float,
    boundary_name: str,
    subdivisions: int,
) -> None:
    delta = (theta_b - theta_a) / subdivisions
    for piece in range(subdivisions):
        theta_1 = theta_a + piece * delta
        theta_2 = theta_a + (piece + 1) * delta
        theta_mid = 0.5 * (theta_1 + theta_2)
        point_1 = geometry.AppendPoint(
            radius * np.cos(theta_1), radius * np.sin(theta_1)
        )
        point_mid = geometry.AppendPoint(
            radius * np.cos(theta_mid), radius * np.sin(theta_mid)
        )
        point_2 = geometry.AppendPoint(
            radius * np.cos(theta_2), radius * np.sin(theta_2)
        )
        geometry.Append(
            ["spline3", point_1, point_mid, point_2],
            bc=boundary_name,
            leftdomain=1,
            rightdomain=0,
        )


def make_mesh(config: Config) -> ngs.Mesh:
    geometry = SplineGeometry()
    for electrode in range(config.n_electrodes):
        segment_start = 2.0 * np.pi * electrode / config.n_electrodes
        segment_stop = 2.0 * np.pi * (electrode + 1) / config.n_electrodes
        electrode_stop = segment_start + config.electrode_coverage * (
            segment_stop - segment_start
        )
        add_arc(
            geometry,
            config.radius_m,
            segment_start,
            electrode_stop,
            f"electrode{electrode + 1}",
            config.arc_subdivisions,
        )
        add_arc(
            geometry,
            config.radius_m,
            electrode_stop,
            segment_stop,
            "insulating",
            config.arc_subdivisions,
        )
    geometry.SetMaterial(1, "domain")
    return ngs.Mesh(geometry.GenerateMesh(maxh=config.maxh_m))


def ngsolve_csr(matrix, shape: tuple[int, int]) -> csr_matrix:
    rows, columns, values = matrix.COO()
    return csr_matrix(
        (
            np.asarray(values, dtype=float),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=shape,
    )


def assemble_blocks(config: Config):
    mesh = make_mesh(config)
    space = ngs.H1(mesh, order=config.potential_order)
    trial, test = space.TnT()
    robin_form = ngs.BilinearForm(space)
    robin_form += ngs.SymbolicBFI(
        config.conductivity_s_per_m * ngs.grad(trial) * ngs.grad(test)
    )
    coupling = np.zeros((space.ndof, config.n_electrodes), dtype=float)
    electrode_diagonal = np.zeros(config.n_electrodes, dtype=float)
    for electrode in range(config.n_electrodes):
        boundary = mesh.Boundaries(f"electrode{electrode + 1}")
        robin_form += ngs.SymbolicBFI(
            trial * test / config.contact_impedance,
            definedon=boundary,
        )
    robin_form.Assemble()
    for electrode in range(config.n_electrodes):
        boundary = mesh.Boundaries(f"electrode{electrode + 1}")
        linear_form = ngs.LinearForm(space)
        linear_form += ngs.SymbolicLFI(
            test / config.contact_impedance,
            definedon=boundary,
        )
        linear_form.Assemble()
        coupling[:, electrode] = linear_form.vec.FV().NumPy()
        electrode_diagonal[electrode] = float(
            ngs.Integrate(1.0 / config.contact_impedance, mesh, definedon=boundary)
        )
    robin_matrix = ngsolve_csr(robin_form.mat, (space.ndof, space.ndof))
    return mesh, space, robin_matrix, coupling, electrode_diagonal


def solve(config: Config) -> tuple[list[dict[str, object]], dict[str, object]]:
    assemble_started = time.perf_counter()
    mesh, space, robin_matrix, coupling, diagonal = assemble_blocks(config)
    assemble_seconds = float(time.perf_counter() - assemble_started)
    currents, labels = current_patterns(config)
    electrode_count = config.n_electrodes

    classic_matrix = bmat(
        [
            [robin_matrix, -csc_matrix(coupling), None],
            [
                -csc_matrix(coupling.T),
                diags(diagonal, format="csc"),
                csc_matrix(np.ones((electrode_count, 1))),
            ],
            [
                None,
                csc_matrix(np.ones((1, electrode_count))),
                csc_matrix((1, 1)),
            ],
        ],
        format="csc",
    )
    classic_rhs = np.zeros(
        (space.ndof + electrode_count + 1, currents.shape[1]), dtype=float
    )
    classic_rhs[space.ndof : space.ndof + electrode_count, :] = currents
    classic_started = time.perf_counter()
    classic_solution = splu(classic_matrix).solve(classic_rhs)
    classic_seconds = float(time.perf_counter() - classic_started)
    classic_voltage = classic_solution[space.ndof : space.ndof + electrode_count, :]

    basis = helmert_basis(electrode_count)
    robin_started = time.perf_counter()
    robin_factor = splu(robin_matrix.tocsc())
    response_basis = robin_factor.solve(coupling @ basis)
    reduced_map = basis.T @ (diagonal[:, None] * basis - coupling.T @ response_basis)
    coefficients = np.linalg.solve(reduced_map, basis.T @ currents)
    robin_voltage = basis @ coefficients
    robin_seconds = float(time.perf_counter() - robin_started)

    rows: list[dict[str, object]] = []
    for formulation, voltage in (
        ("classic", classic_voltage),
        ("robin_transconductance", robin_voltage),
    ):
        for column, (mode, frequency) in enumerate(labels):
            current_norm = float(np.linalg.norm(currents[:, column]))
            voltage_norm = float(np.linalg.norm(voltage[:, column]))
            rows.append(
                {
                    "solver": "NGSolve",
                    "formulation": formulation,
                    "mode": mode,
                    "spatial_frequency": frequency,
                    "current_norm_a": current_norm,
                    "voltage_norm_v": voltage_norm,
                    "characteristic_resistance_ohm": voltage_norm / current_norm,
                }
            )
    voltage_relative_l2 = float(
        np.linalg.norm(robin_voltage - classic_voltage)
        / max(float(np.linalg.norm(classic_voltage)), np.finfo(float).eps)
    )
    response_residual = float(
        np.linalg.norm(robin_matrix @ response_basis - coupling @ basis)
        / max(float(np.linalg.norm(coupling @ basis)), np.finfo(float).eps)
    )
    metadata = {
        "solver": "NGSolve",
        "ngsolve_version": str(ngs.__version__),
        "physical_config": asdict(config),
        "discretization": {
            "vertices": int(mesh.nv),
            "elements": int(mesh.ne),
            "degrees_of_freedom": int(space.ndof),
            "element_family": "NGSolve H1",
            "potential_order": config.potential_order,
            "electrode_integration": "NGSolve boundary SymbolicBFI/SymbolicLFI",
        },
        "linear_solver": {
            "assembly": "NGSolve",
            "classic": "SciPy SuperLU augmented CEM",
            "robin": "SciPy SuperLU A_R plus NumPy dense reduced solve",
            "scalar_dtype": "float64",
        },
        "within_solver": {
            "electrode_voltage_relative_l2": voltage_relative_l2,
            "classic_voltage_balance_max_abs": float(
                np.max(np.abs(np.sum(classic_voltage, axis=0)))
            ),
            "robin_voltage_balance_max_abs": float(
                np.max(np.abs(np.sum(robin_voltage, axis=0)))
            ),
            "assembly_seconds": assemble_seconds,
            "classic_seconds": classic_seconds,
            "robin_seconds": robin_seconds,
        },
        "robin_diagnostics": {
            "rank": int(np.linalg.matrix_rank(reduced_map)),
            "condition_number": float(np.linalg.cond(reduced_map)),
            "response_relative_residual": response_residual,
        },
        "implementation_note": (
            "The FEM blocks are assembled by NGSolve from the author's Robin weak "
            "form; both algebraic formulations are solved from those identical blocks."
        ),
    }
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = solve(Config())
    csv_path = output_dir / "ngsolve_characteristic_resistance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_dir / "ngsolve_report.json"
    report_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"NGSolve CEM benchmark artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
