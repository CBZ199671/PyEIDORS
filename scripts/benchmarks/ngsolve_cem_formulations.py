#!/usr/bin/env python3
"""Fair NGSolve P1 float64 classic/Robin CEM benchmark on a shared mesh."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
import time

import ngsolve as ngs
from netgen.read_gmsh import ReadGmsh
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.cem_fair_common import (
    MESH_FINGERPRINT_SCHEMA,
    benchmark_preassembled_blocks,
    canonical_mesh_fingerprint,
)


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
    potential_order: int = 1
    timing_repeats: int = 11


def current_patterns(config: Config) -> tuple[np.ndarray, list[tuple[str, int]]]:
    frequencies = np.arange(1, config.n_electrodes // 2 + 1, dtype=float)
    index = np.arange(config.n_electrodes, dtype=float)
    angles = (
        2.0 * np.pi * (index + config.electrode_coverage / 2.0) / config.n_electrodes
    )
    cosine = np.cos(angles[:, None] * frequencies[None, :])
    sine = np.sin(angles[:, None] * frequencies[None, :])
    patterns = np.column_stack((cosine, sine)).astype(np.float64, copy=False)
    patterns -= np.mean(patterns, axis=0, keepdims=True)
    labels = [
        *(("cosine", int(k)) for k in frequencies),
        *(("sine", int(k)) for k in frequencies),
    ]
    return patterns, labels


def ngsolve_csr(matrix, shape: tuple[int, int]) -> csr_matrix:
    rows, columns, values = matrix.COO()
    return csr_matrix(
        (
            np.asarray(values, dtype=np.float64),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=shape,
    )


def _imported_mesh_arrays(mesh: ngs.Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(
        [tuple(point.p)[:2] for point in mesh.ngmesh.Points()], dtype=np.float64
    )
    cells = np.asarray(
        [
            [int(vertex.nr) - 1 for vertex in element.vertices]
            for element in mesh.ngmesh.Elements2D()
        ],
        dtype=np.int64,
    )
    boundary_names = tuple(mesh.GetBoundaries())
    edge_rows: list[tuple[int, int, int]] = []
    for segment in mesh.ngmesh.Elements1D():
        name = boundary_names[int(segment.index) - 1]
        match = re.fullmatch(r"electrode_(\d+)", str(name))
        label = int(match.group(1)) if match else 0
        vertices = [int(vertex.nr) - 1 for vertex in segment.vertices]
        edge_rows.append((vertices[0], vertices[1], label))
    return points, cells, np.asarray(edge_rows, dtype=np.int64)


def load_verified_mesh(
    mesh_path: Path,
    metadata_path: Path,
) -> tuple[ngs.Mesh, dict[str, object], float]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    mesh = ngs.Mesh(ReadGmsh(str(mesh_path)))
    import_seconds = float(time.perf_counter() - started)
    nodes, cells, edges = _imported_mesh_arrays(mesh)
    fingerprint = canonical_mesh_fingerprint(nodes, cells, edges)
    expected = str(metadata["mesh_fingerprint"])
    if fingerprint != expected:
        raise RuntimeError(
            "NGSolve common-mesh fingerprint mismatch: "
            f"imported={fingerprint}, expected={expected}"
        )
    expected_counts = (
        int(metadata["nodes"]),
        int(metadata["cells"]),
        int(metadata["boundary_edges"]),
    )
    imported_counts = (nodes.shape[0], cells.shape[0], edges.shape[0])
    if imported_counts != expected_counts:
        raise RuntimeError(
            f"NGSolve imported counts {imported_counts} != expected {expected_counts}"
        )
    verification = {
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "vertices": int(nodes.shape[0]),
        "cells": int(cells.shape[0]),
        "boundary_edges": int(edges.shape[0]),
    }
    return mesh, verification, import_seconds


def assemble_blocks(config: Config, mesh: ngs.Mesh):
    space = ngs.H1(mesh, order=config.potential_order)
    trial, test = space.TnT()
    robin_form = ngs.BilinearForm(space)
    robin_form += ngs.SymbolicBFI(
        config.conductivity_s_per_m * ngs.grad(trial) * ngs.grad(test)
    )
    coupling = np.zeros((space.ndof, config.n_electrodes), dtype=np.float64)
    electrode_diagonal = np.zeros(config.n_electrodes, dtype=np.float64)
    for electrode in range(config.n_electrodes):
        boundary = mesh.Boundaries(f"electrode_{electrode + 1}")
        robin_form += ngs.SymbolicBFI(
            trial * test / config.contact_impedance,
            definedon=boundary,
        )
    robin_form.Assemble()
    for electrode in range(config.n_electrodes):
        boundary = mesh.Boundaries(f"electrode_{electrode + 1}")
        linear_form = ngs.LinearForm(space)
        linear_form += ngs.SymbolicLFI(
            -test / config.contact_impedance,
            definedon=boundary,
        )
        linear_form.Assemble()
        coupling[:, electrode] = linear_form.vec.FV().NumPy()
        electrode_diagonal[electrode] = float(
            ngs.Integrate(1.0 / config.contact_impedance, mesh, definedon=boundary)
        )
    robin_matrix = ngsolve_csr(robin_form.mat, (space.ndof, space.ndof))
    return space, robin_matrix, coupling, np.diag(electrode_diagonal)


def solve(
    config: Config,
    mesh_path: Path,
    metadata_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    mesh, mesh_verification, mesh_import_seconds = load_verified_mesh(
        mesh_path, metadata_path
    )
    assemble_started = time.perf_counter()
    space, robin_matrix, coupling, diagonal = assemble_blocks(config, mesh)
    assembly_seconds = float(time.perf_counter() - assemble_started)
    currents, labels = current_patterns(config)
    timing, voltages, parity = benchmark_preassembled_blocks(
        robin_matrix,
        coupling,
        diagonal,
        currents,
        repeats=config.timing_repeats,
    )
    timing.update(
        {
            "mesh_import_seconds": mesh_import_seconds,
            "assembly_seconds": assembly_seconds,
        }
    )

    rows: list[dict[str, object]] = []
    for formulation, voltage in voltages.items():
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
    classic_voltage = voltages["classic"]
    robin_voltage = voltages["robin_transconductance"]
    metadata = {
        "solver": "NGSolve",
        "ngsolve_version": str(ngs.__version__),
        "physical_config": asdict(config),
        "discretization": {
            **mesh_verification,
            "degrees_of_freedom": int(space.ndof),
            "element_family": "NGSolve P1 H1 triangle",
            "potential_order": 1,
            "electrode_integration": "NGSolve boundary SymbolicBFI/SymbolicLFI",
            "mesh_import_verified": True,
            "common_mesh_role": "imported Gmsh 2.2",
        },
        "linear_solver": {
            "assembly": "NGSolve",
            "classic": "SciPy SuperLU augmented CEM",
            "robin": "SciPy SuperLU A_R plus SciPy dense reduced LU",
            "scalar_dtype": "float64",
        },
        "timing": timing,
        "within_solver": {
            **parity,
            "classic_voltage_balance_max_abs": float(
                np.max(np.abs(np.sum(classic_voltage, axis=0)))
            ),
            "robin_voltage_balance_max_abs": float(
                np.max(np.abs(np.sum(robin_voltage, axis=0)))
            ),
        },
        "raw_electrode_voltages": {
            formulation: np.asarray(voltage, dtype=np.float64).tolist()
            for formulation, voltage in voltages.items()
        },
        "implementation_note": (
            "NGSolve imports the canonical PyEIDORS P1 mesh and re-hashes its "
            "nodes/cells/tagged edges before assembling A_R/C/D. Both formulations "
            "use independent factor states and identical RHS matrices."
        ),
    }
    return rows, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mesh-metadata", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=11)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, metadata = solve(
        Config(timing_repeats=int(args.timing_repeats)),
        args.mesh.resolve(),
        args.mesh_metadata.resolve(),
    )
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
