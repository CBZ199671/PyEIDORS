#!/usr/bin/python3
"""Native GetFEM P1 Robin--transconductance CEM adapter."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import getfem
import numpy as np


REPORT_SCHEMA = "cem-multifem-accuracy-v1"


def _helmert_basis(count: int) -> np.ndarray:
    basis = np.zeros((count, count - 1), dtype=np.float64)
    for column in range(1, count):
        scale = np.sqrt(float(column * (column + 1)))
        basis[:column, column - 1] = 1.0 / scale
        basis[column, column - 1] = -float(column) / scale
    return basis


def _native_solve(matrix: Any, rhs: np.ndarray) -> np.ndarray:
    """Solve each RHS with GetFEM's MUMPS binding."""

    values = np.asarray(rhs, dtype=np.float64)
    if values.ndim == 1:
        solution = getfem.linsolve_mumps(matrix, values)
        return np.asarray(solution, dtype=np.float64).reshape(-1)
    result = np.empty_like(values)
    for column in range(values.shape[1]):
        solution = getfem.linsolve_mumps(matrix, values[:, column])
        result[:, column] = np.asarray(solution, dtype=np.float64).reshape(-1)
    return result


def _point_cells(mesh: Any) -> list[list[int]]:
    convex_ids = np.asarray(mesh.cvid(), dtype=np.int64).reshape(-1)
    point_ids, offsets = mesh.pid_from_cvid(convex_ids)
    point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    cells: list[list[int]] = []
    for index in range(convex_ids.size):
        cell = point_ids[offsets[index] : offsets[index + 1]].tolist()
        if len(cell) != 3:
            raise ValueError("GetFEM imported a non-triangle volume element")
        cells.append(cell)
    return cells


def _tagged_edges(mesh: Any, electrode_count: int) -> list[list[int]]:
    edges: list[list[int]] = []
    for region in range(1, electrode_count + 2):
        faces = np.asarray(mesh.region(region), dtype=np.int64)
        if faces.size == 0:
            continue
        faces = faces.reshape(2, -1)
        label = region if region <= electrode_count else 0
        for column in range(faces.shape[1]):
            face = faces[:, column : column + 1]
            endpoints = np.asarray(mesh.pid_in_faces(face), dtype=np.int64).reshape(-1)
            if endpoints.size != 2:
                raise ValueError("GetFEM boundary region contains a non-edge face")
            edges.append([int(endpoints[0]), int(endpoints[1]), int(label)])
    return edges


def _point_to_dof(mesh: Any, mesh_fem: Any) -> np.ndarray:
    point_ids = np.asarray(mesh.pid(), dtype=np.int64).reshape(-1)
    if not np.array_equal(point_ids, np.arange(point_ids.size, dtype=np.int64)):
        raise ValueError("GetFEM imported non-contiguous point identifiers")
    points = np.asarray(mesh.pts(), dtype=np.float64).T[:, :2]
    dof_points = np.asarray(mesh_fem.basic_dof_nodes(), dtype=np.float64).T[:, :2]
    if dof_points.shape != points.shape:
        raise ValueError("GetFEM P1 DOF count differs from point count")
    mapping = np.full(points.shape[0], -1, dtype=np.int64)
    for point, coordinate in enumerate(points):
        distances = np.max(np.abs(dof_points - coordinate), axis=1)
        matches = np.flatnonzero(distances <= 1.0e-13)
        if matches.size != 1:
            raise ValueError("GetFEM P1 point-to-DOF map is not bijective")
        mapping[point] = int(matches[0])
    if np.unique(mapping).size != mapping.size:
        raise ValueError("GetFEM P1 point-to-DOF map contains duplicates")
    return mapping


def solve(config: dict[str, Any]) -> dict[str, Any]:
    mesh_path = str(Path(config["mesh"]).resolve())
    conductivity = float(config["conductivity"])
    contact_impedance = np.asarray(config["contact_impedance"], dtype=np.float64)
    currents = np.asarray(config["currents"], dtype=np.float64)
    if conductivity <= 0.0 or np.any(contact_impedance <= 0.0):
        raise ValueError("conductivity and contact impedances must be positive")
    if currents.shape[0] != contact_impedance.size:
        raise ValueError("current/electrode shape mismatch")
    if not np.allclose(np.sum(currents, axis=0), 0.0, atol=1.0e-13, rtol=0.0):
        raise ValueError("current patterns must be balanced")

    mesh = getfem.Mesh("import", "gmsh", mesh_path)
    mesh_fem = getfem.MeshFem(mesh, 1)
    mesh_fem.set_classical_fem(1)
    mesh_im = getfem.MeshIm(mesh, 4)
    node_count = int(mesh_fem.nbdof())
    electrode_count = int(contact_impedance.size)
    zero_state = np.zeros(node_count, dtype=np.float64)

    stiffness = getfem.asm_generic(
        mesh_im,
        2,
        f"{conductivity:.17g}*Grad_Test2_u.Grad_Test_u",
        -1,
        "u",
        1,
        mesh_fem,
        zero_state,
    )
    boundary_mass = getfem.Spmat("empty", node_count)
    coupling = np.zeros((node_count, electrode_count), dtype=np.float64)
    electrode_matrix = np.zeros((electrode_count, electrode_count), dtype=np.float64)
    for electrode, impedance in enumerate(contact_impedance, start=1):
        mass = getfem.asm_mass_matrix(mesh_im, mesh_fem, mesh_fem, electrode)
        mass.scale(1.0 / float(impedance))
        boundary_mass = getfem.Spmat("add", boundary_mass, mass)
        column = getfem.asm_generic(
            mesh_im,
            1,
            f"{1.0 / float(impedance):.17g}*Test_u",
            electrode,
            "u",
            1,
            mesh_fem,
            zero_state,
        )
        coupling[:, electrode - 1] = np.asarray(column, dtype=np.float64)
        electrode_matrix[electrode - 1, electrode - 1] = float(
            getfem.asm_generic(
                mesh_im,
                0,
                f"{1.0 / float(impedance):.17g}",
                electrode,
            )
        )

    robin_matrix = getfem.Spmat("add", stiffness, boundary_mass)
    response = _native_solve(robin_matrix, coupling)
    transconductance = electrode_matrix - coupling.T @ response
    basis = _helmert_basis(electrode_count)
    reduced_map = basis.T @ transconductance @ basis
    reduced_sparse = getfem.Spmat("empty", electrode_count - 1)
    reduced_indices = np.arange(electrode_count - 1, dtype=np.int32)
    reduced_sparse.assign(reduced_indices, reduced_indices, reduced_map)
    coefficients = _native_solve(reduced_sparse, basis.T @ currents)
    electrode_voltage = basis @ coefficients
    body_potential = _native_solve(robin_matrix, coupling @ electrode_voltage)

    point_to_dof = _point_to_dof(mesh, mesh_fem)
    points = np.asarray(mesh.pts(), dtype=np.float64).T[:, :2]
    stiffness_dense = np.asarray(stiffness.full(), dtype=np.float64)
    boundary_dense = np.asarray(boundary_mass.full(), dtype=np.float64)
    robin_dense = np.asarray(robin_matrix.full(), dtype=np.float64)
    point_index = np.ix_(point_to_dof, point_to_dof)
    return {
        "schema": REPORT_SCHEMA,
        "solver": "GetFEM",
        "formulation": "robin_transconductance",
        "implementation": {
            "native_assembly": True,
            "framework_version": "5.3",
            "body_solver": "GetFEM-MUMPS",
            "electrode_solver": "GetFEM-MUMPS",
        },
        "discretization": {
            "mesh_fingerprint": config["mesh_fingerprint"],
            "mesh_import_verified": True,
            "potential_order": 1,
            "geometry_order": 1,
            "scalar_dtype": "float64",
            "imported_nodes": points.tolist(),
            "imported_cells_zero_based": _point_cells(mesh),
            "imported_tagged_boundary_edges_zero_based": _tagged_edges(
                mesh, electrode_count
            ),
        },
        "physical_config": {
            "conductivity": conductivity,
            "contact_impedance": contact_impedance.tolist(),
            "currents": currents.tolist(),
        },
        "blocks": {
            "K": stiffness_dense[point_index].tolist(),
            "B": boundary_dense[point_index].tolist(),
            "C_plus": coupling[point_to_dof, :].tolist(),
            "D": electrode_matrix.tolist(),
            "A_R": robin_dense[point_index].tolist(),
        },
        "solution": {
            "T": transconductance.tolist(),
            "reduced_map": reduced_map.tolist(),
            "body_potential": body_potential[point_to_dof, :].tolist(),
            "electrode_voltage": electrode_voltage.tolist(),
        },
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: getfem_cem_robin.py CONFIG_JSON OUTPUT_JSON")
    config_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = solve(config)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
