#!/usr/bin/env python3
"""True-circle CEM h-refinement and cross-FEM continuum-accuracy suite."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for source_path in (ROOT, SRC):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from pyeidors.interop.geometry_exchange import (
    STANDARD_INTEROP_FORMAT,
    build_mesh_from_exchange_mat,
    save_exchange_mat,
)

from scripts.benchmarks.cem_continuum_reference import (
    ContinuumGeometry,
    certify_continuum_reference,
    continuum_current_patterns,
)
from scripts.benchmarks.cem_fair_common import (
    MESH_FINGERPRINT_SCHEMA,
    _as_csc,
    _classic_state,
    _robin_state,
    _solve_classic,
    _solve_robin,
    canonical_mesh_fingerprint,
    write_gmsh22,
)
from scripts.benchmarks.compare_cem_formulations import (
    BenchmarkConfig,
    _assemble_pyeidors_blocks,
    _extract_tagged_boundary_edges,
    _pattern_config,
    configure_fonts,
)


SUITE_SCHEMA = "cem-continuum-circle-suite-v1"
MESH_SCHEMA = "cem-true-circle-p1-h-sequence-v1"
METRIC_SCHEMA = "cem-continuum-accuracy-metrics-v1"
FORMULATIONS = ("classic", "robin_transconductance")
SOLVERS = ("PyEIDORS/DOLFINx", "NGSolve", "EIDORS")
GEOMETRY = ContinuumGeometry()


@dataclass(frozen=True)
class ContinuumCase:
    case_id: str
    label: str
    conductivity: float
    contact_impedance: float
    drive_skip: int
    drive_label: str


@dataclass(frozen=True)
class MeshLevel:
    level_id: str
    target_h: float


CASES = (
    ContinuumCase("C1", "baseline", 0.25, 1.0, 1, "adjacent"),
    ContinuumCase("C2", "low_z", 0.25, 0.125, 1, "adjacent"),
    ContinuumCase("C3", "high_z", 0.25, 8.0, 1, "adjacent"),
    ContinuumCase("C4", "high_sigma", 1.0, 1.0, 1, "adjacent"),
    ContinuumCase("C5", "skip4", 0.25, 1.0, 4, "skip-4"),
)
MESH_LEVELS = (
    MeshLevel("H0", 0.25),
    MeshLevel("H1", 0.125),
    MeshLevel("H2", 0.0625),
    MeshLevel("H3", 0.03125),
)
METRIC_FIELDS = (
    "case_id",
    "mesh_level_id",
    "solver",
    "formulation",
    "target_h",
    "h_max",
    "boundary_chord_max",
    "boundary_sagitta_max",
    "continuum_relative_l2",
    "continuum_max_abs",
    "reference_relative_uncertainty",
    "classic_robin_relative_l2",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    path.write_text(serialized + "\n", encoding="utf-8")


def _periodic_distance(angle: float, center: float) -> float:
    return (angle - center + math.pi) % (2.0 * math.pi) - math.pi


def _circle_breaks_and_labels(
    geometry: ContinuumGeometry,
) -> tuple[np.ndarray, list[int]]:
    half_width = geometry.electrode_angle / 2.0
    endpoints = sorted(
        {float((center - half_width) % (2.0 * math.pi)) for center in geometry.centers}
        | {
            float((center + half_width) % (2.0 * math.pi))
            for center in geometry.centers
        }
    )
    if len(endpoints) != 2 * geometry.n_electrodes:
        raise RuntimeError("true-circle electrode endpoints are not unique")
    labels: list[int] = []
    for index, start in enumerate(endpoints):
        stop = endpoints[(index + 1) % len(endpoints)]
        if index == len(endpoints) - 1:
            stop += 2.0 * math.pi
        midpoint = (start + stop) / 2.0
        label = 0
        for electrode, center in enumerate(geometry.centers, start=1):
            if abs(_periodic_distance(midpoint, float(center))) < half_width:
                label = electrode
                break
        labels.append(label)
    if set(labels) != set(range(geometry.n_electrodes + 1)):
        raise RuntimeError("true-circle arcs do not contain all electrodes and gaps")
    return np.asarray(endpoints, dtype=np.float64), labels


def _extract_linear_gmsh_arrays(
    gmsh: Any,
    arc_entities: list[int],
    arc_labels: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    coordinate_rows = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[:, :2]
    coordinate_by_tag = {
        int(tag): coordinate_rows[index]
        for index, tag in enumerate(np.asarray(node_tags, dtype=np.int64))
    }

    triangle_tags: list[np.ndarray] = []
    element_types, _, element_nodes = gmsh.model.mesh.getElements(2)
    for element_type, connectivity in zip(element_types, element_nodes, strict=True):
        _, dimension, order, node_count, _, _ = gmsh.model.mesh.getElementProperties(
            int(element_type)
        )
        if int(dimension) == 2 and int(order) == 1 and int(node_count) == 3:
            triangle_tags.append(
                np.asarray(connectivity, dtype=np.int64).reshape(-1, 3)
            )
    if len(triangle_tags) != 1:
        raise RuntimeError("Gmsh true-circle mesh must contain one P1 triangle block")
    tagged_edge_tags: list[tuple[int, int, int]] = []
    for entity, label in zip(arc_entities, arc_labels, strict=True):
        edge_types, _, edge_nodes = gmsh.model.mesh.getElements(1, int(entity))
        for element_type, connectivity in zip(edge_types, edge_nodes, strict=True):
            _, dimension, order, node_count, _, _ = (
                gmsh.model.mesh.getElementProperties(int(element_type))
            )
            if int(dimension) != 1 or int(order) != 1 or int(node_count) != 2:
                continue
            for first, second in np.asarray(connectivity, dtype=np.int64).reshape(
                -1, 2
            ):
                tagged_edge_tags.append((int(first), int(second), int(label)))
    if not tagged_edge_tags:
        raise RuntimeError("Gmsh true-circle mesh has no boundary edges")

    triangles_by_tag = triangle_tags[0]
    used_tags = sorted({int(value) for value in triangles_by_tag.reshape(-1)})
    tag_to_index = {tag: index for index, tag in enumerate(used_tags)}
    nodes = np.asarray([coordinate_by_tag[tag] for tag in used_tags], dtype=np.float64)
    cells = np.asarray(
        [[tag_to_index[int(value)] for value in row] for row in triangles_by_tag],
        dtype=np.int64,
    )
    signed_area_twice = (nodes[cells[:, 1], 0] - nodes[cells[:, 0], 0]) * (
        nodes[cells[:, 2], 1] - nodes[cells[:, 0], 1]
    ) - (nodes[cells[:, 2], 0] - nodes[cells[:, 0], 0]) * (
        nodes[cells[:, 1], 1] - nodes[cells[:, 0], 1]
    )
    if np.any(signed_area_twice == 0.0):
        raise RuntimeError("Gmsh true-circle mesh contains a zero-area triangle")
    negative = signed_area_twice < 0.0
    cells[negative, 1], cells[negative, 2] = (
        cells[negative, 2].copy(),
        cells[negative, 1].copy(),
    )
    edges = np.asarray(
        [
            (tag_to_index[first], tag_to_index[second], label)
            for first, second, label in tagged_edge_tags
        ],
        dtype=np.int64,
    )
    return nodes, cells, edges


def _electrode_node_arrays(
    tagged_edges: np.ndarray,
    n_electrodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    for electrode in range(1, int(n_electrodes) + 1):
        selected = tagged_edges[tagged_edges[:, 2] == electrode, :2]
        vertices = np.unique(selected.reshape(-1))
        if vertices.size < 2:
            raise RuntimeError(f"electrode {electrode} has fewer than two mesh nodes")
        rows.append(vertices)
    counts = np.asarray([row.size for row in rows], dtype=np.int64)
    padded = np.zeros((int(n_electrodes), int(np.max(counts))), dtype=np.int64)
    for index, row in enumerate(rows):
        padded[index, : row.size] = row + 1
    return padded, counts


def _mesh_size_metrics(
    nodes: np.ndarray,
    cells: np.ndarray,
    tagged_edges: np.ndarray,
    radius: float,
) -> dict[str, float]:
    triangle_edges = np.vstack(
        (
            cells[:, (0, 1)],
            cells[:, (1, 2)],
            cells[:, (2, 0)],
        )
    )
    triangle_lengths = np.linalg.norm(
        nodes[triangle_edges[:, 0]] - nodes[triangle_edges[:, 1]], axis=1
    )
    boundary_lengths = np.linalg.norm(
        nodes[tagged_edges[:, 0]] - nodes[tagged_edges[:, 1]], axis=1
    )
    boundary_vertices = np.unique(tagged_edges[:, :2])
    circle_error = np.max(
        np.abs(np.linalg.norm(nodes[boundary_vertices], axis=1) - float(radius))
    )
    chord_max = float(np.max(boundary_lengths))
    sagitta = float(radius - math.sqrt(max(radius * radius - chord_max**2 / 4.0, 0.0)))
    return {
        "h_max": float(np.max(triangle_lengths)),
        "boundary_chord_max": chord_max,
        "boundary_sagitta_max": sagitta,
        "circle_radius_max_abs_error": float(circle_error),
    }


def generate_true_circle_mesh(
    output_dir: Path,
    *,
    target_h: float,
    level_id: str,
    geometry: ContinuumGeometry = GEOMETRY,
) -> dict[str, Any]:
    """Generate one linear mesh from exact circular CAD arcs and export it."""

    try:
        import gmsh
    except (ImportError, OSError) as exc:  # pragma: no cover - environment gate
        raise RuntimeError(
            "Gmsh is required for the true-circle mesh sequence"
        ) from exc
    target = float(target_h)
    if not (math.isfinite(target) and target > 0.0):
        raise ValueError("target_h must be finite and positive")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    initialized_here = not bool(gmsh.isInitialized())
    if initialized_here:
        gmsh.initialize()
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"cem_continuum_{level_id.lower()}")
    breaks, arc_labels = _circle_breaks_and_labels(geometry)
    try:
        center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, target)
        points = [
            gmsh.model.geo.addPoint(
                geometry.radius * math.cos(float(angle)),
                geometry.radius * math.sin(float(angle)),
                0.0,
                target,
            )
            for angle in breaks
        ]
        arcs = [
            gmsh.model.geo.addCircleArc(
                points[index], center, points[(index + 1) % len(points)]
            )
            for index in range(len(points))
        ]
        loop = gmsh.model.geo.addCurveLoop(arcs)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        for electrode in range(1, geometry.n_electrodes + 1):
            entities = [
                arc
                for arc, label in zip(arcs, arc_labels, strict=True)
                if label == electrode
            ]
            group = gmsh.model.addPhysicalGroup(1, entities, electrode)
            gmsh.model.setPhysicalName(1, group, f"electrode_{electrode}")
        gaps = [arc for arc, label in zip(arcs, arc_labels, strict=True) if label == 0]
        gap_group = gmsh.model.addPhysicalGroup(1, gaps, geometry.n_electrodes + 1)
        gmsh.model.setPhysicalName(1, gap_group, "insulating")
        domain_group = gmsh.model.addPhysicalGroup(
            2, [surface], geometry.n_electrodes + 2
        )
        gmsh.model.setPhysicalName(2, domain_group, "domain")
        gmsh.option.setNumber("Mesh.MeshSizeMin", target)
        gmsh.option.setNumber("Mesh.MeshSizeMax", target)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), target)
        gmsh.model.mesh.generate(2)
        nodes, cells, tagged_edges = _extract_linear_gmsh_arrays(gmsh, arcs, arc_labels)
    finally:
        if initialized_here:
            gmsh.finalize()
        else:
            gmsh.clear()

    metrics = _mesh_size_metrics(nodes, cells, tagged_edges, radius=geometry.radius)
    if metrics["circle_radius_max_abs_error"] > 2e-12:
        raise RuntimeError("Gmsh boundary nodes left the true circle")
    fingerprint = canonical_mesh_fingerprint(nodes, cells, tagged_edges)
    electrode_nodes, electrode_counts = _electrode_node_arrays(
        tagged_edges, geometry.n_electrodes
    )
    msh_path = output_path / "cem_continuum_common_p1.msh"
    mat_path = output_path / "cem_continuum_common_p1.mat"
    json_path = output_path / "cem_continuum_common_p1.json"
    write_gmsh22(msh_path, nodes, cells, tagged_edges, geometry.n_electrodes)
    payload = {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "schema_version": 3,
        "index_base": 1,
        "dimension": 2,
        "cell_type": "triangle",
        "boundary_entity_type": "edge",
        "source_framework": "true_circle_gmsh_cad",
        "nodes": nodes,
        "elems": cells + 1,
        "boundary_facets": tagged_edges[:, :2] + 1,
        "boundary_edges": tagged_edges[:, :2] + 1,
        "tagged_boundary_edges": np.column_stack(
            (tagged_edges[:, :2] + 1, tagged_edges[:, 2])
        ),
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "n_elec": geometry.n_electrodes,
        "background": 0.25,
        "truth_elem_data": np.full(cells.shape[0], 0.25, dtype=np.float64),
        "contact_impedance": 1.0,
        "mesh_name": f"cem_continuum_{level_id.lower()}",
        "mesh_level": level_id,
        "scenario_name": "true_circle_continuum_mesh_master",
        "electrode_coverage": geometry.electrode_coverage,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        "suite_schema": SUITE_SCHEMA,
    }
    save_exchange_mat(mat_path, payload)
    metadata = {
        "suite_schema": SUITE_SCHEMA,
        "mesh_schema": MESH_SCHEMA,
        "level_id": str(level_id),
        "target_h": target,
        "radius": geometry.radius,
        "n_electrodes": geometry.n_electrodes,
        "electrode_coverage": geometry.electrode_coverage,
        "nodes": int(nodes.shape[0]),
        "cells": int(cells.shape[0]),
        "boundary_edges": int(tagged_edges.shape[0]),
        "electrode_edges": int(np.count_nonzero(tagged_edges[:, 2] > 0)),
        "potential_order": 1,
        "scalar_dtype": "float64",
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": fingerprint,
        **metrics,
        "msh": str(msh_path.resolve()),
        "mat": str(mat_path.resolve()),
        "metadata": str(json_path.resolve()),
    }
    _write_json(json_path, metadata)
    return {
        **metadata,
        "nodes_array": nodes,
        "cells_array": cells,
        "tagged_edges": tagged_edges,
        "electrode_nodes_array": electrode_nodes,
        "electrode_node_counts_array": electrode_counts,
        "msh_path": msh_path,
        "mat_path": mat_path,
        "metadata_path": json_path,
    }


def uncertainty_aware_ranking(
    errors: dict[str, float],
    *,
    reference_relative_uncertainty: float,
) -> dict[str, Any]:
    """Rank only when all adjacent uncertainty intervals are disjoint."""

    uncertainty = float(reference_relative_uncertainty)
    if not (math.isfinite(uncertainty) and uncertainty >= 0.0):
        raise ValueError(
            "reference_relative_uncertainty must be finite and nonnegative"
        )
    rows = []
    for solver, value in errors.items():
        error = float(value)
        if not (math.isfinite(error) and error >= 0.0):
            raise ValueError(f"invalid continuum error for {solver}")
        rows.append(
            {
                "solver": str(solver),
                "error": error,
                "lower": max(0.0, error - uncertainty),
                "upper": error + uncertainty,
            }
        )
    rows.sort(key=lambda row: (row["error"], row["solver"]))
    strict = all(
        first["upper"] < second["lower"]
        for first, second in zip(rows[:-1], rows[1:], strict=True)
    )
    best_upper = rows[0]["upper"] if rows else 0.0
    best_tie = [row["solver"] for row in rows if row["lower"] <= best_upper]
    return {
        "reference_relative_uncertainty": uncertainty,
        "strict_order_supported": bool(strict),
        "ordering": [row["solver"] for row in rows] if strict else None,
        "best_tie": best_tie,
        "intervals": rows,
    }


def shared_reference_sensitivity(
    candidates: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    *,
    primary_reference: str = "final_extrapolated",
) -> dict[str, Any]:
    """Compare solver outputs against correlated variants of one reference."""

    if not candidates:
        raise ValueError("shared-reference sensitivity requires solver candidates")
    if primary_reference not in references:
        raise ValueError(f"missing primary reference {primary_reference}")
    solver_names = [solver for solver in SOLVERS if solver in candidates]
    solver_names.extend(sorted(set(candidates) - set(solver_names)))
    candidate_arrays = {
        solver: np.asarray(candidates[solver], dtype=np.float64)
        for solver in solver_names
    }
    shape = next(iter(candidate_arrays.values())).shape
    if not shape or any(array.shape != shape for array in candidate_arrays.values()):
        raise ValueError("all shared-reference candidates must have one common shape")
    if any(not np.all(np.isfinite(array)) for array in candidate_arrays.values()):
        raise ValueError("shared-reference candidates must be finite")

    reference_arrays: dict[str, np.ndarray] = {}
    reference_rankings: dict[str, list[dict[str, Any]]] = {}
    for name, values in references.items():
        reference = np.asarray(values, dtype=np.float64)
        if reference.shape != shape or not np.all(np.isfinite(reference)):
            raise ValueError(f"invalid shared reference variant {name}")
        if np.linalg.norm(reference) == 0.0:
            raise ValueError(f"shared reference variant {name} has zero norm")
        reference_arrays[str(name)] = reference
        rows = [
            {
                "solver": solver,
                "continuum_relative_l2": _relative_l2(
                    candidate_arrays[solver],
                    reference,
                ),
            }
            for solver in solver_names
        ]
        rows.sort(key=lambda row: (row["continuum_relative_l2"], row["solver"]))
        reference_rankings[str(name)] = rows

    orderings = {
        name: [str(row["solver"]) for row in rows]
        for name, rows in reference_rankings.items()
    }
    ordering_values = [tuple(order) for order in orderings.values()]
    best_values = [order[0] for order in ordering_values]
    primary = reference_arrays[primary_reference]
    primary_norm_squared = float(np.vdot(primary, primary).real)
    pairwise: list[dict[str, Any]] = []
    for first_index, solver_a in enumerate(solver_names):
        for solver_b in solver_names[first_index + 1 :]:
            first = candidate_arrays[solver_a]
            second = candidate_arrays[solver_b]
            first_error = first - primary
            second_error = second - primary
            delta = second - first
            first_norm = float(np.linalg.norm(first))
            second_norm = float(np.linalg.norm(second))
            separation_denominator = 0.5 * (first_norm + second_norm)
            if separation_denominator == 0.0:
                raise ValueError("pairwise solver voltage norm is zero")
            dot = float(np.vdot(first_error, delta).real)
            error_norm = float(np.linalg.norm(first_error))
            delta_norm = float(np.linalg.norm(delta))
            alignment = (
                float(dot / (error_norm * delta_norm))
                if error_norm > 0.0 and delta_norm > 0.0
                else None
            )
            squared_error_difference = float(
                (
                    np.vdot(second_error, second_error).real
                    - np.vdot(first_error, first_error).real
                )
                / primary_norm_squared
            )
            cross_term = float(2.0 * dot / primary_norm_squared)
            delta_squared = float(delta_norm**2 / primary_norm_squared)
            pairwise.append(
                {
                    "solver_a": solver_a,
                    "solver_b": solver_b,
                    "symmetric_relative_voltage_separation": float(
                        delta_norm / separation_denominator
                    ),
                    "primary_reference_error_alignment_cosine": alignment,
                    "squared_error_difference_relative": squared_error_difference,
                    "cross_term_relative": cross_term,
                    "delta_squared_relative": delta_squared,
                    "squared_error_identity_closure_abs": float(
                        abs(squared_error_difference - cross_term - delta_squared)
                    ),
                }
            )
    return {
        "primary_reference": primary_reference,
        "reference_rankings": reference_rankings,
        "reference_orderings": orderings,
        "nominal_ordering": orderings[primary_reference],
        "ordering_stable_across_references": bool(ordering_values)
        and all(order == ordering_values[0] for order in ordering_values[1:]),
        "best_solver_stable_across_references": bool(best_values)
        and all(solver == best_values[0] for solver in best_values[1:]),
        "pairwise_solver_comparisons": pairwise,
    }


def generalized_richardson_triplet(
    coarse: np.ndarray,
    middle: np.ndarray,
    fine: np.ndarray,
    *,
    h_coarse: float,
    h_middle: float,
    h_fine: float,
) -> dict[str, Any] | None:
    """Estimate order and limit for three approximations at nonuniform h."""

    h0, h1, h2 = (float(h_coarse), float(h_middle), float(h_fine))
    if not (math.isfinite(h0) and h0 > h1 > h2 > 0.0):
        raise ValueError("Richardson h values must be finite, positive, and decreasing")
    u0 = np.asarray(coarse, dtype=np.float64)
    u1 = np.asarray(middle, dtype=np.float64)
    u2 = np.asarray(fine, dtype=np.float64)
    if u0.shape != u1.shape or u1.shape != u2.shape:
        raise ValueError("Richardson approximations must have matching shapes")
    difference_coarse = float(np.linalg.norm(u0 - u1))
    difference_fine = float(np.linalg.norm(u1 - u2))
    if not difference_coarse > difference_fine > 0.0:
        return None
    observed_ratio = difference_coarse / difference_fine

    def residual(order: float) -> float:
        numerator = h0**order - h1**order
        denominator = h1**order - h2**order
        return numerator / denominator - observed_ratio

    lower = 1e-3
    upper = 12.0
    lower_value = residual(lower)
    upper_value = residual(upper)
    if not (
        math.isfinite(lower_value)
        and math.isfinite(upper_value)
        and lower_value * upper_value <= 0.0
    ):
        return None
    order = float(brentq(residual, lower, upper, xtol=1e-13, rtol=1e-13))
    middle_power = h1**order
    fine_power = h2**order
    denominator = middle_power - fine_power
    if not (math.isfinite(denominator) and denominator > 0.0):
        return None
    extrapolated = (middle_power * u2 - fine_power * u1) / denominator
    return {
        "observed_order": order,
        "extrapolated": np.asarray(extrapolated, dtype=np.float64),
        "h_values": [h0, h1, h2],
        "difference_ratio": observed_ratio,
    }


def _case_level_directory(
    output_dir: Path,
    case: ContinuumCase,
    level: MeshLevel,
) -> Path:
    return output_dir / "cases" / f"{case.case_id}_{case.label}" / level.level_id


def _serializable_mesh_metadata(mesh: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "nodes_array",
        "cells_array",
        "tagged_edges",
        "electrode_nodes_array",
        "electrode_node_counts_array",
        "msh_path",
        "mat_path",
        "metadata_path",
    }
    return {key: value for key, value in mesh.items() if key not in excluded}


def _write_case_fixture(
    output_dir: Path,
    case: ContinuumCase,
    level: MeshLevel,
    mesh: dict[str, Any],
) -> dict[str, Any]:
    case_dir = _case_level_directory(output_dir, case, level)
    common_dir = case_dir / "common_mesh"
    common_dir.mkdir(parents=True, exist_ok=True)
    msh_path = common_dir / "cem_continuum_common_p1.msh"
    mat_path = common_dir / "cem_continuum_common_p1.mat"
    metadata_path = common_dir / "cem_continuum_common_p1.json"
    nodes = np.asarray(mesh["nodes_array"], dtype=np.float64)
    cells = np.asarray(mesh["cells_array"], dtype=np.int64)
    tagged_edges = np.asarray(mesh["tagged_edges"], dtype=np.int64)
    currents = continuum_current_patterns(
        n_electrodes=GEOMETRY.n_electrodes,
        drive_skip=case.drive_skip,
    )
    write_gmsh22(msh_path, nodes, cells, tagged_edges, GEOMETRY.n_electrodes)
    payload = {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "schema_version": 3,
        "index_base": 1,
        "dimension": 2,
        "cell_type": "triangle",
        "boundary_entity_type": "edge",
        "source_framework": "true_circle_gmsh_cad",
        "nodes": nodes,
        "elems": cells + 1,
        "boundary_facets": tagged_edges[:, :2] + 1,
        "boundary_edges": tagged_edges[:, :2] + 1,
        "tagged_boundary_edges": np.column_stack(
            (tagged_edges[:, :2] + 1, tagged_edges[:, 2])
        ),
        "electrode_nodes": np.asarray(mesh["electrode_nodes_array"], dtype=np.int64),
        "electrode_node_counts": np.asarray(
            mesh["electrode_node_counts_array"], dtype=np.int64
        ),
        "n_elec": GEOMETRY.n_electrodes,
        "background": case.conductivity,
        "truth_elem_data": np.full(cells.shape[0], case.conductivity),
        "contact_impedance": case.contact_impedance,
        "mesh_name": f"cem_continuum_{level.level_id.lower()}",
        "mesh_level": level.level_id,
        "scenario_name": case.label,
        "electrode_coverage": GEOMETRY.electrode_coverage,
        "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
        "mesh_fingerprint": mesh["mesh_fingerprint"],
        "suite_schema": SUITE_SCHEMA,
        "case_id": case.case_id,
        "current_patterns": currents,
        "drive_skip": case.drive_skip,
        "target_h": level.target_h,
        "h_max": mesh["h_max"],
        "boundary_chord_max": mesh["boundary_chord_max"],
        "boundary_sagitta_max": mesh["boundary_sagitta_max"],
    }
    save_exchange_mat(mat_path, payload)
    metadata = {
        "suite_schema": SUITE_SCHEMA,
        "mesh_schema": MESH_SCHEMA,
        "case": asdict(case),
        "case_id": case.case_id,
        "mesh_level_id": level.level_id,
        "target_h": level.target_h,
        **_serializable_mesh_metadata(mesh),
        "conductivity": case.conductivity,
        "contact_impedance": case.contact_impedance,
        "drive_skip": case.drive_skip,
        "current_patterns": currents.tolist(),
        "msh": str(msh_path.resolve()),
        "mat": str(mat_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "case_dir": str(case_dir.resolve()),
    }
    _write_json(metadata_path, metadata)
    return {
        **metadata,
        "msh_path": msh_path,
        "mat_path": mat_path,
        "metadata_path": metadata_path,
        "case_dir_path": case_dir,
    }


def _solve_preassembled(
    robin_matrix: Any,
    coupling: Any,
    electrode_matrix: Any,
    currents: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    a_r = _as_csc(robin_matrix)
    c = _as_csc(coupling)
    d = _as_csc(electrode_matrix)
    current_matrix = np.asarray(currents, dtype=np.float64)
    classic_potential, classic_voltage = _solve_classic(
        _classic_state(a_r, c, d), current_matrix
    )
    robin_potential, robin_voltage = _solve_robin(
        _robin_state(a_r, c, d), current_matrix
    )
    voltage_denominator = max(
        float(np.linalg.norm(classic_voltage)), np.finfo(np.float64).eps
    )
    potential_denominator = max(
        float(np.linalg.norm(classic_potential)), np.finfo(np.float64).eps
    )
    return (
        {
            "classic": np.asarray(classic_voltage, dtype=np.float64),
            "robin_transconductance": np.asarray(robin_voltage, dtype=np.float64),
        },
        {
            "electrode_voltage_relative_l2": float(
                np.linalg.norm(robin_voltage - classic_voltage) / voltage_denominator
            ),
            "body_potential_relative_l2": float(
                np.linalg.norm(robin_potential - classic_potential)
                / potential_denominator
            ),
        },
    )


def run_pyeidors_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Run controlled P1 float64 PyEIDORS on one true-circle fixture."""

    from dolfinx import fem
    from pyeidors.forward import EITForwardModel

    mesh, _ = build_mesh_from_exchange_mat(Path(fixture["mat_path"]))
    config = BenchmarkConfig(
        n_electrodes=GEOMETRY.n_electrodes,
        radius_m=GEOMETRY.radius,
        conductivity_s_per_m=float(fixture["conductivity"]),
        contact_impedance=float(fixture["contact_impedance"]),
        electrode_coverage=GEOMETRY.electrode_coverage,
        mesh_refinement=0,
        potential_order=1,
        timing_repeats=3,
    )
    model = EITForwardModel(
        n_elec=GEOMETRY.n_electrodes,
        pattern_config=_pattern_config(config),
        z=np.full(
            GEOMETRY.n_electrodes,
            float(fixture["contact_impedance"]),
            dtype=np.float64,
        ),
        mesh=mesh,
        potential_order=1,
        linear_backend="scipy",
    )
    if np.dtype(model.scalar_dtype) != np.dtype(np.float64):
        raise RuntimeError("continuum suite requires real float64 PyEIDORS")
    sigma = fem.Function(model.V_sigma)
    sigma.x.array[:] = float(fixture["conductivity"])
    loaded_edges = _extract_tagged_boundary_edges(mesh, list(model.electrode_tags))
    fingerprint = canonical_mesh_fingerprint(
        np.asarray(mesh.coordinates(), dtype=np.float64)[:, :2],
        np.asarray(mesh.cells(), dtype=np.int64),
        loaded_edges,
    )
    if fingerprint != fixture["mesh_fingerprint"]:
        raise RuntimeError("PyEIDORS true-circle common-mesh fingerprint mismatch")
    robin_matrix, coupling, electrode_matrix = _assemble_pyeidors_blocks(model, sigma)
    voltages, parity = _solve_preassembled(
        robin_matrix,
        coupling,
        electrode_matrix,
        np.asarray(fixture["current_patterns"], dtype=np.float64),
    )
    report = {
        "solver": "PyEIDORS/DOLFINx",
        "suite_schema": SUITE_SCHEMA,
        "case_id": fixture["case_id"],
        "mesh_level_id": fixture["mesh_level_id"],
        "physical_config": {
            "radius": GEOMETRY.radius,
            "n_electrodes": GEOMETRY.n_electrodes,
            "electrode_coverage": GEOMETRY.electrode_coverage,
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
            "scalar_dtype": "float64",
            "mesh_fingerprint_schema": MESH_FINGERPRINT_SCHEMA,
            "mesh_fingerprint": fingerprint,
            "mesh_import_verified": True,
            "target_h": fixture["target_h"],
            "h_max": fixture["h_max"],
            "boundary_chord_max": fixture["boundary_chord_max"],
            "boundary_sagitta_max": fixture["boundary_sagitta_max"],
        },
        "linear_solver": {
            "classic": "SciPy SuperLU augmented CEM",
            "robin": "SciPy SuperLU A_R plus dense reduced LU",
            "scalar_dtype": "float64",
        },
        "within_solver": parity,
        "raw_electrode_voltages": {
            name: values.tolist() for name, values in voltages.items()
        },
    }
    _write_json(Path(fixture["case_dir_path"]) / "pyeidors_report.json", report)
    return report


def prepare_suite(output_dir: Path) -> dict[str, Any]:
    """Certify references, generate common meshes, and run PyEIDORS."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    reference_records: dict[str, Any] = {}
    for case in CASES:
        certificate = certify_continuum_reference(
            conductivity=case.conductivity,
            contact_impedance=case.contact_impedance,
            drive_skip=case.drive_skip,
            geometry=GEOMETRY,
        )
        if not certificate["certified"]:
            raise RuntimeError(f"{case.case_id} continuum reference is not certified")
        reference_records[case.case_id] = certificate
        _write_json(output_path / "references" / f"{case.case_id}.json", certificate)

    meshes = [
        generate_true_circle_mesh(
            output_path / "mesh_sequence" / level.level_id,
            target_h=level.target_h,
            level_id=level.level_id,
            geometry=GEOMETRY,
        )
        for level in MESH_LEVELS
    ]
    for coarse, fine in zip(meshes[:-1], meshes[1:], strict=True):
        for metric in ("h_max", "boundary_chord_max", "boundary_sagitta_max"):
            if not float(fine[metric]) < float(coarse[metric]):
                raise RuntimeError(
                    f"true-circle mesh sequence did not decrease {metric}"
                )

    fixtures: list[dict[str, Any]] = []
    for case in CASES:
        for level, mesh in zip(MESH_LEVELS, meshes, strict=True):
            fixture = _write_case_fixture(output_path, case, level, mesh)
            run_pyeidors_fixture(fixture)
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
        "geometry": asdict(GEOMETRY),
        "cases": [asdict(case) for case in CASES],
        "mesh_levels": [_serializable_mesh_metadata(mesh) for mesh in meshes],
        "reference_summaries": {
            case_id: {
                key: value
                for key, value in certificate.items()
                if key
                not in {
                    "reference_voltages",
                    "previous_extrapolated_voltages",
                    "finest_raw_voltages",
                }
            }
            for case_id, certificate in reference_records.items()
        },
        "fixtures": fixtures,
    }
    _write_json(output_path / "suite_manifest.json", manifest)
    return manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_solver_report(report: dict[str, Any], fixture: dict[str, Any]) -> None:
    solver = str(report.get("solver", "unknown"))
    if solver not in SOLVERS:
        raise ValueError(f"unknown continuum solver {solver}")
    if report.get("suite_schema") != SUITE_SCHEMA:
        raise ValueError(f"{solver} continuum suite schema mismatch")
    if report.get("case_id") != fixture["case_id"]:
        raise ValueError(f"{solver} continuum case mismatch")
    if report.get("mesh_level_id") != fixture["mesh_level_id"]:
        raise ValueError(f"{solver} continuum mesh level mismatch")
    discretization = report.get("discretization", {})
    if discretization.get("mesh_fingerprint") != fixture["mesh_fingerprint"]:
        raise ValueError(f"{solver} continuum mesh fingerprint mismatch")
    if not bool(discretization.get("mesh_import_verified", False)):
        raise ValueError(f"{solver} did not verify continuum mesh import")
    if int(discretization.get("potential_order", -1)) != 1:
        raise ValueError(f"{solver} continuum controlled track must use P1")
    if report.get("linear_solver", {}).get("scalar_dtype") != "float64":
        raise ValueError(f"{solver} continuum controlled track must use float64")
    physical = report.get("physical_config", {})
    for key in ("conductivity", "contact_impedance", "drive_skip"):
        if float(physical.get(key, math.nan)) != float(fixture[key]):
            raise ValueError(f"{solver} continuum physical setting mismatch for {key}")
    for formulation in FORMULATIONS:
        voltage = np.asarray(
            report.get("raw_electrode_voltages", {}).get(formulation),
            dtype=np.float64,
        )
        if voltage.shape != (GEOMETRY.n_electrodes, GEOMETRY.n_electrodes):
            raise ValueError(f"{solver} {formulation} continuum voltage shape mismatch")
        if not np.all(np.isfinite(voltage)):
            raise ValueError(f"{solver} {formulation} continuum voltage is nonfinite")


def _relative_l2(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(candidate - reference) / denominator)


def _metric_record(
    report: dict[str, Any],
    fixture: dict[str, Any],
    reference: dict[str, Any],
    formulation: str,
) -> dict[str, Any]:
    raw = report["raw_electrode_voltages"]
    candidate = np.asarray(raw[formulation], dtype=np.float64)
    truth = np.asarray(reference["reference_voltages"], dtype=np.float64)
    classic = np.asarray(raw["classic"], dtype=np.float64)
    robin = np.asarray(raw["robin_transconductance"], dtype=np.float64)
    delta = candidate - truth
    per_rhs = [
        _relative_l2(candidate[:, column], truth[:, column])
        for column in range(truth.shape[1])
    ]
    return {
        "case_id": fixture["case_id"],
        "mesh_level_id": fixture["mesh_level_id"],
        "solver": report["solver"],
        "formulation": formulation,
        "target_h": float(fixture["target_h"]),
        "h_max": float(fixture["h_max"]),
        "boundary_chord_max": float(fixture["boundary_chord_max"]),
        "boundary_sagitta_max": float(fixture["boundary_sagitta_max"]),
        "continuum_relative_l2": _relative_l2(candidate, truth),
        "continuum_max_abs": float(np.max(np.abs(delta))),
        "per_rhs_continuum_relative_l2": per_rhs,
        "reference_relative_uncertainty": float(
            reference["reference_relative_uncertainty"]
        ),
        "classic_robin_relative_l2": _relative_l2(robin, classic),
    }


def _optional_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _convergence_summary(
    metrics: list[dict[str, Any]],
    voltages: dict[tuple[str, str, str, str], np.ndarray],
    references: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in CASES:
        truth = np.asarray(references[case.case_id]["reference_voltages"])
        for solver in SOLVERS:
            for formulation in FORMULATIONS:
                selected = sorted(
                    (
                        item
                        for item in metrics
                        if item["case_id"] == case.case_id
                        and item["solver"] == solver
                        and item["formulation"] == formulation
                    ),
                    key=lambda item: item["target_h"],
                    reverse=True,
                )
                if len(selected) != len(MESH_LEVELS):
                    raise ValueError(
                        f"missing continuum mesh level for {case.case_id}/{solver}/{formulation}"
                    )
                errors = np.asarray(
                    [item["continuum_relative_l2"] for item in selected],
                    dtype=np.float64,
                )
                h_values = np.asarray([item["h_max"] for item in selected])
                pairwise_orders = [
                    float(
                        math.log(errors[index] / errors[index + 1])
                        / math.log(h_values[index] / h_values[index + 1])
                    )
                    for index in range(len(errors) - 1)
                ]
                fitted_order = float(
                    np.polyfit(np.log(h_values[-3:]), np.log(errors[-3:]), 1)[0]
                )
                voltage_levels = [
                    voltages[(case.case_id, item["mesh_level_id"], solver, formulation)]
                    for item in selected
                ]
                previous_triplet = generalized_richardson_triplet(
                    voltage_levels[0],
                    voltage_levels[1],
                    voltage_levels[2],
                    h_coarse=float(h_values[0]),
                    h_middle=float(h_values[1]),
                    h_fine=float(h_values[2]),
                )
                last_triplet = generalized_richardson_triplet(
                    voltage_levels[1],
                    voltage_levels[2],
                    voltage_levels[3],
                    h_coarse=float(h_values[1]),
                    h_middle=float(h_values[2]),
                    h_fine=float(h_values[3]),
                )
                order_previous = (
                    float(previous_triplet["observed_order"])
                    if previous_triplet is not None
                    else math.nan
                )
                order_last = (
                    float(last_triplet["observed_order"])
                    if last_triplet is not None
                    else math.nan
                )
                extrapolated_error = math.nan
                extrapolation_change = math.nan
                if last_triplet is not None:
                    extrapolated = np.asarray(last_triplet["extrapolated"])
                    extrapolated_error = _relative_l2(extrapolated, truth)
                    if previous_triplet is not None:
                        previous = np.asarray(previous_triplet["extrapolated"])
                        extrapolation_change = _relative_l2(extrapolated, previous)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "solver": solver,
                        "formulation": formulation,
                        "pairwise_observed_orders": pairwise_orders,
                        "fitted_observed_order_last_three": fitted_order,
                        "fem_difference_order_previous": _optional_float(
                            order_previous
                        ),
                        "fem_difference_order_last": _optional_float(order_last),
                        "fem_extrapolated_continuum_relative_l2": _optional_float(
                            extrapolated_error
                        ),
                        "successive_fem_extrapolant_relative_change": _optional_float(
                            extrapolation_change
                        ),
                        "finest_continuum_relative_l2": float(errors[-1]),
                    }
                )
    return rows


def _write_metrics_csv(path: Path, metrics: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)


def _plot_convergence(metrics: list[dict[str, Any]], path: Path) -> None:
    configure_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    colors = {
        "PyEIDORS/DOLFINx": "#0072B2",
        "NGSolve": "#D55E00",
        "EIDORS": "#009E73",
    }
    styles = {
        "PyEIDORS/DOLFINx": ("-", "o", 5.0, 5),
        "NGSolve": ("--", "s", 7.0, 4),
        "EIDORS": (":", "^", 9.0, 3),
    }
    for axis, formulation in zip(axes, FORMULATIONS, strict=True):
        for solver in reversed(SOLVERS):
            x_values = []
            y_values = []
            for level in MESH_LEVELS:
                selected = [
                    item["continuum_relative_l2"]
                    for item in metrics
                    if item["mesh_level_id"] == level.level_id
                    and item["solver"] == solver
                    and item["formulation"] == formulation
                ]
                h_selected = [
                    item["h_max"]
                    for item in metrics
                    if item["mesh_level_id"] == level.level_id
                    and item["solver"] == solver
                    and item["formulation"] == formulation
                ]
                x_values.append(float(np.exp(np.mean(np.log(h_selected)))))
                y_values.append(float(np.exp(np.mean(np.log(selected)))))
            axis.loglog(
                x_values,
                y_values,
                linestyle=styles[solver][0],
                marker=styles[solver][1],
                markersize=styles[solver][2],
                markerfacecolor="white",
                markeredgewidth=1.4,
                linewidth=1.8,
                color=colors[solver],
                zorder=styles[solver][3],
                label=solver,
            )
        axis.set_title("Classic CEM" if formulation == "classic" else "Robin CEM")
        axis.set_xlabel(r"Actual $h_{max}$")
        axis.set_ylabel("Geometric-mean relative error")
        axis.grid(True, which="both", alpha=0.28)
        axis.invert_xaxis()
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "True-circle P1 CEM convergence to independent continuum reference "
        "(solver curves coincide)"
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_markdown_report(
    path: Path,
    report: dict[str, Any],
    *,
    plot_reference: str,
) -> None:
    references = report["references"]
    mesh_levels = report["mesh_levels"]
    metrics = report["metrics"]
    convergence = report["convergence"]
    finest = [
        item for item in metrics if item["mesh_level_id"] == MESH_LEVELS[-1].level_id
    ]
    strict_count = sum(
        int(value["strict_order_supported"])
        for value in report["finest_uncertainty_aware_rankings"].values()
    )
    ranking_count = len(report["finest_uncertainty_aware_rankings"])
    shared_results = report["finest_shared_reference_sensitivity"]
    stable_order_count = sum(
        int(value["ordering_stable_across_references"])
        for value in shared_results.values()
    )
    stable_best_count = sum(
        int(value["best_solver_stable_across_references"])
        for value in shared_results.values()
    )
    nominal_win_counts = {solver: 0 for solver in SOLVERS}
    pairwise_comparisons = []
    for value in shared_results.values():
        nominal_win_counts[value["nominal_ordering"][0]] += 1
        pairwise_comparisons.extend(value["pairwise_solver_comparisons"])
    pairwise_separations = [
        item["symmetric_relative_voltage_separation"] for item in pairwise_comparisons
    ]
    identity_closures = [
        item["squared_error_identity_closure_abs"] for item in pairwise_comparisons
    ]
    pyeidors_ngsolve_alignments = [
        item["primary_reference_error_alignment_cosine"]
        for item in pairwise_comparisons
        if item["solver_a"] == "PyEIDORS/DOLFINx"
        and item["solver_b"] == "NGSolve"
        and item["primary_reference_error_alignment_cosine"] is not None
    ]
    nominal_wins_text = "、".join(
        f"{solver} {count}/{ranking_count}"
        for solver, count in nominal_win_counts.items()
    )
    solver_spreads = []
    for case in CASES:
        for formulation in FORMULATIONS:
            selected = [
                item
                for item in finest
                if item["case_id"] == case.case_id
                and item["formulation"] == formulation
            ]
            errors = [item["continuum_relative_l2"] for item in selected]
            spread = max(errors) - min(errors)
            solver_spreads.append(spread)
    classic_robin_max = max(item["classic_robin_relative_l2"] for item in metrics)
    fitted_orders = [item["fitted_observed_order_last_three"] for item in convergence]
    monotone_count = 0
    for case in CASES:
        for solver in SOLVERS:
            for formulation in FORMULATIONS:
                selected = sorted(
                    (
                        item
                        for item in metrics
                        if item["case_id"] == case.case_id
                        and item["solver"] == solver
                        and item["formulation"] == formulation
                    ),
                    key=lambda item: item["target_h"],
                    reverse=True,
                )
                errors = [item["continuum_relative_l2"] for item in selected]
                monotone_count += int(
                    all(
                        fine < coarse
                        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
                    )
                )
    lines = [
        "# 真实圆域 CEM：三种 FEM 的连续问题精度与网格收敛",
        "",
        "## 技术结论",
        "",
        "本实验回答连续物理问题，而非固定有限维矩阵的舍入误差。三种求解器在每一级均导入同一份 P1 网格；Classic 与 Robin 使用相同物理量、原始 SI 电压和零均值规范。",
        "",
        f"以最终 Richardson 外推为共同参考时，最细网格 {ranking_count} 个 case/formulation 的名义第一名计数为：{nominal_wins_text}。但这只是连续总误差的名义顺序，不是离散线性求解精度顺序。",
        "",
        f"共享参考敏感性显示：前一外推、最终外推和最细原始参考三种共同参考下，完整顺序有 {stable_order_count}/{ranking_count} 组不变，第一名有 {stable_best_count}/{ranking_count} 组不变。旧的独立区间规则只有 {strict_count}/{ranking_count} 组区间完全分离；它可以保留为保守界，但不能作为共享参考下的唯一排名判据。",
        "",
        f"全部 {monotone_count}/30 条收敛序列随网格加密单调下降，最后三级拟合阶位于 {min(fitted_orders):.3f}–{max(fitted_orders):.3f}。最细网格三个求解器误差的最大绝对 spread 为 {max(solver_spreads):.3e}，而求解器成对电压分离仅为 {min(pairwise_separations):.3e}–{max(pairwise_separations):.3e}；Classic/Robin 最大内部差为 {classic_robin_max:.3e}。可观测总误差由共同 P1/直边圆域离散主导。",
        "",
        f"![真实圆域 P1 CEM 网格收敛]({plot_reference})",
        "",
        "图中横轴从粗网格向细网格推进，纵轴为五组物理 case 的误差几何平均；三条 solver 曲线在图示尺度上重合，说明共同离散误差远大于 solver 间差异。",
        "",
        "## 独立连续参考解为什么成立",
        "",
        "均匀圆域内部满足 $\\nabla\\cdot(\\sigma\\nabla u)=0$。把调和解按圆周 Fourier 模式展开后，第 $n$ 个非零模式从边界电势到法向电流密度的系数为 $\\sigma |n|/R$，所以逆映射为",
        "",
        "$$\\widehat u_n=\\frac{R}{\\sigma |n|}\\widehat q_n,\\qquad n\\ne0.$$",
        "该 Fourier 乘子就是圆域的解析 Neumann-to-Dirichlet 映射。总注入电流为零，因此 $q$ 的零 Fourier 模式为零；电势常数模式由 $\\sum_lU_l=0$ 唯一确定。数值系统只施加电极上的 $u+zq=U_l$、间隙上的 $q=0$ 和 $\\int_{E_l}q\\,ds=I_l$。这等价于真实圆域上的连续 CEM，而非某个内部三角网格的离散方程。",
        "",
        "边界电流使用周期 midpoint Fourier–Nyström 网格。电极端点与网格单元边界严格对齐，分辨率依次为 5120、10240、20480、40960；最后两组三网格经验阶 Richardson 外推之差定义参考不确定度。只有线性 residual、电流积分 residual、Robin residual、规范 residual 均不超过 $10^{-10}$ 且外推差不超过 $5\\times10^{-6}$ 时才认证。连续参考从不读取 PyEIDORS、NGSolve 或 EIDORS 的组装矩阵。",
        "",
        "### 参考认证",
        "",
        "| Case | 最后观测阶 | 外推相对不确定度 | 最大约束 residual | 认证 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for case in CASES:
        reference = references[case.case_id]
        lines.append(
            f"| {case.case_id} | {reference['observed_order_last']:.4f} | "
            f"{reference['reference_relative_uncertainty']:.3e} | "
            f"{reference['max_constraint_relative_residual']:.3e} | "
            f"{'是' if reference['certified'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 真实圆域公共网格",
            "",
            "网格由 Gmsh 真圆 CAD 圆弧生成，再导出为所有求解器共同使用的线性三角形。边界节点位于真实圆上，弦长和弦—圆弧 sagitta 随加密同步下降。",
            "",
            "| Level | target h | nodes | cells | actual hmax | boundary chord | sagitta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mesh in mesh_levels:
        lines.append(
            f"| {mesh['level_id']} | {mesh['target_h']:.5f} | {mesh['nodes']} | "
            f"{mesh['cells']} | {mesh['h_max']:.5e} | "
            f"{mesh['boundary_chord_max']:.5e} | {mesh['boundary_sagitta_max']:.5e} |"
        )
    lines.extend(
        [
            "",
            "## 最细网格相对连续参考误差",
            "",
            "| Case | Solver | Classic | Robin | Classic/Robin 内部差 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for case in CASES:
        for solver in SOLVERS:
            classic = next(
                item
                for item in finest
                if item["case_id"] == case.case_id
                and item["solver"] == solver
                and item["formulation"] == "classic"
            )
            robin = next(
                item
                for item in finest
                if item["case_id"] == case.case_id
                and item["solver"] == solver
                and item["formulation"] == "robin_transconductance"
            )
            lines.append(
                f"| {case.case_id} | {solver} | {classic['continuum_relative_l2']:.3e} | "
                f"{robin['continuum_relative_l2']:.3e} | "
                f"{classic['classic_robin_relative_l2']:.3e} |"
            )
    lines.extend(
        [
            "",
            "## 保守参考区间检查（不是唯一排名）",
            "",
            "这里把同一个参考不确定度分别加减到每个 solver 的误差上，忽略了误差之间共享同一参考所产生的相关性。因此它是保守边界检查，不是共享参考下的唯一结论。",
            "",
            "| Case | Formulation | 严格顺序成立 | 最优并列集合 |",
            "|---|---|:---:|---|",
        ]
    )
    for key, ranking in report["finest_uncertainty_aware_rankings"].items():
        case_id, formulation = key.split(":", maxsplit=1)
        tie_text = ", ".join(ranking["best_tie"])
        lines.append(
            f"| {case_id} | {formulation} | "
            f"{'是' if ranking['strict_order_supported'] else '否'} | {tie_text} |"
        )
    lines.extend(
        [
            "",
            "## 共享参考敏感性",
            "",
            "三个 solver 在同一行始终使用同一个参考变体，所以参考变化是相关扰动。表中顺序均按误差从小到大，比较前一 Richardson 外推、最终外推和最细 Nyström 原始解；若顺序随共同参考改变，说明 solver 间的微小总误差差不足以支撑稳定品牌排名。",
            "",
            "| Case | Formulation | 前一外推顺序 | 最终外推顺序 | 最细原始顺序 | 全序/第一名稳定 | 最大成对电压分离 |",
            "|---|---|---|---|---|:---:|---:|",
        ]
    )
    for key, sensitivity in shared_results.items():
        case_id, formulation = key.split(":", maxsplit=1)
        orders = sensitivity["reference_orderings"]
        maximum_separation = max(
            item["symmetric_relative_voltage_separation"]
            for item in sensitivity["pairwise_solver_comparisons"]
        )
        stable_text = (
            f"{'是' if sensitivity['ordering_stable_across_references'] else '否'} / "
            f"{'是' if sensitivity['best_solver_stable_across_references'] else '否'}"
        )
        lines.append(
            f"| {case_id} | {formulation} | "
            f"{' < '.join(orders['previous_extrapolated'])} | "
            f"{' < '.join(orders['final_extrapolated'])} | "
            f"{' < '.join(orders['finest_raw'])} | {stable_text} | "
            f"{maximum_separation:.3e} |"
        )
    lines.extend(
        [
            "",
            "## 离散误差、代数误差与偶然抵消",
            "",
            "设连续真值为 $U_*$，同一网格有限维方程的数学精确解为 $U_h^*$，solver 输出为 $\\widehat U_{h,s}$。定义共同的离散误差 $D_h=U_h^*-U_*$ 和 solver 的组装/代数误差 $a_{h,s}=\\widehat U_{h,s}-U_h^*$，则",
            "",
            "$$\\|\\widehat U_{h,s}-U_*\\|^2=\\|D_h\\|^2+2\\langle D_h,a_{h,s}\\rangle+\\|a_{h,s}\\|^2.$$",
            "",
            "连续总误差包含交叉项。即使某个 solver 的 $\\|a_{h,s}\\|$ 更大，只要方向与主导离散误差相反，也可能因抵消得到略小的 $\\|\\widehat U_{h,s}-U_*\\|$。因此连续总误差的名义第一名不能自动解释为线性代数更准确。",
            "",
            f"保存的最细网格输出还逐对验证了精确恒等式 $\\|e_b\\|^2-\\|e_a\\|^2=2\\langle e_a,\\delta\\rangle+\\|\\delta\\|^2$，归一化闭合误差最大为 {max(identity_closures):.3e}。以 PyEIDORS 为锚、NGSolve 为比较对象时，$e_{{Py}}$ 与 $U_{{NG}}-U_{{Py}}$ 的余弦范围为 [{min(pyeidors_ngsolve_alignments):.3f}, {max(pyeidors_ngsolve_alignments):.3f}]；负值直接显示了抵消方向。",
            "",
            "证据层级因此固定为：有理 QQ 实验负责回答同一有限维 P1 系统的离散组装/代数精度；本真实圆域实验负责回答当前网格输出到连续物理解的总误差。没有同网格高精度离散真值时，后者不能覆盖前者的代数精度结论。",
        ]
    )
    lines.extend(
        [
            "",
            "## 收敛与 FEM 外推",
            "",
            "| Case | Solver | Formulation | fitted p | finest error | FEM h→0 error |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in convergence:
        extrapolated = row["fem_extrapolated_continuum_relative_l2"]
        extrapolated_text = "n/a" if extrapolated is None else f"{extrapolated:.3e}"
        lines.append(
            f"| {row['case_id']} | {row['solver']} | {row['formulation']} | "
            f"{row['fitted_observed_order_last_three']:.3f} | "
            f"{row['finest_continuum_relative_l2']:.3e} | {extrapolated_text} |"
        )
    lines.extend(
        [
            "",
            "## 误差含义",
            "",
            "- `continuum_relative_l2`：$\\|U_h-U_{cont}\\|_F/\\|U_{cont}\\|_F$，包含 P1 场离散误差与直边三角形对圆域的几何误差。",
            "- `classic_robin_relative_l2`：同一求解器、同一网格内两种代数实现的差；它主要反映浮点舍入，不是连续 FEM 误差。",
            "- `fem_extrapolated_continuum_relative_l2`：由最后三级公共网格的电极电压独立 Richardson 外推后，与 Fourier–Nyström 连续参考比较。",
            "- `reference_relative_uncertainty`：相邻两次连续参考外推结果的差；对各 solver 独立加减它是保守界，共享参考敏感性才保留 solver 误差之间的相关结构。",
            "",
            "## 限制",
            "",
            "本套件针对均匀、各向同性二维圆域。非圆域、非均匀电导率和三维问题没有解析圆域 NtD 对角化，需要另一套独立高阶体积或边界参考。当前主比较固定为共同 P1/float64，因此回答当前离散输出的连续总误差，而不是各软件可用最高阶单元的能力上限。三种参考变体只检验参考敏感性，不能替代同网格高精度离散真值；代数精度结论应引用有理 QQ 实验。",
            "",
            "## 可复现产物",
            "",
            "- `cem_continuum_accuracy.json`：完整严格 JSON。",
            "- `cem_continuum_accuracy_metrics.csv`：逐 case/mesh/solver/formulation 误差。",
            "- `cem_continuum_convergence.png`：五组物理设置几何平均收敛图。",
            "- `suite_manifest.json`：真实圆网格、指纹、物理配置和参考认证入口。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_suite(output_dir: Path) -> dict[str, Any]:
    """Compare all three FEM solvers with the certified continuum reference."""

    output_path = Path(output_dir).resolve()
    manifest = _load_json(output_path / "suite_manifest.json")
    fixtures = manifest["fixtures"]
    references = {
        case.case_id: _load_json(output_path / "references" / f"{case.case_id}.json")
        for case in CASES
    }
    if not all(reference.get("certified") is True for reference in references.values()):
        raise RuntimeError(
            "all continuum references must be certified before comparison"
        )
    metrics: list[dict[str, Any]] = []
    voltage_lookup: dict[tuple[str, str, str, str], np.ndarray] = {}
    for fixture in fixtures:
        case_dir = Path(fixture["case_dir"])
        report_paths = (
            case_dir / "pyeidors_report.json",
            case_dir / "ngsolve_report.json",
            case_dir / "eidors_report.json",
        )
        reports = [_load_json(path) for path in report_paths]
        if {report.get("solver") for report in reports} != set(SOLVERS):
            raise ValueError("continuum fixture requires three distinct FEM solvers")
        for solver_report in reports:
            _validate_solver_report(solver_report, fixture)
            for formulation in FORMULATIONS:
                voltage = np.asarray(
                    solver_report["raw_electrode_voltages"][formulation],
                    dtype=np.float64,
                )
                voltage_lookup[
                    (
                        fixture["case_id"],
                        fixture["mesh_level_id"],
                        solver_report["solver"],
                        formulation,
                    )
                ] = voltage
                metrics.append(
                    _metric_record(
                        solver_report,
                        fixture,
                        references[fixture["case_id"]],
                        formulation,
                    )
                )
    convergence = _convergence_summary(metrics, voltage_lookup, references)
    finest_rankings: dict[str, Any] = {}
    shared_reference_results: dict[str, Any] = {}
    finest_level = MESH_LEVELS[-1].level_id
    for case in CASES:
        for formulation in FORMULATIONS:
            selected = {
                item["solver"]: item["continuum_relative_l2"]
                for item in metrics
                if item["case_id"] == case.case_id
                and item["mesh_level_id"] == finest_level
                and item["formulation"] == formulation
            }
            finest_rankings[f"{case.case_id}:{formulation}"] = (
                uncertainty_aware_ranking(
                    selected,
                    reference_relative_uncertainty=float(
                        references[case.case_id]["reference_relative_uncertainty"]
                    ),
                )
            )
            reference = references[case.case_id]
            candidates = {
                solver: voltage_lookup[
                    (case.case_id, finest_level, solver, formulation)
                ]
                for solver in SOLVERS
            }
            shared_reference_results[f"{case.case_id}:{formulation}"] = (
                shared_reference_sensitivity(
                    candidates,
                    {
                        "previous_extrapolated": np.asarray(
                            reference["previous_extrapolated_voltages"]
                        ),
                        "final_extrapolated": np.asarray(
                            reference["reference_voltages"]
                        ),
                        "finest_raw": np.asarray(reference["finest_raw_voltages"]),
                    },
                )
            )
    mesh_levels = manifest["mesh_levels"]
    csv_path = output_path / "cem_continuum_accuracy_metrics.csv"
    plot_path = output_path / "cem_continuum_convergence.png"
    report_path = output_path / "cem_continuum_accuracy_report.md"
    _write_metrics_csv(csv_path, metrics)
    _plot_convergence(metrics, plot_path)
    report = {
        "suite_schema": SUITE_SCHEMA,
        "metric_schema": METRIC_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "continuous homogeneous complete-electrode problem on one true unit disk",
        "geometry": asdict(GEOMETRY),
        "cases": [asdict(case) for case in CASES],
        "mesh_levels": mesh_levels,
        "reference_method": {
            "operator": "analytic disk Neumann-to-Dirichlet Fourier multiplier R/(sigma*abs(n))",
            "boundary_method": "midpoint Fourier-Nystrom",
            "extrapolation": "four doubled grids, empirical-order Richardson",
            "uses_interior_fem_mesh": False,
            "uses_candidate_solver_matrix": False,
        },
        "references": references,
        "metrics": metrics,
        "convergence": convergence,
        "finest_uncertainty_aware_rankings": finest_rankings,
        "finest_shared_reference_sensitivity": shared_reference_results,
        "accuracy_evidence_hierarchy": {
            "discrete_algebraic_accuracy": (
                "owned by the exact rational QQ suite on a common finite P1 system"
            ),
            "continuum_total_accuracy": (
                "owned by this true-circle suite against the independent disk reference"
            ),
            "non_override_rule": (
                "continuum total-error ordering does not override exact discrete "
                "algebraic ordering without a same-mesh high-precision discrete target"
            ),
            "decomposition": ("||D_h+a_hs||^2=||D_h||^2+2<D_h,a_hs>+||a_hs||^2"),
        },
        "artifacts": {
            "metrics_csv": csv_path.name,
            "plot": plot_path.name,
            "markdown_report": report_path.name,
        },
    }
    json.dumps(report, allow_nan=False)
    _write_json(output_path / "cem_continuum_accuracy.json", report)
    _write_markdown_report(
        report_path,
        report,
        plot_reference=plot_path.name,
    )
    _write_markdown_report(
        ROOT / "docs" / "benchmarks" / report_path.name,
        report,
        plot_reference=(
            "../../output/cem_continuum_accuracy/cem_continuum_convergence.png"
        ),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "compare"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--output-dir",
            type=Path,
            default=ROOT / "output" / "cem_continuum_accuracy",
        )
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_suite(args.output_dir)
        print(
            f"Prepared {len(manifest['fixtures'])} true-circle FEM fixtures in "
            f"{Path(args.output_dir).resolve()}"
        )
        return 0
    report = compare_suite(args.output_dir)
    print(
        f"Compared {len(report['metrics'])} continuum accuracy records in "
        f"{Path(args.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
