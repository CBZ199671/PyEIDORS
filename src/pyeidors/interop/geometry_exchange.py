"""Standardized geometry exchange helpers for EIDORS <-> PyEIDORS interop."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

STANDARD_INTEROP_FORMAT = "eidors_pyeidors_bridge_v1"

REQUIRED_EXCHANGE_FIELDS = {
    "exchange_format",
    "source_framework",
    "nodes",
    "elems",
    "boundary_edges",
    "electrode_nodes",
    "electrode_node_counts",
    "n_elec",
    "background",
    "truth_elem_data",
    "contact_impedance",
    "mesh_name",
    "mesh_level",
    "scenario_name",
}


def load_forward_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load homogeneous / phantom / difference voltage data from CSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in forward CSV: {path}")

    baseline = np.asarray([float(row["meas_homogeneous"]) for row in rows], dtype=float)
    phantom = np.asarray([float(row["meas_phantom"]) for row in rows], dtype=float)
    if "difference" in reader.fieldnames:
        target_diff = np.asarray([float(row["difference"]) for row in rows], dtype=float)
    else:
        target_diff = phantom - baseline
    return baseline, phantom, target_diff


def export_forward_csv(path: Path, baseline: np.ndarray, phantom: np.ndarray) -> None:
    """Persist homogeneous / phantom / difference voltages to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["meas_homogeneous", "meas_phantom", "difference"],
        )
        writer.writeheader()
        for vh, vi in zip(baseline, phantom):
            writer.writerow(
                {
                    "meas_homogeneous": float(vh),
                    "meas_phantom": float(vi),
                    "difference": float(vi - vh),
                }
            )


def _to_python_dict(mat_obj: Any) -> dict[str, Any]:
    fields = getattr(mat_obj, "_fieldnames", None)
    if not fields:
        return {}
    return {field: getattr(mat_obj, field) for field in fields}


def _infer_electrode_tags(mesh) -> list[int]:
    electrode_tags = list(getattr(mesh, "electrode_tags", []))
    if electrode_tags:
        return [int(tag) for tag in electrode_tags]

    association_table = getattr(mesh, "association_table", {}) or {}
    inferred = []
    for key, value in association_table.items():
        if isinstance(key, str) and key.lower().startswith("electrode"):
            inferred.append(int(value))
    if inferred:
        return sorted(set(inferred))
    return []


def build_electrode_arrays(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Collect padded electrode node ids from a marked PyEIDORS mesh."""
    from fenics import facets

    boundary_markers = getattr(mesh, "boundaries_mf", None)
    electrode_tags = _infer_electrode_tags(mesh)
    if boundary_markers is None or not electrode_tags:
        raise ValueError("Mesh is missing boundary markers or electrode tags")

    coords = np.asarray(mesh.coordinates(), dtype=float)
    center = np.mean(coords, axis=0)

    electrode_lists: list[np.ndarray] = []
    max_nodes = 0
    for tag in electrode_tags:
        node_ids: set[int] = set()
        for facet in facets(mesh):
            if int(boundary_markers[facet]) != int(tag):
                continue
            node_ids.update(int(v) for v in facet.entities(0))
        if not node_ids:
            raise ValueError(f"No boundary nodes found for electrode tag {tag}")
        ordered = np.array(sorted(node_ids), dtype=np.int64)
        local = coords[ordered] - center
        angles = np.arctan2(local[:, 1], local[:, 0])
        ordered = ordered[np.argsort(angles)]
        electrode_lists.append(ordered + 1)
        max_nodes = max(max_nodes, len(ordered))

    electrode_nodes = np.zeros((len(electrode_lists), max_nodes), dtype=np.int64)
    electrode_counts = np.zeros(len(electrode_lists), dtype=np.int64)
    for idx, nodes in enumerate(electrode_lists):
        electrode_nodes[idx, : len(nodes)] = nodes
        electrode_counts[idx] = len(nodes)
    return electrode_nodes, electrode_counts


def build_boundary_edges(mesh) -> np.ndarray:
    """Collect boundary edges from a marked PyEIDORS mesh."""
    from fenics import facets

    boundary_markers = getattr(mesh, "boundaries_mf", None)
    if boundary_markers is None:
        raise ValueError("Mesh is missing boundary markers")

    edges: list[np.ndarray] = []
    for facet in facets(mesh):
        if int(boundary_markers[facet]) == 0:
            continue
        edge = np.asarray(facet.entities(0), dtype=np.int64).reshape(-1) + 1
        if len(edge) != 2:
            continue
        edges.append(edge)
    if not edges:
        raise ValueError("No boundary edges found")
    return np.vstack(edges)


def validate_exchange_payload(payload: dict[str, Any]) -> None:
    """Validate that a payload conforms to the standard interop fields."""
    missing = sorted(REQUIRED_EXCHANGE_FIELDS.difference(payload))
    if missing:
        raise ValueError(f"Exchange payload is missing required fields: {', '.join(missing)}")

    exchange_format = str(np.asarray(payload["exchange_format"]).reshape(-1)[0])
    if exchange_format != STANDARD_INTEROP_FORMAT:
        raise ValueError(
            f"Unsupported exchange format {exchange_format!r}; expected {STANDARD_INTEROP_FORMAT!r}"
        )


def save_exchange_mat(path: Path, payload: dict[str, Any]) -> None:
    """Persist a validated standard payload to MATLAB .mat format."""
    validate_exchange_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, payload)


def _load_standard_electrode_node_lists(payload: dict[str, Any]) -> list[np.ndarray] | None:
    if "electrode_nodes" not in payload:
        return None
    electrode_nodes = np.asarray(payload["electrode_nodes"], dtype=np.int64)
    counts = np.asarray(payload["electrode_node_counts"], dtype=np.int64).reshape(-1)
    node_lists = []
    for row, count in zip(electrode_nodes, counts):
        node_lists.append(np.asarray(row[: int(count)], dtype=np.int64).reshape(-1) - 1)
    return node_lists


def build_mesh_from_exchange_mat(path: Path):
    """Build a FEniCS mesh from a standard or legacy EIDORS/PyEIDORS .mat payload."""
    from fenics import Mesh, MeshEditor, MeshFunction, cells, facets

    payload = loadmat(path, squeeze_me=True, struct_as_record=False)

    nodes = np.asarray(payload["nodes"], dtype=float)
    elems = np.asarray(payload["elems"], dtype=np.int64) - 1
    exchange_format = (
        str(np.asarray(payload.get("exchange_format", "")).reshape(-1)[0])
        if "exchange_format" in payload
        else ""
    )
    standard_electrodes = _load_standard_electrode_node_lists(payload)
    legacy_electrodes = np.asarray(payload["electrodes"]).reshape(-1) if "electrodes" in payload else None

    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "triangle", 2, 2)
    editor.init_vertices(len(nodes))
    editor.init_cells(len(elems))
    for idx, coord in enumerate(nodes):
        editor.add_vertex(idx, coord[:2])
    for idx, elem in enumerate(elems):
        editor.add_cell(idx, elem.astype(np.uintp))
    editor.close()
    mesh.init(1)
    mesh.init(1, 0)
    mesh.init(2, 1)

    boundary_markers = MeshFunction("size_t", mesh, 1, 0)
    association_table: dict[str, int] = {}
    electrode_point_sets: list[set[tuple[float, float]]] = []
    electrode_node_lists: list[np.ndarray]
    if standard_electrodes is not None:
        electrode_node_lists = standard_electrodes
    elif legacy_electrodes is not None:
        electrode_node_lists = []
        for electrode in legacy_electrodes:
            elec_dict = _to_python_dict(electrode)
            elec_nodes = np.asarray(elec_dict["nodes"], dtype=np.int64).reshape(-1) - 1
            electrode_node_lists.append(elec_nodes)
    else:
        raise ValueError(f"Mesh exchange file {path} does not contain electrode definitions")

    for elec_idx, elec_nodes in enumerate(electrode_node_lists, start=1):
        point_set = {
            (float(nodes[node_id, 0]), float(nodes[node_id, 1]))
            for node_id in elec_nodes
        }
        electrode_point_sets.append(point_set)
        association_table[f"electrode_{elec_idx}"] = elec_idx + 1

    gap_tag = len(electrode_point_sets) + 2
    association_table["gaps"] = gap_tag

    for facet in facets(mesh):
        if sum(1 for _ in cells(facet)) != 1:
            continue
        vertex_points = {
            (float(mesh.coordinates()[int(v)][0]), float(mesh.coordinates()[int(v)][1]))
            for v in facet.entities(0)
        }
        marker = gap_tag
        for elec_idx, point_set in enumerate(electrode_point_sets, start=1):
            if vertex_points.issubset(point_set):
                marker = elec_idx + 1
                break
        boundary_markers[facet] = marker

    mesh.boundaries_mf = boundary_markers
    mesh.association_table = association_table
    mesh.electrode_tags = [
        association_table[f"electrode_{idx}"] for idx in range(1, len(electrode_point_sets) + 1)
    ]
    mesh.n_electrodes = len(electrode_point_sets)
    mesh.radius = float(np.max(np.linalg.norm(nodes[:, :2], axis=1)))
    if "mesh_name" in payload:
        mesh.mesh_name = str(np.asarray(payload["mesh_name"]).reshape(-1)[0])
    else:
        mesh.mesh_name = path.stem
    mesh.exchange_format = exchange_format or STANDARD_INTEROP_FORMAT
    return mesh, payload
