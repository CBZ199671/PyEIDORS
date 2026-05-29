"""Standardized geometry exchange helpers for EIDORS <-> PyEIDORS interop."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from pyeidors.femx import build_eit_mesh

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

    fieldnames = set(reader.fieldnames or [])
    target_key = "meas_phantom" if "meas_phantom" in fieldnames else "meas_target"
    if "meas_homogeneous" not in fieldnames or target_key not in fieldnames:
        raise ValueError(
            "Forward CSV must contain 'meas_homogeneous' and either "
            "'meas_phantom' or 'meas_target' columns."
        )

    baseline = np.asarray([float(row["meas_homogeneous"]) for row in rows], dtype=float)
    phantom = np.asarray([float(row[target_key]) for row in rows], dtype=float)
    if "difference" in reader.fieldnames:
        target_diff = np.asarray(
            [float(row["difference"]) for row in rows], dtype=float
        )
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
    """Collect padded electrode node ids from a marked DOLFINx PyEIDORS mesh."""
    boundary_markers = getattr(mesh, "facet_tags", None)
    electrode_tags = _infer_electrode_tags(mesh)
    if boundary_markers is None or not electrode_tags:
        raise ValueError("Mesh is missing facet tags or electrode tags")

    coords = np.asarray(mesh.coordinates(), dtype=float)
    center = np.mean(coords, axis=0)
    mesh_obj = getattr(mesh, "mesh", None)
    if mesh_obj is None:
        raise ValueError("Expected an EITMesh with a DOLFINx mesh")

    fdim = int(mesh_obj.topology.dim) - 1
    mesh_obj.topology.create_connectivity(fdim, 0)
    facet_to_vertex = mesh_obj.topology.connectivity(fdim, 0)
    if facet_to_vertex is None:
        raise ValueError("Mesh is missing facet-to-vertex connectivity")

    electrode_lists: list[np.ndarray] = []
    max_nodes = 0
    for tag in electrode_tags:
        node_ids: set[int] = set()
        for facet_idx in boundary_markers.find(int(tag)):
            node_ids.update(int(v) for v in facet_to_vertex.links(int(facet_idx)))
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
    """Collect one-based boundary edges from a marked DOLFINx PyEIDORS mesh."""
    boundary_markers = getattr(mesh, "facet_tags", None)
    if boundary_markers is None:
        raise ValueError("Mesh is missing facet tags")

    mesh_obj = getattr(mesh, "mesh", None)
    if mesh_obj is None:
        raise ValueError("Expected an EITMesh with a DOLFINx mesh")

    fdim = int(mesh_obj.topology.dim) - 1
    mesh_obj.topology.create_connectivity(fdim, 0)
    facet_to_vertex = mesh_obj.topology.connectivity(fdim, 0)
    if facet_to_vertex is None:
        raise ValueError("Mesh is missing facet-to-vertex connectivity")

    edges: list[np.ndarray] = []
    for facet_idx, marker in zip(boundary_markers.indices, boundary_markers.values):
        if int(marker) == 0:
            continue
        edge = (
            np.asarray(facet_to_vertex.links(int(facet_idx)), dtype=np.int64).reshape(
                -1
            )
            + 1
        )
        if len(edge) != 2:
            continue
        edges.append(edge)
    if not edges:
        raise ValueError("No boundary edges found")
    out = np.empty((len(edges), 2), dtype=np.int64)
    for edge_idx, edge in enumerate(edges):
        out[edge_idx, :] = edge
    return out


def validate_exchange_payload(payload: dict[str, Any]) -> None:
    """Validate that a payload conforms to the standard interop fields."""
    missing = sorted(REQUIRED_EXCHANGE_FIELDS.difference(payload))
    if missing:
        raise ValueError(
            f"Exchange payload is missing required fields: {', '.join(missing)}"
        )

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


def _load_standard_electrode_node_lists(
    payload: dict[str, Any],
) -> list[np.ndarray] | None:
    if "electrode_nodes" not in payload:
        return None
    electrode_nodes = np.atleast_2d(
        np.asarray(payload["electrode_nodes"], dtype=np.int64)
    )
    counts = np.asarray(payload["electrode_node_counts"], dtype=np.int64).reshape(-1)
    node_lists = []
    for row, count in zip(electrode_nodes, counts):
        active_nodes = np.asarray(row[: int(count)], dtype=np.int64).reshape(-1)
        if int(np.min(active_nodes, initial=1)) < 1:
            raise ValueError("'electrode_nodes' must use one-based node ids")
        node_lists.append(active_nodes - 1)
    return node_lists


def _load_nodes(payload: dict[str, Any]) -> np.ndarray:
    nodes = np.asarray(payload["nodes"], dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] < 2:
        raise ValueError(
            "'nodes' must be a two-dimensional array with at least x/y columns"
        )
    return np.ascontiguousarray(nodes[:, :2], dtype=np.float64)


def _load_one_based_connectivity(
    payload: dict[str, Any], field: str, width: int
) -> np.ndarray:
    data = np.asarray(payload[field], dtype=np.int64)
    if data.ndim == 1:
        if data.size % width != 0:
            raise ValueError(
                f"'{field}' cannot be reshaped into width-{width} connectivity"
            )
        data = data.reshape(-1, width)
    if data.ndim != 2 or data.shape[1] != width:
        raise ValueError(
            f"'{field}' must be a two-dimensional array with width {width}"
        )
    if int(np.min(data, initial=1)) < 1:
        raise ValueError(f"'{field}' must use one-based node ids")
    return np.ascontiguousarray(data - 1, dtype=np.int64)


def _create_dolfinx_triangle_mesh(nodes: np.ndarray, elems: np.ndarray):
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    import basix.ufl
    import ufl

    coordinate_element = basix.ufl.element("Lagrange", "triangle", 1, shape=(2,))
    domain = ufl.Mesh(coordinate_element)
    return dmesh.create_mesh(MPI.COMM_WORLD, elems, domain, nodes)


def _standard_facet_tags(
    mesh,
    boundary_edges: np.ndarray,
    electrode_node_lists: list[np.ndarray],
):
    from dolfinx import mesh as dmesh

    association_table: dict[str, int] = {"domain": 1}
    electrode_sets: list[set[int]] = []
    for elec_idx, nodes in enumerate(electrode_node_lists, start=1):
        node_set = {
            int(node_id) for node_id in np.asarray(nodes, dtype=np.int64).reshape(-1)
        }
        electrode_sets.append(node_set)
        association_table[f"electrode_{elec_idx}"] = elec_idx + 1
    gap_tag = len(electrode_sets) + 2
    association_table["gaps"] = gap_tag

    boundary_pairs = {
        tuple(
            sorted(int(value) for value in np.asarray(edge, dtype=np.int64).reshape(-1))
        )
        for edge in boundary_edges
    }

    fdim = int(mesh.topology.dim) - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    facet_map = mesh.topology.index_map(fdim)
    if facet_to_vertex is None or facet_map is None:
        raise ValueError("Unable to build DOLFINx facet connectivity for exchange mesh")

    indices: list[int] = []
    values: list[int] = []
    for facet_idx in range(int(facet_map.size_local)):
        vertices = tuple(sorted(int(v) for v in facet_to_vertex.links(facet_idx)))
        if vertices not in boundary_pairs:
            continue
        marker = gap_tag
        vertex_set = set(vertices)
        for elec_idx, node_set in enumerate(electrode_sets, start=1):
            if vertex_set.issubset(node_set):
                marker = elec_idx + 1
                break
        indices.append(facet_idx)
        values.append(marker)

    if len(indices) != len(boundary_pairs):
        raise ValueError(
            "Exchange payload boundary edges do not match the generated DOLFINx mesh facets"
        )

    order = np.argsort(np.asarray(indices, dtype=np.int64))
    facet_tags = dmesh.meshtags(
        mesh,
        fdim,
        np.asarray(indices, dtype=np.int32)[order],
        np.asarray(values, dtype=np.int32)[order],
    )

    tdim = int(mesh.topology.dim)
    cell_map = mesh.topology.index_map(tdim)
    n_cells = int(cell_map.size_local if cell_map is not None else 0)
    cell_tags = dmesh.meshtags(
        mesh,
        tdim,
        np.arange(n_cells, dtype=np.int32),
        np.ones(n_cells, dtype=np.int32),
    )
    return facet_tags, cell_tags, association_table


def build_mesh_from_exchange_mat(path: Path):
    """Build an EITMesh from a standard EIDORS/PyEIDORS .mat payload."""
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    validate_exchange_payload(payload)

    nodes = _load_nodes(payload)
    elems = _load_one_based_connectivity(payload, "elems", 3)
    boundary_edges = _load_one_based_connectivity(payload, "boundary_edges", 2)
    standard_electrodes = _load_standard_electrode_node_lists(payload)
    if standard_electrodes is None:
        raise ValueError(
            f"Mesh exchange file {path} does not contain electrode definitions"
        )
    n_elec = int(np.asarray(payload["n_elec"]).reshape(-1)[0])
    if n_elec != len(standard_electrodes):
        raise ValueError(
            f"Exchange payload declares n_elec={n_elec}, "
            f"but contains {len(standard_electrodes)} electrode node lists"
        )

    dolfinx_mesh = _create_dolfinx_triangle_mesh(nodes, elems)
    facet_tags, cell_tags, association_table = _standard_facet_tags(
        dolfinx_mesh,
        boundary_edges,
        standard_electrodes,
    )
    mesh = build_eit_mesh(
        dolfinx_mesh,
        facet_tags=facet_tags,
        cell_tags=cell_tags,
        association_table=association_table,
        radius=float(np.max(np.linalg.norm(nodes, axis=1))),
        mesh_file=str(path),
        electrode_vertices=[
            nodes[np.asarray(ids, dtype=np.int64)] for ids in standard_electrodes
        ],
        mesh_family="triangle",
        geometry_version="interop-v1",
        generator_revision="interop-v1",
    )
    if "mesh_name" in payload:
        mesh.mesh_name = str(np.asarray(payload["mesh_name"]).reshape(-1)[0])
    else:
        mesh.mesh_name = path.stem
    mesh.exchange_format = STANDARD_INTEROP_FORMAT
    mesh.n_electrodes = n_elec
    return mesh, payload
