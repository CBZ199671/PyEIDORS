"""Standardized geometry exchange helpers for EIDORS <-> PyEIDORS interop."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from pyeidors.femx import build_eit_mesh

LEGACY_INTEROP_FORMAT = "eidors_pyeidors_bridge_v1"
STANDARD_INTEROP_FORMAT_V2 = "eidors_pyeidors_geometry_v2"
STANDARD_INTEROP_FORMAT = STANDARD_INTEROP_FORMAT_V2
SUPPORTED_INTEROP_FORMATS = frozenset(
    {LEGACY_INTEROP_FORMAT, STANDARD_INTEROP_FORMAT_V2}
)

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

REQUIRED_V2_EXCHANGE_FIELDS = {
    "schema_version",
    "index_base",
    "dimension",
    "cell_type",
    "boundary_entity_type",
    "boundary_facets",
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
        for vh, vi in zip(baseline, phantom, strict=True):
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
    if str(getattr(mesh, "electrode_model", "")).strip().lower() == "pem":
        source_nodes = np.asarray(
            getattr(mesh, "point_electrode_source_nodes", []),
            dtype=np.int64,
        ).reshape(-1)
        if source_nodes.size == 0:
            raise ValueError("PEM mesh is missing exact point electrode source nodes")
        return source_nodes.reshape(-1, 1) + 1, np.ones(
            source_nodes.size,
            dtype=np.int64,
        )

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


def build_boundary_facets(mesh) -> np.ndarray:
    """Collect one-based boundary facets from a marked DOLFINx PyEIDORS mesh."""
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

    facet_width = max(int(mesh_obj.topology.dim), 1)
    facets: list[np.ndarray] = []
    for facet_idx, marker in zip(
        boundary_markers.indices, boundary_markers.values, strict=True
    ):
        if int(marker) == 0:
            continue
        facet = (
            np.asarray(facet_to_vertex.links(int(facet_idx)), dtype=np.int64).reshape(
                -1
            )
            + 1
        )
        if len(facet) != facet_width:
            raise ValueError(
                "Boundary facet width does not match the mesh topological dimension"
            )
        facets.append(facet)
    if not facets:
        raise ValueError("No boundary facets found")
    out = np.empty((len(facets), facet_width), dtype=np.int64)
    for facet_idx, facet in enumerate(facets):
        out[facet_idx, :] = facet
    return out


def build_boundary_edges(mesh) -> np.ndarray:
    """Collect one-based 2D boundary edges (legacy compatibility API)."""
    facets = build_boundary_facets(mesh)
    if facets.shape[1] != 2:
        raise ValueError(
            "build_boundary_edges only supports 2D meshes; "
            "use build_boundary_facets for 3D meshes"
        )
    return facets


def _scalar_text(value: Any) -> str:
    return str(np.asarray(value).reshape(-1)[0])


def _scalar_int(value: Any, field: str) -> int:
    try:
        return int(np.asarray(value).reshape(-1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"'{field}' must be an integer scalar") from exc


def _scalar_bool(value: Any, field: str) -> bool:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"'{field}' must be a logical scalar")
    return bool(array[0])


def _presence_vector(
    payload: dict[str, Any],
    field: str,
    *,
    size: int,
    default: bool,
) -> np.ndarray:
    if field not in payload:
        return np.full(size, default, dtype=bool)
    values = np.asarray(payload[field], dtype=bool).reshape(-1)
    if values.size == 1:
        return np.full(size, bool(values[0]), dtype=bool)
    if values.size != size:
        raise ValueError(f"'{field}' must be scalar or have {size} entries")
    return values


def _string_vector(
    payload: dict[str, Any],
    field: str,
    *,
    size: int,
) -> list[str] | None:
    if field not in payload:
        return None
    raw = np.asarray(payload[field], dtype=object).reshape(-1)
    values = [str(np.asarray(value).reshape(-1)[0]) for value in raw]
    if len(values) != size:
        raise ValueError(f"'{field}' must have one entry per electrode")
    return values


def _validate_element_data(
    value: Any,
    *,
    field: str,
    n_elements: int,
    present: bool,
) -> None:
    data = np.asarray(value)
    if not present:
        if data.size and np.any(np.isfinite(data)):
            raise ValueError(
                f"'{field}' contains finite values while its presence flag is false"
            )
        return
    if data.size == 1:
        return
    if data.ndim == 1 and data.size == n_elements:
        return
    if data.ndim >= 2 and data.shape[0] == n_elements:
        return
    raise ValueError(
        f"'{field}' must be scalar or have n_elements rows (optionally with frames)"
    )


def _connectivity_array(
    payload: dict[str, Any],
    field: str,
    *,
    width: int,
    n_nodes: int,
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
    lower = int(np.min(data, initial=1))
    upper = int(np.max(data, initial=0))
    if lower < 1:
        raise ValueError(f"'{field}' must use one-based node ids")
    if upper > n_nodes:
        raise ValueError(
            f"'{field}' contains node id {upper}, but nodes has only {n_nodes} rows"
        )
    return data


def _electrode_node_matrix(
    payload: dict[str, Any],
    counts: np.ndarray,
) -> np.ndarray:
    n_elec = _scalar_int(payload["n_elec"], "n_elec")
    raw = np.asarray(payload["electrode_nodes"], dtype=np.int64)
    if raw.ndim == 0:
        raw = raw.reshape(1, 1)
    elif raw.ndim == 1:
        if raw.size % n_elec != 0:
            raise ValueError(
                "Flattened 'electrode_nodes' cannot be reshaped to n_elec rows"
            )
        raw = raw.reshape(n_elec, raw.size // n_elec)
    elif raw.ndim != 2:
        raise ValueError("'electrode_nodes' must be scalar, 1D, or 2D")
    if raw.shape[0] != n_elec and raw.shape[1] == n_elec:
        raw = raw.T
    if raw.shape[0] != n_elec or counts.size != n_elec:
        raise ValueError(
            "'electrode_nodes' rows and 'electrode_node_counts' length must "
            "equal n_elec"
        )
    return np.ascontiguousarray(raw, dtype=np.int64)


def validate_exchange_payload(payload: dict[str, Any]) -> None:
    """Validate that a payload conforms to the standard interop fields."""
    missing = sorted(REQUIRED_EXCHANGE_FIELDS.difference(payload))
    if missing:
        raise ValueError(
            f"Exchange payload is missing required fields: {', '.join(missing)}"
        )

    exchange_format = _scalar_text(payload["exchange_format"])
    if exchange_format not in SUPPORTED_INTEROP_FORMATS:
        raise ValueError(
            f"Unsupported exchange format {exchange_format!r}; expected one of "
            f"{sorted(SUPPORTED_INTEROP_FORMATS)!r}"
        )
    if exchange_format == STANDARD_INTEROP_FORMAT_V2:
        missing_v2 = sorted(REQUIRED_V2_EXCHANGE_FIELDS.difference(payload))
        if missing_v2:
            raise ValueError(
                "Geometry v2 payload is missing required fields: "
                + ", ".join(missing_v2)
            )
        if _scalar_int(payload["schema_version"], "schema_version") != 2:
            raise ValueError("'schema_version' must be 2 for Geometry v2")
        if _scalar_int(payload["index_base"], "index_base") != 1:
            raise ValueError("'index_base' must be 1 for Geometry v2")

    nodes = np.asarray(payload["nodes"])
    if nodes.ndim != 2 or nodes.shape[0] == 0 or nodes.shape[1] not in {2, 3}:
        raise ValueError("'nodes' must have shape (n_nodes, 2) or (n_nodes, 3)")
    inferred_dimension = 3 if nodes.shape[1] == 3 else 2
    dimension = (
        _scalar_int(payload["dimension"], "dimension")
        if "dimension" in payload
        else inferred_dimension
    )
    if dimension not in {2, 3}:
        raise ValueError("'dimension' must be 2 or 3")
    if nodes.shape[1] != dimension:
        raise ValueError(
            f"'nodes' has {nodes.shape[1]} columns, inconsistent with dimension={dimension}"
        )

    expected_cell_type = "triangle" if dimension == 2 else "tetrahedron"
    expected_elem_width = dimension + 1
    cell_type = (
        _scalar_text(payload["cell_type"]).strip().lower()
        if "cell_type" in payload
        else expected_cell_type
    )
    if cell_type != expected_cell_type:
        raise ValueError(
            f"dimension={dimension} requires cell_type={expected_cell_type!r}"
        )
    elems = _connectivity_array(
        payload,
        "elems",
        width=expected_elem_width,
        n_nodes=nodes.shape[0],
    )

    boundary_field = (
        "boundary_facets" if "boundary_facets" in payload else "boundary_edges"
    )
    boundary = _connectivity_array(
        payload,
        boundary_field,
        width=dimension,
        n_nodes=nodes.shape[0],
    )
    if "boundary_facets" in payload and "boundary_edges" in payload:
        legacy_boundary = _connectivity_array(
            payload,
            "boundary_edges",
            width=dimension,
            n_nodes=nodes.shape[0],
        )
        canonical = np.sort(boundary, axis=1)
        legacy_canonical = np.sort(legacy_boundary, axis=1)
        canonical = canonical[np.lexsort(canonical.T[::-1])]
        legacy_canonical = legacy_canonical[np.lexsort(legacy_canonical.T[::-1])]
        if not np.array_equal(canonical, legacy_canonical):
            raise ValueError(
                "'boundary_facets' and legacy 'boundary_edges' must describe "
                "the same boundary entities when both are present"
            )
    if "boundary_entity_type" in payload:
        expected_boundary_type = "edge" if dimension == 2 else "triangle"
        boundary_type = _scalar_text(payload["boundary_entity_type"]).strip().lower()
        if boundary_type != expected_boundary_type:
            raise ValueError(
                f"dimension={dimension} requires "
                f"boundary_entity_type={expected_boundary_type!r}"
            )

    n_elec = _scalar_int(payload["n_elec"], "n_elec")
    if n_elec <= 0:
        raise ValueError("'n_elec' must be positive")
    counts = np.asarray(payload["electrode_node_counts"], dtype=np.int64).reshape(-1)
    electrode_nodes = _electrode_node_matrix(payload, counts)
    for row, count in zip(electrode_nodes, counts, strict=True):
        count_value = int(count)
        if count_value <= 0 or count_value > row.size:
            raise ValueError(
                "Each electrode node count must be positive and fit its padded row"
            )
        active = np.asarray(row[:count_value], dtype=np.int64)
        lower = int(np.min(active, initial=1))
        upper = int(np.max(active, initial=0))
        if lower < 1 or upper > nodes.shape[0]:
            raise ValueError(
                "Active electrode node ids must be one-based and within nodes"
            )

    truth_present = (
        _scalar_bool(payload["truth_elem_data_present"], "truth_elem_data_present")
        if "truth_elem_data_present" in payload
        else True
    )
    _validate_element_data(
        payload["truth_elem_data"],
        field="truth_elem_data",
        n_elements=elems.shape[0],
        present=truth_present,
    )
    background_present = (
        _scalar_bool(payload["background_present"], "background_present")
        if "background_present" in payload
        else True
    )
    background = np.asarray(payload["background"]).reshape(-1)
    if background.size != 1:
        raise ValueError("'background' must be a scalar")
    if not background_present and np.any(np.isfinite(background)):
        raise ValueError(
            "'background' contains a finite value while background_present is false"
        )
    if "background_elem_data" in payload:
        background_elem_present = _scalar_bool(
            payload.get("background_elem_data_present", True),
            "background_elem_data_present",
        )
        _validate_element_data(
            payload["background_elem_data"],
            field="background_elem_data",
            n_elements=elems.shape[0],
            present=background_elem_present,
        )
    impedance = np.asarray(payload["contact_impedance"]).reshape(-1)
    if impedance.size not in {1, n_elec}:
        raise ValueError(
            "'contact_impedance' must be scalar or have one value per electrode"
        )
    impedance_present = _presence_vector(
        payload,
        "contact_impedance_present",
        size=n_elec,
        default=True,
    )
    expanded_impedance = (
        np.full(n_elec, impedance[0], dtype=impedance.dtype)
        if impedance.size == 1
        else impedance
    )
    if np.any(~impedance_present & np.isfinite(expanded_impedance)):
        raise ValueError(
            "'contact_impedance' contains finite values for electrodes whose "
            "contact_impedance_present flag is false"
        )
    electrode_models = _string_vector(
        payload,
        "electrode_model",
        size=n_elec,
    )
    if electrode_models is not None:
        supported_models = {"cem", "cem_faces", "point", "distributed_point"}
        unknown = sorted(set(electrode_models).difference(supported_models))
        if unknown:
            raise ValueError(
                "'electrode_model' contains unsupported values: " + ", ".join(unknown)
            )


def source_electrode_models(payload: dict[str, Any]) -> list[str]:
    """Return declared or boundary-derived EIDORS electrode model classes."""

    n_elec = _scalar_int(payload["n_elec"], "n_elec")
    declared = _string_vector(
        payload,
        "electrode_model",
        size=n_elec,
    )
    if declared is not None:
        return declared

    counts = np.asarray(
        payload["electrode_node_counts"],
        dtype=np.int64,
    ).reshape(-1)
    electrode_nodes = _electrode_node_matrix(payload, counts)
    boundary_field = (
        "boundary_facets" if "boundary_facets" in payload else "boundary_edges"
    )
    dimension = _scalar_int(
        payload.get("dimension", np.asarray(payload["nodes"]).shape[1]),
        "dimension",
    )
    boundary = _connectivity_array(
        payload,
        boundary_field,
        width=dimension,
        n_nodes=np.asarray(payload["nodes"]).shape[0],
    )

    models: list[str] = []
    for row, count in zip(electrode_nodes, counts, strict=True):
        active = {
            int(value)
            for value in np.asarray(row[: int(count)], dtype=np.int64).reshape(-1)
        }
        if len(active) == 1:
            models.append("point")
            continue
        has_complete_boundary_facet = any(
            all(int(node_id) in active for node_id in facet)
            for facet in np.asarray(boundary, dtype=np.int64)
        )
        models.append("cem" if has_complete_boundary_facet else "distributed_point")
    return models


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
    counts = np.asarray(payload["electrode_node_counts"], dtype=np.int64).reshape(-1)
    electrode_nodes = _electrode_node_matrix(payload, counts)
    node_lists = []
    for row, count in zip(electrode_nodes, counts, strict=True):
        active_nodes = np.asarray(row[: int(count)], dtype=np.int64).reshape(-1)
        if int(np.min(active_nodes, initial=1)) < 1:
            raise ValueError("'electrode_nodes' must use one-based node ids")
        node_lists.append(active_nodes - 1)
    return node_lists


def _load_nodes(payload: dict[str, Any], *, dimension: int) -> np.ndarray:
    import dolfinx

    geometry_dtype = np.dtype(getattr(dolfinx, "default_real_type", np.float64))
    nodes = np.asarray(payload["nodes"], dtype=geometry_dtype)
    if nodes.ndim != 2 or nodes.shape[1] != dimension:
        raise ValueError(f"'nodes' must have shape (n_nodes, {dimension})")
    return np.ascontiguousarray(nodes, dtype=geometry_dtype)


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


def _create_dolfinx_simplex_mesh(
    nodes: np.ndarray,
    elems: np.ndarray,
    *,
    cell_type: str,
):
    from dolfinx import mesh as dmesh
    from mpi4py import MPI
    import basix.ufl
    import ufl

    coordinate_element = basix.ufl.element(
        "Lagrange",
        cell_type,
        1,
        shape=(nodes.shape[1],),
        dtype=nodes.dtype,
    )
    domain = ufl.Mesh(coordinate_element)
    return dmesh.create_mesh(MPI.COMM_WORLD, elems, domain, nodes)


def source_cell_data_to_local(mesh, values: Any, *, name: str = "cell data"):
    """Map source-file cell data into the imported DOLFINx local cell order."""
    source_indices = np.asarray(
        getattr(mesh, "source_cell_indices", []),
        dtype=np.int64,
    ).reshape(-1)
    if source_indices.size != mesh.num_cells():
        raise ValueError(
            f"{name} cannot be mapped: imported mesh has no complete source-cell map"
        )
    array = np.asarray(values)
    if array.ndim == 0 and source_indices.size == 1 and source_indices[0] == 0:
        return array.reshape(1)
    if array.ndim == 0 or array.shape[0] <= int(np.max(source_indices, initial=-1)):
        raise ValueError(
            f"{name} must have a source-cell leading axis covering "
            f"{int(np.max(source_indices, initial=-1)) + 1} cells"
        )
    return np.asarray(array[source_indices])


def _standard_facet_tags(
    mesh,
    boundary_facets: np.ndarray,
    electrode_node_lists: list[np.ndarray],
    electrode_models: list[str] | None = None,
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

    boundary_entities = {
        tuple(
            sorted(
                int(value) for value in np.asarray(facet, dtype=np.int64).reshape(-1)
            )
        )
        for facet in boundary_facets
    }

    vertex_map = mesh.topology.index_map(0)
    if vertex_map is None:
        raise ValueError("Unable to access DOLFINx vertex map for exchange mesh")
    n_local_vertices = int(vertex_map.size_local + vertex_map.num_ghosts)
    local_to_source = np.asarray(
        mesh.geometry.input_global_indices,
        dtype=np.int64,
    ).reshape(-1)
    if local_to_source.size < n_local_vertices:
        raise ValueError(
            "DOLFINx geometry input map does not cover all local exchange vertices"
        )
    local_to_source = local_to_source[:n_local_vertices]

    fdim = int(mesh.topology.dim) - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    facet_map = mesh.topology.index_map(fdim)
    if facet_to_vertex is None or facet_map is None:
        raise ValueError("Unable to build DOLFINx facet connectivity for exchange mesh")

    boundary_records: list[tuple[int, tuple[int, ...]]] = []
    for facet_idx in range(int(facet_map.size_local)):
        local_vertices = np.asarray(
            facet_to_vertex.links(facet_idx),
            dtype=np.int64,
        )
        source_vertices = tuple(
            sorted(int(value) for value in local_to_source[local_vertices])
        )
        if source_vertices not in boundary_entities:
            continue
        boundary_records.append((facet_idx, source_vertices))

    if len(boundary_records) != len(boundary_entities):
        raise ValueError(
            "Exchange payload boundary facets do not match the generated DOLFINx mesh facets"
        )

    if electrode_models is None:
        point_electrodes = {
            index
            for index, node_set in enumerate(electrode_sets)
            if len(node_set) < int(mesh.topology.dim)
        }
    else:
        point_electrodes = {
            index
            for index, model in enumerate(electrode_models)
            if model in {"point", "distributed_point"}
        }
    native_pem = bool(
        electrode_models
        and len(electrode_models) == len(electrode_sets)
        and all(model == "point" for model in electrode_models)
        and all(len(node_set) == 1 for node_set in electrode_sets)
    )
    markers: dict[int, int] = {}
    marker_counts = np.zeros(len(electrode_sets), dtype=np.int64)
    for facet_idx, source_vertices in boundary_records:
        vertex_set = set(source_vertices)
        for elec_idx, node_set in enumerate(electrode_sets, start=1):
            if elec_idx - 1 not in point_electrodes and vertex_set.issubset(node_set):
                markers[facet_idx] = elec_idx + 1
                marker_counts[elec_idx - 1] += 1
                break

    if not native_pem:
        for point_index in sorted(point_electrodes):
            node_set = electrode_sets[point_index]
            candidates = [
                facet_idx
                for facet_idx, source_vertices in boundary_records
                if node_set.intersection(source_vertices)
            ]
            if not candidates:
                raise ValueError(
                    f"Point electrode {point_index + 1} is not on a boundary facet"
                )
            selected = next(
                (facet_idx for facet_idx in candidates if facet_idx not in markers),
                None,
            )
            if selected is None:
                raise ValueError(
                    "Unable to assign a unique incident boundary facet to point "
                    f"electrode {point_index + 1}"
                )
            markers[selected] = point_index + 2
            marker_counts[point_index] += 1

    for facet_idx, source_vertices in boundary_records:
        if facet_idx in markers:
            continue
        if native_pem:
            markers[facet_idx] = gap_tag
            continue
        point_candidates = [
            point_index
            for point_index in sorted(point_electrodes)
            if electrode_sets[point_index].intersection(source_vertices)
        ]
        if point_candidates:
            selected = min(point_candidates, key=lambda item: marker_counts[item])
            markers[facet_idx] = selected + 2
            marker_counts[selected] += 1
        else:
            markers[facet_idx] = gap_tag

    missing_electrodes = (
        []
        if native_pem
        else [
            str(index + 1)
            for index, count in enumerate(marker_counts)
            if int(count) <= 0
        ]
    )
    if missing_electrodes:
        raise ValueError(
            "Imported electrode definitions have no positive boundary facets: "
            + ", ".join(missing_electrodes)
        )

    indices = np.asarray(sorted(markers), dtype=np.int32)
    values = np.asarray([markers[int(index)] for index in indices], dtype=np.int32)
    order = np.argsort(np.asarray(indices, dtype=np.int64))
    facet_tags = dmesh.meshtags(
        mesh,
        fdim,
        indices[order],
        values[order],
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
    projection = (
        "none"
        if native_pem
        else ("incident_boundary_facets" if point_electrodes else "exact_surface_nodes")
    )
    return facet_tags, cell_tags, association_table, projection


def build_mesh_from_exchange_mat(path: Path):
    """Build an EITMesh from a standard EIDORS/PyEIDORS .mat payload."""
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    validate_exchange_payload(payload)

    node_columns = int(np.asarray(payload["nodes"]).shape[1])
    dimension = (
        _scalar_int(payload["dimension"], "dimension")
        if "dimension" in payload
        else node_columns
    )
    cell_type = "triangle" if dimension == 2 else "tetrahedron"
    nodes = _load_nodes(payload, dimension=dimension)
    elems = _load_one_based_connectivity(payload, "elems", dimension + 1)
    boundary_field = (
        "boundary_facets" if "boundary_facets" in payload else "boundary_edges"
    )
    boundary_facets = _load_one_based_connectivity(
        payload,
        boundary_field,
        dimension,
    )
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
    electrode_models = source_electrode_models(payload)

    dolfinx_mesh = _create_dolfinx_simplex_mesh(
        nodes,
        elems,
        cell_type=cell_type,
    )
    facet_tags, cell_tags, association_table, electrode_projection = (
        _standard_facet_tags(
            dolfinx_mesh,
            boundary_facets,
            standard_electrodes,
            electrode_models,
        )
    )
    center = np.mean(nodes, axis=0)
    mesh = build_eit_mesh(
        dolfinx_mesh,
        facet_tags=facet_tags,
        cell_tags=cell_tags,
        association_table=association_table,
        radius=float(np.max(np.linalg.norm(nodes - center, axis=1))),
        mesh_file=str(path),
        electrode_vertices=[
            nodes[np.asarray(ids, dtype=np.int64)] for ids in standard_electrodes
        ],
        mesh_family=cell_type,
        geometry_version=(
            "interop-v2"
            if _scalar_text(payload["exchange_format"]) == STANDARD_INTEROP_FORMAT_V2
            else "interop-v1"
        ),
        generator_revision="interop-v2",
    )
    if "mesh_name" in payload:
        mesh.mesh_name = str(np.asarray(payload["mesh_name"]).reshape(-1)[0])
    else:
        mesh.mesh_name = path.stem
    mesh.exchange_format = _scalar_text(payload["exchange_format"])
    mesh.n_electrodes = n_elec
    mesh.electrode_projection = electrode_projection
    mesh.source_electrode_models = electrode_models
    original_cell_index = np.asarray(
        dolfinx_mesh.topology.original_cell_index,
        dtype=np.int64,
    ).reshape(-1)
    if original_cell_index.size < mesh.num_cells():
        raise ValueError(
            "DOLFINx original-cell map does not cover all imported local cells"
        )
    mesh.source_cell_indices = np.ascontiguousarray(
        original_cell_index[: mesh.num_cells()],
        dtype=np.int64,
    )
    for field in (
        "background_elem_data",
        "target_elem_data",
        "truth_elem_data",
    ):
        if field in payload:
            payload[field] = source_cell_data_to_local(
                mesh,
                payload[field],
                name=field,
            )
    payload["source_cell_indices"] = mesh.source_cell_indices.copy()
    payload["element_data_order"] = "dolfinx_local"
    if all(model == "point" for model in electrode_models):
        mesh.electrode_model = "pem"
        mesh.point_electrode_source_nodes = np.asarray(
            [int(nodes[0]) for nodes in standard_electrodes],
            dtype=np.int64,
        )
    elif all(model in {"cem", "cem_faces"} for model in electrode_models):
        mesh.electrode_model = "cem"
    elif all(model == "distributed_point" for model in electrode_models):
        mesh.electrode_model = "cem"
    else:
        mesh.electrode_model = "mixed"
    ground_value = payload.get("effective_gnd_node", payload.get("gnd_node"))
    ground_array = (
        np.asarray(ground_value).reshape(-1)
        if ground_value is not None
        else np.empty(0, dtype=float)
    )
    try:
        ground_scalar = (
            float(ground_array[0]) if ground_array.size == 1 else float("nan")
        )
    except (TypeError, ValueError):
        ground_scalar = float("nan")
    if np.isfinite(ground_scalar):
        ground_node = int(ground_scalar)
        if ground_node < 1 or ground_node > nodes.shape[0]:
            raise ValueError(
                f"Effective ground node {ground_node} is outside the imported mesh"
            )
        mesh.gnd_node_source = ground_node - 1
    else:
        mesh.gnd_node_source = None
    return mesh, payload
