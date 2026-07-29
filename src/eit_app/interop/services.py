"""High-level services for the EIDORS <-> PyEIDORS interop hub."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from eit_app.i18n import t
from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors.data.measurement_dataset import MeasurementDataset
from pyeidors.interop import STANDARD_INTEROP_FORMAT
from pyeidors.interop.geometry_exchange import source_electrode_models
from pyeidors.utils.numeric_ops import real_array_if_zero_imaginary

from .bridge_package import (
    CAPTURE_SCRIPT_NAME,
    CONFIG_NAME,
    GEOMETRY_NAME,
    LoadedBridgePackage,
    default_manifest,
    load_bridge_package,
    save_bridge_package,
)
from .environment import (
    EidorsEnvironmentDetector,
    _run_command_capture,
    matlab_command_for_execution,
    matlab_runtime_path,
    to_posix_path,
    to_windows_path,
)
from .matlab_templates import CAPTURE_SCRIPT_TEMPLATE
from .models import (
    EidorsEnvironment,
    EidorsExportJob,
    EidorsImportPreview,
    InteropCapabilityReport,
    ReconstructionPreset,
)

log = logging.getLogger(__name__)

BRIDGE_RUNTIME_NAME = "bridge_runtime.json"
BRIDGE_MANIFEST_ALIAS = "bridge_manifest.json"
CAPTURE_REQUEST_NAME = "capture_request.json"
CAPTURE_REPORT_NAME = "capture_report.json"
RUN_IMPORT_FROM_PYEIDORS_NAME = "run_import_from_pyeidors.m"

_SCRIPT_HINT_PATTERNS: dict[str, re.Pattern[str]] = {
    "mk_common_model": re.compile(r"\bmk_common_model\b", re.IGNORECASE),
    "mk_stim_patterns": re.compile(r"\bmk_stim_patterns\b", re.IGNORECASE),
    "inv_solve": re.compile(r"\binv_solve\b", re.IGNORECASE),
    "fwd_solve": re.compile(r"\bfwd_solve\b", re.IGNORECASE),
    "z_contact": re.compile(r"\bz_contact\b", re.IGNORECASE),
    "n_elec": re.compile(r"\bn_elec\b|\bnelec\b", re.IGNORECASE),
    "vh": re.compile(r"\bvh\b", re.IGNORECASE),
    "vi": re.compile(r"\bvi\b", re.IGNORECASE),
}


def _read_text_if_possible(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def detect_script_hints(path: str | Path) -> dict[str, Any]:
    """Shallow-parse a MATLAB script to infer likely EIDORS content."""

    source = Path(path)
    text = _read_text_if_possible(source)
    hints: dict[str, Any] = {}
    for key, pattern in _SCRIPT_HINT_PATTERNS.items():
        hints[key] = bool(pattern.search(text))

    if hints["inv_solve"] and (hints["vh"] or hints["vi"]):
        script_kind = "difference_inverse"
    elif hints["fwd_solve"] or hints["mk_common_model"]:
        script_kind = "forward_or_model"
    elif "show_fem" in text or "show_slices" in text or "plot(" in text:
        script_kind = "plot_only"
    else:
        script_kind = "unknown"

    stim_pattern = "{ad}" if "{ad}" in text else "{op}" if "{op}" in text else ""
    meas_pattern = stim_pattern

    hints.update(
        {
            "script_kind": script_kind,
            "stim_pattern": stim_pattern,
            "meas_pattern": meas_pattern,
            "path": str(source),
        }
    )
    return hints


def _measurement_summary(measurements: dict[str, np.ndarray] | None) -> dict[str, str]:
    if not measurements:
        return {"status": "No boundary-voltage arrays found in this package."}
    keys = sorted(measurements)
    first_key = next(
        (
            key
            for key in ("target", "ground_truth", "difference", "homogeneous")
            if key in measurements
        ),
        keys[0],
    )
    first = np.asarray(measurements[first_key]).reshape(-1)
    return {
        "arrays": ", ".join(keys),
        "points": str(first.size),
        "difference": "difference" if "difference" in measurements else "n/a",
    }


def _geometry_summary(geometry_payload: dict[str, Any] | None) -> dict[str, str]:
    if not geometry_payload:
        return {"status": "No geometry payload found."}
    nodes = np.asarray(geometry_payload.get("nodes", np.zeros((0, 2))))
    elems = np.asarray(geometry_payload.get("elems", np.zeros((0, 0))))
    n_elec_raw = geometry_payload.get("n_elec", 0)
    try:
        n_elec = int(np.asarray(n_elec_raw).reshape(-1)[0])
    except Exception:
        n_elec = 0
    dimension = "3D" if nodes.ndim == 2 and nodes.shape[1] >= 3 else "2D"
    return {
        "dimension": dimension,
        "nodes": str(int(nodes.shape[0])),
        "elements": str(int(elems.shape[0])),
        "electrodes": str(n_elec),
    }


def _mat_scalar_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    array = np.asarray(value).reshape(-1)
    return default if array.size == 0 else bool(array[0])


def _mat_json_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return {}
    try:
        parsed = json.loads(str(array[0]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _python_scalar(value: Any) -> float | complex:
    scalar = complex(np.asarray(value).reshape(-1)[0])
    if abs(scalar.imag) <= 1.0e-15:
        return float(scalar.real)
    return scalar


def _normalize_contact_impedance(
    value: Any,
    *,
    presence: Any = None,
) -> float | complex | list[float | complex] | None:
    if value is None:
        return None
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return None
    present = (
        np.ones(array.size, dtype=bool)
        if presence is None
        else np.asarray(presence, dtype=bool).reshape(-1)
    )
    if present.size == 1:
        present = np.full(array.size, bool(present[0]), dtype=bool)
    if present.size != array.size and array.size != 1:
        return None
    if not np.all(present):
        return None
    if array.size == 1:
        return _python_scalar(array[0])
    return [_python_scalar(item) for item in array]


def _interop_semantics_from_geometry(
    geometry: dict[str, Any],
    *,
    n_elec: int,
) -> dict[str, Any]:
    semantics = _mat_json_mapping(geometry.get("capture_metadata_json"))
    blockers = [
        str(item) for item in semantics.get("forward_blockers", []) if str(item)
    ]
    warnings = [
        str(item) for item in semantics.get("forward_warnings", []) if str(item)
    ]
    legacy_projection_blocker = (
        "point_or_distributed_point_electrode_requires_explicit_projection_opt_in"
    )
    blockers = [item for item in blockers if item != legacy_projection_blocker]

    source_models = source_electrode_models(geometry)
    if source_models and all(model == "point" for model in source_models):
        electrode_model = "pem"
        contact_impedance_applicable = False
        electrode_projection = "none"
    elif source_models and all(
        model in {"cem", "cem_faces"} for model in source_models
    ):
        electrode_model = "cem"
        contact_impedance_applicable = True
        electrode_projection = "exact_surface_nodes"
    elif source_models and all(model == "distributed_point" for model in source_models):
        electrode_model = "cem"
        contact_impedance_applicable = True
        electrode_projection = "incident_boundary_facets"
        blocker = "distributed_point_electrode_requires_explicit_projection_opt_in"
        if blocker not in blockers:
            blockers.append(blocker)
    else:
        electrode_model = "cem"
        contact_impedance_applicable = True
        electrode_projection = "unsupported_mixed"
        blocker = "mixed_cem_pem_electrode_models_not_supported"
        if blocker not in blockers:
            blockers.append(blocker)

    contact_present = np.asarray(
        geometry.get("contact_impedance_present", np.ones(n_elec, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if contact_present.size == 1:
        contact_present = np.full(n_elec, bool(contact_present[0]), dtype=bool)
    contact_missing = contact_present.size != n_elec or not np.all(contact_present)
    contact_blocker = "contact_impedance_missing_no_eidors_default"
    if contact_missing and contact_impedance_applicable:
        if contact_blocker not in blockers:
            blockers.append(contact_blocker)
    elif not contact_impedance_applicable:
        blockers = [item for item in blockers if item != contact_blocker]

    background_present = _mat_scalar_bool(
        geometry.get("background_present"),
        default=True,
    )
    if not background_present:
        background_elem_present = _mat_scalar_bool(
            geometry.get("background_elem_data_present"),
            default="background_elem_data" in geometry,
        )
        background_elem_data = np.asarray(geometry.get("background_elem_data", []))
        blocker = (
            "background_is_nonuniform_and_not_gui_scalar_compatible"
            if background_elem_present and background_elem_data.size
            else "background_image_missing_or_unmappable"
        )
        if blocker not in blockers:
            blockers.append(blocker)

    if "stimulation_supported" in geometry and not _mat_scalar_bool(
        geometry["stimulation_supported"],
        default=False,
    ):
        blocker = "stimulation_missing_or_unsupported_voltage_interior_complex_pattern"
        if blocker not in blockers:
            blockers.append(blocker)

    return {
        **semantics,
        "source_framework": str(
            np.asarray(geometry.get("source_framework", "unknown")).reshape(-1)[0]
        ),
        "source_electrode_models": source_models,
        "source_electrode_model_class": (
            "point"
            if all(model == "point" for model in source_models)
            else (
                "cem"
                if all(model in {"cem", "cem_faces"} for model in source_models)
                else (
                    "distributed_point"
                    if all(model == "distributed_point" for model in source_models)
                    else "mixed"
                )
            )
        ),
        "electrode_model": electrode_model,
        "electrode_projection": electrode_projection,
        "contact_impedance_applicable": contact_impedance_applicable,
        "effective_gnd_node": (
            int(np.asarray(geometry["effective_gnd_node"]).reshape(-1)[0])
            if "effective_gnd_node" in geometry
            and np.asarray(geometry["effective_gnd_node"]).size == 1
            and np.isfinite(np.asarray(geometry["effective_gnd_node"]).reshape(-1)[0])
            else None
        ),
        "forward_blockers": blockers,
        "forward_warnings": warnings,
        "forward_ready": not blockers,
    }


def _config_from_loaded_package(loaded: LoadedBridgePackage) -> ForwardModelConfig:
    geometry = loaded.geometry_payload or {}
    nodes = np.asarray(geometry.get("nodes", np.zeros((0, 2))), dtype=float)
    base = loaded.forward_model_config or ForwardModelConfig()
    overrides: dict[str, Any] = {}
    if nodes.ndim == 2 and nodes.shape[0] and nodes.shape[1] in {2, 3}:
        dimension = int(nodes.shape[1])
        bounds_min = np.min(nodes, axis=0)
        bounds_max = np.max(nodes, axis=0)
        center = 0.5 * (bounds_min + bounds_max)
        overrides.update(
            {
                "mesh_source": "interop",
                "mesh_path": str(
                    Path(base.mesh_path).resolve()
                    if base.mesh_path
                    else (loaded.root / GEOMETRY_NAME).resolve()
                ),
                "mesh_dimension": dimension,
                "mesh_family": "triangle" if dimension == 2 else "tetrahedron",
                "geometry_version": "interop-v2",
                "radius": float(
                    np.max(np.linalg.norm(nodes[:, :2] - center[:2], axis=1))
                ),
            }
        )
        if dimension == 3:
            overrides.update(
                {
                    "height": float(bounds_max[2] - bounds_min[2]),
                    "z_center": float(center[2]),
                }
            )
    if geometry:
        n_elec = int(np.asarray(geometry.get("n_elec", base.n_elec)).reshape(-1)[0])
        semantics = _interop_semantics_from_geometry(
            geometry,
            n_elec=n_elec,
        )
        overrides.update(
            {
                "n_elec": n_elec,
                "n_rings": 1,
                "electrode_model": semantics["electrode_model"],
                "contact_impedance": _normalize_contact_impedance(
                    geometry.get("contact_impedance"),
                    presence=geometry.get("contact_impedance_present"),
                ),
                "interop_semantics": semantics,
            }
        )
        if semantics["electrode_model"] == "pem":
            overrides.update(
                {
                    "drive_mode": "total_current",
                    "potential_order": 1,
                }
            )
        if _mat_scalar_bool(geometry.get("background_present"), default=True):
            background = np.asarray(geometry.get("background", [])).reshape(-1)
            if background.size == 1 and np.isfinite(background[0]):
                overrides["background_conductivity"] = _python_scalar(background[0])
    custom_patterns = _custom_patterns_from_geometry(geometry)
    if custom_patterns is not None:
        stim_matrix, meas_matrices = custom_patterns
        overrides.update(
            {
                "measurement_protocol": "custom",
                "custom_stim_matrix": stim_matrix,
                "custom_meas_matrices": meas_matrices,
            }
        )
    return base.with_overrides(**overrides)


def _custom_patterns_from_geometry(
    geometry: dict[str, Any],
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    if "stim_matrix" not in geometry or "meas_matrices" not in geometry:
        return None
    if "stimulation_supported" in geometry and not _mat_scalar_bool(
        geometry["stimulation_supported"],
        default=False,
    ):
        return None
    stim_matrix = real_array_if_zero_imaginary(
        geometry["stim_matrix"],
        name="EIDORS effective stimulation matrix",
    )
    if stim_matrix.size == 0:
        return None
    if stim_matrix.ndim == 1:
        stim_matrix = stim_matrix.reshape(1, -1)
    if stim_matrix.ndim != 2 or stim_matrix.shape[0] == 0:
        raise ValueError("'stim_matrix' must have shape (n_stim, n_elec)")

    counts = np.asarray(
        geometry.get("measurement_counts", []),
        dtype=np.int64,
    ).reshape(-1)
    if counts.size != stim_matrix.shape[0]:
        raise ValueError("'measurement_counts' must have one entry per stimulation")
    raw = real_array_if_zero_imaginary(
        geometry["meas_matrices"],
        name="EIDORS measurement matrices",
    )
    if stim_matrix.shape[0] == 1 and raw.ndim == 2:
        raw = raw.reshape(1, *raw.shape)
    elif raw.ndim == 2 and raw.shape == stim_matrix.shape and np.all(counts == 1):
        raw = raw.reshape(stim_matrix.shape[0], 1, stim_matrix.shape[1])
    if raw.ndim != 3 or raw.shape[0] != stim_matrix.shape[0]:
        raise ValueError("'meas_matrices' must have shape (n_stim, max_n_meas, n_elec)")
    matrices: list[np.ndarray] = []
    for stim_index, count in enumerate(counts):
        n_meas = int(count)
        if n_meas <= 0 or n_meas > raw.shape[1]:
            raise ValueError(
                "Each measurement count must be positive and fit meas_matrices"
            )
        matrices.append(np.asarray(raw[stim_index, :n_meas, :], dtype=float))
    return stim_matrix, matrices


def _build_preview(loaded: LoadedBridgePackage) -> EidorsImportPreview:
    geometry_summary = _geometry_summary(loaded.geometry_payload)
    measurement_summary = _measurement_summary(loaded.measurements)
    capability = InteropCapabilityReport(
        can_import_geometry=loaded.geometry_payload is not None,
        can_import_measurements=loaded.measurements is not None,
    )
    forward_cfg = _config_from_loaded_package(loaded)

    recognized = {
        "n_elec": forward_cfg.n_elec,
        "n_rings": forward_cfg.n_rings,
        "stim_pattern": forward_cfg.stim_pattern,
        "meas_pattern": forward_cfg.meas_pattern,
        "rotate_meas": forward_cfg.rotate_meas,
        "use_meas_current": forward_cfg.use_meas_current,
        "use_meas_current_next": forward_cfg.use_meas_current_next,
        "mesh_dimension": forward_cfg.mesh_dimension,
        "mesh_refinement": forward_cfg.mesh_refinement,
        "contact_impedance": forward_cfg.contact_impedance,
        "source_electrode_models": forward_cfg.interop_semantics.get(
            "source_electrode_models", []
        ),
        "forward_ready": not forward_cfg.interop_semantics.get("forward_blockers", []),
        "point_count": forward_cfg.point_count(),
    }
    inferred: dict[str, Any] = {}
    if "config" not in loaded.manifest.files:
        inferred["source"] = (
            "No config.json found; values were inferred from geometry and defaults."
        )

    missing: list[str] = []
    if loaded.geometry_payload is None:
        missing.append("geometry")
    if loaded.measurements is None:
        missing.append("measurements")
    missing.extend(
        str(item)
        for item in forward_cfg.interop_semantics.get("forward_blockers", [])
        if str(item)
    )

    warnings = [
        *loaded.manifest.notes,
        *[
            str(item)
            for item in forward_cfg.interop_semantics.get("forward_warnings", [])
            if str(item)
        ],
    ]
    if loaded.measurements:
        try:
            measurements = loaded.measurements
            if "homogeneous" in measurements:
                MeasurementDataset.from_metadata(
                    np.asarray(measurements["homogeneous"]).reshape(1, -1),
                    forward_cfg.to_mapping(),
                    data_type="real",
                )
            if "target" in measurements:
                MeasurementDataset.from_metadata(
                    np.asarray(measurements["target"]).reshape(1, -1),
                    forward_cfg.to_mapping(),
                    data_type="real",
                )
            capability.can_import_measurements = True
        except Exception as exc:
            capability.can_import_measurements = False
            warnings.append(t("interop.svc.err.boundary_uninterpretable", error=exc))

    return EidorsImportPreview(
        forward_model_config=forward_cfg,
        capability_report=capability,
        geometry_summary=geometry_summary,
        measurement_summary=measurement_summary,
        recognized_fields=recognized,
        inferred_fields=inferred,
        missing_fields=missing,
        warnings=warnings,
    )


def _boundary_entities_from_cells(cell_connectivity: np.ndarray) -> np.ndarray:
    cells = np.asarray(cell_connectivity, dtype=int)
    if cells.ndim != 2 or cells.size == 0:
        return np.empty((0, 0), dtype=np.int64)

    verts_per_cell = cells.shape[1]
    if verts_per_cell == 3:
        local_facets = ((0, 1), (1, 2), (0, 2))
    elif verts_per_cell == 4:
        local_facets = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    else:
        raise ValueError(t("interop.svc.err.exporter_mesh_only"))

    counts: dict[tuple[int, ...], tuple[int, ...]] = {}
    repeats: dict[tuple[int, ...], int] = {}
    for cell in cells:
        for facet in local_facets:
            original = tuple(int(cell[index]) for index in facet)
            key = tuple(sorted(original))
            counts[key] = original
            repeats[key] = repeats.get(key, 0) + 1

    boundary = [counts[key] for key, count in repeats.items() if count == 1]
    if not boundary:
        return np.empty((0, len(local_facets[0])), dtype=np.int64)
    return np.asarray(boundary, dtype=np.int64) + 1


def _infer_electrode_node_groups(
    node_coords: np.ndarray, boundary_entities: np.ndarray, n_elec: int
) -> tuple[np.ndarray, np.ndarray]:
    if node_coords.shape[1] < 2:
        raise ValueError(t("interop.svc.err.need_2d_coords"))

    boundary_nodes = np.unique(np.asarray(boundary_entities, dtype=int).reshape(-1)) - 1
    if boundary_nodes.size == 0:
        raise ValueError(t("interop.svc.err.no_boundary_nodes"))

    coords = node_coords[boundary_nodes, :2]
    center = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
    order = np.argsort(angles)
    ordered_nodes = boundary_nodes[order]
    chunks = np.array_split(ordered_nodes, max(int(n_elec), 1))
    max_len = max(len(chunk) for chunk in chunks)
    padded = np.zeros((len(chunks), max_len), dtype=np.int64)
    counts = np.zeros(len(chunks), dtype=np.int64)
    for index, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        padded[index, : len(chunk)] = chunk + 1
        counts[index] = len(chunk)
    return padded, counts


def build_geometry_payload_from_result(
    *,
    node_coords: np.ndarray,
    cell_connectivity: np.ndarray,
    forward_model_config: ForwardModelConfig,
    truth_elem_data: np.ndarray | None = None,
    background: float | None = None,
    source_framework: str = "pyeidors",
    mesh_name: str = "pyeidors_export",
    scenario_name: str = "bridge_export",
    boundary_facets: np.ndarray | None = None,
    electrode_nodes: np.ndarray | None = None,
    electrode_node_counts: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a Geometry v2 payload from a simulation result."""

    nodes = np.asarray(node_coords, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] not in {2, 3}:
        raise ValueError("Geometry export requires 2D or 3D node coordinates")
    dimension = int(nodes.shape[1])
    elems = np.asarray(cell_connectivity, dtype=np.int64) + 1
    if elems.ndim != 2 or elems.shape[1] != dimension + 1:
        raise ValueError("Geometry export supports only 2D triangles or 3D tetrahedra")
    boundary_entities = (
        np.asarray(boundary_facets, dtype=np.int64)
        if boundary_facets is not None
        else _boundary_entities_from_cells(cell_connectivity)
    )
    if boundary_facets is not None and int(np.min(boundary_entities, initial=1)) < 1:
        boundary_entities = boundary_entities + 1
    electrode_model = str(forward_model_config.electrode_model).strip().lower()
    if electrode_model == "pem" and (
        electrode_nodes is None or electrode_node_counts is None
    ):
        raise ValueError(
            "PEM export requires exact singleton electrode_nodes from the "
            "forward mesh; surface-electrode inference is not permitted."
        )
    if electrode_nodes is None or electrode_node_counts is None:
        electrode_nodes, electrode_counts = _infer_electrode_node_groups(
            nodes,
            boundary_entities,
            forward_model_config.n_elec * max(forward_model_config.n_rings, 1),
        )
    else:
        electrode_nodes = np.asarray(electrode_nodes, dtype=np.int64)
        electrode_counts = np.asarray(
            electrode_node_counts,
            dtype=np.int64,
        ).reshape(-1)
        if electrode_nodes.ndim == 1:
            if electrode_counts.size == 1:
                electrode_nodes = electrode_nodes.reshape(1, -1)
            elif electrode_nodes.size == electrode_counts.size and np.all(
                electrode_counts == 1
            ):
                electrode_nodes = electrode_nodes.reshape(-1, 1)
        if (
            electrode_nodes.ndim != 2
            or electrode_nodes.shape[0] != electrode_counts.size
        ):
            raise ValueError(
                "Exact electrode nodes must have one padded row per electrode"
            )
        active_node_ids = [
            electrode_nodes[index, : int(count)]
            for index, count in enumerate(electrode_counts)
            if int(count) > 0
        ]
        if active_node_ids:
            active_node_ids_flat = np.concatenate(active_node_ids)
            if int(np.min(active_node_ids_flat, initial=1)) == 0:
                electrode_nodes = electrode_nodes.copy()
                for index, count in enumerate(electrode_counts):
                    electrode_nodes[index, : int(count)] += 1
    if electrode_model == "pem" and (
        electrode_counts.size == 0 or np.any(electrode_counts != 1)
    ):
        raise ValueError(
            "PEM export requires exactly one source node for every electrode"
        )
    if truth_elem_data is None:
        truth_elem = np.full(
            elems.shape[0],
            (
                forward_model_config.background_conductivity
                if background is None
                else background
            ),
        )
    else:
        truth_elem = np.asarray(truth_elem_data).reshape(-1)
    background_value = (
        forward_model_config.background_conductivity
        if background is None
        else background
    )
    configured_impedance = forward_model_config.contact_impedance
    if electrode_model == "pem" and configured_impedance is None:
        effective_impedance = 1.0
        impedance_status = "eidors_structural_placeholder_not_used_by_pem"
    elif electrode_model == "pem":
        effective_impedance = (
            configured_impedance
            if isinstance(configured_impedance, (int, float, complex))
            else np.asarray(configured_impedance)
        )
        impedance_status = "source_value_preserved_but_not_used_by_pem"
    elif configured_impedance is None:
        effective_impedance: float | complex | np.ndarray = 0.01
        impedance_status = "pyeidors_runtime_default"
    elif isinstance(configured_impedance, (int, float, complex)):
        effective_impedance = configured_impedance
        impedance_status = "exact_config"
    else:
        effective_impedance = np.asarray(configured_impedance)
        impedance_status = "exact_config"
    contact_unit = (
        "ohm*mesh_coordinate_unit" if dimension == 2 else "ohm*mesh_coordinate_unit^2"
    )
    total_electrodes = forward_model_config.n_elec * max(
        forward_model_config.n_rings, 1
    )
    source_gnd_node = forward_model_config.interop_semantics.get("effective_gnd_node")
    effective_gnd_node = (
        int(source_gnd_node)
        if source_gnd_node is not None and int(source_gnd_node) > 0
        else 1
    )
    export_semantics = {
        "schema": "eidors_pyeidors_capture_semantics_v1",
        "source_framework": "pyeidors",
        "fields": {
            "contact_impedance": {
                "status": impedance_status,
                "effective_value_exported": True,
                "applicable": electrode_model == "cem",
            },
            "background_image": {
                "status": "exact_config",
            },
            "target_image": {
                "status": "exact_result"
                if truth_elem_data is not None
                else "derived_from_background",
            },
            "gnd_node": {
                "status": "derived",
                "effective_source": "pyeidors_export_gauge_choice",
            },
            "normalize_measurements": {
                "status": "exact",
                "effective_source": "pyeidors_forward_unnormalized",
            },
        },
        "electrode_models": [
            "point" if electrode_model == "pem" else "cem"
            for _ in range(total_electrodes)
        ],
        "forward_ready": True,
        "forward_blockers": [],
        "forward_warnings": [],
    }
    payload: dict[str, Any] = {
        "exchange_format": STANDARD_INTEROP_FORMAT,
        "schema_version": 2,
        "index_base": 1,
        "source_framework": source_framework,
        "dimension": dimension,
        "cell_type": "triangle" if dimension == 2 else "tetrahedron",
        "boundary_entity_type": "edge" if dimension == 2 else "triangle",
        "nodes": nodes,
        "elems": elems,
        "boundary_edges": boundary_entities,
        "boundary_facets": boundary_entities,
        "electrode_nodes": electrode_nodes,
        "electrode_node_counts": electrode_counts,
        "n_elec": int(total_electrodes),
        "background": background_value,
        "background_present": True,
        "background_elem_data": np.full(elems.shape[0], background_value),
        "background_elem_data_present": True,
        "truth_elem_data": truth_elem,
        "truth_elem_data_present": True,
        "target_elem_data": truth_elem,
        "contact_impedance": effective_impedance,
        "contact_impedance_present": True,
        "contact_impedance_applicable": electrode_model == "cem",
        "contact_impedance_physical_present": bool(
            electrode_model == "cem" and configured_impedance is not None
        ),
        "contact_impedance_unit": contact_unit,
        "electrode_model": export_semantics["electrode_models"],
        "electrode_projection_required": np.zeros(
            len(export_semantics["electrode_models"]),
            dtype=bool,
        ),
        "model_coordinate_units": "mesh_coordinate_unit",
        "geometry_scale_to_m": float(forward_model_config.geometry_scale_to_m),
        "gnd_node": np.nan,
        "gnd_node_present": False,
        "effective_gnd_node": effective_gnd_node,
        "effective_gnd_node_source": (
            "preserved_eidors_effective_gnd_node"
            if source_gnd_node is not None
            else "pyeidors_export_gauge_choice"
        ),
        "normalize_measurements": False,
        "normalize_measurements_present": True,
        "normalize_measurements_source": "pyeidors_forward_unnormalized",
        "mesh_name": mesh_name,
        "mesh_level": "bridge_export",
        "scenario_name": scenario_name,
    }
    if (
        forward_model_config.custom_stim_matrix is not None
        and forward_model_config.custom_meas_matrices is not None
    ):
        stim_matrix = real_array_if_zero_imaginary(
            forward_model_config.custom_stim_matrix,
            name="PyEIDORS custom stimulation matrix",
        )
        if stim_matrix.ndim == 1:
            stim_matrix = stim_matrix.reshape(1, -1)
        raw_measurements = forward_model_config.custom_meas_matrices
        if isinstance(raw_measurements, (list, tuple)):
            measurement_list = [
                real_array_if_zero_imaginary(
                    matrix,
                    name="PyEIDORS custom measurement matrix",
                )
                for matrix in raw_measurements
            ]
        else:
            measurement_array = real_array_if_zero_imaginary(
                raw_measurements,
                name="PyEIDORS custom measurement matrices",
            )
            if measurement_array.ndim == 2:
                measurement_list = [
                    measurement_array.copy() for _ in range(stim_matrix.shape[0])
                ]
            elif measurement_array.ndim == 3:
                measurement_list = [
                    np.asarray(matrix, dtype=float) for matrix in measurement_array
                ]
            else:
                raise ValueError(
                    "Custom measurement matrices must be 2D, 3D, or a list"
                )
        if len(measurement_list) != stim_matrix.shape[0]:
            raise ValueError(
                "Custom stimulation and measurement matrix counts must match"
            )
        if any(
            matrix.ndim != 2 or matrix.shape[1] != stim_matrix.shape[1]
            for matrix in measurement_list
        ):
            raise ValueError(
                "Each custom measurement matrix must have n_electrodes columns"
            )
        max_measurements = max(matrix.shape[0] for matrix in measurement_list)
        padded = np.zeros(
            (len(measurement_list), max_measurements, stim_matrix.shape[1]),
            dtype=float,
        )
        counts = np.empty(len(measurement_list), dtype=np.int64)
        for index, matrix in enumerate(measurement_list):
            padded[index, : matrix.shape[0], :] = matrix
            counts[index] = matrix.shape[0]
        payload.update(
            {
                "stim_matrix_raw": stim_matrix,
                "stim_matrix": stim_matrix,
                "meas_matrices": padded,
                "measurement_counts": counts,
                "current_density": np.nan,
                "current_density_present": False,
                "current_density_applied": False,
                "stimulation_supported": True,
                "stim_positive_current": np.sum(np.maximum(stim_matrix, 0.0), axis=1),
                "stim_negative_current": -np.sum(np.minimum(stim_matrix, 0.0), axis=1),
                "stim_net_current": np.sum(stim_matrix, axis=1),
                "stim_max_abs_current": np.max(np.abs(stim_matrix), axis=1),
                "stim_balanced": np.abs(np.sum(stim_matrix, axis=1))
                <= (
                    1.0e-12
                    * np.maximum(
                        1.0,
                        np.max(np.abs(stim_matrix), axis=1),
                    )
                ),
            }
        )
        export_semantics["fields"]["stimulation"] = {
            "status": "exact_config",
            "raw_equals_effective": True,
        }
    else:
        export_semantics["fields"]["stimulation"] = {
            "status": "generated_at_eidors_import",
            "drive_value": float(forward_model_config.drive_value),
        }
    payload["capture_metadata_json"] = json.dumps(
        export_semantics,
        ensure_ascii=False,
    )
    return payload


class EidorsBridgeRunner:
    """Run a controlled MATLAB capture to produce a Bridge Package v2."""

    def run_capture(
        self,
        environment: EidorsEnvironment,
        script_path: str | Path,
        output_dir: str | Path,
        *,
        selectors: dict[str, str] | None = None,
    ) -> Path:
        if not environment.matlab_command:
            raise RuntimeError(t("interop.svc.err.no_matlab"))
        if not environment.eidors_startup:
            raise RuntimeError(t("interop.svc.err.no_startup"))

        source = Path(script_path)
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)

        hints = detect_script_hints(source)
        (root / CAPTURE_SCRIPT_NAME).write_text(
            CAPTURE_SCRIPT_TEMPLATE, encoding="utf-8"
        )
        request_payload = {
            "eidors_startup": matlab_runtime_path(
                environment.eidors_startup, environment
            ),
            "target_script": matlab_runtime_path(source, environment),
            "output_dir": matlab_runtime_path(root, environment),
            "script_kind": hints["script_kind"],
        }
        request_payload.update(
            {
                str(key): str(value)
                for key, value in dict(selectors or {}).items()
                if str(value).strip()
            }
        )
        request_path = root / CAPTURE_REQUEST_NAME
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        runtime_root = matlab_runtime_path(root, environment)
        runtime_request = matlab_runtime_path(request_path, environment)
        escaped_root = runtime_root.replace("'", "''")
        escaped_request = runtime_request.replace("'", "''")
        expression = (
            f"addpath('{escaped_root}'); run_capture_from_eidors('{escaped_request}');"
        )
        returncode, stdout, stderr = _run_command_capture(
            [matlab_command_for_execution(environment), "-batch", expression],
            timeout=180,
        )
        if returncode != 0:
            raise RuntimeError(
                stderr.strip() or stdout.strip() or "MATLAB bridge capture failed."
            )

        loaded = load_bridge_package(root)
        geometry_payload = loaded.geometry_payload
        measurements = loaded.measurements
        if geometry_payload is None:
            geometry_path = root / GEOMETRY_NAME
            if geometry_path.exists():
                geometry_payload = loadmat(
                    geometry_path, squeeze_me=True, struct_as_record=False
                )
        existing_cfg = _config_from_loaded_package(loaded)
        forward_cfg = existing_cfg.with_overrides(
            stim_pattern=hints.get("stim_pattern") or existing_cfg.stim_pattern,
            meas_pattern=hints.get("meas_pattern") or existing_cfg.meas_pattern,
        )

        notes: list[str] = []
        report_path = root / CAPTURE_REPORT_NAME
        if report_path.exists():
            notes.append(t("interop.svc.note.capture_report"))
        manifest = default_manifest(
            source_framework="eidors",
            package_kind="captured_script",
            environment=environment.to_mapping(),
            capabilities={
                "can_import_geometry": geometry_payload is not None,
                "can_import_measurements": measurements is not None,
                "can_capture_script": True,
                "can_run_equivalent_forward": not bool(
                    forward_cfg.interop_semantics.get("forward_blockers", [])
                ),
            },
            script_path=str(source),
            script_kind=str(hints.get("script_kind", "unknown")),
            script_hints=hints,
            notes=notes,
        )
        save_bridge_package(
            root,
            manifest,
            geometry_payload=geometry_payload,
            measurements=measurements,
            forward_model_config=forward_cfg,
            include_capture_script=True,
        )
        return root


class EidorsScriptCaptureService:
    """Facade that detects environments and captures user EIDORS scripts."""

    def __init__(
        self,
        detector: EidorsEnvironmentDetector | None = None,
        runner: EidorsBridgeRunner | None = None,
    ) -> None:
        self._detector = detector or EidorsEnvironmentDetector()
        self._runner = runner or EidorsBridgeRunner()

    def detect_environments(
        self,
    ) -> tuple[list[EidorsEnvironment], InteropCapabilityReport]:
        return self._detector.detect()

    def load_profiles(self) -> list[EidorsEnvironment]:
        return self._detector.load_profiles()

    def save_profiles(self, profiles: list[EidorsEnvironment]) -> None:
        self._detector.save_profiles(profiles)

    def save_last_environment(self, environment: EidorsEnvironment) -> None:
        self._detector.save_last_environment(environment)

    def test_matlab(self, environment: EidorsEnvironment) -> tuple[bool, str]:
        return self._detector.test_matlab_launch(environment)

    def test_startup(self, environment: EidorsEnvironment) -> tuple[bool, str]:
        return self._detector.test_eidors_startup(environment)

    def infer_startup_from_source(self, source_path: str | Path) -> str:
        return self._detector.infer_startup_from_source(source_path)

    def capture_or_load(
        self,
        source_path: str | Path,
        environment: EidorsEnvironment | None = None,
        output_dir: str | Path | None = None,
    ) -> LoadedBridgePackage:
        source = Path(to_posix_path(source_path))
        if source.is_dir() and (
            (source / "manifest.json").exists()
            or (source / GEOMETRY_NAME).exists()
            or (source / CONFIG_NAME).exists()
        ):
            return load_bridge_package(source)
        if source.is_file() and source.suffix.lower() in {".mat", ".json"}:
            return load_bridge_package(source)
        if source.is_file() and source.suffix.lower() == ".m":
            if environment is None:
                raise RuntimeError(t("interop.svc.err.need_env_for_script"))
            target_dir = (
                Path(to_posix_path(output_dir))
                if output_dir
                else source.parent / f"{source.stem}_bridge"
            )
            bundle_dir = self._runner.run_capture(environment, source, target_dir)
            return load_bridge_package(bundle_dir)
        raise RuntimeError(t("interop.svc.err.unsupported_source"))


class InteropBundleImporter:
    """Build previews and normalized assets from a bridge package."""

    def load_package(self, path: str | Path) -> LoadedBridgePackage:
        return load_bridge_package(path)

    def preview_package(
        self, path: str | Path
    ) -> tuple[LoadedBridgePackage, EidorsImportPreview]:
        loaded = load_bridge_package(path)
        return loaded, _build_preview(loaded)

    def preview_loaded_package(
        self, loaded: LoadedBridgePackage
    ) -> EidorsImportPreview:
        return _build_preview(loaded)


class InteropBundleExporter:
    """Export PyEIDORS runtime state to a Bridge Package v2 directory."""

    def export_bundle(
        self,
        job: EidorsExportJob,
        *,
        forward_model_config: ForwardModelConfig,
        environment: EidorsEnvironment | None = None,
        geometry_payload: dict[str, Any] | None = None,
        measurements: dict[str, np.ndarray] | None = None,
        reconstruction_preset: ReconstructionPreset | None = None,
        notes: list[str] | None = None,
    ) -> Path:
        root = Path(to_posix_path(job.output_dir))
        root.mkdir(parents=True, exist_ok=True)

        effective_geometry = geometry_payload if job.include_geometry else None
        effective_measurements = measurements if job.include_measurements else None
        effective_notes = list(notes or [])
        if job.include_geometry and effective_geometry is None:
            effective_notes.append(t("interop.svc.note.no_geometry_export"))
        if job.include_measurements and effective_measurements is None:
            effective_notes.append(t("interop.svc.note.no_measurements_export"))

        manifest = default_manifest(
            source_framework="pyeidors",
            package_kind="export_project",
            environment=environment.to_mapping() if environment else {},
            capabilities={
                "can_import_geometry": effective_geometry is not None,
                "can_import_measurements": effective_measurements is not None,
            },
            notes=effective_notes,
        )
        save_bridge_package(
            root,
            manifest,
            geometry_payload=effective_geometry,
            measurements=effective_measurements,
            forward_model_config=forward_model_config,
            reconstruction_preset=reconstruction_preset,
            include_run_in_eidors_script=job.include_scripts
            and effective_geometry is not None,
        )

        runtime_payload = {
            "eidors_startup": to_windows_path(environment.eidors_startup)
            if environment
            else "",
            "geometry_mat": to_windows_path((root / GEOMETRY_NAME).resolve()),
            "measurements_csv": to_windows_path((root / "measurements.csv").resolve()),
            "measurements_mat": to_windows_path((root / "measurements.mat").resolve()),
            "stim_pattern": forward_model_config.stim_pattern,
            "meas_pattern": forward_model_config.meas_pattern,
            "rotate_meas": forward_model_config.rotate_meas,
            "use_meas_current": forward_model_config.use_meas_current,
            "drive_value": forward_model_config.drive_value,
        }
        (root / BRIDGE_RUNTIME_NAME).write_text(
            json.dumps(runtime_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            (root / BRIDGE_MANIFEST_ALIAS).write_text(
                manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        run_script = root / "run_in_eidors.m"
        if run_script.exists():
            (root / RUN_IMPORT_FROM_PYEIDORS_NAME).write_text(
                run_script.read_text(encoding="utf-8"), encoding="utf-8"
            )
        return root


class InteropSmokeValidator:
    """Run a lightweight import smoke check, optionally including inverse solve."""

    def validate(
        self,
        loaded: LoadedBridgePackage,
        *,
        reconstruction_preset: ReconstructionPreset | None = None,
    ) -> dict[str, Any]:
        config = _config_from_loaded_package(loaded)
        config.require_interop_forward_ready()
        measurements = loaded.measurements or {}
        homogeneous = measurements.get("homogeneous")
        target = measurements.get("target")
        difference = measurements.get("difference")

        if target is None and homogeneous is not None and difference is not None:
            target = np.asarray(homogeneous, dtype=float).reshape(-1) + np.asarray(
                difference, dtype=float
            ).reshape(-1)
        if homogeneous is None and target is not None and difference is not None:
            homogeneous = np.asarray(target, dtype=float).reshape(-1) - np.asarray(
                difference, dtype=float
            ).reshape(-1)
        if homogeneous is None or target is None:
            raise RuntimeError(t("interop.svc.err.smoke_needs_two"))

        homogeneous = np.asarray(homogeneous, dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        metadata = {
            **config.to_mapping(),
            "difference_mode": "raw",
            "difference_orientation": "target_minus_reference",
        }

        MeasurementDataset.from_metadata(
            homogeneous.reshape(1, -1), metadata, data_type="real"
        )
        MeasurementDataset.from_metadata(
            target.reshape(1, -1), metadata, data_type="real"
        )

        result: dict[str, Any] = {
            "status": "compatible",
            "n_measurements": int(target.size),
            "mesh_dimension": int(config.mesh_dimension),
            "message": t("interop.svc.smoke.compat_ok", count=int(target.size)),
        }

        preset = (
            reconstruction_preset
            or loaded.reconstruction_preset
            or ReconstructionPreset()
        )
        from pyeidors import EITSystem
        from pyeidors.data import PatternConfig

        pattern = PatternConfig(
            n_elec=config.n_elec,
            n_rings=config.n_rings,
            stim_pattern=config.stim_pattern,
            meas_pattern=config.meas_pattern,
            electrode_layout=config.electrode_layout,
            measurement_protocol=config.measurement_protocol,
            custom_stim_matrix=config.custom_stim_matrix,
            custom_meas_matrices=config.custom_meas_matrices,
            drive_mode=config.drive_mode,
            drive_value=config.drive_value,
            geometry_scale_to_m=config.geometry_scale_to_m,
            electrode_length_m_override=config.electrode_length_m_override,
            use_meas_current=config.use_meas_current,
            use_meas_current_next=config.use_meas_current_next,
            rotate_meas=config.rotate_meas,
            stim_direction=config.stim_direction,
            meas_direction=config.meas_direction,
            stim_first_positive=config.stim_first_positive,
        )
        system = EITSystem(
            n_elec=config.total_electrodes(),
            pattern_config=pattern,
            electrode_model=config.electrode_model,
            contact_impedance=config.contact_impedance,
            base_conductivity=config.background_conductivity,
            regularization_alpha=float(preset.regularization_alpha),
            difference_mode=str(preset.difference_mode),
            difference_orientation=str(preset.difference_orientation),
            potential_order=config.potential_order,
        )
        if config.mesh_source == "interop":
            from pyeidors.interop import build_mesh_from_exchange_mat

            mesh_path = Path(config.mesh_path)
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    "Imported EIDORS geometry file was not found: "
                    f"{mesh_path or '<empty>'}. Reload the Bridge Package."
                )
            imported_mesh, geometry_payload = build_mesh_from_exchange_mat(mesh_path)
            system.setup(
                mesh=imported_mesh,
                initialize_inverse=int(config.mesh_dimension) != 3,
            )
            result.update(
                {
                    "mesh_source": "interop",
                    "geometry_format": str(
                        np.asarray(geometry_payload["exchange_format"]).reshape(-1)[0]
                    ),
                    "n_nodes": imported_mesh.num_vertices(),
                    "n_elements": imported_mesh.num_cells(),
                    "n_electrodes": int(imported_mesh.n_electrodes),
                }
            )
        else:
            system.setup(
                mesh_source="generated",
                dimension=config.mesh_dimension,
                mesh_size=config.mesh_refinement,
                radius=config.radius,
                height=config.height,
                electrode_height_ratio=config.electrode_height_ratio,
                electrode_level_fractions=config.electrode_level_fractions,
                z_center=config.z_center,
                mesh_family=config.mesh_family,
                geometry_version=config.geometry_version,
                initialize_inverse=int(config.mesh_dimension) != 3,
            )

        if int(config.mesh_dimension) == 3:
            result["status"] = "mesh_loaded"
            result["message"] += t("interop.svc.smoke.compat_3d_suffix")
            return result

        ref_data = MeasurementDataset.from_metadata(
            homogeneous.reshape(1, -1), metadata, data_type="real"
        ).to_eit_data(0)
        tgt_data = MeasurementDataset.from_metadata(
            target.reshape(1, -1), metadata, data_type="real"
        ).to_eit_data(0)

        method = str(preset.method or "").strip().lower()
        if method in {"gn-absolute", "eidors_abs_gn"}:
            recon = system.absolute_reconstruct(measurement_data=tgt_data)
        else:
            recon = system.difference_reconstruct(
                measurement_data=tgt_data, reference_data=ref_data
            )

        conductivity = np.asarray(
            getattr(recon, "conductivity", np.asarray([]))
        ).reshape(-1)
        result.update(
            {
                "status": "inverse_ok",
                "n_elements": int(conductivity.size),
                "message": t(
                    "interop.svc.smoke.inverse_ok",
                    count=int(target.size),
                    method=preset.method,
                    n_elements=int(conductivity.size),
                ),
            }
        )
        return result
