"""Bridge Package v3 helpers for EIDORS <-> PyEIDORS exchange."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors.interop import (
    BRIDGE_PACKAGE_FORMAT_V3,
    BridgeV3Package,
    validate_exchange_payload,
    validate_bridge_v3_package,
)
from pyeidors.interop.geometry_exchange import source_electrode_models
from pyeidors.io._json import json_ready

from .matlab_templates import CAPTURE_SCRIPT_TEMPLATE, RUN_IN_EIDORS_TEMPLATE
from .models import (
    BRIDGE_PACKAGE_FORMAT_V3 as APP_BRIDGE_PACKAGE_FORMAT_V3,
    InteropBridgeManifest,
    ReconstructionPreset,
)

MANIFEST_NAME = "manifest.json"
GEOMETRY_NAME = "geometry.mat"
MEASUREMENTS_MAT_NAME = "measurements.mat"
CONFIG_NAME = "model.json"
PRESET_NAME = "reconstruction.json"
CAPTURE_SCRIPT_NAME = "run_capture_from_eidors.m"
RUN_IN_EIDORS_NAME = "run_in_eidors.m"
RUN_IMPORT_FROM_PYEIDORS_NAME = "run_import_from_pyeidors.m"

_PROTOCOL_STAGING_KEYS = {
    "stim_matrix",
    "stim_matrix_raw",
    "meas_matrices",
    "measurement_matrix",
    "measurement_counts",
    "meas_select",
    "meas_select_present",
    "n2e",
    "N2E",
    "qq",
    "QQ",
    "vv",
    "VV",
    "v2meas",
    "current_density",
    "current_density_present",
    "current_density_applied",
    "stimulation_supported",
    "normalize_measurements",
    "normalize_measurements_present",
    "normalize_measurements_source",
    "stim_positive_current",
    "stim_negative_current",
    "stim_net_current",
    "stim_max_abs_current",
    "stim_balanced",
    "volt_matrix",
    "volt_pattern_present",
    "interior_sources",
    "interior_source_counts",
    "stimulation_labels",
}


@dataclass
class LoadedBridgePackage:
    """Loaded bridge package content."""

    root: Path
    manifest: InteropBridgeManifest
    geometry_payload: dict[str, Any] | None = None
    protocol_payload: dict[str, Any] | None = None
    fields_payload: dict[str, Any] | None = None
    measurements: dict[str, np.ndarray] | None = None
    forward_model_config: ForwardModelConfig | None = None
    reconstruction_preset: ReconstructionPreset | None = None


def _json_default(value: Any) -> Any:
    converted = json_ready(value)
    if converted is value:
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )
    return converted


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _split_bridge_v3_staging_payload(
    geometry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = {
        key: value for key, value in geometry.items() if key in _PROTOCOL_STAGING_KEYS
    }
    fields = {
        key: value
        for key, value in geometry.items()
        if key.startswith(("background", "target", "truth", "coarse2fine"))
    }
    return protocol, fields


def default_manifest(
    *,
    source_framework: str,
    package_kind: str,
    environment: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
    files: dict[str, str] | None = None,
    script_path: str = "",
    script_kind: str = "",
    script_hints: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> InteropBridgeManifest:
    return InteropBridgeManifest(
        exchange_format=APP_BRIDGE_PACKAGE_FORMAT_V3,
        source_framework=source_framework,
        package_kind=package_kind,
        created_at_utc=utc_now_iso(),
        environment=dict(environment or {}),
        capabilities=dict(capabilities or {}),
        files=dict(files or {}),
        script_path=script_path,
        script_kind=script_kind,
        script_hints=dict(script_hints or {}),
        notes=list(notes or []),
    )


def save_bridge_package(
    output_dir: str | Path,
    manifest: InteropBridgeManifest,
    *,
    geometry_payload: dict[str, Any] | None = None,
    measurements: dict[str, np.ndarray] | None = None,
    forward_model_config: ForwardModelConfig | None = None,
    reconstruction_preset: ReconstructionPreset | None = None,
    include_capture_script: bool = False,
    include_run_in_eidors_script: bool = False,
) -> Path:
    if geometry_payload is None:
        raise ValueError("Bridge v3 requires geometry_payload")
    geometry = dict(geometry_payload)
    config_mapping = (
        forward_model_config.to_mapping() if forward_model_config is not None else {}
    )
    config_mapping.update({"mesh_source": "interop", "mesh_path": GEOMETRY_NAME})
    n_elec = int(
        np.asarray(geometry.get("n_elec", config_mapping.get("n_elec", 0))).reshape(-1)[
            0
        ]
    )
    dimension = int(
        np.asarray(
            geometry.get(
                "dimension",
                np.asarray(geometry.get("nodes", np.empty((0, 2)))).shape[1],
            )
        ).reshape(-1)[0]
    )
    metadata: dict[str, Any] = {}
    metadata_raw = np.asarray(geometry.get("capture_metadata_json", [])).reshape(-1)
    if metadata_raw.size:
        try:
            parsed = json.loads(str(metadata_raw[0]))
            if isinstance(parsed, dict):
                metadata = parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
    model_payload = {
        "schema_version": 3,
        "forward_model_config": config_mapping,
        "n_elec": n_elec,
        "dimension": dimension,
        "potential_order": int(config_mapping.get("potential_order", 1) or 1),
        "effective_gnd_node": json_ready(geometry.get("effective_gnd_node")),
        "normalize_measurements": bool(
            np.asarray(geometry.get("normalize_measurements", False)).reshape(-1)[0]
        ),
        "forward_ready": bool(metadata.get("forward_ready", True)),
        "forward_blockers": [
            str(item) for item in metadata.get("forward_blockers", [])
        ],
    }
    protocol_payload, fields_payload = _split_bridge_v3_staging_payload(geometry)
    if "stim_matrix" not in protocol_payload:
        custom_stim = config_mapping.get("custom_stim_matrix")
        protocol_payload["stim_matrix"] = (
            np.asarray(custom_stim)
            if custom_stim is not None
            else np.empty((0, n_elec), dtype=float)
        )
    if "meas_matrices" not in protocol_payload:
        custom_meas = config_mapping.get("custom_meas_matrices")
        if custom_meas is not None:
            protocol_payload["meas_matrices"] = np.asarray(
                custom_meas,
                dtype=object,
            )
    if not fields_payload:
        fields_payload["background_present"] = False
    reconstruction_mapping = (
        reconstruction_preset.to_mapping()
        if reconstruction_preset is not None
        else None
    )
    package = BridgeV3Package.write(
        output_dir,
        model=model_payload,
        geometry=geometry,
        protocol=protocol_payload,
        fields=fields_payload,
        measurements=measurements,
        reconstruction=reconstruction_mapping,
        source_framework=manifest.source_framework,
        package_kind=manifest.package_kind,
        provenance={
            "created_at_utc": manifest.created_at_utc or utc_now_iso(),
            "environment": manifest.environment,
            "script_path": manifest.script_path,
            "script_kind": manifest.script_kind,
            "script_hints": manifest.script_hints,
            "notes": manifest.notes,
        },
        capabilities=manifest.capabilities,
    )
    root = package.root
    if include_capture_script:
        (root / CAPTURE_SCRIPT_NAME).write_text(
            CAPTURE_SCRIPT_TEMPLATE, encoding="utf-8"
        )
    if include_run_in_eidors_script:
        (root / RUN_IN_EIDORS_NAME).write_text(RUN_IN_EIDORS_TEMPLATE, encoding="utf-8")
        (root / RUN_IMPORT_FROM_PYEIDORS_NAME).write_text(
            RUN_IN_EIDORS_TEMPLATE, encoding="utf-8"
        )
    return root


def _load_manifest(root: Path) -> InteropBridgeManifest:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        return default_manifest(source_framework="unknown", package_kind="legacy")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return InteropBridgeManifest.from_mapping(payload)


def _load_forward_model_config(
    root: Path, manifest: InteropBridgeManifest
) -> ForwardModelConfig | None:
    config_name = manifest.files.get("model", CONFIG_NAME)
    config_path = root / config_name
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "forward_model_config" in payload:
        return ForwardModelConfig.from_mapping(payload["forward_model_config"])
    return ForwardModelConfig.from_mapping(payload)


def _load_reconstruction_preset(
    root: Path, manifest: InteropBridgeManifest
) -> ReconstructionPreset | None:
    preset_name = manifest.files.get("reconstruction", PRESET_NAME)
    preset_path = root / preset_name
    if not preset_path.exists():
        return None
    return ReconstructionPreset(**json.loads(preset_path.read_text(encoding="utf-8")))


def _public_mat_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("__")}


def _contact_config_value(payload: dict[str, Any]) -> Any:
    if "contact_impedance" not in payload:
        return None
    values = np.asarray(payload["contact_impedance"]).reshape(-1)
    if values.size == 0:
        return None
    present = np.asarray(
        payload.get("contact_impedance_present", np.ones(values.size, dtype=bool)),
        dtype=bool,
    ).reshape(-1)
    if present.size == 1:
        present = np.full(values.size, bool(present[0]), dtype=bool)
    if present.size != values.size or not np.all(present):
        return None

    def scalar(item: Any) -> float | complex:
        number = complex(item)
        return float(number.real) if abs(number.imag) <= 1.0e-15 else number

    if values.size == 1:
        return scalar(values[0])
    return [scalar(item) for item in values]


def _load_measurements(
    root: Path, manifest: InteropBridgeManifest
) -> dict[str, np.ndarray] | None:
    mat_name = manifest.files.get("measurements", MEASUREMENTS_MAT_NAME)
    mat_path = root / mat_name
    if mat_path.exists():
        payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return {
            key: np.asarray(value)
            for key, value in payload.items()
            if not key.startswith("__")
        }
    return None


def load_bridge_package(path: str | Path) -> LoadedBridgePackage:
    package = BridgeV3Package.load(path)
    root = package.root
    manifest = InteropBridgeManifest.from_mapping(dict(package.manifest))
    geometry_payload = dict(package.geometry)
    model_mapping = dict(package.model)
    raw_config = model_mapping.get("forward_model_config", model_mapping)
    forward_model_config = ForwardModelConfig.from_mapping(dict(raw_config))
    geometry_path = root / manifest.files["geometry"]
    forward_model_config = forward_model_config.with_overrides(
        mesh_source="interop",
        mesh_path=str(geometry_path.resolve()),
    )
    reconstruction_preset = None
    if package.reconstruction is not None:
        reconstruction_preset = ReconstructionPreset(**dict(package.reconstruction))
    return LoadedBridgePackage(
        root=root,
        manifest=manifest,
        geometry_payload=geometry_payload,
        protocol_payload=dict(package.protocol),
        fields_payload=dict(package.fields),
        measurements=(
            None
            if package.measurements is None
            else {key: np.asarray(value) for key, value in package.measurements.items()}
        ),
        forward_model_config=forward_model_config,
        reconstruction_preset=reconstruction_preset,
    )


def validate_bridge_package(path: str | Path) -> dict[str, Any]:
    """Validate a Bridge Package v3 directory."""

    source = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "schema": "eidors_pyeidors_bridge_v3_validation_v1",
        "path": str(source.resolve()),
        "valid": False,
        "package_format": "",
        "model_id": "",
        "forward_fingerprint": "",
        "protocol_layout_hash": "",
        "protocol_physics_hash": "",
        "geometry_format": "",
        "dimension": None,
        "cell_type": "",
        "n_nodes": 0,
        "n_elements": 0,
        "n_boundary_facets": 0,
        "n_electrodes": 0,
        "n_stimulations": 0,
        "n_measurements": 0,
        "electrode_definition": "",
        "electrode_models": [],
        "contact_impedance_present": [],
        "contact_impedance_unit": "",
        "background_present": None,
        "stimulation_supported": None,
        "current_density_present": None,
        "current_density_applied": None,
        "stim_positive_current": [],
        "stim_negative_current": [],
        "stim_net_current": [],
        "stim_max_abs_current": [],
        "stim_balanced": [],
        "forward_ready": None,
        "forward_blockers": [],
        "forward_warnings": [],
        "files": {},
        "warnings": warnings,
        "errors": errors,
    }
    core_report = validate_bridge_v3_package(source)
    if not core_report["valid"]:
        errors.extend(str(item) for item in core_report["errors"])
        return report
    try:
        package = BridgeV3Package.load(source)
        root = package.root
        manifest = InteropBridgeManifest.from_mapping(dict(package.manifest))
        geometry = dict(package.geometry)
        protocol = dict(package.protocol)
        fields = dict(package.fields)
        validate_exchange_payload(geometry)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Invalid geometry MAT: {exc}")
        return report
    report.update(
        {
            "package_format": BRIDGE_PACKAGE_FORMAT_V3,
            "model_id": package.model_id,
            "forward_fingerprint": package.forward_fingerprint,
            "protocol_layout_hash": str(package.manifest["protocol_layout_hash"]),
            "protocol_physics_hash": str(package.manifest["protocol_physics_hash"]),
            "files": dict(core_report["files"]),
        }
    )

    nodes = np.asarray(geometry["nodes"])
    elems = np.atleast_2d(np.asarray(geometry["elems"]))
    boundary_key = (
        "boundary_facets" if "boundary_facets" in geometry else "boundary_edges"
    )
    boundary = np.atleast_2d(np.asarray(geometry[boundary_key]))
    dimension = int(nodes.shape[1])
    report.update(
        {
            "geometry_format": str(
                np.asarray(geometry["exchange_format"]).reshape(-1)[0]
            ),
            "dimension": dimension,
            "cell_type": str(
                np.asarray(
                    geometry.get(
                        "cell_type",
                        "triangle" if dimension == 2 else "tetrahedron",
                    )
                ).reshape(-1)[0]
            ),
            "n_nodes": int(nodes.shape[0]),
            "n_elements": int(elems.shape[0]),
            "n_boundary_facets": int(boundary.shape[0]),
            "n_electrodes": int(np.asarray(geometry["n_elec"]).reshape(-1)[0]),
            "electrode_definition": (
                "point_or_lower_dimensional"
                if np.any(
                    np.asarray(
                        geometry["electrode_node_counts"],
                        dtype=np.int64,
                    ).reshape(-1)
                    < dimension
                )
                else "surface_nodes"
            ),
        }
    )
    metadata: dict[str, Any] = {}
    if "capture_metadata_json" in geometry:
        metadata_raw = np.asarray(geometry["capture_metadata_json"]).reshape(-1)
        if metadata_raw.size:
            try:
                parsed = json.loads(str(metadata_raw[0]))
                if isinstance(parsed, dict):
                    metadata = parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                warnings.append("capture_metadata_json could not be decoded.")
    electrode_models = source_electrode_models(geometry)
    report["electrode_models"] = electrode_models
    pem_models = {"point", "distributed_point"}
    cem_models = {"cem", "cem_faces"}
    all_point_electrodes = bool(
        electrode_models and all(model in pem_models for model in electrode_models)
    )
    all_cem_electrodes = bool(
        electrode_models and all(model in cem_models for model in electrode_models)
    )
    report["electrode_model"] = (
        "pem" if all_point_electrodes else ("cem" if all_cem_electrodes else "mixed")
    )
    report["electrode_projection"] = (
        "exact_weighted_n2e"
        if all_point_electrodes
        else ("exact_cem_faces" if all_cem_electrodes else "exact_per_electrode")
    )
    if electrode_models:
        report["electrode_definition"] = (
            "point_or_distributed_point"
            if any(
                model in {"point", "distributed_point"} for model in electrode_models
            )
            else "complete_electrode_model"
        )
    contact_present = np.asarray(
        geometry.get(
            "contact_impedance_present",
            np.ones(report["n_electrodes"], dtype=bool),
        ),
        dtype=bool,
    ).reshape(-1)
    if contact_present.size == 1:
        contact_present = np.full(
            report["n_electrodes"],
            bool(contact_present[0]),
            dtype=bool,
        )
    report["contact_impedance_present"] = contact_present.tolist()
    if "contact_impedance_unit" in geometry:
        report["contact_impedance_unit"] = str(
            np.asarray(geometry["contact_impedance_unit"]).reshape(-1)[0]
        )
    report["background_present"] = bool(
        np.asarray(fields.get("background_present", True)).reshape(-1)[0]
    )
    if "stimulation_supported" in protocol:
        report["stimulation_supported"] = bool(
            np.asarray(protocol["stimulation_supported"]).reshape(-1)[0]
        )
    if "current_density_present" in protocol:
        report["current_density_present"] = bool(
            np.asarray(protocol["current_density_present"]).reshape(-1)[0]
        )
    if "current_density_applied" in protocol:
        report["current_density_applied"] = bool(
            np.asarray(protocol["current_density_applied"]).reshape(-1)[0]
        )
    for source_name, report_name in (
        ("stim_positive_current", "stim_positive_current"),
        ("stim_negative_current", "stim_negative_current"),
        ("stim_net_current", "stim_net_current"),
        ("stim_max_abs_current", "stim_max_abs_current"),
        ("stim_balanced", "stim_balanced"),
    ):
        if source_name in protocol:
            report[report_name] = json_ready(
                np.asarray(protocol[source_name]).reshape(-1)
            )
    blockers = [
        str(item)
        for item in metadata.get("forward_blockers", [])
        if str(item)
        not in {
            "point_or_distributed_point_electrode_requires_explicit_projection_opt_in",
            "distributed_point_electrode_requires_explicit_projection_opt_in",
            "mixed_cem_pem_electrode_models_not_supported",
            "background_is_nonuniform_and_not_gui_scalar_compatible",
        }
    ]
    cem_mask = np.asarray(
        [model in cem_models for model in electrode_models],
        dtype=bool,
    )
    if (
        contact_present.size != report["n_electrodes"]
        or cem_mask.size != report["n_electrodes"]
        or np.any(~contact_present & cem_mask)
    ):
        blockers.append("contact_impedance_missing_no_eidors_default")
    if not report["background_present"]:
        background_elem_present = bool(
            np.asarray(
                fields.get(
                    "background_elem_data_present",
                    "background_elem_data" in fields,
                )
            ).reshape(-1)[0]
        )
        background_elem_data = np.asarray(fields.get("background_elem_data", []))
        if background_elem_present and background_elem_data.size:
            warnings.append(
                "Background uses an exact element field without a scalar summary."
            )
        else:
            blockers.append("background_image_missing_or_unmappable")
    if report["stimulation_supported"] is False:
        blockers.append(
            "stimulation_missing_or_unsupported_voltage_interior_complex_pattern"
        )
    blockers = list(dict.fromkeys(blockers))
    forward_notes = [
        str(item) for item in metadata.get("forward_warnings", []) if str(item)
    ]
    report["forward_blockers"] = blockers
    report["forward_warnings"] = forward_notes
    report["forward_ready"] = not blockers
    warnings.extend(forward_notes)
    if blockers:
        warnings.append(
            "Geometry is valid, but an equivalent PyEIDORS forward solve is "
            "blocked: " + ", ".join(blockers)
        )
    if "stim_matrix" in protocol:
        stim = np.asarray(protocol["stim_matrix"])
        report["n_stimulations"] = int(1 if stim.ndim == 1 else stim.shape[0])
    if "measurement_counts" in protocol:
        report["n_measurements"] = int(
            np.sum(np.asarray(protocol["measurement_counts"], dtype=np.int64))
        )
    else:
        measurements = _load_measurements(root, manifest)
        if measurements:
            first = next(iter(measurements.values()))
            report["n_measurements"] = int(np.asarray(first).size)

    report["valid"] = not errors
    return report
