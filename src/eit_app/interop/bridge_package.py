"""Bridge Package v2 helpers for EIDORS <-> PyEIDORS exchange."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors.interop import (
    SUPPORTED_INTEROP_FORMATS,
    export_forward_csv,
    load_forward_csv,
    save_exchange_mat,
    validate_exchange_payload,
)
from pyeidors.interop.geometry_exchange import source_electrode_models
from pyeidors.io._json import json_ready
from pyeidors.utils.numeric_ops import real_array_if_zero_imaginary

from .matlab_templates import CAPTURE_SCRIPT_TEMPLATE, RUN_IN_EIDORS_TEMPLATE
from .models import (
    BRIDGE_PACKAGE_FORMAT_V2,
    InteropBridgeManifest,
    ReconstructionPreset,
)

MANIFEST_NAME = "manifest.json"
GEOMETRY_NAME = "geometry.mat"
MEASUREMENTS_CSV_NAME = "measurements.csv"
MEASUREMENTS_MAT_NAME = "measurements.mat"
CONFIG_NAME = "config.json"
PRESET_NAME = "reconstruction_preset.json"
CAPTURE_SCRIPT_NAME = "run_capture_from_eidors.m"
RUN_IN_EIDORS_NAME = "run_in_eidors.m"
RUN_IMPORT_FROM_PYEIDORS_NAME = "run_import_from_pyeidors.m"


@dataclass
class LoadedBridgePackage:
    """Loaded bridge package content."""

    root: Path
    manifest: InteropBridgeManifest
    geometry_payload: dict[str, Any] | None = None
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
        exchange_format=BRIDGE_PACKAGE_FORMAT_V2,
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
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {}

    if geometry_payload is not None:
        geometry_path = root / GEOMETRY_NAME
        exchange_format = str(
            np.asarray(geometry_payload.get("exchange_format", "")).reshape(-1)[0]
        )
        if exchange_format in SUPPORTED_INTEROP_FORMATS:
            save_exchange_mat(geometry_path, geometry_payload)
        else:
            savemat(geometry_path, geometry_payload)
        files["geometry"] = GEOMETRY_NAME

    if measurements:
        if {"homogeneous", "target"}.issubset(measurements):
            try:
                homogeneous = real_array_if_zero_imaginary(
                    measurements["homogeneous"],
                    name="homogeneous Bridge measurements",
                ).reshape(-1)
                target = real_array_if_zero_imaginary(
                    measurements["target"],
                    name="target Bridge measurements",
                ).reshape(-1)
            except (TypeError, ValueError):
                savemat(
                    root / MEASUREMENTS_MAT_NAME,
                    {key: np.asarray(value) for key, value in measurements.items()},
                )
                files["measurements_mat"] = MEASUREMENTS_MAT_NAME
            else:
                export_forward_csv(
                    root / MEASUREMENTS_CSV_NAME,
                    homogeneous,
                    target,
                )
                files["measurements_csv"] = MEASUREMENTS_CSV_NAME
        else:
            savemat(
                root / MEASUREMENTS_MAT_NAME,
                {key: np.asarray(value) for key, value in measurements.items()},
            )
            files["measurements_mat"] = MEASUREMENTS_MAT_NAME

    if forward_model_config is not None:
        config_path = root / CONFIG_NAME
        config_mapping = forward_model_config.to_mapping()
        if geometry_payload is not None:
            config_mapping.update(
                {
                    "mesh_source": "interop",
                    "mesh_path": GEOMETRY_NAME,
                }
            )
        config_path.write_text(
            json.dumps(
                {
                    "forward_model_config": config_mapping,
                    "exchange_format": BRIDGE_PACKAGE_FORMAT_V2,
                },
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        files["config"] = CONFIG_NAME

    if reconstruction_preset is not None:
        preset_path = root / PRESET_NAME
        preset_path.write_text(
            json.dumps(
                reconstruction_preset.to_mapping(), ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        files["reconstruction_preset"] = PRESET_NAME

    if include_capture_script:
        (root / CAPTURE_SCRIPT_NAME).write_text(
            CAPTURE_SCRIPT_TEMPLATE, encoding="utf-8"
        )
        files["capture_script"] = CAPTURE_SCRIPT_NAME

    if include_run_in_eidors_script:
        (root / RUN_IN_EIDORS_NAME).write_text(RUN_IN_EIDORS_TEMPLATE, encoding="utf-8")
        (root / RUN_IMPORT_FROM_PYEIDORS_NAME).write_text(
            RUN_IN_EIDORS_TEMPLATE, encoding="utf-8"
        )
        files["run_in_eidors"] = RUN_IN_EIDORS_NAME
        files["run_import_from_pyeidors"] = RUN_IMPORT_FROM_PYEIDORS_NAME

    merged_manifest = InteropBridgeManifest.from_mapping(
        {**manifest.to_mapping(), "files": {**manifest.files, **files}}
    )
    (root / MANIFEST_NAME).write_text(
        json.dumps(
            merged_manifest.to_mapping(),
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
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
    config_name = manifest.files.get("config", CONFIG_NAME)
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
    preset_name = manifest.files.get("reconstruction_preset", PRESET_NAME)
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
    csv_name = manifest.files.get("measurements_csv", MEASUREMENTS_CSV_NAME)
    mat_name = manifest.files.get("measurements_mat", MEASUREMENTS_MAT_NAME)
    csv_path = root / csv_name
    mat_path = root / mat_name
    if csv_path.exists():
        homogeneous, target, difference = load_forward_csv(csv_path)
        return {
            "homogeneous": homogeneous,
            "target": target,
            "difference": difference,
        }
    if mat_path.exists():
        payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        return {
            key: np.asarray(value)
            for key, value in payload.items()
            if not key.startswith("__")
        }
    return None


def load_bridge_package(path: str | Path) -> LoadedBridgePackage:
    source = Path(path)
    if source.is_file() and source.suffix.lower() == ".mat":
        payload = _public_mat_payload(
            loadmat(source, squeeze_me=True, struct_as_record=False)
        )
        manifest = default_manifest(
            source_framework=str(
                np.asarray(payload.get("source_framework", "eidors")).reshape(-1)[0]
            ),
            package_kind="legacy_geometry",
            files={"geometry": source.name},
            notes=[
                "Imported from a legacy .mat exchange payload without Bridge Package v2 manifest."
            ],
        )
        config = ForwardModelConfig.from_mapping(
            {
                "mesh_source": "interop",
                "mesh_path": str(source.resolve()),
                "mesh_dimension": 3
                if np.asarray(payload.get("nodes", np.zeros((0, 2)))).shape[1] >= 3
                else 2,
                "n_elec": int(np.asarray(payload.get("n_elec", 16)).reshape(-1)[0]),
                "contact_impedance": _contact_config_value(payload),
            }
        )
        return LoadedBridgePackage(
            root=source.parent,
            manifest=manifest,
            geometry_payload=payload,
            forward_model_config=config,
        )

    root = source if source.is_dir() else source.parent
    manifest = _load_manifest(root)
    geometry_name = manifest.files.get("geometry", GEOMETRY_NAME)
    geometry_path = root / geometry_name
    geometry_payload = None
    if geometry_path.exists():
        geometry_payload = _public_mat_payload(
            loadmat(geometry_path, squeeze_me=True, struct_as_record=False)
        )

    forward_model_config = _load_forward_model_config(root, manifest)
    if forward_model_config is None and geometry_payload is not None:
        forward_model_config = ForwardModelConfig.from_mapping(
            {
                "mesh_dimension": 3
                if np.asarray(geometry_payload.get("nodes", np.zeros((0, 2)))).shape[1]
                >= 3
                else 2,
                "n_elec": int(
                    np.asarray(geometry_payload.get("n_elec", 16)).reshape(-1)[0]
                ),
                "contact_impedance": _contact_config_value(geometry_payload),
            }
        )
    if forward_model_config is not None and geometry_path.exists():
        forward_model_config = forward_model_config.with_overrides(
            mesh_source="interop",
            mesh_path=str(geometry_path.resolve()),
        )

    return LoadedBridgePackage(
        root=root,
        manifest=manifest,
        geometry_payload=geometry_payload,
        measurements=_load_measurements(root, manifest),
        forward_model_config=forward_model_config,
        reconstruction_preset=_load_reconstruction_preset(root, manifest),
    )


def validate_bridge_package(path: str | Path) -> dict[str, Any]:
    """Validate a Bridge Package v2 directory or a standalone geometry MAT."""

    source = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "schema": "eidors_pyeidors_bridge_validation_v1",
        "path": str(source.resolve()),
        "valid": False,
        "package_format": "",
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
    if not source.exists():
        errors.append(f"Path does not exist: {source}")
        return report

    if source.is_file() and source.suffix.lower() == ".mat":
        root = source.parent
        manifest = default_manifest(
            source_framework="unknown",
            package_kind="standalone_geometry",
            files={"geometry": source.name},
        )
        geometry_path = source
        warnings.append(
            "Standalone geometry MAT has no Bridge Package manifest/config."
        )
    else:
        root = source if source.is_dir() else source.parent
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            errors.append(f"Missing {MANIFEST_NAME}")
            return report
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = InteropBridgeManifest.from_mapping(manifest_payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Invalid {MANIFEST_NAME}: {exc}")
            return report
        report["package_format"] = manifest.exchange_format
        if manifest.exchange_format != BRIDGE_PACKAGE_FORMAT_V2:
            errors.append(
                f"manifest.exchange_format must be {BRIDGE_PACKAGE_FORMAT_V2!r}"
            )
        for role, name in sorted(manifest.files.items()):
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(
                    f"manifest file role {role!r} must use a safe relative path"
                )
                continue
            candidate = root / relative
            report["files"][role] = str(relative)
            if not candidate.is_file():
                errors.append(f"manifest file role {role!r} does not exist: {relative}")
        geometry_name = manifest.files.get("geometry", GEOMETRY_NAME)
        geometry_path = root / geometry_name

    if not geometry_path.is_file():
        errors.append(f"Missing geometry MAT: {geometry_path.name}")
        return report
    try:
        geometry = _public_mat_payload(
            loadmat(geometry_path, squeeze_me=True, struct_as_record=False)
        )
        validate_exchange_payload(geometry)
    except (OSError, TypeError, ValueError) as exc:
        errors.append(f"Invalid geometry MAT: {exc}")
        return report

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
        np.asarray(geometry.get("background_present", True)).reshape(-1)[0]
    )
    if "stimulation_supported" in geometry:
        report["stimulation_supported"] = bool(
            np.asarray(geometry["stimulation_supported"]).reshape(-1)[0]
        )
    if "current_density_present" in geometry:
        report["current_density_present"] = bool(
            np.asarray(geometry["current_density_present"]).reshape(-1)[0]
        )
    if "current_density_applied" in geometry:
        report["current_density_applied"] = bool(
            np.asarray(geometry["current_density_applied"]).reshape(-1)[0]
        )
    for source_name, report_name in (
        ("stim_positive_current", "stim_positive_current"),
        ("stim_negative_current", "stim_negative_current"),
        ("stim_net_current", "stim_net_current"),
        ("stim_max_abs_current", "stim_max_abs_current"),
        ("stim_balanced", "stim_balanced"),
    ):
        if source_name in geometry:
            report[report_name] = json_ready(
                np.asarray(geometry[source_name]).reshape(-1)
            )
    blockers = [str(item) for item in metadata.get("forward_blockers", []) if str(item)]
    if not np.all(contact_present):
        blockers.append("contact_impedance_missing_no_eidors_default")
    if not report["background_present"]:
        background_elem_present = bool(
            np.asarray(
                geometry.get(
                    "background_elem_data_present",
                    "background_elem_data" in geometry,
                )
            ).reshape(-1)[0]
        )
        background_elem_data = np.asarray(geometry.get("background_elem_data", []))
        blockers.append(
            "background_is_nonuniform_and_not_gui_scalar_compatible"
            if background_elem_present and background_elem_data.size
            else "background_image_missing_or_unmappable"
        )
    if report["electrode_definition"] == "point_or_lower_dimensional" or any(
        model in {"point", "distributed_point"} for model in electrode_models
    ):
        blockers.append(
            "point_or_distributed_point_electrode_requires_explicit_projection_opt_in"
        )
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
    if "stim_matrix" in geometry:
        stim = np.asarray(geometry["stim_matrix"])
        report["n_stimulations"] = int(1 if stim.ndim == 1 else stim.shape[0])
    if "measurement_counts" in geometry:
        report["n_measurements"] = int(
            np.sum(np.asarray(geometry["measurement_counts"], dtype=np.int64))
        )
    else:
        measurements = _load_measurements(root, manifest)
        if measurements:
            first = next(iter(measurements.values()))
            report["n_measurements"] = int(np.asarray(first).size)

    report["valid"] = not errors
    return report
