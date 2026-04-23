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
    STANDARD_INTEROP_FORMAT,
    export_forward_csv,
    load_forward_csv,
    save_exchange_mat,
)

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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
        if geometry_payload.get("exchange_format") == STANDARD_INTEROP_FORMAT:
            save_exchange_mat(geometry_path, geometry_payload)
        else:
            savemat(geometry_path, geometry_payload)
        files["geometry"] = GEOMETRY_NAME

    if measurements:
        if {"homogeneous", "target"}.issubset(measurements):
            homogeneous = np.asarray(measurements["homogeneous"], dtype=float).reshape(
                -1
            )
            target = np.asarray(measurements["target"], dtype=float).reshape(-1)
            export_forward_csv(root / MEASUREMENTS_CSV_NAME, homogeneous, target)
            files["measurements_csv"] = MEASUREMENTS_CSV_NAME
        else:
            savemat(
                root / MEASUREMENTS_MAT_NAME,
                {key: np.asarray(value) for key, value in measurements.items()},
            )
            files["measurements_mat"] = MEASUREMENTS_MAT_NAME

    if forward_model_config is not None:
        config_path = root / CONFIG_NAME
        config_path.write_text(
            json.dumps(
                {
                    "forward_model_config": forward_model_config.to_mapping(),
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
        payload = loadmat(source, squeeze_me=True, struct_as_record=False)
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
                "mesh_dimension": 3
                if np.asarray(payload.get("nodes", np.zeros((0, 2)))).shape[1] >= 3
                else 2,
                "n_elec": int(np.asarray(payload.get("n_elec", 16)).reshape(-1)[0]),
                "contact_impedance": payload.get("contact_impedance"),
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
        geometry_payload = loadmat(
            geometry_path, squeeze_me=True, struct_as_record=False
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
                "contact_impedance": geometry_payload.get("contact_impedance"),
            }
        )

    return LoadedBridgePackage(
        root=root,
        manifest=manifest,
        geometry_payload=geometry_payload,
        measurements=_load_measurements(root, manifest),
        forward_model_config=forward_model_config,
        reconstruction_preset=_load_reconstruction_preset(root, manifest),
    )
