"""Bridge Package v3 integrity, identity, and typed exchange models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat, savemat

from pyeidors.cache.keys import (
    CacheKeyParts,
    build_cache_key,
    hash_array,
    hash_file_content,
)
from pyeidors.io._json import json_ready

BRIDGE_PACKAGE_FORMAT_V3 = "eidors_pyeidors_bridge_v3"
GEOMETRY_FORMAT_V3 = "eidors_pyeidors_geometry_v3"
BRIDGE_SCHEMA_VERSION = 3
MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.json"
GEOMETRY_NAME = "geometry.mat"
PROTOCOL_NAME = "protocol.mat"
FIELDS_NAME = "fields.mat"
MEASUREMENTS_NAME = "measurements.mat"
RECONSTRUCTION_NAME = "reconstruction.json"

REQUIRED_FILE_ROLES = ("model", "geometry", "protocol", "fields")
DEFAULT_FILE_NAMES = MappingProxyType(
    {
        "model": MODEL_NAME,
        "geometry": GEOMETRY_NAME,
        "protocol": PROTOCOL_NAME,
        "fields": FIELDS_NAME,
    }
)


def _as_text(value: Any) -> str:
    array = np.asarray(value).reshape(-1)
    return "" if array.size == 0 else str(array[0])


def _as_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_semantic_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            return {
                "dtype": str(array.dtype),
                "shape": [int(item) for item in array.shape],
                "items": [
                    _canonical_semantic_value(item) for item in array.reshape(-1)
                ],
            }
        return {
            "dtype": str(array.dtype),
            "shape": [int(item) for item in array.shape],
            "sha256": hash_array(array),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not str(key).startswith("__")
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_semantic_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    scalar = _as_python_scalar(value)
    if isinstance(scalar, complex):
        return {
            "complex_scalar": {
                "real": float(scalar.real),
                "imag": float(scalar.imag),
            }
        }
    return scalar


def _semantic_fingerprint(namespace: str, payload: Mapping[str, Any]) -> str:
    return build_cache_key(
        CacheKeyParts(
            artifact=namespace,
            namespace="eidors-bridge-v3",
            schema_version=BRIDGE_SCHEMA_VERSION,
            payload=_canonical_semantic_value(payload),
        )
    )


def _normalize_protocol_rows(values: Any) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2:
        raise ValueError("Protocol matrices must be one- or two-dimensional")
    if rows.shape[0] == 0:
        return np.ascontiguousarray(rows)
    if not np.all(np.isfinite(rows)):
        raise ValueError("Protocol matrices must contain only finite values")
    normalized = np.array(rows, copy=True)
    scale = np.max(np.abs(normalized), axis=1)
    if np.any(scale <= 0):
        raise ValueError("Protocol rows must be non-zero")
    normalized = normalized / scale[:, None]
    for index, row in enumerate(normalized):
        active = np.flatnonzero(np.abs(row) > 0)
        if active.size and np.real(row[active[0]]) < 0:
            normalized[index] *= -1
    return normalized


def _protocol_measurement_payload(protocol: Mapping[str, Any]) -> Any:
    if "meas_matrices" in protocol:
        return protocol["meas_matrices"]
    if "measurement_matrix" in protocol:
        return protocol["measurement_matrix"]
    if "v2meas" in protocol:
        return protocol["v2meas"]
    return np.empty((0, 0), dtype=float)


def build_bridge_fingerprints(
    *,
    model: Mapping[str, Any],
    geometry: Mapping[str, Any],
    protocol: Mapping[str, Any],
    fields: Mapping[str, Any],
    measurements: Mapping[str, Any] | None = None,
    reconstruction: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Build semantic package, forward, and protocol identities."""

    stim = protocol.get("stim_matrix", protocol.get("stim_matrix_effective"))
    if stim is None:
        n_elec = int(model.get("n_elec", 0) or 0)
        stim = np.empty((0, max(n_elec, 0)), dtype=float)
    measurement = _protocol_measurement_payload(protocol)
    layout_payload = {
        "stim_normalized": _normalize_protocol_rows(stim),
        "measurement": measurement,
        "measurement_counts": protocol.get("measurement_counts"),
        "meas_select": protocol.get("meas_select"),
    }
    protocol_layout_hash = _semantic_fingerprint(
        "bridge-v3-protocol-layout",
        layout_payload,
    )
    protocol_physics_hash = _semantic_fingerprint(
        "bridge-v3-protocol-physics",
        {
            **layout_payload,
            "stim_effective": np.asarray(stim),
            "current_density": protocol.get("current_density"),
        },
    )
    forward_fingerprint = _semantic_fingerprint(
        "bridge-v3-forward-model",
        {
            "model": model,
            "geometry": geometry,
            "protocol_physics_hash": protocol_physics_hash,
            "background": {
                key: value
                for key, value in fields.items()
                if str(key).startswith("background")
            },
        },
    )
    model_id = _semantic_fingerprint(
        "bridge-v3-model-asset",
        {
            "model": model,
            "geometry": geometry,
            "protocol": protocol,
            "fields": fields,
            "measurements": measurements or {},
            "reconstruction": reconstruction or {},
        },
    )
    return {
        "model_id": model_id,
        "forward_fingerprint": forward_fingerprint,
        "protocol_layout_hash": protocol_layout_hash,
        "protocol_physics_hash": protocol_physics_hash,
    }


@dataclass(frozen=True)
class ElectrodeSpec:
    """Portable ordered CEM or weighted-PEM electrode definition."""

    kind: str
    index_base: int = 1
    source_nodes: tuple[int, ...] = ()
    source_faces: tuple[tuple[int, ...], ...] = ()
    node_weights: tuple[float | complex, ...] = ()
    boundary_kind: str = "exterior"
    contact_impedance: float | complex | None = None
    contact_impedance_present: bool = False
    contact_impedance_applicable: bool = False

    def __post_init__(self) -> None:
        index_base = int(self.index_base)
        if index_base not in {0, 1}:
            raise ValueError("ElectrodeSpec.index_base must be 0 or 1")
        object.__setattr__(self, "index_base", index_base)
        kind = str(self.kind).strip().lower()
        if kind not in {"cem", "pem"}:
            raise ValueError("ElectrodeSpec.kind must be 'cem' or 'pem'")
        object.__setattr__(self, "kind", kind)
        boundary = str(self.boundary_kind).strip().lower()
        if boundary not in {"exterior", "interior", "none"}:
            raise ValueError(
                "ElectrodeSpec.boundary_kind must be exterior, interior, or none"
            )
        object.__setattr__(self, "boundary_kind", boundary)
        if any(int(node) < index_base for node in self.source_nodes):
            raise ValueError("ElectrodeSpec.source_nodes contains an invalid index")
        if kind == "cem":
            if not self.source_faces:
                raise ValueError("CEM electrode requires source_faces")
            if not self.contact_impedance_applicable:
                raise ValueError("CEM contact impedance must be applicable")
            if self.contact_impedance_present:
                if self.contact_impedance is None:
                    raise ValueError(
                        "CEM contact_impedance_present requires contact_impedance"
                    )
                impedance = complex(self.contact_impedance)
                if not np.isfinite(impedance) or np.isclose(abs(impedance), 0.0):
                    raise ValueError(
                        "CEM contact impedance must be finite and non-zero"
                    )
            elif self.contact_impedance is not None:
                raise ValueError(
                    "Missing CEM contact impedance must be represented as None"
                )
        else:
            if not self.source_nodes:
                raise ValueError("PEM electrode requires source_nodes")
            if len(self.source_nodes) != len(self.node_weights):
                raise ValueError("PEM nodes and weights must have equal length")
            weights = np.asarray(self.node_weights)
            if np.iscomplexobj(weights) and not np.allclose(np.imag(weights), 0.0):
                raise ValueError("PEM weights must be real")
            weights = np.real(weights)
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("PEM weights must be finite and non-negative")
            if not np.isclose(np.sum(weights), 1.0):
                raise ValueError("PEM weights must sum to one")
            if self.contact_impedance_applicable:
                raise ValueError("PEM contact impedance is provenance-only")


@dataclass(frozen=True)
class ProtocolSpec:
    """Portable effective current and measurement protocol."""

    stim_matrix: np.ndarray
    meas_matrices: tuple[np.ndarray, ...]
    stim_matrix_raw: np.ndarray | None = None
    meas_select: np.ndarray | None = None
    normalize_measurements: bool = False
    current_density: float | complex | None = None

    def __post_init__(self) -> None:
        stim = np.asarray(self.stim_matrix)
        if stim.ndim == 1:
            stim = stim.reshape(1, -1)
        if stim.ndim != 2 or not np.all(np.isfinite(stim)):
            raise ValueError("ProtocolSpec.stim_matrix must be a finite matrix")
        if len(self.meas_matrices) != stim.shape[0]:
            raise ValueError(
                "ProtocolSpec requires one measurement matrix per stimulation"
            )
        matrices: list[np.ndarray] = []
        for matrix in self.meas_matrices:
            value = np.asarray(matrix)
            if value.ndim == 1:
                value = value.reshape(1, -1)
            if value.ndim != 2 or value.shape[1] != stim.shape[1]:
                raise ValueError("Measurement matrix electrode width mismatch")
            if not np.all(np.isfinite(value)):
                raise ValueError("Measurement matrices must be finite")
            matrices.append(np.ascontiguousarray(value))
        object.__setattr__(self, "stim_matrix", np.ascontiguousarray(stim))
        object.__setattr__(self, "meas_matrices", tuple(matrices))


@dataclass(frozen=True)
class BridgeV3Package:
    """Verified in-memory view of one Bridge Package v3 directory."""

    root: Path
    manifest: Mapping[str, Any]
    model: Mapping[str, Any]
    geometry: Mapping[str, Any]
    protocol: Mapping[str, Any]
    fields: Mapping[str, Any]
    measurements: Mapping[str, Any] | None = None
    reconstruction: Mapping[str, Any] | None = None

    @property
    def model_id(self) -> str:
        return str(self.manifest["model_id"])

    @property
    def forward_fingerprint(self) -> str:
        return str(self.manifest["forward_fingerprint"])

    @classmethod
    def load(cls, path: str | Path, *, verify: bool = True) -> "BridgeV3Package":
        root = Path(path)
        if root.is_file():
            raise ValueError(
                "Bridge v3 requires a package directory; standalone MAT is unsupported"
            )
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError(f"Missing {MANIFEST_NAME}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid {MANIFEST_NAME}: {exc}") from exc
        _validate_manifest_shape(manifest)
        if verify:
            _verify_files(root, manifest)
        files = manifest["files"]
        model = _load_json_file(root, files["model"]["path"])
        geometry = _load_mat_file(root, files["geometry"]["path"])
        protocol = _load_mat_file(root, files["protocol"]["path"])
        fields = _load_mat_file(root, files["fields"]["path"])
        measurements = (
            _load_mat_file(root, files["measurements"]["path"])
            if "measurements" in files
            else None
        )
        reconstruction = (
            _load_json_file(root, files["reconstruction"]["path"])
            if "reconstruction" in files
            else None
        )
        package = cls(
            root=root.resolve(),
            manifest=MappingProxyType(manifest),
            model=MappingProxyType(model),
            geometry=MappingProxyType(geometry),
            protocol=MappingProxyType(protocol),
            fields=MappingProxyType(fields),
            measurements=(
                None if measurements is None else MappingProxyType(measurements)
            ),
            reconstruction=(
                None if reconstruction is None else MappingProxyType(reconstruction)
            ),
        )
        if verify:
            package.verify_semantic_identity()
        return package

    def verify_semantic_identity(self) -> None:
        identities = build_bridge_fingerprints(
            model=self.model,
            geometry=self.geometry,
            protocol=self.protocol,
            fields=self.fields,
            measurements=self.measurements,
            reconstruction=self.reconstruction,
        )
        for name, actual in identities.items():
            expected = str(self.manifest.get(name, ""))
            if actual != expected:
                raise ValueError(
                    f"Bridge v3 semantic identity mismatch for {name}: "
                    f"expected {expected!r}, got {actual!r}"
                )

    @classmethod
    def write(
        cls,
        path: str | Path,
        *,
        model: Mapping[str, Any],
        geometry: Mapping[str, Any],
        protocol: Mapping[str, Any],
        fields: Mapping[str, Any],
        measurements: Mapping[str, Any] | None = None,
        reconstruction: Mapping[str, Any] | None = None,
        source_framework: str = "pyeidors",
        package_kind: str = "bridge",
        provenance: Mapping[str, Any] | None = None,
        capabilities: Mapping[str, Any] | None = None,
    ) -> "BridgeV3Package":
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        geometry_payload = dict(geometry)
        geometry_payload["exchange_format"] = GEOMETRY_FORMAT_V3
        geometry_payload["schema_version"] = BRIDGE_SCHEMA_VERSION
        model_payload = dict(model)
        model_payload.setdefault("schema_version", BRIDGE_SCHEMA_VERSION)
        _write_json_file(root / MODEL_NAME, model_payload)
        savemat(root / GEOMETRY_NAME, geometry_payload)
        savemat(root / PROTOCOL_NAME, dict(protocol))
        savemat(root / FIELDS_NAME, dict(fields))
        role_paths: dict[str, Path] = {
            "model": root / MODEL_NAME,
            "geometry": root / GEOMETRY_NAME,
            "protocol": root / PROTOCOL_NAME,
            "fields": root / FIELDS_NAME,
        }
        if measurements is not None:
            savemat(root / MEASUREMENTS_NAME, dict(measurements))
            role_paths["measurements"] = root / MEASUREMENTS_NAME
        if reconstruction is not None:
            _write_json_file(root / RECONSTRUCTION_NAME, dict(reconstruction))
            role_paths["reconstruction"] = root / RECONSTRUCTION_NAME
        decoded_model = _load_json_file(root, MODEL_NAME)
        decoded_geometry = _load_mat_file(root, GEOMETRY_NAME)
        decoded_protocol = _load_mat_file(root, PROTOCOL_NAME)
        decoded_fields = _load_mat_file(root, FIELDS_NAME)
        decoded_measurements = (
            _load_mat_file(root, MEASUREMENTS_NAME)
            if measurements is not None
            else None
        )
        decoded_reconstruction = (
            _load_json_file(root, RECONSTRUCTION_NAME)
            if reconstruction is not None
            else None
        )
        identities = build_bridge_fingerprints(
            model=decoded_model,
            geometry=decoded_geometry,
            protocol=decoded_protocol,
            fields=decoded_fields,
            measurements=decoded_measurements,
            reconstruction=decoded_reconstruction,
        )
        file_entries = {
            role: {
                "path": file_path.name,
                "size_bytes": int(file_path.stat().st_size),
                "sha256": hash_file_content(file_path),
            }
            for role, file_path in sorted(role_paths.items())
        }
        manifest = {
            "exchange_format": BRIDGE_PACKAGE_FORMAT_V3,
            "schema_version": BRIDGE_SCHEMA_VERSION,
            "source_framework": str(source_framework),
            "package_kind": str(package_kind),
            "model_id": identities["model_id"],
            "forward_fingerprint": identities["forward_fingerprint"],
            "protocol_layout_hash": identities["protocol_layout_hash"],
            "protocol_physics_hash": identities["protocol_physics_hash"],
            "files": file_entries,
            "provenance": dict(provenance or {}),
            "capabilities": dict(capabilities or {}),
        }
        _write_json_file(root / MANIFEST_NAME, manifest)
        return cls.load(root)


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            json_ready(dict(payload)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_json_file(root: Path, relative: str) -> dict[str, Any]:
    path = _safe_package_path(root, relative)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Bridge v3 JSON file {relative!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Bridge v3 JSON file {relative!r} must contain an object")
    return payload


def _load_mat_file(root: Path, relative: str) -> dict[str, Any]:
    path = _safe_package_path(root, relative)
    try:
        payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Bridge v3 MAT file {relative!r}: {exc}") from exc
    return {key: value for key, value in payload.items() if not key.startswith("__")}


def _safe_package_path(root: Path, relative: str) -> Path:
    path = Path(str(relative))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe Bridge v3 relative path: {relative!r}")
    return root / path


def _validate_manifest_shape(manifest: Mapping[str, Any]) -> None:
    if manifest.get("exchange_format") != BRIDGE_PACKAGE_FORMAT_V3:
        raise ValueError(
            "Unsupported Bridge package format "
            f"{manifest.get('exchange_format')!r}; expected "
            f"{BRIDGE_PACKAGE_FORMAT_V3!r}. Bridge v1/v2 is not supported."
        )
    if int(manifest.get("schema_version", -1)) != BRIDGE_SCHEMA_VERSION:
        raise ValueError("Bridge v3 manifest schema_version must be 3")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Bridge v3 manifest.files must be an object")
    missing = [role for role in REQUIRED_FILE_ROLES if role not in files]
    if missing:
        raise ValueError(
            "Bridge v3 manifest is missing required file roles: " + ", ".join(missing)
        )
    for name in (
        "model_id",
        "forward_fingerprint",
        "protocol_layout_hash",
        "protocol_physics_hash",
    ):
        value = str(manifest.get(name, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"Bridge v3 manifest.{name} must be lowercase SHA-256")
    for role, entry in files.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Bridge v3 file role {role!r} must be an object")
        _safe_package_path(Path("."), str(entry.get("path", "")))
        if int(entry.get("size_bytes", -1)) < 0:
            raise ValueError(f"Bridge v3 file role {role!r} has invalid size_bytes")
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Bridge v3 file role {role!r} has invalid SHA-256 digest")


def _verify_files(root: Path, manifest: Mapping[str, Any]) -> None:
    for role, entry in manifest["files"].items():
        path = _safe_package_path(root, str(entry["path"]))
        if not path.is_file():
            raise ValueError(
                f"Bridge v3 file role {role!r} does not exist: {entry['path']}"
            )
        size = int(path.stat().st_size)
        expected_size = int(entry["size_bytes"])
        if size != expected_size:
            raise ValueError(
                f"Bridge v3 file size mismatch for {role!r}: "
                f"expected {expected_size}, got {size}"
            )
        digest = hash_file_content(path)
        if digest != str(entry["sha256"]):
            raise ValueError(f"Bridge v3 SHA-256 mismatch for file role {role!r}")


def validate_bridge_v3_package(path: str | Path) -> dict[str, Any]:
    """Return deterministic validation JSON without raising."""

    source = Path(path)
    report: dict[str, Any] = {
        "schema": "eidors_pyeidors_bridge_v3_validation_v1",
        "path": str(source.resolve()),
        "valid": False,
        "package_format": "",
        "model_id": "",
        "forward_fingerprint": "",
        "protocol_layout_hash": "",
        "protocol_physics_hash": "",
        "files": {},
        "errors": [],
    }
    try:
        package = BridgeV3Package.load(source)
    except (OSError, TypeError, ValueError) as exc:
        report["errors"].append(str(exc))
        return report
    report.update(
        {
            "valid": True,
            "package_format": BRIDGE_PACKAGE_FORMAT_V3,
            "model_id": package.model_id,
            "forward_fingerprint": package.forward_fingerprint,
            "protocol_layout_hash": str(package.manifest["protocol_layout_hash"]),
            "protocol_physics_hash": str(package.manifest["protocol_physics_hash"]),
            "files": {
                role: str(entry["path"])
                for role, entry in package.manifest["files"].items()
            },
        }
    )
    return report


__all__ = [
    "BRIDGE_PACKAGE_FORMAT_V3",
    "BRIDGE_SCHEMA_VERSION",
    "BridgeV3Package",
    "ElectrodeSpec",
    "FIELDS_NAME",
    "GEOMETRY_FORMAT_V3",
    "GEOMETRY_NAME",
    "MANIFEST_NAME",
    "MEASUREMENTS_NAME",
    "MODEL_NAME",
    "PROTOCOL_NAME",
    "ProtocolSpec",
    "RECONSTRUCTION_NAME",
    "build_bridge_fingerprints",
    "validate_bridge_v3_package",
]
