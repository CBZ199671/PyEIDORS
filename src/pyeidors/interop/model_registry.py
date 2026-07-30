"""Immutable Bridge v3 model registry and authoritative runtime contexts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import tempfile
from typing import Any, Mapping

import numpy as np

from pyeidors.runtime_paths import pyeidors_data_path

from .bridge_v3 import BridgeV3Package, ElectrodeSpec, ProtocolSpec
from .geometry_exchange import (
    build_mesh_from_exchange_mat,
    source_cell_data_to_local,
)

MODEL_REGISTRY_SCHEMA_VERSION = 3
MODEL_REGISTRY_DB_NAME = "registry.sqlite3"
MODEL_BINDING_FLOWS = ("simulation", "dataset", "realtime")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _public_mat_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if not str(key).startswith("__")
    }


@dataclass(frozen=True)
class RegisteredModel:
    """One immutable, managed Bridge v3 model asset."""

    model_id: str
    display_name: str
    source_path: str
    asset_path: Path
    forward_fingerprint: str
    protocol_layout_hash: str
    protocol_physics_hash: str
    source_framework: str
    package_kind: str
    capabilities: Mapping[str, Any]
    imported_at_utc: str
    status: str = "ready"
    integrity_error: str = ""

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": "pyeidors_registered_model_v3",
            "model_id": self.model_id,
            "display_name": self.display_name,
            "source_path": self.source_path,
            "asset_path": str(self.asset_path),
            "forward_fingerprint": self.forward_fingerprint,
            "protocol_layout_hash": self.protocol_layout_hash,
            "protocol_physics_hash": self.protocol_physics_hash,
            "source_framework": self.source_framework,
            "package_kind": self.package_kind,
            "capabilities": dict(self.capabilities),
            "imported_at_utc": self.imported_at_utc,
            "status": self.status,
            "integrity_error": self.integrity_error,
        }


class ModelRegistry:
    """Validate, atomically copy, index, and bind immutable v3 packages."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else pyeidors_data_path("models/v3")
        self.root = self.root.expanduser().resolve()
        self.database_path = self.root / MODEL_REGISTRY_DB_NAME
        self.root.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    asset_path TEXT NOT NULL,
                    forward_fingerprint TEXT NOT NULL,
                    protocol_layout_hash TEXT NOT NULL,
                    protocol_physics_hash TEXT NOT NULL,
                    source_framework TEXT NOT NULL,
                    package_kind TEXT NOT NULL,
                    capabilities_json TEXT NOT NULL,
                    imported_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    integrity_error TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS bindings (
                    flow TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    bound_at_utc TEXT NOT NULL,
                    FOREIGN KEY(model_id) REFERENCES models(model_id)
                );
                """
            )
            connection.execute(f"PRAGMA user_version = {MODEL_REGISTRY_SCHEMA_VERSION}")

    def _asset_path(self, model_id: str) -> Path:
        if len(model_id) != 64 or any(
            char not in "0123456789abcdef" for char in model_id
        ):
            raise ValueError("model_id must be a lowercase SHA-256 digest")
        path = (self.root / model_id).resolve()
        if path.parent != self.root:
            raise ValueError("Resolved model asset path escaped the registry root")
        return path

    @staticmethod
    def _reject_symlinks(source: Path) -> None:
        if source.is_symlink():
            raise ValueError("Bridge v3 registry source cannot be a symlink")
        for directory, directory_names, file_names in os.walk(
            source,
            followlinks=False,
        ):
            base = Path(directory)
            for name in [*directory_names, *file_names]:
                if (base / name).is_symlink():
                    raise ValueError(
                        "Bridge v3 registry packages cannot contain symlinks"
                    )

    @staticmethod
    def _make_read_only(root: Path) -> None:
        for directory, _, file_names in os.walk(root):
            for name in file_names:
                path = Path(directory) / name
                mode = path.stat().st_mode
                path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def _copy_atomically(self, source: Path, destination: Path) -> None:
        self._reject_symlinks(source)
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".register-{destination.name[:12]}-",
                dir=self.root,
            )
        ).resolve()
        if temporary_root.parent != self.root:
            raise RuntimeError("Registry staging path escaped the registry root")
        staging = temporary_root / "package"
        try:
            shutil.copytree(source, staging)
            staged = BridgeV3Package.load(staging)
            if staged.model_id != destination.name:
                raise ValueError("Copied Bridge v3 package changed semantic identity")
            if destination.exists():
                existing = BridgeV3Package.load(destination)
                if existing.model_id != destination.name:
                    raise ValueError("Existing registry asset has invalid identity")
            else:
                os.replace(staging, destination)
                self._make_read_only(destination)
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)

    @staticmethod
    def _registered_from_row(row: sqlite3.Row) -> RegisteredModel:
        return RegisteredModel(
            model_id=str(row["model_id"]),
            display_name=str(row["display_name"]),
            source_path=str(row["source_path"]),
            asset_path=Path(str(row["asset_path"])),
            forward_fingerprint=str(row["forward_fingerprint"]),
            protocol_layout_hash=str(row["protocol_layout_hash"]),
            protocol_physics_hash=str(row["protocol_physics_hash"]),
            source_framework=str(row["source_framework"]),
            package_kind=str(row["package_kind"]),
            capabilities=json.loads(str(row["capabilities_json"])),
            imported_at_utc=str(row["imported_at_utc"]),
            status=str(row["status"]),
            integrity_error=str(row["integrity_error"]),
        )

    def register(
        self,
        package_path: str | Path,
        *,
        display_name: str | None = None,
    ) -> RegisteredModel:
        source = Path(package_path).expanduser().resolve()
        package = BridgeV3Package.load(source)
        destination = self._asset_path(package.model_id)
        if source != destination:
            self._copy_atomically(source, destination)
        else:
            BridgeV3Package.load(destination)
        imported_at = _utc_now()
        name = str(display_name or package.model.get("name") or source.name)
        capabilities = dict(package.manifest.get("capabilities", {}))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO models (
                    model_id, display_name, source_path, asset_path,
                    forward_fingerprint, protocol_layout_hash,
                    protocol_physics_hash, source_framework, package_kind,
                    capabilities_json, imported_at_utc, status, integrity_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', '')
                ON CONFLICT(model_id) DO UPDATE SET
                    display_name=excluded.display_name,
                    source_path=excluded.source_path,
                    status='ready',
                    integrity_error=''
                """,
                (
                    package.model_id,
                    name,
                    str(source),
                    str(destination),
                    package.forward_fingerprint,
                    str(package.manifest["protocol_layout_hash"]),
                    str(package.manifest["protocol_physics_hash"]),
                    str(package.manifest.get("source_framework", "unknown")),
                    str(package.manifest.get("package_kind", "bridge")),
                    _json_text(capabilities),
                    imported_at,
                ),
            )
        return self.get(package.model_id)

    def get(self, model_id: str) -> RegisteredModel:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown registered model_id: {model_id}")
        return self._registered_from_row(row)

    def list_models(self) -> list[RegisteredModel]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM models ORDER BY imported_at_utc DESC, model_id"
            ).fetchall()
        return [self._registered_from_row(row) for row in rows]

    def _mark_corrupt(self, model_id: str, error: Exception) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE models
                SET status = 'corrupt', integrity_error = ?
                WHERE model_id = ?
                """,
                (str(error), model_id),
            )

    def load_package(self, model_id: str) -> BridgeV3Package:
        registered = self.get(model_id)
        try:
            package = BridgeV3Package.load(registered.asset_path)
            if package.model_id != model_id:
                raise ValueError("Registered model_id does not match package identity")
        except (OSError, TypeError, ValueError) as exc:
            self._mark_corrupt(model_id, exc)
            raise ValueError(
                f"Registered Bridge v3 model {model_id} failed integrity: {exc}"
            ) from exc
        if registered.status != "ready" or registered.integrity_error:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE models
                    SET status = 'ready', integrity_error = ''
                    WHERE model_id = ?
                    """,
                    (model_id,),
                )
        return package

    def bind(self, flow: str, model_id: str) -> RegisteredModel:
        normalized_flow = str(flow).strip().lower()
        if normalized_flow not in MODEL_BINDING_FLOWS:
            raise ValueError("flow must be one of: " + ", ".join(MODEL_BINDING_FLOWS))
        self.load_package(model_id)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO bindings(flow, model_id, bound_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(flow) DO UPDATE SET
                    model_id=excluded.model_id,
                    bound_at_utc=excluded.bound_at_utc
                """,
                (normalized_flow, model_id, _utc_now()),
            )
        return self.get(model_id)

    def apply_to_all(self, model_id: str) -> dict[str, RegisteredModel]:
        self.load_package(model_id)
        with self._connect() as connection:
            now = _utc_now()
            connection.executemany(
                """
                INSERT INTO bindings(flow, model_id, bound_at_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(flow) DO UPDATE SET
                    model_id=excluded.model_id,
                    bound_at_utc=excluded.bound_at_utc
                """,
                [(flow, model_id, now) for flow in MODEL_BINDING_FLOWS],
            )
        registered = self.get(model_id)
        return {flow: registered for flow in MODEL_BINDING_FLOWS}

    def bindings(self) -> dict[str, RegisteredModel]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT b.flow, m.*
                FROM bindings AS b
                JOIN models AS m ON m.model_id = b.model_id
                ORDER BY b.flow
                """
            ).fetchall()
        return {str(row["flow"]): self._registered_from_row(row) for row in rows}

    def bound_model(self, flow: str) -> RegisteredModel | None:
        return self.bindings().get(str(flow).strip().lower())


def _scalar_int(value: Any, name: str) -> int:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    return int(array[0])


def _protocol_spec(
    protocol: Mapping[str, Any],
    *,
    n_elec: int,
) -> ProtocolSpec:
    if "stim_matrix" not in protocol or "meas_matrices" not in protocol:
        raise ValueError("Bridge v3 protocol has no executable current patterns")
    stim = np.asarray(protocol["stim_matrix"])
    if stim.ndim == 0 and n_elec == 1:
        stim = stim.reshape(1, 1)
    elif stim.ndim == 1 and stim.size == n_elec:
        stim = stim.reshape(1, n_elec)
    elif stim.ndim == 1 and n_elec == 1:
        stim = stim.reshape(-1, 1)
    if stim.ndim != 2 or stim.shape[1] != n_elec:
        raise ValueError("Bridge v3 stimulation shape is ambiguous")
    if np.iscomplexobj(stim) and not np.allclose(np.imag(stim), 0.0):
        raise ValueError("Bridge v3 executable stimulation must be real")
    stim = np.real(stim)
    counts = np.asarray(protocol.get("measurement_counts", []), dtype=np.int64)
    counts = counts.reshape(-1)
    if counts.size != stim.shape[0] or np.any(counts <= 0):
        raise ValueError("Bridge v3 measurement_counts is invalid")
    raw_meas = np.asarray(protocol["meas_matrices"])
    expected_shape = (stim.shape[0], int(np.max(counts)), n_elec)
    squeezed_shape = tuple(size for size in expected_shape if size != 1)
    if raw_meas.shape == expected_shape:
        restored = raw_meas
    elif raw_meas.shape == squeezed_shape or (
        not squeezed_shape and raw_meas.ndim == 0
    ):
        restored = raw_meas.reshape(expected_shape)
    else:
        raise ValueError(
            "Bridge v3 measurement matrix shape cannot be restored exactly"
        )
    matrices = tuple(
        np.asarray(restored[index, : int(count), :])
        for index, count in enumerate(counts)
    )
    raw_stim = protocol.get("stim_matrix_raw")
    return ProtocolSpec(
        stim_matrix=stim,
        stim_matrix_raw=None if raw_stim is None else np.asarray(raw_stim),
        meas_matrices=matrices,
        meas_select=(
            None
            if "meas_select" not in protocol
            else np.asarray(protocol["meas_select"])
        ),
        normalize_measurements=bool(
            np.asarray(protocol.get("normalize_measurements", False)).reshape(-1)[0]
        ),
        current_density=(
            None
            if "current_density" not in protocol
            or not bool(
                np.asarray(protocol.get("current_density_present", False)).reshape(-1)[
                    0
                ]
            )
            else np.asarray(protocol["current_density"]).reshape(-1)[0]
        ),
    )


def _source_element_field(
    fields: Mapping[str, Any],
    *,
    name: str,
    n_elements: int,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    if name not in fields:
        if fallback is None:
            raise ValueError(f"Bridge v3 fields is missing {name}")
        return np.array(fallback, copy=True)
    values = np.asarray(fields[name])
    if values.ndim == 0:
        values = values.reshape(1)
    if values.shape[0] != n_elements:
        raise ValueError(f"{name} must have one source row per element")
    return np.ascontiguousarray(values)


def _effective_measurement_matrices(
    protocol: ProtocolSpec,
) -> tuple[np.ndarray, ...]:
    """Apply an order-preserving EIDORS meas_select to protocol rows."""

    selector = protocol.meas_select
    if selector is None or np.asarray(selector).size == 0:
        return protocol.meas_matrices
    total = sum(int(matrix.shape[0]) for matrix in protocol.meas_matrices)
    values = np.asarray(selector).reshape(-1)
    if values.dtype == np.bool_ or (
        values.size == total and np.all(np.isin(values, [0, 1]))
    ):
        if values.size != total:
            raise ValueError("Bridge v3 meas_select mask length mismatch")
        selected = np.flatnonzero(values.astype(bool))
    else:
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError("Bridge v3 meas_select indices must be finite integers")
        selected = values.astype(np.int64)
        if selected.size and np.min(selected) >= 1 and np.max(selected) <= total:
            selected = selected - 1
        elif selected.size and (np.min(selected) < 0 or np.max(selected) >= total):
            raise ValueError("Bridge v3 meas_select index is out of range")
    if selected.size != np.unique(selected).size or np.any(np.diff(selected) <= 0):
        raise ValueError(
            "Bridge v3 meas_select must be a unique order-preserving selection"
        )

    matrices: list[np.ndarray] = []
    offset = 0
    for matrix in protocol.meas_matrices:
        stop = offset + int(matrix.shape[0])
        local = selected[(selected >= offset) & (selected < stop)] - offset
        matrices.append(np.ascontiguousarray(matrix[local]))
        offset = stop
    return tuple(matrices)


@dataclass
class ModelContext:
    """Fully resolved immutable model data used by every runtime workflow."""

    registered: RegisteredModel
    package: BridgeV3Package
    mesh: Any
    geometry: dict[str, Any]
    electrode_specs: tuple[ElectrodeSpec, ...]
    protocol: ProtocolSpec
    background_source: np.ndarray
    background_local: np.ndarray
    target_source: np.ndarray
    target_local: np.ndarray
    coarse2fine_source: np.ndarray | None
    coarse2fine_local: np.ndarray | None
    cache_key: str

    @property
    def effective_meas_matrices(self) -> tuple[np.ndarray, ...]:
        return _effective_measurement_matrices(self.protocol)

    @property
    def measurement_count(self) -> int:
        return sum(int(matrix.shape[0]) for matrix in self.effective_meas_matrices)

    def create_system(
        self,
        *,
        initialize_inverse: bool = False,
        runtime_stim_matrix: Any | None = None,
        **overrides: Any,
    ):
        """Create the authoritative EITSystem without regenerating geometry."""

        from pyeidors import EITSystem
        from pyeidors.data import PatternConfig

        kinds = {spec.kind for spec in self.electrode_specs}
        electrode_model = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        stimulation = self.protocol.stim_matrix
        if runtime_stim_matrix is not None:
            stimulation = np.asarray(runtime_stim_matrix)
            if stimulation.shape != self.protocol.stim_matrix.shape:
                raise ValueError("Runtime stimulation shape differs from Bridge v3")
            if np.iscomplexobj(stimulation) and not np.allclose(
                np.imag(stimulation),
                0.0,
            ):
                raise ValueError("Runtime stimulation must be real")
            stimulation = np.ascontiguousarray(
                np.real(stimulation),
                dtype=np.float64,
            )
            if not np.all(np.isfinite(stimulation)):
                raise ValueError("Runtime stimulation must be finite")
            if np.any(np.linalg.norm(stimulation, axis=1) <= 1.0e-12):
                raise ValueError("Runtime stimulation rows must be non-zero")
        pattern = PatternConfig(
            n_elec=len(self.electrode_specs),
            n_rings=1,
            measurement_protocol="custom",
            custom_stim_matrix=stimulation,
            custom_meas_matrices=list(self.effective_meas_matrices),
            drive_mode="total_current",
        )
        potential_order = int(self.package.model.get("potential_order", 1) or 1)
        options = {
            "n_elec": len(self.electrode_specs),
            "pattern_config": pattern,
            "electrode_model": electrode_model,
            "contact_impedance": None,
            "base_conductivity": np.asarray(self.background_source).reshape(-1)[0],
            "potential_order": potential_order,
        }
        options.update(overrides)
        system = EITSystem(**options)
        system.setup(mesh=self.mesh, initialize_inverse=initialize_inverse)
        return system


class ModelContextFactory:
    """The sole Bridge v3 package-to-runtime model resolution path."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def create(self, model: str | RegisteredModel) -> ModelContext:
        model_id = model.model_id if isinstance(model, RegisteredModel) else str(model)
        registered = self.registry.get(model_id)
        package = self.registry.load_package(model_id)
        geometry_path = registered.asset_path / "geometry.mat"
        mesh, geometry = build_mesh_from_exchange_mat(geometry_path)
        electrode_specs = tuple(getattr(mesh, "electrode_specs", ()) or ())
        if not electrode_specs:
            raise ValueError("Imported Bridge v3 mesh has no electrode_specs")
        protocol = _protocol_spec(
            package.protocol,
            n_elec=len(electrode_specs),
        )
        n_elements = int(np.asarray(package.geometry["elems"]).shape[0])
        fields = _public_mat_payload(package.fields)
        background_source = _source_element_field(
            fields,
            name="background_elem_data",
            n_elements=n_elements,
        )
        target_source = _source_element_field(
            fields,
            name="target_elem_data",
            n_elements=n_elements,
            fallback=background_source,
        )
        background_local = source_cell_data_to_local(
            mesh,
            background_source,
            name="background_elem_data",
        )
        target_local = source_cell_data_to_local(
            mesh,
            target_source,
            name="target_elem_data",
        )
        coarse2fine_source = None
        coarse2fine_local = None
        if "coarse2fine" in fields and np.asarray(fields["coarse2fine"]).size:
            coarse2fine_source = np.asarray(fields["coarse2fine"])
            if coarse2fine_source.ndim == 1:
                if coarse2fine_source.size % n_elements:
                    raise ValueError("coarse2fine source row count is ambiguous")
                coarse2fine_source = coarse2fine_source.reshape(n_elements, -1)
            coarse2fine_local = source_cell_data_to_local(
                mesh,
                coarse2fine_source,
                name="coarse2fine",
            )
        return ModelContext(
            registered=registered,
            package=package,
            mesh=mesh,
            geometry=geometry,
            electrode_specs=electrode_specs,
            protocol=protocol,
            background_source=background_source,
            background_local=background_local,
            target_source=target_source,
            target_local=target_local,
            coarse2fine_source=coarse2fine_source,
            coarse2fine_local=coarse2fine_local,
            cache_key=package.forward_fingerprint,
        )

    def for_flow(self, flow: str) -> ModelContext:
        registered = self.registry.bound_model(flow)
        if registered is None:
            raise KeyError(f"No Bridge v3 model is bound to flow {flow!r}")
        return self.create(registered)


__all__ = [
    "MODEL_BINDING_FLOWS",
    "MODEL_REGISTRY_SCHEMA_VERSION",
    "ModelContext",
    "ModelContextFactory",
    "ModelRegistry",
    "RegisteredModel",
]
