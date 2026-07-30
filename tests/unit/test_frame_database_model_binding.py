"""Frame database schema-v3 and immutable model binding gates."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sqlite3

import numpy as np
import pytest

from eit_app.controllers.reconstruction_controller import (
    ReconstructionRequest,
    clear_reconstruction_system_cache,
    run_reconstruction_request,
)
from eit_app.models.frame_database import FrameDatabase
from eit_app.models.frame_model import FrameData
from pyeidors.interop import prove_protocol_mapping


def _create_v2_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            session_dir TEXT NOT NULL UNIQUE,
            started_at TEXT NOT NULL,
            n_elec INTEGER,
            stim_pattern TEXT,
            meas_pattern TEXT,
            frequency_hz INTEGER,
            frequency_hz_min INTEGER,
            frequency_hz_max INTEGER,
            stim_amp_uA INTEGER,
            voltage_amp_level INTEGER,
            transport_type TEXT,
            mea_mode INTEGER,
            notes TEXT,
            metadata_json TEXT
        );
        CREATE TABLE frames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            frame_index INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            csv_path TEXT NOT NULL UNIQUE,
            yaml_path TEXT,
            frame_metadata_json TEXT,
            UNIQUE(session_id, frame_index)
        );
        PRAGMA user_version = 2;
        """
    )
    session_dir = path.parent / "legacy-session"
    session_dir.mkdir()
    connection.execute(
        """
        INSERT INTO sessions (
            name, session_dir, started_at, n_elec, metadata_json
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            "legacy",
            str(session_dir.resolve()),
            "2026-01-01T00:00:00",
            2,
            "{}",
        ),
    )
    connection.commit()
    connection.close()


class _Registry:
    def __init__(self, *, n_elec: int = 2) -> None:
        self.registered = SimpleNamespace(
            model_id="a" * 64,
            forward_fingerprint="b" * 64,
            protocol_layout_hash="c" * 64,
            protocol_physics_hash="d" * 64,
        )
        self.package = SimpleNamespace(
            model={"n_elec": n_elec},
            geometry={"n_elec": n_elec},
        )
        self.loaded: list[str] = []

    def load_package(self, model_id: str):
        self.loaded.append(model_id)
        if model_id != self.registered.model_id:
            raise KeyError(model_id)
        return self.package

    def get(self, model_id: str):
        if model_id != self.registered.model_id:
            raise KeyError(model_id)
        return self.registered


def _identity_mapping() -> dict[str, object]:
    stim = np.asarray([[1.0, -1.0]])
    meas = (np.asarray([[1.0, -1.0]]),)
    return prove_protocol_mapping(
        model_stim_matrix=stim,
        model_meas_matrices=meas,
        hardware_stim_matrix=stim,
        hardware_meas_matrices=meas,
    ).to_mapping()


def test_v758_v2_migration_adds_nullable_model_fields_without_guessing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frames.sqlite3"
    _create_v2_database(path)

    database = FrameDatabase(path)
    try:
        columns = {
            str(row["name"])
            for row in database._conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        assert {
            "model_id",
            "forward_fingerprint",
            "protocol_layout_hash",
            "protocol_physics_hash",
            "channel_mapping_json",
        } <= columns
        assert database._conn.execute("PRAGMA user_version").fetchone()[0] == 3
        session = database.get_session(1)
        assert session is not None
        assert session["model_id"] is None
        assert session["channel_mapping_json"] is None
    finally:
        database.close()


def test_v758_session_binding_validates_and_persists_exact_identity(
    tmp_path: Path,
) -> None:
    database = FrameDatabase(tmp_path / "frames.sqlite3")
    registry = _Registry()
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_id = database.add_session(session_dir, {"n_elec": 2})
    mapping = _identity_mapping()

    try:
        bound = database.bind_session_model(
            session_id,
            registry.registered.model_id,
            registry=registry,
            channel_mapping=mapping,
        )
        assert registry.loaded == [registry.registered.model_id]
        assert bound["model_id"] == registry.registered.model_id
        assert bound["forward_fingerprint"] == registry.registered.forward_fingerprint
        assert bound["protocol_layout_hash"] == registry.registered.protocol_layout_hash
        assert (
            bound["protocol_physics_hash"] == registry.registered.protocol_physics_hash
        )
        assert json.loads(bound["channel_mapping_json"]) == mapping
    finally:
        database.close()


@pytest.mark.parametrize(
    ("metadata", "registry", "message"),
    [
        ({"n_elec": 3}, _Registry(n_elec=2), "electrode count mismatch"),
        (
            {"n_elec": 2, "protocol_layout_hash": "e" * 64},
            _Registry(n_elec=2),
            "protocol layout hash mismatch",
        ),
    ],
)
def test_v758_invalid_session_binding_is_not_persisted(
    tmp_path: Path,
    metadata: dict[str, object],
    registry: _Registry,
    message: str,
) -> None:
    database = FrameDatabase(tmp_path / "frames.sqlite3")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_id = database.add_session(session_dir, metadata)

    try:
        with pytest.raises(ValueError, match=message):
            database.bind_session_model(
                session_id,
                registry.registered.model_id,
                registry=registry,
            )
        session = database.get_session(session_id)
        assert session is not None
        assert session["model_id"] is None
    finally:
        database.close()


def test_v758_bound_reconstruction_uses_model_context_without_generated_mesh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyeidors.interop as interop

    clear_reconstruction_system_cache()
    created: list[dict[str, object]] = []
    reconstructed: list[tuple[object, object]] = []
    registered = SimpleNamespace(
        model_id="a" * 64,
        forward_fingerprint="b" * 64,
        protocol_layout_hash="c" * 64,
        protocol_physics_hash="d" * 64,
    )

    class FakeSystem:
        def difference_reconstruct(self, *, measurement_data, reference_data):
            reconstructed.append((measurement_data, reference_data))
            return SimpleNamespace(
                conductivity=np.asarray([1.25, 1.5]),
                measured=np.asarray([0.2]),
                simulated=np.asarray([0.15]),
            )

    context = SimpleNamespace(
        registered=registered,
        electrode_specs=(
            SimpleNamespace(kind="pem"),
            SimpleNamespace(kind="pem"),
        ),
        protocol=SimpleNamespace(
            stim_matrix=np.asarray([[1.0, -1.0]]),
            normalize_measurements=True,
        ),
        effective_meas_matrices=(np.asarray([[1.0, -1.0]]),),
        measurement_count=1,
        coarse2fine_local=None,
        cache_key=registered.forward_fingerprint,
        mesh=SimpleNamespace(
            coordinates=lambda: np.asarray([[0.0, 0.0], [1.0, 0.0]]),
            cells=lambda: np.asarray([[0, 1]], dtype=np.int32),
        ),
        create_system=lambda **kwargs: created.append(dict(kwargs)) or FakeSystem(),
    )

    class FakeFactory:
        def __init__(self, registry) -> None:
            assert registry is not None

        def create(self, model_id: str):
            assert model_id == registered.model_id
            return context

    monkeypatch.setattr(interop, "ModelRegistry", lambda: object())
    monkeypatch.setattr(interop, "ModelContextFactory", FakeFactory)
    frame = FrameData(
        real=np.asarray([0.1]),
        imag=np.asarray([0.0]),
        timestamp=0.0,
        frame_index=0,
    )
    target = FrameData(
        real=np.asarray([0.2]),
        imag=np.asarray([0.0]),
        timestamp=1.0,
        frame_index=1,
    )
    request = ReconstructionRequest(
        reference_frame=frame,
        target_frame=target,
        method="gn-difference",
        metadata={
            "request_source": "db",
            "model_id": registered.model_id,
            "forward_fingerprint": registered.forward_fingerprint,
            "protocol_layout_hash": registered.protocol_layout_hash,
            "protocol_physics_hash": registered.protocol_physics_hash,
            "channel_mapping": _identity_mapping(),
        },
    )

    result = run_reconstruction_request(request)

    assert result.error_msg is None
    np.testing.assert_allclose(result.conductivity, [1.25, 1.5])
    assert result.metadata["reconstruction_runtime"] == "bridge_v3_bound_model"
    assert created and created[0]["initialize_inverse"] is True
    np.testing.assert_allclose(created[0]["runtime_stim_matrix"], [[1.0, -1.0]])
    assert created[0]["difference_mode"] == "normalized"
    assert len(reconstructed) == 1
    clear_reconstruction_system_cache()
