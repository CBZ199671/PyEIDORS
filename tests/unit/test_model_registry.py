"""Managed Bridge v3 registry and authoritative ModelContext gates."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import stat
import subprocess
import sys

import numpy as np
import pytest
import pyeidors.interop.model_registry as model_registry_module

from pyeidors.interop import (
    BridgeV3Package,
    MODEL_BINDING_FLOWS,
    ModelContextFactory,
    ModelRegistry,
)


def _write_mixed_package(root: Path) -> BridgeV3Package:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    geometry = {
        "index_base": 1,
        "source_framework": "pyeidors",
        "dimension": 2,
        "cell_type": "triangle",
        "boundary_entity_type": "edge",
        "nodes": nodes,
        "elems": np.asarray([[1, 2, 3], [1, 3, 4]], dtype=np.int64),
        "boundary_facets": np.asarray(
            [[1, 2], [2, 3], [3, 4], [4, 1]],
            dtype=np.int64,
        ),
        "boundary_edges": np.asarray(
            [[1, 2], [2, 3], [3, 4], [4, 1]],
            dtype=np.int64,
        ),
        "electrode_nodes": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "electrode_node_counts": np.asarray([2, 2], dtype=np.int64),
        "electrode_model": ["cem_faces", "distributed_point"],
        "electrode_boundary_kind": ["exterior", "none"],
        "pem_node_weights": np.asarray([[0.0, 0.0], [0.25, 0.75]]),
        "cem_face_nodes": np.asarray([[1, 2]], dtype=np.int64),
        "cem_face_node_counts": np.asarray([2], dtype=np.int64),
        "cem_face_electrode": np.asarray([1], dtype=np.int64),
        "n_elec": 2,
        "background": 1.0,
        "background_present": True,
        "background_elem_data": np.asarray([1.0, 2.0]),
        "background_elem_data_present": True,
        "truth_elem_data": np.asarray([1.0, 1.5]),
        "truth_elem_data_present": True,
        "target_elem_data": np.asarray([1.0, 1.5]),
        "contact_impedance": np.asarray([0.02, np.nan]),
        "contact_impedance_present": np.asarray([True, False]),
        "contact_impedance_applicable": np.asarray([True, False]),
        "effective_gnd_node": 1,
        "normalize_measurements": False,
        "mesh_name": "registry_mixed",
        "mesh_level": "unit",
        "scenario_name": "registry",
    }
    protocol = {
        "stim_matrix": np.asarray([[1.0, -1.0]]),
        "stim_matrix_raw": np.asarray([[1.0, -1.0]]),
        "meas_matrices": np.asarray([[[1.0, -1.0]]]),
        "measurement_counts": np.asarray([1], dtype=np.int64),
        "stimulation_supported": True,
        "normalize_measurements": False,
    }
    fields = {
        "background": 1.0,
        "background_present": True,
        "background_elem_data": np.asarray([1.0, 2.0]),
        "target_elem_data": np.asarray([1.0, 1.5]),
        "coarse2fine": np.asarray([[1.0], [0.5]]),
    }
    return BridgeV3Package.write(
        root,
        model={
            "schema_version": 3,
            "name": "registry mixed model",
            "n_elec": 2,
            "dimension": 2,
            "potential_order": 1,
            "forward_ready": True,
        },
        geometry=geometry,
        protocol=protocol,
        fields=fields,
        capabilities={"forward_ready": True},
    )


def test_v756_registry_atomically_deduplicates_and_survives_source_delete(
    tmp_path: Path,
) -> None:
    source = _write_mixed_package(tmp_path / "source")
    registry = ModelRegistry(tmp_path / "registry")

    first = registry.register(source.root, display_name="Mixed")
    second = registry.register(source.root, display_name="Mixed renamed")

    assert first.model_id == second.model_id
    assert len(registry.list_models()) == 1
    assert second.asset_path == registry.root / second.model_id
    assert second.asset_path != source.root
    shutil.rmtree(source.root)
    loaded = registry.load_package(second.model_id)
    assert loaded.model_id == second.model_id
    assert registry.get(second.model_id).display_name == "Mixed renamed"


def test_v756_registry_marks_tampered_managed_asset_corrupt(
    tmp_path: Path,
) -> None:
    source = _write_mixed_package(tmp_path / "source")
    registry = ModelRegistry(tmp_path / "registry")
    registered = registry.register(source.root)
    geometry_path = registered.asset_path / "geometry.mat"
    geometry_path.chmod(geometry_path.stat().st_mode | stat.S_IWUSR)
    with geometry_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ValueError, match="failed integrity"):
        registry.load_package(registered.model_id)

    damaged = registry.get(registered.model_id)
    assert damaged.status == "corrupt"
    assert "mismatch" in damaged.integrity_error


def test_v757_registry_bindings_and_context_use_exact_fields_and_protocol(
    tmp_path: Path,
) -> None:
    source = _write_mixed_package(tmp_path / "source")
    registry = ModelRegistry(tmp_path / "registry")
    registered = registry.register(source.root)

    applied = registry.apply_to_all(registered.model_id)
    assert set(applied) == set(MODEL_BINDING_FLOWS)
    assert {flow: model.model_id for flow, model in registry.bindings().items()} == {
        flow: registered.model_id for flow in MODEL_BINDING_FLOWS
    }

    context = ModelContextFactory(registry).for_flow("simulation")
    source_indices = np.asarray(context.mesh.source_cell_indices, dtype=np.int64)
    np.testing.assert_allclose(
        context.background_local,
        context.background_source[source_indices],
    )
    np.testing.assert_allclose(
        context.target_local,
        context.target_source[source_indices],
    )
    np.testing.assert_allclose(context.protocol.stim_matrix, [[1.0, -1.0]])
    np.testing.assert_allclose(
        context.protocol.meas_matrices[0],
        [[1.0, -1.0]],
    )
    assert [spec.kind for spec in context.electrode_specs] == ["cem", "pem"]
    assert context.cache_key == registered.forward_fingerprint
    assert context.coarse2fine_local is not None

    system = context.create_system(initialize_inverse=False)
    data = system.forward_solve(context.background_local)
    assert np.asarray(data.meas).shape == (1,)
    assert np.isfinite(np.asarray(data.meas)).all()


def test_v765_registry_closes_every_sqlite_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_connect = sqlite3.connect
    tracked: list[TrackingConnection] = []

    class TrackingConnection(sqlite3.Connection):
        closed_by_registry = False

        def close(self) -> None:
            self.closed_by_registry = True
            super().close()

    def tracking_connect(*args: object, **kwargs: object) -> TrackingConnection:
        kwargs["factory"] = TrackingConnection
        connection = real_connect(*args, **kwargs)
        assert isinstance(connection, TrackingConnection)
        tracked.append(connection)
        return connection

    monkeypatch.setattr(model_registry_module.sqlite3, "connect", tracking_connect)
    registry = ModelRegistry(tmp_path / "registry")
    assert not registry.list_models()
    assert tracked
    assert all(connection.closed_by_registry for connection in tracked)


def test_t605_cli_registers_managed_v3_package(tmp_path: Path) -> None:
    source = _write_mixed_package(tmp_path / "source")
    registry_root = tmp_path / "registry"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyeidors.interop.cli",
            "register",
            str(source.root),
            "--name",
            "CLI mixed",
            "--registry-dir",
            str(registry_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["model_id"] == source.model_id
    assert payload["display_name"] == "CLI mixed"
    assert Path(payload["asset_path"]).parent == registry_root.resolve()
