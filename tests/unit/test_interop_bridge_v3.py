"""Bridge Package v3-only identity and integrity gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eit_app.interop.bridge_package import (
    default_manifest,
    load_bridge_package,
    save_bridge_package,
    validate_bridge_package,
)
from eit_app.models.forward_model_config import ForwardModelConfig
from pyeidors.interop import (
    BRIDGE_PACKAGE_FORMAT_V3,
    BridgeV3Package,
    ElectrodeSpec,
    GEOMETRY_FORMAT_V3,
    ProtocolSpec,
    build_bridge_fingerprints,
    validate_bridge_v3_package,
)


def _payloads() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    model = {
        "schema_version": 3,
        "n_elec": 4,
        "dimension": 3,
        "potential_order": 1,
        "forward_ready": True,
    }
    geometry = {
        "index_base": 1,
        "source_framework": "pyeidors",
        "dimension": 3,
        "cell_type": "tetrahedron",
        "boundary_entity_type": "triangle",
        "nodes": np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        "elems": np.asarray([[1, 2, 3, 4]], dtype=np.int64),
        "boundary_facets": np.asarray(
            [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
            dtype=np.int64,
        ),
        "boundary_edges": np.asarray(
            [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
            dtype=np.int64,
        ),
        "electrode_nodes": np.asarray(
            [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
            dtype=np.int64,
        ),
        "electrode_node_counts": np.full(4, 3, dtype=np.int64),
        "n_elec": 4,
        "background": 1.0,
        "truth_elem_data": np.asarray([2.0]),
        "contact_impedance": np.full(4, 0.01),
        "mesh_name": "v3-unit-tetra",
        "mesh_level": "unit",
        "scenario_name": "v3",
    }
    stim = np.asarray(
        [
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 1.0, -1.0, 0.0],
        ]
    )
    protocol = {
        "stim_matrix": stim,
        "stim_matrix_raw": stim,
        "measurement_matrix": np.asarray(
            [[0.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, -1.0]]
        ),
        "measurement_counts": np.asarray([1, 1], dtype=np.int64),
        "normalize_measurements": False,
    }
    fields = {
        "background_elem_data": np.asarray([1.0]),
        "target_elem_data": np.asarray([2.0]),
    }
    return model, geometry, protocol, fields


def _write_package(root: Path) -> BridgeV3Package:
    model, geometry, protocol, fields = _payloads()
    return BridgeV3Package.write(
        root,
        model=model,
        geometry=geometry,
        protocol=protocol,
        fields=fields,
        measurements={
            "homogeneous": np.asarray([0.1, 0.2]),
            "target": np.asarray([0.2, 0.3]),
        },
    )


def test_v753_v3_roundtrip_has_required_identity_and_files(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "bridge_v3")
    report = validate_bridge_v3_package(package.root)

    assert report["valid"] is True
    assert report["package_format"] == BRIDGE_PACKAGE_FORMAT_V3
    assert package.geometry["exchange_format"] == GEOMETRY_FORMAT_V3
    assert package.geometry["schema_version"] == 3
    assert len(package.model_id) == 64
    assert len(package.forward_fingerprint) == 64
    assert set(package.manifest["files"]) >= {
        "model",
        "geometry",
        "protocol",
        "fields",
        "measurements",
    }
    assert package.fields["background_elem_data"] == pytest.approx(1.0)


def test_v776_rewrite_removes_omitted_optional_package_files(
    tmp_path: Path,
) -> None:
    model, geometry, protocol, fields = _payloads()
    root = tmp_path / "bridge_v3"
    BridgeV3Package.write(
        root,
        model=model,
        geometry=geometry,
        protocol=protocol,
        fields=fields,
        measurements={"homogeneous": np.asarray([0.1, 0.2])},
        reconstruction={"method": "noser_rm"},
    )

    package = BridgeV3Package.write(
        root,
        model=model,
        geometry=geometry,
        protocol=protocol,
        fields=fields,
    )

    assert package.measurements is None
    assert package.reconstruction is None
    assert not (root / "measurements.mat").exists()
    assert not (root / "reconstruction.json").exists()
    assert set(package.manifest["files"]) == {
        "model",
        "geometry",
        "protocol",
        "fields",
    }


def test_v764_singleton_complex_measurement_has_stable_identity(
    tmp_path: Path,
) -> None:
    model, geometry, protocol, fields = _payloads()
    package = BridgeV3Package.write(
        tmp_path / "complex_scalar",
        model=model,
        geometry=geometry,
        protocol=protocol,
        fields=fields,
        measurements={
            "homogeneous": np.asarray([1.0 + 0.25j]),
            "target": np.asarray([1.1 + 0.5j]),
        },
    )

    reloaded = BridgeV3Package.load(package.root)

    assert reloaded.model_id == package.model_id
    assert reloaded.measurements["homogeneous"] == pytest.approx(1.0 + 0.25j)


def test_v753_v3_integrity_rejects_tampered_file(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "bridge_v3")
    geometry_path = package.root / "geometry.mat"
    with geometry_path.open("ab") as stream:
        stream.write(b"tamper")

    report = validate_bridge_v3_package(package.root)

    assert report["valid"] is False
    assert any(
        "size mismatch" in error or "SHA-256 mismatch" in error
        for error in report["errors"]
    )
    with pytest.raises(ValueError, match="mismatch"):
        BridgeV3Package.load(package.root)


def test_v753_v3_only_rejects_v2_and_standalone_mat(tmp_path: Path) -> None:
    package = _write_package(tmp_path / "bridge_v3")
    manifest_path = package.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exchange_format"] = "eidors_pyeidors_bridge_v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_bridge_v3_package(package.root)

    assert report["valid"] is False
    assert any("v1/v2" in error for error in report["errors"])
    standalone = package.root / "geometry.mat"
    with pytest.raises(ValueError, match="standalone MAT"):
        BridgeV3Package.load(standalone)


def test_v753_protocol_layout_ignores_only_drive_amplitude() -> None:
    model, geometry, protocol, fields = _payloads()
    first = build_bridge_fingerprints(
        model=model,
        geometry=geometry,
        protocol=protocol,
        fields=fields,
    )
    scaled_protocol = dict(protocol)
    scaled_protocol["stim_matrix"] = np.asarray(protocol["stim_matrix"]) * 0.005
    second = build_bridge_fingerprints(
        model=model,
        geometry=geometry,
        protocol=scaled_protocol,
        fields=fields,
    )

    assert first["protocol_layout_hash"] == second["protocol_layout_hash"]
    assert first["protocol_physics_hash"] != second["protocol_physics_hash"]
    assert first["forward_fingerprint"] != second["forward_fingerprint"]
    assert first["model_id"] != second["model_id"]


def test_v753_typed_electrode_and_protocol_contracts() -> None:
    pem = ElectrodeSpec(
        kind="pem",
        source_nodes=(1, 2),
        node_weights=(0.25, 0.75),
        boundary_kind="none",
        contact_impedance=0.01,
        contact_impedance_present=True,
        contact_impedance_applicable=False,
    )
    cem = ElectrodeSpec(
        kind="cem",
        source_nodes=(1, 2),
        source_faces=((1, 2),),
        boundary_kind="exterior",
        contact_impedance=0.01,
        contact_impedance_present=True,
        contact_impedance_applicable=True,
    )
    protocol = ProtocolSpec(
        stim_matrix=np.asarray([[1.0, -1.0]]),
        meas_matrices=(np.asarray([[1.0, -1.0]]),),
    )

    assert pem.kind == "pem"
    assert cem.kind == "cem"
    assert protocol.stim_matrix.shape == (1, 2)
    missing_cem = ElectrodeSpec(
        kind="cem",
        source_nodes=(1, 2),
        source_faces=((1, 2),),
        boundary_kind="exterior",
        contact_impedance=None,
        contact_impedance_present=False,
        contact_impedance_applicable=True,
    )
    assert missing_cem.contact_impedance is None
    with pytest.raises(ValueError, match="sum to one"):
        ElectrodeSpec(
            kind="pem",
            source_nodes=(1, 2),
            node_weights=(0.2, 0.2),
        )
    with pytest.raises(ValueError, match="contact_impedance_present"):
        ElectrodeSpec(
            kind="cem",
            source_faces=((1, 2),),
            contact_impedance_present=True,
            contact_impedance_applicable=True,
        )


def test_v753_manifest_schema_is_valid_json() -> None:
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "schemas"
        / "interop"
        / "eidors_pyeidors_bridge_v3.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["exchange_format"]["const"] == (
        BRIDGE_PACKAGE_FORMAT_V3
    )
    assert schema["properties"]["schema_version"]["const"] == 3


def test_v753_application_bridge_uses_v3_core_roundtrip(tmp_path: Path) -> None:
    _, geometry, protocol, fields = _payloads()
    geometry.update(protocol)
    geometry.update(fields)
    config = ForwardModelConfig(
        mesh_dimension=3,
        mesh_family="tetrahedron",
        n_elec=4,
        electrode_model="cem",
        measurement_protocol="custom",
        custom_stim_matrix=protocol["stim_matrix"],
        custom_meas_matrices=[protocol["measurement_matrix"]],
        contact_impedance=[0.01] * 4,
    )
    root = save_bridge_package(
        tmp_path / "application_bridge_v3",
        default_manifest(source_framework="pyeidors", package_kind="unit_test"),
        geometry_payload=geometry,
        forward_model_config=config,
    )

    report = validate_bridge_package(root)
    loaded = load_bridge_package(root)

    assert report["valid"] is True
    assert report["package_format"] == BRIDGE_PACKAGE_FORMAT_V3
    assert report["model_id"] == loaded.manifest.model_id
    assert loaded.forward_model_config is not None
    assert loaded.forward_model_config.mesh_source == "interop"
    assert (
        Path(loaded.forward_model_config.mesh_path) == (root / "geometry.mat").resolve()
    )
