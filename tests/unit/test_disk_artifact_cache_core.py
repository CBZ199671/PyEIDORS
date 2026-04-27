"""T82 phase 1: persistent disk artifact key/manifest core."""

from __future__ import annotations

from pathlib import Path

from pyeidors.cache.disk_artifacts import (
    build_disk_artifact_key,
    build_disk_artifact_manifest,
    build_disk_artifact_subkey,
    build_disk_artifact_subkeys,
    file_fingerprint,
    file_sha256,
    stable_json_digest,
)


def test_stable_json_digest_is_order_independent() -> None:
    left = {"mesh": {"gdim": 3, "n_elec": 48}, "arrays": ["RM", "Y"]}
    right = {"arrays": ["RM", "Y"], "mesh": {"n_elec": 48, "gdim": 3}}
    assert stable_json_digest(left) == stable_json_digest(right)


def test_disk_artifact_key_excludes_output_file_locations(tmp_path: Path) -> None:
    payload = {
        "format": "dolfinx-xdmf-hdf5",
        "gdim": 3,
        "mesh_content_signature": {"geometry_hash": "g", "topology_hash": "t"},
    }
    path_a = tmp_path / "a" / "mesh.xdmf"
    path_b = tmp_path / "b" / "mesh.xdmf"
    path_a.parent.mkdir()
    path_b.parent.mkdir()
    path_a.write_text("<xdmf/>", encoding="utf-8")
    path_b.write_text("<xdmf/>", encoding="utf-8")

    manifest_a = build_disk_artifact_manifest(
        "dolfinx-mesh-cache",
        payload,
        files={"xdmf": path_a},
        metadata={"cache_format": "dolfinx-xdmf-hdf5"},
    )
    manifest_b = build_disk_artifact_manifest(
        "dolfinx-mesh-cache",
        payload,
        files={"xdmf": path_b},
        metadata={"cache_format": "dolfinx-xdmf-hdf5"},
    )

    assert manifest_a.artifact_key == manifest_b.artifact_key
    assert manifest_a.artifact_key == build_disk_artifact_key(
        "dolfinx-mesh-cache",
        payload,
    )
    meta_a = manifest_a.to_metadata()
    meta_b = manifest_b.to_metadata()
    assert meta_a["files"]["xdmf"]["path"] != meta_b["files"]["xdmf"]["path"]
    assert meta_a["artifact_key"] == meta_b["artifact_key"]


def test_disk_artifact_subkeys_are_artifact_kind_independent() -> None:
    payload = {
        "gdim": 3,
        "mesh_content_signature": {"geometry_hash": "g", "topology_hash": "t"},
        "association_table": {"domain": 1, "electrode_1": 2},
    }
    mesh_manifest = build_disk_artifact_manifest(
        "dolfinx-mesh-cache",
        {"format": "dolfinx-xdmf-hdf5", **payload},
        subkey_payloads={"mesh_provenance": payload},
    )
    hdf5_manifest = build_disk_artifact_manifest(
        "hdf5-artifact",
        {"schema": "unit-cache-v1", "arrays": {"RM": {"shape": [2, 2]}}},
        subkey_payloads={"mesh_provenance": payload},
    )

    expected = build_disk_artifact_subkey("mesh_provenance", payload)
    assert mesh_manifest.artifact_key != hdf5_manifest.artifact_key
    assert mesh_manifest.to_metadata()["subkeys"]["mesh_provenance"] == expected
    assert hdf5_manifest.to_metadata()["subkeys"]["mesh_provenance"] == expected
    assert build_disk_artifact_subkeys({"mesh_provenance": payload}) == {
        "mesh_provenance": expected
    }


def test_file_fingerprint_and_sha256_are_optional_for_manifest_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.h5"
    target.write_bytes(b"payload")

    fingerprint = file_fingerprint(target, include_sha256=True)
    assert fingerprint is not None
    assert fingerprint["exists"] is True
    assert fingerprint["size"] == 7
    assert fingerprint["sha256"] == file_sha256(target)

    missing = file_fingerprint(tmp_path / "missing.h5", include_sha256=True)
    assert missing is not None
    assert missing["exists"] is False
    assert "sha256" not in missing


def test_file_fingerprint_supports_adios2_directory_artifacts(tmp_path: Path) -> None:
    adios_dir = tmp_path / "mesh_adios4dolfinx.bp"
    adios_dir.mkdir()
    (adios_dir / "md.idx").write_text("index", encoding="utf-8")

    fingerprint = file_fingerprint(adios_dir, include_sha256=True)

    assert fingerprint is not None
    assert fingerprint["exists"] is True
    assert fingerprint["is_dir"] is True
    assert fingerprint["path"].endswith(".bp")
    assert "sha256" not in fingerprint
