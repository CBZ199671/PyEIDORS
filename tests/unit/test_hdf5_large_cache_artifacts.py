from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from pyeidors.io.hdf5_artifacts import (
    HDF5LazyDataset,
    migrate_npz_to_hdf5,
    read_hdf5_artifact,
    verify_hdf5_artifact_checksums,
    write_large_cache_hdf5_artifact,
)


def _mesh_provenance_payload() -> dict[str, object]:
    return {
        "format": "dolfinx-xdmf-hdf5",
        "gdim": 2,
        "mesh_content_signature": {
            "tdim": 2,
            "geometry_hash": "geom",
            "topology_hash": "topo",
        },
        "association_table": {"domain": 1, "electrode_1": 2},
        "physical_groups": {
            "domain": {"tag": 1, "dim": 2},
            "electrode_1": {"tag": 2, "dim": 1},
        },
        "mesh_family": "tet",
        "geometry_version": "unit",
        "generator_revision": "unit-rev",
        "structured_sidecar_signature": None,
        "structured_sidecar_version": None,
    }


def _greit_component_arrays() -> dict[str, np.ndarray]:
    y = np.arange(24, dtype=np.float64).reshape(6, 4) / 10.0
    d = np.arange(40, dtype=np.float64).reshape(10, 4) / 20.0
    return {
        "RM": np.arange(60, dtype=np.float64).reshape(10, 6),
        "Y": y,
        "D": d,
        "PJt": d @ y.T,
        "M": y @ y.T + 0.01 * np.eye(6),
        "vh": np.linspace(1.0, 2.0, 6),
        "vi": y + 1.0,
        "xyzr": np.vstack([np.arange(4), np.arange(4) + 1.0, np.ones(4), np.ones(4)]),
    }


def test_large_cache_hdf5_writes_chunked_compressed_checksummed_greit_components(
    tmp_path,
) -> None:
    arrays = _greit_component_arrays()
    path = write_large_cache_hdf5_artifact(
        tmp_path / "greit_large_cache.h5",
        arrays,
        {"package_role": "greit-eidors-components"},
        schema="unit-greit-large-cache-v1",
    )

    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema"] == "unit-greit-large-cache-v1"
        for name in ("RM", "Y", "D", "PJt", "M", "vh", "vi", "xyzr"):
            dataset = handle["arrays"][name]
            assert dataset.compression == "gzip"
            assert dataset.chunks is not None
            assert dataset.attrs["sha256"]
            assert dataset.attrs["compression"] == "gzip"
            assert dataset.attrs["chunks_json"] != "[]"

    digests = verify_hdf5_artifact_checksums(path)
    assert set(arrays).issubset(digests)

    lazy_artifact = read_hdf5_artifact(path, lazy=True)
    assert lazy_artifact.metadata["large_cache"] is True
    assert lazy_artifact.metadata["checksum_algorithm"] == "sha256"
    assert len(lazy_artifact.metadata["artifact_key"]) == 64
    assert (
        lazy_artifact.metadata["artifact_manifest"]["artifact_kind"] == "hdf5-artifact"
    )
    rm = lazy_artifact.arrays["RM"]
    assert isinstance(rm, HDF5LazyDataset)
    assert rm.shape == arrays["RM"].shape
    assert rm.compression == "gzip"
    assert rm.chunks is not None
    np.testing.assert_allclose(rm[0, :3], arrays["RM"][0, :3])
    np.testing.assert_allclose(np.asarray(lazy_artifact.arrays["PJt"]), arrays["PJt"])


def test_large_cache_hdf5_artifact_key_ignores_output_path(tmp_path) -> None:
    arrays = _greit_component_arrays()
    metadata = {"package_role": "greit-eidors-components"}
    path_a = write_large_cache_hdf5_artifact(
        tmp_path / "a" / "greit_cache.h5",
        arrays,
        metadata,
        schema="unit-greit-large-cache-v1",
    )
    path_b = write_large_cache_hdf5_artifact(
        tmp_path / "b" / "greit_cache.h5",
        arrays,
        metadata,
        schema="unit-greit-large-cache-v1",
    )

    meta_a = read_hdf5_artifact(path_a, lazy=True).metadata
    meta_b = read_hdf5_artifact(path_b, lazy=True).metadata

    assert meta_a["artifact_key"] == meta_b["artifact_key"]
    assert (
        meta_a["artifact_manifest"]["files"]["artifact"]["path"]
        != meta_b["artifact_manifest"]["files"]["artifact"]["path"]
    )

    changed = dict(arrays)
    changed["RM"] = arrays["RM"] + 1.0
    path_c = write_large_cache_hdf5_artifact(
        tmp_path / "c" / "greit_cache.h5",
        changed,
        metadata,
        schema="unit-greit-large-cache-v1",
    )
    meta_c = read_hdf5_artifact(path_c, lazy=True).metadata
    assert meta_c["artifact_key"] != meta_a["artifact_key"]


def test_large_cache_hdf5_records_optional_mesh_provenance_subkey(tmp_path) -> None:
    mesh_provenance = _mesh_provenance_payload()
    path = write_large_cache_hdf5_artifact(
        tmp_path / "greit_with_mesh_subkey.h5",
        _greit_component_arrays(),
        {"package_role": "greit-eidors-components"},
        schema="unit-greit-large-cache-v1",
        subkey_payloads={"mesh_provenance": mesh_provenance},
    )

    metadata = read_hdf5_artifact(path, lazy=True).metadata

    assert metadata["artifact_manifest"]["subkeys"]["mesh_provenance"]
    assert (
        metadata["artifact_manifest"]["key_payload"]["metadata"]["package_role"]
        == "greit-eidors-components"
    )


def test_legacy_hdf5_without_manifest_gets_in_memory_artifact_key(tmp_path) -> None:
    arrays = _greit_component_arrays()
    path = write_large_cache_hdf5_artifact(
        tmp_path / "legacy_without_manifest.h5",
        arrays,
        {"package_role": "greit-eidors-components"},
        schema="unit-greit-large-cache-v1",
    )
    expected_key = read_hdf5_artifact(path, lazy=True).metadata["artifact_key"]

    with h5py.File(path, "a") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        metadata.pop("artifact_key", None)
        metadata.pop("artifact_manifest", None)
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)

    artifact = read_hdf5_artifact(path, lazy=True)

    assert artifact.metadata["artifact_key"] == expected_key
    assert artifact.metadata["artifact_manifest"]["artifact_kind"] == "hdf5-artifact"
    with h5py.File(path, "r") as handle:
        persisted = json.loads(str(handle.attrs["metadata_json"]))
    assert "artifact_key" not in persisted
    assert "artifact_manifest" not in persisted


def test_hdf5_artifact_checksum_mismatch_fails_eager_and_lazy_reads(tmp_path) -> None:
    path = write_large_cache_hdf5_artifact(
        tmp_path / "corruptible.h5",
        {"RM": np.eye(3, dtype=np.float64)},
        schema="unit-checksum-v1",
    )
    with h5py.File(path, "a") as handle:
        handle["arrays"]["RM"].attrs["sha256"] = "not-the-real-digest"

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_hdf5_artifact(path)

    lazy_artifact = read_hdf5_artifact(path, lazy=True, verify_checksums=False)
    rm = lazy_artifact.arrays["RM"]
    np.testing.assert_allclose(rm.read(verify_checksum=False), np.eye(3))
    with pytest.raises(ValueError, match="checksum mismatch"):
        rm.read(verify_checksum=True)


def test_npz_migration_is_read_only_source_and_lazy_hdf5_destination(tmp_path) -> None:
    source = tmp_path / "legacy_components.npz"
    arrays = _greit_component_arrays()
    np.savez_compressed(source, **arrays)

    target = migrate_npz_to_hdf5(
        source,
        schema="unit-migrated-large-cache-v1",
        metadata={"package_role": "legacy-greit-migration"},
    )

    assert source.exists()
    assert target == tmp_path / "legacy_components.h5"
    artifact = read_hdf5_artifact(target, lazy=True)
    assert artifact.metadata["migrated_from"] == str(source)
    assert artifact.metadata["legacy_format"] == "npz"
    assert artifact.metadata["legacy_source_read_only"] is True
    assert artifact.metadata["package_role"] == "legacy-greit-migration"
    np.testing.assert_allclose(np.asarray(artifact.arrays["Y"]), arrays["Y"])
