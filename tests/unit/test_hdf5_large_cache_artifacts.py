from __future__ import annotations

import hashlib
import inspect
import json

import h5py
import numpy as np
import pytest

import pyeidors.io.hdf5_artifacts as hdf5_mod
from pyeidors.io.hdf5_artifacts import (
    HDF5LazyDataset,
    _array_digest,
    _dataset_array_digest,
    migrate_npz_to_hdf5,
    read_hdf5_artifact,
    verify_hdf5_artifact_checksums,
    write_hdf5_artifact,
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


def _legacy_array_digest(value: object) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    if arr.dtype.kind in {"S", "U", "O"}:
        flat = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in arr.reshape(-1)
        ]
        arr = np.asarray(flat, dtype=np.str_).reshape(arr.shape)
    encoded = (
        str(arr.dtype).encode()
        + b"|"
        + json.dumps([int(v) for v in arr.shape]).encode()
        + b"|"
        + arr.tobytes()
    )
    return hashlib.sha256(encoded).hexdigest()


def test_hdf5_array_digest_streams_payload_without_tobytes_copy() -> None:
    numeric = np.arange(24, dtype=np.float64).reshape(6, 4)[:, ::2]
    strings = np.asarray([["alpha", "beta"], ["gamma", "delta"]], dtype=np.str_)

    assert _array_digest(numeric) == _legacy_array_digest(numeric)
    assert _array_digest(strings) == _legacy_array_digest(strings)
    source = inspect.getsource(_array_digest)
    source += inspect.getsource(hdf5_mod._update_digest_with_array_bytes)
    assert "memoryview" in source
    assert ".tobytes(" not in source
    assert "np.ascontiguousarray(np.asarray(value))" not in source


def test_v570_hdf5_array_digest_chunks_noncontiguous_numeric_views(
    monkeypatch,
) -> None:
    base = np.arange(1024 * 64, dtype=np.float32).reshape(1024, 64)
    view = base[::2, ::2]
    expected = _legacy_array_digest(view)
    full_contiguous_nbytes = np.ascontiguousarray(view).nbytes
    copied_nbytes: list[int] = []
    real_ascontiguousarray = hdf5_mod.np.ascontiguousarray

    def _tracking_ascontiguousarray(value, *args, **kwargs):
        copied_nbytes.append(int(np.asarray(value).nbytes))
        return real_ascontiguousarray(value, *args, **kwargs)

    monkeypatch.setattr(hdf5_mod, "_DATASET_DIGEST_CHUNK_BYTES", 2048)
    monkeypatch.setattr(
        hdf5_mod.np,
        "ascontiguousarray",
        _tracking_ascontiguousarray,
    )

    assert _array_digest(view) == expected
    assert copied_nbytes
    assert max(copied_nbytes) < full_contiguous_nbytes


def test_v564_verify_hdf5_checksums_streams_numeric_datasets(
    tmp_path,
    monkeypatch,
) -> None:
    arrays = {"RM": np.arange(96, dtype=np.float64).reshape(12, 8)}
    path = write_large_cache_hdf5_artifact(
        tmp_path / "streaming_verify.h5",
        arrays,
        {"role": "unit"},
        schema="unit-streaming-verify-v1",
    )

    with h5py.File(path, "r") as handle:
        dataset = handle["arrays"]["RM"]
        assert _dataset_array_digest(dataset) == _array_digest(arrays["RM"])

    real_asarray = hdf5_mod.np.asarray

    def _guard_asarray(value, *args, **kwargs):
        if isinstance(value, h5py.Dataset):
            raise AssertionError("verify should not materialize full HDF5 datasets")
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(hdf5_mod.np, "asarray", _guard_asarray)

    digests = verify_hdf5_artifact_checksums(path)

    assert set(digests) == {"RM"}
    source = inspect.getsource(verify_hdf5_artifact_checksums)
    source += inspect.getsource(_dataset_array_digest)
    assert "np.asarray(dataset)" not in source
    assert "dataset[start:stop]" in source


def test_large_cache_artifacts_are_written_swmr_ready(tmp_path) -> None:
    arrays = {"RM": np.arange(12, dtype=np.float32).reshape(3, 4)}

    path = write_large_cache_hdf5_artifact(
        tmp_path / "swmr_ready.h5",
        arrays,
        {"role": "unit"},
        schema="unit-swmr-ready-v1",
    )

    with h5py.File(path, "r", libver="latest", swmr=True) as handle:
        assert bool(handle.attrs["swmr_ready"]) is True
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        assert metadata["hdf5_swmr_ready"] is True
        assert metadata["hdf5_libver"] == "latest"

    artifact = read_hdf5_artifact(path, lazy=True, swmr=True)
    assert isinstance(artifact.arrays["RM"], HDF5LazyDataset)
    assert artifact.arrays["RM"].info.swmr is True
    np.testing.assert_array_equal(artifact.arrays["RM"].read(), arrays["RM"])


def test_hdf5_writer_reuses_manifest_digests_for_dataset_attrs(
    tmp_path,
    monkeypatch,
) -> None:
    import pyeidors.io.hdf5_artifacts as hdf5_mod

    arrays = {
        "RM": np.arange(12, dtype=np.float64).reshape(3, 4),
        "weights": np.linspace(1.0, 2.0, 4),
    }
    real_digest = hdf5_mod._array_digest
    calls: list[str] = []

    def _counting_digest(value: object) -> str:
        arr = np.asarray(value)
        calls.append(str(arr.shape))
        return real_digest(value)

    monkeypatch.setattr(hdf5_mod, "_array_digest", _counting_digest)

    path = write_hdf5_artifact(
        tmp_path / "single_digest_pass.h5",
        arrays,
        {"role": "unit"},
        schema="unit-single-digest-pass-v1",
    )

    assert len(calls) == len(arrays)
    artifact = read_hdf5_artifact(path, lazy=True)
    for name, values in arrays.items():
        expected = real_digest(values)
        assert (
            artifact.metadata["artifact_manifest"]["key_payload"]["arrays"][name][
                "sha256"
            ]
            == expected
        )
        with h5py.File(path, "r") as handle:
            assert handle["arrays"][name].attrs["sha256"] == expected


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


def test_v565_legacy_hdf5_manifest_fallback_streams_dataset_digest(
    tmp_path,
    monkeypatch,
) -> None:
    arrays = {"RM": np.arange(120, dtype=np.float64).reshape(15, 8)}
    path = write_large_cache_hdf5_artifact(
        tmp_path / "legacy_missing_digest.h5",
        arrays,
        {"package_role": "legacy-fallback"},
        schema="unit-legacy-manifest-streaming-v1",
    )
    with h5py.File(path, "a") as handle:
        metadata = json.loads(str(handle.attrs["metadata_json"]))
        metadata.pop("artifact_key", None)
        metadata.pop("artifact_manifest", None)
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        del handle["arrays"]["RM"].attrs["sha256"]

    real_asarray = hdf5_mod.np.asarray

    def _guard_asarray(value, *args, **kwargs):
        if isinstance(value, h5py.Dataset):
            raise AssertionError("manifest fallback should stream HDF5 datasets")
        return real_asarray(value, *args, **kwargs)

    monkeypatch.setattr(hdf5_mod.np, "asarray", _guard_asarray)

    artifact = read_hdf5_artifact(path, lazy=True, verify_checksums=False)

    digest = artifact.metadata["artifact_manifest"]["key_payload"]["arrays"]["RM"][
        "sha256"
    ]
    assert digest == _array_digest(arrays["RM"])
    source = inspect.getsource(hdf5_mod._hdf5_manifest_array_payload_from_group)
    assert "np.asarray(dataset)" not in source
    assert "_dataset_array_digest(dataset)" in source


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
