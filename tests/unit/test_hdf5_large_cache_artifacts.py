from __future__ import annotations

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
    rm = lazy_artifact.arrays["RM"]
    assert isinstance(rm, HDF5LazyDataset)
    assert rm.shape == arrays["RM"].shape
    assert rm.compression == "gzip"
    assert rm.chunks is not None
    np.testing.assert_allclose(rm[0, :3], arrays["RM"][0, :3])
    np.testing.assert_allclose(np.asarray(lazy_artifact.arrays["PJt"]), arrays["PJt"])


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
