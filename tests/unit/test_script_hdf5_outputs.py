from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyeidors.io.hdf5_artifacts import read_hdf5_artifact
from scripts.common.hdf5_outputs import (
    GALLERY_ARRAYS_SCHEMA,
    hdf5_output_path,
    read_output_bundle,
    write_output_bundle,
)


def test_script_output_bundle_writes_hdf5_with_metadata(tmp_path: Path) -> None:
    path = write_output_bundle(
        tmp_path / "outputs.h5",
        {
            "delta_sigma": np.array([0.1, -0.2]),
            "rmse_abs": 1.25,
        },
        {"package_role": "difference_reconstruction_outputs"},
        schema=GALLERY_ARRAYS_SCHEMA,
    )

    assert path == tmp_path / "outputs.h5"
    assert not (tmp_path / "outputs.npz").exists()

    artifact = read_hdf5_artifact(path)
    assert artifact.schema == GALLERY_ARRAYS_SCHEMA
    assert artifact.metadata["artifact_format"] == "hdf5"
    assert artifact.metadata["package_role"] == "difference_reconstruction_outputs"
    np.testing.assert_allclose(artifact.arrays["delta_sigma"], [0.1, -0.2])
    assert float(artifact.arrays["rmse_abs"]) == pytest.approx(1.25)


def test_script_output_bundle_writes_string_arrays(tmp_path: Path) -> None:
    path = write_output_bundle(
        tmp_path / "metrics.h5",
        {"metric_names": np.array(["rmse", "pearson"])},
        {"package_role": "metric_names"},
    )

    payload = read_output_bundle(path)

    assert payload["metric_names"].astype(str).tolist() == ["rmse", "pearson"]


def test_script_output_bundle_reader_keeps_legacy_npz_read_only(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy_outputs.npz"
    np.savez(legacy, values=np.array([1, 2, 3]))

    payload = read_output_bundle(legacy)

    np.testing.assert_array_equal(payload["values"], [1, 2, 3])
    assert legacy.exists()
    assert not (tmp_path / "legacy_outputs.h5").exists()


def test_hdf5_output_path_rejects_new_numpy_archive_suffix(tmp_path: Path) -> None:
    assert hdf5_output_path(tmp_path / "result_arrays.h5").suffix == ".h5"
    assert hdf5_output_path(tmp_path / "result_arrays.hdf5").suffix == ".hdf5"
    with pytest.raises(ValueError, match="HDF5 output bundle"):
        hdf5_output_path(tmp_path / "result_arrays.npz")
