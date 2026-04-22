from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eit_app.i18n import set_language, t
from eit_app.io.hdf5_packages import (
    DATASET_MESH_SCHEMA,
    DATASET_SAMPLE_SCHEMA,
    SIMULATION_RESULTS_SCHEMA,
    normalize_hdf5_package_path,
    write_dataset_mesh_info_package,
    write_dataset_sample_package,
    write_simulation_results_package,
)
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact


def test_dataset_generator_writes_mesh_and_sample_hdf5_packages(tmp_path: Path) -> None:
    mesh_path = write_dataset_mesh_info_package(
        tmp_path,
        node_coords=np.array([[0.0, 0.0], [1.0, 0.0]]),
        cell_connectivity=np.array([[0, 1]], dtype=np.int32),
        n_electrodes=16,
        homogeneous_voltages=np.array([1.0, 2.0, 3.0]),
        forward_model_config={"mesh_dimension": 2, "n_elec": 16},
        total_electrodes=16,
    )
    sample_path = write_dataset_sample_package(
        tmp_path,
        7,
        ground_truth=np.array([0.9, 1.1]),
        boundary_voltages=np.array([1.2, 2.3]),
        background_conductivity=1.0,
        n_inhomogeneities=2,
    )

    assert mesh_path == tmp_path / "mesh_info.h5"
    assert sample_path == tmp_path / "sample_000007.h5"
    assert not (tmp_path / "mesh_info.npz").exists()
    assert not (tmp_path / "sample_000007.npz").exists()

    mesh = read_hdf5_artifact(mesh_path)
    assert mesh.schema == DATASET_MESH_SCHEMA
    assert mesh.metadata["artifact_format"] == "hdf5"
    assert mesh.metadata["forward_model_config"]["mesh_dimension"] == 2
    np.testing.assert_allclose(mesh.arrays["homogeneous_voltages"], [1.0, 2.0, 3.0])
    assert int(mesh.arrays["total_electrodes"]) == 16

    sample = read_hdf5_artifact(sample_path)
    assert sample.schema == DATASET_SAMPLE_SCHEMA
    assert sample.metadata["sample_index"] == 7
    np.testing.assert_allclose(sample.arrays["ground_truth"], [0.9, 1.1])
    assert int(sample.arrays["n_inhomogeneities"]) == 2


def test_simulation_results_export_writes_hdf5_and_normalizes_suffix(
    tmp_path: Path,
) -> None:
    path = write_simulation_results_package(
        tmp_path / "sim_result",
        ground_truth=np.array([1.0, 0.8]),
        boundary_voltages=np.array([0.1, 0.2]),
        homogeneous_voltages=np.array([0.0, 0.0]),
        node_coords=np.array([[0.0, 0.0, 0.0]]),
        cell_connectivity=np.array([[0]], dtype=np.int32),
    )

    assert path == tmp_path / "sim_result.h5"
    artifact = read_hdf5_artifact(path)
    assert artifact.schema == SIMULATION_RESULTS_SCHEMA
    assert artifact.metadata["package_role"] == "simulation_results"
    np.testing.assert_allclose(artifact.arrays["boundary_voltages"], [0.1, 0.2])


def test_hdf5_package_path_rejects_numpy_archive_suffix(tmp_path: Path) -> None:
    assert normalize_hdf5_package_path(tmp_path / "export").suffix == ".h5"
    assert normalize_hdf5_package_path(tmp_path / "export.hdf5").suffix == ".hdf5"
    with pytest.raises(ValueError, match="HDF5 package path"):
        normalize_hdf5_package_path(tmp_path / "legacy.npz")


def test_dataset_and_simulation_i18n_labels_are_hdf5() -> None:
    try:
        set_language("en", persist=False)
        assert "mesh_info.h5" in t("dataset.artifacts.item1")
        assert "sample_000000.h5" in t("dataset.artifacts.item2")
        assert ".npz" not in t("dataset.artifacts.item1")
        assert "HDF5 package" in t("sim.results.save_dialog_filter")

        set_language("zh", persist=False)
        assert "mesh_info.h5" in t("dataset.artifacts.item1")
        assert "sample_000000.h5" in t("dataset.artifacts.item2")
        assert ".npz" not in t("dataset.artifacts.item2")
        assert "HDF5" in t("sim.results.save_dialog_filter")
    finally:
        set_language("en", persist=False)
