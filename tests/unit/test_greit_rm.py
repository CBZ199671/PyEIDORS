"""Tests for 3D GREIT reconstruction-matrix helpers."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from pyeidors.data.channels import bad_channel_mask
from pyeidors.inverse import (
    GREIT_METRIC_KEYS,
    GREITRM,
    VoxelGrid,
    build_3d_greit_rm,
    generate_spherical_targets,
    greit_metrics,
    load_greit_rm,
    migrate_greit_rm_to_hdf5,
    write_greit_metrics_artifact,
)


def test_generate_spherical_targets_for_3d_voxel_grid() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        shape=(2, 2, 2),
    )

    spheres = generate_spherical_targets(grid, radius=0.2, kind="sphere")
    blobs = generate_spherical_targets(
        grid,
        centers=[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
        radius=0.75,
        kind="blob",
        amplitude=2.0,
    )

    assert spheres.shape == (8, 8)
    assert spheres.masks.shape == (8, 8)
    assert np.all(np.count_nonzero(spheres.masks, axis=1) == 1)
    assert blobs.shape == (2, 8)
    assert np.all(blobs.values >= 0.0)
    assert blobs.metadata["kind"] == "blob"
    assert blobs.metadata["voxel_shape"] == (2, 2, 2)


def test_build_3d_greit_rm_reconstructs_training_sphere_and_saves_artifact(
    tmp_path,
) -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        shape=(2, 2, 2),
    )
    n_param = grid.num_cells()
    jacobian = np.vstack(
        [
            np.eye(n_param, dtype=float),
            np.array([[3.0, -2.0, 1.0, -4.0, 2.5, 0.5, -1.5, 6.0]]),
        ]
    )
    mask = bad_channel_mask(jacobian.shape[0], bad_channels=[8])
    weights = np.array([2.0, 1.0, 0.75, 1.5, 3.0, 0.5, 2.5, 1.25, 1e6])
    artifact_path = tmp_path / "greit_rm.h5"

    greit = build_3d_greit_rm(
        jacobian=jacobian,
        inverse_mesh=grid,
        target_radius=0.2,
        noise_figure=1e-8,
        channel_mask=mask,
        measurement_weights=weights,
        artifact_path=artifact_path,
    )

    assert isinstance(greit, GREITRM)
    assert artifact_path.exists()
    assert greit.shape == (n_param, jacobian.shape[0])
    assert greit.metadata["algorithm"] == "greit-3d"
    assert greit.metadata["synthetic_target_count"] == 8
    assert greit.metadata["online_hot_path"] == "rm_matmul"
    assert greit.metadata["artifact_schema"] == "pyeidors-greit-rm-hdf5-v1"
    assert greit.metadata["artifact_format"] == "hdf5"
    assert greit.metadata["bad_channel_count"] == 1

    target = np.asarray(greit.training_targets[3], dtype=float)
    reference = np.linspace(2.0, 4.0, jacobian.shape[0])
    normalized = jacobian @ target
    normalized[8] = 99.0
    frame = reference * (1.0 + normalized)

    reconstruction = greit.reconstruct(
        frame,
        normalize=True,
        v_ref=reference,
        device="cpu",
        return_metadata=True,
    )

    np.testing.assert_allclose(
        reconstruction.values,
        target.reshape(grid.shape),
        atol=2e-8,
    )
    assert reconstruction.metadata["algorithm"] == "greit-3d"
    assert reconstruction.metadata["online_hot_path"] == "rm_matmul"
    assert reconstruction.metadata["voxel_shape"] == grid.shape
    prepared = greit.prepare_online(
        device="cpu",
        dtype="float32",
        cache_key="unit-greit-rm",
    )
    prepared_reconstruction = prepared.reconstruct(
        frame,
        normalize=True,
        v_ref=reference,
        device="cpu",
        dtype="float32",
        return_metadata=True,
    )
    np.testing.assert_allclose(
        prepared_reconstruction.values,
        target.reshape(grid.shape),
        atol=1e-6,
    )
    assert prepared_reconstruction.metadata["rm_prepare_mode"] == "reused_handle"
    assert prepared_reconstruction.metadata["rm_dtype"] == "float32"
    assert prepared_reconstruction.metadata["rm_cache_key"] == "unit-greit-rm"
    metrics = greit_metrics(
        reconstruction.values,
        np.asarray(greit.training_targets[3], dtype=bool),
        centers=grid.cell_centers(),
    )
    assert set(metrics) == set(GREIT_METRIC_KEYS)
    assert metrics["AR"] == pytest.approx(1.0, abs=2e-8)
    assert metrics["PE"] == pytest.approx(0.0, abs=2e-8)
    assert metrics["RNG"] == pytest.approx(0.0, abs=2e-8)

    loaded = load_greit_rm(artifact_path)
    np.testing.assert_allclose(loaded.rm, greit.rm)
    np.testing.assert_allclose(loaded.training_targets, greit.training_targets)
    np.testing.assert_allclose(loaded.training_responses, greit.training_responses)
    np.testing.assert_allclose(
        loaded.reconstruct(frame, normalize=True, v_ref=reference, device="cpu"),
        target.reshape(grid.shape),
        atol=2e-8,
    )


def test_load_greit_rm_reads_legacy_npz_and_migrates_to_hdf5(tmp_path) -> None:
    legacy_path = tmp_path / "legacy_greit_rm.npz"
    rm = np.eye(2, dtype=np.float64)
    np.savez_compressed(
        legacy_path,
        rm=rm,
        metadata_json=np.asarray(json.dumps({"algorithm": "greit-3d"})),
        voxel_shape=np.asarray([2], dtype=np.int64),
        channel_mask=np.asarray([], dtype=bool),
        measurement_weights=np.asarray([], dtype=np.float64),
        training_targets=np.eye(2, dtype=np.float64),
        training_responses=np.eye(2, dtype=np.float64),
    )

    legacy = load_greit_rm(legacy_path)
    assert legacy.metadata["legacy_read_only"] is True
    np.testing.assert_allclose(legacy.rm, rm)

    migrated_path = migrate_greit_rm_to_hdf5(legacy_path)
    migrated = load_greit_rm(migrated_path)
    assert migrated_path.suffix == ".h5"
    assert migrated.metadata["artifact_schema"] == "pyeidors-greit-rm-hdf5-v1"
    assert migrated.metadata["migrated_from"] == str(legacy_path)
    np.testing.assert_allclose(migrated.rm, rm)


def test_build_3d_greit_rm_accepts_explicit_target_matrix() -> None:
    jacobian = np.eye(3, dtype=float)
    targets = np.eye(3, dtype=float)

    greit = build_3d_greit_rm(
        jacobian=jacobian,
        targets=targets,
        noise_figure=1e-8,
    )

    np.testing.assert_allclose(
        greit.reconstruct(targets[1], normalize=False, device="cpu"),
        targets[1],
        atol=1e-8,
    )
    assert greit.metadata["target_kind"] == "provided"


def test_build_3d_greit_rm_validates_inputs() -> None:
    with pytest.raises(ValueError, match="inverse_mesh is required"):
        build_3d_greit_rm(jacobian=np.eye(2), targets=None)
    with pytest.raises(ValueError, match="3D inverse cell centers"):
        generate_spherical_targets(
            VoxelGrid.from_bounds([0.0, 0.0], [1.0, 1.0], shape=(1, 1))
        )
    with pytest.raises(ValueError, match="targets parameter dimension"):
        build_3d_greit_rm(jacobian=np.eye(2), targets=np.ones((1, 3)))


def test_greit_metrics_for_perfect_single_voxel_target() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        shape=(3, 3, 3),
    )
    image = np.zeros(grid.shape, dtype=float)
    image[1, 1, 1] = 1.0
    target_mask = image.astype(bool)

    metrics = greit_metrics(image, target_mask, centers=grid.cell_centers())

    assert list(metrics.keys()) == list(GREIT_METRIC_KEYS)
    assert metrics["AR"] == pytest.approx(1.0)
    assert metrics["PE"] == pytest.approx(0.0)
    assert metrics["RES"] == pytest.approx((1.0 / 27.0) ** (1.0 / 3.0))
    assert metrics["SD"] == pytest.approx(0.0)
    assert metrics["RNG"] == pytest.approx(0.0)


def test_greit_metrics_detects_ringing_outside_quarter_max_set() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        shape=(3, 3, 3),
    )
    image = np.full(grid.shape, -0.1, dtype=float)
    image[1, 1, 1] = 1.0
    target_mask = np.zeros(grid.shape, dtype=bool)
    target_mask[1, 1, 1] = True

    metrics = greit_metrics(image, target_mask, centers=grid.cell_centers())

    assert metrics["RNG"] > 2.0
    assert metrics["SD"] == pytest.approx(0.0)


def test_write_greit_metrics_artifact_json_and_csv(tmp_path) -> None:
    metrics = {
        "AR": 1.0,
        "PE": 0.0,
        "RES": 0.25,
        "SD": 0.0,
        "RNG": 0.0,
        "case": "perfect",
    }
    json_path = write_greit_metrics_artifact(
        metrics,
        tmp_path / "metrics.json",
        metadata={"suite": "unit"},
    )
    csv_path = write_greit_metrics_artifact([metrics], tmp_path / "metrics.csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "pyeidors-greit-metrics-v1"
    assert payload["metric_keys"] == list(GREIT_METRIC_KEYS)
    assert payload["metadata"] == {"suite": "unit"}
    assert payload["records"][0]["case"] == "perfect"

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["case"] == "perfect"
    for key in GREIT_METRIC_KEYS:
        assert key in rows[0]


def test_write_greit_metrics_artifact_requires_full_metric_set(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing keys"):
        write_greit_metrics_artifact(
            {"AR": 1.0, "PE": 0.0, "RES": 0.1, "SD": 0.0},
            tmp_path / "bad.json",
        )
    with pytest.raises(ValueError, match="must end with"):
        write_greit_metrics_artifact(
            {key: 0.0 for key in GREIT_METRIC_KEYS},
            tmp_path / "bad.txt",
        )
