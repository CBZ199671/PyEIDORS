"""Tests for 3D GREIT reconstruction-matrix helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.data.channels import bad_channel_mask
from pyeidors.inverse import (
    GREITRM,
    VoxelGrid,
    build_3d_greit_rm,
    generate_spherical_targets,
    load_greit_rm,
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
    artifact_path = tmp_path / "greit_rm.npz"

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
    assert greit.metadata["artifact_schema"] == "pyeidors-greit-rm-v1"
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

    loaded = load_greit_rm(artifact_path)
    np.testing.assert_allclose(loaded.rm, greit.rm)
    np.testing.assert_allclose(loaded.training_targets, greit.training_targets)
    np.testing.assert_allclose(loaded.training_responses, greit.training_responses)
    np.testing.assert_allclose(
        loaded.reconstruct(frame, normalize=True, v_ref=reference, device="cpu"),
        target.reshape(grid.shape),
        atol=2e-8,
    )


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
