"""Tests for 3D GREIT reconstruction-matrix helpers."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.data.channels import bad_channel_mask
from pyeidors.io.hdf5_artifacts import read_hdf5_artifact
from pyeidors.inverse import (
    GREIT3DDistribution,
    GREIT_CACHE_SIGNATURE_SCHEMA,
    GREITDesiredImages,
    GREIT_EIDORS_HDF5_SCHEMA,
    GREITFiniteTargetResponses,
    GREIT_METRIC_KEYS,
    GREITRM,
    VoxelGrid,
    build_3d_greit_rm,
    build_greit_desired_images,
    build_greit_finite_target_responses,
    build_greit_rm_from_eidors_components,
    build_greit3d_distribution,
    generate_spherical_targets,
    greit_metrics,
    load_greit_rm,
    migrate_greit_rm_to_hdf5,
    write_greit_metrics_artifact,
)


class _FiniteTargetForwardModel:
    def __init__(self, centers: np.ndarray, measurement_matrix: np.ndarray) -> None:
        self.centers = np.asarray(centers, dtype=float)
        self.measurement_matrix = np.asarray(measurement_matrix, dtype=float)
        self.solve_calls = 0

    def cell_centers(self) -> np.ndarray:
        return self.centers

    def fwd_solve(self, image):
        self.solve_calls += 1
        meas = self.measurement_matrix @ np.asarray(image.elem_data, dtype=float)
        return SimpleNamespace(meas=meas), None


class _BatchFiniteTargetForwardModel(_FiniteTargetForwardModel):
    def __init__(self, centers: np.ndarray, measurement_matrix: np.ndarray) -> None:
        super().__init__(centers, measurement_matrix)
        self.batch_calls = 0

    def fwd_solve_batch(self, images):
        self.batch_calls += 1
        return [self.fwd_solve(image)[0] for image in images]


def test_greit3d_distribution_explicit_planes_use_eidors_x_fastest_order() -> None:
    distribution = build_greit3d_distribution(
        xvec=[0.0, 1.0, 2.0],
        yvec=[10.0, 20.0, 30.0],
        zvec=[100.0, 200.0],
    )

    assert isinstance(distribution, GREIT3DDistribution)
    np.testing.assert_allclose(
        distribution.centers,
        np.array(
            [
                [0.5, 15.0, 150.0],
                [1.5, 15.0, 150.0],
                [0.5, 25.0, 150.0],
                [1.5, 25.0, 150.0],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(distribution.distr, distribution.centers.T)
    assert distribution.volume_mask.shape == (2, 2, 1)
    assert distribution.metadata["parameter_order"] == "eidors_ndgrid_x_fastest"
    assert distribution.metadata["n_targets"] == 4


def test_greit3d_distribution_imgsz_downsample_and_point_in_volume() -> None:
    distribution = build_greit3d_distribution(
        imgsz=[4, 4, 4],
        bounds=[[-1.0, -1.0, 0.0], [1.0, 1.0, 2.0]],
        downsample=[2, 1],
        point_in_volume=lambda xyz: np.linalg.norm(xyz[:, :2], axis=1) <= 0.8,
    )

    assert distribution.metadata["downsample_factors"] == (2, 2, 2)
    assert distribution.metadata["downsample_phases"] == (1, 1, 1)
    np.testing.assert_allclose(distribution.x_pts, [-0.25, 0.75])
    np.testing.assert_allclose(distribution.y_pts, [-0.25, 0.75])
    np.testing.assert_allclose(distribution.z_pts, [0.75, 1.75])
    assert distribution.volume_mask.shape == (2, 2, 2)
    assert distribution.inside_mask.shape == (8,)
    assert 0 < distribution.num_cells() < 8
    assert np.all(np.linalg.norm(distribution.centers[:, :2], axis=1) <= 0.8)


def test_greit3d_distribution_rejects_2d_raster_parity_options() -> None:
    with pytest.raises(TypeError):
        build_greit3d_distribution(xg=[0.0, 1.0], yg=[0.0, 1.0])  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="xvec or imgsz"):
        build_greit3d_distribution(yvec=[0.0, 1.0], zvec=[0.0, 1.0])


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


def test_build_greit_finite_target_responses_ratio_mode_and_contract() -> None:
    centers = np.array(
        [
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 0.5],
        ],
        dtype=float,
    )
    measurement_matrix = np.array(
        [
            [1.0, 2.0, 0.5, 0.25],
            [0.5, 1.0, 1.5, 2.0],
            [2.0, 0.0, 1.0, 0.5],
        ],
        dtype=float,
    )
    model = _FiniteTargetForwardModel(centers, measurement_matrix)
    channel_mask = bad_channel_mask(3, bad_channels=[1])

    responses = build_greit_finite_target_responses(
        model,
        centers=[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        target_radius=0.2,
        target_plane="z",
        target_offset=0.5,
        target_contrast=[0.5, 1.0],
        normalize=True,
        channel_mask=channel_mask,
        measurement_weights=np.array([4.0, 9.0, 16.0]),
    )

    assert isinstance(responses, GREITFiniteTargetResponses)
    expected_conductivities = np.array(
        [
            [1.5, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    expected_vh = measurement_matrix @ np.ones(4, dtype=float)
    expected_vi = np.column_stack(
        [measurement_matrix @ row for row in expected_conductivities]
    )
    expected_y = expected_vi / expected_vh.reshape(-1, 1) - 1.0
    expected_contracted = expected_y.copy()
    expected_contracted[0, :] *= 2.0
    expected_contracted[1, :] = 0.0
    expected_contracted[2, :] *= 4.0

    np.testing.assert_allclose(responses.vh, expected_vh)
    np.testing.assert_allclose(responses.vi, expected_vi)
    np.testing.assert_allclose(responses.y, expected_y)
    np.testing.assert_allclose(responses.contracted_y, expected_contracted)
    np.testing.assert_allclose(responses.conductivities, expected_conductivities)
    np.testing.assert_allclose(
        responses.xyzr,
        np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.2, 0.2],
            ],
            dtype=float,
        ),
    )
    assert responses.n_measurements == 3
    assert responses.n_targets == 2
    assert responses.metadata["training_mode"] == "forward"
    assert responses.metadata["eidors_parity"] is True
    assert responses.metadata["difference_normalization"] == "ratio"
    assert responses.metadata["target_plane"] == "z"
    assert responses.metadata["target_offset"] == 0.5
    assert responses.metadata["bad_channel_count"] == 1
    assert responses.metadata["measurement_weight_kind"] == "diagonal"
    assert model.solve_calls == 3


def test_build_greit_finite_target_responses_raw_batch_and_cache() -> None:
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    measurement_matrix = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.5, 0.5],
        ],
        dtype=float,
    )
    model = _BatchFiniteTargetForwardModel(centers, measurement_matrix)
    cache: dict[str, GREITFiniteTargetResponses] = {}

    responses = build_greit_finite_target_responses(
        model,
        centers=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        target_size=0.25,
        target_offset=[0.0, 0.0, 0.0],
        target_contrast=0.25,
        normalize=False,
        batch_size=1,
        response_cache=cache,
        cache_key="unit-forward-responses",
    )
    cached = build_greit_finite_target_responses(
        model,
        centers=[[99.0, 99.0, 99.0]],
        target_size=1.0,
        normalize=True,
        response_cache=cache,
        cache_key="unit-forward-responses",
    )

    expected_vh = measurement_matrix @ np.ones(3, dtype=float)
    expected_vi = np.column_stack(
        [
            measurement_matrix @ np.array([1.25, 1.0, 1.0]),
            measurement_matrix @ np.array([1.0, 1.0, 1.25]),
        ]
    )
    np.testing.assert_allclose(responses.vh, expected_vh)
    np.testing.assert_allclose(responses.vi, expected_vi)
    np.testing.assert_allclose(responses.y, expected_vi - expected_vh.reshape(-1, 1))
    np.testing.assert_allclose(responses.contracted_y, responses.y)
    assert responses.metadata["difference_normalization"] == "raw"
    assert responses.metadata["batch_size"] == 1
    assert responses.metadata["cache_key"] == "unit-forward-responses"
    assert responses.metadata["cache_hit"] is False
    assert model.batch_calls == 2
    assert cached.metadata["cache_hit"] is True
    np.testing.assert_allclose(cached.y, responses.y)


def test_build_greit_desired_images_default_sigmoid_is_independent_from_targets() -> (
    None
):
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        shape=(3, 3, 3),
    )
    center = np.array([[1.5, 1.5, 1.5]], dtype=float)
    raw_targets = generate_spherical_targets(
        grid,
        centers=center,
        radius=0.51,
        kind="sphere",
    )

    desired = build_greit_desired_images(
        grid,
        xyz=center,
        radius=0.8,
        target_values=raw_targets,
    )

    assert isinstance(desired, GREITDesiredImages)
    assert desired.shape == (grid.num_cells(), 1)
    assert desired.metadata["builder"] == "GREIT_desired_img"
    assert desired.metadata["desired_solution_fn"] == "GREIT_desired_img_sigmoid"
    assert desired.metadata["eidors_component_parity"] is True
    assert desired.metadata["target_values_used"] is False
    assert desired.metadata["target_values_requires_explicit_opt_in"] is True
    assert desired.metadata["d_shape"] == (grid.num_cells(), 1)

    raw_d = raw_targets.values.T
    assert not np.allclose(desired.values, raw_d)
    distances = np.linalg.norm(grid.cell_centers() - center, axis=1)
    center_idx = int(np.argmin(distances))
    neighbor_idx = int(np.argsort(distances)[1])
    far_idx = int(np.argmax(distances))
    assert desired.values[center_idx, 0] > 0.99
    assert 0.0 < desired.values[neighbor_idx, 0] < desired.values[center_idx, 0]
    assert desired.values[far_idx, 0] == pytest.approx(0.0)


def test_build_greit_desired_images_uses_custom_solution_fn_signature() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        shape=(2, 2, 2),
    )
    xyz = np.array(
        [
            [0.5, 1.5],
            [0.5, 1.5],
            [0.5, 1.5],
        ],
        dtype=float,
    )

    def custom_desired(xyz_arg, radius_arg, options):
        assert xyz_arg.shape == (3, 2)
        np.testing.assert_allclose(radius_arg, [0.25, 0.5])
        assert options["n_rec_parameters"] == grid.num_cells()
        centers = np.asarray(options["rec_centers"], dtype=float)
        return centers[:, [0]] + xyz_arg[[0], :]

    desired = build_greit_desired_images(
        grid,
        xyz=xyz,
        radius=[0.25, 0.5],
        desired_solution_fn=custom_desired,
    )

    expected = grid.cell_centers()[:, [0]] + xyz[[0], :]
    np.testing.assert_allclose(desired.values, expected)
    assert desired.shape == (grid.num_cells(), 2)
    assert desired.metadata["desired_solution_fn"] == "custom_desired"
    assert desired.metadata["eidors_component_parity"] is False
    assert desired.metadata["target_values_used"] is False


def test_build_greit_desired_images_target_values_requires_explicit_opt_in() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        shape=(2, 2, 2),
    )
    targets = generate_spherical_targets(
        grid,
        centers=[[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
        radius=0.2,
        kind="sphere",
    )

    with pytest.raises(ValueError, match="explicit target_values"):
        build_greit_desired_images(
            grid,
            xyz=targets.centers,
            radius=targets.radii,
            desired_solution_fn="target_values",
        )

    desired = build_greit_desired_images(
        grid,
        xyz=targets.centers,
        radius=targets.radii,
        desired_solution_fn="target_values",
        target_values=targets,
    )

    np.testing.assert_allclose(desired.values, targets.values.T)
    assert desired.metadata["desired_solution_fn"] == "target_values_explicit_opt_in"
    assert desired.metadata["eidors_component_parity"] is False
    assert desired.metadata["target_values_used"] is True


def test_eidors_greit_hdf5_artifact_stores_model_components_and_signature(
    tmp_path,
) -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 1.0],
        shape=(2, 2, 1),
    )
    measurement_matrix = np.array(
        [
            [1.0, 0.5, 0.25, 0.0],
            [0.0, 1.0, -0.5, 0.75],
            [0.25, -0.25, 1.0, 0.5],
        ],
        dtype=float,
    )
    model = _FiniteTargetForwardModel(grid.cell_centers(), measurement_matrix)
    responses = build_greit_finite_target_responses(
        model,
        centers=grid.cell_centers()[:2],
        target_radius=0.4,
        target_contrast=0.35,
        normalize=True,
    )
    desired = build_greit_desired_images(grid, responses=responses)
    artifact_path = tmp_path / "eidors_greit_components.h5"

    greit = build_greit_rm_from_eidors_components(
        responses,
        desired,
        weight=0.25,
        noise_covar=1.5,
        artifact_path=artifact_path,
        fwd_model_signature="unit-fwd-signature",
        keep_model_components=True,
    )

    expected_pjt = desired.values @ responses.contracted_y.T
    artifact = read_hdf5_artifact(artifact_path)
    assert artifact.schema == GREIT_EIDORS_HDF5_SCHEMA
    assert artifact.metadata["artifact_schema"] == GREIT_EIDORS_HDF5_SCHEMA
    assert artifact.metadata["cache_signature_schema"] == GREIT_CACHE_SIGNATURE_SCHEMA
    assert artifact.metadata["cache_signature_hash"] == greit.cache_signature
    assert artifact.metadata["cache_signature_payload"]["schema"] == (
        GREIT_CACHE_SIGNATURE_SCHEMA
    )
    assert artifact.metadata["cache_signature_payload"]["training_mode"] == "forward"
    assert artifact.metadata["cache_signature_payload"]["fwd_model_signature"] == (
        "unit-fwd-signature"
    )
    for name in (
        "RM",
        "PJt",
        "M",
        "Sn",
        "noiselev",
        "weight",
        "vh",
        "vi",
        "xyzr",
        "D",
        "Y",
        "rec_model",
        "fwd_model_signature",
    ):
        assert name in artifact.arrays
    np.testing.assert_allclose(artifact.arrays["PJt"], expected_pjt)
    np.testing.assert_allclose(artifact.arrays["D"], desired.values)
    np.testing.assert_allclose(artifact.arrays["Y"], responses.contracted_y)
    assert artifact.arrays["noiselev"].shape == (1,)
    assert artifact.arrays["weight"].shape == (1,)

    loaded = load_greit_rm(artifact_path)
    np.testing.assert_allclose(loaded.rm, greit.rm)
    np.testing.assert_allclose(loaded.pjt, expected_pjt)
    np.testing.assert_allclose(loaded.m, greit.m)
    np.testing.assert_allclose(loaded.y, responses.contracted_y)
    np.testing.assert_allclose(loaded.d, desired.values)
    np.testing.assert_allclose(loaded.vh, responses.vh)
    np.testing.assert_allclose(loaded.vi, responses.vi)
    np.testing.assert_allclose(loaded.xyzr, responses.xyzr)
    np.testing.assert_allclose(loaded.rec_model, grid.cell_centers())
    assert loaded.fwd_model_signature == "unit-fwd-signature"
    assert loaded.cache_signature == greit.cache_signature


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
    assert greit.metadata["eidors_parity"] is False
    assert greit.metadata["training_mode"] == "linearized"
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
