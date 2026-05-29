"""Tests for 3D GREIT reconstruction-matrix helpers."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

import pyeidors.inverse.greit as greit_module
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
    GREITNativeTrainingPipeline,
    VoxelGrid,
    build_3d_greit_rm,
    build_greit_desired_images,
    build_greit_finite_target_responses,
    build_greit_rm_from_eidors_components,
    build_greit3d_distribution,
    build_native_greit_training_pipeline,
    generate_spherical_targets,
    greit_metrics,
    load_greit_rm,
    migrate_greit_rm_to_hdf5,
    write_greit_metrics_artifact,
)


def test_greit_array_digest_streams_payload_without_tobytes_copy() -> None:
    array = np.arange(16, dtype=np.float32).reshape(4, 4)[::2, ::2]
    contiguous = np.ascontiguousarray(array)
    expected = hashlib.sha256(
        str(contiguous.dtype).encode("utf-8")
        + b"|"
        + json.dumps([int(v) for v in contiguous.shape], sort_keys=True).encode("utf-8")
        + b"|"
        + contiguous.tobytes()
    ).hexdigest()

    assert greit_module._array_digest(array) == expected
    source = inspect.getsource(greit_module._array_digest)
    assert "update_digest_with_array_payload" in source
    assert ".tobytes(" not in source
    assert "np.ascontiguousarray(np.asarray(value))" not in source


def test_v452_greit_infer_center_spacing_reuses_unique_order_without_diff_subset() -> (
    None
):
    source = inspect.getsource(greit_module._infer_center_spacing)
    helper_source = inspect.getsource(greit_module._median_positive_adjacent_spacing)

    assert "np.diff" not in source
    assert "np.sort(coords)" not in source
    assert "diffs[diffs >" not in source
    assert "_median_positive_adjacent_spacing(coords)" in source
    assert "np.diff" not in helper_source
    assert "diffs[diffs >" not in helper_source
    assert "overwrite_input=True" in helper_source

    centers = np.array(
        [
            [4.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    spacing = greit_module._infer_center_spacing(centers)

    np.testing.assert_allclose(spacing, np.array([1.0, 0.5, 0.0]))


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


def test_v282_greit_training_response_matrices_direct_fill(monkeypatch) -> None:
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
            [0.5, 1.0, -1.0],
        ],
        dtype=float,
    )
    conductivities = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [0.25, 0.75, 1.25],
        ],
        dtype=float,
    )
    model = _BatchFiniteTargetForwardModel(centers, measurement_matrix)
    expected_vi = measurement_matrix @ conductivities.T

    def _fail_column_stack(*_args, **_kwargs):
        raise AssertionError("GREIT training response matrices must direct-fill")

    monkeypatch.setattr(greit_module.np, "column_stack", _fail_column_stack)

    vi = greit_module._solve_measurement_batch(
        model,
        conductivities,
        batch_size=2,
    )
    np.testing.assert_allclose(vi, expected_vi)
    assert model.batch_calls == 2

    y = np.array(
        [
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=float,
    )
    contracted, contract = greit_module._contract_training_responses(
        y,
        channel_mask=None,
        measurement_weights=None,
    )
    np.testing.assert_allclose(contracted, y)
    assert contract is not None
    assert "np.column_stack" not in inspect.getsource(
        greit_module._solve_measurement_batch
    )
    assert "np.column_stack" not in inspect.getsource(
        greit_module._contract_training_responses
    )


def test_v283_finite_target_conductivities_use_work_vectors() -> None:
    source = inspect.getsource(greit_module._build_finite_target_conductivities)
    assert "np.linalg.norm" not in source
    assert "center.reshape" not in source
    assert "append(" not in source

    fwd_centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    background = np.ones(3, dtype=float)
    target_centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=float,
    )
    actual = greit_module._build_finite_target_conductivities(
        fwd_centers,
        background=background,
        target_centers=target_centers,
        target_radii=np.array([0.25, 0.01], dtype=float),
        target_contrasts=np.array([0.5, -0.25], dtype=float),
    )

    np.testing.assert_allclose(
        actual,
        np.array(
            [
                [1.5, 1.0, 1.0],
                [1.0, 0.75, 1.0],
            ],
            dtype=float,
        ),
    )


def test_v402_finite_target_conductivities_reuse_mask_and_where_add(
    monkeypatch,
) -> None:
    source = inspect.getsource(greit_module._build_finite_target_conductivities)
    assert "dist2 <= radius2" not in source
    assert "sigma[mask]" not in source
    assert "inside_mask = np.empty(fwd_centers.shape[0], dtype=bool)" in source
    assert "np.less_equal(dist2, radius2, out=inside_mask)" in source
    assert "np.add(sigma, float(contrast), out=sigma, where=inside_mask)" in source

    original_less_equal = np.less_equal
    mask_ids: list[int] = []

    def _capture_less_equal(left, right, out=None, **kwargs):
        assert out is not None
        mask_ids.append(id(out))
        return original_less_equal(left, right, out=out, **kwargs)

    monkeypatch.setattr(greit_module.np, "less_equal", _capture_less_equal)
    actual = greit_module._build_finite_target_conductivities(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        background=np.ones(3, dtype=float),
        target_centers=np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=float,
        ),
        target_radii=np.array([0.25, 0.01], dtype=float),
        target_contrasts=np.array([0.5, -0.25], dtype=float),
    )

    assert len(mask_ids) == 2
    assert len(set(mask_ids)) == 1
    np.testing.assert_allclose(
        actual,
        np.array(
            [
                [1.5, 1.0, 1.0],
                [1.0, 0.75, 1.0],
            ],
            dtype=float,
        ),
    )


def test_v454_finite_target_conductivity_positivity_check_uses_reduction() -> None:
    source = inspect.getsource(greit_module._build_finite_target_conductivities)

    assert "np.any(sigma <= 0.0)" not in source
    assert "float(np.min(sigma)) <= 0.0" in source

    with pytest.raises(ValueError, match="finite-target conductivity"):
        greit_module._build_finite_target_conductivities(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            background=np.ones(2, dtype=float),
            target_centers=np.array([[0.0, 0.0, 0.0]], dtype=float),
            target_radii=np.array([0.25], dtype=float),
            target_contrasts=np.array([-1.5], dtype=float),
        )


def test_v455_background_conductivity_positivity_check_uses_reduction() -> None:
    source = inspect.getsource(greit_module._resolve_background_conductivity)

    assert "np.any(array <= 0.0)" not in source
    assert "float(np.min(array)) <= 0.0" in source

    with pytest.raises(ValueError, match="background_conductivity"):
        greit_module._resolve_background_conductivity(
            np.array([1.0, 0.0, 2.0], dtype=float),
            n_cells=3,
        )


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


def test_v216_greit3d_distribution_avoids_meshgrid() -> None:
    source = inspect.getsource(greit_module.build_greit3d_distribution)

    assert "np.meshgrid" not in source
    assert "_cartesian_centers_from_axes" in source


def test_v448_greit3d_distribution_direct_fills_inside_centers() -> None:
    source = inspect.getsource(greit_module.build_greit3d_distribution)
    helper_source = inspect.getsource(greit_module._compact_centers_by_mask)

    assert "candidate_centers[inside_mask]" not in source
    assert "_compact_centers_by_mask(" in source
    assert "candidate_centers," in source
    assert "inside_mask," in source
    assert "out = np.empty(" in helper_source
    assert "out[write_idx] = values[row_idx]" in helper_source

    distribution = build_greit3d_distribution(
        imgsz=[2, 2, 1],
        bounds=[[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]],
        point_in_volume=np.array([True, False, False, True], dtype=bool),
    )

    assert distribution.centers.flags.c_contiguous
    np.testing.assert_allclose(
        distribution.centers,
        np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 0.5]], dtype=np.float64),
    )
    np.testing.assert_allclose(distribution.distr, distribution.centers.T)


def test_v453_greit3d_distribution_reuses_inside_count_for_compaction() -> None:
    source = inspect.getsource(greit_module.build_greit3d_distribution)
    helper_source = inspect.getsource(greit_module._compact_centers_by_mask)

    assert "np.any(inside_mask)" not in source
    assert "inside_count = int(np.count_nonzero(inside_mask))" in source
    assert "count=inside_count" in source
    assert "count: int | None = None" in helper_source

    distribution = build_greit3d_distribution(
        imgsz=[2, 2, 1],
        bounds=[[0.0, 0.0, 0.0], [2.0, 2.0, 1.0]],
        point_in_volume=np.array([True, False, False, True], dtype=bool),
    )

    assert distribution.metadata["n_targets"] == 2
    assert distribution.centers.shape == (2, 3)


def test_v236_cartesian_centers_fill_axes_without_tile_repeat() -> None:
    source = inspect.getsource(greit_module._cartesian_centers_from_axes)

    assert "np.tile" not in source
    assert "np.repeat" not in source
    assert "_fill_repeated_axis_column" in source
    centers_c = greit_module._cartesian_centers_from_axes(
        (np.array([0.0, 1.0]), np.array([10.0, 20.0, 30.0])),
        order="C",
    )
    centers_f = greit_module._cartesian_centers_from_axes(
        (np.array([0.0, 1.0]), np.array([10.0, 20.0, 30.0])),
        order="F",
    )

    np.testing.assert_allclose(
        centers_c,
        np.array(
            [
                [0.0, 10.0],
                [0.0, 20.0],
                [0.0, 30.0],
                [1.0, 10.0],
                [1.0, 20.0],
                [1.0, 30.0],
            ],
            dtype=float,
        ),
    )
    np.testing.assert_allclose(
        centers_f,
        np.array(
            [
                [0.0, 10.0],
                [1.0, 10.0],
                [0.0, 20.0],
                [1.0, 20.0],
                [0.0, 30.0],
                [1.0, 30.0],
            ],
            dtype=float,
        ),
    )


def test_v217_metric_centers_avoid_meshgrid() -> None:
    source = inspect.getsource(greit_module._metric_centers)

    assert "np.meshgrid" not in source
    centers = greit_module._metric_centers(None, (2, 3, 4), n_cells=24)
    assert centers.shape == (24, 3)
    np.testing.assert_allclose(
        centers[:5],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 1.0, 0.0],
        ],
    )
    np.testing.assert_allclose(centers[-1], [1.0, 2.0, 3.0])


def test_v218_gauss_reference_offsets_avoid_meshgrid() -> None:
    source = inspect.getsource(greit_module._gauss_reference_offsets)

    assert "np.meshgrid" not in source
    offsets, weights = greit_module._gauss_reference_offsets(3, 2)

    assert offsets.shape == (8, 3)
    assert weights.shape == (8,)
    assert np.sum(weights) == pytest.approx(1.0)
    np.testing.assert_allclose(offsets[0], [-0.28867513459481287] * 3)
    np.testing.assert_allclose(offsets[-1], [0.28867513459481287] * 3)


def test_v219_default_radius_uses_kdtree_not_all_pairs() -> None:
    source = inspect.getsource(greit_module._nearest_unique_center_distance)
    radius_source = inspect.getsource(greit_module._default_radius)

    assert "cKDTree" in source
    assert "[:, None" not in source
    assert "[None, :" not in source
    assert "positive_nearest" not in source
    assert "np.isfinite(nearest_values) &" not in source
    assert "min_positive_finite_value(nearest_values" in source
    assert "_nearest_unique_center_distance" in radius_source

    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert greit_module._default_radius(centers) == pytest.approx(1.02)


def test_v220_nearest_center_distance_uses_shared_kdtree_helper() -> None:
    source = inspect.getsource(greit_module._nearest_center_distance)

    assert "_nearest_unique_center_distance" in source
    assert "[:, None" not in source
    assert "[None, :" not in source

    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert greit_module._nearest_center_distance(centers) == pytest.approx(2.0)


def test_v221_adaptive_gauss_streams_boundary_distances() -> None:
    source = inspect.getsource(greit_module._greit_sigmoid_adaptive_gauss)

    assert "center_distances" not in source
    assert "xyz_matrix.T[None" not in source
    assert "_distances_to_point" in source


def test_v222_sample_average_streams_target_distances() -> None:
    source = inspect.getsource(greit_module._greit_sigmoid_average_over_samples)

    assert "_distances_to_point" in source
    assert "flat_samples - xyz_matrix" not in source
    distances = greit_module._distances_to_point(
        np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float64),
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
    )
    np.testing.assert_allclose(distances, [0.0, 5.0])


def test_v224_desired_image_hot_path_streams_offsets_not_samples() -> None:
    sigmoid_source = inspect.getsource(greit_module.greit_desired_image_sigmoid)
    adaptive_source = inspect.getsource(greit_module._greit_sigmoid_adaptive_gauss)
    average_source = inspect.getsource(greit_module._greit_sigmoid_average_over_offsets)
    module_source = inspect.getsource(greit_module)

    assert "_desired_cell_offsets" in sigmoid_source
    assert "_desired_cell_samples" not in sigmoid_source
    assert "_desired_cell_samples" not in module_source
    assert "_desired_cell_offsets" in adaptive_source
    assert "_desired_cell_samples" not in adaptive_source
    assert "fine_samples" not in adaptive_source
    assert "rec_centers[refine, None" not in adaptive_source
    assert "_distances_to_shifted_centers" in average_source
    assert "[:, None" not in average_source
    assert "[None, :" not in average_source

    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        shape=(2, 1, 1),
    )
    centers = grid.cell_centers()
    options = {"desired_img_gauss_order": 2}
    offsets, weights, extents = greit_module._desired_cell_offsets(
        grid,
        centers,
        mode="gauss",
        options=options,
    )
    samples = centers[:, None, :3] + offsets[None, :, :] * extents[:, None, :]
    xyz_matrix = np.array([[0.5], [0.5], [0.5]], dtype=np.float64)
    radii = np.array([0.3], dtype=np.float64)
    steepness = np.array([10.0], dtype=np.float64)

    streamed = greit_module._greit_sigmoid_average_over_offsets(
        centers,
        extents,
        offsets,
        weights,
        xyz_matrix,
        radii,
        steepness,
    )
    sampled = greit_module._greit_sigmoid_average_over_samples(
        samples,
        weights,
        xyz_matrix,
        radii,
        steepness,
    )

    np.testing.assert_allclose(streamed, sampled)


def test_v460_desired_extent_active_axes_avoid_bool_matrix() -> None:
    gauss_source = inspect.getsource(greit_module._gauss_offsets_for_extents)
    offsets_source = inspect.getsource(greit_module._desired_cell_offsets)
    helper_source = inspect.getsource(greit_module._active_extent_axes)

    assert "extents > np.finfo" not in gauss_source
    assert "extents > np.finfo" not in offsets_source
    assert "_active_extent_axes(extents)" in gauss_source
    assert "_active_extent_axes(extents)" in offsets_source
    assert "np.max(arr[:, axis])" in helper_source

    extents = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.0, 0.25, 0.0],
        ],
        dtype=np.float64,
    )
    active, active_count = greit_module._active_extent_axes(extents)

    assert active.tolist() == [False, True, False]
    assert active_count == 1

    offsets, weights = greit_module._gauss_offsets_for_extents(extents, 2)

    assert offsets.shape[1] == 3
    assert weights.shape[0] == offsets.shape[0]
    np.testing.assert_allclose(offsets[:, 0], 0.0)
    np.testing.assert_allclose(offsets[:, 2], 0.0)


def test_v464_desired_cell_extents_negative_check_uses_min_reduction() -> None:
    source = inspect.getsource(greit_module._as_desired_cell_extents)

    assert "np.any(array < 0.0)" not in source
    assert "np.min(array)" in source

    extents = greit_module._as_desired_cell_extents(
        np.array([[0.0, 0.5], [0.25, 0.0]], dtype=np.float64),
        n_cells=2,
    )

    np.testing.assert_allclose(
        extents,
        np.array([[0.0, 0.5, 0.0], [0.25, 0.0, 0.0]], dtype=np.float64),
    )
    with pytest.raises(ValueError, match="non-negative"):
        greit_module._as_desired_cell_extents(
            np.array([[0.0, -0.5, 0.0]], dtype=np.float64),
            n_cells=1,
        )


def test_greit_cell_extent_vector_broadcast_uses_direct_fill() -> None:
    source = inspect.getsource(greit_module._desired_cell_extents)
    source += inspect.getsource(greit_module._as_desired_cell_extents)
    source += inspect.getsource(greit_module._as_extent_vector)
    source += inspect.getsource(greit_module._repeat_extent_rows)
    assert "broadcast_to" not in source
    assert "np.repeat" not in source
    assert "np.copyto" in source

    centers = np.zeros((4, 3), dtype=np.float64)
    extents = greit_module._desired_cell_extents(
        rec_model={},
        rec_centers=centers,
        options={"desired_img_cell_spacing": [0.1, 0.2, 0.3]},
    )

    assert extents.flags.c_contiguous
    np.testing.assert_allclose(extents, np.tile([0.1, 0.2, 0.3], (4, 1)))


def test_v225_center_desired_image_streams_target_columns() -> None:
    source = inspect.getsource(greit_module._greit_sigmoid_from_centers)

    assert "_distances_to_point" in source
    assert "_greit_sigmoid_1d" in source
    assert "[:, None" not in source
    assert "[None, :" not in source

    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
        ],
        dtype=np.float64,
    )
    xyz_matrix = np.array(
        [
            [0.0, 3.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    radii = np.array([1.0, 1.0], dtype=np.float64)
    steepness = np.array([10.0, 10.0], dtype=np.float64)
    streamed = greit_module._greit_sigmoid_from_centers(
        centers,
        xyz_matrix,
        radii,
        steepness,
    )
    distances = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float64)
    expected = greit_module._greit_sigmoid_from_distances(
        distances,
        radii,
        steepness,
    )
    np.testing.assert_allclose(streamed, expected)


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


def test_v284_greit_target_generation_streams_distances_and_rows() -> None:
    source = inspect.getsource(greit_module.generate_spherical_targets)
    ball_source = inspect.getsource(greit_module._equivalent_ball_mask)

    assert "np.linalg.norm" not in source
    assert "center.reshape" not in source
    assert "append(" not in source
    assert "_squared_distances_to_point" in source
    assert "target[~mask]" not in source
    assert "np.multiply(target, mask, out=target)" in source
    assert "np.linalg.norm" not in ball_source
    assert "_squared_distances_to_point" in ball_source

    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        shape=(2, 2, 2),
    )
    centers = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]], dtype=float)
    spheres = generate_spherical_targets(
        grid,
        centers=centers,
        radius=0.25,
        kind="sphere",
        amplitude=3.0,
    )
    blobs = generate_spherical_targets(
        grid,
        centers=centers,
        radius=0.75,
        kind="blob",
        amplitude=2.0,
    )

    expected_spheres = np.zeros((2, 8), dtype=float)
    expected_spheres[0, 0] = 3.0
    expected_spheres[1, -1] = 3.0
    np.testing.assert_allclose(spheres.values, expected_spheres)

    dist2 = np.sum((grid.cell_centers() - centers[0]) ** 2, axis=1)
    expected_blob = 2.0 * np.exp(-0.5 * dist2 / (0.75 * 0.75))
    expected_blob[dist2 > 0.75 * 0.75] = 0.0
    np.testing.assert_allclose(blobs.values[0], expected_blob)

    selected = greit_module._equivalent_ball_mask(
        grid.cell_centers(),
        np.ones(8, dtype=float),
        center=centers[0],
        target_volume=2.0,
    )
    assert np.count_nonzero(selected) == 2
    assert selected[0]


def test_v286_as_xyz_points_direct_fills_z_padding(monkeypatch) -> None:
    def _fail_column_stack(*_args, **_kwargs):
        raise AssertionError("XYZ padding must direct-fill")

    monkeypatch.setattr(greit_module.np, "column_stack", _fail_column_stack)
    source = inspect.getsource(greit_module._as_xyz_points)
    assert "np.column_stack" not in source

    points2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    xyz = greit_module._as_xyz_points(points2d, name="test points")
    np.testing.assert_allclose(
        xyz,
        np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]], dtype=float),
    )
    assert xyz.flags.c_contiguous


def test_v465_as_xyz_points_finite_scan_uses_bounded_work_buffer() -> None:
    source = inspect.getsource(greit_module._as_xyz_points)
    model_source = inspect.getsource(greit_module._model_nodes)
    helper_source = inspect.getsource(greit_module._all_finite_values)

    assert "np.isfinite(points).all()" not in source
    assert "_all_finite_values(points)" in source
    assert "_all_finite_values(nodes)" in model_source
    assert "finite_checked=True" in model_source
    assert "out=work_view" in helper_source

    assert greit_module._all_finite_values(np.array([[1.0, 2.0]], dtype=np.float64))
    assert not greit_module._all_finite_values(
        np.array([[1.0, np.nan]], dtype=np.float64)
    )
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._as_xyz_points(
            np.array([[1.0, np.inf]], dtype=np.float64),
            name="test points",
        )

    nodes = greit_module._model_nodes(
        {"nodes": np.array([[1.0, np.nan, 0.0]], dtype=np.float64)}
    )
    assert nodes is None


def test_v295_greit_xyzr_and_distribution_bounds_direct_fill(monkeypatch) -> None:
    def _fail_vstack(*_args, **_kwargs):
        raise AssertionError("GREIT xyzr/bounds assembly must direct-fill")

    monkeypatch.setattr(greit_module.np, "vstack", _fail_vstack)
    for obj in (
        greit_module.build_greit_finite_target_responses,
        greit_module._resolve_distribution_bounds,
    ):
        assert "np.vstack" not in inspect.getsource(obj)

    centers = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)
    model = _FiniteTargetForwardModel(centers, np.eye(2, dtype=float))
    responses = build_greit_finite_target_responses(
        model,
        centers=centers,
        target_radius=[0.1, 0.2],
        target_contrast=[0.5, 1.0],
        normalize=False,
    )
    np.testing.assert_allclose(
        responses.xyzr,
        np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.1, 0.2],
            ],
            dtype=float,
        ),
    )
    bounds = greit_module._resolve_distribution_bounds(
        {"nodes": np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float)},
        None,
    )
    np.testing.assert_allclose(bounds, np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))


def test_v461_inside_mask_bounds_fallback_avoids_bool_matrix() -> None:
    inside_source = inspect.getsource(greit_module._inside_mask_from_model_nodes)
    helper_source = inspect.getsource(greit_module._points_within_bounds_mask)

    assert "np.all(" not in inside_source
    assert "& (centers[:, :3] <=" not in inside_source
    assert "_points_within_bounds_mask" in inside_source
    assert "np.greater_equal" in helper_source
    assert "np.less_equal" in helper_source
    assert "out=axis_mask" in helper_source

    model = {"nodes": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)}
    centers = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.1, 0.5, 0.5],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    mask = greit_module._inside_mask_from_model_nodes(model, centers)

    assert mask.flags.c_contiguous
    assert mask.tolist() == [True, False, True]


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


def test_v586_target_plane_offset_reuses_centers_when_no_offset() -> None:
    centers = np.ascontiguousarray(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]], dtype=np.float64)
    )

    resolved, metadata = greit_module._apply_target_plane_offset(
        centers,
        target_plane=None,
        target_offset=None,
    )

    assert resolved.flags.c_contiguous
    assert np.shares_memory(resolved, centers)
    assert metadata == {"target_plane": None, "target_offset": None}
    source = inspect.getsource(greit_module._apply_target_plane_offset)
    assert "np.asarray(centers, dtype=np.float64).copy()" not in source
    assert 'np.array(center_values, dtype=np.float64, copy=True, order="C")' in source


def test_build_greit_finite_target_responses_applies_measurement_order_before_contract() -> (
    None
):
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
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    model = _FiniteTargetForwardModel(centers, measurement_matrix)
    order = np.array([2, 0, 3, 1], dtype=np.int64)

    responses = build_greit_finite_target_responses(
        model,
        centers=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        target_radius=0.25,
        target_contrast=0.5,
        normalize=False,
        measurement_order=order,
        channel_mask=[False, True, False, False],
        measurement_weights=np.array([1.0, 4.0, 9.0, 16.0]),
    )

    expected_vh_raw = measurement_matrix @ np.ones(3, dtype=float)
    expected_vi_raw = np.column_stack(
        [
            measurement_matrix @ np.array([1.5, 1.0, 1.0]),
            measurement_matrix @ np.array([1.0, 1.0, 1.5]),
        ]
    )
    expected_y = expected_vi_raw[order, :] - expected_vh_raw[order].reshape(-1, 1)
    expected_contracted = expected_y.copy()
    expected_contracted[1, :] = 0.0
    expected_contracted[2, :] *= 3.0
    expected_contracted[3, :] *= 4.0

    np.testing.assert_allclose(responses.vh, expected_vh_raw[order])
    np.testing.assert_allclose(responses.vi, expected_vi_raw[order, :])
    np.testing.assert_allclose(responses.y, expected_y)
    np.testing.assert_allclose(responses.contracted_y, expected_contracted)
    assert responses.metadata["measurement_order_source"] == "provided"
    assert responses.metadata["measurement_order_first_indices"] == (2, 0, 3, 1)
    assert responses.metadata["bad_channel_count"] == 1


def test_v457_measurement_order_range_check_uses_min_max_reductions() -> None:
    source = inspect.getsource(greit_module._resolve_measurement_order)

    assert "(order < 0) | (order >=" not in source
    assert "int(np.min(order)) < 0" in source
    assert "int(np.max(order)) >=" in source

    resolved, metadata = greit_module._resolve_measurement_order(
        np.array([2, 0, 1], dtype=np.int64),
        n_measurements=3,
    )

    np.testing.assert_array_equal(resolved, np.array([2, 0, 1], dtype=np.int64))
    assert metadata["measurement_order_source"] == "provided"
    with pytest.raises(ValueError, match="out of range"):
        greit_module._resolve_measurement_order(
            np.array([0, 3, 1], dtype=np.int64),
            n_measurements=3,
        )


def test_v459_measurement_order_permutation_check_avoids_unique_sort() -> None:
    source = inspect.getsource(greit_module._resolve_measurement_order)
    helper_source = inspect.getsource(greit_module._measurement_order_flags)

    assert "np.unique(order)" not in source
    assert "np.array_equal(order, identity)" not in source
    assert "_measurement_order_flags(order)" in source
    assert "seen = np.zeros(values.size, dtype=bool)" in helper_source

    resolved, metadata = greit_module._resolve_measurement_order(
        np.array([0, 1, 2], dtype=np.int64),
        n_measurements=3,
    )

    assert resolved is None
    assert metadata["measurement_order_source"] == "identity"
    with pytest.raises(ValueError, match="permutation"):
        greit_module._resolve_measurement_order(
            np.array([0, 1, 1], dtype=np.int64),
            n_measurements=3,
        )


def test_v458_vh_normalization_checks_min_abs_without_full_bool_arrays() -> None:
    calc_source = inspect.getsource(greit_module._calc_greit_difference_data)
    ensure_source = inspect.getsource(greit_module._ensure_nonzero_vh_for_normalization)
    helper_source = inspect.getsource(greit_module._min_abs_value)

    assert "np.any(np.abs(vh)" not in calc_source
    assert "np.any(np.abs(vh)" not in ensure_source
    assert "_min_abs_value(vh)" in calc_source
    assert "_min_abs_value(vh)" in ensure_source
    assert "np.abs(arr[start:stop], out=work_view)" in helper_source

    vh = np.array([2.0, -4.0, 8.0], dtype=np.float64)
    vi = np.array([[4.0, 6.0], [-8.0, -4.0], [8.0, 16.0]], dtype=np.float64)

    y = greit_module._calc_greit_difference_data(vh, vi, normalize=True)

    np.testing.assert_allclose(y, vi / vh.reshape(-1, 1) - 1.0)
    assert greit_module._min_abs_value(vh, chunk_size=2) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="non-zero homogeneous"):
        greit_module._calc_greit_difference_data(
            np.array([1.0, 0.0, 2.0], dtype=np.float64),
            np.ones((3, 1), dtype=np.float64),
            normalize=True,
        )


def test_v466_greit_response_finite_checks_use_bounded_scan() -> None:
    diff_source = inspect.getsource(greit_module._calc_greit_difference_data)
    vector_source = inspect.getsource(greit_module._measurement_vector_from_result)

    assert "np.isfinite(y).all()" not in diff_source
    assert "np.isfinite(vector).all()" not in vector_source
    assert "_all_finite_values(y)" in diff_source
    assert "_all_finite_values(vector)" in vector_source

    y = greit_module._calc_greit_difference_data(
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([[2.0], [4.0]], dtype=np.float64),
        normalize=True,
    )
    np.testing.assert_allclose(y, [[1.0], [1.0]])

    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._calc_greit_difference_data(
            np.array([1.0], dtype=np.float64),
            np.array([[np.inf]], dtype=np.float64),
            normalize=False,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._measurement_vector_from_result(
            SimpleNamespace(meas=np.array([1.0, np.nan], dtype=np.float64))
        )


def test_v470_greit_artifact_finite_checks_use_bounded_scan() -> None:
    for obj in (
        greit_module._validate_training_response_matrix,
        greit_module._validate_desired_component_matrix,
        greit_module._noise_covar_matrix,
        greit_module._validate_pjt_cache,
        greit_module._measurement_noise_matrix,
        greit_module._eidors_nf_vh_vector,
        greit_module._eidors_nf_measurement_matrix,
        greit_module._eidors_nf_volume_weights,
    ):
        source = inspect.getsource(obj)
        assert "np.isfinite(matrix).all()" not in source
        assert "np.isfinite(vector).all()" not in source
        assert "np.isfinite(noise).all()" not in source
        assert "_all_finite_values(" in source

    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._validate_training_response_matrix([[1.0, np.nan]])
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._validate_desired_component_matrix([[np.inf]], n_targets=1)
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._noise_covar_matrix(
            np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64),
            n_measurements=2,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._validate_pjt_cache(
            np.array([[np.nan, 1.0]], dtype=np.float64),
            n_rec_parameters=1,
            n_measurements=2,
            dtype=np.float64,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._measurement_noise_matrix(
            np.array([[np.nan]], dtype=np.float64),
            y=np.zeros((1, 1), dtype=np.float64),
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._eidors_nf_vh_vector(
            np.array([1.0, np.nan], dtype=np.float64),
            n_measurements=2,
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._eidors_nf_measurement_matrix(
            np.array([[1.0], [np.inf]], dtype=np.float64),
            n_measurements=2,
            name="signal_y",
        )
    with pytest.raises(FloatingPointError, match="non-finite"):
        greit_module._eidors_nf_volume_weights(
            np.array([1.0, np.nan], dtype=np.float64),
            n_rec_parameters=2,
        )


def test_v495_greit_remaining_finite_guards_use_bounded_scan() -> None:
    checked = (
        greit_module._as_desired_cell_extents,
        greit_module._as_extent_vector,
        greit_module._desired_adaptive_band,
        greit_module.calc_greit_rm,
        greit_module._resolve_jacobian,
        greit_module._resolve_targets,
        greit_module._resolve_distribution_bounds,
        greit_module._resolve_axis_edges,
        greit_module._desired_radii,
        greit_module._as_3d_offset,
        greit_module._resolve_background_conductivity,
        greit_module._as_target_contrasts,
        greit_module._forward_cell_centers,
        greit_module._desired_rec_centers,
        greit_module._as_eidors_xyz,
        greit_module._desired_steepness,
        greit_module._validate_desired_matrix,
        greit_module._cell_centers,
        greit_module._as_centers,
        greit_module._nearest_unique_center_distance,
        greit_module._measurement_regularisation,
        greit_module._validate_log10_bracket,
        greit_module._rec_model_array,
        greit_module._as_flat_image,
        greit_module._as_cell_volumes,
        greit_module._metric_centers,
        greit_module._as_target_values,
    )
    old_patterns = (
        "np.isfinite(array).all()",
        "np.isfinite(extent).all()",
        "np.isfinite(band).all()",
        "np.isfinite(rm).all()",
        "np.isfinite(matrix).all()",
        "np.isfinite(values).all()",
        "np.isfinite(arr).all()",
        "np.isfinite(radii).all()",
        "np.isfinite(offset).all()",
        "np.isfinite(centers).all()",
        "np.isfinite(xyz).all()",
        "np.isfinite(embedded_radii).all()",
        "np.isfinite(steepness).all()",
        "np.isfinite(center_matrix).all()",
        "np.isfinite([lo, hi]).all()",
        "np.isfinite(volumes).all()",
        "np.isfinite(coords).all()",
        "np.isfinite(target).all()",
    )

    for obj in checked:
        source = inspect.getsource(obj)
        assert "_all_finite_values(" in source
        for old_pattern in old_patterns:
            assert old_pattern not in source


def test_v504_greit_noise_regularisation_diagonal_uses_in_place_add() -> None:
    calc_source = inspect.getsource(greit_module.calc_greit_rm)
    noise_source = inspect.getsource(greit_module._noise_covar_term)
    compat_source = inspect.getsource(greit_module._noise_covar_matrix)
    regularisation_source = inspect.getsource(greit_module._measurement_regularisation)

    assert "y_matrix @ y_matrix.T + (noiselev * noiselev) * sn" not in calc_source
    assert "add_scaled_diagonal_in_place(m, sn, noiselev * noiselev)" in calc_source
    assert "add_scaled_values_in_place(m, sn, noiselev * noiselev)" in calc_source
    assert "scalar * np.eye" not in noise_source
    assert "np.eye(n_measurements" not in regularisation_source
    assert "np.diag(array) if array.ndim == 1" not in regularisation_source
    assert "add_scaled_diagonal_in_place(matrix, term, 1.0)" in compat_source

    y = np.array(
        [[1.0, 0.5], [0.25, 1.5], [1.0, -0.5]],
        dtype=float,
    )
    d = np.array([[0.5, 1.0], [-1.0, 0.25]], dtype=float)
    diagonal_noise = np.array([1.0, 2.0, 4.0], dtype=float)
    dense_noise = np.diag(diagonal_noise)

    diagonal_result = greit_module.calc_greit_rm(
        y,
        d,
        weight=0.2,
        noise_covar=diagonal_noise,
    )
    dense_result = greit_module.calc_greit_rm(
        y,
        d,
        weight=0.2,
        noise_covar=dense_noise,
    )

    np.testing.assert_allclose(diagonal_result.rm, dense_result.rm)
    assert diagonal_result.metadata["noise_covar_source"] == "diagonal"
    assert diagonal_result.metadata["sn_kind"] == "diagonal"
    assert diagonal_result.metadata["sn_shape"] == (3, 3)


def test_v500_greit_range_guards_use_reductions_without_comparison_masks() -> None:
    checked = (
        greit_module._as_desired_cell_extents,
        greit_module._infer_center_spacing,
        greit_module._desired_adaptive_band,
        greit_module._resolve_distribution_bounds,
        greit_module._parse_imgsz,
        greit_module._resolve_axis_edges,
        greit_module._parse_downsample,
        greit_module._resolve_targets,
        greit_module._desired_radii,
        greit_module._desired_steepness,
    )
    old_patterns = (
        "np.any(extent < 0.0)",
        "np.any(spacing > 0.0)",
        "np.any(band < 0.0)",
        "np.any(arr[1] <= arr[0])",
        "np.any(upper <= lower)",
        "np.any(arr <= 0)",
        "np.any(np.diff(arr) <= 0.0)",
        "np.any(factors <= 0)",
        "np.any(phases < 0)",
        "np.any(phases >= factors)",
        "np.any(radii <= 0.0)",
        "np.any(steepness <= 0.0)",
    )

    for obj in checked:
        source = inspect.getsource(obj)
        for old_pattern in old_patterns:
            assert old_pattern not in source


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
    assert desired.metadata["desired_solution_fn"] == "GREIT_desired_img_sigmoid:gauss"
    assert desired.metadata["desired_image_sampling"] == "gauss"
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
    assert desired.values[center_idx, 0] > 0.9
    assert 0.0 < desired.values[neighbor_idx, 0] < desired.values[center_idx, 0]
    assert 0.0 < desired.values[far_idx, 0] < 0.01


def test_build_greit_desired_images_supports_sampling_modes() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        shape=(1, 1, 1),
    )
    center = np.array([[0.5, 0.5, 0.5]], dtype=float)

    center_sampled = build_greit_desired_images(
        grid,
        xyz=center,
        radius=0.2,
        desired_solution_fn="center",
        desired_options={"desired_img_threshold": 0.0},
    )
    gauss = build_greit_desired_images(
        grid,
        xyz=center,
        radius=0.2,
        desired_solution_fn="gauss",
        desired_options={"desired_img_threshold": 0.0, "desired_img_gauss_order": 3},
    )
    adaptive = build_greit_desired_images(
        grid,
        xyz=center,
        radius=0.2,
        desired_solution_fn="adaptive_gauss",
        desired_options={
            "desired_img_threshold": 0.0,
            "desired_img_adaptive_base_order": 2,
            "desired_img_adaptive_fine_order": 5,
        },
    )
    sobol = build_greit_desired_images(
        grid,
        xyz=center,
        radius=0.2,
        desired_solution_fn="sobol_qmc",
        desired_options={
            "desired_img_threshold": 0.0,
            "desired_img_sobol_samples": 16,
            "desired_img_sobol_seed": 7,
        },
    )

    assert center_sampled.metadata["desired_image_sampling"] == "center"
    assert gauss.metadata["desired_image_sampling"] == "gauss"
    assert adaptive.metadata["desired_image_sampling"] == "adaptive_gauss"
    assert sobol.metadata["desired_image_sampling"] == "sobol_qmc"
    assert center_sampled.values[0, 0] > gauss.values[0, 0]
    assert center_sampled.values[0, 0] > adaptive.values[0, 0]
    assert np.isfinite(sobol.values).all()


def test_build_greit_desired_images_adaptive_gauss_refines_boundary_cells() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
        shape=(2, 1, 1),
    )
    xyz = np.array([[0.5, 0.5, 0.5]], dtype=float)

    coarse = build_greit_desired_images(
        grid,
        xyz=xyz,
        radius=0.5,
        desired_solution_fn="gauss",
        desired_options={"desired_img_threshold": 0.0, "desired_img_gauss_order": 2},
    )
    adaptive = build_greit_desired_images(
        grid,
        xyz=xyz,
        radius=0.5,
        desired_solution_fn="adaptive_gauss",
        desired_options={
            "desired_img_threshold": 0.0,
            "desired_img_adaptive_base_order": 2,
            "desired_img_adaptive_fine_order": 5,
        },
    )
    fine = build_greit_desired_images(
        grid,
        xyz=xyz,
        radius=0.5,
        desired_solution_fn="gauss",
        desired_options={"desired_img_threshold": 0.0, "desired_img_gauss_order": 5},
    )

    assert np.linalg.norm(adaptive.values - fine.values) < np.linalg.norm(
        coarse.values - fine.values
    )


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
    assert artifact.metadata["large_cache"] is True
    assert artifact.metadata["checksum_algorithm"] == "sha256"
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
    lazy_artifact = read_hdf5_artifact(artifact_path, lazy=True)
    assert lazy_artifact.arrays["RM"].compression == "gzip"
    assert lazy_artifact.arrays["RM"].chunks is not None

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


def test_build_native_greit_training_pipeline_uses_native_forward_y_d_and_rm_formula(
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
            [1.5, 0.0, 0.0, -0.5],
        ],
        dtype=float,
    )
    model = _FiniteTargetForwardModel(grid.cell_centers(), measurement_matrix)
    artifact_path = tmp_path / "native_training_pipeline.h5"

    pipeline = build_native_greit_training_pipeline(
        model,
        rec_model=grid,
        centers=grid.cell_centers()[:2],
        target_radius=0.4,
        target_contrast=0.25,
        normalize=False,
        desired_solution_fn="gauss",
        desired_options={
            "desired_img_threshold": 0.0,
            "desired_img_gauss_order": 3,
        },
        weight=0.2,
        noise_covar=1.5,
        artifact_path=artifact_path,
        fwd_model_signature="native-unit-fwd",
    )

    assert isinstance(pipeline, GREITNativeTrainingPipeline)
    assert artifact_path.exists()
    assert model.solve_calls == 3
    assert pipeline.metadata["training_data_source"] == "native_pyeidors_forward"
    assert pipeline.greit.metadata["uses_eidors_exported_vh_vi_d"] is False
    assert pipeline.greit.metadata["training_mode"] == "forward"
    assert pipeline.greit.metadata["eidors_parity"] is True
    assert pipeline.greit.metadata["difference_normalization"] == "raw"
    assert pipeline.greit.metadata["desired_solution_fn"] == (
        "GREIT_desired_img_sigmoid:gauss"
    )

    expected_conductivities = np.array(
        [
            [1.25, 1.0, 1.0, 1.0],
            [1.0, 1.25, 1.0, 1.0],
        ],
        dtype=float,
    )
    expected_vh = measurement_matrix @ np.ones(4, dtype=float)
    expected_vi = np.column_stack(
        [measurement_matrix @ row for row in expected_conductivities]
    )
    np.testing.assert_allclose(pipeline.responses.vh, expected_vh)
    np.testing.assert_allclose(pipeline.responses.vi, expected_vi)
    np.testing.assert_allclose(
        pipeline.responses.y,
        expected_vi - expected_vh.reshape(-1, 1),
    )
    np.testing.assert_allclose(pipeline.greit.y, pipeline.responses.contracted_y)
    np.testing.assert_allclose(pipeline.greit.d, pipeline.desired_images.values)

    y = pipeline.responses.contracted_y
    d = pipeline.desired_images.values
    expected_pjt = d @ y.T
    noiselev = 0.2 * float(np.mean(np.abs(y)))
    expected_m = y @ y.T + (noiselev * noiselev) * 1.5 * np.eye(y.shape[0])
    expected_rm = np.linalg.solve(expected_m.T, expected_pjt.T).T

    np.testing.assert_allclose(pipeline.greit.pjt, expected_pjt)
    np.testing.assert_allclose(pipeline.greit.m, expected_m)
    np.testing.assert_allclose(pipeline.greit.rm, expected_rm)
    np.testing.assert_allclose(pipeline.rm, expected_rm)


def test_native_greit_pipeline_supports_2d_centers_and_eidors_nf1_search() -> None:
    centers_2d = np.array(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=float,
    )
    measurement_matrix = np.array(
        [
            [1.0, 0.2, 0.1, 0.3],
            [0.3, 1.2, 0.4, 0.2],
            [0.2, 0.4, 1.1, 0.5],
            [0.5, 0.2, 0.3, 1.3],
        ],
        dtype=float,
    )
    model = _FiniteTargetForwardModel(centers_2d, measurement_matrix)

    pipeline = build_native_greit_training_pipeline(
        model,
        rec_model=centers_2d,
        centers=centers_2d[:3],
        target_radius=0.55,
        target_contrast=0.2,
        normalize=False,
        desired_solution_fn="center",
        weight_strategy="eidors_nf1",
        target_noise_figure=1.0,
        fwd_model_signature="native-2d-unit-fwd",
    )

    assert pipeline.metadata["greit_weight_strategy"] == "eidors_nf1"
    assert pipeline.metadata["greit_weight_source"] == "eidors_nf1_search"
    assert pipeline.metadata["greit_noise_figure_target"] == pytest.approx(1.0)
    assert pipeline.metadata["greit_weight_search"]["weight"] == pytest.approx(
        pipeline.greit.metadata["weight"]
    )
    assert pipeline.responses.xyzr.shape == (4, 3)
    np.testing.assert_allclose(pipeline.responses.xyzr[2], 0.0)
    assert pipeline.desired_images.rec_centers.shape == (4, 3)
    np.testing.assert_allclose(pipeline.desired_images.rec_centers[:, 2], 0.0)
    assert pipeline.greit.rec_model.shape == (4, 3)


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


def test_v237_weighted_centroid_avoids_broadcast_matrix() -> None:
    source = inspect.getsource(greit_module._weighted_centroid)

    assert "weights[:, None]" not in source
    assert "coords * weights" not in source
    assert "weights @ coords" in source
    coords = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=float)
    weights = np.array([1.0, 3.0], dtype=float)
    np.testing.assert_allclose(
        greit_module._weighted_centroid(coords, weights),
        np.array([1.5, 2.5, 3.5], dtype=float),
    )


def test_v440_greit_metrics_uses_masked_weighted_sums_without_subset_copies() -> None:
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
    metrics_source = inspect.getsource(greit_module.greit_metrics)
    helper_source = inspect.getsource(greit_module._masked_weighted_sum)
    assert "_masked_weighted_sum" in metrics_source
    assert "where=qmi" in metrics_source
    assert "weights[qmi]" not in metrics_source
    assert "signed_image[qmi]" not in metrics_source
    assert "weights[opposite]" not in metrics_source
    assert "signed_image[opposite]" not in metrics_source
    assert "qmi & ~equivalent_ball" not in metrics_source
    assert "values_arr[mask" not in helper_source
    assert "weights_arr[mask" not in helper_source


def test_v456_greit_metric_cell_volume_positivity_uses_reduction() -> None:
    source = inspect.getsource(greit_module._as_cell_volumes)

    assert "np.any(volumes <= 0.0)" not in source
    assert "float(np.min(volumes)) <= 0.0" in source

    np.testing.assert_allclose(
        greit_module._as_cell_volumes([1.0, 2.0, 3.0], n_cells=3),
        np.array([1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError, match="cell_volumes"):
        greit_module._as_cell_volumes([1.0, 0.0, 2.0], n_cells=3)


def test_v443_greit_metrics_default_target_uses_masked_centroid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        shape=(3, 3, 3),
    )
    image = np.zeros(grid.shape, dtype=float)
    image[1, 1, 1] = 1.0
    target_mask = np.zeros(grid.shape, dtype=bool)
    target_mask[1, 1, 1] = True

    def _raise_if_called(*_args, **_kwargs):
        raise AssertionError("default target mask must not materialize target values")

    monkeypatch.setattr(greit_module, "_as_target_values", _raise_if_called)

    metrics = greit_metrics(image, target_mask, centers=grid.cell_centers())

    assert metrics["AR"] == pytest.approx(1.0)
    assert metrics["PE"] == pytest.approx(0.0)
    metrics_source = inspect.getsource(greit_module.greit_metrics)
    centroid_source = inspect.getsource(greit_module._masked_weighted_centroid)
    assert "if target_values is None:" in metrics_source
    assert "_masked_weighted_centroid(coords, weights, mask)" in metrics_source
    assert "target_integral = float(np.sum(weights, where=mask))" in metrics_source
    assert "mask.astype(np.float64)" not in metrics_source
    assert (
        "_masked_weighted_sum(coords[:, axis], weights_arr, mask_arr)"
        in centroid_source
    )


def test_v444_greit_metrics_positive_target_reuses_image_for_signed_view() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [3.0, 3.0, 3.0],
        shape=(3, 3, 3),
    )
    image = np.zeros(grid.shape, dtype=float)
    image[1, 1, 1] = 1.0
    target_mask = np.zeros(grid.shape, dtype=bool)
    target_mask[1, 1, 1] = True

    metrics = greit_metrics(image, target_mask, centers=grid.cell_centers())

    assert metrics["AR"] == pytest.approx(1.0)
    source = inspect.getsource(greit_module.greit_metrics)
    assert "signed_image = image if signal_sign > 0.0 else -image" in source
    assert "signed_image = signal_sign * image" not in source


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
