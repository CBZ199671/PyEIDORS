from __future__ import annotations

import inspect

import numpy as np

import eit_app.controllers.dataset_generator_controller as dataset_controller
import eit_app.controllers.forward_solver_controller as fwd_controller
from eit_app.controllers.forward_solver_controller import _paint_shape
from eit_app.controllers.forward_solver_controller import _cell_volume_sample_points
from eit_app.models.simulation_state import InhomogeneitySpec


def test_v315_volume_fraction_streams_samples_without_chunk_tensor():
    x0, x1 = -0.04, 0.04
    y0, y1 = -0.04, 0.04
    z_levels = np.array([-0.08, -0.056, -0.024, 0.0, 0.024, 0.056, 0.08])
    cell_vertices = []
    for z0, z1 in zip(z_levels[:-1], z_levels[1:], strict=True):
        cell_vertices.append(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )
    cell_vertices_arr = np.asarray(cell_vertices, dtype=float)
    centers = cell_vertices_arr.mean(axis=1)
    full_values = np.ones(centers.shape[0], dtype=float)
    streamed_values = np.ones_like(full_values)
    spec = InhomogeneitySpec(shape="circle", size_x=0.063, conductivity=2.0)

    _paint_shape(
        full_values,
        centers,
        spec,
        mesh_dimension=3,
        cell_vertices=cell_vertices_arr,
    )
    node_coords = cell_vertices_arr.reshape(-1, 3)
    cell_connectivity = np.arange(node_coords.shape[0], dtype=np.int32).reshape(-1, 8)
    _paint_shape(
        streamed_values,
        centers,
        spec,
        mesh_dimension=3,
        node_coords=node_coords,
        cell_connectivity=cell_connectivity,
    )

    np.testing.assert_allclose(streamed_values, full_values)
    streaming_source = inspect.getsource(
        fwd_controller._apply_volume_fraction_streaming
    )
    assert "np.take" in streaming_source
    assert "_cell_volume_sample_points" not in streaming_source
    assert "vertices = np.take" not in streaming_source
    assert "coords[cells" not in streaming_source
    assert "node_coords[cell_connectivity]" not in inspect.getsource(
        fwd_controller.execute_forward_request
    )
    assert "node_coords[cell_connectivity]" not in inspect.getsource(
        dataset_controller._DatasetGeneratorWorker.run
    )


def test_v374_volume_fraction_streaming_preserves_float32_coords(monkeypatch):
    node_coords = np.array(
        [
            [-0.05, -0.05, -0.05],
            [0.05, -0.05, -0.05],
            [0.05, 0.05, -0.05],
            [-0.05, 0.05, -0.05],
            [-0.05, -0.05, 0.05],
            [0.05, -0.05, 0.05],
            [0.05, 0.05, 0.05],
            [-0.05, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    cell_connectivity = np.arange(8, dtype=np.int32).reshape(1, 8)
    values = np.ones(1, dtype=np.float32)
    seen_sample_dtypes: list[np.dtype] = []
    original_empty = fwd_controller.np.empty

    def _capture_empty(shape, dtype=float, *args, **kwargs):
        if tuple(shape) == (1, 3):
            seen_sample_dtypes.append(np.dtype(dtype))
        return original_empty(shape, dtype=dtype, *args, **kwargs)

    def _inside_fn(samples):
        assert np.asarray(samples).dtype == np.dtype(np.float32)
        return np.ones(samples.shape[:2], dtype=bool)

    monkeypatch.setattr(fwd_controller.np, "empty", _capture_empty)

    applied = fwd_controller._apply_volume_fraction_streaming(
        values,
        node_coords,
        cell_connectivity,
        _inside_fn,
        2.0,
        chunk_size=1,
    )

    assert applied
    assert seen_sample_dtypes
    assert set(seen_sample_dtypes) == {np.dtype(np.float32)}
    assert values[0] == np.float32(2.0)

    source = inspect.getsource(fwd_controller._apply_volume_fraction_streaming)
    assert "np.asarray(node_coords, dtype=np.float64)" not in source
    assert "dtype=coords_xyz.dtype" in source


def test_v381_volume_fraction_ratio_buffers_preserve_float32(monkeypatch):
    apply_source = inspect.getsource(fwd_controller._apply_volume_fraction)
    streaming_source = inspect.getsource(
        fwd_controller._apply_volume_fraction_streaming
    )

    assert "inside.mean(axis=1, dtype=fraction_dtype)" in apply_source
    assert "inside_counts = np.zeros(n_chunk, dtype=np.float64)" not in streaming_source
    assert "inside_counts = np.zeros(n_chunk, dtype=fraction_dtype)" in streaming_source

    values = np.ones(2, dtype=np.float32)
    sample_points = np.zeros((2, 2, 3), dtype=np.float32)
    inside = np.array([[True, False], [True, True]], dtype=bool)

    applied = fwd_controller._apply_volume_fraction(
        values,
        sample_points,
        inside,
        2.0,
    )

    assert applied
    assert values.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(values, np.array([1.5, 2.0], dtype=np.float32))

    node_coords = np.array(
        [
            [-0.05, -0.05, -0.05],
            [0.05, -0.05, -0.05],
            [0.05, 0.05, -0.05],
            [-0.05, 0.05, -0.05],
            [-0.05, -0.05, 0.05],
            [0.05, -0.05, 0.05],
            [0.05, 0.05, 0.05],
            [-0.05, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    cell_connectivity = np.arange(8, dtype=np.int32).reshape(1, 8)
    streamed_values = np.ones(1, dtype=np.float32)
    seen_zero_dtypes: list[np.dtype] = []
    original_zeros = fwd_controller.np.zeros

    def _capture_zeros(shape, *args, dtype=None, **kwargs):
        if shape == 1:
            seen_zero_dtypes.append(np.dtype(dtype))
        return original_zeros(shape, *args, dtype=dtype, **kwargs)

    monkeypatch.setattr(fwd_controller.np, "zeros", _capture_zeros)

    streamed = fwd_controller._apply_volume_fraction_streaming(
        streamed_values,
        node_coords,
        cell_connectivity,
        lambda samples: np.ones(samples.shape[:2], dtype=bool),
        2.0,
        chunk_size=1,
    )

    assert streamed
    assert seen_zero_dtypes == [np.dtype(np.float32)]
    assert streamed_values[0] == np.float32(2.0)


def test_v437_paint_shape_applies_masks_with_copyto_where(monkeypatch) -> None:
    paint_source = inspect.getsource(fwd_controller._paint_shape)
    helper_source = inspect.getsource(fwd_controller._paint_values_where)

    assert "values[dist2 <" not in paint_source
    assert "values[norm <" not in paint_source
    assert "values[mask]" not in paint_source
    assert "_paint_values_where(" in paint_source
    assert "np.copyto(values, conductivity, where=mask)" in helper_source

    original_copyto = fwd_controller.np.copyto
    copyto_calls: list[tuple[tuple[int, ...], np.dtype]] = []

    def _record_copyto(dst, src, *args, where=True, **kwargs):
        copyto_calls.append((np.asarray(where).shape, np.asarray(dst).dtype))
        return original_copyto(dst, src, *args, where=where, **kwargs)

    monkeypatch.setattr(fwd_controller.np, "copyto", _record_copyto)
    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    values = np.ones(3, dtype=np.float32)
    spec = InhomogeneitySpec(shape="rectangle", size_x=1.0, size_y=1.0)

    _paint_shape(values, centers, spec, mesh_dimension=3)

    assert copyto_calls == [((3,), np.dtype(np.float32))]
    np.testing.assert_array_equal(values, np.array([2.0, 1.0, 1.0], dtype=np.float32))


def test_v612_rectangle_center_mask_uses_bounded_axis_work_buffers() -> None:
    helper_source = inspect.getsource(fwd_controller._paint_axis_aligned_box_centers)
    paint_source = inspect.getsource(fwd_controller._paint_shape)

    assert "np.abs(centers[:, 0]" not in paint_source
    assert "np.abs(centers[:, 1]" not in paint_source
    assert "np.abs(centers[:, 2]" not in paint_source
    assert "np.subtract(axis_values, center, out=axis_work_chunk)" in helper_source
    assert "np.abs(axis_work_chunk, out=axis_work_chunk)" in helper_source
    assert "_paint_values_where(values[start:stop], mask_chunk" in helper_source

    centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 1.1],
        ],
        dtype=np.float32,
    )
    values = np.ones(centers.shape[0], dtype=np.float32)

    fwd_controller._paint_axis_aligned_box_centers(
        values,
        centers,
        ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        2.0,
        chunk_size=2,
    )

    np.testing.assert_array_equal(
        values,
        np.array([2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32),
    )


def test_v380_legacy_volume_fraction_cell_vertices_preserve_float32(
    monkeypatch,
) -> None:
    cell_vertices = np.array(
        [
            [
                [-0.05, -0.05, -0.05],
                [0.05, -0.05, -0.05],
                [0.05, 0.05, -0.05],
                [-0.05, 0.05, -0.05],
                [-0.05, -0.05, 0.05],
                [0.05, -0.05, 0.05],
                [0.05, 0.05, 0.05],
                [-0.05, 0.05, 0.05],
            ]
        ],
        dtype=np.float32,
    )

    samples = _cell_volume_sample_points(cell_vertices)

    assert samples is not None
    assert samples.dtype == np.dtype(np.float32)

    source = inspect.getsource(fwd_controller._cell_volume_sample_points)
    paint_source = inspect.getsource(fwd_controller._paint_shape)
    assert "np.asarray(cell_vertices, dtype=np.float64)" not in source
    assert "np.asarray(cell_vertices, dtype=float)" not in paint_source

    seen_dtypes: list[np.dtype] = []

    def _capture_sample_points(vertices):
        seen_dtypes.append(np.asarray(vertices).dtype)
        return None

    monkeypatch.setattr(
        fwd_controller,
        "_cell_volume_sample_points",
        _capture_sample_points,
    )
    values = np.ones(1, dtype=np.float32)
    centers = cell_vertices.mean(axis=1)
    spec = InhomogeneitySpec(shape="circle", size_x=0.063, conductivity=2.0)

    _paint_shape(
        values,
        centers,
        spec,
        mesh_dimension=3,
        cell_vertices=cell_vertices,
    )

    assert seen_dtypes == [np.dtype(np.float32)]


def test_v375_dataset_generator_reuses_forward_geometry_extractor() -> None:
    source = inspect.getsource(dataset_controller._DatasetGeneratorWorker.run)

    assert "cell_midpoints" not in source
    assert "cells_conn.links(i) for i in range" not in source
    assert "_forward_mesh_geometry_arrays(" in source


def test_v577_dataset_generator_reuses_clean_measurement_views() -> None:
    source = inspect.getsource(dataset_controller._DatasetGeneratorWorker.run)

    assert "data.meas.copy()" not in source
    assert "data_homog.meas.copy()" not in source
    assert "_forward_measurement_values(" in source
