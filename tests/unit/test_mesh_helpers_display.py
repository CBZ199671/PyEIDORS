from __future__ import annotations

import inspect

import numpy as np

from eit_app.ui.conductivity_image_widget import _project_cells_to_triangles
import eit_app.ui.mesh_helpers as mesh_helpers
from eit_app.ui.mesh_helpers import cell_to_node_average, extract_boundary_triangles


def test_v198_cell_to_node_average_avoids_repeat_temporary(monkeypatch) -> None:
    cells = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)
    values = np.array([1.0, 3.0], dtype=np.float32)

    def _fail_repeat(*_args, **_kwargs):
        raise AssertionError("cell_to_node_average must not materialize repeat values")

    monkeypatch.setattr(np, "repeat", _fail_repeat)

    node_values = cell_to_node_average(values, cells, 4)

    assert node_values.dtype == np.float32
    np.testing.assert_allclose(
        node_values,
        np.array([2.0, 1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_v198_cell_to_node_average_fills_orphans_without_widening() -> None:
    cells = np.array([[0, 1, 2]], dtype=np.int32)
    values = np.array([2.0], dtype=np.float32)

    node_values = cell_to_node_average(values, cells, 5)

    assert node_values.dtype == np.float32
    np.testing.assert_allclose(
        node_values,
        np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )


def test_v346_cell_to_node_average_fills_in_place_without_subset_copies() -> None:
    source = inspect.getsource(cell_to_node_average)

    assert "np.where" not in source
    assert "[np.isfinite" not in source
    assert "np.divide(" in source
    assert "out=node_sum" in source

    cells = np.array([[0, 1], [1, 2]], dtype=np.int32)
    values = np.array([1.0, np.inf], dtype=np.float32)

    node_values = cell_to_node_average(values, cells, 4)

    assert node_values.dtype == np.float32
    np.testing.assert_array_equal(
        node_values,
        np.array([1.0, np.inf, np.inf, 1.0], dtype=np.float32),
    )


def test_v438_cell_to_node_average_fills_orphans_and_nan_with_copyto(
    monkeypatch,
) -> None:
    source = inspect.getsource(cell_to_node_average)

    assert "node_values[~touched]" not in source
    assert "node_values[nan_mask]" not in source
    assert "nan_mask = np.isnan(node_values)" not in source
    assert "np.logical_not(touched, out=touched)" in source
    assert "np.isnan(node_values, out=touched)" in source
    assert "np.copyto(node_values, mean, where=touched)" in source
    assert "np.copyto(node_values, np.nan, where=touched)" in source

    original_copyto = np.copyto
    copyto_calls: list[tuple[np.dtype, tuple[int, ...], object]] = []

    def _record_copyto(dst, src, *args, where=True, **kwargs):
        copyto_calls.append((np.asarray(dst).dtype, np.asarray(where).shape, src))
        return original_copyto(dst, src, *args, where=where, **kwargs)

    monkeypatch.setattr(np, "copyto", _record_copyto)

    cells = np.array([[0, 1], [1, 2]], dtype=np.int32)
    values = np.array([1.0, np.inf], dtype=np.float32)

    node_values = cell_to_node_average(values, cells, 4)

    assert copyto_calls == [
        (np.dtype(np.float32), (4,), np.nan),
        (np.dtype(np.float32), (4,), 1.0),
    ]
    np.testing.assert_array_equal(
        node_values,
        np.array([1.0, np.inf, np.inf, 1.0], dtype=np.float32),
    )


def test_v472_mesh_helper_all_finite_reuses_chunk_work_buffer() -> None:
    source = inspect.getsource(mesh_helpers._all_finite)

    assert "np.isfinite(arr[start : start + chunk_items]).all()" not in source
    assert "np.isfinite(chunk, out=chunk_mask)" in source
    assert mesh_helpers._all_finite(np.array([1.0, 2.0], dtype=np.float32))
    assert not mesh_helpers._all_finite(np.array([1.0, np.nan], dtype=np.float32))


def test_v478_cell_to_node_average_reuses_touched_mask_for_nan_fill() -> None:
    source = inspect.getsource(cell_to_node_average)
    helper_source = inspect.getsource(mesh_helpers._finite_sum_count)

    assert "finite_mask = np.isfinite(node_values)" not in source
    assert "np.any(nan_mask)" not in source
    assert "np.isnan(node_values, out=touched)" in source
    assert "_finite_sum_count(node_values)" in source
    assert "np.isfinite(chunk, out=chunk_mask)" in helper_source

    total, count = mesh_helpers._finite_sum_count(
        np.array([1.0, np.nan, np.inf, 3.0], dtype=np.float32)
    )

    assert total == 4.0
    assert count == 2


def test_v363_cell_to_node_average_integer_fallback_uses_float32() -> None:
    source = inspect.getsource(cell_to_node_average)

    assert "dtype=np.float64" not in source

    cells = np.array([[0, 1], [1, 2]], dtype=np.int32)
    values = np.array([1, 3], dtype=np.int32)

    node_values = cell_to_node_average(values, cells, 4)

    assert node_values.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        node_values,
        np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float32),
    )


def test_v200_extract_boundary_triangles_preserves_int32_triangle_view() -> None:
    cells = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)

    triangles, sources = extract_boundary_triangles(cells, return_sources=True)

    assert triangles.dtype == np.int32
    assert np.shares_memory(triangles, cells)
    assert sources is not None
    assert sources.dtype == np.int32
    np.testing.assert_array_equal(sources, [0, 1])


def test_v383_extract_boundary_triangles_direct_fills_tetra_faces() -> None:
    cells = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=np.int32,
    )
    source = inspect.getsource(extract_boundary_triangles)

    assert "kept = [" not in source
    assert "np.asarray([face for face" not in source
    assert "np.asarray([idx for _" not in source
    assert "kept_count = sum(" in source
    assert "triangles = np.empty((kept_count, 3), dtype=np.int32)" in source

    triangles, sources = extract_boundary_triangles(cells, return_sources=True)

    assert triangles.dtype == np.dtype(np.int32)
    assert sources is not None
    assert sources.dtype == np.dtype(np.int32)
    assert len(triangles) == 6
    internal_face = np.array([1, 2, 3], dtype=np.int32)
    sorted_faces = np.sort(triangles, axis=1)
    assert not bool(np.any(np.all(sorted_faces == internal_face, axis=1)))
    np.testing.assert_array_equal(np.bincount(sources, minlength=2), [3, 3])


def test_v200_project_cells_to_triangles_preserves_int32_triangle_view() -> None:
    cells = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)
    x = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    triangles, sources = _project_cells_to_triangles(cells, x, y)

    assert triangles.dtype == np.int32
    assert np.shares_memory(triangles, cells)
    assert sources.dtype == np.int32
    np.testing.assert_array_equal(sources, [0, 1])


def test_v273_project_quad_cells_to_triangles_direct_fills_sources(
    monkeypatch,
) -> None:
    cells = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.int32)
    x = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 2.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)

    def _fail_repeat(*_args, **_kwargs):
        raise AssertionError("quad projection sources must not use np.repeat")

    monkeypatch.setattr(np, "repeat", _fail_repeat)

    triangles, sources = _project_cells_to_triangles(cells, x, y)

    assert triangles.dtype == np.int32
    assert sources.dtype == np.int32
    np.testing.assert_array_equal(
        triangles,
        np.array([[0, 1, 2], [0, 2, 3], [1, 4, 5], [1, 5, 2]], dtype=np.int32),
    )
    np.testing.assert_array_equal(sources, [0, 0, 1, 1])
    assert "np.repeat" not in inspect.getsource(_project_cells_to_triangles)


def test_v382_project_tetra_cells_direct_fills_boundary_faces() -> None:
    cells = np.array(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=np.int32,
    )
    x = np.array([0.0, 1.0, 0.0, 0.2, 1.0], dtype=np.float32)
    y = np.array([0.0, 0.0, 1.0, 0.3, 1.0], dtype=np.float32)
    z = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    source = inspect.getsource(_project_cells_to_triangles)

    assert "np.asarray([face for face" not in source
    assert "np.asarray([idx for _" not in source
    assert "kept_count = sum(" in source
    assert "triangles = np.empty((kept_count, 3), dtype=np.int32)" in source

    triangles, sources = _project_cells_to_triangles(cells, x, y, z)

    assert triangles.dtype == np.dtype(np.int32)
    assert sources.dtype == np.dtype(np.int32)
    assert len(triangles) == 6
    internal_face = np.array([1, 2, 3], dtype=np.int32)
    sorted_faces = np.sort(triangles, axis=1)
    assert not bool(np.any(np.all(sorted_faces == internal_face, axis=1)))
    np.testing.assert_array_equal(np.bincount(sources, minlength=2), [3, 3])
