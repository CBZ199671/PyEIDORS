"""Tests for GUI array-derived geometry process cache."""

from __future__ import annotations

import inspect

import numpy as np

import eit_app.ui.array_geometry_cache as array_geometry_cache_module
from eit_app.ui.array_geometry_cache import (
    ARRAY_GEOMETRY_CACHE_SCHEMA,
    _compute_cell_centers,
    array_geometry_cache_entries,
    array_geometry_cache_snapshot,
    array_geometry_cache_stats,
    array_geometry_signature,
    cached_cell_centers,
    clear_array_geometry_cache,
)


def test_array_geometry_signature_streams_payload_without_tobytes_copy() -> None:
    source = inspect.getsource(array_geometry_cache_module._hash_array_into)
    source += inspect.getsource(array_geometry_cache_module._signature_for_arrays)
    assert "update_digest_with_array_payload" in source
    assert ".tobytes(" not in source


def test_array_geometry_cache_hashes_noncontiguous_views_without_local_copy() -> None:
    clear_array_geometry_cache()
    coords_wide = np.asarray(
        [
            [0.0, 0.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 11.0],
            [0.0, 1.0, 0.0, 12.0],
            [0.0, 0.0, 1.0, 13.0],
        ],
        dtype=np.float32,
    )
    cells_wide = np.asarray([[0, 99, 1, 99, 2, 99, 3, 99]], dtype=np.int32)
    coords_view = coords_wide[:, :3]
    cells_view = cells_wide[:, ::2]

    assert not coords_view.flags.c_contiguous
    assert not cells_view.flags.c_contiguous

    first = cached_cell_centers(coords_view, cells_view, coordinate_dims=3)
    second = cached_cell_centers(
        np.ascontiguousarray(coords_view),
        np.ascontiguousarray(cells_view),
        coordinate_dims=3,
    )

    assert first is second
    np.testing.assert_allclose(first, [[0.25, 0.25, 0.25]])
    source = inspect.getsource(array_geometry_cache_module._as_hashable_arrays)
    assert "np.ascontiguousarray(coords_raw" not in source
    assert "np.ascontiguousarray(cells_raw" not in source


def test_cached_cell_centers_reuses_same_geometry_content() -> None:
    clear_array_geometry_cache()
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)

    first = cached_cell_centers(coords, cells, coordinate_dims=3)
    second = cached_cell_centers(coords.copy(), cells.copy(), coordinate_dims=3)

    assert first is second
    assert first is not None
    np.testing.assert_allclose(first, [[0.25, 0.25, 0.25]])
    assert not first.flags.writeable
    stats = array_geometry_cache_stats()
    assert stats["items"] == 1
    assert stats["misses"] == 1
    assert stats["hits"] == 1


def test_cached_cell_centers_preserves_integer_connectivity_dtype() -> None:
    clear_array_geometry_cache()
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)

    centers = cached_cell_centers(coords, cells, coordinate_dims=3)
    entries = array_geometry_cache_entries()

    assert centers is not None
    assert entries[0]["node_dtype"] == "float32"
    assert entries[0]["cell_dtype"] == "int32"


def test_compute_cell_centers_streams_vertex_slices(monkeypatch) -> None:
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=np.int32,
    )
    original_take = np.take
    take_shapes: list[tuple[int, ...]] = []

    def _counted_take(array, indices, *args, **kwargs):
        take_shapes.append(tuple(np.asarray(indices).shape))
        return original_take(array, indices, *args, **kwargs)

    monkeypatch.setattr(np, "take", _counted_take)

    centers = _compute_cell_centers(coords, cells)

    assert centers.dtype == np.float32
    assert take_shapes == [(2,), (2,), (2,), (2,)]
    assert not centers.flags.writeable
    np.testing.assert_allclose(
        centers,
        [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )


def test_array_geometry_signature_changes_after_input_mutation() -> None:
    clear_array_geometry_cache()
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.asarray([[0, 1, 2]], dtype=np.int64)

    signature_before = array_geometry_signature(coords, cells)
    center_before = cached_cell_centers(coords, cells)
    coords[1, 0] = 2.0
    signature_after = array_geometry_signature(coords, cells)
    center_after = cached_cell_centers(coords, cells)

    assert signature_before != signature_after
    np.testing.assert_allclose(center_before, [[1.0 / 3.0, 1.0 / 3.0]])
    np.testing.assert_allclose(center_after, [[2.0 / 3.0, 1.0 / 3.0]])
    assert array_geometry_cache_stats()["items"] == 2


def test_cached_cell_centers_rejects_invalid_connectivity() -> None:
    clear_array_geometry_cache()
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    cells = np.asarray([[0, 3]], dtype=np.int64)

    assert cached_cell_centers(coords, cells) is None
    stats = array_geometry_cache_stats()
    assert stats["invalid"] == 1
    assert stats["items"] == 0


def test_array_geometry_cache_snapshot_reports_json_safe_entry_metadata() -> None:
    clear_array_geometry_cache()
    coords = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells = np.asarray([[0, 1, 2]], dtype=np.int32)

    cached_cell_centers(coords, cells)

    entries = array_geometry_cache_entries()
    assert len(entries) == 1
    assert entries[0]["node_shape"] == (3, 2)
    assert entries[0]["cell_shape"] == (1, 3)
    snapshot = array_geometry_cache_snapshot()
    assert snapshot["schema"] == ARRAY_GEOMETRY_CACHE_SCHEMA
    assert snapshot["process_local"] is True
    assert snapshot["stats"]["items"] == 1
    assert snapshot["entries"][0]["signature_prefix"]


def test_array_geometry_cache_does_not_retain_oversize_centers(monkeypatch) -> None:
    clear_array_geometry_cache()
    monkeypatch.setenv("EIT_APP_ARRAY_GEOMETRY_CACHE_MAX_BYTES", "4")
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)

    first = cached_cell_centers(coords, cells, coordinate_dims=3)
    second = cached_cell_centers(coords, cells, coordinate_dims=3)

    assert first is not None
    assert second is not None
    assert first is not second
    stats = array_geometry_cache_stats()
    assert stats["items"] == 0
    assert stats["bytes"] == 0
    assert stats["oversize"] == 2
    assert stats["misses"] == 2


def test_array_geometry_cache_eviction_obeys_byte_budget(monkeypatch) -> None:
    clear_array_geometry_cache()
    monkeypatch.setenv("EIT_APP_ARRAY_GEOMETRY_CACHE_MAX_BYTES", "16")
    cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    coords_a = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    coords_b = coords_a.copy()
    coords_b[1, 0] = 2.0

    cached_cell_centers(coords_a, cells, coordinate_dims=3)
    cached_cell_centers(coords_b, cells, coordinate_dims=3)

    stats = array_geometry_cache_stats()
    entries = array_geometry_cache_entries()
    assert stats["items"] == 1
    assert stats["bytes"] <= stats["max_bytes"]
    assert stats["evictions"] == 1
    assert len(entries) == 1
