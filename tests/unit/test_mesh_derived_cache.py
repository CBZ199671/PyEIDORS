"""Tests for mesh-derived HDF5 geometry cache artifacts."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from pyeidors.data.structures import EITMesh
import pyeidors.geometry.derived_cache as derived_cache_module
from pyeidors.geometry.derived_cache import (
    MESH_DERIVED_SCHEMA,
    _axis_aligned_hexa_volume_from_indices,
    _cell_centers,
    _cell_measures,
    _tetra_volume_from_indices,
    build_mesh_derived_arrays,
    load_or_build_mesh_derived_artifact,
    mesh_derived_cache_path,
    mesh_derived_signature,
    mesh_derived_signature_payload,
)


class _FakeIndexMap:
    def __init__(self, size_local: int) -> None:
        self.size_local = int(size_local)


class _FakeConnectivity:
    def __init__(self, rows: np.ndarray) -> None:
        self._rows = np.asarray(rows, dtype=np.int32)
        self.link_calls = 0

    def links(self, idx: int) -> np.ndarray:
        self.link_calls += 1
        return self._rows[int(idx)]


class _FakeTopology:
    def __init__(self, *, dim: int, cells: np.ndarray) -> None:
        self.dim = int(dim)
        self._cells = np.asarray(cells, dtype=np.int32)
        self._connectivity = _FakeConnectivity(self._cells)

    def create_connectivity(self, _from: int, _to: int) -> None:
        return None

    def connectivity(self, from_dim: int, to_dim: int):
        if (int(from_dim), int(to_dim)) == (self.dim, 0):
            return self._connectivity
        return None

    def index_map(self, dim: int):
        if int(dim) == self.dim:
            return _FakeIndexMap(self._cells.shape[0])
        return None


def _fake_mesh(coords, cells, *, dim: int):
    arr = np.asarray(coords, dtype=np.float64)
    return SimpleNamespace(
        geometry=SimpleNamespace(x=arr, dim=arr.shape[1]),
        topology=_FakeTopology(dim=dim, cells=np.asarray(cells, dtype=np.int32)),
    )


def _fake_eit_mesh(coords, cells, *, dim: int) -> EITMesh:
    return EITMesh(
        mesh=_fake_mesh(coords, cells, dim=dim),
        facet_tags=None,
        mesh_file="fake.msh",
        mesh_family="unit",
    )


def test_build_mesh_derived_arrays_for_triangle_mesh() -> None:
    mesh = _fake_mesh(
        coords=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        cells=[[0, 1, 2]],
        dim=2,
    )

    derived = build_mesh_derived_arrays(mesh)

    assert derived.metadata["artifact_schema"] == MESH_DERIVED_SCHEMA
    np.testing.assert_allclose(derived.cell_centers, [[1.0 / 3.0, 1.0 / 3.0]])
    np.testing.assert_allclose(derived.cell_measures, [0.5])


def test_v397_mesh_derived_build_reuses_extracted_arrays_for_signature() -> None:
    mesh = _fake_mesh(
        coords=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        cells=[[0, 1, 2, 3], [1, 2, 3, 4]],
        dim=3,
    )
    connectivity = mesh.topology._connectivity

    derived = build_mesh_derived_arrays(mesh)

    assert connectivity.link_calls == 2
    assert derived.metadata["signature"] == mesh_derived_signature(mesh)
    assert derived.metadata["signature_payload"]["n_cells"] == 2
    assert connectivity.link_calls == 4


def test_v592_mesh_derived_cells_use_flat_connectivity_array() -> None:
    class _FlatConnectivity:
        def __init__(self, rows: np.ndarray) -> None:
            self.rows = np.asarray(rows, dtype=np.int64)
            self.array = self.rows.reshape(-1)
            self.offsets = np.arange(
                0,
                self.array.size + 1,
                self.rows.shape[1],
                dtype=np.int64,
            )

        def links(self, idx: int) -> np.ndarray:
            raise AssertionError(f"flat connectivity path should not call links({idx})")

    class _FlatTopology:
        dim = 3

        def __init__(self, rows: np.ndarray) -> None:
            self._connectivity = _FlatConnectivity(rows)

        def create_connectivity(self, _from: int, _to: int) -> None:
            return None

        def connectivity(self, from_dim: int, to_dim: int):
            if (int(from_dim), int(to_dim)) == (3, 0):
                return self._connectivity
            return None

        def index_map(self, dim: int):
            if int(dim) == 3:
                return _FakeIndexMap(self._connectivity.rows.shape[0])
            return None

    cells = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    mesh = SimpleNamespace(
        geometry=SimpleNamespace(
            x=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=np.float32,
            ),
            dim=3,
        ),
        topology=_FlatTopology(cells),
    )

    derived = build_mesh_derived_arrays(mesh)

    assert derived.cell_connectivity.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(derived.cell_connectivity, cells.astype(np.int32))
    np.testing.assert_allclose(
        derived.cell_centers,
        [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )
    source = inspect.getsource(derived_cache_module._mesh_cells)
    assert "flat_arr[start:stop]" in source
    assert "connectivity.links(0)" in source


def test_v212_mesh_derived_centers_stream_vertex_slices(monkeypatch) -> None:
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
    cells = np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int32)
    original_take = np.take
    calls: list[tuple[tuple[int, ...], bool]] = []

    def _count_take(a, indices, axis=None, out=None, **kwargs):
        calls.append((np.asarray(indices).shape, out is not None))
        return original_take(a, indices, axis=axis, out=out, **kwargs)

    monkeypatch.setattr(np, "take", _count_take)

    centers = _cell_centers(coords, cells)

    assert centers.dtype == np.float32
    assert calls == [((2,), True), ((2,), True), ((2,), True), ((2,), True)]
    np.testing.assert_allclose(
        centers,
        [[0.25, 0.25, 0.25], [0.5, 0.5, 0.5]],
    )


def test_v212_mesh_derived_measures_do_not_materialize_full_cell_points() -> None:
    source = inspect.getsource(_cell_measures)

    assert "points = coords[cells]" not in source
    assert "coords[cells]" not in source
    assert "coords[cell]" not in source


def test_v394_mesh_derived_measures_direct_fill_without_list_staging() -> None:
    source = inspect.getsource(_cell_measures)

    assert "np.asarray(\n            [" not in source
    assert "[_polygon_area" not in source
    assert "[_polyhedron_volume" not in source
    assert "measures = np.empty((cells.shape[0],), dtype=coords.dtype)" in source

    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3], [0, 4, 2, 3]], dtype=np.int32)

    measures = _cell_measures(coords, cells, tdim=3)

    assert measures.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(measures, [1.0 / 6.0, 2.0 / 6.0])


def test_v395_mesh_derived_tetra_measures_avoid_cell_gather(
    monkeypatch,
) -> None:
    source = inspect.getsource(_cell_measures)
    helper_source = inspect.getsource(_tetra_volume_from_indices)

    assert "cells.shape[1] == 4" in source
    assert "_tetra_volume_from_indices(coords, cell)" in source
    assert "coords[cell]" not in helper_source

    def _fail_polyhedron_volume(*_args, **_kwargs):
        raise AssertionError("tetra measures must bypass polyhedron gather path")

    monkeypatch.setattr(
        derived_cache_module, "_polyhedron_volume", _fail_polyhedron_volume
    )
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3], [0, 4, 2, 3]], dtype=np.int32)

    measures = _cell_measures(coords, cells, tdim=3)

    assert measures.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(measures, [1.0 / 6.0, 2.0 / 6.0])
    assert _tetra_volume_from_indices(coords, cells[0]) == 1.0 / 6.0


def test_v396_mesh_derived_axis_aligned_hexa_measures_avoid_convex_hull(
    monkeypatch,
) -> None:
    source = inspect.getsource(_cell_measures)
    helper_source = inspect.getsource(_axis_aligned_hexa_volume_from_indices)

    assert "cells.shape[1] == 8" in source
    assert "_axis_aligned_hexa_volume_from_indices(coords, cell)" in source
    assert "coords[cell]" not in helper_source

    def _fail_polyhedron_volume(*_args, **_kwargs):
        raise AssertionError("axis-aligned hexa measures must bypass ConvexHull path")

    monkeypatch.setattr(
        derived_cache_module, "_polyhedron_volume", _fail_polyhedron_volume
    )
    coords = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
            [2.0, 0.0, 4.0],
            [0.0, 3.0, 4.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    cells = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)

    measures = _cell_measures(coords, cells, tdim=3)

    assert measures.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(measures, [24.0])
    assert _axis_aligned_hexa_volume_from_indices(coords, cells[0]) == 24.0


def test_v431_mesh_derived_fallback_measures_reuse_cell_vertex_buffer(
    monkeypatch,
) -> None:
    source = inspect.getsource(_cell_measures)
    helper_source = inspect.getsource(derived_cache_module._fill_cell_vertices)

    assert "coords[cell]" not in source
    assert "_fill_cell_vertices(cell_vertices, coords, cell)" in source
    assert "dtype=np.float64" in source
    assert "coords[cell]" not in helper_source

    polygon_buffer_ids: list[int] = []

    def _fake_polygon_area(points: np.ndarray) -> float:
        polygon_buffer_ids.append(id(points))
        assert points.dtype == np.dtype(np.float64)
        return float(len(polygon_buffer_ids))

    monkeypatch.setattr(derived_cache_module, "_polygon_area", _fake_polygon_area)
    coords_2d = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    cells_2d = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    polygon_measures = _cell_measures(coords_2d, cells_2d, tdim=2)

    assert len(set(polygon_buffer_ids)) == 1
    assert polygon_measures.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(polygon_measures, [1.0, 2.0])

    polyhedron_buffer_ids: list[int] = []
    polyhedron_shapes: list[tuple[int, ...]] = []

    def _force_polyhedron_fallback(*_args, **_kwargs) -> float:
        return float("nan")

    def _fake_polyhedron_volume(points: np.ndarray) -> float:
        polyhedron_buffer_ids.append(id(points))
        polyhedron_shapes.append(points.shape)
        assert points.dtype == np.dtype(np.float64)
        return float(len(polyhedron_buffer_ids))

    monkeypatch.setattr(
        derived_cache_module,
        "_axis_aligned_hexa_volume_from_indices",
        _force_polyhedron_fallback,
    )
    monkeypatch.setattr(
        derived_cache_module, "_polyhedron_volume", _fake_polyhedron_volume
    )
    coords_3d = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.1],
            [1.0, 0.0, 1.2],
            [0.0, 1.0, 1.4],
            [1.2, 1.0, 1.0],
            [0.4, 0.3, 1.8],
        ],
        dtype=np.float32,
    )
    hexa_cells = np.asarray(
        [[0, 1, 2, 4, 3, 5, 6, 7], [1, 2, 4, 8, 5, 6, 7, 3]],
        dtype=np.int32,
    )
    generic_cells = np.asarray([[0, 1, 2, 3, 4], [1, 2, 3, 4, 8]], dtype=np.int32)

    hexa_measures = _cell_measures(coords_3d, hexa_cells, tdim=3)
    generic_measures = _cell_measures(coords_3d, generic_cells, tdim=3)

    assert polyhedron_shapes == [(8, 3), (8, 3), (5, 3), (5, 3)]
    assert len(set(polyhedron_buffer_ids[:2])) == 1
    assert len(set(polyhedron_buffer_ids[2:])) == 1
    assert hexa_measures.dtype == np.dtype(np.float32)
    assert generic_measures.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(hexa_measures, [1.0, 2.0])
    np.testing.assert_allclose(generic_measures, [3.0, 4.0])


def test_v296_tetra_volume_direct_fills_determinant_matrix(monkeypatch) -> None:
    def _fail_vstack(*_args, **_kwargs):
        raise AssertionError("tetra volume must not call np.vstack")

    monkeypatch.setattr(derived_cache_module.np, "vstack", _fail_vstack)
    source = inspect.getsource(derived_cache_module._tetra_volume)
    assert "np.vstack" not in source

    volume = derived_cache_module._tetra_volume(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
    )
    assert volume == 1.0 / 6.0


def test_mesh_derived_signature_changes_with_connectivity() -> None:
    mesh_a = _fake_mesh(
        coords=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        cells=[[0, 1, 2]],
        dim=2,
    )
    mesh_b = _fake_mesh(
        coords=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        cells=[[0, 2, 1]],
        dim=2,
    )

    assert mesh_derived_signature(mesh_a) != mesh_derived_signature(mesh_b)
    payload = mesh_derived_signature_payload(mesh_a)
    assert payload["n_cells"] == 1
    assert payload["vertices_per_cell"] == 3


def test_load_or_build_mesh_derived_artifact_round_trips_hdf5(tmp_path) -> None:
    mesh = _fake_mesh(
        coords=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        cells=[[0, 1, 2, 3]],
        dim=3,
    )

    first = load_or_build_mesh_derived_artifact(mesh, cache_dir=tmp_path)
    second = load_or_build_mesh_derived_artifact(mesh, cache_dir=tmp_path)

    assert first.path == second.path
    assert mesh_derived_cache_path(tmp_path, mesh_derived_signature(mesh)).exists()
    np.testing.assert_allclose(second.cell_centers, [[0.25, 0.25, 0.25]])
    np.testing.assert_allclose(second.cell_measures, [1.0 / 6.0])


def test_mesh_derived_artifact_stores_electrodes_and_visual_point_cloud(
    tmp_path,
) -> None:
    mesh = _fake_eit_mesh(
        coords=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        cells=[[0, 1, 2, 3]],
        dim=3,
    )
    mesh.electrode_vertices = [
        np.asarray([[1.0, 0.0, -0.2], [1.0, 0.1, -0.2]], dtype=np.float64),
        np.asarray([[0.0, 1.0, 0.2], [0.1, 1.0, 0.2]], dtype=np.float64),
    ]

    derived = load_or_build_mesh_derived_artifact(mesh, cache_dir=tmp_path)

    assert derived.electrode_vertices is not None
    assert derived.electrode_vertex_offsets is not None
    np.testing.assert_allclose(
        derived.electrode_vertices,
        [[1.0, 0.0, -0.2], [1.0, 0.1, -0.2], [0.0, 1.0, 0.2], [0.1, 1.0, 0.2]],
    )
    np.testing.assert_array_equal(derived.electrode_vertex_offsets, [0, 2, 4])
    np.testing.assert_allclose(derived.visual_point_cloud_centers, derived.cell_centers)
    np.testing.assert_array_equal(derived.visual_point_cloud_cell_indices, [0])
    assert derived.metadata["electrode_patch_count"] == 2
    assert derived.metadata["visual_point_cloud_count"] == 1


def test_eit_mesh_reuses_process_derived_arrays_for_cells_and_centers() -> None:
    eit_mesh = _fake_eit_mesh(
        coords=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        cells=[[0, 1, 2]],
        dim=2,
    )
    connectivity = eit_mesh.mesh.topology._connectivity

    np.testing.assert_array_equal(eit_mesh.cells(), [[0, 1, 2]])
    np.testing.assert_allclose(eit_mesh.cell_centers(), [[1.0 / 3.0, 1.0 / 3.0]])
    np.testing.assert_allclose(eit_mesh.cell_measures(), [0.5])
    calls_after_first_build = connectivity.link_calls
    np.testing.assert_array_equal(eit_mesh.cells(), [[0, 1, 2]])
    np.testing.assert_allclose(eit_mesh.cell_centers(), [[1.0 / 3.0, 1.0 / 3.0]])

    assert connectivity.link_calls == calls_after_first_build
