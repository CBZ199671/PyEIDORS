"""Tests for dual-mesh coarse-to-fine projection helpers."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import pyeidors.inverse.dual_mesh as dual_mesh_module
from pyeidors.inverse.dual_mesh import CellMesh, DualMesh, VoxelGrid, coarse2fine


def _cell_mesh_from_centers(centers: np.ndarray, *, name: str = "fine") -> CellMesh:
    centers = np.asarray(centers, dtype=float)
    coords: list[np.ndarray] = []
    cells: list[list[int]] = []
    eps = 1e-3
    if centers.shape[1] == 2:
        offsets = np.array([[-eps, -eps], [eps, 0.0], [0.0, eps]], dtype=float)
    elif centers.shape[1] == 3:
        offsets = np.array(
            [
                [-eps, -eps, -eps],
                [eps, 0.0, 0.0],
                [0.0, eps, 0.0],
                [0.0, 0.0, eps],
            ],
            dtype=float,
        )
    else:
        raise ValueError("test helper only supports 2D or 3D centers")

    for center in centers:
        start = len(coords)
        coords.extend(center + offsets)
        cells.append(list(range(start, start + offsets.shape[0])))
    return CellMesh(np.asarray(coords, dtype=float), np.asarray(cells), name=name)


def test_v211_cell_mesh_centers_stream_vertex_slices(monkeypatch) -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    mesh = CellMesh(coords, cells, name="streamed-centers")
    original_take = np.take
    calls: list[tuple[tuple[int, ...], bool]] = []

    def _count_take(a, indices, axis=None, out=None, **kwargs):
        calls.append((np.asarray(indices).shape, out is not None))
        return original_take(a, indices, axis=axis, out=out, **kwargs)

    monkeypatch.setattr(dual_mesh_module.np, "take", _count_take)

    centers = mesh.cell_centers()

    assert calls == [((2,), True), ((2,), True), ((2,), True), ((2,), True)]
    np.testing.assert_allclose(
        centers,
        np.array(
            [
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        ),
    )


def test_v213_generic_cell_centers_stream_vertex_slices(monkeypatch) -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    mesh = SimpleNamespace(coordinates=coords, cells=cells)
    original_take = np.take
    calls: list[tuple[tuple[int, ...], bool]] = []

    def _count_take(a, indices, axis=None, out=None, **kwargs):
        calls.append((np.asarray(indices).shape, out is not None))
        return original_take(a, indices, axis=axis, out=out, **kwargs)

    monkeypatch.setattr(dual_mesh_module.np, "take", _count_take)

    centers = dual_mesh_module._cell_centers(mesh)

    assert calls == [((2,), True), ((2,), True), ((2,), True), ((2,), True)]
    np.testing.assert_allclose(
        centers,
        np.array(
            [
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        ),
    )


def test_v214_cell_mesh_locator_has_no_full_cell_vertices_expansion() -> None:
    source = inspect.getsource(dual_mesh_module._locate_points_in_cell_mesh)

    assert "cell_vertices = coords[cells]" not in source
    assert "cell_vertices[cell_idx]" not in source
    assert "np.concatenate" not in source


def test_v399_cell_mesh_locator_reuses_1d_bbox_candidate_mask() -> None:
    source = inspect.getsource(dual_mesh_module._locate_points_in_cell_mesh)
    helper_source = inspect.getsource(dual_mesh_module._bbox_candidate_mask)

    assert "np.all((point >= mins) & (point <= maxs), axis=1)" not in source
    assert "_bbox_candidate_mask(point, mins, maxs, candidate_mask)" in source
    assert "candidate_mask = np.empty(cells.shape[0], dtype=bool)" in source
    assert "where=out" in helper_source

    mins = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], dtype=float)
    maxs = np.array([[0.5, 0.5], [2.0, 2.0], [-0.5, -0.5]], dtype=float)
    mask = np.empty(3, dtype=bool)

    result = dual_mesh_module._bbox_candidate_mask(
        np.array([0.25, 0.25], dtype=float),
        mins,
        maxs,
        mask,
    )

    assert result is mask
    np.testing.assert_array_equal(mask, [True, False, False])


def test_v400_cell_mesh_locator_iterates_candidate_mask_without_flatnonzero(
    monkeypatch,
) -> None:
    source = inspect.getsource(dual_mesh_module._locate_points_in_cell_mesh)

    assert "np.flatnonzero(candidate_mask)" not in source
    assert "for cell_idx, is_candidate in enumerate(candidate_mask)" in source

    def _fail_flatnonzero(*_args, **_kwargs):
        raise AssertionError("locator must not allocate candidate index arrays")

    monkeypatch.setattr(dual_mesh_module.np, "flatnonzero", _fail_flatnonzero)
    coarse = CellMesh(
        coordinates=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        cells=np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=int),
        name="coarse-tetra",
    )
    fine = _cell_mesh_from_centers(
        np.array([[0.1, 0.1, 0.1], [1.2, 0.1, 0.1]], dtype=float),
        name="fine-tetra",
    )

    projection = coarse2fine(fine, coarse)

    np.testing.assert_allclose(
        projection.toarray(),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )


def test_v401_cell_mesh_locator_reuses_candidate_vertex_buffer(monkeypatch) -> None:
    source = inspect.getsource(dual_mesh_module._locate_points_in_cell_mesh)
    helper_source = inspect.getsource(dual_mesh_module._fill_cell_vertices)

    assert "vertices = coords[cells" not in source
    assert "coords[cells[cell_idx]]" not in source
    assert "_fill_cell_vertices(coords, cells[cell_idx], candidate_vertices)" in source
    assert (
        "candidate_vertices = np.empty((cells.shape[1], coords.shape[1]),"
        " dtype=coords.dtype)" in source
    )
    assert "coords[cell" not in helper_source

    seen_vertex_buffers: list[int] = []
    original_point_in_simplex = dual_mesh_module._point_in_simplex

    def _capture_point_in_simplex(point, vertices, *, tol):
        seen_vertex_buffers.append(id(vertices))
        return original_point_in_simplex(point, vertices, tol=tol)

    monkeypatch.setattr(
        dual_mesh_module,
        "_point_in_simplex",
        _capture_point_in_simplex,
    )
    coarse = CellMesh(
        coordinates=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        cells=np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=int),
        name="coarse-tetra",
    )
    fine = _cell_mesh_from_centers(
        np.array([[0.1, 0.1, 0.1], [1.2, 0.1, 0.1]], dtype=float),
        name="fine-tetra",
    )

    projection = coarse2fine(fine, coarse)

    assert len(seen_vertex_buffers) >= 2
    assert len(set(seen_vertex_buffers)) == 1
    np.testing.assert_allclose(
        projection.toarray(),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )


def test_v215_voxel_grid_cell_centers_avoid_meshgrid() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 3.0, 4.0],
        shape=(2, 3, 4),
    )

    centers = grid.cell_centers()

    source = inspect.getsource(VoxelGrid.cell_centers)
    assert "meshgrid" not in source
    assert "np.stack" not in source
    assert "np.repeat" not in source
    assert "np.tile" not in source
    assert "_fill_repeated_axis_column" in source
    helper_source = inspect.getsource(dual_mesh_module._fill_repeated_axis_column)
    assert "as_strided" in helper_source
    assert centers.shape == (24, 3)
    np.testing.assert_allclose(
        centers[:5],
        [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 1.5],
            [0.5, 0.5, 2.5],
            [0.5, 0.5, 3.5],
            [0.5, 1.5, 0.5],
        ],
    )
    np.testing.assert_allclose(centers[-1], [1.5, 2.5, 3.5])


def test_v492_dual_mesh_array_validators_use_bounded_scans() -> None:
    coord_source = inspect.getsource(dual_mesh_module._as_coordinates)
    cells_source = inspect.getsource(dual_mesh_module._as_cells)
    grid_source = inspect.getsource(VoxelGrid.__post_init__)

    assert "all_finite_values(coords)" in coord_source
    assert "np.isfinite(coords).all()" not in coord_source
    assert "np.min(cells)" in cells_source
    assert "np.any(cells < 0)" not in cells_source
    assert "all_finite_values(self.origin)" in grid_source
    assert "all_finite_values(self.spacing)" in grid_source
    assert "np.any(~np.isfinite(self.origin))" not in grid_source
    assert "np.any(~np.isfinite(self.spacing))" not in grid_source
    assert "np.any(self.spacing <= 0.0)" not in grid_source


def test_v445_voxel_grid_locate_points_direct_fills_inside_and_outside_rows() -> None:
    grid = VoxelGrid.from_bounds(
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        shape=(2, 2, 2),
    )
    points = np.array(
        [
            [0.25, 0.25, 0.25],
            [1.25, 0.25, 0.25],
            [3.0, 0.25, 0.25],
        ],
        dtype=float,
    )

    indices = grid.locate_points(points, outside="nearest")

    np.testing.assert_array_equal(indices[:2], np.array([0, 4], dtype=np.int64))
    assert int(indices[2]) in {4, 5, 6, 7}
    source = inspect.getsource(VoxelGrid.locate_points)
    assert "(scaled >= 0) & (scaled < upper)" not in source
    assert "np.all(" not in source
    assert "scaled[inside]" not in source
    assert "pts[~inside]" not in source
    assert "indices[~inside]" not in source
    assert "_inside_scaled_rows(scaled, upper)" in source
    assert "_ravel_scaled_rows(scaled, self.shape)" in source
    assert "_compact_rows_where(" in source
    assert "pts, outside_mask, count=outside_count" in source
    assert "_scatter_values_where(" in source


def test_v446_voxel_grid_scaled_point_indices_reuses_float_work_buffer() -> None:
    pts = np.array([[0.25, 1.25], [1.99, -0.1]], dtype=float)
    origin = np.array([0.0, 0.0], dtype=float)
    spacing = np.array([1.0, 0.5], dtype=float)

    scaled = dual_mesh_module._scaled_point_indices(pts, origin, spacing)

    np.testing.assert_array_equal(scaled, np.array([[0, 2], [1, -1]], dtype=np.int64))
    helper_source = inspect.getsource(dual_mesh_module._scaled_point_indices)
    locate_source = inspect.getsource(VoxelGrid.locate_points)
    assert "np.subtract(pts, origin, out=scaled_work)" in helper_source
    assert "np.divide(scaled_work, spacing, out=scaled_work)" in helper_source
    assert "np.floor(scaled_work, out=scaled_work)" in helper_source
    assert "np.floor((pts - self.origin) / self.spacing)" not in locate_source
    assert "_scaled_point_indices(pts, self.origin, self.spacing)" in locate_source


def test_v451_voxel_grid_outside_count_reused_for_row_compaction(monkeypatch) -> None:
    source = inspect.getsource(VoxelGrid.locate_points)
    helper_source = inspect.getsource(dual_mesh_module._compact_rows_where)

    assert "np.any(outside_mask)" not in source
    assert "outside_count = int(np.count_nonzero(outside_mask))" in source
    assert "_compact_rows_where(pts, outside_mask, count=outside_count)" in source
    assert "count: int | None = None" in helper_source

    def _fail_count_nonzero(*_args, **_kwargs):
        raise AssertionError("caller should provide known row count")

    monkeypatch.setattr(dual_mesh_module.np, "count_nonzero", _fail_count_nonzero)
    rows = np.arange(12, dtype=np.float64).reshape(4, 3)
    mask = np.array([False, True, False, True], dtype=bool)

    compact = dual_mesh_module._compact_rows_where(rows, mask, count=2)

    np.testing.assert_array_equal(compact, rows[[1, 3]])


def test_coarse2fine_maps_fine_cells_to_voxel_cells() -> None:
    fine = _cell_mesh_from_centers(
        np.array(
            [
                [0.25, 0.25],
                [1.25, 0.25],
                [0.25, 1.25],
                [1.25, 1.25],
            ],
            dtype=float,
        )
    )
    coarse = VoxelGrid(
        origin=np.array([0.0, 0.0]), spacing=np.array([1.0, 1.0]), shape=(2, 2)
    )

    projection = coarse2fine(fine, coarse)

    assert sparse.isspmatrix_csr(projection)
    assert projection.shape == (4, 4)
    assert projection.nnz == 4
    np.testing.assert_allclose(
        projection.toarray(),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )


def test_coarse2fine_uses_tetra_containment_before_nearest() -> None:
    coarse = CellMesh(
        coordinates=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        cells=np.array([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=int),
        name="coarse-tetra",
    )
    fine = _cell_mesh_from_centers(
        np.array([[0.1, 0.1, 0.1], [1.2, 0.1, 0.1]], dtype=float),
        name="fine-tetra",
    )

    projection = coarse2fine(fine, coarse)

    np.testing.assert_allclose(
        projection.toarray(),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float),
    )


def test_coarse2fine_falls_back_to_nearest_or_raises_for_outside_points() -> None:
    fine = _cell_mesh_from_centers(np.array([[3.0, 0.5]], dtype=float))
    coarse = VoxelGrid(
        origin=np.array([0.0, 0.0]), spacing=np.array([1.0, 1.0]), shape=(2, 1)
    )

    projection = coarse2fine(fine, coarse)
    np.testing.assert_allclose(
        projection.toarray(), np.array([[0.0, 1.0]], dtype=float)
    )

    with pytest.raises(ValueError, match="outside the coarse voxel grid"):
        coarse2fine(fine, coarse, outside="raise")


def test_dual_mesh_projects_and_restricts_cell_values() -> None:
    fine = _cell_mesh_from_centers(
        np.array([[0.25, 0.25], [1.25, 0.25], [1.25, 0.75]], dtype=float)
    )
    coarse = VoxelGrid(
        origin=np.array([0.0, 0.0]), spacing=np.array([1.0, 1.0]), shape=(2, 1)
    )
    dual = DualMesh(fine_mesh=fine, coarse_mesh=coarse)

    assert dual.summary() == {
        "n_fine_cells": 3,
        "n_coarse_cells": 2,
        "projection_nnz": 3,
        "method": "piecewise_constant",
        "outside": "nearest",
    }
    np.testing.assert_allclose(
        dual.project_to_fine(np.array([10.0, 20.0])), [10.0, 20.0, 20.0]
    )
    np.testing.assert_allclose(
        dual.restrict_to_coarse(np.array([1.0, 3.0, 5.0])), [1.0, 4.0]
    )


def test_dual_mesh_validates_projection_shape_and_value_lengths() -> None:
    fine = _cell_mesh_from_centers(np.array([[0.25, 0.25], [1.25, 0.25]], dtype=float))
    coarse = VoxelGrid(
        origin=np.array([0.0, 0.0]), spacing=np.array([1.0, 1.0]), shape=(2, 1)
    )

    with pytest.raises(ValueError, match="projection shape mismatch"):
        DualMesh(fine, coarse, projection=sparse.eye(1, 2, format="csr"))

    dual = DualMesh(fine, coarse)
    with pytest.raises(ValueError, match="Expected 2 coarse values"):
        dual.project_to_fine(np.ones(1))
    with pytest.raises(ValueError, match="Expected 2 fine values"):
        dual.restrict_to_coarse(np.ones(1))
