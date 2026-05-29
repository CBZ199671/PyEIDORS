"""Contracts for shared graph-operator core consolidation."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from pyeidors.inverse.dual_mesh import VoxelGrid
from pyeidors.inverse.prior import _graph_core as graph_core_module
from pyeidors.inverse.prior import laplace as laplace_module
from pyeidors.inverse.prior._graph_core import (
    cell_volumes,
    difference_from_edges,
    dolfinx_cell_difference_operator,
    graph_edges_and_volumes,
    laplacian_from_edges,
)
from pyeidors.inverse.prior.laplace import (
    graph_difference_operator,
    graph_laplacian,
)
from pyeidors.inverse.regularization import smoothness as smooth_module


class _FakeIndexMap:
    def __init__(self, size_local: int):
        self.size_local = int(size_local)


class _FakeConnectivity:
    def __init__(self, links_map):
        self._links_map = {
            int(key): np.asarray(value, dtype=np.int32)
            for key, value in links_map.items()
        }

    def links(self, idx: int):
        return self._links_map[int(idx)]


class _FakeTopology:
    def __init__(self, *, dim: int, facet_size: int | None, connectivity):
        self.dim = int(dim)
        self._facet_size = facet_size
        self._connectivity = connectivity

    def create_connectivity(self, _from: int, _to: int) -> None:
        return None

    def connectivity(self, _from: int, _to: int):
        return self._connectivity

    def index_map(self, dim: int):
        if int(dim) != self.dim - 1 or self._facet_size is None:
            return None
        return _FakeIndexMap(self._facet_size)


def _fake_mesh(*, facet_size: int | None, links_map):
    return SimpleNamespace(
        topology=_FakeTopology(
            dim=2,
            facet_size=facet_size,
            connectivity=None if links_map is None else _FakeConnectivity(links_map),
        )
    )


def test_graph_core_matches_voxel_laplace_and_ltl_contract() -> None:
    grid = VoxelGrid.from_bounds([0.0, 0.0], [2.0, 2.0], shape=(2, 2))

    n_cells, edges, volumes = graph_edges_and_volumes(grid)
    difference = difference_from_edges(n_cells, edges, volumes=volumes, weight="unit")
    laplace = laplacian_from_edges(n_cells, edges, volumes=volumes, weight="unit")

    assert n_cells == grid.num_cells()
    np.testing.assert_allclose(
        difference.toarray(), graph_difference_operator(grid).toarray()
    )
    np.testing.assert_allclose(
        (2.0 * laplace).toarray(),
        graph_laplacian(grid).toarray(),
    )
    np.testing.assert_allclose((difference.T @ difference).toarray(), laplace.toarray())


def test_dolfinx_cell_difference_wrapper_matches_shared_core() -> None:
    mesh = _fake_mesh(facet_size=3, links_map={0: [0, 1], 1: [1], 2: [1, 2]})

    core = dolfinx_cell_difference_operator(mesh, 3)
    wrapper = smooth_module._cell_difference_operator(mesh, 3)

    expected = np.array(
        [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]],
        dtype=float,
    )
    np.testing.assert_allclose(core.toarray(), expected)
    np.testing.assert_allclose(wrapper.toarray(), expected)


def test_dolfinx_cell_difference_core_keeps_empty_edge_shape() -> None:
    missing_connectivity = _fake_mesh(facet_size=1, links_map=None)
    missing_index_map = _fake_mesh(facet_size=None, links_map={0: [0, 1]})
    boundary_only = _fake_mesh(facet_size=2, links_map={0: [0], 1: [1]})

    assert dolfinx_cell_difference_operator(missing_connectivity, 3).shape == (0, 3)
    assert dolfinx_cell_difference_operator(missing_index_map, 3).shape == (0, 3)
    assert dolfinx_cell_difference_operator(boundary_only, 3).shape == (0, 3)


def test_laplace_and_regularization_wrappers_are_thin_facades() -> None:
    laplace_source = inspect.getsource(
        laplace_module.graph_laplacian
    ) + inspect.getsource(laplace_module.graph_difference_operator)
    smooth_source = inspect.getsource(smooth_module._cell_difference_operator)

    assert "graph_edges_and_volumes" in laplace_source
    assert "laplacian_from_edges" in laplace_source
    assert "difference_from_edges" in laplace_source
    assert "np.ndindex" not in laplace_source
    assert "_cell_edges_from_shared_facets" not in laplace_source

    assert "dolfinx_cell_difference_operator" in smooth_source
    assert "rows.extend" not in smooth_source
    assert "facet_to_cell" not in smooth_source


def test_v501_graph_core_negative_cell_guard_uses_min_reduction() -> None:
    source = inspect.getsource(graph_core_module.extract_cells)

    assert "np.min(cells, initial=0)" in source
    assert "np.any(cells < 0)" not in source


def test_v432_graph_core_cell_volumes_reuse_simplex_work_buffers(monkeypatch) -> None:
    source = inspect.getsource(cell_volumes)
    fill_source = inspect.getsource(graph_core_module._fill_cell_vertices)
    gram_source = inspect.getsource(graph_core_module._simplex_gram_from_vertices)

    assert "coords[cell]" not in source
    assert "vertices[1:] - vertices[0]" not in source
    assert "_fill_cell_vertices(cell_vertices, coords, cell)" in source
    assert "np.matmul(active_basis.T, active_basis, out=active_gram)" in gram_source
    assert "coords[cell]" not in fill_source

    original_fill = graph_core_module._fill_cell_vertices
    fill_ids: list[int] = []

    def _record_fill(out: np.ndarray, coords: np.ndarray, cell: np.ndarray) -> None:
        fill_ids.append(id(out))
        original_fill(out, coords, cell)

    original_det = graph_core_module.np.linalg.det
    gram_base_ids: list[int] = []

    def _record_det(arr: np.ndarray) -> float:
        base = arr
        while getattr(base, "base", None) is not None:
            base = base.base
        gram_base_ids.append(id(base))
        return float(original_det(arr))

    monkeypatch.setattr(graph_core_module, "_fill_cell_vertices", _record_fill)
    monkeypatch.setattr(graph_core_module.np.linalg, "det", _record_det)

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
    mesh = SimpleNamespace(coordinates=lambda: coords)

    volumes = cell_volumes(mesh, cells, n_cells=2)

    assert len(set(fill_ids)) == 1
    assert len(set(gram_base_ids)) == 1
    np.testing.assert_allclose(volumes, [1.0, 2.0])
