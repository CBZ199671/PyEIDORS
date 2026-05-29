"""Edge-branch tests for femx helper utilities."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

import pyeidors.femx.helpers as helpers_module
from pyeidors.femx.helpers import (
    cell_midpoints,
    estimate_radius,
    mesh_cell_vertices,
    mesh_facet_vertices,
)


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
    def __init__(
        self, *, dim: int, index_maps: dict[int, _FakeIndexMap | None], connectivities
    ):
        self.dim = int(dim)
        self._index_maps = index_maps
        self._connectivities = connectivities

    def create_entities(self, _dim: int) -> None:
        return None

    def create_connectivity(self, _from: int, _to: int) -> None:
        return None

    def index_map(self, dim: int):
        return self._index_maps.get(int(dim))

    def connectivity(self, from_dim: int, to_dim: int):
        return self._connectivities.get((int(from_dim), int(to_dim)))


def _fake_mesh(*, coords, topology: _FakeTopology):
    return SimpleNamespace(
        geometry=SimpleNamespace(
            x=np.asarray(coords, dtype=float),
            dim=np.asarray(coords, dtype=float).shape[1],
        ),
        topology=topology,
    )


def test_mesh_cell_vertices_and_midpoints_cover_empty_connectivity_paths():
    none_topology = _FakeTopology(
        dim=2, index_maps={2: _FakeIndexMap(1)}, connectivities={(2, 0): None}
    )
    none_mesh = _fake_mesh(coords=[[0.0, 0.0]], topology=none_topology)
    assert mesh_cell_vertices(none_mesh).shape == (0, 0)

    zero_topology = _FakeTopology(
        dim=2,
        index_maps={2: _FakeIndexMap(0)},
        connectivities={(2, 0): _FakeConnectivity({0: [0, 1, 2]})},
    )
    zero_mesh = _fake_mesh(
        coords=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], topology=zero_topology
    )
    assert mesh_cell_vertices(zero_mesh).shape == (0, 0)
    assert cell_midpoints(zero_mesh).shape == (0, 2)


def test_mesh_facet_vertices_and_radius_cover_empty_cases():
    none_conn_topology = _FakeTopology(
        dim=2, index_maps={1: _FakeIndexMap(1)}, connectivities={(1, 0): None}
    )
    none_conn_mesh = _fake_mesh(coords=[[0.0, 0.0]], topology=none_conn_topology)
    assert mesh_facet_vertices(none_conn_mesh).shape == (0, 0)

    none_index_topology = _FakeTopology(
        dim=2,
        index_maps={1: None},
        connectivities={(1, 0): _FakeConnectivity({0: [0, 1]})},
    )
    none_index_mesh = _fake_mesh(
        coords=[[0.0, 0.0], [1.0, 0.0]], topology=none_index_topology
    )
    assert mesh_facet_vertices(none_index_mesh).shape == (0, 0)

    zero_index_topology = _FakeTopology(
        dim=2,
        index_maps={1: _FakeIndexMap(0)},
        connectivities={(1, 0): _FakeConnectivity({0: [0, 1]})},
    )
    zero_index_mesh = _fake_mesh(
        coords=[[0.0, 0.0], [1.0, 0.0]], topology=zero_index_topology
    )
    assert mesh_facet_vertices(zero_index_mesh).shape == (0, 0)

    empty_radius_mesh = SimpleNamespace(
        geometry=SimpleNamespace(x=np.zeros((0, 2), dtype=float), dim=2),
        topology=zero_index_topology,
    )
    assert estimate_radius(empty_radius_mesh) == 0.0


def test_v289_cell_midpoints_and_radius_avoid_expanded_coordinate_temporaries():
    source_midpoints = inspect.getsource(helpers_module.cell_midpoints)
    source_radius = inspect.getsource(helpers_module.estimate_radius)
    assert "coords[c2v]" not in source_midpoints
    assert "np.linalg.norm" not in source_radius
    assert "squared_distances_to_point" in source_radius

    topology = _FakeTopology(
        dim=2,
        index_maps={2: _FakeIndexMap(2)},
        connectivities={(2, 0): _FakeConnectivity({0: [0, 1, 2], 1: [1, 3, 2]})},
    )
    mesh = _fake_mesh(
        coords=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        topology=topology,
    )

    np.testing.assert_allclose(
        cell_midpoints(mesh),
        np.array([[2.0 / 3.0, 2.0 / 3.0], [4.0 / 3.0, 4.0 / 3.0]], dtype=float),
    )
    assert estimate_radius(mesh) == np.sqrt(2.0)


def test_v398_mesh_connectivity_helpers_direct_fill_without_list_staging():
    source_cells = inspect.getsource(helpers_module.mesh_cell_vertices)
    source_facets = inspect.getsource(helpers_module.mesh_facet_vertices)

    assert "[connectivity.links" not in source_cells
    assert "[connectivity.links" not in source_facets
    assert "np.empty((num_cells, verts_per_cell), dtype=np.int32)" in source_cells
    assert "np.empty((num_facets, verts_per_facet), dtype=np.int32)" in source_facets

    topology = _FakeTopology(
        dim=2,
        index_maps={2: _FakeIndexMap(2), 1: _FakeIndexMap(3)},
        connectivities={
            (2, 0): _FakeConnectivity({0: [0, 1, 2], 1: [1, 3, 2]}),
            (1, 0): _FakeConnectivity({0: [0, 1], 1: [1, 2], 2: [2, 0]}),
        },
    )
    mesh = _fake_mesh(
        coords=[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]],
        topology=topology,
    )

    np.testing.assert_array_equal(mesh_cell_vertices(mesh), [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_array_equal(mesh_facet_vertices(mesh), [[0, 1], [1, 2], [2, 0]])
