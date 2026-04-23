"""Additional branch coverage for regularization helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pyeidors.inverse.regularization import smoothness as smooth_module
from pyeidors.inverse.regularization.smoothness import (
    NOSERRegularization,
    SmoothnessRegularization,
    TotalVariationRegularization,
    _cell_difference_operator,
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


def _fake_model(*, n_elements: int = 3, mesh=None):
    index_map = SimpleNamespace(size_local=n_elements)
    dofmap = SimpleNamespace(index_map=index_map, index_map_bs=1)
    v_sigma = SimpleNamespace(dofmap=dofmap)
    return SimpleNamespace(mesh=mesh or object(), V_sigma=v_sigma)


def test_cell_difference_operator_handles_missing_and_empty_connectivity():
    mesh_missing = SimpleNamespace(
        topology=_FakeTopology(dim=2, facet_size=1, connectivity=None)
    )
    assert _cell_difference_operator(mesh_missing, 3).shape == (0, 3)

    mesh_missing_map = SimpleNamespace(
        topology=_FakeTopology(
            dim=2, facet_size=None, connectivity=_FakeConnectivity({0: [0, 1]})
        )
    )
    assert _cell_difference_operator(mesh_missing_map, 3).shape == (0, 3)

    mesh_boundary_only = SimpleNamespace(
        topology=_FakeTopology(
            dim=2, facet_size=2, connectivity=_FakeConnectivity({0: [0], 1: [1]})
        )
    )
    assert _cell_difference_operator(mesh_boundary_only, 3).shape == (0, 3)


def test_cell_difference_operator_builds_rows_and_smoothness_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    mesh = SimpleNamespace(
        topology=_FakeTopology(
            dim=2,
            facet_size=3,
            connectivity=_FakeConnectivity({0: [0, 1], 1: [1], 2: [1, 2]}),
        )
    )
    L = _cell_difference_operator(mesh, 3)
    assert L.shape == (2, 3)
    np.testing.assert_allclose(
        L.toarray(), np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]], dtype=float)
    )

    monkeypatch.setattr(
        smooth_module,
        "_cell_difference_operator",
        lambda _mesh, n: csr_matrix((0, n), dtype=np.float64),
    )
    reg = SmoothnessRegularization(_fake_model(n_elements=3, mesh=mesh), alpha=0.5)
    matrix = reg.create_matrix()
    np.testing.assert_allclose(matrix.toarray(), 0.5 * np.eye(3))


def test_total_variation_reference_validation_identity_and_nonlinear(
    monkeypatch: pytest.MonkeyPatch,
):
    reg_bad = TotalVariationRegularization(
        _fake_model(n_elements=3),
        reference_conductivity=np.array([1.0, 2.0], dtype=float),
    )
    with pytest.raises(ValueError, match="must match the number of elements"):
        reg_bad._reference_vector()

    monkeypatch.setattr(
        smooth_module,
        "_cell_difference_operator",
        lambda _mesh, n: csr_matrix((0, n), dtype=np.float64),
    )
    reg = TotalVariationRegularization(
        _fake_model(n_elements=3), alpha=2.0, epsilon=1e-3, reference_conductivity=1.0
    )
    matrix = reg.create_matrix()
    np.testing.assert_allclose(matrix.toarray(), 2.0 * np.eye(3))

    nonlinear = reg.create_nonlinear_term(np.array([1.0, 1.5, 2.0], dtype=float))
    assert nonlinear.shape == (3, 3)
    assert np.isfinite(nonlinear).all()


def test_noser_floor_paths(monkeypatch: pytest.MonkeyPatch):
    class _FakeFunction:
        def __init__(self, _space):
            self.x = SimpleNamespace(array=np.zeros(3, dtype=float))

    monkeypatch.setattr(smooth_module.fem, "Function", _FakeFunction)

    reg = NOSERRegularization(
        _fake_model(n_elements=3),
        jacobian_calculator=SimpleNamespace(
            calculate=lambda _sigma: np.zeros((2, 3), dtype=float)
        ),
        base_conductivity=1.0,
        alpha=1.5,
        exponent=0.5,
        floor=0.7,
        adaptive_floor=False,
    )
    diag = reg._compute_baseline_diag()
    np.testing.assert_allclose(diag, np.full(3, 0.7, dtype=float))

    reg._baseline_diag = np.array([1.0, 4.0, 9.0], dtype=float)
    matrix = reg.create_matrix()
    np.testing.assert_allclose(
        matrix.diagonal(), 1.5 * np.array([1.0, 2.0, 3.0], dtype=float)
    )
