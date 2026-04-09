"""Branch coverage for sparse Bayesian backend wrappers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.inverse.solvers import sparse_bayesian_backends as backend_module
from pyeidors.inverse.solvers import sparse_bayesian_engine as engine_module


class _DummyBackend(backend_module.SparseBayesianBackendMixin):
    def __init__(self):
        self.verbose = True
        self.config = SimpleNamespace(smoothing_beta=0.25)
        self._cached_coarse_matrices = {}


def test_sparse_backend_wrapper_methods_delegate(monkeypatch: pytest.MonkeyPatch):
    backend = _DummyBackend()

    monkeypatch.setattr(
        backend_module,
        "coarse_initialization",
        lambda *args: np.array([1.0, 2.0], dtype=float),
    )
    out = backend._coarse_initialization(
        np.eye(2, dtype=float),
        np.ones(2, dtype=float),
        0.1,
        1.0,
        [np.array([0, 1], dtype=int)],
        2,
        None,
    )
    np.testing.assert_allclose(out, np.array([1.0, 2.0], dtype=float))


def test_solve_with_cuqi_map_without_warm_start_uses_disp_only():
    backend = _DummyBackend()
    calls: list[dict[str, object]] = []

    class _Estimate:
        @staticmethod
        def to_numpy():
            return np.array([0.2, 0.3], dtype=float)

    class _Problem:
        def MAP(self, **kwargs):
            calls.append(dict(kwargs))
            return _Estimate()

    out = backend._solve_with_cuqi_map(_Problem(), warm_start=None)
    np.testing.assert_allclose(out, np.array([0.2, 0.3], dtype=float))
    assert calls == [{"disp": True}]


def test_sparse_backend_import_guards_raise_cleanly(monkeypatch: pytest.MonkeyPatch):
    backend = _DummyBackend()
    monkeypatch.setattr(engine_module, "LinearModel", None, raising=False)
    monkeypatch.setattr(engine_module, "SmoothedLaplace", None, raising=False)
    monkeypatch.setattr(engine_module, "Gaussian", None, raising=False)
    monkeypatch.setattr(engine_module, "BayesianProblem", None, raising=False)

    with pytest.raises(ImportError, match="CUQIpy is required"):
        backend._linear_model(np.eye(2, dtype=float))

    with pytest.raises(ImportError, match="CUQIpy is required"):
        backend._sparse_prior(target_dim=2, prior_scale=0.5)

    with pytest.raises(ImportError, match="CUQIpy is required"):
        backend._gaussian_likelihood("latent", noise_sigma=0.1)

    with pytest.raises(ImportError, match="CUQIpy is required"):
        backend._bayesian_problem("y", "x")
