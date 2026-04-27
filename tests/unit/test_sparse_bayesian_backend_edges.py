"""Branch coverage for sparse Bayesian wrapper methods.

T76 Path C folded the historical ``SparseBayesianBackendMixin`` into
:class:`SparseBayesianReconstructor`. These tests now exercise the
wrapper methods directly on the reconstructor (constructed via
``__new__`` with a hand-built minimal namespace) so the same branch
coverage carries over without depending on the deleted mixin module.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyeidors.inverse.solvers import sparse_bayesian_engine as engine_module


def _bare_reconstructor() -> engine_module.SparseBayesianReconstructor:
    """Build a reconstructor without invoking ``__init__`` (skips CUQIpy + EITSystem)."""
    rec = engine_module.SparseBayesianReconstructor.__new__(
        engine_module.SparseBayesianReconstructor
    )
    rec.verbose = True
    rec.config = SimpleNamespace(smoothing_beta=0.25)
    rec._cached_coarse_matrices = {}
    return rec


def test_coarse_initialization_wrapper_delegates_to_module(
    monkeypatch: pytest.MonkeyPatch,
):
    rec = _bare_reconstructor()
    monkeypatch.setattr(
        engine_module,
        "coarse_initialization",
        lambda *args: np.array([1.0, 2.0], dtype=float),
    )
    out = rec._coarse_initialization(
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
    rec = _bare_reconstructor()
    calls: list[dict[str, object]] = []

    class _Estimate:
        @staticmethod
        def to_numpy():
            return np.array([0.2, 0.3], dtype=float)

    class _Problem:
        def MAP(self, **kwargs):
            calls.append(dict(kwargs))
            return _Estimate()

    out = rec._solve_with_cuqi_map(_Problem(), warm_start=None)
    np.testing.assert_allclose(out, np.array([0.2, 0.3], dtype=float))
    assert calls == [{"disp": True}]


def test_cuqi_adapter_import_guards_raise_cleanly(monkeypatch: pytest.MonkeyPatch):
    rec = _bare_reconstructor()
    monkeypatch.setattr(engine_module, "LinearModel", None, raising=False)
    monkeypatch.setattr(engine_module, "SmoothedLaplace", None, raising=False)
    monkeypatch.setattr(engine_module, "Gaussian", None, raising=False)
    monkeypatch.setattr(engine_module, "BayesianProblem", None, raising=False)

    with pytest.raises(ImportError, match="CUQIpy is required"):
        rec._linear_model(np.eye(2, dtype=float))

    with pytest.raises(ImportError, match="CUQIpy is required"):
        rec._sparse_prior(target_dim=2, prior_scale=0.5)

    with pytest.raises(ImportError, match="CUQIpy is required"):
        rec._gaussian_likelihood("latent", noise_sigma=0.1)

    with pytest.raises(ImportError, match="CUQIpy is required"):
        rec._bayesian_problem("y", "x")
