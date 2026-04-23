"""Additional cache and helper coverage for sparse Bayesian engine."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyeidors.inverse.solvers import sparse_bayesian_engine as sparse_module


class _Lookup:
    hit = True
    layer = "process"
    artifact = "jacobian"
    key = "demo"


class _EnabledCacheManager:
    def __init__(self):
        self.enabled = True
        self.semantic_calls = 0
        self.regular_calls = 0

    def get_or_compute_semantic(self, **kwargs):
        self.semantic_calls += 1
        return kwargs["compute_fn"](), _Lookup()

    def get_or_compute(self, **kwargs):
        self.regular_calls += 1
        return kwargs["compute_fn"](), _Lookup()


def test_constructor_success_forward_measurement_and_create_image(monkeypatch):
    monkeypatch.setattr(sparse_module, "_CUQI_AVAILABLE", True)
    monkeypatch.setattr(
        sparse_module,
        "create_pde_model",
        lambda eit_system: (
            "fake-pde",
            lambda conductivity: np.asarray(conductivity, dtype=float)[:2],
            SimpleNamespace(n_elements=3, n_measurements=2),
        ),
    )

    eit_system = SimpleNamespace(fwd_model="fwd-model")
    rec = sparse_module.SparseBayesianReconstructor(eit_system, verbose=False)

    assert rec.eit_system is eit_system
    assert rec.n_elements == 3
    np.testing.assert_allclose(
        rec._forward_measurement(np.array([1.0, 2.0, 3.0])), np.array([1.0, 2.0])
    )
    image = rec._create_homogeneous_image(1.25)
    np.testing.assert_allclose(image.elem_data, np.array([1.25, 1.25, 1.25]))


def test_prepare_jacobian_and_hierarchy_use_enabled_cache_manager(monkeypatch):
    monkeypatch.setattr(
        sparse_module, "model_signature_from_forward_model", lambda _fwd: "model-sig"
    )
    monkeypatch.setattr(
        sparse_module,
        "pattern_signature_from_forward_model",
        lambda _fwd: "pattern-sig",
    )
    monkeypatch.setattr(
        sparse_module,
        "backend_signature_from_forward_model",
        lambda _fwd: "backend-sig",
    )

    rec = sparse_module.SparseBayesianReconstructor.__new__(
        sparse_module.SparseBayesianReconstructor
    )
    rec.config = sparse_module.SparseBayesianConfig(
        cache_jacobian=True, coarse_levels=(4, 2), coarse_group_size=3
    )
    rec.n_elements = 4
    rec.n_measurements = 3
    rec._cached_jacobian = None
    rec._cached_baseline = None
    rec._cached_basis = object()
    rec._cached_reduced_matrix = object()
    rec._cached_U = object()
    rec._cached_singular = object()
    rec._coarse_levels_cache = {}
    rec._cached_coarse_matrices = {"old": np.eye(1)}
    rec.fwd_model = SimpleNamespace()
    rec._eit_pde = SimpleNamespace(
        jacobian_wrt_parameter=lambda wrt: np.eye(3, 4, dtype=float)
    )
    rec.eit_system = SimpleNamespace(cache_manager=_EnabledCacheManager())

    baseline = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    jac = rec._prepare_jacobian(baseline)
    np.testing.assert_allclose(jac, np.eye(3, 4, dtype=float))
    np.testing.assert_allclose(rec._cached_baseline, baseline)
    assert rec.eit_system.cache_manager.semantic_calls == 1
    assert rec._cached_basis is None
    assert rec._cached_reduced_matrix is None
    assert rec._cached_U is None
    assert rec._cached_singular is None
    assert rec._cached_coarse_matrices == {}

    hierarchy = rec._build_coarse_hierarchy()
    assert hierarchy
    assert rec.eit_system.cache_manager.regular_calls == 1


def test_sparse_engine_noise_floor_and_forward_without_to_numpy():
    rec = sparse_module.SparseBayesianReconstructor.__new__(
        sparse_module.SparseBayesianReconstructor
    )
    rec.config = sparse_module.SparseBayesianConfig(noise_rel=0.0, noise_floor=2e-3)
    rec._cuqi_model = lambda conductivity: np.asarray(conductivity, dtype=float)[:3]
    rec.n_elements = 5
    rec.fwd_model = "unused"

    np.testing.assert_allclose(
        rec._forward_measurement(np.array([1.0, 2.0, 3.0, 4.0], dtype=float)),
        np.array([1.0, 2.0, 3.0], dtype=float),
    )
    assert (
        rec._estimate_noise_level(np.array([np.nan, np.nan], dtype=float))
        == rec.config.noise_floor
    )
