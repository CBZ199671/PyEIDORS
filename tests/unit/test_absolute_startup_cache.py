"""Tests for absolute startup cache wiring in GN runtime."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pyeidors.inverse.solvers.gauss_newton_runtime as gn_runtime


class _FakeLookup:
    def __init__(self, *, hit: bool, layer: str, artifact: str, key: str):
        self.hit = hit
        self.layer = layer
        self.artifact = artifact
        self.key = key


class _FakeCacheManager:
    def __init__(self):
        self.calls = 0

    def get_or_compute_semantic(self, **kwargs):
        self.calls += 1
        compute_fn = kwargs["compute_fn"]
        value = compute_fn()
        return value, _FakeLookup(
            hit=False,
            layer="compute",
            artifact="absolute_startup_jacobian",
            key="startup-key",
        )


def _fake_sigma(values: np.ndarray):
    return SimpleNamespace(x=SimpleNamespace(array=np.asarray(values, dtype=float)))


def test_startup_cache_lookup_uses_cache_manager(monkeypatch):
    monkeypatch.setattr(gn_runtime, "model_signature_from_forward_model", lambda _: "model")
    monkeypatch.setattr(gn_runtime, "pattern_signature_from_forward_model", lambda _: "pattern")
    monkeypatch.setattr(gn_runtime, "backend_signature_from_forward_model", lambda _: "backend")

    cache = _FakeCacheManager()
    jacobian = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    reconstructor = SimpleNamespace(
        solver_mode="fast",
        absolute_startup_cache=True,
        cache_manager=cache,
        fwd_model=SimpleNamespace(),
        negate_jacobian=False,
        jacobian_calculator=SimpleNamespace(
            calculate=lambda sigma_current, method: jacobian,
        ),
        linear_solver="auto",
        preconditioner="auto",
        line_search_mode="fast",
        jacobian_update_every=2,
        jacobian_reuse_tol=1e-3,
    )

    cached, lookup = gn_runtime._startup_cache_lookup(
        reconstructor,
        _fake_sigma(np.array([1.0, 1.0], dtype=float)),
        "efficient",
    )

    assert cache.calls == 1
    np.testing.assert_allclose(cached, jacobian)
    assert lookup["layer"] == "compute"
    assert lookup["artifact"] == "absolute_startup_jacobian"


def test_startup_cache_lookup_disabled_returns_none():
    reconstructor = SimpleNamespace(
        solver_mode="strict",
        absolute_startup_cache=True,
        cache_manager=None,
    )
    cached, lookup = gn_runtime._startup_cache_lookup(
        reconstructor,
        _fake_sigma(np.array([1.0], dtype=float)),
        "efficient",
    )
    assert cached is None
    assert lookup["layer"] == "disabled"
