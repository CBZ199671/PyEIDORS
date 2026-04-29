"""T6 — process-local Jacobian cache primitive contract (V9, V17).

Mirrors the V16 / V17 guard contract from
``tests/unit/test_forward_process_setup_cache.py`` for the Jacobian
side. The cache must:

- reject empty ``sigma_fingerprint`` (V9 fingerprints required to
  distinguish two stored Jacobians);
- reject both-empty mesh identifier (V17 mirror of V16);
- produce a byte-stable SHA256 hex of the canonical JSON payload;
- round-trip arrays put under a key, return ``None`` on miss;
- evict the oldest entry once item budget is reached;
- be cleared by :func:`clear_process_jacobian_cache`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyeidors.inverse.jacobian import (
    build_process_jacobian_key,
    clear_process_jacobian_cache,
    get_process_cached_jacobian,
    process_jacobian_cache_stats,
    put_process_cached_jacobian,
)
from pyeidors.inverse.jacobian.process_jacobian_cache import (
    _PROCESS_JACOBIAN_CACHE_MAX_ITEMS,
)


@pytest.fixture(autouse=True)
def _isolate_jacobian_cache():
    """Each test starts with an empty cache; tear it down again."""
    clear_process_jacobian_cache()
    try:
        yield
    finally:
        clear_process_jacobian_cache()


def test_empty_sigma_fingerprint_rejected():
    with pytest.raises(ValueError, match="sigma_fingerprint"):
        build_process_jacobian_key(
            sigma_fingerprint="",
            mesh_file="cylinder_16e.msh",
        )


def test_empty_mesh_identifier_rejected():
    with pytest.raises(ValueError, match="mesh_file or mesh_content_hash"):
        build_process_jacobian_key(
            sigma_fingerprint="abc123",
            mesh_file=None,
            mesh_content_hash="",
        )


def test_key_byte_stable_for_identical_payload():
    """Same inputs in different call order → identical key (sort_keys)."""
    a = build_process_jacobian_key(
        sigma_fingerprint="abc",
        mesh_file="m1.msh",
        jacobian_method="direct",
        backend_signature={"x": 1, "y": 2},
        pattern_signature={"n": 16},
        model_signature={"dofs": 1024},
    )
    b = build_process_jacobian_key(
        backend_signature={"y": 2, "x": 1},
        model_signature={"dofs": 1024},
        pattern_signature={"n": 16},
        sigma_fingerprint="abc",
        mesh_file="m1.msh",
        jacobian_method="direct",
    )
    assert a == b
    # SHA256 hex is 64 characters
    assert len(a) == 64


def test_distinct_keys_for_different_axes():
    base_kwargs = dict(
        sigma_fingerprint="abc",
        mesh_file="m1.msh",
        jacobian_method="direct",
        backend_signature={"backend": "petsc"},
    )
    key_base = build_process_jacobian_key(**base_kwargs)
    key_sigma = build_process_jacobian_key(
        **{**base_kwargs, "sigma_fingerprint": "def"}
    )
    key_mesh_file = build_process_jacobian_key(**{**base_kwargs, "mesh_file": "m2.msh"})
    key_mesh_hash = build_process_jacobian_key(
        sigma_fingerprint="abc",
        mesh_file=None,
        mesh_content_hash="hash1",
        jacobian_method="direct",
        backend_signature={"backend": "petsc"},
    )
    key_method = build_process_jacobian_key(
        **{**base_kwargs, "jacobian_method": "linearized"}
    )
    key_backend = build_process_jacobian_key(
        **{**base_kwargs, "backend_signature": {"backend": "scipy"}}
    )
    key_calculator = build_process_jacobian_key(
        **{
            **base_kwargs,
            "calculator_signature": {
                "qualname": "EidorsJacobianAdapter",
                "sign_convention": "-dV/dsigma_eidors_canonical",
            },
        }
    )
    keys = {key_base, key_sigma, key_mesh_file, key_mesh_hash, key_method, key_backend}
    keys.add(key_calculator)
    assert len(keys) == 7


def test_get_returns_none_for_missing_key():
    assert get_process_cached_jacobian("does-not-exist") is None


def test_put_then_get_round_trip():
    key = build_process_jacobian_key(
        sigma_fingerprint="abc",
        mesh_content_hash="hash1",
    )
    payload = np.linspace(0.0, 1.0, 24, dtype=np.float64).reshape(4, 6)
    put_process_cached_jacobian(key, payload)
    cached = get_process_cached_jacobian(key)
    assert cached is not None
    np.testing.assert_array_equal(cached, payload)
    assert cached.dtype == payload.dtype
    assert cached.shape == payload.shape


def test_lru_eviction_drops_oldest_entry():
    max_items = int(_PROCESS_JACOBIAN_CACHE_MAX_ITEMS)
    keys = []
    for idx in range(max_items + 2):
        key = build_process_jacobian_key(
            sigma_fingerprint=f"sigma-{idx:08x}",
            mesh_content_hash="mesh",
        )
        put_process_cached_jacobian(key, np.full(3, float(idx), dtype=np.float64))
        keys.append(key)
    stats = process_jacobian_cache_stats()
    assert stats["items"] == max_items
    assert stats["max_items"] == max_items
    # Oldest two entries evicted; tail still cached.
    assert get_process_cached_jacobian(keys[0]) is None
    assert get_process_cached_jacobian(keys[1]) is None
    for key in keys[2:]:
        assert get_process_cached_jacobian(key) is not None


def test_clear_drops_all_entries():
    key = build_process_jacobian_key(
        sigma_fingerprint="abc",
        mesh_content_hash="hash1",
    )
    put_process_cached_jacobian(key, np.zeros(3))
    assert process_jacobian_cache_stats()["items"] == 1
    clear_process_jacobian_cache()
    assert process_jacobian_cache_stats()["items"] == 0
    assert get_process_cached_jacobian(key) is None
