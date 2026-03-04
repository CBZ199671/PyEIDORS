"""Extended tests for the phase-2 cache manager."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyeidors.cache import CacheKeyParts, CacheManager, CachePolicy, build_cache_key


def test_cache_key_stable_for_semantically_equal_payload():
    parts_a = CacheKeyParts(
        artifact="jacobian",
        payload={"b": [3, 4], "a": {"x": 1, "y": 2}},
        namespace="unit",
        code_fingerprint="abc",
    )
    parts_b = CacheKeyParts(
        artifact="jacobian",
        payload={"a": {"y": 2, "x": 1}, "b": [3, 4]},
        namespace="unit",
        code_fingerprint="abc",
    )
    assert build_cache_key(parts_a) == build_cache_key(parts_b)


def test_cache_manager_process_hit():
    manager = CacheManager(scope="process", policy=CachePolicy(process_max_bytes=2 * 1024**2))
    calls = {"count": 0}

    def _compute():
        calls["count"] += 1
        return np.arange(8, dtype=float)

    v1, lookup1 = manager.get_or_compute(
        artifact="jacobian",
        payload={"id": 1},
        compute_fn=_compute,
        persist=False,
    )
    v2, lookup2 = manager.get_or_compute(
        artifact="jacobian",
        payload={"id": 1},
        compute_fn=_compute,
        persist=False,
    )

    assert calls["count"] == 1
    assert lookup1.hit is False
    assert lookup2.hit is True
    np.testing.assert_allclose(v1, v2)
    stats = manager.stats()
    assert stats["process_hits"] >= 1


def test_cache_manager_disk_persistence(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    policy = CachePolicy(process_max_bytes=1024, disk_max_bytes=8 * 1024**2)
    manager1 = CacheManager(scope="both", cache_dir=cache_dir, policy=policy)
    manager1.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 0.1, "shape": (4, 4)},
        compute_fn=lambda: {"lu": np.eye(4), "piv": np.arange(4)},
        persist=True,
    )

    manager2 = CacheManager(scope="both", cache_dir=cache_dir, policy=policy)
    calls = {"count": 0}

    def _compute():
        calls["count"] += 1
        return {"lu": np.eye(4), "piv": np.arange(4)}

    _, lookup = manager2.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 0.1, "shape": (4, 4)},
        compute_fn=_compute,
        persist=True,
    )

    assert lookup.hit is True
    assert lookup.layer in {"disk", "process"}
    assert calls["count"] == 0
    assert manager2.stats()["disk_hits"] >= 1


def test_cache_manager_off_scope_and_invalidation_paths(tmp_path: Path):
    disabled = CacheManager(scope="off")
    calls = {"count": 0}

    def _compute_disabled():
        calls["count"] += 1
        return {"value": 1}

    value, lookup = disabled.get_or_compute(
        artifact="mesh_bundle",
        payload={"id": 1},
        compute_fn=_compute_disabled,
        persist=True,
    )
    assert value == {"value": 1}
    assert lookup.layer == "disabled"
    assert calls["count"] == 1
    assert disabled.enabled is False

    manager = CacheManager(scope="both", cache_dir=tmp_path / "cache-two", policy=CachePolicy())
    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 1},
        compute_fn=lambda: np.eye(2),
        persist=True,
        cost=3.5,
    )
    # Explicit cost branch and both-layer invalidation/clear.
    removed = manager.invalidate(reason="test-clean")
    assert removed >= 1
    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"alpha": 2},
        compute_fn=lambda: np.eye(2),
        persist=True,
    )
    manager.clear(scope="process")
    stats_after_process_clear = manager.stats()
    assert stats_after_process_clear["process_items"] == 0
    manager.clear(scope="disk")
    stats_after_disk_clear = manager.stats()
    assert stats_after_disk_clear["disk_items"] == 0
