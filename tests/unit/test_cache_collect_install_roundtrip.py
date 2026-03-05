"""Tests for cache collect/install roundtrip behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyeidors.cache import CacheManager, CachePolicy


def test_collect_recent_with_values_and_install_roundtrip(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    policy = CachePolicy(process_max_bytes=8 * 1024**2, disk_max_bytes=8 * 1024**2)
    manager = CacheManager(scope="both", cache_dir=cache_dir, policy=policy)

    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"case": "a"},
        compute_fn=lambda: np.eye(4),
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        persist=True,
    )
    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"case": "b"},
        compute_fn=lambda: np.eye(4) * 2.0,
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        persist=True,
    )

    snapshot = manager.collect_recent(
        names=["inv_solve_diff_GN_one_step", "inv_solve_diff_GN_one_step"],
        limit_per_name=1,
        namespace="difference",
        include_value=True,
    )
    collected = snapshot["inv_solve_diff_GN_one_step"]
    assert len(collected) == 2
    assert all("val" in item for item in collected)

    manager.clear(scope="both")
    calls = {"count": 0}

    def _compute() -> np.ndarray:
        calls["count"] += 1
        return np.eye(4)

    value_cold, cold_lookup = manager.get_or_compute(
        artifact="single_step_operator",
        payload={"case": "a"},
        compute_fn=_compute,
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        persist=True,
    )
    assert cold_lookup.hit is False
    assert calls["count"] == 1
    np.testing.assert_allclose(value_cold, np.eye(4))

    manager.clear(scope="both")
    installed = manager.install_to_cache(snapshot, target_layers="both")
    assert installed >= 2

    value_warm, warm_lookup = manager.get_or_compute(
        artifact="single_step_operator",
        payload={"case": "a"},
        compute_fn=_compute,
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        persist=True,
    )
    assert warm_lookup.hit is True
    assert calls["count"] == 1
    np.testing.assert_allclose(value_warm, np.eye(4))
