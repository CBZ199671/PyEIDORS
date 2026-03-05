"""Tests for EIDORS-style cache control plane behavior."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from pyeidors.cache import CacheManager, CachePolicy


def test_cache_enable_disable_and_debug_status():
    manager = CacheManager(scope="process")

    assert manager.status() == 1.0
    assert manager.debug_status() == 0.0

    manager.set_enabled(False, "calc_jacobian")
    assert manager.status() == 0.5
    assert manager.status("calc_jacobian") == 0.0
    assert manager.status("inv_solve_diff_GN_one_step") == 1.0

    manager.set_enabled(True, "calc_jacobian")
    assert manager.status() == 1.0
    assert manager.status("calc_jacobian") == 1.0

    manager.set_enabled(False)
    assert manager.status() == 0.0
    manager.set_enabled(True)
    assert manager.status() == 1.0

    manager.set_debug(True, "calc_jacobian")
    assert manager.debug_status() == 0.5
    assert manager.debug_status("calc_jacobian") == 1.0
    assert manager.debug_status("inv_solve_diff_GN_one_step") == 0.0
    manager.set_debug(False, "calc_jacobian")
    assert manager.debug_status() == 0.0


def test_cache_disable_name_bypasses_reuse():
    manager = CacheManager(scope="process")
    calls = {"count": 0}

    def _compute() -> np.ndarray:
        calls["count"] += 1
        return np.arange(8, dtype=float)

    manager.set_enabled(False, "calc_jacobian")
    _, lookup1 = manager.get_or_compute(
        artifact="jacobian",
        payload={"x": 1},
        compute_fn=_compute,
        name="calc_jacobian",
    )
    _, lookup2 = manager.get_or_compute(
        artifact="jacobian",
        payload={"x": 1},
        compute_fn=_compute,
        name="calc_jacobian",
    )

    assert calls["count"] == 2
    assert lookup1.layer == "disabled"
    assert lookup2.layer == "disabled"


def test_cache_boost_priority_and_time_clear(tmp_path: Path):
    manager = CacheManager(
        scope="both",
        cache_dir=tmp_path / "cache",
        policy=CachePolicy(process_max_bytes=8 * 1024**2, disk_max_bytes=8 * 1024**2),
    )

    manager.boost_priority(2.0)
    assert manager.boost_priority(0.0) == 2.0
    manager.boost_priority(-2.0)
    assert manager.boost_priority(0.0) == 0.0

    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"id": "old"},
        compute_fn=lambda: np.ones(256, dtype=float),
        name="inv_solve_diff_GN_one_step",
        persist=True,
    )
    barrier = time.time()
    time.sleep(0.02)
    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"id": "new"},
        compute_fn=lambda: np.ones(512, dtype=float),
        name="inv_solve_diff_GN_one_step",
        persist=True,
    )

    removed_old = manager.clear_old(barrier)
    assert removed_old >= 1

    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"id": "newer"},
        compute_fn=lambda: np.ones(128, dtype=float),
        name="inv_solve_diff_GN_one_step",
        persist=True,
    )
    newer_barrier = time.time()
    time.sleep(0.01)
    manager.get_or_compute(
        artifact="single_step_operator",
        payload={"id": "latest"},
        compute_fn=lambda: np.ones(64, dtype=float),
        name="inv_solve_diff_GN_one_step",
        persist=True,
    )
    removed_new = manager.clear_new(newer_barrier)
    assert removed_new >= 1
