"""Tests for score-aware cache eviction and clear_max behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyeidors.cache.store_disk import DiskCacheStore
from pyeidors.cache.store_process import ProcessCacheStore


def test_process_store_clear_max_prefers_high_score_entries():
    store = ProcessCacheStore(max_bytes=10 * 1024 * 1024)
    payload = np.ones(1024, dtype=np.float64)

    store.put(
        "high-effort",
        payload,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=1000.0,
        priority=2.0,
    )
    store.put(
        "low-effort-1",
        payload,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=1.0,
        priority=0.0,
    )
    store.put(
        "low-effort-2",
        payload,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=1.0,
        priority=0.0,
    )

    target = payload.nbytes + 512
    removed = store.clear_max(target)
    assert removed >= 1

    entries = store.list_entries()
    keys = {item["key"] for item in entries}
    assert "high-effort" in keys


def test_disk_store_clear_max_prefers_high_score_entries(tmp_path: Path):
    store = DiskCacheStore(tmp_path / "cache", max_bytes=10 * 1024 * 1024, compress_payloads=False)
    payload = np.ones(1024, dtype=np.float64)

    store.put(
        "high-effort",
        payload,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=1000.0,
        priority=2.0,
    )
    store.put(
        "low-effort-1",
        payload,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=1.0,
        priority=0.0,
    )
    store.put(
        "low-effort-2",
        payload,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=1.0,
        priority=0.0,
    )

    target = payload.nbytes + 512
    removed = store.clear_max(target)
    assert removed >= 1

    entries = store.list_entries()
    keys = {item["key"] for item in entries}
    assert "high-effort" in keys

