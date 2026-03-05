"""Tests for EIDORS-style rank eviction ordering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyeidors.cache.store_disk import DiskCacheStore
from pyeidors.cache.store_process import ProcessCacheStore


def test_process_store_eviction_prefers_high_score_eff_and_small_size():
    store = ProcessCacheStore(max_bytes=10 * 1024 * 1024)
    small = np.ones(128, dtype=np.float64)
    large = np.ones(4096, dtype=np.float64)

    store.put(
        "keep-high-effort",
        small,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=1000.0,
        priority=1.0,
    )
    store.put(
        "keep-small-size",
        small,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=10.0,
        priority=0.0,
    )
    store.put(
        "evict-large-same-effort",
        large,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=10.0,
        priority=0.0,
    )

    target_bytes = small.nbytes * 2 + 32
    removed = store.clear_max(target_bytes)
    assert removed >= 1

    keys = {entry["key"] for entry in store.list_entries()}
    assert "keep-high-effort" in keys
    assert "keep-small-size" in keys
    assert "evict-large-same-effort" not in keys


def test_disk_store_eviction_prefers_high_score_eff_and_small_size(tmp_path: Path):
    store = DiskCacheStore(
        tmp_path / "cache",
        max_bytes=10 * 1024 * 1024,
        compress_payloads=False,
    )
    small = np.ones(128, dtype=np.float64)
    large = np.ones(4096, dtype=np.float64)

    store.put(
        "keep-high-effort",
        small,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=1000.0,
        priority=1.0,
    )
    store.put(
        "keep-small-size",
        small,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=10.0,
        priority=0.0,
    )
    store.put(
        "evict-large-same-effort",
        large,
        artifact="single_step_operator",
        name="inv_solve_diff_GN_one_step",
        namespace="difference",
        cost=1.0,
        effort=10.0,
        priority=0.0,
    )

    entries_before = {entry["key"]: entry for entry in store.list_entries()}
    target_bytes = (
        int(entries_before["keep-high-effort"]["size_bytes"])
        + int(entries_before["keep-small-size"]["size_bytes"])
        + 64
    )
    removed = store.clear_max(target_bytes)
    assert removed >= 1

    keys = {entry["key"] for entry in store.list_entries()}
    assert "keep-high-effort" in keys
    assert "keep-small-size" in keys
    assert "evict-large-same-effort" not in keys
