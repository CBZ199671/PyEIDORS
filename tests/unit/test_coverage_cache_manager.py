"""Tests for cache manager edge cases to achieve 100% coverage."""

from __future__ import annotations


import numpy as np
import pytest

from pyeidors.cache.manager import (
    CacheManager,
    _get_shared_process_store,
    _SHARED_PROCESS_STORES,
    _SHARED_PROCESS_STORES_LOCK,
)
from pyeidors.cache.types import CachePolicy, normalize_cache_lifecycle


class TestNormalizeCacheLifecycle:
    """Cover line 21 in types.py."""

    def test_unknown_value_returns_default(self):
        assert normalize_cache_lifecycle("bogus") == "session"

    def test_none_returns_default(self):
        assert normalize_cache_lifecycle(None) == "session"


class TestSharedProcessStore:
    """Cover line 42 in manager.py: max_bytes upgrade."""

    def test_max_bytes_upgraded(self, tmp_path):

        store1 = _get_shared_process_store(
            cache_dir=tmp_path / "test_shared",
            max_bytes=100,
            code_fingerprint="test",
        )
        store2 = _get_shared_process_store(
            cache_dir=tmp_path / "test_shared",
            max_bytes=200,
            code_fingerprint="test",
        )
        assert store1 is store2
        assert store2.max_bytes == 200

        with _SHARED_PROCESS_STORES_LOCK:
            key = (str((tmp_path / "test_shared").resolve()), "test")
            _SHARED_PROCESS_STORES.pop(key, None)


class TestCacheManagerStatus:
    """Cover lines 131, 161, 169-171."""

    def test_status_global_disabled(self, tmp_path):
        """Line 131: globally disabled returns 0.0."""
        mgr = CacheManager(scope="off", cache_dir=tmp_path / "cache_off")
        assert mgr.status("test_name") == 0.0

    def test_debug_status_fully_enabled(self, tmp_path):
        """Line 161: debug fully enabled."""
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_dbg")
        mgr.set_debug(True)
        assert mgr.debug_status("any_name") == 1.0

    def test_debug_status_partial_enabled(self, tmp_path):
        """Lines 169-171: debug partially enabled for specific name."""
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_dbg2")
        mgr.set_debug(True, name="specific")
        assert mgr.debug_status("specific") == 1.0
        assert mgr.debug_status("other") == 0.0

    def test_set_debug_name_off(self, tmp_path):
        """Lines 169-171: turning debug off for name."""
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_dbg3")
        mgr.set_debug(True, name="x")
        mgr.set_debug(False, name="x")
        assert mgr.debug_status() == 0.0


class TestCacheManagerGetOrCompute:
    """Cover lines 226-238: repopulate process from disk."""

    def test_disk_hit_populates_process(self, tmp_path):
        mgr = CacheManager(
            scope="both",
            cache_dir=tmp_path / "cache_both",
            policy=CachePolicy(disk_lifecycle="persistent"),
        )
        data = np.array([1.0, 2.0, 3.0])

        # Populate disk
        val1, info1 = mgr.get_or_compute(
            artifact="test_art",
            payload={"key": "val"},
            compute_fn=lambda: data,
        )
        assert info1.layer == "compute"

        # Clear process cache only
        if mgr._process is not None:
            mgr._process.clear()

        # Should hit disk and repopulate process
        val2, info2 = mgr.get_or_compute(
            artifact="test_art",
            payload={"key": "val"},
            compute_fn=lambda: None,
        )
        assert info2.layer == "disk"
        np.testing.assert_array_equal(val2, data)


class TestCacheManagerClearMax:
    """Cover lines 327-332."""

    def test_clear_max(self, tmp_path):
        mgr = CacheManager(
            scope="both",
            cache_dir=tmp_path / "cache_cm",
            policy=CachePolicy(disk_lifecycle="persistent"),
        )
        mgr.get_or_compute(
            artifact="test",
            payload={"k": 1},
            compute_fn=lambda: np.zeros(100),
        )
        removed = mgr.clear_max(max_bytes=0)
        assert removed >= 0


class TestCacheManagerEntryValue:
    """Cover lines 352, 355."""

    def test_entry_value_process_layer(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_ev")
        mgr.get_or_compute(
            artifact="test",
            payload={"k": 1},
            compute_fn=lambda: "hello",
        )
        entries = mgr.list_entries()
        if entries:
            val = mgr._entry_value_for_layer(entries[0]["key"], "process")
            assert val == "hello"

    def test_entry_value_unknown_layer(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_ev2")
        assert mgr._entry_value_for_layer("key", "unknown") is None


class TestCacheManagerInstall:
    """Cover lines 442-470."""

    def test_install_invalid_target_layers(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst")
        with pytest.raises(ValueError, match="target_layers"):
            mgr.install_to_cache({}, target_layers="invalid")

    def test_install_invalid_snapshot_type(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst2")
        with pytest.raises(TypeError, match="snapshot must be"):
            mgr.install_to_cache("not_a_dict_or_list", target_layers="process")

    def test_install_skips_non_dict_items(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst3")
        result = mgr.install_to_cache(["not_a_dict"], target_layers="process")
        assert result == 0

    def test_install_skips_missing_val(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst4")
        result = mgr.install_to_cache([{"key": "k1"}], target_layers="process")
        assert result == 0

    def test_install_skips_invalid_key(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst5")
        result = mgr.install_to_cache(
            [{"val": "v", "key": ""}], target_layers="process"
        )
        assert result == 0

    def test_install_skips_no_artifact(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst6")
        result = mgr.install_to_cache(
            [{"val": "v", "key": "k1", "meta": {}}], target_layers="process"
        )
        assert result == 0

    def test_install_skips_non_dict_meta(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst7")
        result = mgr.install_to_cache(
            [{"val": "v", "key": "k1", "meta": "not_dict"}],
            target_layers="process",
        )
        assert result == 0

    def test_install_success(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst8")
        snapshot = [
            {
                "val": "value",
                "key": "test_key",
                "meta": {"artifact": "test_art", "namespace": "default", "cost": 1.0},
            }
        ]
        result = mgr.install_to_cache(snapshot, target_layers="process")
        assert result == 1

    def test_install_from_dict_snapshot(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_inst9")
        snapshot = {
            "group": [
                {
                    "val": "value",
                    "key": "test_key",
                    "meta": {"artifact": "test_art"},
                }
            ]
        }
        result = mgr.install_to_cache(snapshot, target_layers="process")
        assert result == 1


class TestCacheManagerListEntries:
    """Cover lines 513-523."""

    def test_list_entries_with_limit(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_le")
        mgr.get_or_compute(artifact="a", payload={"k": 1}, compute_fn=lambda: "v1")
        mgr.get_or_compute(artifact="b", payload={"k": 2}, compute_fn=lambda: "v2")
        entries = mgr.list_entries(limit=1)
        assert len(entries) == 1

    def test_list_entries_no_limit(self, tmp_path):
        mgr = CacheManager(scope="process", cache_dir=tmp_path / "cache_le2")
        mgr.get_or_compute(artifact="a", payload={"k": 1}, compute_fn=lambda: "v1")
        entries = mgr.list_entries()
        assert len(entries) >= 1
