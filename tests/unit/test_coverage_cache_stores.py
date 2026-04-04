"""Tests for cache store_disk and store_process edge cases."""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from pyeidors.cache.store_disk import DiskCacheStore
from pyeidors.cache.store_process import ProcessCacheStore


class TestDiskCacheStoreEdgeCases:
    """Cover uncovered lines in store_disk.py."""

    def _make_store(self, tmp_path, **kwargs):
        defaults = dict(max_bytes=10 * 1024 * 1024, compress_payloads=True)
        defaults.update(kwargs)
        return DiskCacheStore(tmp_path / "cache", **defaults)

    def test_schema_migration_executes(self, tmp_path):
        """Line 116: migration for missing columns."""
        store = self._make_store(tmp_path)
        # Re-init should not fail (columns already exist)
        store._init_db()

    def test_is_expired_no_ttl_returns_false(self, tmp_path):
        """Line 138: ttl is None returns False."""
        store = self._make_store(tmp_path)
        assert store._is_expired(time.time() - 100, None) is False

    def test_put_serialization_failure(self, tmp_path):
        """Lines 208-209: serialization fails."""
        store = self._make_store(tmp_path)
        unpicklable = lambda: None  # noqa: E731
        result = store.put("key", unpicklable, artifact="test", cost=1.0)
        assert result is False

    def test_remove_entry_file_unlink_fails(self, tmp_path):
        """Lines 283-284: file unlink exception."""
        store = self._make_store(tmp_path)
        store.put("key1", "value1", artifact="test", cost=1.0)
        with store._session() as conn:
            with mock.patch.object(Path, "unlink", side_effect=PermissionError):
                store._remove_entry(conn, "key1", Path("/nonexistent"))

    def test_eviction_when_over_max_bytes(self, tmp_path):
        """Lines 297-308: eviction logic."""
        store = self._make_store(tmp_path, max_bytes=100)
        large_data = np.zeros(1000, dtype=np.float64)
        store.put("key1", large_data, artifact="test", cost=1.0)
        store.put("key2", large_data, artifact="test", cost=1.0)

    def test_clear_name_with_namespace(self, tmp_path):
        """Line 329: namespace filtering in clear_name."""
        store = self._make_store(tmp_path)
        store.put("k1", "v1", artifact="test", cost=1.0, name="a", namespace="ns1")
        store.put("k2", "v2", artifact="test", cost=1.0, name="a", namespace="ns2")
        removed = store.clear_name("a", namespace="ns1")
        assert removed == 1

    def test_clear_name_without_namespace(self, tmp_path):
        """Line 329: without namespace."""
        store = self._make_store(tmp_path)
        store.put("k1", "v1", artifact="test", cost=1.0, name="a")
        removed = store.clear_name("a")
        assert removed == 1

    def test_clear_max_already_under(self, tmp_path):
        """Line 356: total already under target."""
        store = self._make_store(tmp_path)
        store.put("k1", "tiny", artifact="test", cost=1.0)
        removed = store.clear_max(max_bytes=10 * 1024 * 1024)
        assert removed == 0

    def test_get_value_missing_key(self, tmp_path):
        """Line 416: key not found."""
        store = self._make_store(tmp_path)
        assert store.get_value("nonexistent") is None

    def test_get_value_deserialize_fails(self, tmp_path):
        """Lines 421-422: deserialization failure."""
        store = self._make_store(tmp_path)
        store.put("k1", "hello", artifact="test", cost=1.0)
        entry = store.list_entries(limit=1)
        assert len(entry) > 0
        # Corrupt the file
        with store._session() as conn:
            row = conn.execute("SELECT file_path FROM cache_entries WHERE cache_key = ?", ("k1",)).fetchone()
        Path(row[0]).write_bytes(b"corrupted data")
        assert store.get_value("k1") is None

    def test_list_entries_with_limit(self, tmp_path):
        """Lines 449-450: limit in list_entries."""
        store = self._make_store(tmp_path)
        store.put("k1", "v1", artifact="test", cost=1.0, name="a")
        store.put("k2", "v2", artifact="test", cost=1.0, name="a")
        entries = store.list_entries(name="a", limit=1)
        assert len(entries) == 1

    def test_collect_recent(self, tmp_path):
        """Lines 483-490: collect_recent method."""
        store = self._make_store(tmp_path)
        store.put("k1", "v1", artifact="test", cost=1.0, name="a")
        result = store.collect_recent(names=["a", "b"], limit_per_name=1)
        assert "a" in result
        assert "b" in result


class TestProcessCacheStoreEdgeCases:
    """Cover uncovered lines in store_process.py."""

    def test_put_replaces_existing(self):
        """Line 129: existing entry replaced."""
        store = ProcessCacheStore(max_bytes=10000)
        store.put("k1", "old", artifact="test", cost=1.0)
        store.put("k1", "new", artifact="test", cost=1.0)
        assert store.get("k1") == "new"

    def test_clear_name_skips_non_matching_name(self):
        """Line 180: skip non-matching name."""
        store = ProcessCacheStore(max_bytes=10000)
        store.put("k1", "v1", artifact="test", cost=1.0, name="a")
        store.put("k2", "v2", artifact="test", cost=1.0, name="b")
        removed = store.clear_name("a")
        assert removed == 1
        assert store.get_value("k2") == "v2"

    def test_clear_name_skips_non_matching_namespace(self):
        """Line 182: skip non-matching namespace."""
        store = ProcessCacheStore(max_bytes=10000)
        store.put("k1", "v1", artifact="test", cost=1.0, name="a", namespace="ns1")
        store.put("k2", "v2", artifact="test", cost=1.0, name="a", namespace="ns2")
        removed = store.clear_name("a", namespace="ns1")
        assert removed == 1

    def test_get_value_existing(self):
        """Lines 225-229: get_value returns entry value."""
        store = ProcessCacheStore(max_bytes=10000)
        store.put("k1", "hello", artifact="test", cost=1.0)
        assert store.get_value("k1") == "hello"

    def test_get_value_missing(self):
        """Lines 225-229: get_value returns None for missing."""
        store = ProcessCacheStore(max_bytes=10000)
        assert store.get_value("nonexistent") is None

    def test_list_entries_name_filter(self):
        """Lines 242, 244: name/namespace filter."""
        store = ProcessCacheStore(max_bytes=10000)
        store.put("k1", "v1", artifact="test", cost=1.0, name="a", namespace="ns1")
        store.put("k2", "v2", artifact="test", cost=1.0, name="b", namespace="ns1")
        store.put("k3", "v3", artifact="test", cost=1.0, name="a", namespace="ns2")
        entries = store.list_entries(name="a", namespace="ns1")
        assert len(entries) == 1

    def test_list_entries_with_limit(self):
        """Line 266: limit in list_entries."""
        store = ProcessCacheStore(max_bytes=10000)
        store.put("k1", "v1", artifact="test", cost=1.0, name="a")
        store.put("k2", "v2", artifact="test", cost=1.0, name="a")
        entries = store.list_entries(name="a", limit=1)
        assert len(entries) == 1
