"""Coverage-focused tests for cache key/process/disk stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from pyeidors.cache.keys import _normalize, hash_array, hash_path
from pyeidors.cache.store_disk import DiskCacheStore
from pyeidors.cache.store_process import ProcessCacheStore, estimate_object_size_bytes


@dataclass(frozen=True)
class _Payload:
    name: str
    count: int


def test_normalize_and_hash_helpers_cover_special_types(tmp_path: Path):
    payload = {
        "path": tmp_path / "mesh.msh",
        "blob": b"abc",
        "arr": np.arange(6, dtype=np.float64).reshape(2, 3),
        "scalar_float": np.float64(1.5),
        "scalar_int": np.int32(7),
        "scalar_bool": np.bool_(True),
        "set_values": {3, 1, 2},
        "data": _Payload("demo", 2),
    }
    normalized = _normalize(payload)

    assert normalized["path"].endswith("mesh.msh")
    assert "__bytes__" in normalized["blob"]
    assert normalized["arr"]["__ndarray__"] is True
    assert normalized["scalar_float"] == 1.5
    assert normalized["scalar_int"] == 7
    assert normalized["scalar_bool"] is True
    assert normalized["set_values"] == [1, 2, 3]
    assert normalized["data"]["name"] == "demo"

    missing = tmp_path / "missing.file"
    hash_missing_a = hash_path(missing)
    hash_missing_b = hash_path(missing)
    assert hash_missing_a == hash_missing_b

    real_file = tmp_path / "mesh.msh"
    real_file.write_text("v1", encoding="utf-8")
    hash_real_a = hash_path(real_file)
    time.sleep(0.001)
    real_file.write_text("v2", encoding="utf-8")
    hash_real_b = hash_path(real_file)
    assert hash_real_a != hash_real_b

    arr64 = np.array([1.0, 2.0], dtype=np.float64)
    arr32 = arr64.astype(np.float32)
    assert hash_array(arr64) != hash_array(arr32)


def test_estimate_object_size_and_process_store_eviction():
    class _Unpicklable:
        def __getstate__(self):
            raise RuntimeError("nope")

    assert estimate_object_size_bytes(np.zeros(4, dtype=np.float64)) >= 32
    assert estimate_object_size_bytes(b"abc") == 3
    assert estimate_object_size_bytes("x") == 64
    assert estimate_object_size_bytes({"a": 1, "b": [1, 2, 3]}) > 96
    assert estimate_object_size_bytes([1, 2, 3]) > 96
    assert estimate_object_size_bytes(_Unpicklable()) == 1024

    store = ProcessCacheStore(max_bytes=256)
    store.put("k1", np.arange(100, dtype=np.float64), artifact="jacobian", cost=1.0)
    # Oversized insertion forces eviction of least-recently-used entries.
    store.put("k2", np.arange(200, dtype=np.float64), artifact="jacobian", cost=1.0)
    assert store.stats()["items"] <= 1

    store_zero = ProcessCacheStore(max_bytes=0)
    store_zero.put("k", {"a": 1}, artifact="mesh_bundle", cost=1.0)
    assert store_zero.get("k") is None
    assert store_zero.stats()["items"] == 0

    store2 = ProcessCacheStore(max_bytes=2048)
    store2.put("alpha-1", [1, 2, 3], artifact="a", cost=1.0)
    store2.put("alpha-2", [4, 5, 6], artifact="a", cost=1.0)
    store2.put("beta-1", [7], artifact="b", cost=1.0)
    removed = store2.invalidate(prefix="alpha")
    assert removed == 2
    assert store2.get("beta-1") == [7]
    store2.clear()
    assert store2.stats()["items"] == 0


def test_disk_store_ttl_invalidate_and_corruption_recovery(tmp_path: Path):
    cache_root = tmp_path / "disk-cache"
    store = DiskCacheStore(cache_root, max_bytes=1024**2, compress_payloads=False, default_ttl_seconds=0.01)

    assert store.put("k1", {"v": 1}, artifact="jacobian", cost=1.0)
    assert store.get("k1") == {"v": 1}
    time.sleep(0.02)
    # Expired entries are removed and treated as misses.
    assert store.get("k1") is None

    assert store.put("prefix_a", {"v": "a"}, artifact="jacobian", cost=1.0)
    assert store.put("prefix_b", {"v": "b"}, artifact="jacobian", cost=1.0)
    assert store.put("other_c", {"v": "c"}, artifact="jacobian", cost=1.0)
    removed = store.invalidate(prefix="prefix_")
    assert removed == 2
    assert store.get("other_c") == {"v": "c"}

    assert store.put("corrupt", {"v": 99}, artifact="jacobian", cost=1.0)
    stats_before = store.stats()["items"]
    entry_file = next((cache_root / "objects" / "jacobian").glob("corrupt*"))
    entry_file.write_bytes(b"not-a-pickle")
    assert store.get("corrupt") is None
    assert store.stats()["items"] == stats_before - 1

    read_only = DiskCacheStore(tmp_path / "readonly", max_bytes=1024, read_only=True)
    assert read_only.put("x", {"v": 1}, artifact="mesh_bundle", cost=1.0) is False


def test_disk_store_size_zero_evicts_everything(tmp_path: Path):
    store = DiskCacheStore(tmp_path / "size-zero", max_bytes=0, compress_payloads=True)
    assert store.put("a", np.arange(10), artifact="jacobian", cost=1.0)
    assert store.stats()["items"] == 0
    assert store.get("a") is None
