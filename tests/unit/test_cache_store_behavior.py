"""Coverage-focused tests for cache key/process/disk stores."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

import pyeidors.cache.keys as cache_keys_mod
from pyeidors.cache.keys import _normalize, hash_array, hash_array_payload, hash_path
from pyeidors.cache.manager import CacheManager
from pyeidors.cache.store_disk import DiskCacheStore
from pyeidors.cache.store_process import ProcessCacheStore, estimate_object_size_bytes
from pyeidors.cache.types import CachePolicy


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


def test_cache_hash_helpers_stream_payloads_without_legacy_full_copy(tmp_path: Path):
    arr = np.arange(48, dtype=np.float64).reshape(6, 8)[:, ::2]
    arr_c = np.ascontiguousarray(arr)
    legacy_array_payload = f"{arr_c.dtype}:{arr_c.shape}:".encode("utf-8")
    expected_array_hash = hashlib.sha256(
        legacy_array_payload + arr_c.tobytes()
    ).hexdigest()
    expected_content_hash = hashlib.sha256(arr.tobytes()).hexdigest()
    expected_prefixed_hash = hashlib.sha256(b"prefix\0" + arr_c.tobytes()).hexdigest()

    assert hash_array(arr) == expected_array_hash
    assert _normalize(arr)["sha256"] == expected_content_hash
    assert hash_array_payload(arr) == expected_content_hash
    assert hash_array_payload(arr, prefix=b"prefix\0") == expected_prefixed_hash
    assert ".tobytes(" not in inspect.getsource(hash_array)
    assert ".tobytes(" not in inspect.getsource(hash_array_payload)

    payload_path = tmp_path / "large-ish.bin"
    payload_path.write_bytes((b"abcdef0123456789" * 1024) + b"tail")
    stat = payload_path.stat()
    content_hash = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    legacy_path_payload = (
        f"{payload_path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}::{content_hash}"
    ).encode("utf-8")

    assert hash_path(payload_path) == hashlib.sha256(legacy_path_payload).hexdigest()
    assert ".read_bytes(" not in inspect.getsource(hash_path)


def test_v571_cache_hash_helpers_chunk_noncontiguous_numeric_views(monkeypatch):
    base = np.arange(2048 * 64, dtype=np.float32).reshape(2048, 64)
    view = base[::2, ::2]
    full_contiguous = np.ascontiguousarray(view)
    expected = hashlib.sha256(full_contiguous.tobytes()).hexdigest()
    copied_nbytes: list[int] = []
    real_ascontiguousarray = cache_keys_mod.np.ascontiguousarray

    def _tracking_ascontiguousarray(value, *args, **kwargs):
        copied_nbytes.append(int(np.asarray(value).nbytes))
        return real_ascontiguousarray(value, *args, **kwargs)

    monkeypatch.setattr(cache_keys_mod, "_HASH_CHUNK_BYTES", 4096)
    monkeypatch.setattr(
        cache_keys_mod.np,
        "ascontiguousarray",
        _tracking_ascontiguousarray,
    )

    assert hash_array_payload(view) == expected
    assert copied_nbytes
    assert max(copied_nbytes) < full_contiguous.nbytes


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


def test_process_store_size_estimate_avoids_pickle_for_array_backed_objects(
    monkeypatch,
) -> None:
    from scipy import sparse

    import pyeidors.cache.store_process as store_process

    _ = monkeypatch
    assert "pickle.dumps" not in inspect.getsource(
        store_process._estimate_object_size_bytes
    )
    arr = np.arange(128, dtype=np.float64)
    wrapped = SimpleNamespace(arr=arr, label="wrapped")
    csr = sparse.eye(16, dtype=np.float64, format="csr")

    assert estimate_object_size_bytes(wrapped) >= arr.nbytes
    assert estimate_object_size_bytes(csr) >= (
        csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
    )

    cyclic: list[object] = []
    cyclic.append(cyclic)
    assert estimate_object_size_bytes(cyclic) >= 96


def test_process_store_admission_rejects_oversize_and_immediate_eviction():
    store = ProcessCacheStore(max_bytes=256)
    assert store.put("hot-a", b"a" * 120, artifact="jacobian", cost=1.0, priority=100)
    assert store.put("hot-b", b"b" * 100, artifact="jacobian", cost=1.0, priority=100)

    assert not store.put("cold", b"c" * 80, artifact="jacobian", cost=1.0)
    assert store.get("cold") is None
    assert store.get("hot-a") == b"a" * 120
    assert store.get("hot-b") == b"b" * 100

    stats = store.stats()
    assert stats["admission_rejections"] == 1
    assert stats["admission_rejected_bytes"] == 80
    assert stats["admission_rejection_reasons"] == {"would_evict_immediately": 1}

    assert not store.put("huge", b"x" * 512, artifact="jacobian", cost=1000.0)
    assert store.get("huge") is None
    stats = store.stats()
    assert stats["admission_rejections"] == 2
    assert stats["admission_rejection_reasons"]["entry_too_large"] == 1


def test_cache_manager_exposes_process_admission_rejection_stats(tmp_path: Path):
    manager = CacheManager(
        scope="process",
        cache_dir=tmp_path / "cache",
        policy=CachePolicy(process_max_bytes=128),
        code_fingerprint="admission-test",
    )

    value, lookup = manager.get_or_compute(
        artifact="jacobian",
        payload={"case": "oversize"},
        compute_fn=lambda: b"x" * 256,
        persist=False,
    )

    assert value == b"x" * 256
    assert lookup.layer == "compute"
    stats = manager.stats()
    assert stats["process_items"] == 0
    assert stats["process_admission_rejections"] == 1
    assert stats["process_admission_rejected_bytes"] == 256
    assert stats["process_admission_rejection_reasons"] == {"entry_too_large": 1}


def test_disk_store_ttl_invalidate_and_corruption_recovery(tmp_path: Path):
    cache_root = tmp_path / "disk-cache"
    store = DiskCacheStore(
        cache_root, max_bytes=1024**2, compress_payloads=False, default_ttl_seconds=0.01
    )

    assert store.put("k1", {"v": 1}, artifact="jacobian", cost=1.0)
    assert store.get("k1") == {"v": 1}
    time.sleep(0.02)
    # Expired entries are removed and treated as misses.
    assert store.get("k1") is None

    # Disable TTL for the remaining assertions so this test is not wall-clock flaky.
    store.default_ttl_seconds = None
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


def test_disk_store_streams_pickle_payloads_without_whole_file_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fail_read_bytes(self):  # noqa: ANN001
        raise AssertionError(f"unexpected whole-file read: {self}")

    def _fail_write_bytes(self, data):  # noqa: ANN001
        del data
        raise AssertionError(f"unexpected whole-file write: {self}")

    monkeypatch.setattr(Path, "read_bytes", _fail_read_bytes)
    monkeypatch.setattr(Path, "write_bytes", _fail_write_bytes)
    store = DiskCacheStore(
        tmp_path / "streaming-disk-cache",
        max_bytes=8 * 1024**2,
        compress_payloads=True,
    )
    payload = {"arr": np.arange(2048, dtype=np.float64), "label": "stream"}

    assert store.put("stream-key", payload, artifact="jacobian", cost=1.0)
    restored = store.get("stream-key")
    restored_by_value = store.get_value("stream-key")

    assert restored["label"] == "stream"
    np.testing.assert_allclose(restored["arr"], payload["arr"])
    np.testing.assert_allclose(restored_by_value["arr"], payload["arr"])
    assert "pickle.dumps" not in inspect.getsource(DiskCacheStore)
    assert ".read_bytes(" not in inspect.getsource(DiskCacheStore.get)
    assert ".read_bytes(" not in inspect.getsource(DiskCacheStore.get_value)
    assert ".write_bytes(" not in inspect.getsource(DiskCacheStore.put)
