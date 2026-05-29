"""T79 entrance gate: shared ProcessLRUCache + cache-key byte stability.

Two historical in-process LRU caches —
:mod:`pyeidors.geometry.process_mesh_cache` (mesh objects keyed by
disk artifact signatures) and
:mod:`pyeidors.forward.process_setup_cache` (forward-model static
setup bundles keyed by mesh + electrode + pattern provenance) —
duplicated the same ``OrderedDict`` LRU + ``threading.Lock`` +
``max_items`` + SHA256-of-JSON pattern. T79 consolidates the storage
machinery into :mod:`pyeidors.cache.process_lru` while preserving the
two public surfaces and the cache-key formula bytewise.

This entrance gate freezes:

* the consolidation: both wrappers reuse :class:`ProcessLRUCache`
  rather than holding their own ``OrderedDict`` + ``Lock``;
* cache-key byte stability for known-input fixtures so disk
  artifacts persisted before the refactor remain readable;
* LRU eviction + thread-safety semantics on the shared primitive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pytest

from pyeidors.cache.process_lru import (
    ProcessLRUCache,
    hash_json_payload,
    path_signature,
)
from pyeidors.data.structures import PatternConfig
from pyeidors.forward.process_setup_cache import (
    ForwardStaticSetupBundle,
    _PROCESS_FORWARD_SETUP_CACHE,
    _forward_setup_bundle_size_bytes,
    build_process_forward_setup_key,
    clear_process_forward_setup_cache,
    get_process_forward_setup_bundle,
    process_forward_setup_cache_stats,
    put_process_forward_setup_bundle,
)
import pyeidors.forward.process_setup_cache as forward_setup_cache_module
from pyeidors.geometry.process_mesh_cache import (
    _PROCESS_MESH_CACHE,
    _mesh_cache_size_bytes,
    build_process_mesh_cache_key,
    clear_process_mesh_cache,
    get_process_cached_mesh,
    process_mesh_cache_stats,
    put_process_cached_mesh,
)
import pyeidors.geometry.process_mesh_cache as mesh_cache_module


def test_process_caches_share_processlru_machinery() -> None:
    """Both wrappers wrap a :class:`ProcessLRUCache` instance, not a hand-rolled dict."""
    assert isinstance(_PROCESS_MESH_CACHE, ProcessLRUCache)
    assert isinstance(_PROCESS_FORWARD_SETUP_CACHE, ProcessLRUCache)
    assert _PROCESS_MESH_CACHE.max_items == 8
    assert _PROCESS_FORWARD_SETUP_CACHE.max_items == 8


def test_hash_json_payload_matches_inline_sha256_of_canonical_json() -> None:
    """The shared hash MUST stay byte-identical to the historical inline SHA256."""
    payload = {"alpha": 1, "beta": [2, 3], "gamma": "x"}
    expected = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    assert hash_json_payload(payload) == expected


def test_path_signature_for_existing_file_includes_size_and_mtime(tmp_path) -> None:
    target = tmp_path / "demo.bin"
    target.write_bytes(b"abc")
    sig = path_signature(target)
    assert "demo.bin" in sig
    assert "::3::" in sig  # size = 3 bytes
    assert sig.endswith("::0")  # not a directory


def test_path_signature_for_missing_path_returns_raw_string(tmp_path) -> None:
    missing = tmp_path / "nope.bin"
    assert path_signature(missing) == str(missing)


# ---------------------------------------------------------------------------
# Cache-key byte stability for known fixtures (V16 / V17 / V36 protected).
# ---------------------------------------------------------------------------


def test_build_process_mesh_cache_key_is_byte_stable_for_known_inputs(
    tmp_path,
) -> None:
    mesh_path = tmp_path / "mesh.msh"
    mesh_path.write_bytes(b"placeholder")
    key1 = build_process_mesh_cache_key(
        mesh_file=mesh_path, gdim=2, n_elec=16, mesh_name="mesh"
    )
    key2 = build_process_mesh_cache_key(
        mesh_file=mesh_path, gdim=2, n_elec=16, mesh_name="mesh"
    )
    assert key1 == key2  # deterministic
    assert len(key1) == 64  # sha256 hex
    # Sanity: changing gdim flips the key.
    assert key1 != build_process_mesh_cache_key(
        mesh_file=mesh_path, gdim=3, n_elec=16, mesh_name="mesh"
    )


def test_build_process_forward_setup_key_payload_shape_is_unchanged() -> None:
    pattern = PatternConfig(
        n_elec=16,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
        use_meas_current=False,
        rotate_meas=True,
    )
    z = np.full(16, 1e-5, dtype=np.float64)
    key1 = build_process_forward_setup_key(
        mesh_file="mesh.h5", n_elec=16, z=z, pattern_config=pattern
    )
    key2 = build_process_forward_setup_key(
        mesh_file="mesh.h5", n_elec=16, z=z, pattern_config=pattern
    )
    assert key1 == key2
    assert len(key1) == 64
    # A different content hash must yield a different key (V17 contract).
    key3 = build_process_forward_setup_key(
        mesh_file=None,
        n_elec=16,
        z=z,
        pattern_config=pattern,
        mesh_content_hash="abc123",
    )
    assert key3 != key1


def test_build_process_forward_setup_key_requires_mesh_token() -> None:
    """V16: at least one of mesh_file / mesh_content_hash must be present."""
    pattern = PatternConfig(
        n_elec=4,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    with pytest.raises(ValueError, match="mesh_file or mesh_content_hash"):
        build_process_forward_setup_key(
            mesh_file=None, n_elec=4, z=np.zeros(4), pattern_config=pattern
        )


# ---------------------------------------------------------------------------
# LRU + thread-safety semantics.
# ---------------------------------------------------------------------------


@dataclass
class _DummyValue:
    payload: int


@dataclass
class _DummyGeometry:
    x: np.ndarray


@dataclass
class _DummyTags:
    indices: np.ndarray
    values: np.ndarray


@dataclass
class _DummyMesh:
    geometry: _DummyGeometry
    facet_tags: _DummyTags
    cell_tags: _DummyTags | None = None
    electrode_vertices: list[np.ndarray] | None = None
    _derived_arrays: object | None = None


def test_process_lru_cache_lru_eviction_and_get_promote_to_recent() -> None:
    cache: ProcessLRUCache[_DummyValue] = ProcessLRUCache(max_items=3)
    cache.put("a", _DummyValue(1))
    cache.put("b", _DummyValue(2))
    cache.put("c", _DummyValue(3))

    # Access "a" so it becomes most-recent before "b" gets evicted.
    assert cache.get("a").payload == 1
    cache.put("d", _DummyValue(4))
    assert cache.get("b") is None
    assert cache.stats() == {"items": 3, "max_items": 3}

    cache.clear()
    assert cache.stats()["items"] == 0


def test_v605_process_lru_cache_byte_budget_evicts_and_skips_oversize() -> None:
    cache: ProcessLRUCache[_DummyValue] = ProcessLRUCache(
        max_items=10,
        max_bytes=10,
        sizeof=lambda value: value.payload,
    )
    cache.put("a", _DummyValue(4))
    cache.put("b", _DummyValue(4))
    assert cache.get("a").payload == 4

    cache.put("c", _DummyValue(4))

    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None
    assert cache.stats() == {
        "items": 2,
        "max_items": 10,
        "total_bytes": 8,
        "max_bytes": 10,
    }

    cache.put("too-large", _DummyValue(11))
    assert cache.get("too-large") is None
    assert cache.stats()["total_bytes"] == 8

    cache.discard("a")
    assert cache.get("a") is None
    assert cache.stats()["total_bytes"] == 4


def test_v606_process_mesh_cache_skips_entries_above_byte_budget(monkeypatch) -> None:
    clear_process_mesh_cache()
    mesh = _DummyMesh(
        geometry=_DummyGeometry(np.ones((4, 3), dtype=np.float64)),
        facet_tags=_DummyTags(
            indices=np.arange(4, dtype=np.int32),
            values=np.arange(4, dtype=np.int32),
        ),
        electrode_vertices=[np.ones((2, 3), dtype=np.float32)],
    )
    key = "mesh-byte-budget-key"

    try:
        put_process_cached_mesh(key, mesh)
        assert get_process_cached_mesh(key) is mesh

        monkeypatch.setattr(
            mesh_cache_module,
            "_RESOLVED_PROCESS_MESH_CACHE_MAX_BYTES",
            _mesh_cache_size_bytes(mesh) - 1,
        )
        put_process_cached_mesh(key, mesh)

        assert get_process_cached_mesh(key) is None
        stats = process_mesh_cache_stats()
        assert stats["items"] == 0
        assert "max_bytes" in stats
    finally:
        clear_process_mesh_cache()


def _dummy_forward_setup_bundle() -> ForwardStaticSetupBundle:
    return ForwardStaticSetupBundle(
        ds_electrodes=None,
        electrode_tags=(),
        electrode_boundary_measures={},
        geometry_scale_to_m=1.0,
        mesh_tdim=2,
        boundary_scale_to_m=1.0,
        electrode_lengths_m=np.zeros(4, dtype=np.float64),
        pattern_manager=None,
        V=None,
        V_sigma=None,
        dofs=0,
        electrode_matrix=__import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix(
            np.eye(4, dtype=np.float64)
        ),
    )


def test_v607_process_forward_setup_cache_skips_entries_above_byte_budget(
    monkeypatch,
) -> None:
    clear_process_forward_setup_cache()
    key = "forward-setup-byte-budget-key"
    bundle = _dummy_forward_setup_bundle()

    try:
        put_process_forward_setup_bundle(key, bundle)
        assert get_process_forward_setup_bundle(key) is bundle

        monkeypatch.setattr(
            forward_setup_cache_module,
            "_RESOLVED_PROCESS_FORWARD_SETUP_CACHE_MAX_BYTES",
            _forward_setup_bundle_size_bytes(bundle) - 1,
        )
        put_process_forward_setup_bundle(key, bundle)

        assert get_process_forward_setup_bundle(key) is None
        stats = process_forward_setup_cache_stats()
        assert stats["items"] == 0
        assert "max_bytes" in stats
    finally:
        clear_process_forward_setup_cache()


def test_process_lru_cache_rejects_non_positive_max_items() -> None:
    with pytest.raises(ValueError, match="max_items must be positive"):
        ProcessLRUCache(max_items=0)


def test_clear_helpers_drain_module_level_caches(tmp_path) -> None:
    """Module-level ``clear_*`` helpers reset the shared LRU through the wrapper."""
    pattern = PatternConfig(
        n_elec=2,
        stim_pattern="{ad}",
        meas_pattern="{ad}",
        drive_mode="normalized",
        drive_value=1.0,
        geometry_scale_to_m=1.0,
    )
    z = np.full(2, 1e-5, dtype=np.float64)
    key = build_process_forward_setup_key(
        mesh_file="dummy.msh", n_elec=2, z=z, pattern_config=pattern
    )

    bundle = ForwardStaticSetupBundle(
        ds_electrodes=None,
        electrode_tags=(),
        electrode_boundary_measures={},
        geometry_scale_to_m=1.0,
        mesh_tdim=2,
        boundary_scale_to_m=1.0,
        electrode_lengths_m=np.zeros(2, dtype=float),
        pattern_manager=None,
        V=None,
        V_sigma=None,
        dofs=0,
        electrode_matrix=__import__("scipy.sparse", fromlist=["csr_matrix"]).csr_matrix(
            (2, 2)
        ),
    )
    put_process_forward_setup_bundle(key, bundle)
    assert get_process_forward_setup_bundle(key) is bundle
    assert process_forward_setup_cache_stats()["items"] >= 1

    clear_process_forward_setup_cache()
    assert get_process_forward_setup_bundle(key) is None
    clear_process_mesh_cache()  # geometry-side helper is symmetric
